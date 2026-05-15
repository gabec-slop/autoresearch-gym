from __future__ import annotations

import argparse
import json
import sys
import time
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from autoresearch_gym.runner import render_rollouts, session_run
from autoresearch_gym.runner.experiment import DEFAULT_VISUAL_CONTROL, normalize_visual_control, write_json_atomic


def _repo_root() -> Path:
    return Path.cwd()


def _slugify(value: str) -> str:
    return session_run._slugify(value)  # type: ignore[attr-defined]


def cmd_run(argv: list[str]) -> int:
    session_run.main(argv)
    return 0


def cmd_init_session(args: argparse.Namespace) -> int:
    session_dir = args.base_dir / "sessions" / f"{time.strftime('%Y%m%d-%H%M%S')}-{_slugify(args.label)}"
    session_dir.mkdir(parents=True, exist_ok=False)
    candidates_dir = session_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = session_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    log_path = session_dir / "outer_loop_log.md"
    log_path.write_text(session_run.SESSION_LOG_TEMPLATE, encoding="utf-8")
    session_meta = {
        "session_dir": str(session_dir),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "search_mode": args.search_mode,
        "benchmark_path": str(args.benchmark.resolve()),
        "seed_candidate_path": str(args.seed_candidate.resolve()),
        "candidates_dir": str(candidates_dir.resolve()),
        "next_candidate_path": str((candidates_dir / "pass01_baseline.py").resolve()),
        "runs_dir": str(runs_dir),
        "results_path": str(session_dir / "results.jsonl"),
        "log_path": str(log_path),
    }
    (session_dir / "session.json").write_text(json.dumps(session_meta, indent=2), encoding="utf-8")
    pointer = {
        "session_path": session_dir.relative_to(Path.cwd()).as_posix() if session_dir.is_relative_to(Path.cwd()) else str(session_dir),
        "session_dir": str(session_dir.resolve()),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tag": "init-session",
        "search_mode": args.search_mode,
        "source": "autoresearch-gym init-session",
    }
    args.base_dir.mkdir(parents=True, exist_ok=True)
    (args.base_dir / "live_session.json").write_text(json.dumps(pointer, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "session_dir": str(session_dir),
                "seed_candidate": str(args.seed_candidate.resolve()),
                "next_candidate": str(candidates_dir / "pass01_baseline.py"),
            },
            indent=2,
        )
    )
    return 0


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: object, dashboard_root: Path, **kwargs: object) -> None:
        self.dashboard_root = dashboard_root.resolve()
        super().__init__(*args, **kwargs)

    def _send_json(self, status: int, payload: dict[str, object]) -> None:
        encoded = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def _resolve_session_dir(self, session: object) -> Path:
        if not isinstance(session, str) or not session.strip():
            raise ValueError("missing session")
        session_path = Path(session.strip().lstrip("/"))
        if session_path.is_absolute() or ".." in session_path.parts:
            raise ValueError("invalid session path")
        resolved = (self.dashboard_root / session_path).resolve()
        if not resolved.is_relative_to(self.dashboard_root):
            raise ValueError("session path escapes dashboard root")
        return resolved

    def _control_path_for_request(self, session: object) -> Path:
        session_dir = self._resolve_session_dir(session)
        return session_dir / "live" / "control.json"

    def _repo_relative(self, path: Path) -> str:
        return path.resolve().relative_to(self.dashboard_root).as_posix()

    def _read_live_session_path(self) -> str | None:
        pointer_path = self.dashboard_root / "autoresearch_runs" / "live_session.json"
        if not pointer_path.exists():
            return None
        try:
            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        value = pointer.get("session_path") or pointer.get("session") or pointer.get("path")
        return str(value).strip("/ ") if value else None

    def _session_summary(self, session_dir: Path, current_session: str | None) -> dict[str, object] | None:
        metrics_path = session_dir / "live" / "current_run_metrics.json"
        results_path = session_dir / "results.jsonl"
        session_json_path = session_dir / "session.json"
        if not any(path.exists() for path in (metrics_path, results_path, session_json_path)):
            return None

        metrics: dict[str, object] = {}
        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                metrics = {}

        run = metrics.get("run") if isinstance(metrics.get("run"), dict) else {}
        current = metrics.get("current") if isinstance(metrics.get("current"), dict) else {}
        assert isinstance(run, dict)
        assert isinstance(current, dict)
        path = self._repo_relative(session_dir)
        updated_at = metrics_path.stat().st_mtime if metrics_path.exists() else session_dir.stat().st_mtime
        status = current.get("status") or run.get("status") or ("complete" if results_path.exists() else "session")
        label = run.get("tag") or run.get("run_id") or session_dir.name
        return {
            "path": path,
            "name": Path(path).name,
            "label": str(label),
            "status": str(status),
            "updated_at": updated_at,
            "updated_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(updated_at)),
            "episodes": current.get("episodes_complete"),
            "avg_return": current.get("avg_return"),
            "success_rate": current.get("success_rate"),
            "is_current": path == current_session,
        }

    def _list_sessions(self) -> list[dict[str, object]]:
        sessions_dir = self.dashboard_root / "autoresearch_runs" / "sessions"
        if not sessions_dir.exists():
            return []
        current_session = self._read_live_session_path()
        summaries: list[dict[str, object]] = []
        seen_paths: set[str] = set()
        for session_dir in sessions_dir.iterdir():
            if not session_dir.is_dir():
                continue
            summary = self._session_summary(session_dir, current_session)
            if summary is None:
                continue
            path = str(summary["path"])
            if path in seen_paths:
                continue
            seen_paths.add(path)
            summaries.append(summary)
        return sorted(summaries, key=lambda item: float(item["updated_at"]), reverse=True)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/_autoresearch/sessions":
            self._send_json(200, {"ok": True, "current_session": self._read_live_session_path(), "sessions": self._list_sessions()})
            return
        if parsed.path != "/_autoresearch/control":
            return super().do_GET()
        try:
            params = parse_qs(parsed.query)
            control_path = self._control_path_for_request(params.get("session", [""])[0])
            payload = json.loads(control_path.read_text(encoding="utf-8")) if control_path.exists() else DEFAULT_VISUAL_CONTROL
            self._send_json(200, {"ok": True, "control": normalize_visual_control(payload)})
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self._send_json(400, {"ok": False, "error": str(exc)})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/_autoresearch/control":
            self._send_json(404, {"ok": False, "error": "unknown endpoint"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(min(length, 64_000))
            request = json.loads(raw.decode("utf-8")) if raw else {}
            control_path = self._control_path_for_request(request.get("session"))
            incoming = request.get("control")
            if not isinstance(incoming, dict):
                incoming = {key: value for key, value in request.items() if key != "session"}
            existing = {}
            if control_path.exists():
                try:
                    existing = json.loads(control_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    existing = {}
            merged = normalize_visual_control({**existing, **incoming})
            write_json_atomic(control_path, merged)
            self._send_json(200, {"ok": True, "control": merged, "control_path": str(control_path)})
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self._send_json(400, {"ok": False, "error": str(exc)})


def cmd_dashboard(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    handler = partial(DashboardHandler, directory=str(root), dashboard_root=root)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://{args.host}:{args.port}/dashboard/"
    if args.session:
        url += f"?session={args.session}"
    print(f"Serving {root}")
    print(url)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard server.")
    finally:
        server.server_close()
    return 0


def cmd_render_rollouts(argv: list[str]) -> int:
    render_rollouts.main(argv)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="autoresearch-gym", description="Run and inspect Gymnasium autoresearch sessions.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("run", help="Run one already-authored fixed-budget candidate. Arguments are passed to the runner.")

    init_parser = subparsers.add_parser(
        "init-session",
        help="Create a session directory and record the selected seed without creating pass files.",
    )
    package_dir = Path(__file__).resolve().parent
    default_task_dir = package_dir / "tasks" / "hopper_v0"
    init_parser.add_argument("--label", required=True)
    init_parser.add_argument("--benchmark", type=Path, default=default_task_dir / "benchmark.json")
    init_parser.add_argument("--seed-candidate", type=Path, default=default_task_dir / "seed_trainable.py")
    init_parser.add_argument("--base-dir", type=Path, default=Path.cwd() / "autoresearch_runs")
    init_parser.add_argument("--search-mode", choices=["linear"], default="linear")

    dashboard_parser = subparsers.add_parser("dashboard", help="Serve the static dashboard from the current repo.")
    dashboard_parser.add_argument("--host", default="127.0.0.1")
    dashboard_parser.add_argument("--port", type=int, default=4174)
    dashboard_parser.add_argument("--root", type=Path, default=_repo_root())
    dashboard_parser.add_argument("--session", default=None)
    dashboard_parser.add_argument("--open", action="store_true")

    subparsers.add_parser("render-rollouts", help="Render rollout GIFs from a saved MuJoCo checkpoint run.")
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["run"]:
        return cmd_run(argv[1:])
    if argv[:1] == ["render-rollouts"]:
        return cmd_render_rollouts(argv[1:])
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "init-session":
        return cmd_init_session(args)
    if args.command == "dashboard":
        return cmd_dashboard(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
