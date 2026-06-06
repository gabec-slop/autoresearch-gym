from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from autoresearch_gym.runner import render_rollouts, session_run
from autoresearch_gym.external.remote_session import run_session_doctor
from autoresearch_gym.runner.dashboard_server import (
    DASHBOARD_MANAGER,
    dashboard_url,
    ensure_session_dashboard,
    find_available_port,
    session_dashboard_status,
    terminate_session_dashboard,
)
from autoresearch_gym.runner.experiment import DEFAULT_VISUAL_CONTROL, normalize_visual_control, select_device, write_json_atomic


def _repo_root() -> Path:
    return Path.cwd()


def _slugify(value: str) -> str:
    return session_run._slugify(value)  # type: ignore[attr-defined]


def _looks_like_dashboard_session_dir(path: Path) -> bool:
    return any((path / name).exists() for name in ("live", "session.json", "results.jsonl", "outer_loop_log.md"))


def resolve_dashboard_session_dir(dashboard_root: Path, session: object) -> Path:
    if not isinstance(session, str) or not session.strip():
        raise ValueError("missing session")
    root = dashboard_root.resolve()
    raw_session = session.strip().rstrip("/")
    session_path = Path(raw_session)
    if session_path.is_absolute():
        resolved = session_path.resolve()
        if not resolved.is_relative_to(root) and not _looks_like_dashboard_session_dir(resolved):
            raise ValueError("absolute session path is not an autoresearch session")
        return resolved
    if ".." in session_path.parts:
        raise ValueError("invalid session path")
    resolved = (root / raw_session.lstrip("/")).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("session path escapes dashboard root")
    return resolved


def resolve_dashboard_artifact_path(dashboard_root: Path, session: object, artifact_path: object) -> Path:
    if not isinstance(artifact_path, str) or not artifact_path.strip():
        raise ValueError("missing artifact path")
    raw_path = artifact_path.strip().replace("\\", "/")
    if raw_path.startswith(("http://", "https://", "data:")):
        raise ValueError("remote artifact URLs are fetched directly by the browser")

    root = dashboard_root.resolve()
    session_dir = resolve_dashboard_session_dir(root, session)
    if Path(raw_path).is_absolute():
        resolved = Path(raw_path).resolve()
    else:
        relative_path = raw_path.lstrip("/")
        root_candidate = (root / relative_path).resolve()
        session_candidate = (session_dir / relative_path).resolve()
        resolved = root_candidate if root_candidate.exists() else session_candidate

    allowed_roots = [root, session_dir]
    if not any(resolved.is_relative_to(allowed_root) for allowed_root in allowed_roots):
        raise ValueError("artifact path escapes dashboard root and session")
    if not resolved.exists() or not resolved.is_file():
        raise FileNotFoundError(str(resolved))
    return resolved


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
        "execution_target": args.execution_target,
        "candidates_dir": str(candidates_dir.resolve()),
        "next_candidate_path": str((candidates_dir / "pass01_baseline.py").resolve()),
        "runs_dir": str(runs_dir),
        "results_path": str(session_dir / "results.jsonl"),
        "log_path": str(log_path),
    }
    doctor = None
    if not args.skip_doctor:
        try:
            doctor = run_session_doctor(
                args.benchmark,
                execution_target=args.execution_target,
                target_config_path=args.target_config,
                repo_root=Path.cwd(),
                timeout=args.doctor_timeout,
            )
        except Exception as exc:  # noqa: BLE001 - init records doctor failures instead of losing the session.
            doctor = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "checks": [
                    {
                        "name": "session_doctor",
                        "status": "fail",
                        "message": str(exc),
                    }
                ],
            }
        doctor_path = session_dir / "doctor.json"
        write_json_atomic(doctor_path, doctor)
        session_meta["doctor_path"] = str(doctor_path)
        session_meta["doctor_ok"] = bool(doctor.get("ok"))
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
    dashboard = None
    if args.dashboard:
        dashboard = ensure_session_dashboard(
            session_dir,
            host=args.dashboard_host,
            port=args.dashboard_port,
            port_end=args.dashboard_port_end,
            root=Path.cwd(),
            ready_timeout=args.dashboard_ready_timeout,
        )
    print(
        json.dumps(
            {
                "session_dir": str(session_dir),
                "seed_candidate": str(args.seed_candidate.resolve()),
                "next_candidate": str(candidates_dir / "pass01_baseline.py"),
                "doctor": doctor,
                "dashboard": dashboard,
            },
            indent=2,
        )
    )
    if args.strict_doctor and doctor is not None and not doctor.get("ok"):
        return 1
    return 0


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: object, dashboard_root: Path, **kwargs: object) -> None:
        self.dashboard_root = dashboard_root.resolve()
        super().__init__(*args, **kwargs)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _send_json(self, status: int, payload: dict[str, object]) -> None:
        encoded = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def _resolve_session_dir(self, session: object) -> Path:
        return resolve_dashboard_session_dir(self.dashboard_root, session)

    def _control_path_for_request(self, session: object) -> Path:
        session_dir = self._resolve_session_dir(session)
        return session_dir / "live" / "control.json"

    def _repo_relative(self, path: Path) -> str:
        resolved = path.resolve()
        try:
            return resolved.relative_to(self.dashboard_root).as_posix()
        except ValueError:
            return str(resolved)

    def _read_live_session_path(self) -> str | None:
        pointer_path = self.dashboard_root / "autoresearch_runs" / "live_session.json"
        if not pointer_path.exists():
            return None
        try:
            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        value = pointer.get("session_path") or pointer.get("session") or pointer.get("path")
        if not value:
            return None
        raw_value = str(value).strip()
        return raw_value.rstrip("/") if Path(raw_value).is_absolute() else raw_value.strip("/ ")

    def _send_file(self, path: Path) -> None:
        stat = path.stat()
        self.send_response(200)
        self.send_header("Content-Type", self.guess_type(str(path)))
        self.send_header("Content-Length", str(stat.st_size))
        self.send_header("Last-Modified", self.date_time_string(stat.st_mtime))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        with path.open("rb") as handle:
            shutil.copyfileobj(handle, self.wfile)

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
        if parsed.path == "/_autoresearch/identity":
            self._send_json(
                200,
                {
                    "ok": True,
                    "managed_by": DASHBOARD_MANAGER,
                    "root": str(self.dashboard_root),
                    "pid": os.getpid(),
                },
            )
            return
        if parsed.path == "/_autoresearch/sessions":
            self._send_json(200, {"ok": True, "current_session": self._read_live_session_path(), "sessions": self._list_sessions()})
            return
        if parsed.path == "/_autoresearch/artifact":
            try:
                params = parse_qs(parsed.query)
                path = resolve_dashboard_artifact_path(
                    self.dashboard_root,
                    params.get("session", [""])[0],
                    params.get("path", [""])[0],
                )
                self._send_file(path)
            except FileNotFoundError as exc:
                self._send_json(404, {"ok": False, "error": str(exc)})
            except (OSError, ValueError) as exc:
                self._send_json(400, {"ok": False, "error": str(exc)})
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
    port = args.port if args.no_port_probe else find_available_port(args.host, args.port, args.port_end)
    handler = partial(DashboardHandler, directory=str(root), dashboard_root=root)
    server = ThreadingHTTPServer((args.host, port), handler)
    url = dashboard_url(args.host, port, args.session)
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


def cmd_session_dashboard(args: argparse.Namespace) -> int:
    session_dir = args.session_dir.resolve()
    if args.action == "ensure":
        payload = ensure_session_dashboard(
            session_dir,
            host=args.host,
            port=args.port,
            port_end=args.port_end,
            root=args.root,
            ready_timeout=args.ready_timeout,
            force_restart=args.force_restart,
        )
        print(json.dumps({"ok": bool(payload.get("ready")), "action": "ensure", "dashboard": payload}, indent=2))
        return 0 if payload.get("ready") else 1
    if args.action == "status":
        payload = session_dashboard_status(session_dir)
        print(json.dumps(payload, indent=2))
        return 0 if payload.get("ok") else 1
    if args.action == "teardown":
        payload = terminate_session_dashboard(session_dir, wait_seconds=args.wait_seconds)
        print(json.dumps(payload, indent=2))
        return 0 if payload.get("ok") else 1
    raise ValueError(f"unknown session-dashboard action: {args.action}")


def cmd_render_rollouts(argv: list[str]) -> int:
    render_rollouts.main(argv)
    return 0


def _nvidia_smi_gpus() -> list[dict[str, object]]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if completed.returncode != 0:
        return []
    gpus: list[dict[str, object]] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3 or not parts[0]:
            continue
        try:
            memory_mb: int | None = int(float(parts[1]))
        except ValueError:
            memory_mb = None
        gpus.append({"name": parts[0], "memory_total_mb": memory_mb, "driver_version": parts[2]})
    return gpus


def cmd_doctor(args: argparse.Namespace) -> int:
    if getattr(args, "benchmark", None) is not None or getattr(args, "execution_target", None) is not None:
        benchmark = args.benchmark
        if benchmark is None:
            package_dir = Path(__file__).resolve().parent
            benchmark = package_dir / "tasks" / "hopper_v0" / "benchmark.json"
        payload = run_session_doctor(
            benchmark,
            execution_target=getattr(args, "execution_target", None),
            target_config_path=getattr(args, "target_config", None),
            repo_root=Path.cwd(),
            timeout=getattr(args, "timeout", 60.0),
        )
        print(json.dumps(payload, indent=2))
        if args.strict and not payload["ok"]:
            return 1
        return 0

    checks: list[dict[str, object]] = []
    torch_info: dict[str, object] = {"installed": False}
    selected_device = "unavailable"
    try:
        import torch

        mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
        torch_info.update(
            {
                "installed": True,
                "version": getattr(torch, "__version__", None),
                "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device_count": int(torch.cuda.device_count()),
                "cuda_devices": [
                    torch.cuda.get_device_name(index)
                    for index in range(torch.cuda.device_count())
                    if torch.cuda.is_available()
                ],
                "mps_available": bool(mps_backend and mps_backend.is_available()),
            }
        )
        selected_device = str(select_device(args.device))
    except ModuleNotFoundError as exc:
        torch_info["error"] = str(exc)

    nvidia_gpus = _nvidia_smi_gpus()
    checks.append(
        {
            "name": "nvidia_cuda_torch",
            "status": "warn" if nvidia_gpus and not torch_info.get("cuda_available") else "ok",
            "message": (
                "NVIDIA GPU detected by nvidia-smi, but this Python environment cannot use CUDA through PyTorch."
                if nvidia_gpus and not torch_info.get("cuda_available")
                else "No NVIDIA/PyTorch CUDA mismatch detected."
            ),
        }
    )
    payload = {
        "ok": all(check["status"] == "ok" for check in checks),
        "selected_device": selected_device,
        "torch": torch_info,
        "nvidia_smi_gpus": nvidia_gpus,
        "checks": checks,
    }
    print(json.dumps(payload, indent=2))
    if args.strict and not payload["ok"]:
        return 1
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
    init_parser.add_argument(
        "--execution-target",
        default=None,
        help="Optional target to doctor at session init. Passes still choose their target explicitly.",
    )
    init_parser.add_argument(
        "--target-config",
        type=Path,
        default=None,
        help="Ignored TOML target config used when --execution-target is provided.",
    )
    init_parser.add_argument(
        "--skip-doctor",
        action="store_true",
        help="Create the session without writing a benchmark/target doctor report.",
    )
    init_parser.add_argument(
        "--strict-doctor",
        action="store_true",
        help="Exit nonzero after creating the session if the doctor report is not ok.",
    )
    init_parser.add_argument("--doctor-timeout", type=float, default=60.0)
    init_parser.set_defaults(dashboard=True)
    init_parser.add_argument(
        "--no-dashboard",
        action="store_false",
        dest="dashboard",
        help="Create the session without starting the session dashboard.",
    )
    init_parser.add_argument("--dashboard-host", default="127.0.0.1")
    init_parser.add_argument("--dashboard-port", type=int, default=4174)
    init_parser.add_argument("--dashboard-port-end", type=int, default=4199)
    init_parser.add_argument("--dashboard-ready-timeout", type=float, default=5.0)

    dashboard_parser = subparsers.add_parser("dashboard", help="Serve the static dashboard from the current repo.")
    dashboard_parser.add_argument("--host", default="127.0.0.1")
    dashboard_parser.add_argument("--port", type=int, default=4174)
    dashboard_parser.add_argument("--port-end", type=int, default=4199)
    dashboard_parser.add_argument("--no-port-probe", action="store_true")
    dashboard_parser.add_argument("--root", type=Path, default=_repo_root())
    dashboard_parser.add_argument("--session", default=None)
    dashboard_parser.add_argument("--open", action="store_true")

    session_dashboard_parser = subparsers.add_parser(
        "session-dashboard",
        help="Ensure, inspect, or tear down the dashboard service for one autoresearch session.",
    )
    session_dashboard_parser.add_argument("action", choices=["ensure", "status", "teardown"])
    session_dashboard_parser.add_argument("--session-dir", type=Path, required=True)
    session_dashboard_parser.add_argument("--host", default="127.0.0.1")
    session_dashboard_parser.add_argument("--port", type=int, default=4174)
    session_dashboard_parser.add_argument("--port-end", type=int, default=4199)
    session_dashboard_parser.add_argument("--root", type=Path, default=_repo_root())
    session_dashboard_parser.add_argument("--ready-timeout", type=float, default=5.0)
    session_dashboard_parser.add_argument("--force-restart", action="store_true")
    session_dashboard_parser.add_argument("--wait-seconds", type=float, default=3.0)

    subparsers.add_parser("render-rollouts", help="Render rollout GIFs from a saved MuJoCo checkpoint run.")

    doctor_parser = subparsers.add_parser("doctor", help="Check simulator and accelerator installation state.")
    doctor_parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    doctor_parser.add_argument(
        "--benchmark",
        type=Path,
        default=None,
        help="Run the benchmark-aware session doctor instead of the basic accelerator check.",
    )
    doctor_parser.add_argument(
        "--execution-target",
        default=None,
        help="Run the benchmark-aware doctor on this SSH target.",
    )
    doctor_parser.add_argument("--target-config", type=Path, default=None)
    doctor_parser.add_argument("--timeout", type=float, default=60.0)
    doctor_parser.add_argument("--strict", action="store_true", help="Exit nonzero when a check reports a warning.")
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
    if args.command == "session-dashboard":
        return cmd_session_dashboard(args)
    if args.command == "doctor":
        return cmd_doctor(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
