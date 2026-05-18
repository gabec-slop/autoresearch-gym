from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

try:
    from .experiment import run_experiment
except ImportError:  # pragma: no cover - supports direct script execution.
    from experiment import run_experiment


SESSION_LOG_TEMPLATE = """# Outer Loop Research Log

This log is session-local. Update it immediately after every pass in this session.

Candidate files must be authored by the model one at a time. Pass 1 must be a
verbatim copy of the selected seed trainable, usually
`candidates/pass01_baseline.py`. Scripts may run an already-authored candidate,
but must not generate, mutate, queue, or prewrite candidate files.

For each pass, record:
- pass number / epoch
- stage: smoke or confirmation
- run id / tag
- lineage mode: from_scratch or warm_start
- parent run id / checkpoint when warm-started
- search mode: linear
- mutation family / proposal id
- mutation summary
- primary metric value
- relevant secondary metrics
- decision: keep / revert / prune / keep_on_frontier / promote
- interpretation
- next planned mutation

Most recent entries should appear at the bottom.
"""


def _checkpoint_from_run_dir(out_dir: Path, run_id: str) -> Path:
    return out_dir / run_id / "agent_checkpoint.pt"


def _sort_key(payload: dict) -> tuple[float, float, str]:
    return (
        float(payload.get("score", 0.0)),
        float(payload.get("eval", {}).get("avg_return", 0.0)),
        str(payload.get("run_id", "")),
    )


def resolve_init_checkpoint(args: argparse.Namespace) -> Path | None:
    if args.init_checkpoint is not None:
        return args.init_checkpoint.resolve()
    if args.init_from_run is not None:
        candidate = _checkpoint_from_run_dir(args.out_dir, args.init_from_run)
        if not candidate.exists():
            raise FileNotFoundError(f"Checkpoint for run '{args.init_from_run}' was not found at {candidate}")
        return candidate.resolve()
    if not args.init_from_best:
        return None

    if not args.results_path.exists():
        raise FileNotFoundError(f"Cannot resolve --init-from-best because results file does not exist: {args.results_path}")

    best_payload: dict | None = None
    with args.results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            checkpoint_path = payload.get("artifacts", {}).get("checkpoint_path")
            if checkpoint_path is None:
                run_id = payload.get("run_id")
                if run_id is None:
                    continue
                checkpoint_path = str(_checkpoint_from_run_dir(args.out_dir, run_id))
            if not Path(checkpoint_path).exists():
                continue
            if best_payload is None or _sort_key(payload) > _sort_key(best_payload):
                best_payload = payload

    if best_payload is None:
        raise FileNotFoundError("Could not find any recorded checkpoint to satisfy --init-from-best")

    checkpoint_path = best_payload.get("artifacts", {}).get("checkpoint_path")
    if checkpoint_path is None:
        checkpoint_path = str(_checkpoint_from_run_dir(args.out_dir, best_payload["run_id"]))
    return Path(checkpoint_path).resolve()


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-._")
    return slug or "session"


def _repo_relative_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _repo_relative_unresolved_path(path: Path) -> str:
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return str(path)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def write_live_session_pointer(base_dir: Path, session_dir: Path, args: argparse.Namespace) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    latest_alias = base_dir / "sessions" / "latest"
    pointer = {
        "session_path": _repo_relative_path(session_dir),
        "session_dir": str(session_dir.resolve()),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tag": args.tag,
        "search_mode": args.search_mode,
        "source": "session_run",
        "latest_alias_path": _repo_relative_unresolved_path(latest_alias),
    }
    try:
        latest_alias.parent.mkdir(parents=True, exist_ok=True)
        if latest_alias.is_symlink() or latest_alias.is_file():
            latest_alias.unlink()
        if not latest_alias.exists():
            latest_alias.symlink_to(session_dir.resolve(), target_is_directory=True)
    except OSError as error:
        pointer["latest_alias_error"] = str(error)
    (base_dir / "live_session.json").write_text(json.dumps(pointer, indent=2), encoding="utf-8")


def resolve_layout(args: argparse.Namespace) -> tuple[Path, Path | None, Path | None]:
    if args.session_dir is not None and args.candidate is None:
        raise ValueError(
            "Autoresearch session runs require --candidate pointing to an already-authored "
            "session-local candidates/passNN_*.py file. For pass01, copy the selected seed "
            "verbatim to candidates/pass01_baseline.py, then run that file explicitly."
        )

    if args.session_dir is not None:
        session_dir = args.session_dir.resolve()
        session_dir.mkdir(parents=True, exist_ok=True)
    else:
        session_dir = None

    if session_dir is None:
        if args.candidate is None:
            args.candidate = args.seed_candidate
        out_dir = args.out_dir
        results_path = None if args.no_record else args.results_path
        return out_dir, results_path, None

    args.candidate = args.candidate.resolve()
    candidates_dir = session_dir / "candidates"
    if not _is_relative_to(args.candidate, candidates_dir):
        raise ValueError(
            "Autoresearch session candidates must live under the session-local candidates/ "
            f"directory: {candidates_dir}"
        )

    out_dir = session_dir / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = None if args.no_record else (session_dir / "results.jsonl")
    log_path = session_dir / "outer_loop_log.md"
    if not log_path.exists():
        log_path.write_text(SESSION_LOG_TEMPLATE, encoding="utf-8")
    session_meta = {
        "session_dir": str(session_dir),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "search_mode": args.search_mode,
        "benchmark_path": str(args.benchmark.resolve()),
        "candidate_path": str(args.candidate.resolve()),
        "seed_candidate_path": str(args.seed_candidate.resolve()),
        "runs_dir": str(out_dir),
        "results_path": str(results_path) if results_path is not None else None,
        "log_path": str(log_path),
    }
    (session_dir / "session.json").write_text(json.dumps(session_meta, indent=2), encoding="utf-8")
    write_live_session_pointer(args.base_dir, session_dir, args)
    return out_dir, results_path, session_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one already-authored fixed-budget RL autoresearch candidate.")
    package_dir = Path(__file__).resolve().parents[1]
    base_dir = Path.cwd() / "autoresearch_runs"
    default_task_dir = package_dir / "tasks" / "bat_to_goal_v0"
    parser.set_defaults(base_dir=base_dir)
    parser.add_argument("--benchmark", type=Path, default=default_task_dir / "benchmark.json")
    parser.add_argument("--seed-candidate", type=Path, default=default_task_dir / "seed_trainable.py")
    parser.add_argument(
        "--candidate",
        type=Path,
        default=None,
        help=(
            "Candidate module to execute. In autoresearch sessions, pass an "
            "already-authored candidates/passNN_*.py file; pass01 should be a "
            "verbatim copy of the selected seed."
        ),
    )
    parser.add_argument("--tag", type=str, default="trial")
    parser.add_argument("--out-dir", type=Path, default=base_dir / "runs")
    parser.add_argument("--results-path", type=Path, default=base_dir / "results.jsonl")
    parser.add_argument("--train-episodes", type=int, default=None)
    parser.add_argument("--train-seconds", type=float, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument(
        "--headless-env",
        action="store_true",
        help=(
            "Attempt to override environment construction to render_mode=None. "
            "Panda/PyBullet envs that require a render mode keep the benchmark "
            "render_mode and record the fallback in the final JSON summary."
        ),
    )
    parser.add_argument(
        "--compact-status",
        action="store_true",
        help="Write compact periodic run status lines to stderr. Final stdout JSON is unchanged.",
    )
    parser.add_argument(
        "--compact-status-file",
        type=Path,
        default=None,
        help="Write the same compact status lines to this file so humans or agents can tail it while the run continues.",
    )
    parser.add_argument(
        "--status-interval-seconds",
        type=float,
        default=10.0,
        help="Minimum seconds between compact status lines.",
    )
    parser.add_argument("--no-record", action="store_true")
    parser.add_argument(
        "--search-mode",
        choices=["linear"],
        default="linear",
        help="Outer-loop search policy metadata for dashboards and logs.",
    )
    parser.add_argument("--mutation-family", type=str, default=None)
    parser.add_argument("--proposal-id", type=str, default=None)
    parser.add_argument("--prompt-strategy", type=str, default=None)
    parser.add_argument("--parent-run-id", type=str, default=None)
    parser.add_argument("--archive-role", type=str, default=None)
    parser.add_argument("--decision", type=str, default=None)
    parser.add_argument("--notes", type=str, default=None)
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--batch-candidate-index", type=int, default=None)
    parser.add_argument("--screening-stage", choices=["smoke", "confirmation"], default=None)
    parser.add_argument("--parent-kind", type=str, default=None)
    parser.add_argument("--confirmation-source-run-id", type=str, default=None)
    parser.add_argument("--session-dir", type=Path, default=None)
    init_group = parser.add_mutually_exclusive_group()
    init_group.add_argument("--init-checkpoint", type=Path, default=None)
    init_group.add_argument("--init-from-run", type=str, default=None)
    init_group.add_argument("--init-from-best", action="store_true")
    return parser.parse_args(argv)


def evolution_metadata(args: argparse.Namespace, init_checkpoint: Path | None) -> dict:
    metadata = {
        "search_mode": args.search_mode,
        "mutation_family": args.mutation_family,
        "proposal_id": args.proposal_id,
        "prompt_strategy": args.prompt_strategy,
        "declared_parent_run_id": args.parent_run_id,
        "archive_role": args.archive_role,
        "decision": args.decision,
        "notes": args.notes,
        "epoch": args.epoch,
        "batch_candidate_index": args.batch_candidate_index,
        "screening_stage": args.screening_stage,
        "parent_kind": args.parent_kind,
        "confirmation_source_run_id": args.confirmation_source_run_id,
        "uses_warm_start": init_checkpoint is not None,
    }
    return {key: value for key, value in metadata.items() if value is not None}


def append_outer_loop_log(session_dir: Path | None, summary: dict) -> None:
    if session_dir is None:
        return
    log_path = session_dir / "outer_loop_log.md"
    candidate = summary.get("candidate") if isinstance(summary.get("candidate"), dict) else {}
    train = summary.get("train") if isinstance(summary.get("train"), dict) else {}
    eval_summary = summary.get("eval") if isinstance(summary.get("eval"), dict) else {}
    objective = summary.get("objective") if isinstance(summary.get("objective"), dict) else {}
    lines = [
        "",
        f"## {summary.get('tag', summary.get('run_id', 'run'))}",
        "",
        f"- run_id: `{summary.get('run_id', '-')}`",
        f"- completed_at: {time.strftime('%Y-%m-%dT%H:%M:%S')}",
        f"- train_stop_reason: {train.get('stop_reason', '-')}",
        f"- train_steps: {train.get('total_steps', '-')}",
        f"- train_avg_return: {train.get('avg_return', '-')}",
        f"- eval_avg_return: {eval_summary.get('avg_return', '-')}",
        f"- eval_success_rate: {eval_summary.get('success_rate', '-')}",
        f"- score: {objective.get('value', summary.get('score', '-'))}",
    ]
    description = candidate.get("description") if isinstance(candidate, dict) else None
    if description:
        lines.extend(["", str(description)])
    lines.append("")
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    out_dir, results_path, session_dir = resolve_layout(args)
    args.out_dir = out_dir
    args.results_path = results_path if results_path is not None else args.results_path
    init_checkpoint = resolve_init_checkpoint(args)
    summary = run_experiment(
        benchmark_path=args.benchmark,
        candidate_path=args.candidate,
        tag=args.tag,
        out_dir=out_dir,
        results_path=results_path,
        train_episodes_override=args.train_episodes,
        train_seconds_override=args.train_seconds,
        eval_episodes_override=args.eval_episodes,
        init_checkpoint=init_checkpoint,
        session_dir=session_dir,
        evolution_metadata=evolution_metadata(args, init_checkpoint),
        headless_env=args.headless_env,
        status_interval_seconds=args.status_interval_seconds,
        compact_status=args.compact_status,
        compact_status_file=args.compact_status_file,
    )
    append_outer_loop_log(session_dir, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
