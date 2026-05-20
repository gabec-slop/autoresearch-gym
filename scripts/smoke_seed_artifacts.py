from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from check_trainable_contract import validate_records, validate_summary


@dataclass(frozen=True)
class SeedCase:
    name: str
    benchmark: str
    seed: str


SEED_CASES: tuple[SeedCase, ...] = (
    SeedCase(
        "inverted-pendulum",
        "autoresearch_gym/tasks/inverted_pendulum_v5/benchmark.json",
        "autoresearch_gym/tasks/inverted_pendulum_v5/seed_trainable.py",
    ),
    SeedCase(
        "hopper",
        "autoresearch_gym/tasks/hopper_v0/benchmark.json",
        "autoresearch_gym/tasks/hopper_v0/seed_trainable.py",
    ),
    SeedCase(
        "hopper-vector",
        "autoresearch_gym/tasks/hopper_v0/benchmark_vectorized_wall_clock.json",
        "autoresearch_gym/tasks/hopper_v0/seed_trainable_vectorized.py",
    ),
    SeedCase(
        "fetch-push",
        "autoresearch_gym/tasks/fetch_push_dense_v0/benchmark.json",
        "autoresearch_gym/tasks/fetch_push_dense_v0/seed_trainable.py",
    ),
    SeedCase(
        "fetch-push-her",
        "autoresearch_gym/tasks/fetch_push_dense_v0/benchmark.json",
        "autoresearch_gym/tasks/fetch_push_dense_v0/seed_trainable_her.py",
    ),
    SeedCase(
        "panda-pick",
        "autoresearch_gym/tasks/panda_pick_and_place_v0/benchmark.json",
        "autoresearch_gym/tasks/panda_pick_and_place_v0/seed_trainable.py",
    ),
    SeedCase(
        "panda-pick-her",
        "autoresearch_gym/tasks/panda_pick_and_place_v0/benchmark.json",
        "autoresearch_gym/tasks/panda_pick_and_place_v0/seed_trainable_her.py",
    ),
    SeedCase(
        "bat-to-goal",
        "autoresearch_gym/tasks/bat_to_goal_v0/benchmark.json",
        "autoresearch_gym/tasks/bat_to_goal_v0/seed_trainable.py",
    ),
    SeedCase(
        "bat-to-goal-vector",
        "autoresearch_gym/tasks/bat_to_goal_v0/benchmark_vectorized_wall_clock.json",
        "autoresearch_gym/tasks/bat_to_goal_v0/seed_trainable_vectorized.py",
    ),
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def repo_path(path: Path, value: str | None) -> Path | None:
    if not value:
        return None
    candidate = Path(value)
    return candidate if candidate.is_absolute() else path / candidate


def write_control(session_dir: Path, mode: str) -> None:
    live_dir = session_dir / "live"
    live_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "visual_mode": mode,
        "control_poll_seconds": 0.0,
        "jpeg_quality": 70,
    }
    if mode == "live_frame":
        payload["live_frame_interval_seconds"] = 0.0
    elif mode == "sampled_trajectory":
        payload["trajectory_sample_rate"] = 1.0
        payload["trajectory_frame_stride"] = 1
        payload["trajectory_playback_fps"] = 20.0
    (live_dir / "control.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def candidate_for_session(repo_root: Path, session_dir: Path, seed: str) -> Path:
    candidates_dir = session_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    candidate = candidates_dir / "pass01_baseline.py"
    shutil.copy2(repo_root / seed, candidate)
    return candidate


def run_case(repo_root: Path, case: SeedCase, mode: str, output_root: Path, timeout: float) -> dict[str, Any]:
    session_dir = output_root / "sessions" / f"{case.name}-{mode}"
    if session_dir.exists():
        shutil.rmtree(session_dir)
    session_dir.mkdir(parents=True)
    write_control(session_dir, mode)
    candidate = candidate_for_session(repo_root, session_dir, case.seed)
    cmd = [
        sys.executable,
        "-m",
        "autoresearch_gym.cli",
        "run",
        "--benchmark",
        case.benchmark,
        "--seed-candidate",
        case.seed,
        "--session-dir",
        str(session_dir),
        "--candidate",
        str(candidate),
        "--tag",
        f"artifact-smoke-{case.name}-{mode}",
        "--train-episodes",
        "1",
        "--eval-episodes",
        "1",
        "--no-train-probe",
        "--compact-status-file",
        str(session_dir / "live" / "status.log"),
    ]
    completed = subprocess.run(
        cmd,
        cwd=repo_root,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    result: dict[str, Any] = {
        "case": case.name,
        "mode": mode,
        "session_dir": str(session_dir),
        "returncode": completed.returncode,
        "errors": [],
    }
    if completed.returncode != 0:
        result["errors"].append(f"run exited {completed.returncode}: {completed.stderr[-2000:]}")
        return result

    runs = sorted((session_dir / "runs").glob("*/summary.json"))
    if not runs:
        result["errors"].append("missing run summary")
        return result
    summary_path = runs[-1]
    summary = read_json(summary_path)
    train_episodes_path = summary_path.parent / "train_episodes.json"
    metrics_path = session_dir / "live" / "current_run_metrics.json"
    result["run_id"] = summary_path.parent.name

    result["errors"].extend(validate_summary(summary, require_gradient_updates=True))
    if train_episodes_path.exists():
        result["errors"].extend(validate_records(read_json(train_episodes_path)))
    else:
        result["errors"].append("missing train_episodes.json")

    if not metrics_path.exists():
        result["errors"].append("missing live/current_run_metrics.json")
        return result
    metrics = read_json(metrics_path)
    current = metrics.get("current", {})
    for key in ("env_steps", "completed_episodes", "episode_batches", "avg_return", "success_rate"):
        if key not in current:
            result["errors"].append(f"live current missing {key}")
    losses = metrics.get("latest_losses")
    train = summary.get("train", {})
    if train.get("gradient_updates") is None:
        result["errors"].append("summary train missing gradient_updates")
    if isinstance(losses, dict) and losses and losses.get("gradient_updates") is None:
        result["errors"].append("live latest_losses missing gradient_updates")

    visual = metrics.get("visual", {})
    if mode == "live_frame":
        frame_path = repo_path(repo_root, visual.get("live_frame_path"))
        if frame_path is None or not frame_path.exists() or frame_path.stat().st_size <= 0:
            result["errors"].append("missing or empty live frame")
    elif mode == "sampled_trajectory":
        manifest_path = repo_path(repo_root, visual.get("trajectory_manifest_path"))
        if manifest_path is None or not manifest_path.exists():
            result["errors"].append("missing sampled trajectory manifest")
        else:
            manifest = read_json(manifest_path)
            frame_count = int(manifest.get("frame_count") or 0)
            if frame_count < 2:
                result["errors"].append(f"sampled trajectory has too few frames: {frame_count}")
            for frame in manifest.get("frames", []):
                frame_path = repo_path(repo_root, frame)
                if frame_path is None or not frame_path.exists() or frame_path.stat().st_size <= 0:
                    result["errors"].append(f"missing or empty sampled frame: {frame}")
                    break

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run artifact-level smoke tests for package seed trainables.")
    parser.add_argument("--output-root", type=Path, default=Path("/private/tmp/autoresearch-seed-artifact-smoke"))
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--case", action="append", choices=[case.name for case in SEED_CASES])
    parser.add_argument("--mode", action="append", choices=["sampled_trajectory", "live_frame"])
    args = parser.parse_args()

    repo_root = Path.cwd().resolve()
    selected_names = set(args.case or [case.name for case in SEED_CASES])
    selected_modes = tuple(args.mode or ("sampled_trajectory", "live_frame"))
    args.output_root.mkdir(parents=True, exist_ok=True)

    results = []
    for case in SEED_CASES:
        if case.name not in selected_names:
            continue
        for mode in selected_modes:
            result = run_case(repo_root, case, mode, args.output_root, args.timeout)
            results.append(result)
            status = "ok" if not result["errors"] else "FAIL"
            print(f"{case.name} {mode}: {status}", flush=True)
            for error in result["errors"]:
                print(f"  - {error}", flush=True)

    failures = [result for result in results if result["errors"]]
    print(json.dumps({"ok": not failures, "results": results}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
