from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat

try:
    from check_trainable_contract import validate_records, validate_summary
except ModuleNotFoundError:  # pragma: no cover - used when imported as scripts.smoke_seed_artifacts.
    from scripts.check_trainable_contract import validate_records, validate_summary


@dataclass(frozen=True)
class SeedCase:
    name: str
    benchmark: str
    seed: str
    visual_artifact_smoke: bool = True
    required_modules: tuple[str, ...] = ()
    benchmark_overrides: dict[str, Any] | None = None


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
        "panda-mjwarp",
        "autoresearch_gym/tasks/panda_pick_and_place_mjwarp_v0/benchmark.json",
        "autoresearch_gym/tasks/panda_pick_and_place_mjwarp_v0/seed_trainable.py",
        required_modules=("mujoco", "robot_descriptions"),
        benchmark_overrides={
            "max_steps": 4,
            "env_kwargs": {"backend": "mujoco", "num_envs": 1, "steps_per_env_per_iteration": 4},
        },
    ),
    SeedCase(
        "panda-mjwarp-tqc-her-ee",
        "autoresearch_gym/tasks/panda_pick_and_place_mjwarp_v0/benchmark.json",
        "autoresearch_gym/tasks/panda_pick_and_place_mjwarp_v0/seed_trainable_tqc_her_ee.py",
        visual_artifact_smoke=False,
        required_modules=("mujoco", "mujoco_warp", "robot_descriptions"),
        benchmark_overrides={
            "max_steps": 4,
            "env_kwargs": {
                "backend": "mujoco_warp",
                "num_envs": 512,
                "steps_per_env_per_iteration": 20,
            },
        },
    ),
    SeedCase(
        "panda-mjwarp-pandagym-dense",
        "autoresearch_gym/tasks/panda_pick_and_place_mjwarp_pandagym_dense_v0/benchmark.json",
        "autoresearch_gym/tasks/panda_pick_and_place_mjwarp_pandagym_dense_v0/seed_trainable.py",
        visual_artifact_smoke=False,
        required_modules=("mujoco", "mujoco_warp", "robot_descriptions"),
        benchmark_overrides={
            "max_steps": 4,
            "env_kwargs": {
                "backend": "mujoco_warp",
                "num_envs": 128,
                "steps_per_env_per_iteration": 8,
            },
        },
    ),
    SeedCase(
        "panda-mjwarp-pandagym-dense-tqc-her-ee",
        "autoresearch_gym/tasks/panda_pick_and_place_mjwarp_pandagym_dense_v0/benchmark.json",
        "autoresearch_gym/tasks/panda_pick_and_place_mjwarp_pandagym_dense_v0/seed_trainable_tqc_her_ee.py",
        visual_artifact_smoke=False,
        required_modules=("mujoco", "mujoco_warp", "robot_descriptions"),
        benchmark_overrides={
            "max_steps": 4,
            "env_kwargs": {
                "backend": "mujoco_warp",
                "num_envs": 128,
                "steps_per_env_per_iteration": 8,
            },
        },
    ),
    SeedCase(
        "panda-mjwarp-pandagym-dense-guided-warmup",
        "autoresearch_gym/tasks/panda_pick_and_place_mjwarp_pandagym_dense_v0/benchmark.json",
        "autoresearch_gym/tasks/panda_pick_and_place_mjwarp_pandagym_dense_v0/seed_trainable_guided_warmup.py",
        visual_artifact_smoke=False,
        required_modules=("mujoco", "mujoco_warp", "robot_descriptions"),
        benchmark_overrides={
            "max_steps": 4,
            "env_kwargs": {
                "backend": "mujoco_warp",
                "num_envs": 128,
                "steps_per_env_per_iteration": 8,
            },
        },
    ),
    SeedCase(
        "so101-reach-mujoco",
        "autoresearch_gym/tasks/so101_reach_mujoco_v0/benchmark.json",
        "autoresearch_gym/tasks/so101_reach_mujoco_v0/seed_trainable.py",
        required_modules=("mujoco",),
        benchmark_overrides={
            "max_steps": 4,
        },
    ),
    SeedCase(
        "so101-reach-vectorized-mujoco",
        "autoresearch_gym/tasks/so101_reach_mujoco_v0/benchmark_vectorized_wall_clock.json",
        "autoresearch_gym/tasks/so101_reach_mujoco_v0/seed_trainable_vectorized.py",
        required_modules=("mujoco",),
        benchmark_overrides={
            "max_steps": 4,
            "train_seconds": 2,
            "eval_episodes": 1,
        },
    ),
    SeedCase(
        "so101-reach-mjwarp-mujoco",
        "autoresearch_gym/tasks/so101_reach_mujoco_v0/benchmark_mjwarp_wall_clock.json",
        "autoresearch_gym/tasks/so101_reach_mujoco_v0/seed_trainable_vectorized.py",
        required_modules=("mujoco", "mujoco_warp"),
        benchmark_overrides={
            "max_steps": 4,
            "train_seconds": 2,
            "eval_episodes": 1,
            "env_kwargs": {"num_envs": 2},
        },
    ),
    SeedCase(
        "so101-reach-vision-mujoco",
        "autoresearch_gym/tasks/so101_reach_mujoco_v0/benchmark_vision.json",
        "autoresearch_gym/tasks/so101_reach_mujoco_v0/seed_trainable_pixel_actor_critic.py",
        required_modules=("mujoco",),
        benchmark_overrides={
            "max_steps": 4,
            "train_episodes": 1,
            "eval_episodes": 1,
        },
    ),
    SeedCase(
        "so101-cube-to-bin-mujoco",
        "autoresearch_gym/tasks/so101_cube_to_bin_mujoco_v0/benchmark.json",
        "autoresearch_gym/tasks/so101_cube_to_bin_mujoco_v0/seed_trainable.py",
        required_modules=("mujoco",),
        benchmark_overrides={
            "max_steps": 4,
        },
    ),
    SeedCase(
        "so101-cube-to-bin-mjwarp-mujoco",
        "autoresearch_gym/tasks/so101_cube_to_bin_mujoco_v0/benchmark_mjwarp_wall_clock.json",
        "autoresearch_gym/tasks/so101_cube_to_bin_mujoco_v0/seed_trainable_vectorized.py",
        required_modules=("mujoco", "mujoco_warp"),
        benchmark_overrides={
            "max_steps": 4,
            "train_seconds": 2,
            "eval_episodes": 1,
            "env_kwargs": {"num_envs": 2},
        },
    ),
    SeedCase(
        "so101-cube-to-bin-vision-mujoco",
        "autoresearch_gym/tasks/so101_cube_to_bin_mujoco_v0/benchmark_vision.json",
        "autoresearch_gym/tasks/so101_cube_to_bin_mujoco_v0/seed_trainable_pixel_actor_critic.py",
        required_modules=("mujoco",),
        benchmark_overrides={
            "max_steps": 4,
            "train_episodes": 1,
            "eval_episodes": 1,
        },
    ),
    SeedCase(
        "so101-vial-to-rack-mujoco",
        "autoresearch_gym/tasks/so101_vial_to_rack_mujoco_v0/benchmark.json",
        "autoresearch_gym/tasks/so101_vial_to_rack_mujoco_v0/seed_trainable.py",
        required_modules=("mujoco",),
        benchmark_overrides={
            "max_steps": 4,
        },
    ),
    SeedCase(
        "so101-vial-to-rack-mjwarp-mujoco",
        "autoresearch_gym/tasks/so101_vial_to_rack_mujoco_v0/benchmark_mjwarp_wall_clock.json",
        "autoresearch_gym/tasks/so101_vial_to_rack_mujoco_v0/seed_trainable_vectorized.py",
        required_modules=("mujoco", "mujoco_warp"),
        benchmark_overrides={
            "max_steps": 4,
            "train_seconds": 2,
            "eval_episodes": 1,
            "env_kwargs": {"num_envs": 2},
        },
    ),
    SeedCase(
        "so101-vial-to-rack-vision-mujoco",
        "autoresearch_gym/tasks/so101_vial_to_rack_mujoco_v0/benchmark_vision.json",
        "autoresearch_gym/tasks/so101_vial_to_rack_mujoco_v0/seed_trainable_pixel_actor_critic.py",
        required_modules=("mujoco",),
        benchmark_overrides={
            "max_steps": 4,
            "train_episodes": 1,
            "eval_episodes": 1,
        },
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
    SeedCase(
        "unitree-g1-mjlab",
        "autoresearch_gym/tasks/unitree_g1_motion_mirror_v0/benchmark.json",
        "autoresearch_gym/tasks/unitree_g1_motion_mirror_v0/seed_trainable.py",
        visual_artifact_smoke=False,
        benchmark_overrides={
            "max_steps": 8,
            "env_kwargs": {"dry_run": True},
            "execution_backend": {"dry_run": True},
        },
    ),
    SeedCase(
        "unitree-g1-lower-level",
        "autoresearch_gym/tasks/unitree_g1_motion_mirror_v0/benchmark_lower_level.json",
        "autoresearch_gym/tasks/unitree_g1_motion_mirror_v0/seed_trainable_lower_level_cleanrl.py",
        visual_artifact_smoke=False,
        benchmark_overrides={
            "max_steps": 8,
            "env_kwargs": {"num_envs": 2, "steps_per_env_per_iteration": 4},
        },
    ),
    SeedCase(
        "unitree-go2-mjlab",
        "autoresearch_gym/tasks/unitree_go2_rough_locomotion_v0/benchmark.json",
        "autoresearch_gym/tasks/unitree_go2_rough_locomotion_v0/seed_trainable.py",
        visual_artifact_smoke=False,
        benchmark_overrides={
            "max_steps": 8,
            "env_kwargs": {"dry_run": True},
            "execution_backend": {"dry_run": True},
        },
    ),
    SeedCase(
        "unitree-go2-mjlab-staged",
        "autoresearch_gym/tasks/unitree_go2_rough_locomotion_v0/benchmark.json",
        "autoresearch_gym/tasks/unitree_go2_rough_locomotion_v0/seed_trainable_staged_curriculum.py",
        visual_artifact_smoke=False,
        benchmark_overrides={
            "max_steps": 8,
            "env_kwargs": {"dry_run": True},
            "execution_backend": {"dry_run": True},
        },
    ),
    SeedCase(
        "unitree-go2-lower-level",
        "autoresearch_gym/tasks/unitree_go2_rough_locomotion_v0/benchmark_lower_level.json",
        "autoresearch_gym/tasks/unitree_go2_rough_locomotion_v0/seed_trainable_lower_level_cleanrl.py",
        visual_artifact_smoke=False,
        benchmark_overrides={
            "max_steps": 8,
            "env_kwargs": {"num_envs": 2, "steps_per_env_per_iteration": 4},
        },
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
        "render_width": 720,
        "render_height": 480,
    }
    if mode == "live_frame":
        payload["live_frame_interval_seconds"] = 0.0
    elif mode == "sampled_trajectory":
        payload["trajectory_sample_rate"] = 1.0
        payload["trajectory_frame_stride"] = 1
        payload["trajectory_playback_fps"] = 20.0
    (live_dir / "control.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def deep_update(payload: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(payload)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def benchmark_for_session(repo_root: Path, session_dir: Path, case: SeedCase) -> Path:
    source = repo_root / case.benchmark
    if not case.benchmark_overrides:
        return source
    benchmark_dir = session_dir / "benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    payload = deep_update(read_json(source), case.benchmark_overrides)
    eval_case_bank = payload.get("eval_case_bank")
    if eval_case_bank:
        source_eval_cases = source.parent / str(eval_case_bank)
        if source_eval_cases.exists():
            shutil.copy2(source_eval_cases, benchmark_dir / str(eval_case_bank))
    target = benchmark_dir / source.name
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def candidate_for_session(repo_root: Path, session_dir: Path, seed: str) -> Path:
    candidates_dir = session_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    candidate = candidates_dir / "pass01_baseline.py"
    shutil.copy2(repo_root / seed, candidate)
    return candidate


def visual_artifact_smoke_enabled(case: SeedCase) -> bool:
    if not case.visual_artifact_smoke:
        return False
    if os.environ.get("AUTORESEARCH_SMOKE_VISUALS") == "1":
        return True
    return sys.platform != "darwin"


def missing_required_modules(case: SeedCase) -> list[str]:
    return [module for module in case.required_modules if find_spec(module) is None]


def validate_image_file(path: Path, label: str, expected_size: tuple[int, int] | None = None) -> list[str]:
    errors: list[str] = []
    try:
        with Image.open(path) as image:
            width, height = image.size
            if width <= 1 or height <= 1:
                errors.append(f"{label} image is too small: {width}x{height}")
                return errors
            if expected_size is not None and (width, height) != expected_size:
                errors.append(f"{label} image has wrong size: {width}x{height}, expected {expected_size[0]}x{expected_size[1]}")
            stat = ImageStat.Stat(image.convert("RGB"))
            if sum(float(value) for value in stat.var) <= 0.0:
                errors.append(f"{label} image is blank/uniform: {path}")
    except Exception as exc:
        errors.append(f"{label} image is unreadable: {path}: {exc}")
    return errors


def run_case(repo_root: Path, case: SeedCase, mode: str, output_root: Path, timeout: float) -> dict[str, Any]:
    session_dir = output_root / "sessions" / f"{case.name}-{mode}"
    if session_dir.exists():
        shutil.rmtree(session_dir)
    session_dir.mkdir(parents=True)
    missing_modules = missing_required_modules(case)
    if missing_modules:
        return {
            "case": case.name,
            "mode": mode,
            "session_dir": str(session_dir),
            "returncode": None,
            "skipped": f"missing optional modules: {', '.join(missing_modules)}",
            "errors": [],
        }
    write_control(session_dir, mode)
    candidate = candidate_for_session(repo_root, session_dir, case.seed)
    benchmark = benchmark_for_session(repo_root, session_dir, case)
    cmd = [
        sys.executable,
        "-m",
        "autoresearch_gym.cli",
        "run",
        "--benchmark",
        str(benchmark),
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
        "--no-dashboard",
        "--no-train-probe",
        "--compact-status-file",
        str(session_dir / "live" / "status.log"),
    ]
    visual_enabled = visual_artifact_smoke_enabled(case)
    if not visual_enabled:
        cmd.append("--headless-env")
    result: dict[str, Any] = {
        "case": case.name,
        "mode": mode,
        "session_dir": str(session_dir),
        "returncode": None,
        "errors": [],
    }
    try:
        completed = subprocess.run(
            cmd,
            cwd=repo_root,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        result["returncode"] = 124
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        result["errors"].append(
            f"run timed out after {timeout:.1f}s: stdout={str(stdout)[-1000:]} stderr={str(stderr)[-1000:]}"
        )
        return result
    result["returncode"] = completed.returncode
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
    if not visual_enabled:
        result["visual_artifact_smoke"] = "skipped-headless"
    elif mode == "live_frame":
        frame_path = repo_path(repo_root, visual.get("live_frame_path"))
        if frame_path is None or not frame_path.exists() or frame_path.stat().st_size <= 0:
            result["errors"].append("missing or empty live frame")
        else:
            result["errors"].extend(validate_image_file(frame_path, "live frame", expected_size=(720, 480)))
        live_feed_paths = visual.get("live_feed_paths")
        if isinstance(live_feed_paths, dict) and live_feed_paths:
            for feed_name, feed_path_value in live_feed_paths.items():
                feed_path = repo_path(repo_root, feed_path_value)
                if feed_path is None or not feed_path.exists() or feed_path.stat().st_size <= 0:
                    result["errors"].append(f"missing or empty live feed {feed_name}: {feed_path_value}")
                    break
                image_errors = validate_image_file(feed_path, f"live feed {feed_name}")
                if image_errors:
                    result["errors"].extend(image_errors)
                    break
    elif mode == "sampled_trajectory":
        manifest_path = repo_path(repo_root, visual.get("trajectory_manifest_path"))
        if manifest_path is None or not manifest_path.exists():
            result["errors"].append("missing sampled trajectory manifest")
        else:
            manifest = read_json(manifest_path)
            if manifest.get("width") != 720 or manifest.get("height") != 480:
                result["errors"].append(
                    f"sampled trajectory manifest has wrong size: {manifest.get('width')}x{manifest.get('height')}"
                )
            frame_count = int(manifest.get("frame_count") or 0)
            if frame_count < 2:
                result["errors"].append(f"sampled trajectory has too few frames: {frame_count}")
            for frame in manifest.get("frames", []):
                frame_path = repo_path(repo_root, frame)
                if frame_path is None or not frame_path.exists() or frame_path.stat().st_size <= 0:
                    result["errors"].append(f"missing or empty sampled frame: {frame}")
                    break
                image_errors = validate_image_file(frame_path, f"sampled frame {frame}", expected_size=(720, 480))
                if image_errors:
                    result["errors"].extend(image_errors)
                    break
            steps = manifest.get("steps")
            if isinstance(steps, list) and steps:
                first_feeds = steps[0].get("feeds") if isinstance(steps[0], dict) else None
                if not isinstance(first_feeds, dict) or not first_feeds:
                    result["errors"].append("sampled trajectory step missing feeds")
                else:
                    for feed_name, feed_path_value in first_feeds.items():
                        feed_path = repo_path(repo_root, feed_path_value)
                        if feed_path is None or not feed_path.exists() or feed_path.stat().st_size <= 0:
                            result["errors"].append(f"missing or empty sampled feed {feed_name}: {feed_path_value}")
                            break
                        image_errors = validate_image_file(feed_path, f"sampled feed {feed_name}")
                        if image_errors:
                            result["errors"].extend(image_errors)
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
            status = "SKIP" if result.get("skipped") else "ok" if not result["errors"] else "FAIL"
            print(f"{case.name} {mode}: {status}", flush=True)
            if result.get("skipped"):
                print(f"  - {result['skipped']}", flush=True)
            for error in result["errors"]:
                print(f"  - {error}", flush=True)

    failures = [result for result in results if result["errors"]]
    print(json.dumps({"ok": not failures, "results": results}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
