from __future__ import annotations

from importlib import resources

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from autoresearch_gym.external.base import ArtifactSet, CommandSpec, RunBundle
from autoresearch_gym.runner.curves import make_train_episode_record

DASHBOARD_FRAME_WIDTH = 720
DASHBOARD_FRAME_HEIGHT = 480
DASHBOARD_FRAME_SIZE = (DASHBOARD_FRAME_WIDTH, DASHBOARD_FRAME_HEIGHT)
DEFAULT_TRAJECTORY_PLAYBACK_FPS = 20.0


BRIDGE_PACKAGE = "autoresearch_gym.external.mjlab_bridges"
MJLAB_ROLLOUT_BRIDGE = "mjlab_rollout_bridge.py"
MJLAB_TRAIN_BRIDGE = "mjlab_train_bridge.py"


def _bridge_source_text(filename: str) -> str:
    return resources.files(BRIDGE_PACKAGE).joinpath(filename).read_text(encoding="utf-8")


def _copy_bridge_script(filename: str, destination: Path) -> None:
    resource = resources.files(BRIDGE_PACKAGE).joinpath(filename)
    with resources.as_file(resource) as source:
        shutil.copy2(source, destination)


class UnitreeExternalBackend:
    """Unitree external backend scaffold for MJLab/Isaac-style task deployment.

    This backend validates that the configured upstream assets exist and produces
    normalized train/eval/media artifacts. Simulator-specific PPO commands can
    replace the worker internals without changing the runner or target contract.
    """

    def build_bundle(self, bundle: RunBundle) -> RunBundle:
        bundle.external_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": bundle.run_id,
            "tag": bundle.tag,
            "task_family": bundle.execution_backend.get("task_family", "unitree"),
            "required_paths": bundle.execution_backend.get("required_paths", []),
            "dry_run": bool(bundle.execution_backend.get("dry_run", True)),
            "benchmark": {
                "name": bundle.benchmark.name,
                "env_id": bundle.benchmark.env_id,
                "env_kwargs": bundle.benchmark.env_kwargs,
                "train_episodes": bundle.train_episodes,
                "train_seconds": bundle.train_seconds,
                "eval_episodes": bundle.eval_episodes,
                "max_steps": bundle.max_steps,
                "primary_metric": bundle.benchmark.primary_metric,
            },
            "candidate": bundle.candidate_metadata,
            "eval_cases": bundle.eval_cases,
            "compact_status_file": str(bundle.compact_status_file) if bundle.compact_status_file is not None else None,
            "session_dir": str(bundle.session_dir) if bundle.session_dir is not None else None,
            "repo_root": str(Path.cwd()),
        }
        (bundle.external_dir / "bundle.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return bundle

    def training_command(self, bundle: RunBundle) -> CommandSpec:
        command = self._command("train", bundle)
        if bundle.train_seconds is not None:
            command.timeout_seconds = max(float(command.timeout_seconds or 0.0), float(bundle.train_seconds) + 900.0)
        return command

    def eval_command(self, bundle: RunBundle, checkpoint_path: Path) -> CommandSpec:
        return self._command("eval", bundle, checkpoint_path)

    def media_command(self, bundle: RunBundle, checkpoint_path: Path) -> CommandSpec | None:
        return self._command("media", bundle, checkpoint_path)

    def normalize_train(self, artifacts: ArtifactSet) -> dict[str, Any]:
        return json.loads((artifacts.root / "train_result.json").read_text(encoding="utf-8"))

    def normalize_eval(self, artifacts: ArtifactSet) -> dict[str, Any]:
        return json.loads((artifacts.root / "eval_result.json").read_text(encoding="utf-8"))

    def normalize_media(self, artifacts: ArtifactSet) -> dict[str, Any]:
        path = artifacts.root / "media_result.json"
        if not path.exists():
            return {"media_available": False}
        payload = json.loads(path.read_text(encoding="utf-8"))
        frame_path = artifacts.root / "current_run_frame.jpg"
        if frame_path.exists():
            payload["live_frame_path"] = _dashboard_path(frame_path)
            visual = payload.setdefault("visual", {})
            if isinstance(visual, dict):
                visual["live_frame_path"] = _dashboard_path(frame_path)
        manifest_path = artifacts.root / "trajectories" / "sample_000001" / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            frame_dir = manifest_path.parent
            local_frames = sorted(frame_dir.glob("frame_*.jpg"))
            manifest["frames"] = [_dashboard_path(frame) for frame in local_frames]
            manifest["latest_frame_path"] = manifest["frames"][-1] if manifest["frames"] else None
            manifest["frame_count"] = len(manifest["frames"])
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            payload["trajectory_manifest_path"] = _dashboard_path(manifest_path)
            payload["trajectory_latest_frame_path"] = manifest["latest_frame_path"]
            visual = payload.setdefault("visual", {})
            if isinstance(visual, dict):
                visual["trajectory_manifest_path"] = payload["trajectory_manifest_path"]
                visual["trajectory_latest_frame_path"] = payload["trajectory_latest_frame_path"]
                visual["sampled_status"] = manifest.get("status", "completed")
                visual["latest_sample_index"] = manifest.get("sample_index", 1)
        return payload

    def _command(self, mode: str, bundle: RunBundle, checkpoint_path: Path | None = None) -> CommandSpec:
        argv = [
            sys.executable,
            "-m",
            "autoresearch_gym.external.unitree_backend",
            "--mode",
            mode,
            "--bundle",
            str(bundle.external_dir / "bundle.json"),
            "--out-dir",
            str(bundle.external_dir),
        ]
        if checkpoint_path is not None:
            argv.extend(["--checkpoint", str(checkpoint_path)])
        return CommandSpec(argv=argv, cwd=Path.cwd(), label=f"unitree-{mode}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _dashboard_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _check_required_paths(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    checks = []
    for raw in bundle.get("required_paths") or []:
        path = Path(str(raw))
        checks.append({"path": str(path), "exists": path.exists()})
    return checks


def _repo_root(bundle: dict[str, Any]) -> Path:
    raw = str(bundle.get("repo_root") or Path.cwd())
    if os.name == "nt" and raw.startswith("/") and ":" not in raw[:4]:
        return Path.cwd().resolve()
    configured = Path(raw)
    if configured.exists():
        return configured.resolve()
    return Path.cwd().resolve()


def _unitree_root(bundle: dict[str, Any]) -> Path:
    return _repo_root(bundle) / ".external" / "unitree_rl_mjlab"


def _mjlab_root(bundle: dict[str, Any]) -> Path:
    return _repo_root(bundle) / ".external" / "mjlab"


def _mjlab_python(bundle: dict[str, Any]) -> Path | str:
    configured = os.environ.get("UNITREE_MJLAB_PYTHON")
    if configured:
        return configured
    scripts = "Scripts" if os.name == "nt" else "bin"
    suffix = "python.exe" if os.name == "nt" else "python"
    candidate = _mjlab_root(bundle) / ".venv" / scripts / suffix
    return candidate if candidate.exists() else sys.executable


def _task_id(bundle: dict[str, Any]) -> str:
    task_family = str(bundle.get("task_family", "unitree"))
    if task_family == "g1_motion_mirror":
        return "Unitree-G1-Tracking-No-State-Estimation"
    if task_family == "go2_rough_locomotion":
        return "Unitree-Go2-Rough"
    raise ValueError(f"unknown Unitree task_family: {task_family}")


def _experiment_name(bundle: dict[str, Any]) -> str:
    return "g1_tracking" if str(bundle.get("task_family")) == "g1_motion_mirror" else "go2_velocity"


def _motion_file(bundle: dict[str, Any]) -> str | None:
    if str(bundle.get("task_family")) != "g1_motion_mirror":
        return None
    for raw in bundle.get("required_paths") or []:
        if str(raw).endswith(".npz"):
            path = Path(str(raw))
            return str(path if path.is_absolute() else (_repo_root(bundle) / path))
    return None


def _candidate_recipe(bundle: dict[str, Any]) -> dict[str, Any]:
    candidate = bundle.get("candidate") if isinstance(bundle.get("candidate"), dict) else {}
    recipe = candidate.get("recipe") if isinstance(candidate.get("recipe"), dict) else {}
    return recipe


def _recipe_section(recipe: dict[str, Any], key: str) -> dict[str, Any]:
    value = recipe.get(key)
    return value if isinstance(value, dict) else {}


def _parallel_env_count(bundle: dict[str, Any], *, for_eval: bool = False) -> int:
    recipe = _candidate_recipe(bundle)
    runner = _recipe_section(recipe, "runner")
    environment = _recipe_section(recipe, "environment")
    env_kwargs = bundle.get("benchmark", {}).get("env_kwargs", {})
    key = "eval_num_envs" if for_eval else "num_envs"
    value = None
    if for_eval:
        value = runner.get("eval_num_envs") or environment.get("eval_num_envs")
    else:
        value = recipe.get("num_envs")
        if value is None:
            value = runner.get("num_envs") or environment.get("num_envs")
    if value is None and isinstance(env_kwargs, dict):
        value = env_kwargs.get(key)
    if value is None:
        value = 16 if for_eval else 1024
    return max(1, int(value))


def _steps_per_env(bundle: dict[str, Any]) -> int:
    recipe = _candidate_recipe(bundle)
    runner = _recipe_section(recipe, "runner")
    env_kwargs = bundle.get("benchmark", {}).get("env_kwargs", {})
    value = recipe.get("steps_per_env_per_iteration") or runner.get("steps_per_env_per_iteration") or runner.get("num_steps_per_env") or (
        env_kwargs.get("steps_per_env_per_iteration") if isinstance(env_kwargs, dict) else None
    )
    return max(1, int(value or 24))


def _learning_iterations(bundle: dict[str, Any]) -> int:
    recipe = _candidate_recipe(bundle)
    runner = _recipe_section(recipe, "runner")
    value = recipe.get("max_iterations") or recipe.get("learning_iterations") or runner.get("max_iterations") or runner.get("learning_iterations")
    if value is None:
        benchmark = bundle.get("benchmark", {})
        train_seconds = benchmark.get("train_seconds")
        if train_seconds is not None:
            num_envs = _parallel_env_count(bundle, for_eval=False)
            task_family = str(bundle.get("task_family") or bundle.get("execution_backend", {}).get("task_family", ""))
            default_seconds_per_iteration = 2.6 if "g1" in task_family else 1.8 * max(1.0, num_envs / 1024.0)
            seconds_per_iteration = float(runner.get("seconds_per_iteration_estimate") or default_seconds_per_iteration)
            value = max(1, int(float(train_seconds) / max(0.1, seconds_per_iteration)))
        else:
            value = benchmark.get("train_episodes", 1)
    return max(1, int(value))


def _mjlab_probe_interval_iterations(recipe: dict[str, Any], bundle: dict[str, Any], iterations: int) -> int:
    runner = _recipe_section(recipe, "runner")
    explicit = runner.get("probe_interval_iterations")
    if explicit is not None:
        return max(0, int(explicit))
    target_raw = runner.get("target_policy_probe_count") or os.environ.get("UNITREE_MJLAB_TARGET_POLICY_PROBES") or 5
    target = max(1, int(target_raw))
    return max(1, int(math.ceil(max(1, int(iterations)) / target)))


def _seed(bundle: dict[str, Any], default: int) -> int:
    recipe = _candidate_recipe(bundle)
    runner = _recipe_section(recipe, "runner")
    return int(recipe.get("seed") or runner.get("seed") or bundle.get("benchmark", {}).get("seed") or default)


def _external_env(bundle: dict[str, Any]) -> dict[str, str]:
    current = os.environ.copy()
    paths = [
        str(_mjlab_root(bundle) / "src"),
        str(_unitree_root(bundle)),
        str(_repo_root(bundle)),
    ]
    existing = current.get("PYTHONPATH")
    if existing:
        paths.append(existing)
    current["PYTHONPATH"] = os.pathsep.join(paths)
    current["MUJOCO_GL"] = "glfw" if os.name == "nt" else "egl"
    current.setdefault("PYTHONIOENCODING", "utf-8")
    current.setdefault("PYTHONUTF8", "1")
    current.setdefault("WANDB_MODE", "offline")
    return current


def _run_subprocess(argv: list[str], *, cwd: Path, env: dict[str, str], timeout: float | None, stdout_path: Path, stderr_path: Path) -> subprocess.CompletedProcess[str]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [str(arg) for arg in argv],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )
    stdout = result.stdout or ""
    stderr = result.stderr or ""
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            f"MJLab command failed with exit code {result.returncode}: {' '.join(map(str, argv))}\n"
            f"stdout:\n{stdout[-2000:]}\n"
            f"stderr:\n{stderr[-2000:]}"
        )
    return result


def _find_latest_run_dir(bundle: dict[str, Any], started_at: float) -> Path:
    root = _unitree_root(bundle) / "logs" / "rsl_rl" / _experiment_name(bundle)
    candidates = [path for path in root.glob("*") if path.is_dir()] if root.exists() else []
    run_id = str(bundle["run_id"])
    tagged = [path for path in candidates if run_id in path.name]
    if tagged:
        return max(tagged, key=lambda path: path.stat().st_mtime)
    recent = [path for path in candidates if path.stat().st_mtime >= started_at - 5]
    if recent:
        return max(recent, key=lambda path: path.stat().st_mtime)
    raise FileNotFoundError(f"could not locate MJLab run directory under {root}")


def _checkpoint_from_run_dir(run_dir: Path) -> Path:
    checkpoints = list(run_dir.glob("model_*.pt"))
    if not checkpoints:
        checkpoints = list(run_dir.rglob("model_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"no model_*.pt checkpoint was written under {run_dir}")

    def checkpoint_index(path: Path) -> int:
        match = re.search(r"model_(\d+)\.pt$", path.name)
        return int(match.group(1)) if match else -1

    return max(checkpoints, key=lambda path: (checkpoint_index(path), path.stat().st_mtime))


def _write_status(bundle: dict[str, Any], out_dir: Path, status: str, total_steps: int = 0) -> None:
    line = f"st={status} step={total_steps} task={bundle.get('task_family', 'unitree')}"
    status_path = bundle.get("compact_status_file")
    if status_path:
        path = Path(status_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    if bundle.get("session_dir"):
        live_dir = Path(bundle["session_dir"]) / "live"
        live_dir.mkdir(parents=True, exist_ok=True)
        with (live_dir / "status.log").open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def _validate(bundle: dict[str, Any], out_dir: Path) -> None:
    checks = _check_required_paths(bundle)
    _write_json(out_dir / "preflight_result.json", {"checks": checks, "ok": all(check["exists"] for check in checks)})
    if bundle.get("dry_run"):
        return
    missing = [check["path"] for check in checks if not check["exists"]]
    if missing:
        raise FileNotFoundError("Missing required Unitree external paths: " + ", ".join(missing))


def _task_metrics(task_family: str, idx: int) -> dict[str, float]:
    if task_family == "g1_motion_mirror":
        return {
            "mpkpe": 0.08 + idx * 0.005,
            "r_mpkpe": 0.05 + idx * 0.004,
            "foot_slip": 0.02 + idx * 0.002,
            "fall_rate": 0.0,
        }
    return {
        "command_tracking_error": 0.18 + idx * 0.01,
        "survival_rate": 1.0,
        "fall_rate": 0.0,
        "energy_cost": 0.4 + idx * 0.02,
    }


def _train_last_metrics(task_family: str, records: list[dict[str, Any]]) -> dict[str, float]:
    metrics = {"gradient_updates": 0.0, "external_backend": 1.0}
    if not records:
        return metrics
    info_records = [record.get("info_metrics", {}) for record in records if isinstance(record.get("info_metrics"), dict)]
    if task_family == "g1_motion_mirror":
        mpkpe_values = [float(info["mpkpe"]) for info in info_records if "mpkpe" in info]
        root_values = [float(info["r_mpkpe"]) for info in info_records if "r_mpkpe" in info]
        foot_values = [float(info["foot_slip"]) for info in info_records if "foot_slip" in info]
        if mpkpe_values:
            metrics["mpkpe_error"] = float(np.mean(mpkpe_values))
            metrics["tracking_error"] = float(np.mean(mpkpe_values))
        if root_values:
            metrics["root_mpkpe_error"] = float(np.mean(root_values))
        if foot_values:
            metrics["foot_slip_error"] = float(np.mean(foot_values))
        return metrics

    command_values = [float(info["command_tracking_error"]) for info in info_records if "command_tracking_error" in info]
    energy_values = [float(info["energy_cost"]) for info in info_records if "energy_cost" in info]
    if command_values:
        metrics["command_tracking_error"] = float(np.mean(command_values))
        metrics["tracking_error"] = float(np.mean(command_values))
    if energy_values:
        metrics["energy_error"] = float(np.mean(energy_values))
    return metrics


def _run_train(bundle: dict[str, Any], out_dir: Path) -> None:
    _validate(bundle, out_dir)
    if not bundle.get("dry_run"):
        _run_mjlab_train(bundle, out_dir)
        return
    task_family = str(bundle.get("task_family", "unitree"))
    episodes = max(1, min(int(bundle["benchmark"]["train_episodes"]), 3))
    records = []
    for idx in range(episodes):
        metrics = _task_metrics(task_family, idx)
        return_value = 1.0 - float(metrics.get("mpkpe", metrics.get("command_tracking_error", 0.1)))
        records.append(
            make_train_episode_record(
                episode=idx + 1,
                return_value=return_value,
                length=24,
                success=True,
                step=(idx + 1) * 24,
                elapsed_seconds=0.05 * (idx + 1),
                info_metrics=metrics,
            )
        )
    total_steps = int(sum(record["length"] for record in records))
    payload = {
        "episode_records": records,
        "total_steps": total_steps,
        "env_steps": total_steps,
        "episodes_completed": episodes,
        "completed_episodes": episodes,
        "episode_batches": episodes,
        "gradient_updates": 0,
        "last_metrics": _train_last_metrics(task_family, records),
        "stop_reason": "external_smoke_complete",
        "external_backend_scaffold": True,
    }
    _write_json(out_dir / "train_result.json", payload)
    (out_dir / "agent_checkpoint.pt").write_text(f"{task_family} external scaffold checkpoint\n", encoding="utf-8")
    _write_status(bundle, out_dir, "run", total_steps)


def _run_eval(bundle: dict[str, Any], out_dir: Path) -> None:
    _validate(bundle, out_dir)
    if not bundle.get("dry_run"):
        checkpoint = out_dir / "agent_checkpoint.pt"
        _run_mjlab_rollout(bundle, out_dir, checkpoint, mode="eval")
        return
    task_family = str(bundle.get("task_family", "unitree"))
    cases = bundle.get("eval_cases") or []
    episodes = int(bundle["benchmark"]["eval_episodes"])
    records = []
    for idx in range(episodes):
        case = cases[idx] if idx < len(cases) else {}
        metrics = _task_metrics(task_family, idx)
        if task_family == "g1_motion_mirror":
            episode_return = -float(metrics["mpkpe"])
            success = bool(metrics["mpkpe"] < 0.12)
        else:
            episode_return = 1.0 - float(metrics["command_tracking_error"]) - 0.05 * float(metrics["energy_cost"])
            success = bool(metrics["fall_rate"] == 0.0 and metrics["survival_rate"] >= 1.0)
        records.append(
            {
                "episode": idx + 1,
                "seed": 9000 + idx,
                "return": float(episode_return),
                "length": int(bundle["benchmark"]["max_steps"]),
                "success": success,
                "case_label": str(case.get("name", f"case-{idx + 1:02d}")),
                "info_metrics": metrics,
            }
        )
    summary = {
        "episodes": episodes,
        "avg_return": float(np.mean([record["return"] for record in records])) if records else 0.0,
        "avg_length": float(np.mean([record["length"] for record in records])) if records else 0.0,
        "success_rate": float(np.mean([1.0 if record["success"] else 0.0 for record in records])) if records else 0.0,
        "episode_records": records,
    }
    for key in records[0]["info_metrics"].keys() if records else []:
        summary[f"avg_{key}"] = float(np.mean([record["info_metrics"][key] for record in records]))
    _write_json(out_dir / "eval_result.json", summary)
    _write_status(bundle, out_dir, "eval", 0)


def _run_mjlab_train(bundle: dict[str, Any], out_dir: Path) -> None:
    task_id = _task_id(bundle)
    recipe = _candidate_recipe(bundle)
    num_envs = _parallel_env_count(bundle)
    steps_per_env = _steps_per_env(bundle)
    iterations = _learning_iterations(bundle)
    seed = _seed(bundle, 42)
    unitree_root = _unitree_root(bundle)
    log_dir = out_dir / "command_logs"
    started_at = time.time()
    script = _write_train_script(out_dir).resolve()
    rollout_script = _write_rollout_script(out_dir).resolve()
    recipe_path = (out_dir / "mjlab_recipe.json").resolve()
    resolved_path = (out_dir / "mjlab_resolved_config.json").resolve()
    _write_json(recipe_path, recipe)
    probe_interval = _mjlab_probe_interval_iterations(recipe, bundle, iterations)
    probe_num_envs = min(_parallel_env_count(bundle, for_eval=True), int(os.environ.get("UNITREE_MJLAB_PROBE_NUM_ENVS_MAX", "16")))
    probe_steps = min(int(bundle["benchmark"].get("max_steps") or 200), int(os.environ.get("UNITREE_MJLAB_PROBE_STEPS_MAX", "120")))
    argv: list[str | Path] = [
        _mjlab_python(bundle),
        script,
        "--task-id",
        task_id,
        "--run-id",
        str(bundle["run_id"]),
        "--num-envs",
        str(num_envs),
        "--iterations",
        str(iterations),
        "--seed",
        str(seed),
        "--gpu-id",
        "0",
        "--recipe-json",
        recipe_path,
        "--resolved-json",
        resolved_path,
        "--out-dir",
        out_dir.resolve(),
        "--steps-per-env",
        str(steps_per_env),
        "--budget-mode",
        "time" if bundle["benchmark"].get("train_seconds") is not None else "episodes",
        "--rollout-script",
        rollout_script,
        "--probe-interval-iterations",
        str(probe_interval),
        "--probe-num-envs",
        str(probe_num_envs),
        "--probe-steps",
        str(probe_steps),
        "--sample-rollout-frame-count",
        str(int(_recipe_section(recipe, "runner").get("sample_rollout_frame_count") or 24)),
        "--sample-trajectory-source",
        str(_recipe_section(recipe, "runner").get("sample_trajectory_source") or "runner_eval"),
    ]
    if bundle["benchmark"].get("train_seconds") is not None:
        argv.extend(["--train-seconds", str(bundle["benchmark"].get("train_seconds"))])
    motion_file = _motion_file(bundle)
    if motion_file:
        argv.extend(["--motion-file", motion_file])
    _run_subprocess(
        [str(arg) for arg in argv],
        cwd=unitree_root,
        env=_external_env(bundle),
        timeout=float(
            os.environ.get(
                "UNITREE_MJLAB_TRAIN_TIMEOUT_SECONDS",
                str(float(bundle["benchmark"].get("train_seconds") or 900.0) + 900.0),
            )
        ),
        stdout_path=log_dir / "mjlab-train.stdout.log",
        stderr_path=log_dir / "mjlab-train.stderr.log",
    )
    run_dir = _find_latest_run_dir(bundle, started_at)
    checkpoint = _checkpoint_from_run_dir(run_dir)
    shutil.copy2(checkpoint, out_dir / "agent_checkpoint.pt")
    total_steps = int(iterations * num_envs * steps_per_env)
    partial_path = out_dir / "train_result_partial.json"
    if partial_path.exists():
        payload = json.loads(partial_path.read_text(encoding="utf-8"))
    else:
        payload = {}
    records = payload.get("episode_records") if isinstance(payload.get("episode_records"), list) else []
    collection_records = [record for record in records if str(record.get("record_type") or "train_episode") != "policy_probe"]
    if not collection_records:
        records = []
        for idx in range(iterations):
            records.append(
                make_train_episode_record(
                    episode=idx + 1,
                    return_value=0.0,
                    length=steps_per_env,
                    success=False,
                    step=(idx + 1) * num_envs * steps_per_env,
                    elapsed_seconds=max(0.0, time.time() - started_at),
                    info_metrics={
                        "mjlab_num_envs": float(num_envs),
                        "mjlab_iteration": float(idx + 1),
                        "samples_per_iteration": float(num_envs * steps_per_env),
                        "mjlab_scalar_status": "unavailable",
                    },
                )
            )
            records[-1]["record_type"] = "train_collection_window"
            records[-1]["episodes_in_window"] = num_envs
            records[-1]["env_steps_in_window"] = num_envs * steps_per_env
        collection_records = records
    last_metrics = payload.get("last_metrics") if isinstance(payload.get("last_metrics"), dict) else {}
    last_metrics.update(
        {
            "gradient_updates": float(len(collection_records) or iterations),
            "mjlab_num_envs": float(num_envs),
            "samples_per_iteration": float(num_envs * steps_per_env),
            "policy_probe_count": float(len([record for record in records if str(record.get("record_type")) == "policy_probe"])),
        }
    )
    payload = {
        "episode_records": records,
        "total_steps": total_steps,
        "env_steps": total_steps,
        "episodes_completed": int(iterations * num_envs),
        "completed_episodes": int(iterations * num_envs),
        "episode_batches": len(collection_records) or iterations,
        "gradient_updates": len(collection_records) or iterations,
        "avg_return": float(np.mean([float(record.get("return", 0.0)) for record in collection_records])) if collection_records else 0.0,
        "avg_length": float(np.mean([float(record.get("length", steps_per_env)) for record in collection_records])) if collection_records else float(steps_per_env),
        "last_metrics": last_metrics,
        "stop_reason": "mjlab_train_complete",
        "external_backend_scaffold": False,
        "mjlab": {
            "task_id": task_id,
            "run_dir": str(run_dir),
            "checkpoint_path": str(checkpoint),
            "num_envs": num_envs,
            "steps_per_env_per_iteration": steps_per_env,
            "learning_iterations": iterations,
            "recipe_path": str(recipe_path),
            "resolved_config_path": str(resolved_path),
        },
    }
    if resolved_path.exists():
        resolved_payload = json.loads(resolved_path.read_text(encoding="utf-8"))
        payload["mjlab"]["applied_overrides"] = resolved_payload.get("applied", [])
        payload["mjlab"]["skipped_overrides"] = resolved_payload.get("skipped", [])
    _write_json(out_dir / "train_result.json", payload)
    _write_status(bundle, out_dir, "run", total_steps)


def _write_rollout_script(out_dir: Path) -> Path:
    script_path = out_dir / "mjlab_rollout_bridge.py"
    _copy_bridge_script(MJLAB_ROLLOUT_BRIDGE, script_path)
    return script_path


def _write_train_script(out_dir: Path) -> Path:
    script_path = out_dir / "mjlab_train_bridge.py"
    _copy_bridge_script(MJLAB_TRAIN_BRIDGE, script_path)
    return script_path


def _run_mjlab_rollout(
    bundle: dict[str, Any],
    out_dir: Path,
    checkpoint: Path,
    *,
    mode: str,
    frame_dir: Path | None = None,
) -> dict[str, Any]:
    if not checkpoint.exists():
        raise FileNotFoundError(f"MJLab checkpoint does not exist: {checkpoint}")
    task_id = _task_id(bundle)
    num_envs = min(_parallel_env_count(bundle, for_eval=True), int(os.environ.get("UNITREE_MJLAB_EVAL_NUM_ENVS_MAX", "64")))
    steps = min(int(bundle["benchmark"].get("max_steps") or 200), int(os.environ.get("UNITREE_MJLAB_ROLLOUT_STEPS_MAX", "200")))
    out_json = (out_dir / f"{mode}_rollout.json").resolve()
    script = _write_rollout_script(out_dir).resolve()
    log_dir = out_dir / "command_logs"
    argv: list[str | Path] = [
        _mjlab_python(bundle),
        script,
        "--task-id",
        task_id,
        "--checkpoint",
        checkpoint.resolve(),
        "--out-json",
        out_json,
        "--num-envs",
        str(num_envs),
        "--steps",
        str(steps),
        "--seed",
        str(_seed(bundle, 9000)),
    ]
    motion_file = _motion_file(bundle)
    if motion_file:
        argv.extend(["--motion-file", motion_file])
    if frame_dir is not None:
        argv.extend(
            [
                "--frame-dir",
                frame_dir.resolve(),
                "--frame-count",
                "24",
                "--render-width",
                str(DASHBOARD_FRAME_WIDTH),
                "--render-height",
                str(DASHBOARD_FRAME_HEIGHT),
                "--no-terminations",
            ]
        )
    _run_subprocess(
        [str(arg) for arg in argv],
        cwd=_unitree_root(bundle),
        env=_external_env(bundle),
        timeout=float(os.environ.get("UNITREE_MJLAB_ROLLOUT_TIMEOUT_SECONDS", "300")),
        stdout_path=log_dir / f"mjlab-{mode}.stdout.log",
        stderr_path=log_dir / f"mjlab-{mode}.stderr.log",
    )
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    if mode == "eval":
        info_metrics = {
            "avg_step_reward": float(payload.get("avg_step_reward", 0.0)),
            "done_fraction": float(payload.get("done_fraction", 0.0)),
            "mjlab_num_envs": float(payload.get("num_envs", num_envs)),
        }
        if "avg_mpkpe" in payload:
            info_metrics["mpkpe"] = float(payload["avg_mpkpe"])
            info_metrics["r_mpkpe"] = float(payload.get("avg_r_mpkpe", 0.0))
        records = [
            {
                "episode": 1,
                "seed": _seed(bundle, 9000),
                "return": float(payload.get("return", 0.0)),
                "length": int(payload.get("steps", steps)),
                "case_label": "mjlab-vector-rollout",
                "info_metrics": info_metrics,
            }
        ]
        summary = {
            "episodes": 1,
            "avg_return": float(payload.get("return", 0.0)),
            "avg_length": float(payload.get("steps", steps)),
            "avg_step_reward": float(payload.get("avg_step_reward", 0.0)),
            "done_fraction": float(payload.get("done_fraction", 0.0)),
            "mjlab_num_envs": float(payload.get("num_envs", num_envs)),
            "metric_source": "mjlab_rollout_reward",
            "episode_records": records,
        }
        if "avg_mpkpe" in payload:
            summary["avg_mpkpe"] = float(payload["avg_mpkpe"])
            summary["avg_r_mpkpe"] = float(payload.get("avg_r_mpkpe", 0.0))
        _write_json(out_dir / "eval_result.json", summary)
        _write_status(bundle, out_dir, "eval", 0)
    return payload


def _normalize_dashboard_frame(path: Path) -> Path:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != DASHBOARD_FRAME_SIZE:
            image = image.resize(DASHBOARD_FRAME_SIZE, Image.Resampling.LANCZOS)
            image = image.filter(ImageFilter.UnsharpMask(radius=1.0, percent=90, threshold=3))
        image.save(path, format="JPEG", quality=95, subsampling=0)
    return path


def _run_media(bundle: dict[str, Any], out_dir: Path) -> None:
    _validate(bundle, out_dir)
    if not bundle.get("dry_run"):
        checkpoint = Path(str(bundle.get("checkpoint") or out_dir / "agent_checkpoint.pt"))
        if not checkpoint.exists():
            checkpoint = out_dir / "agent_checkpoint.pt"
        frame_path = out_dir / "current_run_frame.jpg"
        trajectory_dir = out_dir / "trajectories" / "sample_000001"
        trajectory_dir.mkdir(parents=True, exist_ok=True)
        rollout = _run_mjlab_rollout(bundle, out_dir, checkpoint, mode="media", frame_dir=trajectory_dir)
        frames = [_normalize_dashboard_frame(Path(path)) for path in rollout.get("frames", [])]
        if not frames:
            raise RuntimeError("MJLab media rollout completed but did not write any frames")
        shutil.copy2(frames[-1], frame_path)
        manifest = {
            "run_id": bundle["run_id"],
            "tag": bundle["tag"],
            "sample_index": 1,
            "episode": 1,
            "status": "completed",
            "updated_at": time.time(),
            "frame_count": len(frames),
            "frames": [str(path) for path in frames],
            "latest_frame_path": str(frames[-1]),
            "playback_fps": DEFAULT_TRAJECTORY_PLAYBACK_FPS,
            "frame_stride": 1,
            "sample_rate": 1.0,
            "width": DASHBOARD_FRAME_WIDTH,
            "height": DASHBOARD_FRAME_HEIGHT,
            "source": "mjlab",
            "task_id": _task_id(bundle),
        }
        manifest_path = trajectory_dir / "manifest.json"
        _write_json(manifest_path, manifest)
        media = {
            "media_available": True,
            "live_frame_path": str(frame_path),
            "trajectory_manifest_path": str(manifest_path),
            "trajectory_latest_frame_path": str(frames[-1]),
            "visual": {
                "mode": "sampled_trajectory",
                "live_frame_path": str(frame_path),
                "trajectory_manifest_path": str(manifest_path),
                "trajectory_latest_frame_path": str(frames[-1]),
                "sampled_status": "completed",
                "latest_sample_index": 1,
                "source": "mjlab",
            },
        }
        _write_json(out_dir / "media_result.json", media)
        return
    media = {
        "media_available": False,
        "visual": {
            "mode": "off",
            "sampled_status": "unavailable",
            "disabled_reason": "unitree_dry_run_has_no_real_renderer",
        },
    }
    _write_json(out_dir / "media_result.json", media)
    if bundle.get("session_dir"):
        live_dir = Path(bundle["session_dir"]) / "live"
        live_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            live_dir / "current_run_metrics.json",
            {
                "run": {
                    "run_id": bundle["run_id"],
                    "tag": bundle["tag"],
                    "status": "finished",
                    "candidate": bundle.get("candidate", {}),
                },
                "current": {"status": "finished", "step": 0, "episodes_complete": bundle["benchmark"]["eval_episodes"]},
                "visual": {
                    "mode": "off",
                    "sampled_status": "unavailable",
                    "disabled_reason": "unitree_dry_run_has_no_real_renderer",
                },
            },
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval", "media"], required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args(argv)
    bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
    if args.checkpoint is not None:
        bundle["checkpoint"] = str(args.checkpoint)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    time.sleep(0.01)
    if args.mode == "train":
        _run_train(bundle, args.out_dir)
    elif args.mode == "eval":
        _run_eval(bundle, args.out_dir)
    else:
        _run_media(bundle, args.out_dir)


if __name__ == "__main__":
    main()
