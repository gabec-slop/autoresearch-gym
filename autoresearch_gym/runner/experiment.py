from __future__ import annotations

import json
import os
import importlib.util
import platform
import re
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import gymnasium as gym
import numpy as np
from PIL import Image

from autoresearch_gym.runner.curves import (
    aggregate_info_metrics,
    collection_episode_records,
    is_policy_probe_record,
    make_policy_probe_record,
    scalar_info_metrics,
    validate_train_curve_contract,
)

try:
    import pybullet

    SIM_RECOVERABLE_ERRORS = (pybullet.error, RuntimeError, TypeError, KeyError, ValueError)
except ModuleNotFoundError:
    pybullet = None  # type: ignore[assignment]
    SIM_RECOVERABLE_ERRORS = (RuntimeError, TypeError, KeyError, ValueError)

ROOT_DIR = Path(__file__).resolve().parents[2]

import sys

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def require_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Torch is required to run training candidates. Install a task extra, "
            "for example `pip install -e '.[mujoco]'` or `pip install -e '.[panda]'`."
        ) from exc
    return torch


@dataclass
class TrainProbeSpec:
    enabled: bool = True
    interval_seconds: float = 5.0
    episodes: int = 3
    seed_start: int = 900_000


@dataclass
class BenchmarkSpec:
    name: str
    env_id: str
    env_kwargs: dict[str, Any]
    train_episodes: int
    train_seconds: float | None
    eval_episodes: int
    max_steps: int
    reward_type: str | None
    render_mode: str | None
    primary_metric: str
    primary_metric_mode: str
    train_seed: int
    eval_seed_start: int
    device: str
    eval_case_bank: Path | None
    train_probe: TrainProbeSpec = field(default_factory=TrainProbeSpec)


def load_benchmark(path: Path) -> BenchmarkSpec:
    payload = json.loads(path.read_text())
    env_kwargs = dict(payload.get("env_kwargs", {}))
    if "render_mode" in payload:
        env_kwargs.setdefault("render_mode", payload["render_mode"])
    if "reward_type" in payload:
        env_kwargs.setdefault("reward_type", payload["reward_type"])
    max_steps = int(payload.get("max_steps", env_kwargs.get("max_steps", 0)))
    probe_payload = payload.get("train_probe") or {}
    train_probe = TrainProbeSpec(
        enabled=bool(probe_payload.get("enabled", True)),
        interval_seconds=float(probe_payload.get("interval_seconds", 5.0)),
        episodes=int(probe_payload.get("episodes", 3)),
        seed_start=int(probe_payload.get("seed_start", 900_000)),
    )
    return BenchmarkSpec(
        name=payload["name"],
        env_id=str(payload["env_id"]),
        env_kwargs=env_kwargs,
        train_episodes=int(payload["train_episodes"]),
        train_seconds=(
            float(payload["train_seconds"])
            if payload.get("train_seconds") is not None
            else float(payload["train_wall_clock_seconds"])
            if payload.get("train_wall_clock_seconds") is not None
            else None
        ),
        eval_episodes=int(payload["eval_episodes"]),
        max_steps=max_steps,
        reward_type=payload.get("reward_type", env_kwargs.get("reward_type")),
        render_mode=payload.get("render_mode", env_kwargs.get("render_mode")),
        primary_metric=str(payload["primary_metric"]),
        primary_metric_mode=str(payload.get("primary_metric_mode", "maximize")),
        train_seed=int(payload["train_seed"]),
        eval_seed_start=int(payload["eval_seed_start"]),
        device=str(payload["device"]),
        eval_case_bank=(path.parent / payload["eval_case_bank"]).resolve() if payload.get("eval_case_bank") else None,
        train_probe=train_probe,
    )


PYBULLET_RENDER_REQUIRED_ENV_IDS = {
    "AutoresearchPandaPickAndPlaceDense-v0",
    "PandaBatToGoal-v0",
}


def apply_headless_env_override(benchmark: BenchmarkSpec) -> dict[str, Any]:
    if benchmark.env_id in PYBULLET_RENDER_REQUIRED_ENV_IDS:
        return {
            "requested": True,
            "effective": False,
            "reason": "env_requires_render_mode",
            "message": (
                "This Panda/PyBullet environment requires render_mode='rgb_array' "
                "or 'human'; keeping the benchmark render_mode."
            ),
        }
    benchmark.env_kwargs["render_mode"] = None
    benchmark.render_mode = None
    return {
        "requested": True,
        "effective": True,
        "reason": None,
        "message": "Environment construction was overridden to render_mode=None.",
    }


def select_device(preference: str) -> Any:
    torch = require_torch()
    pref = preference.lower()
    if pref in {"auto", "mps"} and torch.backends.mps.is_available():
        return torch.device("mps")
    if pref in {"auto", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(f"Unsupported JSON value: {type(value)!r}")


def normalize_run_tag(tag: str) -> str:
    """Collapse accidental duplicate pass prefixes such as pass02-pass02-foo."""
    normalized = tag.strip()
    duplicate_pass_prefix = re.match(r"^(pass\d+)-\1-(.+)$", normalized)
    if duplicate_pass_prefix:
        return f"{duplicate_pass_prefix.group(1)}-{duplicate_pass_prefix.group(2)}"
    return normalized


def load_trainable_module(candidate_path: Path) -> ModuleType:
    resolved = candidate_path.resolve()
    module_name = f"autoresearch_candidate_{time.time_ns()}"
    spec = importlib.util.spec_from_file_location(module_name, resolved)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load candidate module from {resolved}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    for attr in ["get_candidate", "RewardRecipeWrapper", "train_agent", "save_agent_checkpoint"]:
        if not hasattr(module, attr):
            raise AttributeError(f"Candidate module {resolved} is missing required attribute {attr}")
    return module


def env_kwargs_for_candidate(benchmark: BenchmarkSpec, control_type: str | None = None) -> dict[str, Any]:
    env_kwargs = dict(benchmark.env_kwargs)
    if control_type is not None:
        env_kwargs.setdefault("control_type", control_type)
    return env_kwargs


def candidate_metadata(candidate: Any) -> dict[str, Any]:
    if isinstance(candidate, str):
        return {"description": candidate}
    if isinstance(candidate, dict):
        return candidate
    if is_dataclass(candidate):
        return asdict(candidate)
    if hasattr(candidate, "__dict__"):
        return dict(vars(candidate))
    return {"repr": repr(candidate)}


def make_env(
    benchmark: BenchmarkSpec,
    control_type: str | None,
    reward_recipe: str | None,
    reward_wrapper_cls: Any,
) -> gym.Env[np.ndarray, np.ndarray]:
    env = gym.make(benchmark.env_id, **env_kwargs_for_candidate(benchmark, control_type))
    try:
        return reward_wrapper_cls(env, reward_recipe)
    except TypeError:
        if reward_recipe is None:
            return reward_wrapper_cls(env)
        raise


def make_eval_env(benchmark: BenchmarkSpec, control_type: str | None) -> gym.Env[np.ndarray, np.ndarray]:
    return gym.make(benchmark.env_id, **env_kwargs_for_candidate(benchmark, control_type))


def load_eval_cases(benchmark: BenchmarkSpec) -> list[dict[str, Any]] | None:
    if benchmark.eval_case_bank is None:
        return None
    payload = json.loads(benchmark.eval_case_bank.read_text(encoding="utf-8"))
    cases = list(payload.get("cases", []))
    return cases if cases else None


def evaluate_agent(agent: Any, benchmark: BenchmarkSpec, candidate: Any) -> dict[str, Any]:
    if hasattr(agent, "evaluate"):
        return agent.evaluate(benchmark=benchmark, candidate=candidate)

    env = make_eval_env(benchmark, getattr(candidate, "control_type", None))
    episode_records: list[dict[str, Any]] = []
    eval_cases = load_eval_cases(benchmark)

    for idx in range(benchmark.eval_episodes):
        seed = benchmark.eval_seed_start + idx
        reset_options = None
        case_label = None
        if eval_cases is not None and idx < len(eval_cases):
            reset_options = {"fixed_case": eval_cases[idx]}
            case_label = str(eval_cases[idx].get("name", f"case-{idx + 1:02d}"))
        try:
            obs, info = env.reset(seed=seed, options=reset_options)
        except SIM_RECOVERABLE_ERRORS:
            obs, info = env.reset()

        terminated = False
        truncated = False
        episode_return = 0.0
        episode_length = 0

        while not (terminated or truncated):
            action = agent.act(obs, deterministic=True)
            try:
                obs, reward, terminated, truncated, info = env.step(action)
            except SIM_RECOVERABLE_ERRORS:
                terminated = True
                truncated = False
                reward = -3.0
                info = {"is_success": False}
            episode_return += float(reward)
            episode_length += 1

        episode_records.append(
            {
                "episode": idx + 1,
                "seed": seed,
                "return": float(episode_return),
                "length": int(episode_length),
                "success": bool(info.get("is_success", False)),
                "info_metrics": scalar_info_metrics(info),
                "case_label": case_label,
            }
        )

    env.close()
    summary = {
        "episodes": benchmark.eval_episodes,
        "success_rate": float(np.mean([1.0 if e["success"] else 0.0 for e in episode_records])) if episode_records else 0.0,
        "avg_return": float(np.mean([e["return"] for e in episode_records])) if episode_records else 0.0,
        "avg_length": float(np.mean([e["length"] for e in episode_records])) if episode_records else 0.0,
        "episode_records": episode_records,
    }
    summary.update(aggregate_info_metrics(episode_records))
    return summary


def generic_policy_probe(
    agent: Any,
    benchmark: BenchmarkSpec,
    candidate: Any,
    *,
    episodes: int,
    seed_start: int,
) -> dict[str, Any]:
    if not hasattr(agent, "act"):
        raise TypeError("agent does not expose act(obs, deterministic=True)")

    env = make_eval_env(benchmark, getattr(candidate, "control_type", None))
    episode_records: list[dict[str, Any]] = []
    try:
        for idx in range(episodes):
            seed = int(seed_start) + idx
            try:
                obs, info = env.reset(seed=seed)
            except SIM_RECOVERABLE_ERRORS:
                obs, info = env.reset()
            terminated = False
            truncated = False
            episode_return = 0.0
            episode_length = 0

            while not (terminated or truncated) and episode_length < benchmark.max_steps:
                action = agent.act(obs, deterministic=True)
                try:
                    obs, reward, terminated, truncated, info = env.step(action)
                except SIM_RECOVERABLE_ERRORS:
                    terminated = True
                    truncated = False
                    reward = -3.0
                    info = {"is_success": False}
                episode_return += float(reward)
                episode_length += 1

            episode_records.append(
                {
                    "episode": idx + 1,
                    "seed": seed,
                    "return": float(episode_return),
                    "length": int(episode_length),
                    "success": bool(info.get("is_success", False)),
                    "info_metrics": scalar_info_metrics(info),
                }
            )
    finally:
        env.close()

    return {
        "episodes": int(episodes),
        "seed_start": int(seed_start),
        "avg_return": float(np.mean([e["return"] for e in episode_records])) if episode_records else 0.0,
        "avg_length": float(np.mean([e["length"] for e in episode_records])) if episode_records else 0.0,
        "success_rate": (
            float(np.mean([1.0 if e["success"] else 0.0 for e in episode_records]))
            if episode_records
            else 0.0
        ),
        "episode_records": episode_records,
    }


def make_policy_probe_callback(
    trainable_module: ModuleType,
    benchmark: BenchmarkSpec,
    candidate: Any,
    device: Any,
) -> Any:
    probe = benchmark.train_probe
    status = "disabled" if not probe.enabled else "waiting"
    last_probe_elapsed: float | None = None
    probe_count = 0
    probe_records: list[dict[str, Any]] = []
    started_at = time.time()

    def maybe_probe(**kwargs: Any) -> dict[str, Any]:
        nonlocal status, last_probe_elapsed, probe_count
        if not probe.enabled:
            return {"train_probe_status": status}
        if kwargs.get("status") != "running":
            return {"train_probe_status": status}
        agent = kwargs.get("agent")
        if agent is None:
            status = "unsupported"
            return {"train_probe_status": status}
        episode_records = kwargs.get("episode_records")
        if not isinstance(episode_records, list):
            return {"train_probe_status": status}
        display_records = [*episode_records, *probe_records]
        elapsed = float(kwargs.get("elapsed_seconds") or (time.time() - started_at))
        if last_probe_elapsed is not None and elapsed - last_probe_elapsed < probe.interval_seconds:
            return {"episode_records": display_records, "train_probe_status": status}
        if last_probe_elapsed is None and elapsed < probe.interval_seconds:
            return {"episode_records": display_records, "train_probe_status": status}

        seed_start = int(probe.seed_start + probe_count * probe.episodes)
        try:
            if hasattr(trainable_module, "probe_policy"):
                probe_summary = trainable_module.probe_policy(
                    agent,
                    benchmark,
                    candidate,
                    device,
                    episodes=int(probe.episodes),
                    seed_start=seed_start,
                )
            else:
                probe_summary = generic_policy_probe(
                    agent,
                    benchmark,
                    candidate,
                    episodes=int(probe.episodes),
                    seed_start=seed_start,
                )
        except Exception as exc:  # pragma: no cover - probe failures must not fail training.
            status = f"failed: {exc}"
            last_probe_elapsed = elapsed
            return {"train_probe_status": status}

        record = make_policy_probe_record(
            episode=len(display_records) + 1,
            return_value=float(probe_summary.get("avg_return", 0.0)),
            length=float(probe_summary.get("avg_length", 0.0)),
            step=int(kwargs.get("total_steps") or 0),
            elapsed_seconds=elapsed,
            probe_episodes=int(probe_summary.get("episodes") or probe.episodes),
            probe_seed_start=seed_start,
            success_rate=float(probe_summary.get("success_rate", 0.0)),
        )
        probe_records.append(record)
        display_records.append(record)
        probe_count += 1
        last_probe_elapsed = elapsed
        status = "ok"
        return {
            "episode_records": display_records,
            "train_probe_status": status,
            "train_probe_return": record["return"],
            "train_probe_length": record["length"],
        }

    maybe_probe.probe_records = probe_records  # type: ignore[attr-defined]
    return maybe_probe


def append_result(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=json_default) + "\n")


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".{os.getpid()}.{time.time_ns()}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")
    last_error: PermissionError | None = None
    for _ in range(100):
        try:
            tmp_path.replace(path)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.05)
    tmp_path.unlink(missing_ok=True)
    raise last_error if last_error is not None else RuntimeError(f"failed to write {path}")


def write_frame_atomic(path: Path, frame: np.ndarray, quality: int = 75) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.ascontiguousarray(frame[:, :, :3], dtype=np.uint8)
    tmp_path = path.with_suffix(path.suffix + f".{os.getpid()}.{time.time_ns()}.tmp")
    Image.fromarray(image).save(tmp_path, format="JPEG", quality=quality, optimize=False)
    last_error: PermissionError | None = None
    for _ in range(100):
        try:
            tmp_path.replace(path)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.05)
    tmp_path.unlink(missing_ok=True)
    raise last_error if last_error is not None else RuntimeError(f"failed to write {path}")


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT_DIR))
    except ValueError:
        return str(path.resolve())


VISUAL_MODES = {"off", "live_frame", "sampled_trajectory"}
DEFAULT_VISUAL_CONTROL: dict[str, Any] = {
    "visual_mode": "sampled_trajectory",
    "live_frame_interval_seconds": 5.0,
    "trajectory_sample_rate": 0.05,
    "trajectory_frame_stride": 2,
    "trajectory_playback_fps": 20.0,
    "jpeg_quality": 70,
    "control_poll_seconds": 0.0,
}


def _bounded_float(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(numeric):
        return default
    return float(min(max(numeric, minimum), maximum))


def _bounded_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return default
    return int(min(max(numeric, minimum), maximum))


def normalize_visual_control(payload: dict[str, Any] | None) -> dict[str, Any]:
    control = dict(DEFAULT_VISUAL_CONTROL)
    if not isinstance(payload, dict):
        return control

    mode = str(payload.get("visual_mode", control["visual_mode"]))
    if mode in VISUAL_MODES:
        control["visual_mode"] = mode
    control["live_frame_interval_seconds"] = _bounded_float(
        payload.get("live_frame_interval_seconds"),
        float(control["live_frame_interval_seconds"]),
        0.25,
        120.0,
    )
    control["trajectory_sample_rate"] = _bounded_float(
        payload.get("trajectory_sample_rate"),
        float(control["trajectory_sample_rate"]),
        0.0,
        1.0,
    )
    control["trajectory_frame_stride"] = _bounded_int(
        payload.get("trajectory_frame_stride"),
        int(control["trajectory_frame_stride"]),
        1,
        10_000,
    )
    control["trajectory_playback_fps"] = _bounded_float(
        payload.get("trajectory_playback_fps"),
        float(control["trajectory_playback_fps"]),
        1.0,
        60.0,
    )
    control["jpeg_quality"] = _bounded_int(payload.get("jpeg_quality"), int(control["jpeg_quality"]), 20, 95)
    control["control_poll_seconds"] = _bounded_float(
        payload.get("control_poll_seconds"),
        float(control["control_poll_seconds"]),
        0.0,
        10.0,
    )
    return control


def should_sample_visual_episode(episode: int, control: dict[str, Any]) -> bool:
    sample_rate = float(control.get("trajectory_sample_rate", 0.0) or 0.0)
    if sample_rate <= 0.0:
        return False
    if sample_rate >= 1.0:
        return True
    interval = max(1, int(round(1.0 / sample_rate)))
    return episode == 1 or episode % interval == 0


def make_live_writer(
    session_dir: Path | None,
    run_id: str,
    tag: str,
    benchmark: BenchmarkSpec,
    candidate: Any,
    *,
    headless_env: bool = False,
):
    if session_dir is None:
        return None

    live_dir = session_dir / "live"
    metrics_path = live_dir / "current_run_metrics.json"
    frame_path = live_dir / "current_run_frame.jpg"
    control_path = live_dir / "control.json"
    trajectories_dir = live_dir / "trajectories" / run_id
    started_at = time.time()
    last_frame_at = 0.0
    last_control_read_at = 0.0
    visual_control = dict(DEFAULT_VISUAL_CONTROL)
    visual_disabled_reason = "headless_env" if headless_env else None
    if visual_disabled_reason is not None:
        visual_control["visual_mode"] = "off"
    visual_control_error: str | None = None
    sampled_episode: int | None = None
    sampled_manifest_path: Path | None = None
    sampled_frames: list[str] = []
    sampled_status = "idle"
    sampled_last_step = -1
    sampled_trajectory_count = 0
    sampled_trajectory_index: int | None = None
    latest_trajectory_index: int | None = None
    latest_trajectory_manifest_path: Path | None = None
    latest_trajectory_frame_path: Path | None = None
    visual_episode = 0
    visual_sampling_eligible = False
    mujoco_renderers: dict[int, Any] = {}

    if visual_disabled_reason is not None or not control_path.exists():
        write_json_atomic(control_path, visual_control)

    def render_live_frame(env: gym.Env[np.ndarray, np.ndarray]) -> np.ndarray | None:
        render_env = getattr(env, "unwrapped", env)
        model = getattr(render_env, "model", None)
        data = getattr(render_env, "data", None)
        if model is not None and data is not None:
            try:
                import mujoco

                renderer_key = id(render_env)
                renderer = mujoco_renderers.get(renderer_key)
                if renderer is None:
                    renderer = mujoco.Renderer(model, height=360, width=480)
                    mujoco_renderers[renderer_key] = renderer
                mujoco.mj_forward(model, data)
                renderer.update_scene(data)
                return renderer.render()
            except Exception:
                return None

        try:
            return render_env.render(width=720, height=480)
        except TypeError:
            try:
                return render_env.render()
            except Exception:
                return None
        except Exception:
            return None

    def read_visual_control(force: bool = False) -> dict[str, Any]:
        nonlocal last_control_read_at, visual_control, visual_control_error
        if visual_disabled_reason is not None:
            return visual_control
        now = time.perf_counter()
        poll_seconds = float(visual_control.get("control_poll_seconds", DEFAULT_VISUAL_CONTROL["control_poll_seconds"]))
        if not force and now - last_control_read_at < poll_seconds:
            return visual_control
        last_control_read_at = now
        try:
            payload = json.loads(control_path.read_text(encoding="utf-8")) if control_path.exists() else {}
            visual_control = normalize_visual_control(payload)
            visual_control_error = None
        except (OSError, json.JSONDecodeError) as exc:
            visual_control_error = str(exc)
        return visual_control

    def write_sampled_manifest(status: str, episode: int | None, reason: str | None = None) -> None:
        if sampled_manifest_path is None or episode is None:
            return
        payload = {
            "run_id": run_id,
            "tag": tag,
            "sample_index": sampled_trajectory_index,
            "episode": int(episode),
            "status": status,
            "reason": reason,
            "updated_at": time.time(),
            "frame_count": len(sampled_frames),
            "frames": sampled_frames,
            "latest_frame_path": sampled_frames[-1] if sampled_frames else None,
            "playback_fps": float(visual_control.get("trajectory_playback_fps", 20.0)),
            "frame_stride": int(visual_control.get("trajectory_frame_stride", 2)),
            "sample_rate": float(visual_control.get("trajectory_sample_rate", 0.05)),
        }
        write_json_atomic(sampled_manifest_path, payload)

    def stop_sampled_episode(status: str, reason: str | None = None) -> None:
        nonlocal sampled_episode, sampled_manifest_path, sampled_frames, sampled_status, sampled_last_step, sampled_trajectory_index
        if sampled_episode is not None:
            write_sampled_manifest(status, sampled_episode, reason)
        sampled_episode = None
        sampled_manifest_path = None
        sampled_frames = []
        sampled_status = "idle" if status == "completed" else status
        sampled_last_step = -1
        sampled_trajectory_index = None

    def start_sampled_episode(episode: int) -> None:
        nonlocal sampled_episode, sampled_manifest_path, sampled_frames, sampled_status, sampled_last_step
        nonlocal sampled_trajectory_count, sampled_trajectory_index, latest_trajectory_index
        sampled_trajectory_count += 1
        sampled_trajectory_index = sampled_trajectory_count
        latest_trajectory_index = sampled_trajectory_index
        sampled_episode = int(episode)
        sampled_status = "recording"
        sampled_last_step = -1
        sampled_frames = []
        sampled_dir = trajectories_dir / f"episode_{episode:06d}"
        sampled_manifest_path = sampled_dir / "manifest.json"
        write_sampled_manifest("recording", sampled_episode)

    def sample_trajectory_frame(env: gym.Env[np.ndarray, np.ndarray], episode_length: int, reason: str) -> None:
        nonlocal sampled_last_step, latest_trajectory_manifest_path, latest_trajectory_frame_path
        if sampled_episode is None or sampled_manifest_path is None:
            return
        if episode_length == sampled_last_step and reason != "episode_end":
            return
        frame = render_live_frame(env)
        if frame is None:
            write_sampled_manifest("render_unavailable", sampled_episode, "renderer returned no frame")
            return
        frame_path = sampled_manifest_path.parent / f"frame_{len(sampled_frames):04d}.jpg"
        write_frame_atomic(frame_path, frame, quality=int(visual_control.get("jpeg_quality", 70)))
        sampled_frames.append(repo_relative(frame_path))
        sampled_last_step = int(episode_length)
        latest_trajectory_manifest_path = sampled_manifest_path
        latest_trajectory_frame_path = frame_path
        write_sampled_manifest("recording", sampled_episode)

    def visual_reset(env: gym.Env[np.ndarray, np.ndarray]) -> None:
        nonlocal visual_episode, visual_sampling_eligible, sampled_status
        visual_episode += 1
        control = read_visual_control(force=True)
        visual_sampling_eligible = (
            str(control.get("visual_mode", DEFAULT_VISUAL_CONTROL["visual_mode"])) == "sampled_trajectory"
            and should_sample_visual_episode(visual_episode, control)
        )
        if sampled_episode is not None:
            stop_sampled_episode("interrupted", "env_reset")
        if visual_sampling_eligible:
            start_sampled_episode(visual_episode)
            sample_trajectory_frame(env, 0, "episode_start")
        elif str(control.get("visual_mode")) == "sampled_trajectory":
            sampled_status = "skipped"

    def visual_step(
        env: gym.Env[np.ndarray, np.ndarray],
        episode_length: int,
        terminated: bool,
        truncated: bool,
    ) -> None:
        nonlocal sampled_status
        control = read_visual_control(force=True)
        visual_mode = str(control.get("visual_mode", DEFAULT_VISUAL_CONTROL["visual_mode"]))
        if sampled_episode is not None and visual_mode != "sampled_trajectory":
            stop_sampled_episode("interrupted", "visual_mode_changed")
            return
        if visual_mode != "sampled_trajectory":
            return
        if sampled_episode is None:
            sampled_status = "waiting_for_episode_boundary" if not visual_sampling_eligible else sampled_status
            return
        stride = int(control.get("trajectory_frame_stride", 2))
        episode_complete = bool(terminated or truncated)
        if episode_complete or episode_length % stride == 0:
            sample_trajectory_frame(env, episode_length, "episode_end" if episode_complete else "stride")
        if episode_complete:
            stop_sampled_episode("completed")

    class LiveVisualWrapper(gym.Wrapper[np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
        def __init__(self, env: gym.Env[np.ndarray, np.ndarray]) -> None:
            super().__init__(env)
            self._live_episode_length = 0

        def __getattr__(self, name: str) -> Any:
            if name.startswith("_"):
                raise AttributeError(name)
            return getattr(self.env, name)

        def reset(self, *args: Any, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
            result = self.env.reset(*args, **kwargs)
            self._live_episode_length = 0
            visual_reset(self)
            return result

        def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
            obs, reward, terminated, truncated, info = self.env.step(action)
            self._live_episode_length += 1
            visual_step(self, self._live_episode_length, bool(terminated), bool(truncated))
            return obs, reward, terminated, truncated, info

    def wrap_env(env: gym.Env[np.ndarray, np.ndarray]) -> gym.Env[np.ndarray, np.ndarray]:
        return LiveVisualWrapper(env)

    def write_live(
        *,
        status: str,
        episode_records: list[dict[str, Any]],
        total_steps: int,
        last_metrics: dict[str, float] | None,
        env: gym.Env[np.ndarray, np.ndarray] | None = None,
        current_episode: int | None = None,
        episode_return: float = 0.0,
        episode_length: int = 0,
    ) -> dict[str, Any]:
        nonlocal last_frame_at, sampled_status
        control = read_visual_control()
        visual_mode = str(control.get("visual_mode", DEFAULT_VISUAL_CONTROL["visual_mode"]))
        episode_number = int(current_episode or len(episode_records) + 1)

        if sampled_episode is not None and visual_mode != "sampled_trajectory":
            stop_sampled_episode("interrupted", "visual_mode_changed")

        collection_records = collection_episode_records(episode_records)
        probe_records = [record for record in episode_records if is_policy_probe_record(record)]
        avg_return = float(np.mean([entry["return"] for entry in collection_records])) if collection_records else 0.0
        success_rate = (
            float(np.mean([1.0 if entry["success"] else 0.0 for entry in collection_records]))
            if collection_records
            else 0.0
        )
        info_aggregates = aggregate_info_metrics(collection_records)
        if probe_records:
            latest_probe = probe_records[-1]
            info_aggregates["policy_probe_return"] = float(latest_probe.get("return", 0.0))
            info_aggregates["policy_probe_length"] = float(latest_probe.get("length", 0.0))
            if "elapsed_seconds" in latest_probe:
                info_aggregates["policy_probe_elapsed_seconds"] = float(latest_probe["elapsed_seconds"])

        def write_metrics() -> None:
            live_frame_path = repo_relative(frame_path) if visual_mode == "live_frame" and frame_path.exists() else None
            trajectory_manifest = (
                repo_relative(latest_trajectory_manifest_path)
                if latest_trajectory_manifest_path is not None and latest_trajectory_manifest_path.exists()
                else None
            )
            trajectory_latest_frame = (
                repo_relative(latest_trajectory_frame_path)
                if latest_trajectory_frame_path is not None and latest_trajectory_frame_path.exists()
                else None
            )
            visual_payload = {
                "mode": visual_mode,
                "control_path": repo_relative(control_path),
                "control_error": visual_control_error,
                "live_frame_path": live_frame_path,
                "trajectory_manifest_path": trajectory_manifest,
                "trajectory_latest_frame_path": trajectory_latest_frame,
                "sampled_status": sampled_status,
                "active_sampled_episode": sampled_episode,
                "active_sample_index": sampled_trajectory_index,
                "latest_sample_index": latest_trajectory_index,
                "disabled_reason": visual_disabled_reason,
            }
            write_json_atomic(
                metrics_path,
                {
                    "run": {
                        "run_id": run_id,
                        "tag": tag,
                        "status": status,
                        "started_at": started_at,
                        "updated_at": time.time(),
                        "train_episodes": benchmark.train_episodes,
                        "train_seconds": benchmark.train_seconds,
                        "eval_episodes": benchmark.eval_episodes,
                        "max_steps": benchmark.max_steps,
                        "render_mode": benchmark.render_mode,
                        "candidate": candidate_metadata(candidate),
                        "frame_path": live_frame_path,
                        "trajectory_manifest_path": trajectory_manifest,
                        "trajectory_latest_frame_path": trajectory_latest_frame,
                        "visual_control": control,
                        "visual": visual_payload,
                    },
                    "current": {
                        "status": status,
                        "step": int(total_steps),
                        "episode": episode_number,
                        "episode_return": float(episode_return),
                        "episode_length": int(episode_length),
                        "avg_return": avg_return,
                        "success_rate": success_rate,
                        "info_metrics": info_aggregates,
                        "episodes_complete": len(collection_records),
                    },
                    "episodes": episode_records[-400:],
                    "latest_losses": last_metrics,
                    "visual_control": control,
                    "visual": visual_payload,
                },
            )

        # Write metrics before rendering so a slow or broken renderer cannot make
        # the dashboard appear dead.
        write_metrics()

        if env is not None and visual_mode == "live_frame":
            now = time.perf_counter()
            frame_interval = float(control.get("live_frame_interval_seconds", 5.0))
            if status in {"starting", "finished"} or now - last_frame_at >= frame_interval:
                frame = render_live_frame(env)
                if frame is not None:
                    write_frame_atomic(frame_path, frame, quality=int(control.get("jpeg_quality", 70)))
                    last_frame_at = now
                    write_metrics()
        return {
            "visual_mode": visual_mode,
            "sampled_trajectory_active": bool(visual_mode == "sampled_trajectory" and sampled_episode is not None),
            "sampled_trajectory_stride": int(control.get("trajectory_frame_stride", 2)),
            "sampled_trajectory_sample_index": sampled_trajectory_index,
        }

    write_live.wrap_env = wrap_env  # type: ignore[attr-defined]
    return write_live


def compact_status_line(
    *,
    elapsed_seconds: float,
    train_seconds: float | None,
    train_episodes: int | None,
    status: str,
    episode_records: list[dict[str, Any]],
    total_steps: int,
    last_metrics: dict[str, float] | None,
    current_episode: int | None = None,
    episode_return: float = 0.0,
    episode_length: int = 0,
) -> str:
    total_seconds = max(int(round(elapsed_seconds)), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    elapsed_label = f"{hours:02d}:{minutes:02d}:{secs:02d}"
    status_label = {
        "running": "run",
        "evaluating": "eval",
        "finished": "done",
    }.get(status, status)
    collection_records = collection_episode_records(episode_records)
    avg_return = float(np.mean([entry["return"] for entry in collection_records])) if collection_records else 0.0
    success_rate = (
        float(np.mean([1.0 if entry["success"] else 0.0 for entry in collection_records]))
        if collection_records
        else 0.0
    )
    update_label = "?" if last_metrics is None else "Y"
    if train_seconds is not None and train_seconds > 0:
        progress_fraction = min(max(elapsed_seconds / train_seconds, 0.0), 1.0)
        budget_elapsed_seconds = min(total_seconds, max(int(round(train_seconds)), 1))
        budget_label = f"time={budget_elapsed_seconds}/{int(round(train_seconds))}s"
    else:
        episode_budget = max(int(train_episodes or 0), 1)
        progress_fraction = min(max(len(collection_records) / episode_budget, 0.0), 1.0)
        budget_label = f"eps={len(collection_records)}/{episode_budget}"
    return (
        f"t={elapsed_label} pct={100.0 * progress_fraction:.1f} {budget_label} st={status_label} step={int(total_steps)} "
        f"ep={int(current_episode or len(collection_records) + 1)} done={len(collection_records)} "
        f"avg={avg_return:.3f} succ={success_rate:.3f} cur={float(episode_return):.3f} "
        f"len={int(episode_length)} upd={update_label}"
    )


def make_compact_status_writer(
    interval_seconds: float,
    *,
    train_seconds: float | None = None,
    train_episodes: int | None = None,
    emit_stderr: bool = True,
    compact_status_file: Path | None = None,
):
    started_at = time.perf_counter()
    last_emit_at = 0.0
    interval = max(float(interval_seconds), 0.1)
    if compact_status_file is not None:
        compact_status_file.parent.mkdir(parents=True, exist_ok=True)
        compact_status_file.write_text("", encoding="utf-8")

    def write_status(
        *,
        status: str,
        episode_records: list[dict[str, Any]],
        total_steps: int,
        last_metrics: dict[str, float] | None,
        env: gym.Env[np.ndarray, np.ndarray] | None = None,
        current_episode: int | None = None,
        episode_return: float = 0.0,
        episode_length: int = 0,
    ) -> None:
        del env
        nonlocal last_emit_at
        now = time.perf_counter()
        force = status in {"evaluating", "finished"} or last_emit_at == 0.0
        if not force and now - last_emit_at < interval:
            return
        last_emit_at = now
        line = compact_status_line(
            elapsed_seconds=now - started_at,
            train_seconds=train_seconds,
            train_episodes=train_episodes,
            status=status,
            episode_records=episode_records,
            total_steps=total_steps,
            last_metrics=last_metrics,
            current_episode=current_episode,
            episode_return=episode_return,
            episode_length=episode_length,
        )
        if emit_stderr:
            print(line, file=sys.stderr, flush=True)
        if compact_status_file is not None:
            with compact_status_file.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    return write_status


def combine_live_callbacks(*callbacks: Any):
    active_callbacks = [callback for callback in callbacks if callback is not None]
    if not active_callbacks:
        return None

    def combined_callback(**kwargs: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for callback in active_callbacks:
            result = callback(**kwargs)
            if isinstance(result, dict):
                payload.update(result)
                kwargs.update(result)
        return payload

    return combined_callback


def public_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in summary.items() if key != "episode_records"}


def normalize_train_summary_curve(train_summary: dict[str, Any]) -> None:
    records = train_summary.get("episode_records")
    if not isinstance(records, list):
        return
    train_summary.setdefault("last_metrics", None)
    train_summary.setdefault("total_steps", 0)
    collection_records = collection_episode_records(records)
    train_summary["episodes_completed"] = len(collection_records)
    train_summary["avg_return"] = (
        float(np.mean([record["return"] for record in collection_records])) if collection_records else 0.0
    )
    train_summary["success_rate"] = (
        float(np.mean([1.0 if record.get("success") else 0.0 for record in collection_records]))
        if collection_records
        else 0.0
    )
    train_summary["avg_length"] = (
        float(np.mean([record["length"] for record in collection_records])) if collection_records else 0.0
    )
    probe_records = [record for record in records if is_policy_probe_record(record)]
    train_summary["policy_probe_records"] = len(probe_records)
    if probe_records:
        latest_probe = probe_records[-1]
        train_summary["latest_policy_probe_return"] = latest_probe.get("return")
        train_summary["latest_policy_probe_length"] = latest_probe.get("length")


def merge_policy_probe_records(train_summary: dict[str, Any], probe_records: list[dict[str, Any]]) -> None:
    if not probe_records:
        return
    records = train_summary.get("episode_records")
    if not isinstance(records, list):
        return
    merged = [*records, *probe_records]
    train_summary["episode_records"] = sorted(
        merged,
        key=lambda record: (
            float(record.get("elapsed_seconds", float("inf"))),
            int(record.get("step", 0) or 0),
            int(record.get("episode", 0) or 0),
        ),
    )


def resolve_metric(summary: dict[str, Any], metric: str) -> float:
    if "." in metric:
        value: Any = summary
        for part in metric.split("."):
            value = value[part]
        return float(value)

    if metric.startswith("eval_"):
        return float(summary["eval"][metric.removeprefix("eval_")])
    if metric.startswith("train_"):
        return float(summary["train"][metric.removeprefix("train_")])
    if metric in summary:
        return float(summary[metric])
    if metric in summary.get("eval", {}):
        return float(summary["eval"][metric])
    raise KeyError(f"Primary metric '{metric}' was not found in run summary")


def comparable_score(metric_value: float, mode: str) -> float:
    normalized = mode.lower()
    if normalized in {"maximize", "max", "higher_is_better"}:
        return metric_value
    if normalized in {"minimize", "min", "lower_is_better"}:
        return -metric_value
    raise ValueError(f"Unsupported primary_metric_mode: {mode}")


def nvidia_smi_sample() -> dict[str, Any] | None:
    query = (
        "name,utilization.gpu,utilization.memory,memory.used,memory.total,"
        "power.draw,power.limit"
    )
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None

    line = completed.stdout.strip().splitlines()[0] if completed.stdout.strip() else ""
    if not line:
        return None
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 7:
        return None

    def as_float(value: str) -> float | None:
        try:
            return float(value)
        except ValueError:
            return None

    return {
        "name": parts[0],
        "gpu_util_percent": as_float(parts[1]),
        "memory_util_percent": as_float(parts[2]),
        "memory_used_mb": as_float(parts[3]),
        "memory_total_mb": as_float(parts[4]),
        "power_draw_w": as_float(parts[5]),
        "power_limit_w": as_float(parts[6]),
    }


def mps_sample() -> dict[str, Any] | None:
    torch = require_torch()
    if not hasattr(torch, "mps"):
        return None
    try:
        return {
            "mps_current_allocated_mb": torch.mps.current_allocated_memory() / (1024 * 1024),
            "mps_driver_allocated_mb": torch.mps.driver_allocated_memory() / (1024 * 1024),
            "mps_recommended_max_memory_mb": torch.mps.recommended_max_memory() / (1024 * 1024),
        }
    except RuntimeError:
        return None


class UtilizationMonitor:
    def __init__(self, device: Any, interval_seconds: float = 1.0) -> None:
        self.device = device
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._wall_start = 0.0
        self._cpu_start = 0.0
        self._wall_end = 0.0
        self._cpu_end = 0.0

    def __enter__(self) -> "UtilizationMonitor":
        torch = require_torch()
        self._wall_start = time.perf_counter()
        self._cpu_start = time.process_time()
        if self.device.type == "cuda" and torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats(self.device)
            except RuntimeError:
                pass
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        torch = require_torch()
        if self.device.type == "cuda" and torch.cuda.is_available():
            try:
                torch.cuda.synchronize(self.device)
            except RuntimeError:
                pass
        elif self.device.type == "mps" and hasattr(torch, "mps"):
            try:
                torch.mps.synchronize()
            except RuntimeError:
                pass
        self._wall_end = time.perf_counter()
        self._cpu_end = time.process_time()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(0.1, self.interval_seconds * 2))

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            sample = nvidia_smi_sample() if self.device.type == "cuda" else None
            if sample is None and self.device.type == "mps":
                sample = mps_sample()
            if sample is not None:
                sample["sampled_at"] = time.time()
                self.samples.append(sample)
            self._stop.wait(self.interval_seconds)

    def summary(self, train_summary: dict[str, Any]) -> dict[str, Any]:
        torch = require_torch()
        wall_clock = max(self._wall_end - self._wall_start, 1e-9)
        process_cpu_seconds = max(self._cpu_end - self._cpu_start, 0.0)
        total_steps = int(train_summary.get("total_steps", 0) or 0)
        gradient_updates_raw = train_summary.get("gradient_updates")
        gradient_updates = int(gradient_updates_raw or 0) if gradient_updates_raw is not None else None
        steps_per_second = total_steps / wall_clock if total_steps else 0.0
        # Missing gradient-update instrumentation is distinct from a measured
        # zero; surface it as null so summaries do not imply learning never ran.
        updates_per_second = gradient_updates / wall_clock if gradient_updates is not None else None

        summary: dict[str, Any] = {
            "device": str(self.device),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "wall_clock_seconds": wall_clock,
            "process_cpu_seconds": process_cpu_seconds,
            "process_cpu_util_percent": 100.0 * process_cpu_seconds / wall_clock,
            "steps_per_second": steps_per_second,
            "gradient_updates": gradient_updates,
            "updates_per_second": updates_per_second,
            "sample_count": len(self.samples),
        }

        if self.device.type == "cuda" and torch.cuda.is_available():
            try:
                props = torch.cuda.get_device_properties(self.device)
                summary["cuda_device_name"] = props.name
                summary["cuda_total_memory_mb"] = props.total_memory / (1024 * 1024)
                summary["cuda_peak_memory_allocated_mb"] = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)
                summary["cuda_peak_memory_reserved_mb"] = torch.cuda.max_memory_reserved(self.device) / (1024 * 1024)
            except RuntimeError:
                pass
        elif self.device.type == "cpu":
            nvidia_sample = nvidia_smi_sample()
            if nvidia_sample and nvidia_sample.get("name"):
                summary["visible_nvidia_device_name"] = nvidia_sample["name"]

        numeric_fields = [
            "gpu_util_percent",
            "memory_util_percent",
            "memory_used_mb",
            "memory_total_mb",
            "power_draw_w",
            "power_limit_w",
            "mps_current_allocated_mb",
            "mps_driver_allocated_mb",
            "mps_recommended_max_memory_mb",
        ]
        for field in numeric_fields:
            values = [float(sample[field]) for sample in self.samples if sample.get(field) is not None]
            if values:
                summary[f"avg_{field}"] = float(np.mean(values))
                summary[f"max_{field}"] = float(np.max(values))
        if self.samples and self.samples[0].get("name"):
            summary["nvidia_smi_device_name"] = self.samples[0]["name"]
        summary["notes"] = utilization_notes(summary, train_summary)
        return summary


def utilization_notes(utilization: dict[str, Any], train_summary: dict[str, Any]) -> str:
    device = utilization.get("cuda_device_name") or utilization.get("nvidia_smi_device_name") or utilization.get("device", "unknown device")
    steps_per_second = float(utilization.get("steps_per_second", 0.0) or 0.0)
    updates_per_second_raw = utilization.get("updates_per_second")
    fragments = [
        f"Training ran on {device}.",
        f"Throughput was {steps_per_second:.1f} environment steps/sec",
    ]
    if updates_per_second_raw is not None:
        updates_per_second = float(updates_per_second_raw or 0.0)
        fragments.append(f"and {updates_per_second:.1f} reported gradient updates/sec")
    fragments[-1] += "."
    if updates_per_second_raw is None:
        fragments.append(
            "The trainable did not report gradient_updates, so updates_per_second is unavailable rather than a measured zero."
        )

    avg_gpu = utilization.get("avg_gpu_util_percent")
    max_gpu = utilization.get("max_gpu_util_percent")
    if avg_gpu is not None and max_gpu is not None:
        fragments.append(f"NVIDIA GPU utilization averaged {float(avg_gpu):.1f}% and peaked at {float(max_gpu):.1f}%.")
        if float(avg_gpu) < 50.0:
            fragments.append(
                "The run appears GPU-underutilized; candidates may try larger batches, more update work per collection step, or more vector environments."
            )
        elif float(avg_gpu) < 85.0:
            fragments.append(
                "The run used the GPU moderately; candidates can still explore batch size, UTD ratio, and vector environment count."
            )
        else:
            fragments.append(
                "The run appears close to GPU-saturated; candidates should prefer better learning efficiency over simply adding compute."
            )
    else:
        cpu_util = float(utilization.get("process_cpu_util_percent", 0.0) or 0.0)
        visible_nvidia = utilization.get("visible_nvidia_device_name")
        if visible_nvidia and str(utilization.get("device")) == "cpu":
            fragments.append(
                f"`nvidia-smi` can see {visible_nvidia}, but PyTorch selected CPU; check for a CPU-only Torch wheel or request `device=cuda` explicitly."
            )
        avg_mps_driver = utilization.get("avg_mps_driver_allocated_mb")
        max_mps_driver = utilization.get("max_mps_driver_allocated_mb")
        mps_limit = utilization.get("max_mps_recommended_max_memory_mb") or utilization.get("avg_mps_recommended_max_memory_mb")
        if avg_mps_driver is not None and max_mps_driver is not None:
            fragments.append(
                f"Apple MPS driver memory averaged {float(avg_mps_driver):.1f} MB and peaked at {float(max_mps_driver):.1f} MB."
            )
            if mps_limit:
                memory_fraction = 100.0 * float(max_mps_driver) / max(float(mps_limit), 1e-9)
                fragments.append(f"Peak MPS memory was about {memory_fraction:.1f}% of the recommended working-set limit.")
            fragments.append(
                f"Process CPU time was {cpu_util:.1f}% of wall time. macOS/MPS does not expose NVIDIA-style GPU utilization through nvidia-smi."
            )
        else:
            fragments.append(
                f"No GPU utilization samples were available; process CPU time was {cpu_util:.1f}% of wall time."
            )

    if train_summary.get("vector_envs") is not None:
        fragments.append(
            f"The trainable reported {train_summary.get('vector_envs')} vector environments and {train_summary.get('gradient_updates', 0)} gradient updates."
        )
    return " ".join(fragments)


def run_experiment(
    benchmark_path: Path,
    candidate_path: Path,
    tag: str,
    out_dir: Path,
    results_path: Path | None,
    train_episodes_override: int | None = None,
    train_seconds_override: float | None = None,
    eval_episodes_override: int | None = None,
    init_checkpoint: Path | None = None,
    session_dir: Path | None = None,
    evolution_metadata: dict[str, Any] | None = None,
    headless_env: bool = False,
    status_interval_seconds: float = 10.0,
    compact_status: bool = False,
    compact_status_file: Path | None = None,
    train_probe_enabled: bool | None = None,
    train_probe_interval_seconds: float | None = None,
    train_probe_episodes: int | None = None,
) -> dict[str, Any]:
    benchmark = load_benchmark(benchmark_path)
    trainable_module = load_trainable_module(candidate_path)
    candidate = trainable_module.get_candidate()
    headless_env_state = {
        "requested": bool(headless_env),
        "effective": False,
        "reason": None,
        "message": None,
    }
    if train_episodes_override is not None:
        benchmark.train_episodes = int(train_episodes_override)
    if train_seconds_override is not None:
        benchmark.train_seconds = float(train_seconds_override)
    if eval_episodes_override is not None:
        benchmark.eval_episodes = int(eval_episodes_override)
    if train_probe_enabled is not None:
        benchmark.train_probe.enabled = bool(train_probe_enabled)
    if train_probe_interval_seconds is not None:
        benchmark.train_probe.interval_seconds = float(train_probe_interval_seconds)
    if train_probe_episodes is not None:
        benchmark.train_probe.episodes = int(train_probe_episodes)
    if headless_env:
        headless_env_state = apply_headless_env_override(benchmark)

    device = select_device(benchmark.device)
    tag = normalize_run_tag(tag)
    run_id = time.strftime("%Y%m%d-%H%M%S") + f"-{tag}"
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    live_writer = make_live_writer(session_dir, run_id, tag, benchmark, candidate, headless_env=headless_env)
    compact_status_enabled = bool(compact_status or compact_status_file is not None)
    status_writer = (
        make_compact_status_writer(
            status_interval_seconds,
            train_seconds=benchmark.train_seconds,
            train_episodes=benchmark.train_episodes,
            emit_stderr=bool(compact_status),
            compact_status_file=compact_status_file,
        )
        if compact_status_enabled
        else None
    )
    probe_callback = make_policy_probe_callback(trainable_module, benchmark, candidate, device)
    live_callback = combine_live_callbacks(probe_callback, live_writer, status_writer)

    def make_training_env(control_type: str | None = None, reward_recipe: str | None = None) -> gym.Env[np.ndarray, np.ndarray]:
        env = make_env(
            benchmark,
            control_type,
            reward_recipe,
            trainable_module.RewardRecipeWrapper,
        )
        wrap_env = getattr(live_writer, "wrap_env", None) if live_writer is not None else None
        return wrap_env(env) if wrap_env is not None else env

    with UtilizationMonitor(device) as utilization_monitor:
        agent, train_summary = trainable_module.train_agent(
            benchmark,
            make_training_env,
            candidate,
            device,
            init_checkpoint=init_checkpoint,
            live_callback=live_callback,
        )
    merge_policy_probe_records(train_summary, getattr(probe_callback, "probe_records", []))
    validate_train_curve_contract(train_summary)
    normalize_train_summary_curve(train_summary)
    utilization_summary = utilization_monitor.summary(train_summary)
    if live_callback is not None:
        live_callback(
            status="evaluating",
            episode_records=train_summary["episode_records"],
            total_steps=train_summary.get("total_steps", 0),
            last_metrics=train_summary.get("last_metrics"),
        )
    eval_summary = evaluate_agent(agent, benchmark, candidate)
    if live_callback is not None:
        live_callback(
            status="finished",
            episode_records=train_summary["episode_records"],
            total_steps=train_summary.get("total_steps", 0),
            last_metrics=train_summary.get("last_metrics"),
        )
    checkpoint_path = run_dir / "agent_checkpoint.pt"
    trainable_module.save_agent_checkpoint(
        agent,
        checkpoint_path,
        metadata={
            "run_id": run_id,
            "tag": tag,
            "candidate": candidate_metadata(candidate),
        },
    )

    summary = {
        "run_id": run_id,
        "tag": tag,
        "session": {
            "session_dir": str(session_dir) if session_dir is not None else None,
            "results_path": str(results_path) if results_path is not None else None,
            "log_path": str(session_dir / "outer_loop_log.md") if session_dir is not None else None,
        },
        "lineage": {
            "mode": "warm_start" if init_checkpoint is not None else "from_scratch",
            "init_checkpoint": str(init_checkpoint) if init_checkpoint is not None else None,
            "parent_run_id": train_summary.get("resumed_from", {}).get("run_id") if train_summary.get("resumed_from") else None,
            "parent_tag": train_summary.get("resumed_from", {}).get("tag") if train_summary.get("resumed_from") else None,
        },
        "run_options": {
            "headless_env": bool(headless_env_state["effective"]),
            "headless_env_requested": bool(headless_env_state["requested"]),
            "headless_env_effective": bool(headless_env_state["effective"]),
            "headless_env_reason": headless_env_state["reason"],
            "headless_env_message": headless_env_state["message"],
            "compact_status": bool(compact_status_enabled),
            "compact_status_stderr": bool(compact_status),
            "compact_status_file": str(compact_status_file) if compact_status_file is not None else None,
            "status_interval_seconds": float(status_interval_seconds),
        },
        "benchmark": {
            "name": benchmark.name,
            "env_id": benchmark.env_id,
            "env_kwargs": benchmark.env_kwargs,
            "train_episodes": benchmark.train_episodes,
            "train_seconds": benchmark.train_seconds,
            "eval_episodes": benchmark.eval_episodes,
            "max_steps": benchmark.max_steps,
            "primary_metric": benchmark.primary_metric,
            "primary_metric_mode": benchmark.primary_metric_mode,
            "device": str(device),
            "eval_case_bank": str(benchmark.eval_case_bank) if benchmark.eval_case_bank is not None else None,
            "train_probe": asdict(benchmark.train_probe),
        },
        "candidate": candidate_metadata(candidate),
        "train": public_summary(train_summary),
        "eval": public_summary(eval_summary),
        "system_utilization": utilization_summary,
        "system_utilization_notes": utilization_summary["notes"],
        "artifacts": {
            "checkpoint_path": str(checkpoint_path),
        },
        "evolution": evolution_metadata or {},
    }
    metric_value = resolve_metric(summary, benchmark.primary_metric)
    summary["objective"] = {
        "metric": benchmark.primary_metric,
        "mode": benchmark.primary_metric_mode,
        "value": metric_value,
    }
    summary["score"] = comparable_score(metric_value, benchmark.primary_metric_mode)

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=json_default), encoding="utf-8")
    (run_dir / "train_episodes.json").write_text(json.dumps(train_summary["episode_records"], indent=2, default=json_default), encoding="utf-8")
    (run_dir / "eval_episodes.json").write_text(json.dumps(eval_summary["episode_records"], indent=2, default=json_default), encoding="utf-8")
    (run_dir / "candidate_snapshot.json").write_text(
        json.dumps(
            {
                "candidate": candidate_metadata(candidate),
            },
            indent=2,
            default=json_default,
        ),
        encoding="utf-8",
    )
    (run_dir / "benchmark_snapshot.json").write_text(benchmark_path.read_text(encoding="utf-8"), encoding="utf-8")
    (run_dir / "trainable_snapshot.py").write_text(candidate_path.read_text(encoding="utf-8"), encoding="utf-8")

    if results_path is not None:
        append_result(results_path, summary)
    return summary
