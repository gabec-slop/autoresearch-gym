from __future__ import annotations

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
from PIL import Image, ImageDraw

from autoresearch_gym.external.base import ArtifactSet, CommandSpec, RunBundle
from autoresearch_gym.runner.curves import make_train_episode_record

DASHBOARD_FRAME_WIDTH = 720
DASHBOARD_FRAME_HEIGHT = 480
DASHBOARD_FRAME_SIZE = (DASHBOARD_FRAME_WIDTH, DASHBOARD_FRAME_HEIGHT)
DEFAULT_TRAJECTORY_PLAYBACK_FPS = 20.0


MJLAB_ROLLOUT_SCRIPT = r'''
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "glfw" if os.name == "nt" else "egl")

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
import mjlab.utils.os as mjlab_os
from mjlab.utils.torch import configure_torch_backends

import mjlab.tasks  # noqa: F401

if not hasattr(mjlab_os, "update_assets"):
    def update_assets(assets, asset_dir, meshdir):
        asset_root = Path(asset_dir)
        prefix = str(meshdir or "").replace("\\", "/").strip("/")
        for path in asset_root.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(asset_root).as_posix()
            data = path.read_bytes()
            assets[rel] = data
            if prefix:
                assets[f"{prefix}/{rel}"] = data
    mjlab_os.update_assets = update_assets

import src.tasks  # noqa: F401
from src.tasks.tracking.mdp.metrics import compute_mpkpe, compute_root_relative_mpkpe  # noqa: E402


def _configure_motion(env_cfg, motion_file: str | None) -> None:
    if "motion" not in env_cfg.commands:
        return
    motion_cmd = env_cfg.commands["motion"]
    if not isinstance(motion_cmd, MotionCommandCfg):
        return
    if motion_file is None:
        raise ValueError("tracking rollout requires --motion-file")
    motion_path = Path(motion_file).expanduser().resolve()
    if not motion_path.exists():
        raise FileNotFoundError(f"motion file not found: {motion_path}")
    motion_cmd.motion_file = str(motion_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--motion-file", default=None)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=9000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--frame-dir", default=None)
    parser.add_argument("--frame-count", type=int, default=24)
    parser.add_argument("--no-terminations", action="store_true")
    args = parser.parse_args()

    configure_torch_backends()
    os.environ.setdefault("MUJOCO_GL", "egl")
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env_cfg = load_env_cfg(args.task_id, play=True)
    agent_cfg = load_rl_cfg(args.task_id)
    env_cfg.seed = args.seed
    env_cfg.scene.num_envs = max(1, int(args.num_envs))
    env_cfg.sim.nconmax = max(int(env_cfg.sim.nconmax or 0), 256)
    env_cfg.sim.njmax = max(int(env_cfg.sim.njmax or 0), 512)
    if args.no_terminations:
        env_cfg.terminations = {}
    _configure_motion(env_cfg, args.motion_file)

    render_mode = "rgb_array" if args.frame_dir else None
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(args.task_id) or MjlabOnPolicyRunner
    runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
    runner.load(str(Path(args.checkpoint).resolve()), load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)

    obs = wrapped.get_observations()
    rewards = []
    done_counts = []
    mpkpe_values = []
    r_mpkpe_values = []
    frame_paths = []
    frame_dir = Path(args.frame_dir) if args.frame_dir else None
    if frame_dir is not None:
        frame_dir.mkdir(parents=True, exist_ok=True)
    frame_stride = max(1, args.steps // max(1, args.frame_count))

    for step in range(max(1, int(args.steps))):
        with torch.no_grad():
            actions = policy(obs)
        obs, reward, dones, extras = wrapped.step(actions)
        del extras
        rewards.append(float(reward.detach().mean().cpu()))
        done_counts.append(float(dones.detach().float().mean().cpu()))
        if "motion" in getattr(wrapped.unwrapped.command_manager, "active_terms", []):
            motion_command = wrapped.unwrapped.command_manager.get_term("motion")
            mpkpe_values.append(float(compute_mpkpe(motion_command).detach().mean().cpu()))
            r_mpkpe_values.append(float(compute_root_relative_mpkpe(motion_command).detach().mean().cpu()))
        if frame_dir is not None and len(frame_paths) < args.frame_count and step % frame_stride == 0:
            frame = wrapped.unwrapped.render()
            if frame is not None:
                if isinstance(frame, np.ndarray) and frame.ndim == 4:
                    frame = frame[0]
                frame = np.asarray(frame)
                if frame.dtype != np.uint8:
                    frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
                frame_path = frame_dir / f"frame_{len(frame_paths):04d}.jpg"
                imageio.imwrite(frame_path, frame)
                frame_paths.append(str(frame_path))

    wrapped.close()
    total_return = float(np.sum(rewards))
    payload = {
        "task_id": args.task_id,
        "steps": int(args.steps),
        "num_envs": int(args.num_envs),
        "device": device,
        "avg_step_reward": float(np.mean(rewards)) if rewards else 0.0,
        "return": total_return,
        "done_fraction": float(np.mean(done_counts)) if done_counts else 0.0,
        "frames": frame_paths,
    }
    if mpkpe_values:
        payload["avg_mpkpe"] = float(np.mean(mpkpe_values))
        payload["avg_r_mpkpe"] = float(np.mean(r_mpkpe_values)) if r_mpkpe_values else 0.0
    Path(args.out_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
'''


MJLAB_TRAIN_SCRIPT = r'''
from __future__ import annotations

import argparse
import copy
import inspect
import json
import os
import re
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw" if os.name == "nt" else "egl")

from scripts.train import TrainConfig, run_train  # noqa: E402

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_runner_cls  # noqa: E402
from mjlab.utils.torch import configure_torch_backends  # noqa: E402

import mjlab.tasks  # noqa: F401,E402
import mjlab.utils.os as mjlab_os  # noqa: E402

if not hasattr(mjlab_os, "update_assets"):
    def update_assets(assets, asset_dir, meshdir):
        asset_root = Path(asset_dir)
        prefix = str(meshdir or "").replace("\\", "/").strip("/")
        for path in asset_root.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(asset_root).as_posix()
            data = path.read_bytes()
            assets[rel] = data
            if prefix:
                assets[f"{prefix}/{rel}"] = data
    mjlab_os.update_assets = update_assets

import src.tasks  # noqa: F401,E402

try:
    from src.tasks.tracking.rl.runner import MotionTrackingOnPolicyRunner

    _original_tracking_save = MotionTrackingOnPolicyRunner.save

    def _windows_safe_tracking_save(self, path, infos=None):
        return _original_tracking_save(self, str(path).replace("\\", "/"), infos)

    MotionTrackingOnPolicyRunner.save = _windows_safe_tracking_save
except Exception:
    pass


def _is_mapping(value):
    return isinstance(value, dict)


def _coerce(value):
    if isinstance(value, list):
        return tuple(_coerce(item) for item in value)
    if isinstance(value, dict):
        return {key: _coerce(item) for key, item in value.items() if item is not None}
    return value


def _set_attr(target, name, value, applied, skipped, label):
    if value is None:
        return
    if target is None:
        skipped.append({"field": label, "reason": "missing target"})
        return
    if hasattr(target, name):
        try:
            setattr(target, name, _coerce(value))
        except Exception as exc:
            skipped.append({"field": label, "reason": f"{type(exc).__name__}: {exc}"})
            return
        applied.append(label)
        return
    skipped.append({"field": label, "reason": "missing attribute"})


def _mapping_get(container, key):
    if isinstance(container, dict):
        return container.get(key)
    if hasattr(container, key):
        return getattr(container, key)
    return None


def _mapping_pop(container, key):
    if isinstance(container, dict):
        container.pop(key, None)
        return True
    return False


def _update_params(term, updates, applied, skipped, label):
    if not _is_mapping(updates):
        return
    func = getattr(term, "func", None)
    allowed = None
    if func is not None:
        try:
            signature = inspect.signature(func)
            has_var_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
            if not has_var_kwargs:
                allowed = set(signature.parameters) - {"env", "env_ids", "self"}
        except Exception:
            allowed = None
    params = getattr(term, "params", None)
    if isinstance(params, dict):
        for key, value in updates.items():
            if value is None:
                continue
            if allowed is not None and key not in allowed:
                skipped.append({"field": f"{label}.params.{key}", "reason": "unsupported function parameter"})
                continue
            params[key] = _coerce(value)
            applied.append(f"{label}.params.{key}")
        return
    for key, value in updates.items():
        if allowed is not None and key not in allowed:
            skipped.append({"field": f"{label}.params.{key}", "reason": "unsupported function parameter"})
            continue
        _set_attr(params, key, value, applied, skipped, f"{label}.params.{key}")


def _apply_model_cfg(model_cfg, section, applied, skipped, label):
    if not _is_mapping(section):
        return
    for key in ("hidden_dims", "activation", "obs_normalization", "rnn_type", "rnn_hidden_dim", "rnn_num_layers", "class_name"):
        _set_attr(model_cfg, key, section.get(key), applied, skipped, f"{label}.{key}")
    distribution = getattr(model_cfg, "distribution_cfg", None)
    for key in ("init_std", "std_type", "noise_std_type"):
        _set_attr(distribution, key, section.get(key), applied, skipped, f"{label}.distribution_cfg.{key}")
    if _is_mapping(section.get("distribution_cfg")):
        for key, value in section["distribution_cfg"].items():
            _set_attr(distribution, key, value, applied, skipped, f"{label}.distribution_cfg.{key}")


def _apply_ppo_cfg(algorithm_cfg, section, applied, skipped):
    if not _is_mapping(section):
        return
    aliases = {"gae_lambda": "lam", "clip_range": "clip_param"}
    supported = (
        "num_learning_epochs",
        "num_mini_batches",
        "learning_rate",
        "schedule",
        "gamma",
        "lam",
        "entropy_coef",
        "desired_kl",
        "max_grad_norm",
        "value_loss_coef",
        "use_clipped_value_loss",
        "clip_param",
        "normalize_advantage_per_mini_batch",
        "optimizer",
        "share_cnn_encoders",
        "class_name",
    )
    for key, value in section.items():
        target_key = aliases.get(key, key)
        if target_key in supported:
            _set_attr(algorithm_cfg, target_key, value, applied, skipped, f"ppo.{target_key}")
        elif value is not None:
            skipped.append({"field": f"ppo.{key}", "reason": "unsupported ppo field"})


def _apply_runner_cfg(cfg, section, run_id, applied, skipped):
    if not _is_mapping(section):
        return
    for key in ("seed", "num_steps_per_env", "max_iterations", "save_interval", "logger", "wandb_project", "wandb_tags", "resume", "load_run", "load_checkpoint", "clip_actions", "upload_model", "obs_groups"):
        _set_attr(cfg.agent, key, section.get(key), applied, skipped, f"runner.{key}")
    suffix = section.get("run_name_suffix")
    if suffix:
        cfg.agent.run_name = f"{run_id}_{suffix}"
        applied.append("runner.run_name_suffix")


def _apply_environment_cfg(cfg, section, applied, skipped):
    if not _is_mapping(section):
        return
    for key in ("episode_length_s", "decimation"):
        _set_attr(cfg.env, key, section.get(key), applied, skipped, f"environment.{key}")
    if section.get("action_scale") is not None:
        action = _mapping_get(getattr(cfg.env, "actions", {}), "joint_pos")
        _set_attr(action, "scale", section.get("action_scale"), applied, skipped, "environment.action_scale")
    sim = getattr(cfg.env, "sim", None)
    sim_section = section.get("sim") if _is_mapping(section.get("sim")) else {}
    for key in ("nconmax", "njmax"):
        _set_attr(sim, key, sim_section.get(key), applied, skipped, f"environment.sim.{key}")
    mujoco = getattr(sim, "mujoco", None)
    for key in ("timestep", "iterations", "ls_iterations"):
        _set_attr(mujoco, key, sim_section.get(key), applied, skipped, f"environment.sim.{key}")
    for key in ("video", "video_length", "video_interval", "enable_nan_guard"):
        _set_attr(cfg, key, section.get(key), applied, skipped, f"environment.{key}")


def _apply_command_cfg(cfg, command_name, section, applied, skipped, label):
    if not _is_mapping(section):
        return
    command = _mapping_get(getattr(cfg.env, "commands", {}), command_name)
    if command is None:
        skipped.append({"field": label, "reason": "missing command"})
        return
    for key in ("resampling_time_range", "debug_vis", "rel_standing_envs", "heading_command", "heading_control_stiffness", "pose_range", "velocity_range", "joint_position_range"):
        _set_attr(command, key, section.get(key), applied, skipped, f"{label}.{key}")
    ranges = section.get("ranges")
    if _is_mapping(ranges):
        target_ranges = getattr(command, "ranges", None)
        if isinstance(target_ranges, dict):
            for key, value in ranges.items():
                if value is None:
                    continue
                target_ranges[key] = _coerce(value)
                applied.append(f"{label}.ranges.{key}")
        else:
            for key, value in ranges.items():
                _set_attr(target_ranges, key, value, applied, skipped, f"{label}.ranges.{key}")


def _apply_rewards(cfg, weights, params, applied, skipped):
    rewards = getattr(cfg.env, "rewards", {})
    if _is_mapping(weights):
        for name, weight in weights.items():
            term = _mapping_get(rewards, name)
            if term is None:
                skipped.append({"field": f"reward_weights.{name}", "reason": "missing reward"})
                continue
            _set_attr(term, "weight", weight, applied, skipped, f"reward_weights.{name}")
    if _is_mapping(params):
        for name, updates in params.items():
            term = _mapping_get(rewards, name)
            if term is None:
                skipped.append({"field": f"reward_params.{name}", "reason": "missing reward"})
                continue
            _update_params(term, updates, applied, skipped, f"reward_params.{name}")


def _apply_term_collection(cfg, collection_name, overrides, applied, skipped):
    if not _is_mapping(overrides):
        return
    collection = getattr(cfg.env, collection_name, {})
    for name, spec in overrides.items():
        if not _is_mapping(spec):
            continue
        if spec.get("enabled") is False:
            if _mapping_pop(collection, name):
                applied.append(f"{collection_name}.{name}.enabled")
            else:
                skipped.append({"field": f"{collection_name}.{name}.enabled", "reason": "cannot remove term"})
            continue
        term = _mapping_get(collection, name)
        if term is None:
            skipped.append({"field": f"{collection_name}.{name}", "reason": "missing term"})
            continue
        _set_attr(term, "interval_range_s", spec.get("interval_range_s"), applied, skipped, f"{collection_name}.{name}.interval_range_s")
        _update_params(term, spec.get("params"), applied, skipped, f"{collection_name}.{name}")


def _apply_terrain_cfg(cfg, section, applied, skipped):
    if not _is_mapping(section):
        return
    scene = getattr(cfg.env, "scene", None)
    _set_attr(scene, "max_init_terrain_level", section.get("max_init_terrain_level"), applied, skipped, "terrain.max_init_terrain_level")
    terrain = getattr(scene, "terrain", None)
    _set_attr(terrain, "max_init_terrain_level", section.get("max_init_terrain_level"), applied, skipped, "terrain.terrain.max_init_terrain_level")
    _set_attr(terrain, "extent", section.get("terrain_extent"), applied, skipped, "terrain.extent")


def _load_recipe(path):
    if not path:
        return {}
    recipe_path = Path(path)
    if not recipe_path.exists():
        return {}
    payload = json.loads(recipe_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _json_write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for attempt in range(20):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.05 * (attempt + 1))
    tmp.replace(path)


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return int(default)


def _checkpoint_index(path):
    match = re.search(r"model_(\d+)\.pt$", Path(path).name)
    return int(match.group(1)) if match else -1


def _load_scalar_events(log_dir):
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception:
        return {}, "tensorboard_unavailable"
    scalars = {}
    event_dirs = sorted({path.parent for path in Path(log_dir).rglob("events.out.tfevents*")})
    for event_dir in event_dirs:
        try:
            accumulator = EventAccumulator(str(event_dir), size_guidance={"scalars": 0})
            accumulator.Reload()
            tags = accumulator.Tags().get("scalars", [])
            for tag in tags:
                scalars.setdefault(tag, [])
                scalars[tag].extend(accumulator.Scalars(tag))
        except Exception:
            continue
    for tag, events in list(scalars.items()):
        deduped = {}
        for event in events:
            deduped[int(event.step)] = event
        scalars[tag] = [deduped[key] for key in sorted(deduped)]
    return scalars, "ok" if scalars else "waiting_for_tensorboard"


def _select_tag(scalars, candidates):
    tags = list(scalars)
    lower = {tag: tag.lower() for tag in tags}
    for candidate in candidates:
        candidate_lower = candidate.lower()
        for tag in tags:
            if lower[tag] == candidate_lower:
                return tag
    for candidate in candidates:
        candidate_lower = candidate.lower()
        for tag in tags:
            if candidate_lower in lower[tag]:
                return tag
    return None


def _scalar_latest_metrics(scalars):
    metrics = {}
    for tag, events in scalars.items():
        if not events:
            continue
        key = _scalar_metric_key(tag)
        if not key:
            continue
        metrics[key[-80:]] = _safe_float(events[-1].value)
    return metrics


def _scalar_metric_key(tag):
    return re.sub(r"[^a-zA-Z0-9_]+", "_", str(tag).strip().lower()).strip("_")


CURRICULUM_SIGNAL_KEYS = (
    "episode_reward_track_linear_velocity",
    "episode_reward_track_angular_velocity",
    "episode_reward_body_orientation_l2",
    "episode_reward_pose",
    "episode_reward_body_ang_vel",
    "episode_reward_action_rate_l2",
    "episode_reward_stand_still",
    "episode_reward_foot_gait",
    "episode_reward_foot_clearance",
    "episode_reward_foot_slip",
    "episode_reward_soft_landing",
    "episode_reward_is_terminated",
    "episode_reward_motion_global_root_pos",
    "episode_reward_motion_global_root_ori",
    "episode_reward_motion_body_pos",
    "episode_reward_motion_body_ori",
    "episode_reward_motion_body_lin_vel",
    "episode_reward_motion_body_ang_vel",
    "episode_reward_joint_limit",
    "episode_reward_self_collisions",
    "episode_termination_time_out",
    "episode_termination_fell_over",
    "episode_termination_illegal_contact",
    "episode_termination_anchor_pos",
    "episode_termination_anchor_ori",
    "episode_termination_ee_body_pos",
    "metrics_mpkpe",
    "metrics_r_mpkpe",
    "metrics_twist_error_vel_xy",
    "metrics_twist_error_vel_yaw",
    "metrics_slip_velocity_mean",
    "metrics_landing_force_mean",
    "curriculum_terrain_levels",
)


def _curriculum_signal_events(scalars):
    by_key = {}
    for tag, events in scalars.items():
        key = _scalar_metric_key(tag)[-80:]
        if key in CURRICULUM_SIGNAL_KEYS and events:
            by_key[key] = sorted(events, key=lambda event: int(getattr(event, "step", 0)))
    return by_key


def _latest_signal_value(events, event_step):
    selected = None
    target_step = int(event_step)
    for candidate in events:
        if int(getattr(candidate, "step", 0)) > target_step:
            break
        selected = candidate
    return None if selected is None else _safe_float(selected.value)


def _command_velocity_stages(recipe):
    curriculum = recipe.get("curriculum_overrides") if _is_mapping(recipe) else None
    command_vel = curriculum.get("command_vel") if _is_mapping(curriculum) else None
    params = command_vel.get("params") if _is_mapping(command_vel) else None
    stages = params.get("velocity_stages") if _is_mapping(params) else None
    if not isinstance(stages, list):
        return []
    normalized = []
    for index, stage in enumerate(stages):
        if not _is_mapping(stage):
            continue
        normalized.append(
            {
                "index": index,
                "step": _safe_int(stage.get("step"), -1),
                "lin_vel_x": stage.get("lin_vel_x"),
                "lin_vel_y": stage.get("lin_vel_y"),
                "ang_vel_z": stage.get("ang_vel_z"),
            }
        )
    return sorted(normalized, key=lambda item: int(item.get("step", -1)))


def _range_metrics(prefix, values):
    if not isinstance(values, (list, tuple)) or len(values) < 2:
        return {}
    return {
        f"{prefix}_min": _safe_float(values[0]),
        f"{prefix}_max": _safe_float(values[1]),
    }


def _command_stage_metrics(record_index, *, steps_per_env, command_stages):
    if not command_stages:
        return {}
    common_step = max(0, int(record_index) * int(steps_per_env))
    active = command_stages[0]
    for stage in command_stages:
        if common_step >= int(stage.get("step", -1)):
            active = stage
        else:
            break
    metrics = {
        "curriculum_command_stage": float(active.get("index", 0)),
        "curriculum_command_stage_start_step": float(active.get("step", -1)),
        "curriculum_command_common_step": float(common_step),
    }
    metrics.update(_range_metrics("curriculum_command_lin_vel_x", active.get("lin_vel_x")))
    metrics.update(_range_metrics("curriculum_command_lin_vel_y", active.get("lin_vel_y")))
    metrics.update(_range_metrics("curriculum_command_ang_vel_z", active.get("ang_vel_z")))
    return metrics


def _normal_diagnostic_series_spec(raw):
    if not _is_mapping(raw):
        return None
    key = str(raw.get("key") or raw.get("metric") or raw.get("value_key") or "").strip()
    if not key:
        return None
    item = dict(raw)
    item["key"] = key
    item.setdefault("label", key.replace("_", " "))
    item.setdefault("source", "info_metrics")
    item.setdefault("chart", "normalized_line")
    item.setdefault("record_type", "train_collection_window")
    return item


def _recipe_diagnostic_series(recipe):
    diagnostic = recipe.get("diagnostic_series") if _is_mapping(recipe) else None
    if not diagnostic:
        return None
    if isinstance(diagnostic, list):
        raw_series = diagnostic
        base = {}
    elif _is_mapping(diagnostic):
        raw_series = diagnostic.get("series")
        base = {key: value for key, value in diagnostic.items() if key != "series"}
    else:
        return None
    if not isinstance(raw_series, list):
        return None
    series = []
    for raw in raw_series:
        item = _normal_diagnostic_series_spec(raw)
        if item is not None:
            series.append(item)
    if not series:
        return None
    return {
        "title": "Training diagnostics",
        "description": "Normalized diagnostic curves emitted by the training run.",
        "x_axis": "elapsed_seconds",
        **base,
        "series": series,
    }


def _diagnostic_series_metadata(records, recipe):
    diagnostic = _recipe_diagnostic_series(recipe)
    if diagnostic is None:
        return None
    available = {
        key
        for record in records
        for key, value in (record.get("info_metrics") or {}).items()
        if isinstance(value, (int, float))
    }
    series = []
    for spec in diagnostic["series"]:
        if spec["key"] not in available:
            continue
        series.append(dict(spec))
    if not series:
        return None
    return {**diagnostic, "series": series}


def _event_step_to_env_step(event_step, *, num_envs, steps_per_env):
    step = max(0, _safe_int(event_step))
    if step <= 0:
        return int(num_envs * steps_per_env)
    if step < 1_000_000:
        return int(step * num_envs * steps_per_env)
    return int(step)


def _records_from_scalars(scalars, *, num_envs, steps_per_env, started_at, command_stages=None):
    reward_tag = _select_tag(
        scalars,
        (
            "train/mean_reward",
            "charts/episodic_return",
            "episode/return",
            "episodic_return",
            "mean_reward",
            "reward",
        ),
    )
    length_tag = _select_tag(scalars, ("train/mean_episode_length", "charts/episodic_length", "episode/length", "length"))
    if reward_tag is None:
        return []
    length_by_step = {}
    if length_tag is not None:
        length_by_step = {int(event.step): _safe_float(event.value, steps_per_env) for event in scalars.get(length_tag, [])}
    signal_events = _curriculum_signal_events(scalars)
    records = []
    for index, event in enumerate(scalars.get(reward_tag, [])):
        env_step = _event_step_to_env_step(event.step, num_envs=num_envs, steps_per_env=steps_per_env)
        elapsed = max(0.0, _safe_float(getattr(event, "wall_time", 0.0)) - float(started_at))
        if elapsed <= 0.0:
            elapsed = max(0.0, time.time() - float(started_at))
        info_metrics = {
            "mjlab_num_envs": float(num_envs),
            "mjlab_iteration": float(index + 1),
            "samples_per_iteration": float(num_envs * steps_per_env),
        }
        info_metrics.update(
            _command_stage_metrics(
                index + 1,
                steps_per_env=steps_per_env,
                command_stages=command_stages or [],
            )
        )
        for key, events in signal_events.items():
            value = _latest_signal_value(events, event.step)
            if value is not None:
                info_metrics[key] = value
        records.append(
            {
                "record_type": "train_collection_window",
                "episode": index + 1,
                "return": _safe_float(event.value),
                "length": float(length_by_step.get(int(event.step), steps_per_env)),
                "success": False,
                "episodes_in_window": int(num_envs),
                "step": env_step,
                "env_steps_in_window": int(num_envs * steps_per_env),
                "elapsed_seconds": elapsed,
                "info_metrics": info_metrics,
            }
        )
    return records


def _read_probe_records(path):
    probe_path = Path(path)
    if not probe_path.exists():
        return []
    records = []
    for line in probe_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except Exception:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def _tensor_column_stats(prefix, tensor, columns):
    metrics = {}
    if tensor is None:
        return metrics
    try:
        data = tensor.detach().float()
        if data.ndim == 1:
            data = data.reshape(1, -1)
        for name, column in columns:
            if int(column) >= data.shape[1]:
                continue
            values = data[:, int(column)]
            metrics[f"{prefix}_{name}_mean"] = float(values.mean().cpu())
            metrics[f"{prefix}_{name}_std"] = float(values.std(unbiased=False).cpu())
            metrics[f"{prefix}_{name}_min"] = float(values.min().cpu())
            metrics[f"{prefix}_{name}_max"] = float(values.max().cpu())
    except Exception:
        return metrics
    return metrics


def _current_command_metrics(env):
    manager = getattr(env, "command_manager", None)
    if manager is None:
        return {}
    active_terms = set(getattr(manager, "active_terms", []) or [])
    for name in ("twist", "base_velocity", "velocity"):
        if active_terms and name not in active_terms:
            continue
        try:
            command = manager.get_command(name)
        except Exception:
            command = None
        metrics = _tensor_column_stats(
            f"command_{name}",
            command,
            (("lin_vel_x", 0), ("lin_vel_y", 1), ("ang_vel_z", 2)),
        )
        if metrics:
            metrics["command_active_term"] = name
            return metrics
    return {}


def _merge_command_metric_samples(samples):
    merged = {}
    if not samples:
        return merged
    keys = sorted({key for sample in samples for key in sample if key != "command_active_term"})
    for key in keys:
        values = [sample[key] for sample in samples if isinstance(sample.get(key), (int, float))]
        if values:
            merged[key] = float(np.mean(values))
    for sample in reversed(samples):
        if sample.get("command_active_term"):
            merged["command_active_term"] = sample["command_active_term"]
            break
    return merged


def _latest_live_visual(out_dir):
    sample_root = Path(out_dir) / "trajectories"
    manifests = sorted(sample_root.glob("sample_*/manifest.json"), key=lambda path: path.stat().st_mtime)
    if not manifests:
        return {}
    try:
        manifest = json.loads(manifests[-1].read_text(encoding="utf-8"))
    except Exception:
        return {}
    latest_frame = manifest.get("latest_frame_path")
    return {
        "mode": "sampled_trajectory",
        "live_frame_path": str(Path(out_dir) / "current_run_frame.jpg"),
        "trajectory_manifest_path": str(manifests[-1]),
        "trajectory_latest_frame_path": str(latest_frame) if latest_frame else None,
        "sampled_status": manifest.get("status", "completed"),
        "latest_sample_index": manifest.get("sample_index", 1),
        "source": str(manifest.get("source") or "mjlab_live_probe"),
    }


def _write_live_sample_manifest(out_dir, *, run_id, tag, sample_index, checkpoint_index, step, frame_paths, source="mjlab_live_probe"):
    out_dir = Path(out_dir)
    frames = [str(_normalize_dashboard_frame(Path(path))) for path in frame_paths]
    if not frames:
        return None
    manifest = {
        "run_id": run_id,
        "tag": tag,
        "sample_index": int(sample_index),
        "checkpoint_index": int(checkpoint_index),
        "episode": int(sample_index),
        "step": int(step),
        "status": "completed",
        "updated_at": time.time(),
        "frame_count": len(frames),
        "frames": frames,
        "latest_frame_path": frames[-1],
        "playback_fps": DEFAULT_TRAJECTORY_PLAYBACK_FPS,
        "frame_stride": 1,
        "sample_rate": 1.0,
        "width": DASHBOARD_FRAME_WIDTH,
        "height": DASHBOARD_FRAME_HEIGHT,
        "source": str(source),
    }
    manifest_path = out_dir / "trajectories" / f"sample_{int(sample_index):06d}" / "manifest.json"
    _json_write(manifest_path, manifest)
    try:
        import shutil

        shutil.copy2(frames[-1], out_dir / "current_run_frame.jpg")
    except Exception:
        pass
    return manifest_path


def _candidate_metadata(out_dir):
    try:
        payload = json.loads((Path(out_dir) / "bundle.json").read_text(encoding="utf-8"))
    except Exception:
        return {}
    candidate = payload.get("candidate")
    return candidate if isinstance(candidate, dict) else {}


def _write_partial_train_payload(
    *,
    out_dir,
    log_dir,
    run_id,
    tag,
    task_id,
    status,
    budget_mode,
    train_seconds,
    num_envs,
    steps_per_env,
    iterations,
    started_at,
    recipe_json=None,
    stop_reason=None,
    event_status="unknown",
):
    out_dir = Path(out_dir)
    scalars, scalar_status = _load_scalar_events(log_dir)
    recipe = _load_recipe(recipe_json)
    command_stages = _command_velocity_stages(recipe)
    records = _records_from_scalars(
        scalars,
        num_envs=num_envs,
        steps_per_env=steps_per_env,
        started_at=started_at,
        command_stages=command_stages,
    )
    probe_records = _read_probe_records(out_dir / "policy_probe_records.jsonl")
    all_records = sorted([*records, *probe_records], key=lambda item: (_safe_float(item.get("step")), item.get("record_type") == "policy_probe"))
    last_record = records[-1] if records else None
    total_steps = int(last_record.get("step", 0)) if last_record else 0
    if total_steps <= 0:
        total_steps = int(max(0, min(iterations, len(records))) * num_envs * steps_per_env)
    latest_metrics = _scalar_latest_metrics(scalars)
    latest_metrics.update(
        {
            "gradient_updates": float(len(records)),
            "mjlab_num_envs": float(num_envs),
            "samples_per_iteration": float(num_envs * steps_per_env),
            "mjlab_scalar_status": scalar_status,
        }
    )
    if last_record:
        for key, value in (last_record.get("info_metrics") or {}).items():
            if str(key).startswith("curriculum_command_"):
                latest_metrics[key] = value
    if event_status:
        latest_metrics["mjlab_live_status"] = str(event_status)
    probe_count = len(probe_records)
    latest_metrics["policy_probe_count"] = float(probe_count)
    if probe_records:
        latest_probe = probe_records[-1]
        latest_metrics["policy_probe_return"] = _safe_float(latest_probe.get("return", 0.0))
        latest_metrics["policy_probe_length"] = _safe_float(latest_probe.get("length", 0.0))
        if "elapsed_seconds" in latest_probe:
            latest_metrics["policy_probe_elapsed_seconds"] = _safe_float(latest_probe.get("elapsed_seconds", 0.0))
    avg_return = float(sum(record["return"] for record in records) / len(records)) if records else 0.0
    diagnostic_series = _diagnostic_series_metadata(records, recipe)
    payload = {
        "episode_records": all_records,
        "total_steps": total_steps,
        "env_steps": total_steps,
        "episodes_completed": int(len(records) * num_envs),
        "episode_batches": len(records),
        "gradient_updates": len(records),
        "avg_return": avg_return,
        "avg_length": float(sum(record["length"] for record in records) / len(records)) if records else 0.0,
        "last_metrics": latest_metrics,
        "stop_reason": stop_reason or ("mjlab_train_running" if status == "running" else "mjlab_train_complete"),
        "external_backend_scaffold": False,
        "mjlab_live": {
            "task_id": task_id,
            "scalar_status": scalar_status,
            "record_count": len(records),
            "probe_count": probe_count,
            "status": status,
        },
    }
    if diagnostic_series is not None:
        payload["diagnostic_series"] = diagnostic_series
    _json_write(out_dir / "train_result_partial.json", payload)
    if status != "running":
        _json_write(out_dir / "train_result.json", payload)
    live_visual = _latest_live_visual(out_dir)
    live_payload = {
        "run": {
            "run_id": run_id,
            "tag": tag,
            "status": status,
            "started_at": started_at,
            "updated_at": time.time(),
            "budget_mode": budget_mode,
            "train_seconds": train_seconds,
            "candidate": _candidate_metadata(out_dir),
            "frame_path": live_visual.get("live_frame_path"),
            "trajectory_manifest_path": live_visual.get("trajectory_manifest_path"),
            "trajectory_latest_frame_path": live_visual.get("trajectory_latest_frame_path"),
            "visual": live_visual,
        },
        "current": {
            "status": status,
            "step": total_steps,
            "env_steps": total_steps,
            "episodes_complete": int(len(records) * num_envs),
            "completed_episodes": int(len(records) * num_envs),
            "episode_batches": len(records),
            "avg_return": avg_return,
            "success_rate": 0.0,
            "info_metrics": latest_metrics,
        },
        "episodes": all_records,
        "latest_losses": latest_metrics,
        "diagnostic_series": diagnostic_series,
        "visual": live_visual,
    }
    if diagnostic_series is not None:
        live_payload["run"]["diagnostic_series"] = diagnostic_series
    _json_write(out_dir / "live" / "current_run_metrics.json", live_payload)
    with (out_dir / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time": time.time(), "status": status, "records": len(records), "probes": len(probe_records), "step": total_steps}) + "\n")
    with (out_dir / "live" / "status.log").open("a", encoding="utf-8") as handle:
        handle.write(f"st={status} step={total_steps} task={task_id} records={len(records)} probes={len(probe_records)}\n")
    return payload


def _run_checkpoint_probe(
    *,
    rollout_script,
    task_id,
    checkpoint,
    out_dir,
    num_envs,
    steps,
    seed,
    motion_file,
    python_executable,
    timeout,
    frame_dir=None,
    frame_count=0,
):
    out_json = Path(out_dir) / "policy_probes" / f"{Path(checkpoint).stem}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    log_dir = Path(out_dir) / "policy_probes" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    argv = [
        python_executable,
        str(rollout_script),
        "--task-id",
        str(task_id),
        "--checkpoint",
        str(checkpoint),
        "--out-json",
        str(out_json),
        "--num-envs",
        str(num_envs),
        "--steps",
        str(steps),
        "--seed",
        str(seed),
    ]
    if motion_file:
        argv.extend(["--motion-file", str(motion_file)])
    if frame_dir is not None and int(frame_count) > 0:
        argv.extend(["--frame-dir", str(frame_dir), "--frame-count", str(int(frame_count)), "--no-terminations"])
    started = time.time()
    try:
        completed = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False, env=_probe_subprocess_env())
    except subprocess.TimeoutExpired as exc:
        (log_dir / f"{Path(checkpoint).stem}.timeout.txt").write_text(
            f"timeout_seconds={timeout}\nelapsed_seconds={time.time() - started:.3f}\ncmd={' '.join(argv)}\n",
            encoding="utf-8",
        )
        if exc.stdout:
            (log_dir / f"{Path(checkpoint).stem}.stdout.log").write_text(str(exc.stdout), encoding="utf-8")
        if exc.stderr:
            (log_dir / f"{Path(checkpoint).stem}.stderr.log").write_text(str(exc.stderr), encoding="utf-8")
        raise
    (log_dir / f"{Path(checkpoint).stem}.stdout.log").write_text(completed.stdout or "", encoding="utf-8")
    (log_dir / f"{Path(checkpoint).stem}.stderr.log").write_text(completed.stderr or "", encoding="utf-8")
    if completed.returncode != 0:
        (log_dir / f"{Path(checkpoint).stem}.failed.txt").write_text(
            f"returncode={completed.returncode}\ncmd={' '.join(argv)}\n",
            encoding="utf-8",
        )
        return None
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    return payload


def _probe_subprocess_env():
    env = os.environ.copy()
    env["MUJOCO_GL"] = "glfw" if os.name == "nt" else env.get("MUJOCO_GL", "egl")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("WANDB_MODE", "offline")
    return env


def _run_train_context_probe_subprocess(
    *,
    train_script,
    task_id,
    run_id,
    num_envs,
    iterations,
    seed,
    gpu_id,
    recipe_json,
    out_dir,
    steps_per_env,
    budget_mode,
    train_seconds,
    checkpoint,
    checkpoint_index,
    probe_num_envs,
    probe_steps,
    frame_dir,
    frame_count,
    timeout,
):
    out_dir = Path(out_dir)
    checkpoint = Path(checkpoint)
    out_json = out_dir / "policy_probes" / f"{checkpoint.stem}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "policy_probes" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    argv = [
        sys.executable,
        str(train_script),
        "--task-id",
        str(task_id),
        "--run-id",
        str(run_id),
        "--num-envs",
        str(num_envs),
        "--iterations",
        str(iterations),
        "--seed",
        str(seed),
        "--gpu-id",
        str(gpu_id),
        "--steps-per-env",
        str(steps_per_env),
        "--budget-mode",
        str(budget_mode),
        "--probe-checkpoint",
        str(checkpoint),
        "--probe-checkpoint-index",
        str(checkpoint_index),
        "--probe-out-json",
        str(out_json),
        "--probe-num-envs",
        str(probe_num_envs),
        "--probe-steps",
        str(probe_steps),
        "--probe-seed",
        str(900000 + int(checkpoint_index)),
        "--probe-frame-count",
        str(int(frame_count)),
    ]
    if recipe_json:
        argv.extend(["--recipe-json", str(recipe_json)])
    if train_seconds is not None:
        argv.extend(["--train-seconds", str(train_seconds)])
    if frame_dir is not None and int(frame_count) > 0:
        argv.extend(["--probe-frame-dir", str(frame_dir)])
    started = time.time()
    try:
        completed = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False, env=_probe_subprocess_env())
    except subprocess.TimeoutExpired as exc:
        (log_dir / f"{checkpoint.stem}.timeout.txt").write_text(
            f"timeout_seconds={timeout}\nelapsed_seconds={time.time() - started:.3f}\ncmd={' '.join(argv)}\n",
            encoding="utf-8",
        )
        if exc.stdout:
            (log_dir / f"{checkpoint.stem}.stdout.log").write_text(str(exc.stdout), encoding="utf-8")
        if exc.stderr:
            (log_dir / f"{checkpoint.stem}.stderr.log").write_text(str(exc.stderr), encoding="utf-8")
        raise
    (log_dir / f"{checkpoint.stem}.stdout.log").write_text(completed.stdout or "", encoding="utf-8")
    (log_dir / f"{checkpoint.stem}.stderr.log").write_text(completed.stderr or "", encoding="utf-8")
    if completed.returncode != 0:
        (log_dir / f"{checkpoint.stem}.failed.txt").write_text(
            f"returncode={completed.returncode}\ncmd={' '.join(argv)}\n",
            encoding="utf-8",
        )
        return None
    return json.loads(out_json.read_text(encoding="utf-8"))


def _run_train_context_sample(
    *,
    cfg,
    task_id,
    checkpoint,
    num_envs,
    steps,
    seed,
    frame_dir,
    frame_count,
    checkpoint_index,
    steps_per_env,
):
    configure_torch_backends()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env_cfg = copy.deepcopy(cfg.env)
    agent_cfg = copy.deepcopy(cfg.agent)
    env_cfg.seed = int(seed)
    requested_num_envs = max(1, int(num_envs))
    env_cfg.scene.num_envs = 1 if frame_dir is not None else requested_num_envs
    env_cfg.sim.nconmax = max(int(env_cfg.sim.nconmax or 0), 256)
    env_cfg.sim.njmax = max(int(env_cfg.sim.njmax or 0), 512)

    render_mode = "rgb_array" if frame_dir is not None else None
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)
    common_step = max(0, int(checkpoint_index) * int(steps_per_env))
    try:
        env.common_step_counter = common_step
        env.curriculum_manager.compute(env_ids=torch.arange(env.num_envs, device=device))
    except Exception:
        pass
    wrapped = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
    runner.load(str(Path(checkpoint).resolve()), load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)

    obs = wrapped.get_observations()
    rewards = []
    done_counts = []
    command_metric_samples = []
    frame_paths = []
    if frame_dir is not None:
        frame_dir = Path(frame_dir)
        frame_dir.mkdir(parents=True, exist_ok=True)
    frame_stride = max(1, int(steps) // max(1, int(frame_count)))
    for step in range(max(1, int(steps))):
        with torch.no_grad():
            actions = policy(obs)
        obs, reward, dones, _ = wrapped.step(actions)
        rewards.append(float(reward.detach().mean().cpu()))
        done_counts.append(float(dones.detach().float().mean().cpu()))
        command_metrics = _current_command_metrics(wrapped.unwrapped)
        if command_metrics:
            command_metric_samples.append(command_metrics)
        if frame_dir is not None and len(frame_paths) < int(frame_count) and step % frame_stride == 0:
            frame = wrapped.unwrapped.render()
            if frame is not None:
                if isinstance(frame, np.ndarray) and frame.ndim == 4:
                    frame = frame[0]
                frame = np.asarray(frame)
                if frame.dtype != np.uint8:
                    frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
                frame_path = frame_dir / f"frame_{len(frame_paths):04d}.jpg"
                imageio.imwrite(frame_path, frame)
                frame_paths.append(str(frame_path))
    wrapped.close()
    payload = {
        "task_id": str(task_id),
        "steps": int(steps),
        "num_envs": int(env_cfg.scene.num_envs),
        "requested_num_envs": int(requested_num_envs),
        "device": device,
        "avg_step_reward": float(np.mean(rewards)) if rewards else 0.0,
        "return": float(np.sum(rewards)),
        "done_fraction": float(np.mean(done_counts)) if done_counts else 0.0,
        "frames": frame_paths,
        "sample_source": "mjlab_train_context",
    }
    command_metrics = _merge_command_metric_samples(command_metric_samples)
    if command_metrics:
        payload["command_metrics"] = command_metrics
    return payload


def _monitor_live_training(stop_event, *, cfg, out_dir, log_dir, run_id, tag, task_id, budget_mode, train_seconds, num_envs, steps_per_env, iterations, seed, gpu_id, recipe_json, started_at, rollout_script, motion_file, probe_interval_iterations, probe_num_envs, probe_steps, sample_rollout_frame_count, sample_trajectory_source):
    probed = set()
    out_dir = Path(out_dir)
    while not stop_event.is_set():
        try:
            payload = _write_partial_train_payload(
                out_dir=out_dir,
                log_dir=log_dir,
                run_id=run_id,
                tag=tag,
                task_id=task_id,
                status="running",
                budget_mode=budget_mode,
                train_seconds=train_seconds,
                num_envs=num_envs,
                steps_per_env=steps_per_env,
                iterations=iterations,
                started_at=started_at,
                recipe_json=recipe_json,
                event_status="monitoring",
            )
            if rollout_script and probe_interval_iterations > 0:
                checkpoints = sorted(Path(log_dir).rglob("model_*.pt"), key=lambda path: (_checkpoint_index(path), path.stat().st_mtime))
                for checkpoint in checkpoints:
                    index = _checkpoint_index(checkpoint)
                    if index <= 0 or index in probed or index % int(probe_interval_iterations) != 0:
                        continue
                    try:
                        sample_index = len(probed) + 1
                        frame_dir = None
                        if int(sample_rollout_frame_count) > 0:
                            frame_dir = out_dir / "trajectories" / f"sample_{sample_index:06d}"
                            frame_dir.mkdir(parents=True, exist_ok=True)
                        if str(sample_trajectory_source) == "train_context":
                            rollout = _run_train_context_probe_subprocess(
                                train_script=Path(__file__).resolve(),
                                task_id=task_id,
                                run_id=run_id,
                                num_envs=num_envs,
                                iterations=iterations,
                                seed=seed,
                                gpu_id=gpu_id,
                                recipe_json=recipe_json,
                                out_dir=out_dir,
                                steps_per_env=steps_per_env,
                                budget_mode=budget_mode,
                                train_seconds=train_seconds,
                                checkpoint=checkpoint,
                                checkpoint_index=index,
                                probe_num_envs=probe_num_envs,
                                probe_steps=probe_steps,
                                frame_dir=frame_dir,
                                frame_count=sample_rollout_frame_count,
                                timeout=180,
                            )
                        else:
                            rollout = _run_checkpoint_probe(
                                rollout_script=rollout_script,
                                task_id=task_id,
                                checkpoint=checkpoint,
                                out_dir=out_dir,
                                num_envs=probe_num_envs,
                                steps=probe_steps,
                                seed=900000 + index,
                                motion_file=motion_file,
                                python_executable=sys.executable,
                                timeout=180,
                                frame_dir=frame_dir,
                                frame_count=sample_rollout_frame_count,
                            )
                    except Exception as exc:
                        with (out_dir / "live" / "status.log").open("a", encoding="utf-8") as handle:
                            handle.write(f"st=probe_failed step={index} task={task_id} error={type(exc).__name__}\n")
                        probed.add(index)
                        continue
                    if rollout is None:
                        with (out_dir / "live" / "status.log").open("a", encoding="utf-8") as handle:
                            handle.write(f"st=probe_failed step={index} task={task_id} error=no_rollout\n")
                        probed.add(index)
                        continue
                    step = int(index * num_envs * steps_per_env)
                    manifest_path = None
                    if int(sample_rollout_frame_count) > 0:
                        manifest_path = _write_live_sample_manifest(
                            out_dir,
                            run_id=run_id,
                            tag=tag,
                            sample_index=len(probed) + 1,
                            checkpoint_index=index,
                            step=step,
                            frame_paths=rollout.get("frames", []),
                            source=rollout.get("sample_source") or "mjlab_live_probe",
                        )
                    record = {
                        "record_type": "policy_probe",
                        "episode": int(index * num_envs),
                        "return": _safe_float(rollout.get("return", 0.0)),
                        "length": _safe_float(rollout.get("steps", probe_steps)),
                        "success": False,
                        "step": step,
                        "elapsed_seconds": max(0.0, time.time() - started_at),
                        "probe_episodes": int(rollout.get("num_envs", probe_num_envs)),
                        "probe_seed_start": 900000 + index,
                        "deterministic": True,
                        "info_metrics": {
                            "policy_probe_return": _safe_float(rollout.get("return", 0.0)),
                            "policy_probe_length": _safe_float(rollout.get("steps", probe_steps)),
                            "policy_probe_episodes": float(rollout.get("num_envs", probe_num_envs)),
                            "avg_step_reward": _safe_float(rollout.get("avg_step_reward", 0.0)),
                            "done_fraction": _safe_float(rollout.get("done_fraction", 0.0)),
                        },
                    }
                    if manifest_path is not None:
                        record["trajectory_manifest_path"] = str(manifest_path)
                        record["trajectory_latest_frame_path"] = str(Path(out_dir) / "current_run_frame.jpg")
                    if "avg_mpkpe" in rollout:
                        record["info_metrics"]["mpkpe"] = _safe_float(rollout.get("avg_mpkpe", 0.0))
                        record["info_metrics"]["r_mpkpe"] = _safe_float(rollout.get("avg_r_mpkpe", 0.0))
                    with (out_dir / "policy_probe_records.jsonl").open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record) + "\n")
                    probed.add(index)
                    _write_partial_train_payload(
                        out_dir=out_dir,
                        log_dir=log_dir,
                        run_id=run_id,
                        tag=tag,
                        task_id=task_id,
                        status="running",
                        budget_mode=budget_mode,
                        train_seconds=train_seconds,
                        num_envs=num_envs,
                        steps_per_env=steps_per_env,
                        iterations=iterations,
                        started_at=started_at,
                        recipe_json=recipe_json,
                        event_status="probe_recorded",
                    )
        except Exception:
            error_dir = out_dir / "live"
            error_dir.mkdir(parents=True, exist_ok=True)
            with (error_dir / "monitor_errors.log").open("a", encoding="utf-8") as handle:
                handle.write(f"\n[{time.time()}]\n")
                handle.write(traceback.format_exc())
            with (error_dir / "status.log").open("a", encoding="utf-8") as handle:
                handle.write(f"st=monitor_error task={task_id} error=see_monitor_errors\n")
        stop_event.wait(5.0)
    _write_partial_train_payload(
        out_dir=out_dir,
        log_dir=log_dir,
        run_id=run_id,
        tag=tag,
        task_id=task_id,
        status="finished",
        budget_mode=budget_mode,
        train_seconds=train_seconds,
        num_envs=num_envs,
        steps_per_env=steps_per_env,
        iterations=iterations,
        started_at=started_at,
        recipe_json=recipe_json,
        stop_reason="mjlab_train_complete",
        event_status="finished",
    )


def _apply_mjlab_recipe(cfg, recipe, run_id):
    applied = []
    skipped = []
    if not _is_mapping(recipe):
        return {"applied": applied, "skipped": [{"field": "recipe", "reason": "not a mapping"}]}
    _apply_runner_cfg(cfg, recipe.get("runner"), run_id, applied, skipped)
    _apply_model_cfg(getattr(cfg.agent, "actor", None), recipe.get("actor"), applied, skipped, "actor")
    _apply_model_cfg(getattr(cfg.agent, "critic", None), recipe.get("critic"), applied, skipped, "critic")
    _apply_ppo_cfg(getattr(cfg.agent, "algorithm", None), recipe.get("ppo"), applied, skipped)
    _apply_environment_cfg(cfg, recipe.get("environment"), applied, skipped)
    _apply_command_cfg(cfg, "motion", recipe.get("motion_command"), applied, skipped, "motion_command")
    _apply_command_cfg(cfg, "twist", recipe.get("twist_command"), applied, skipped, "twist_command")
    _apply_rewards(cfg, recipe.get("reward_weights"), recipe.get("reward_params"), applied, skipped)
    _apply_term_collection(cfg, "events", recipe.get("event_overrides"), applied, skipped)
    _apply_term_collection(cfg, "terminations", recipe.get("termination_overrides"), applied, skipped)
    if hasattr(cfg.env, "curriculum"):
        _apply_term_collection(cfg, "curriculum", recipe.get("curriculum_overrides"), applied, skipped)
    elif hasattr(cfg.env, "curriculums"):
        _apply_term_collection(cfg, "curriculums", recipe.get("curriculum_overrides"), applied, skipped)
    _apply_terrain_cfg(cfg, recipe.get("terrain"), applied, skipped)
    return {"applied": applied, "skipped": skipped}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--motion-file", default=None)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--recipe-json", default=None)
    parser.add_argument("--resolved-json", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--steps-per-env", type=int, default=24)
    parser.add_argument("--budget-mode", default="episodes")
    parser.add_argument("--train-seconds", type=float, default=None)
    parser.add_argument("--rollout-script", default=None)
    parser.add_argument("--probe-interval-iterations", type=int, default=100)
    parser.add_argument("--probe-num-envs", type=int, default=16)
    parser.add_argument("--probe-steps", type=int, default=120)
    parser.add_argument("--probe-checkpoint", default=None)
    parser.add_argument("--probe-checkpoint-index", type=int, default=0)
    parser.add_argument("--probe-out-json", default=None)
    parser.add_argument("--probe-seed", type=int, default=None)
    parser.add_argument("--probe-frame-dir", default=None)
    parser.add_argument("--probe-frame-count", type=int, default=0)
    parser.add_argument("--sample-rollout-frame-count", type=int, default=24)
    parser.add_argument("--sample-trajectory-source", choices=["fallback", "train_context"], default="fallback")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ["MUJOCO_GL"] = "glfw" if os.name == "nt" else "egl"
    cfg = TrainConfig.from_task(args.task_id)
    recipe = _load_recipe(args.recipe_json)
    resolved = _apply_mjlab_recipe(cfg, recipe, args.run_id)
    cfg.env.scene.num_envs = int(args.num_envs)
    cfg.agent.max_iterations = int(args.iterations)
    if int(args.probe_interval_iterations) > 0:
        current_save_interval = int(getattr(cfg.agent, "save_interval", 0) or 0)
        if current_save_interval <= 0 or current_save_interval > int(args.probe_interval_iterations):
            cfg.agent.save_interval = int(args.probe_interval_iterations)
            resolved["applied"].append("runner.save_interval<=probe_interval_iterations")
    if not getattr(cfg.agent, "run_name", None):
        cfg.agent.run_name = args.run_id
    cfg.agent.seed = int(args.seed)
    cfg = replace(cfg, motion_file=args.motion_file, gpu_ids=[int(args.gpu_id)])
    if args.resolved_json:
        Path(args.resolved_json).write_text(
            json.dumps({"recipe": recipe, "applied": resolved["applied"], "skipped": resolved["skipped"]}, indent=2),
            encoding="utf-8",
        )
    if args.probe_checkpoint:
        if not args.probe_out_json:
            raise ValueError("--probe-out-json is required with --probe-checkpoint")
        rollout = _run_train_context_sample(
            cfg=cfg,
            task_id=args.task_id,
            checkpoint=Path(args.probe_checkpoint),
            num_envs=int(args.probe_num_envs),
            steps=int(args.probe_steps),
            seed=int(args.probe_seed if args.probe_seed is not None else 900000 + int(args.probe_checkpoint_index)),
            frame_dir=Path(args.probe_frame_dir) if args.probe_frame_dir else None,
            frame_count=int(args.probe_frame_count),
            checkpoint_index=int(args.probe_checkpoint_index),
            steps_per_env=int(args.steps_per_env),
        )
        Path(args.probe_out_json).write_text(json.dumps(rollout, indent=2), encoding="utf-8")
        return

    log_root_path = Path("logs") / "rsl_rl" / cfg.agent.experiment_name
    log_dir = log_root_path / (datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{args.run_id}")
    started_at = time.time()
    stop_event = threading.Event()
    monitor = None
    if args.out_dir:
        monitor = threading.Thread(
            target=_monitor_live_training,
            kwargs={
                "stop_event": stop_event,
                "cfg": cfg,
                "out_dir": Path(args.out_dir),
                "log_dir": log_dir,
                "run_id": args.run_id,
                "tag": args.run_id,
                "task_id": args.task_id,
                "budget_mode": args.budget_mode,
                "train_seconds": args.train_seconds,
                "num_envs": int(args.num_envs),
                "steps_per_env": int(args.steps_per_env),
                "iterations": int(args.iterations),
                "seed": int(args.seed),
                "gpu_id": int(args.gpu_id),
                "recipe_json": Path(args.recipe_json) if args.recipe_json else None,
                "started_at": started_at,
                "rollout_script": Path(args.rollout_script) if args.rollout_script else None,
                "motion_file": args.motion_file,
                "probe_interval_iterations": int(args.probe_interval_iterations),
                "probe_num_envs": int(args.probe_num_envs),
                "probe_steps": int(args.probe_steps),
                "sample_rollout_frame_count": int(args.sample_rollout_frame_count),
                "sample_trajectory_source": args.sample_trajectory_source,
            },
            daemon=True,
        )
        monitor.start()
    try:
        run_train(args.task_id, cfg, log_dir)
    finally:
        if monitor is not None:
            stop_event.set()
            monitor.join(timeout=30.0)
    print(f"[AUTORESEARCH] log_dir={log_dir}")


if __name__ == "__main__":
    main()
'''


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
        str(_recipe_section(recipe, "runner").get("sample_trajectory_source") or "fallback"),
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
    script_path.write_text(MJLAB_ROLLOUT_SCRIPT, encoding="utf-8")
    return script_path


def _write_train_script(out_dir: Path) -> Path:
    script_path = out_dir / "mjlab_train_bridge.py"
    script_path.write_text(MJLAB_TRAIN_SCRIPT, encoding="utf-8")
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
        argv.extend(["--frame-dir", frame_dir.resolve(), "--frame-count", "24", "--no-terminations"])
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


def _line(draw: ImageDraw.ImageDraw, points: tuple[tuple[int, int], ...], fill: tuple[int, int, int], width: int = 5) -> None:
    for left, right in zip(points, points[1:]):
        draw.line((*left, *right), fill=fill, width=width)


def _joint(draw: ImageDraw.ImageDraw, xy: tuple[int, int], fill: tuple[int, int, int], radius: int = 6) -> None:
    x, y = xy
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill)


def _draw_g1_motion_frame(draw: ImageDraw.ImageDraw, frame_index: int) -> None:
    phase = frame_index / 5.0
    hip = (210, 132 - int(5 * np.sin(phase * np.pi)))
    head = (210, 58)
    neck = (210, 78)
    left_shoulder = (184, 88)
    right_shoulder = (236, 88)
    left_hand = (150 - int(16 * phase), 116 - int(28 * phase))
    right_hand = (270 + int(8 * phase), 104 + int(10 * phase))
    left_knee = (184, 166)
    right_knee = (242 + int(18 * phase), 144 - int(56 * phase))
    left_foot = (170, 214)
    right_foot = (300 + int(22 * phase), 98 - int(36 * phase))
    robot = (84, 210, 255)
    accent = (255, 180, 84)

    _line(draw, (head, neck, hip), robot, 6)
    _line(draw, (left_shoulder, neck, right_shoulder), robot, 5)
    _line(draw, (left_shoulder, left_hand), robot, 5)
    _line(draw, (right_shoulder, right_hand), robot, 5)
    _line(draw, (hip, left_knee, left_foot), robot, 6)
    _line(draw, (hip, right_knee, right_foot), accent, 7)
    for point in (head, neck, hip, left_hand, right_hand, left_knee, right_knee, left_foot, right_foot):
        _joint(draw, point, accent if point == right_foot else robot, 5)
    draw.ellipse((196, 42, 224, 70), outline=(232, 246, 255), width=3)
    draw.arc((278, 50, 380, 150), 210, 330, fill=(255, 180, 84), width=4)
    draw.text((24, 20), "G1 motion mirroring: side kick", fill=(245, 245, 245))
    draw.text((24, 38), f"sampled rollout frame {frame_index + 1}/6", fill=(180, 210, 230))


def _draw_go2_rough_frame(draw: ImageDraw.ImageDraw, frame_index: int) -> None:
    phase = frame_index / 5.0
    body_x = 120 + int(28 * frame_index)
    body_y = 112 - int(6 * np.sin(phase * np.pi * 2))
    robot = (140, 255, 152)
    accent = (255, 180, 84)
    terrain = [(0, 218), (45, 208), (90, 224), (135, 196), (180, 212), (225, 188), (270, 206), (315, 180), (360, 202), (420, 190)]
    _line(draw, tuple(terrain), (86, 116, 96), 5)

    body = (body_x, body_y, body_x + 92, body_y + 42)
    draw.rounded_rectangle(body, radius=16, outline=robot, width=5)
    draw.ellipse((body_x + 72, body_y - 4, body_x + 112, body_y + 30), outline=(232, 246, 255), width=4)
    legs = [
        ((body_x + 16, body_y + 36), (body_x + 2, body_y + 82), (body_x - 10 + int(10 * phase), body_y + 106)),
        ((body_x + 36, body_y + 38), (body_x + 50, body_y + 78), (body_x + 62 - int(8 * phase), body_y + 100)),
        ((body_x + 60, body_y + 38), (body_x + 46, body_y + 80), (body_x + 36 + int(14 * phase), body_y + 112)),
        ((body_x + 82, body_y + 36), (body_x + 98, body_y + 74), (body_x + 116 - int(10 * phase), body_y + 94)),
    ]
    for index, leg in enumerate(legs):
        _line(draw, leg, accent if index in (0, 3) else robot, 5)
        _joint(draw, leg[-1], accent if index in (0, 3) else robot, 5)
    draw.text((24, 20), "Go2 rough-terrain locomotion", fill=(245, 245, 245))
    draw.text((24, 38), f"sampled rollout frame {frame_index + 1}/6", fill=(180, 210, 230))
    draw.text((276, 30), "velocity command", fill=(180, 210, 230))
    draw.line((268, 58, 360, 58), fill=(84, 210, 255), width=4)
    draw.polygon(((360, 58), (346, 50), (346, 66)), fill=(84, 210, 255))


def _normalize_dashboard_frame(path: Path) -> Path:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != DASHBOARD_FRAME_SIZE:
            image = image.resize(DASHBOARD_FRAME_SIZE, Image.Resampling.BILINEAR)
        image.save(path, format="JPEG", quality=90)
    return path


def _write_frame(path: Path, task_family: str, frame_index: int = 0) -> None:
    image = Image.new("RGB", (420, 260), (13, 18, 26))
    draw = ImageDraw.Draw(image)
    for y in range(0, 260, 26):
        draw.line((0, y, 420, y), fill=(24, 34, 46), width=1)
    draw.rectangle((8, 8, 412, 252), outline=(62, 82, 105), width=2)
    if task_family == "g1_motion_mirror":
        _draw_g1_motion_frame(draw, frame_index)
    else:
        _draw_go2_rough_frame(draw, frame_index)
    image = image.resize(DASHBOARD_FRAME_SIZE, Image.Resampling.BILINEAR)
    image.save(path, format="JPEG", quality=90)


def _write_sampled_trajectory(out_dir: Path, bundle: dict[str, Any]) -> Path:
    trajectory_dir = out_dir / "trajectories" / "sample_000001"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    task_family = str(bundle.get("task_family", "unitree"))
    for idx in range(6):
        frame_path = trajectory_dir / f"frame_{idx:04d}.jpg"
        _write_frame(frame_path, task_family, idx)
        frames.append(str(frame_path))
    manifest = {
        "run_id": bundle["run_id"],
        "tag": bundle["tag"],
        "sample_index": 1,
        "episode": 1,
        "status": "completed",
        "updated_at": time.time(),
        "frame_count": len(frames),
        "frames": frames,
        "latest_frame_path": frames[-1],
        "playback_fps": DEFAULT_TRAJECTORY_PLAYBACK_FPS,
        "frame_stride": 1,
        "sample_rate": 1.0,
        "width": DASHBOARD_FRAME_WIDTH,
        "height": DASHBOARD_FRAME_HEIGHT,
    }
    manifest_path = trajectory_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path


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
    task_family = str(bundle.get("task_family", "unitree"))
    frame_path = out_dir / "current_run_frame.jpg"
    _write_frame(frame_path, task_family)
    manifest_path = _write_sampled_trajectory(out_dir, bundle)
    media = {
        "media_available": True,
        "live_frame_path": str(frame_path),
        "trajectory_manifest_path": str(manifest_path),
        "trajectory_latest_frame_path": str(manifest_path.parent / "frame_0005.jpg"),
        "visual": {
            "mode": "sampled_trajectory",
            "live_frame_path": str(frame_path),
            "trajectory_manifest_path": str(manifest_path),
            "trajectory_latest_frame_path": str(manifest_path.parent / "frame_0005.jpg"),
            "sampled_status": "completed",
            "latest_sample_index": 1,
        },
    }
    _write_json(out_dir / "media_result.json", media)
    if bundle.get("session_dir"):
        live_dir = Path(bundle["session_dir"]) / "live"
        live_dir.mkdir(parents=True, exist_ok=True)
        live_frame = live_dir / "current_run_frame.jpg"
        _write_frame(live_frame, task_family)
        live_trajectory_dir = live_dir / "trajectories" / bundle["run_id"] / "episode_000001"
        live_trajectory_dir.mkdir(parents=True, exist_ok=True)
        live_frames = []
        for idx in range(6):
            live_sample = live_trajectory_dir / f"frame_{idx:04d}.jpg"
            _write_frame(live_sample, task_family, idx)
            live_frames.append(_dashboard_path(live_sample))
        live_manifest = live_trajectory_dir / "manifest.json"
        _write_json(
            live_manifest,
            {
                "run_id": bundle["run_id"],
                "tag": bundle["tag"],
                "sample_index": 1,
                "episode": 1,
                "status": "completed",
                "updated_at": time.time(),
                "frame_count": len(live_frames),
                "frames": live_frames,
                "latest_frame_path": live_frames[-1],
                "playback_fps": DEFAULT_TRAJECTORY_PLAYBACK_FPS,
                "frame_stride": 1,
                "sample_rate": 1.0,
                "width": DASHBOARD_FRAME_WIDTH,
                "height": DASHBOARD_FRAME_HEIGHT,
            },
        )
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
                    "mode": "sampled_trajectory",
                    "live_frame_path": _dashboard_path(live_frame),
                    "trajectory_manifest_path": _dashboard_path(live_manifest),
                    "trajectory_latest_frame_path": live_frames[-1],
                    "sampled_status": "completed",
                    "latest_sample_index": 1,
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
