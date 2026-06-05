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

try:
    from PIL import Image, ImageFilter  # noqa: E402
except Exception:
    Image = None
    ImageFilter = None

DASHBOARD_FRAME_WIDTH = 720
DASHBOARD_FRAME_HEIGHT = 480
DASHBOARD_FRAME_SIZE = (DASHBOARD_FRAME_WIDTH, DASHBOARD_FRAME_HEIGHT)
DEFAULT_TRAJECTORY_PLAYBACK_FPS = 20.0


def _normalize_dashboard_frame(path: Path) -> Path:
    if Image is None:
        return path
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != DASHBOARD_FRAME_SIZE:
            image = image.resize(DASHBOARD_FRAME_SIZE, Image.Resampling.LANCZOS)
            if ImageFilter is not None:
                image = image.filter(ImageFilter.UnsharpMask(radius=1.0, percent=90, threshold=3))
        image.save(path, format="JPEG", quality=95, subsampling=0)
    return path

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


def _set_if_present(target, name, value):
    if target is None or not hasattr(target, name):
        return False
    try:
        setattr(target, name, value)
        return True
    except Exception:
        return False


def _configure_render_resolution(env_cfg, width, height):
    for section_name in ("viewer", "render", "renderer", "sim"):
        section = getattr(env_cfg, section_name, None)
        if section is None:
            continue
        _set_if_present(section, "width", int(width))
        _set_if_present(section, "height", int(height))
        _set_if_present(section, "render_width", int(width))
        _set_if_present(section, "render_height", int(height))
        _set_if_present(section, "resolution", (int(width), int(height)))
        _set_if_present(section, "size", (int(width), int(height)))


def _write_frame(path, frame, width, height):
    frame = np.asarray(frame)
    if frame.dtype != np.uint8:
        frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
    if Image is None:
        imageio.imwrite(path, frame)
        return
    image = Image.fromarray(frame).convert("RGB")
    if image.size != (int(width), int(height)):
        image = image.resize((int(width), int(height)), Image.Resampling.LANCZOS)
        if ImageFilter is not None:
            image = image.filter(ImageFilter.UnsharpMask(radius=1.0, percent=90, threshold=3))
    image.save(path, format="JPEG", quality=95, subsampling=0)


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
        argv.extend(
            [
                "--frame-dir",
                str(frame_dir),
                "--frame-count",
                str(int(frame_count)),
                "--render-width",
                str(DASHBOARD_FRAME_WIDTH),
                "--render-height",
                str(DASHBOARD_FRAME_HEIGHT),
                "--no-terminations",
            ]
        )
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
        argv.extend(
            [
                "--probe-frame-dir",
                str(frame_dir),
                "--probe-render-width",
                str(DASHBOARD_FRAME_WIDTH),
                "--probe-render-height",
                str(DASHBOARD_FRAME_HEIGHT),
            ]
        )
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
    render_width,
    render_height,
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
    if frame_dir is not None:
        _configure_render_resolution(env_cfg, int(render_width), int(render_height))

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
                frame_path = frame_dir / f"frame_{len(frame_paths):04d}.jpg"
                _write_frame(frame_path, frame, int(render_width), int(render_height))
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
                        if str(sample_trajectory_source) == "candidate_provided":
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
    parser.add_argument("--probe-render-width", type=int, default=720)
    parser.add_argument("--probe-render-height", type=int, default=480)
    parser.add_argument("--sample-rollout-frame-count", type=int, default=24)
    parser.add_argument("--sample-trajectory-source", choices=["runner_eval", "candidate_provided"], default="runner_eval")
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
            render_width=int(args.probe_render_width),
            render_height=int(args.probe_render_height),
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
