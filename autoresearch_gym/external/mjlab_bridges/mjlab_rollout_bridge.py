from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

try:
    from PIL import Image, ImageFilter
except Exception:
    Image = None
    ImageFilter = None

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


def _set_if_present(target, name: str, value) -> bool:
    if target is None or not hasattr(target, name):
        return False
    try:
        setattr(target, name, value)
        return True
    except Exception:
        return False


def _configure_render_resolution(env_cfg, width: int, height: int) -> None:
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


def _write_frame(path: Path, frame, width: int, height: int) -> None:
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
    parser.add_argument("--render-width", type=int, default=720)
    parser.add_argument("--render-height", type=int, default=480)
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
    if args.frame_dir:
        _configure_render_resolution(env_cfg, int(args.render_width), int(args.render_height))
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
                frame_path = frame_dir / f"frame_{len(frame_paths):04d}.jpg"
                _write_frame(frame_path, frame, int(args.render_width), int(args.render_height))
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
