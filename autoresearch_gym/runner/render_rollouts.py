from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw


def _require_mujoco() -> Any:
    try:
        import mujoco
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("MuJoCo rendering requires the `mujoco` extra.") from exc
    return mujoco


def _require_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Checkpoint rollout rendering requires torch. Install a task extra first.") from exc
    return torch


def load_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("render_trainable_snapshot", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import trainable snapshot: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def annotate(frame: np.ndarray, text: str) -> Image.Image:
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    draw = ImageDraw.Draw(image)
    pad = 6
    bbox = draw.textbbox((0, 0), text)
    width = bbox[2] - bbox[0] + pad * 2
    height = bbox[3] - bbox[1] + pad * 2
    draw.rectangle((0, 0, width, height), fill=(0, 0, 0))
    draw.text((pad, pad), text, fill=(255, 255, 255))
    return image


def make_env(env_id: str, max_episode_steps: int | None, trainable: Any) -> gym.Env[Any, Any]:
    kwargs: dict[str, Any] = {"render_mode": None}
    if max_episode_steps is not None:
        kwargs["max_episode_steps"] = max_episode_steps
    env = gym.make(env_id, **kwargs)
    return trainable.RewardRecipeWrapper(env, getattr(trainable, "REWARD_RECIPE", None))


def render_frame(env: gym.Env[Any, Any], renderer: Any, camera: str | int | None) -> np.ndarray:
    render_env = getattr(env, "unwrapped", env)
    if camera is None:
        renderer.update_scene(render_env.data)
    else:
        renderer.update_scene(render_env.data, camera=camera)
    return renderer.render()


def run_rollout(
    env: gym.Env[Any, Any],
    agent: Any,
    renderer: Any,
    camera: str | int | None,
    seed: int,
    max_steps: int,
    stride: int,
    frames_dir: Path,
    rollout_index: int,
) -> tuple[list[Image.Image], dict[str, Any]]:
    obs, _ = env.reset(seed=seed)
    env.action_space.seed(seed)
    frames: list[Image.Image] = []
    total_return = 0.0
    final_info: dict[str, Any] = {}
    terminated = False
    truncated = False
    step = 0
    for step in range(max_steps):
        action = agent.act(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_return += float(reward)
        final_info = dict(info)
        if step % stride == 0:
            frame = render_frame(env, renderer, camera)
            image = annotate(frame, f"seed={seed} step={step + 1} return={total_return:.1f}")
            if len(frames) < 10:
                image.save(frames_dir / f"rollout{rollout_index:02d}_frame{len(frames):03d}.png")
            frames.append(image)
        if terminated or truncated:
            break
    if not frames:
        frame = render_frame(env, renderer, camera)
        image = annotate(frame, f"seed={seed} step={step + 1} return={total_return:.1f}")
        image.save(frames_dir / f"rollout{rollout_index:02d}_frame000.png")
        frames.append(image)
    summary = {
        "seed": seed,
        "return": total_return,
        "length": step + 1,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "info": {
            key: float(value)
            for key, value in final_info.items()
            if isinstance(value, (int, float, np.integer, np.floating))
        },
    }
    return frames, summary


def write_gif(frames: list[Image.Image], out_path: Path, fps: int) -> None:
    if not frames:
        raise RuntimeError("No frames were rendered")
    duration_ms = int(1000 / fps)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render deterministic MuJoCo rollout GIFs from a saved candidate checkpoint.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[3000, 3001, 3002])
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--camera", default="track")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    mujoco = _require_mujoco()
    torch = _require_torch()

    run_dir = args.run_dir.resolve()
    trainable = load_module(run_dir / "trainable_snapshot.py")
    benchmark = json.loads((run_dir / "benchmark_snapshot.json").read_text(encoding="utf-8"))
    env_id = benchmark.get("env_id", "Hopper-v5")
    max_episode_steps = benchmark.get("max_steps")

    env = make_env(env_id, max_episode_steps, trainable)
    agent = trainable.Agent(env, torch.device(args.device))
    trainable.load_agent_checkpoint(agent, run_dir / "agent_checkpoint.pt")
    render_env = getattr(env, "unwrapped", env)
    renderer = mujoco.Renderer(render_env.model, height=args.height, width=args.width)

    all_frames: list[Image.Image] = []
    summaries: list[dict[str, Any]] = []
    frames_dir = args.out.with_suffix("").with_name(args.out.stem + "_frames")
    frames_dir.mkdir(parents=True, exist_ok=True)
    try:
        for rollout_index, seed in enumerate(args.seeds, start=1):
            frames, summary = run_rollout(
                env,
                agent,
                renderer,
                args.camera if args.camera else None,
                seed,
                args.max_steps,
                args.stride,
                frames_dir,
                rollout_index,
            )
            summaries.append(summary)
            all_frames.extend(frames)
    finally:
        renderer.close()
        env.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_gif(all_frames, args.out, args.fps)
    summary_path = args.out.with_suffix(".json")
    summary_path.write_text(json.dumps({"rollouts": summaries}, indent=2), encoding="utf-8")
    print(json.dumps({"gif": str(args.out), "summary": str(summary_path), "frames_dir": str(frames_dir)}, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
