from __future__ import annotations

import time
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from autoresearch_gym.runner.curves import elapsed_seconds_since, make_train_episode_record, scalar_info_metrics
from autoresearch_gym.tasks.so101_reach_mujoco_v0.seed_trainable import (
    ALGORITHM,
    CONTROL_TYPE,
    REPLAY_SIZE,
    REWARD_RECIPE,
    Agent,
    ReplayBuffer,
    RewardRecipeWrapper,
    load_agent_checkpoint,
    save_agent_checkpoint,
)


EXP_NAME = "so101_mujoco_reach_vectorized_sac_seed"
NUM_ENVS = 8
BATCH_SIZE = 512
LEARNING_STARTS = 1_024
UPDATE_AFTER = 1_024
GRADIENT_UPDATES_PER_VECTOR_STEP = 4
LIVE_CALLBACK_EVERY_STEPS = 512
RENDER_SIDECAR_ENABLED = True
VECTOR_ENV_MODE = "sync"
MAX_ENV_STEPS_SAFETY_CAP = 2_000_000


def get_candidate() -> dict[str, Any]:
    return {
        "description": (
            "SO-101 MuJoCo reach vectorized SAC baseline. Uses headless SyncVectorEnv "
            "workers for collection, a single live sidecar env for dashboard frames "
            "and probes, and larger replay/update batches for fixed wall-clock "
            "Windows runs."
        ),
        "recipe": {
            "algorithm": ALGORITHM,
            "reward_recipe": REWARD_RECIPE,
            "control": "normalized_position_targets",
            "vector_envs": NUM_ENVS,
            "batch_size": BATCH_SIZE,
            "gradient_updates_per_vector_step": GRADIENT_UPDATES_PER_VECTOR_STEP,
        },
    }


def _make_headless_env(benchmark: Any, seed_offset: int):
    def thunk():
        env_kwargs = dict(getattr(benchmark, "env_kwargs", {}) or {})
        env_kwargs["render_mode"] = None
        if int(getattr(benchmark, "max_steps", 0) or 0) > 0:
            env_kwargs.setdefault("max_episode_steps", int(benchmark.max_steps))
        env = gym.make(benchmark.env_id, **env_kwargs)
        wrapped = RewardRecipeWrapper(env, REWARD_RECIPE)
        wrapped.action_space.seed(int(benchmark.train_seed) + seed_offset)
        return wrapped

    return thunk


def _info_for_env(infos: Any, index: int, num_envs: int) -> dict[str, Any]:
    if not isinstance(infos, dict):
        return {}
    final_infos = infos.get("final_info")
    if final_infos is not None and index < len(final_infos) and final_infos[index] is not None:
        return dict(final_infos[index])
    env_info: dict[str, Any] = {}
    for key, value in infos.items():
        if key.startswith("_") or key == "final_info":
            continue
        try:
            if isinstance(value, np.ndarray) and len(value) == num_envs:
                item = value[index]
                env_info[key] = item.item() if hasattr(item, "item") else item
            elif isinstance(value, (list, tuple)) and len(value) == num_envs:
                env_info[key] = value[index]
        except TypeError:
            continue
    return env_info


def train_agent(
    benchmark: Any,
    env_factory: Any,
    candidate: Any,
    device: torch.device,
    init_checkpoint: str | Any | None = None,
    live_callback: Any | None = None,
) -> tuple[Agent, dict[str, Any]]:
    del candidate
    np.random.seed(int(benchmark.train_seed))
    torch.manual_seed(int(benchmark.train_seed))

    probe_env = env_factory(control_type=CONTROL_TYPE, reward_recipe=REWARD_RECIPE)
    render_env = probe_env if RENDER_SIDECAR_ENABLED else None
    render_obs = None
    if render_env is not None:
        render_obs, _ = render_env.reset(seed=int(benchmark.train_seed) + 900_000)
        render_env.action_space.seed(int(benchmark.train_seed) + 900_000)
    agent = Agent(probe_env, device=device)
    agent.batch_size = BATCH_SIZE
    resumed_from = load_agent_checkpoint(agent, init_checkpoint) if init_checkpoint is not None else None
    replay = ReplayBuffer(agent.obs_dim, agent.act_dim, REPLAY_SIZE)

    env_fns = [_make_headless_env(benchmark, idx) for idx in range(NUM_ENVS)]
    envs = gym.vector.SyncVectorEnv(env_fns)
    obs, _ = envs.reset(seed=[int(benchmark.train_seed) + idx for idx in range(NUM_ENVS)])

    global_step = 0
    gradient_updates = 0
    started_at = time.time()
    budget_seconds = getattr(benchmark, "train_seconds", None)
    deadline = started_at + float(budget_seconds) if budget_seconds is not None else None
    episode_records: list[dict[str, Any]] = []
    last_metrics: dict[str, float] | None = None
    active_returns = np.zeros(NUM_ENVS, dtype=np.float64)
    active_lengths = np.zeros(NUM_ENVS, dtype=np.int64)
    live_step = 0
    render_episode_return = 0.0
    render_episode_length = 0

    def should_continue() -> bool:
        if len(episode_records) >= int(benchmark.train_episodes):
            return False
        if global_step >= MAX_ENV_STEPS_SAFETY_CAP:
            return False
        if deadline is not None and time.time() >= deadline:
            return False
        return True

    def advance_render_env() -> tuple[float, int]:
        nonlocal render_obs, render_episode_return, render_episode_length
        if render_env is None or render_obs is None:
            return render_episode_return, render_episode_length
        action = render_env.action_space.sample() if global_step < LEARNING_STARTS else agent.act(render_obs, deterministic=True)
        render_obs, reward, terminated, truncated, _ = render_env.step(action)
        render_episode_return += float(reward)
        render_episode_length += 1
        if terminated or truncated:
            finished_return = render_episode_return
            finished_length = render_episode_length
            render_obs, _ = render_env.reset(seed=int(benchmark.train_seed) + 900_000 + len(episode_records))
            render_episode_return = 0.0
            render_episode_length = 0
            return finished_return, finished_length
        return render_episode_return, render_episode_length

    try:
        if live_callback is not None:
            live_callback(
                status="running",
                episode_records=episode_records,
                total_steps=global_step,
                last_metrics=last_metrics,
                env=render_env,
                current_episode=1,
                episode_return=render_episode_return,
                episode_length=render_episode_length,
                agent=agent,
                elapsed_seconds=elapsed_seconds_since(started_at),
            )

        while should_continue():
            if global_step < LEARNING_STARTS:
                action = np.stack([envs.single_action_space.sample() for _ in range(NUM_ENVS)]).astype(np.float32)
            else:
                action = agent.act_batch(obs, deterministic=False)

            next_obs, rewards, terminations, truncations, infos = envs.step(action)
            dones = np.logical_or(terminations, truncations)
            for env_index in range(NUM_ENVS):
                replay.add(
                    obs[env_index],
                    action[env_index],
                    float(rewards[env_index]),
                    next_obs[env_index],
                    bool(terminations[env_index]),
                )
            obs = next_obs
            active_returns += rewards.astype(np.float64)
            active_lengths += 1
            global_step += NUM_ENVS
            sidecar_return, sidecar_length = advance_render_env()

            for env_index in np.flatnonzero(dones):
                info = _info_for_env(infos, int(env_index), NUM_ENVS)
                record = make_train_episode_record(
                    episode=len(episode_records) + 1,
                    step=global_step,
                    return_value=float(active_returns[env_index]),
                    length=int(active_lengths[env_index]),
                    success=bool(info.get("is_success", False)),
                    elapsed_seconds=elapsed_seconds_since(started_at),
                    info_metrics=scalar_info_metrics(info),
                    env_index=int(env_index),
                )
                episode_records.append(record)
                active_returns[env_index] = 0.0
                active_lengths[env_index] = 0
                if len(episode_records) >= int(benchmark.train_episodes):
                    break

            if global_step >= UPDATE_AFTER and replay.size >= BATCH_SIZE:
                for _ in range(GRADIENT_UPDATES_PER_VECTOR_STEP):
                    if deadline is not None and time.time() >= deadline:
                        break
                    last_metrics = agent.update(replay)
                    gradient_updates += 1
                if last_metrics is not None:
                    last_metrics = {
                        **last_metrics,
                        "gradient_updates": float(gradient_updates),
                        "num_envs": float(NUM_ENVS),
                    }

            if live_callback is not None and (
                global_step == NUM_ENVS or global_step - live_step >= LIVE_CALLBACK_EVERY_STEPS
            ):
                live_step = global_step
                live_callback(
                    status="running",
                    episode_records=episode_records,
                    total_steps=global_step,
                    last_metrics=last_metrics,
                    env=render_env,
                    current_episode=len(episode_records) + 1,
                    episode_return=sidecar_return,
                    episode_length=sidecar_length,
                    agent=agent,
                    elapsed_seconds=elapsed_seconds_since(started_at),
                )
    finally:
        envs.close()
        if render_env is not None:
            render_env.close()

    wall_clock = time.time() - started_at
    stop_reason = (
        "episode_cap_reached"
        if len(episode_records) >= int(benchmark.train_episodes)
        else "max_env_steps_safety_cap_reached"
        if global_step >= MAX_ENV_STEPS_SAFETY_CAP
        else "time_budget_exhausted"
        if deadline is not None and time.time() >= deadline
        else "loop_exited"
    )
    returns = [float(record["return"]) for record in episode_records]
    successes = [1.0 if record.get("success") else 0.0 for record in episode_records]
    return agent, {
        "algorithm": ALGORITHM,
        "episodes": int(benchmark.train_episodes),
        "episodes_completed": len(episode_records),
        "time_budget_seconds": float(budget_seconds) if budget_seconds is not None else None,
        "stop_reason": stop_reason,
        "total_steps": global_step,
        "env_steps": global_step,
        "completed_episodes": len(episode_records),
        "episode_batches": len(episode_records),
        "avg_return": float(np.mean(returns)) if returns else 0.0,
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "avg_length": float(np.mean([record["length"] for record in episode_records])) if episode_records else 0.0,
        "last_metrics": last_metrics,
        "gradient_updates": gradient_updates,
        "episode_records": episode_records,
        "wall_clock_seconds": wall_clock,
        "vector_envs": NUM_ENVS,
        "vector_backend": VECTOR_ENV_MODE,
        "visual_sampling": "single rgb_array sidecar env stepped by current policy",
        "init_checkpoint": str(init_checkpoint) if init_checkpoint is not None else None,
        "resumed_from": resumed_from,
    }
