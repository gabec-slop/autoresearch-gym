from __future__ import annotations

import random
import time
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from autoresearch_gym.runner.curves import elapsed_seconds_since, make_train_episode_record, scalar_info_metrics
from autoresearch_gym.tasks.bat_to_goal_v0.seed_trainable import (
    CandidateSpec,
    ReplayBuffer,
    RewardRecipeWrapper,
    build_agent,
    get_candidate as get_base_candidate,
    load_agent_checkpoint,
    save_agent_checkpoint,
)


# Bat-to-goal is PyBullet-backed on this Mac, so this is simulator batching rather
# than MJX-style one-kernel physics. Keep these constants obvious for candidate
# mutation under a fixed wall-clock benchmark.
NUM_ENVS = 64
VECTOR_ENV_MODE = "async"
ASYNC_CONTEXT = "fork"
GRADIENT_UPDATES_PER_VECTOR_STEP = 1
LIVE_CALLBACK_EVERY_STEPS = 80
MAX_ENV_STEPS_SAFETY_CAP = 1_000_000


def get_candidate() -> CandidateSpec:
    base = get_base_candidate()
    hyperparameters = dict(base.hyperparameters)
    hyperparameters.update(
        {
            "batch_size": 256,
            "start_steps": 256,
            "update_after": 128,
            "gradient_steps": GRADIENT_UPDATES_PER_VECTOR_STEP,
        }
    )
    return CandidateSpec(
        description=(
            "vectorized SAC seed: 64 PyBullet bat-to-goal training envs with "
            "async-fork collection, one rgb_array sidecar env for live frames, "
            "and fixed single-env evaluation."
        ),
        control_type=base.control_type,
        algorithm=base.algorithm,
        reward_recipe=base.reward_recipe,
        hidden_dims=base.hidden_dims,
        curriculum=None,
        hyperparameters=hyperparameters,
    )


def _action_batch(agent: Any, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
    obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32), dtype=torch.float32, device=agent.device)
    with torch.no_grad():
        if hasattr(agent.actor, "act"):
            action_t = agent.actor.act(obs_t, deterministic=deterministic)
        else:
            action_t = agent.actor(obs_t)
    return action_t.cpu().numpy()


def _make_training_env(benchmark: Any, candidate: CandidateSpec, seed_offset: int):
    def thunk():
        env_kwargs = dict(getattr(benchmark, "env_kwargs", {}))
        horizon = int(getattr(benchmark, "max_steps"))
        env_kwargs["max_steps"] = horizon
        env_kwargs["max_episode_steps"] = horizon
        if candidate.control_type is not None:
            env_kwargs.setdefault("control_type", candidate.control_type)
        env = gym.make(benchmark.env_id, **env_kwargs)
        wrapped = RewardRecipeWrapper(env, candidate.reward_recipe)
        wrapped.action_space.seed(int(benchmark.train_seed) + seed_offset)
        return wrapped

    return thunk


def _info_for_env(infos: Any, index: int) -> dict[str, Any]:
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
            if isinstance(value, np.ndarray) and len(value) > index:
                item = value[index]
                env_info[key] = item.item() if hasattr(item, "item") else item
            elif isinstance(value, (list, tuple)) and len(value) > index:
                env_info[key] = value[index]
        except TypeError:
            continue
    return env_info


def _build_vector_envs(benchmark: Any, candidate: CandidateSpec):
    env_fns = [_make_training_env(benchmark, candidate, idx) for idx in range(NUM_ENVS)]
    vector_backend = VECTOR_ENV_MODE
    try:
        if VECTOR_ENV_MODE == "async":
            vector_backend = f"async-{ASYNC_CONTEXT or 'default'}"
            return gym.vector.AsyncVectorEnv(env_fns, context=ASYNC_CONTEXT), vector_backend
        return gym.vector.SyncVectorEnv(env_fns), vector_backend
    except Exception:
        return gym.vector.SyncVectorEnv(env_fns), "sync-fallback"


def train_agent(
    benchmark: Any,
    env_factory: Any,
    candidate: CandidateSpec,
    device: torch.device,
    init_checkpoint: str | Any | None = None,
    live_callback: Any | None = None,
) -> tuple[Any, dict[str, Any]]:
    random.seed(benchmark.train_seed)
    np.random.seed(benchmark.train_seed)
    torch.manual_seed(benchmark.train_seed)

    render_env = env_factory(candidate.control_type, candidate.reward_recipe)
    render_obs, _ = render_env.reset(seed=benchmark.train_seed + 900_000)
    render_env.action_space.seed(benchmark.train_seed + 900_000)

    obs_dim = int(np.prod(render_env.observation_space.shape))
    act_dim = int(np.prod(render_env.action_space.shape))
    agent = build_agent(obs_dim, render_env.action_space.high, candidate, device)
    resumed_from = load_agent_checkpoint(agent, init_checkpoint) if init_checkpoint is not None else None

    replay = ReplayBuffer(obs_dim, act_dim, int(candidate.hyperparameters["replay_size"]))
    envs, vector_backend = _build_vector_envs(benchmark, candidate)
    obs, _ = envs.reset(seed=[benchmark.train_seed + idx for idx in range(NUM_ENVS)])

    total_steps = 0
    gradient_updates = 0
    last_metrics: dict[str, float] | None = None
    episode_records: list[dict[str, Any]] = []
    active_returns = np.zeros(NUM_ENVS, dtype=np.float64)
    active_lengths = np.zeros(NUM_ENVS, dtype=np.int64)
    render_episode_return = 0.0
    render_episode_length = 0
    live_step = 0

    started_at = time.time()
    budget_seconds = getattr(benchmark, "train_seconds", None)
    deadline = started_at + float(budget_seconds) if budget_seconds is not None else None
    start_steps = int(candidate.hyperparameters["start_steps"])
    update_after = int(candidate.hyperparameters["update_after"])
    batch_size = int(candidate.hyperparameters["batch_size"])
    gradient_steps = int(candidate.hyperparameters["gradient_steps"])

    def should_continue() -> bool:
        if len(episode_records) >= benchmark.train_episodes:
            return False
        if total_steps >= MAX_ENV_STEPS_SAFETY_CAP:
            return False
        if deadline is not None and time.time() >= deadline:
            return False
        return True

    def advance_render_env() -> tuple[float, int]:
        nonlocal render_obs, render_episode_return, render_episode_length
        if total_steps < start_steps:
            action = render_env.action_space.sample()
        else:
            action = agent.act(render_obs, deterministic=True)
        render_obs, reward, terminated, truncated, _ = render_env.step(action)
        render_episode_return += float(reward)
        render_episode_length += 1
        if terminated or truncated:
            finished_return = render_episode_return
            finished_length = render_episode_length
            render_obs, _ = render_env.reset(seed=benchmark.train_seed + 900_000 + len(episode_records))
            render_episode_return = 0.0
            render_episode_length = 0
            return finished_return, finished_length
        return render_episode_return, render_episode_length

    if live_callback is not None:
        live_callback(
            status="running",
            episode_records=episode_records,
            total_steps=total_steps,
            last_metrics=last_metrics,
            env=render_env,
            current_episode=1,
            episode_return=render_episode_return,
            episode_length=render_episode_length,
            agent=agent,
            elapsed_seconds=elapsed_seconds_since(started_at),
        )

    while should_continue():
        if total_steps < start_steps:
            actions = np.stack([envs.single_action_space.sample() for _ in range(NUM_ENVS)]).astype(np.float32)
        else:
            actions = _action_batch(agent, obs, deterministic=False)

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        dones = np.logical_or(terminations, truncations)
        for env_index in range(NUM_ENVS):
            replay.add(
                obs[env_index],
                actions[env_index],
                float(rewards[env_index]),
                next_obs[env_index],
                bool(terminations[env_index]),
            )
        obs = next_obs
        active_returns += rewards.astype(np.float64)
        active_lengths += 1
        total_steps += NUM_ENVS
        sidecar_return, sidecar_length = advance_render_env()

        for env_index in np.flatnonzero(dones):
            info = _info_for_env(infos, int(env_index))
            episode_records.append(
                make_train_episode_record(
                    episode=len(episode_records) + 1,
                    return_value=float(active_returns[env_index]),
                    length=int(active_lengths[env_index]),
                    success=bool(info.get("is_success", False)),
                    step=total_steps,
                    elapsed_seconds=elapsed_seconds_since(started_at),
                    info_metrics=scalar_info_metrics(info),
                    contacted_ball=bool(info.get("contacted_ball", False)),
                    env_index=int(env_index),
                )
            )
            active_returns[env_index] = 0.0
            active_lengths[env_index] = 0
            if len(episode_records) >= benchmark.train_episodes:
                break

        if total_steps >= update_after and replay.size >= batch_size:
            for _ in range(gradient_steps):
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

        if live_callback is not None and (total_steps == NUM_ENVS or total_steps - live_step >= LIVE_CALLBACK_EVERY_STEPS):
            live_step = total_steps
            live_callback(
                status="running",
                episode_records=episode_records,
                total_steps=total_steps,
                last_metrics=last_metrics,
                env=render_env,
                current_episode=len(episode_records) + 1,
                episode_return=sidecar_return,
                episode_length=sidecar_length,
                agent=agent,
                elapsed_seconds=elapsed_seconds_since(started_at),
            )

    envs.close()
    render_env.close()
    wall_clock = time.time() - started_at
    if len(episode_records) >= benchmark.train_episodes:
        stop_reason = "episode_cap_reached"
    elif total_steps >= MAX_ENV_STEPS_SAFETY_CAP:
        stop_reason = "max_env_steps_safety_cap_reached"
    elif deadline is not None and time.time() >= deadline:
        stop_reason = "time_budget_exhausted"
    else:
        stop_reason = "loop_exited"

    return agent, {
        "episodes": benchmark.train_episodes,
        "episodes_completed": len(episode_records),
        "time_budget_seconds": float(budget_seconds) if budget_seconds is not None else None,
        "stop_reason": stop_reason,
        "total_steps": total_steps,
        "env_steps": total_steps,
        "completed_episodes": len(episode_records),
        "episode_batches": len(episode_records),
        "avg_return": float(np.mean([e["return"] for e in episode_records])) if episode_records else 0.0,
        "success_rate": float(np.mean([1.0 if e["success"] else 0.0 for e in episode_records])) if episode_records else 0.0,
        "contacted_ball_rate": (
            float(np.mean([1.0 if e.get("contacted_ball") else 0.0 for e in episode_records]))
            if episode_records
            else 0.0
        ),
        "avg_length": float(np.mean([e["length"] for e in episode_records])) if episode_records else 0.0,
        "last_metrics": last_metrics,
        "episode_records": episode_records,
        "wall_clock_seconds": wall_clock,
        "vector_envs": NUM_ENVS,
        "vector_backend": vector_backend,
        "gradient_updates": gradient_updates,
        "visual_sampling": "single rgb_array sidecar env stepped by current deterministic policy",
        "init_checkpoint": str(init_checkpoint) if init_checkpoint is not None else None,
        "resumed_from": resumed_from,
        "curriculum": candidate.curriculum,
    }
