from __future__ import annotations

import time
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from autoresearch_gym.runner.curves import elapsed_seconds_since, make_train_episode_record, scalar_info_metrics


ALGORITHM = "so101_scripted_pick_place_baseline"
CONTROL_TYPE = None
REWARD_RECIPE = "task_dense"
ACTION_BLEND = 0.40
CTRL_LOW = np.asarray([-1.91986, -1.74533, -1.69, -1.65806, -2.74385, -0.17453], dtype=np.float32)
CTRL_HIGH = np.asarray([1.91986, 1.74533, 1.69, 1.65806, 2.84121, 1.74533], dtype=np.float32)
HOME_QPOS = np.asarray([0.0, -1.35, 1.69, 0.20, 0.0, -0.16], dtype=np.float32)
HOME_ACTION = (2.0 * (HOME_QPOS - CTRL_LOW) / (CTRL_HIGH - CTRL_LOW) - 1.0).astype(np.float32)


def flatten_observation(obs: Any) -> np.ndarray:
    if isinstance(obs, dict):
        pieces = []
        for key in ("observation", "achieved_goal", "desired_goal"):
            if key in obs:
                pieces.append(np.asarray(obs[key], dtype=np.float32).reshape(-1))
        return np.concatenate(pieces).astype(np.float32, copy=False)
    return np.asarray(obs, dtype=np.float32).reshape(-1)


class RewardRecipeWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env[Any, Any], recipe: str | None = None) -> None:
        super().__init__(env)
        self.recipe = recipe or REWARD_RECIPE
        if self.recipe != "task_dense":
            raise ValueError(f"Unknown SO-101 pick-place reward recipe: {self.recipe}")
        self.observation_space = _flatten_observation_space(env.observation_space)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        return flatten_observation(obs), info

    def step(self, action: np.ndarray):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["training_reward"] = float(reward)
        return flatten_observation(obs), float(reward), terminated, truncated, info


class Agent:
    def __init__(self, env_or_obs_dim: Any, action_dim: int | None = None) -> None:
        if hasattr(env_or_obs_dim, "action_space"):
            action_shape = getattr(env_or_obs_dim.action_space, "shape", (6,))
            self.action_dim = int(np.prod(action_shape))
        else:
            self.action_dim = int(action_dim or 6)
        self.best_action = np.zeros(self.action_dim, dtype=np.float32)
        self.best_action[: min(self.action_dim, HOME_ACTION.size)] = HOME_ACTION[: min(self.action_dim, HOME_ACTION.size)]

    def act(self, obs: Any, deterministic: bool = False) -> np.ndarray:
        del deterministic
        flat = flatten_observation(obs)
        ee_to_object = flat[9:12] if flat.size >= 12 else np.zeros(3, dtype=np.float32)
        object_to_target = flat[12:15] if flat.size >= 15 else np.zeros(3, dtype=np.float32)
        close_to_object = float(np.linalg.norm(ee_to_object)) < 0.065
        desired_delta = ee_to_object + (0.75 * object_to_target if close_to_object else 0.0)

        action = self.best_action.copy()
        if self.action_dim >= 5:
            action[0] = np.clip(action[0] + 2.0 * desired_delta[1], -1.0, 1.0)
            action[1] = np.clip(action[1] - 1.7 * desired_delta[2] - 0.7 * desired_delta[0], -1.0, 1.0)
            action[2] = np.clip(action[2] + 1.5 * desired_delta[0] + 1.1 * desired_delta[2], -1.0, 1.0)
            action[3] = np.clip(action[3] - 1.1 * desired_delta[2], -1.0, 1.0)
            action[4] = np.clip(action[4] + 0.4 * desired_delta[1], -1.0, 1.0)
        if self.action_dim >= 6:
            action[5] = -0.72 if close_to_object else 0.28
        return np.clip((1.0 - ACTION_BLEND) * self.best_action + ACTION_BLEND * action, -1.0, 1.0)

    def checkpoint_state(self) -> dict[str, Any]:
        return {"algorithm": ALGORITHM, "best_action": self.best_action.tolist(), "action_dim": self.action_dim}

    def load_checkpoint_state(self, state: dict[str, Any]) -> None:
        if state.get("algorithm") != ALGORITHM:
            raise ValueError(f"Checkpoint algorithm mismatch: {state.get('algorithm')}")
        self.action_dim = int(state.get("action_dim", self.action_dim))
        self.best_action = np.asarray(state.get("best_action", np.zeros(self.action_dim)), dtype=np.float32)


def save_agent_checkpoint(agent: Agent, checkpoint_path: str | Any, metadata: dict[str, Any] | None = None) -> None:
    torch.save({"agent_state": agent.checkpoint_state(), "metadata": metadata or {}}, checkpoint_path)


def load_agent_checkpoint(agent: Agent, checkpoint_path: str | Any) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    agent.load_checkpoint_state(payload["agent_state"])
    return payload.get("metadata", {})


def train_agent(
    benchmark: Any,
    env_factory: Any,
    candidate: Any,
    device: torch.device,
    init_checkpoint: str | Any | None = None,
    live_callback: Any | None = None,
) -> tuple[Agent, dict[str, Any]]:
    del candidate, device
    env = env_factory(control_type=CONTROL_TYPE, reward_recipe=REWARD_RECIPE)
    agent = Agent(env)
    if init_checkpoint is not None:
        load_agent_checkpoint(agent, init_checkpoint)
    rng = np.random.default_rng(int(benchmark.train_seed))
    episode_records: list[dict[str, Any]] = []
    total_steps = 0
    started_at = time.perf_counter()
    train_seconds = getattr(benchmark, "train_seconds", None)
    stop_reason = "episode_budget_exhausted"
    try:
        for episode in range(int(benchmark.train_episodes)):
            if train_seconds is not None and elapsed_seconds_since(started_at) >= float(train_seconds):
                stop_reason = "time_budget_exhausted"
                break
            obs, info = env.reset(seed=int(benchmark.train_seed) + episode)
            if live_callback is not None and episode == 0:
                live_callback(
                    status="starting",
                    episode_records=episode_records,
                    total_steps=total_steps,
                    last_metrics=None,
                    env=env,
                    current_episode=episode + 1,
                    episode_return=0.0,
                    episode_length=0,
                )
            episode_return = 0.0
            episode_length = 0
            terminated = False
            truncated = False
            while not (terminated or truncated):
                action = agent.act(obs, deterministic=False)
                if episode > 0:
                    action = np.clip(action + rng.normal(0.0, 0.07, size=agent.action_dim), -1.0, 1.0).astype(np.float32)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_return += float(reward)
                episode_length += 1
                total_steps += 1
                if train_seconds is not None and elapsed_seconds_since(started_at) >= float(train_seconds):
                    truncated = True
                    stop_reason = "time_budget_exhausted"
            record = make_train_episode_record(
                episode=episode + 1,
                step=total_steps,
                return_value=episode_return,
                length=episode_length,
                success=bool(info.get("is_success", False)),
                elapsed_seconds=elapsed_seconds_since(started_at),
                info_metrics=scalar_info_metrics(info),
            )
            episode_records.append(record)
            if live_callback is not None:
                live_callback(
                    status="running",
                    episode_records=episode_records,
                    total_steps=total_steps,
                    last_metrics=None,
                    env=env,
                    current_episode=episode + 1,
                    episode_return=episode_return,
                    episode_length=episode_length,
                )
            if stop_reason == "time_budget_exhausted":
                break
    finally:
        env.close()
    return agent, {
        "algorithm": ALGORITHM,
        "total_steps": total_steps,
        "env_steps": total_steps,
        "gradient_updates": 0,
        "episodes_completed": len(episode_records),
        "episode_batches": len(episode_records),
        "episode_records": episode_records,
        "stop_reason": stop_reason,
        "last_metrics": None,
    }


def _flatten_observation_space(space: gym.Space[Any]) -> gym.spaces.Box:
    if not isinstance(space, gym.spaces.Dict):
        assert isinstance(space, gym.spaces.Box)
        return gym.spaces.Box(
            low=np.asarray(space.low, dtype=np.float32).reshape(-1),
            high=np.asarray(space.high, dtype=np.float32).reshape(-1),
            dtype=np.float32,
        )
    lows = []
    highs = []
    for key in ("observation", "achieved_goal", "desired_goal"):
        subspace = space.spaces[key]
        assert isinstance(subspace, gym.spaces.Box)
        lows.append(np.asarray(subspace.low, dtype=np.float32).reshape(-1))
        highs.append(np.asarray(subspace.high, dtype=np.float32).reshape(-1))
    return gym.spaces.Box(
        low=np.concatenate(lows).astype(np.float32),
        high=np.concatenate(highs).astype(np.float32),
        dtype=np.float32,
    )
