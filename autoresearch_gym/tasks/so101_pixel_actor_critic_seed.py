from __future__ import annotations

import time
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from autoresearch_gym.envs.vision import PixelObservationWrapper
from autoresearch_gym.runner.curves import elapsed_seconds_since, make_train_episode_record, scalar_info_metrics


EXP_NAME = "so101_pixel_actor_critic"
ALGORITHM = "pixel_actor_critic"
REWARD_RECIPE = "task_dense"
POLICY_FEEDS = ("wrist",)
IMAGE_SIZE = (84, 84)
FRAME_STACK = 3
GAMMA = 0.97
LR = 3e-4
ENTROPY_COEF = 0.002
VALUE_COEF = 0.5


def get_candidate() -> dict[str, Any]:
    return {
        "description": (
            "SO-101 low-resolution pixel actor-critic seed. The policy observes the "
            "arm-mounted wrist feed plus proprioception; world/overview frames remain "
            "debug-only synchronized trajectory feeds."
        ),
        "recipe": {
            "algorithm": ALGORITHM,
            "reward_recipe": REWARD_RECIPE,
            "policy_feeds": list(POLICY_FEEDS),
            "image_size": list(IMAGE_SIZE),
            "frame_stack": FRAME_STACK,
            "runner": {"sample_trajectory_source": "runner_eval"},
        },
    }


class RewardRecipeWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env[Any, Any], recipe: str | None = None) -> None:
        if recipe not in {None, REWARD_RECIPE}:
            raise ValueError(f"Unknown SO-101 pixel reward recipe: {recipe}")
        obs_space = getattr(env, "observation_space", None)
        if isinstance(obs_space, gym.spaces.Dict) and "pixels" in obs_space.spaces:
            wrapped = env
        else:
            wrapped = PixelObservationWrapper(
                env,
                policy_feeds=POLICY_FEEDS,
                image_size=IMAGE_SIZE,
                frame_stack=FRAME_STACK,
                include_proprio=True,
            )
        super().__init__(wrapped)
        self.recipe = REWARD_RECIPE


class PixelActorCritic(nn.Module):
    def __init__(self, pixel_shape: tuple[int, int, int], proprio_dim: int, action_dim: int) -> None:
        super().__init__()
        channels, height, width = pixel_shape
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            encoded_dim = int(self.encoder(torch.zeros(1, channels, height, width)).shape[-1])
        hidden_dim = 256
        self.trunk = nn.Sequential(nn.Linear(encoded_dim + proprio_dim, hidden_dim), nn.ReLU())
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.full((action_dim,), -0.7))
        self.value = nn.Linear(hidden_dim, 1)

    def features(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        pixels = obs["pixels"].float() / 255.0
        encoded = self.encoder(pixels)
        proprio = obs.get("proprio")
        if proprio is None:
            proprio = torch.zeros((pixels.shape[0], 0), dtype=pixels.dtype, device=pixels.device)
        return self.trunk(torch.cat([encoded, proprio.float()], dim=-1))

    def forward(self, obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.features(obs)
        mean = self.actor_mean(features)
        log_std = torch.clamp(self.actor_log_std, -5.0, 1.0).expand_as(mean)
        value = self.value(features).squeeze(-1)
        return mean, log_std, value


class Agent:
    def __init__(self, env_or_obs_space: Any, action_dim: int | None = None, device: torch.device | None = None) -> None:
        self.device = device or torch.device("cpu")
        if hasattr(env_or_obs_space, "observation_space"):
            obs_space = env_or_obs_space.observation_space
            action_shape = getattr(env_or_obs_space.action_space, "shape", (6,))
            action_dim = int(np.prod(action_shape))
        else:
            obs_space = env_or_obs_space
        assert isinstance(obs_space, gym.spaces.Dict)
        pixel_space = obs_space.spaces["pixels"]
        assert isinstance(pixel_space, gym.spaces.Box)
        proprio_space = obs_space.spaces.get("proprio")
        proprio_dim = int(np.prod(proprio_space.shape)) if isinstance(proprio_space, gym.spaces.Box) else 0
        self.model = PixelActorCritic(tuple(int(v) for v in pixel_space.shape), proprio_dim, int(action_dim or 6)).to(self.device)

    def _tensor_obs(self, obs: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        tensors = {
            "pixels": torch.as_tensor(obs["pixels"], dtype=torch.uint8, device=self.device).unsqueeze(0),
        }
        if "proprio" in obs:
            tensors["proprio"] = torch.as_tensor(obs["proprio"], dtype=torch.float32, device=self.device).unsqueeze(0)
        return tensors

    def act(self, obs: Any, deterministic: bool = False) -> np.ndarray:
        assert isinstance(obs, dict)
        with torch.no_grad():
            mean, log_std, _value = self.model(self._tensor_obs(obs))
            if deterministic:
                raw_action = mean
            else:
                raw_action = Normal(mean, log_std.exp()).sample()
            action = torch.tanh(raw_action)
        return action.squeeze(0).detach().cpu().numpy().astype(np.float32)

    def sample_action(self, obs: dict[str, np.ndarray]) -> tuple[np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std, value = self.model(self._tensor_obs(obs))
        dist = Normal(mean, log_std.exp())
        raw_action = dist.rsample()
        action = torch.tanh(raw_action)
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action.squeeze(0).detach().cpu().numpy().astype(np.float32), log_prob.squeeze(0), value.squeeze(0), entropy.squeeze(0)


def save_agent_checkpoint(agent: Agent, checkpoint_path: str | Any, metadata: dict[str, Any] | None = None) -> None:
    torch.save({"model_state": agent.model.state_dict(), "metadata": metadata or {}}, checkpoint_path)


def load_agent_checkpoint(agent: Agent, checkpoint_path: str | Any) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location=agent.device)
    agent.model.load_state_dict(payload["model_state"])
    return payload.get("metadata", {})


def train_agent(
    benchmark: Any,
    env_factory: Any,
    candidate: Any,
    device: torch.device,
    init_checkpoint: str | Any | None = None,
    live_callback: Any | None = None,
) -> tuple[Agent, dict[str, Any]]:
    del candidate
    env = env_factory(reward_recipe=REWARD_RECIPE)
    agent = Agent(env, device=device)
    if init_checkpoint is not None:
        load_agent_checkpoint(agent, init_checkpoint)
    optimizer = torch.optim.Adam(agent.model.parameters(), lr=LR)
    episode_records: list[dict[str, Any]] = []
    total_steps = 0
    gradient_updates = 0
    last_metrics: dict[str, float] | None = None
    started_at = time.time()
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
                    last_metrics=last_metrics,
                    env=env,
                    current_episode=episode + 1,
                    episode_return=0.0,
                    episode_length=0,
                )
            rewards: list[float] = []
            log_probs: list[torch.Tensor] = []
            values: list[torch.Tensor] = []
            entropies: list[torch.Tensor] = []
            episode_return = 0.0
            episode_length = 0
            terminated = False
            truncated = False
            while not (terminated or truncated):
                action, log_prob, value, entropy = agent.sample_action(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                rewards.append(float(reward))
                log_probs.append(log_prob)
                values.append(value)
                entropies.append(entropy)
                episode_return += float(reward)
                episode_length += 1
                total_steps += 1
                if train_seconds is not None and elapsed_seconds_since(started_at) >= float(train_seconds):
                    truncated = True
                    stop_reason = "time_budget_exhausted"
            if rewards:
                returns = _discounted_returns(rewards, GAMMA, device)
                values_t = torch.stack(values)
                log_probs_t = torch.stack(log_probs)
                entropies_t = torch.stack(entropies)
                advantages = returns - values_t.detach()
                policy_loss = -(log_probs_t * advantages).mean()
                value_loss = torch.mean((values_t - returns) ** 2)
                entropy_bonus = entropies_t.mean()
                loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy_bonus
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(agent.model.parameters(), 5.0)
                optimizer.step()
                gradient_updates += 1
                last_metrics = {
                    "loss": float(loss.detach().cpu()),
                    "policy_loss": float(policy_loss.detach().cpu()),
                    "value_loss": float(value_loss.detach().cpu()),
                    "entropy": float(entropy_bonus.detach().cpu()),
                    "gradient_updates": float(gradient_updates),
                }
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
                    last_metrics=last_metrics,
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
        "gradient_updates": gradient_updates,
        "episodes_completed": len(episode_records),
        "completed_episodes": len(episode_records),
        "episode_batches": len(episode_records),
        "episode_records": episode_records,
        "stop_reason": stop_reason,
        "last_metrics": last_metrics,
    }


def _discounted_returns(rewards: list[float], gamma: float, device: torch.device) -> torch.Tensor:
    returns: list[float] = []
    running = 0.0
    for reward in reversed(rewards):
        running = float(reward) + gamma * running
        returns.append(running)
    returns.reverse()
    tensor = torch.as_tensor(returns, dtype=torch.float32, device=device)
    if tensor.numel() > 1:
        tensor = (tensor - tensor.mean()) / (tensor.std(unbiased=False) + 1e-6)
    return tensor
