from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from torch import nn
from torch.distributions import Normal

from autoresearch_gym.runner.curves import elapsed_seconds_since, make_train_episode_record


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


@dataclass(frozen=True)
class MbpoRecipe:
    control_type: str | None
    reward_recipe: str | None
    hidden_dims: tuple[int, ...]
    batch_size: int
    replay_size: int
    gamma: float
    tau: float
    start_steps: int
    update_after: int
    gradient_steps: int
    actor_lr: float
    critic_lr: float
    alpha_lr: float
    init_temperature: float
    model_hidden_dims: tuple[int, ...]
    ensemble_size: int
    model_lr: float
    model_warmup_steps: int
    model_train_interval: int
    model_train_epochs: int
    model_batch_size: int
    model_rollout_horizon: int
    model_rollout_starts: int
    model_replay_size: int
    model_batch_fraction: float
    uncertainty_threshold: float
    priority_fraction: float

    @classmethod
    def from_candidate(cls, candidate: dict[str, Any]) -> "MbpoRecipe":
        recipe = dict(candidate.get("recipe", {}))
        sac = dict(recipe.get("sac", {}))
        world_model = dict(recipe.get("world_model", {}))
        return cls(
            control_type=recipe.get("control_type"),
            reward_recipe=recipe.get("reward_recipe"),
            hidden_dims=tuple(int(v) for v in sac.get("hidden_dims", (256, 256))),
            batch_size=int(sac.get("batch_size", 128)),
            replay_size=int(sac.get("replay_size", 100_000)),
            gamma=float(sac.get("gamma", 0.99)),
            tau=float(sac.get("tau", 0.005)),
            start_steps=int(sac.get("start_steps", 256)),
            update_after=int(sac.get("update_after", 128)),
            gradient_steps=int(sac.get("gradient_steps", 1)),
            actor_lr=float(sac.get("actor_lr", 3e-4)),
            critic_lr=float(sac.get("critic_lr", 3e-4)),
            alpha_lr=float(sac.get("alpha_lr", 3e-4)),
            init_temperature=float(sac.get("init_temperature", 0.1)),
            model_hidden_dims=tuple(int(v) for v in world_model.get("hidden_dims", (256, 256, 256))),
            ensemble_size=int(world_model.get("ensemble_size", 5)),
            model_lr=float(world_model.get("lr", 3e-4)),
            model_warmup_steps=int(world_model.get("warmup_steps", 1_000)),
            model_train_interval=int(world_model.get("train_interval", 500)),
            model_train_epochs=int(world_model.get("train_epochs", 3)),
            model_batch_size=int(world_model.get("batch_size", 256)),
            model_rollout_horizon=int(world_model.get("rollout_horizon", 1)),
            model_rollout_starts=int(world_model.get("rollout_starts", 256)),
            model_replay_size=int(world_model.get("model_replay_size", 100_000)),
            model_batch_fraction=float(world_model.get("batch_fraction", 0.25)),
            uncertainty_threshold=float(world_model.get("uncertainty_threshold", 1.0)),
            priority_fraction=float(world_model.get("priority_fraction", 0.5)),
        )


class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, capacity: int) -> None:
        self.capacity = int(capacity)
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        priority: float = 0.0,
    ) -> None:
        self.obs[self.ptr] = np.asarray(obs, dtype=np.float32)
        self.actions[self.ptr] = np.asarray(action, dtype=np.float32)
        self.rewards[self.ptr] = float(reward)
        self.next_obs[self.ptr] = np.asarray(next_obs, dtype=np.float32)
        self.dones[self.ptr] = float(done)
        self.priorities[self.ptr] = float(priority)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_batch(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        for idx in range(obs.shape[0]):
            self.add(obs[idx], actions[idx], float(rewards[idx]), next_obs[idx], bool(dones[idx]))

    def sample(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, ...]:
        idx = np.random.randint(0, self.size, size=int(batch_size))
        return self._take(idx, device)

    def sample_mixed_with(self, other: "ReplayBuffer", batch_size: int, fraction_other: float, device: torch.device):
        other_count = min(int(batch_size * fraction_other), other.size)
        real_count = int(batch_size) - other_count
        if other_count <= 0:
            return self.sample(batch_size, device)
        real_idx = np.random.randint(0, self.size, size=real_count)
        other_idx = np.random.randint(0, other.size, size=other_count)
        obs = np.concatenate([self.obs[real_idx], other.obs[other_idx]], axis=0)
        actions = np.concatenate([self.actions[real_idx], other.actions[other_idx]], axis=0)
        rewards = np.concatenate([self.rewards[real_idx], other.rewards[other_idx]], axis=0)
        next_obs = np.concatenate([self.next_obs[real_idx], other.next_obs[other_idx]], axis=0)
        dones = np.concatenate([self.dones[real_idx], other.dones[other_idx]], axis=0)
        order = np.random.permutation(obs.shape[0])
        return (
            torch.as_tensor(obs[order], device=device),
            torch.as_tensor(actions[order], device=device),
            torch.as_tensor(rewards[order], device=device),
            torch.as_tensor(next_obs[order], device=device),
            torch.as_tensor(dones[order], device=device),
        )

    def sample_start_states(self, count: int, priority_fraction: float) -> np.ndarray:
        count = min(int(count), self.size)
        if count <= 0:
            return np.zeros((0, self.obs.shape[1]), dtype=np.float32)
        priority_count = min(count, int(round(count * priority_fraction)))
        random_count = count - priority_count
        indices: list[np.ndarray] = []
        priority_pool = np.flatnonzero(self.priorities[: self.size] > 0.0)
        if priority_count > 0 and priority_pool.size > 0:
            replace = priority_pool.size < priority_count
            indices.append(np.random.choice(priority_pool, size=priority_count, replace=replace))
        else:
            random_count = count
        if random_count > 0:
            indices.append(np.random.randint(0, self.size, size=random_count))
        return self.obs[np.concatenate(indices)].astype(np.float32, copy=True)

    def _take(self, idx: np.ndarray, device: torch.device) -> tuple[torch.Tensor, ...]:
        return (
            torch.as_tensor(self.obs[idx], device=device),
            torch.as_tensor(self.actions[idx], device=device),
            torch.as_tensor(self.rewards[idx], device=device),
            torch.as_tensor(self.next_obs[idx], device=device),
            torch.as_tensor(self.dones[idx], device=device),
        )


def build_mlp(in_dim: int, hidden_dims: tuple[int, ...], out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = int(in_dim)
    for width in hidden_dims:
        layers.extend([nn.Linear(last, int(width)), nn.ReLU()])
        last = int(width)
    layers.append(nn.Linear(last, int(out_dim)))
    return nn.Sequential(*layers)


class SquashedGaussianActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: tuple[int, ...], action_scale: torch.Tensor) -> None:
        super().__init__()
        self.net = build_mlp(obs_dim, hidden_dims, act_dim * 2)
        self.register_buffer("action_scale", action_scale)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.net(obs).chunk(2, dim=-1)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1.0)
        return mean, log_std

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(obs)
        dist = Normal(mean, log_std.exp())
        pre_tanh = dist.rsample()
        squashed = torch.tanh(pre_tanh)
        action = squashed * self.action_scale
        log_prob = dist.log_prob(pre_tanh) - torch.log(self.action_scale * (1.0 - squashed.pow(2)) + 1e-6)
        return action, log_prob.sum(dim=-1, keepdim=True)

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        mean, log_std = self(obs)
        squashed = torch.tanh(mean if deterministic else Normal(mean, log_std.exp()).sample())
        return squashed * self.action_scale


class Critic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: tuple[int, ...]) -> None:
        super().__init__()
        self.q = build_mlp(obs_dim + act_dim, hidden_dims, 1)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.q(torch.cat([obs, act], dim=-1))


class _MbpoAgent:
    def __init__(self, obs_dim: int, act_high: np.ndarray, recipe: MbpoRecipe, device: torch.device) -> None:
        self.device = device
        self.observation_adapter: Callable[[Any], np.ndarray] | None = None
        self.act_dim = int(act_high.shape[0])
        self.batch_size = recipe.batch_size
        self.gamma = recipe.gamma
        self.tau = recipe.tau
        action_scale = torch.as_tensor(act_high, dtype=torch.float32, device=device)
        self.actor = SquashedGaussianActor(obs_dim, self.act_dim, recipe.hidden_dims, action_scale).to(device)
        self.q1 = Critic(obs_dim, self.act_dim, recipe.hidden_dims).to(device)
        self.q2 = Critic(obs_dim, self.act_dim, recipe.hidden_dims).to(device)
        self.q1_target = Critic(obs_dim, self.act_dim, recipe.hidden_dims).to(device)
        self.q2_target = Critic(obs_dim, self.act_dim, recipe.hidden_dims).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=recipe.actor_lr)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=recipe.critic_lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=recipe.critic_lr)
        self.target_entropy = -float(self.act_dim)
        self.log_alpha = torch.tensor(math.log(recipe.init_temperature), dtype=torch.float32, device=device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=recipe.alpha_lr)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        if self.observation_adapter is not None:
            obs = self.observation_adapter(obs)
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).reshape(1, -1)
        with torch.no_grad():
            action = self.actor.act(obs_t, deterministic=deterministic)
        return action.squeeze(0).cpu().numpy()

    def update(self, real_replay: ReplayBuffer, model_replay: ReplayBuffer | None = None, model_fraction: float = 0.0) -> dict[str, float]:
        if model_replay is not None and model_replay.size > 0 and model_fraction > 0.0:
            obs, act, rew, next_obs, done = real_replay.sample_mixed_with(model_replay, self.batch_size, model_fraction, self.device)
        else:
            obs, act, rew, next_obs, done = real_replay.sample(self.batch_size, self.device)
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_obs)
            target_q = torch.min(self.q1_target(next_obs, next_action), self.q2_target(next_obs, next_action))
            target = rew + (1.0 - done) * self.gamma * (target_q - self.alpha.detach() * next_log_prob)
        q1_loss = nn.functional.mse_loss(self.q1(obs, act), target)
        q2_loss = nn.functional.mse_loss(self.q2(obs, act), target)
        self.q1_opt.zero_grad()
        q1_loss.backward()
        self.q1_opt.step()
        self.q2_opt.zero_grad()
        q2_loss.backward()
        self.q2_opt.step()
        sampled_action, log_prob = self.actor.sample(obs)
        actor_loss = (self.alpha.detach() * log_prob - torch.min(self.q1(obs, sampled_action), self.q2(obs, sampled_action))).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()
        with torch.no_grad():
            for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
                target_param.data.mul_(1.0 - self.tau).add_(self.tau * param.data)
            for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
                target_param.data.mul_(1.0 - self.tau).add_(self.tau * param.data)
        return {
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(((q1_loss + q2_loss) * 0.5).item()),
            "alpha": float(self.alpha.item()),
            "model_replay_size": float(model_replay.size if model_replay is not None else 0),
        }

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "algorithm": "cleanrl_sac_mbpo",
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "q1_opt": self.q1_opt.state_dict(),
            "q2_opt": self.q2_opt.state_dict(),
            "alpha_opt": self.alpha_opt.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
        }

    def load_checkpoint_state(self, state: dict[str, Any]) -> None:
        if state.get("algorithm") != "cleanrl_sac_mbpo":
            raise ValueError("Checkpoint algorithm mismatch for CleanRL MBPO agent")
        self.actor.load_state_dict(state["actor"])
        self.q1.load_state_dict(state["q1"])
        self.q2.load_state_dict(state["q2"])
        self.q1_target.load_state_dict(state["q1_target"])
        self.q2_target.load_state_dict(state["q2_target"])
        self.actor_opt.load_state_dict(state["actor_opt"])
        self.q1_opt.load_state_dict(state["q1_opt"])
        self.q2_opt.load_state_dict(state["q2_opt"])
        self.alpha_opt.load_state_dict(state["alpha_opt"])
        with torch.no_grad():
            self.log_alpha.copy_(state["log_alpha"].to(self.device))


class Normalizer:
    def __init__(self, in_mean: np.ndarray, in_std: np.ndarray, out_mean: np.ndarray, out_std: np.ndarray) -> None:
        self.in_mean = in_mean.astype(np.float32)
        self.in_std = np.maximum(in_std.astype(np.float32), 1e-6)
        self.out_mean = out_mean.astype(np.float32)
        self.out_std = np.maximum(out_std.astype(np.float32), 1e-6)


class WorldModelEnsemble:
    def __init__(self, obs_dim: int, act_dim: int, recipe: MbpoRecipe, device: torch.device) -> None:
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = device
        self.models = nn.ModuleList(
            [build_mlp(obs_dim + act_dim, recipe.model_hidden_dims, obs_dim + 2).to(device) for _ in range(recipe.ensemble_size)]
        )
        self.optimizers = [torch.optim.Adam(model.parameters(), lr=recipe.model_lr) for model in self.models]
        self.normalizer: Normalizer | None = None

    def train_from_replay(self, replay: ReplayBuffer, epochs: int, batch_size: int) -> dict[str, float]:
        if replay.size < max(2, batch_size):
            return {}
        inputs = np.concatenate([replay.obs[: replay.size], replay.actions[: replay.size]], axis=1)
        targets = np.concatenate(
            [
                replay.next_obs[: replay.size] - replay.obs[: replay.size],
                replay.rewards[: replay.size],
                replay.dones[: replay.size],
            ],
            axis=1,
        )
        self.normalizer = Normalizer(inputs.mean(axis=0), inputs.std(axis=0), targets.mean(axis=0), targets.std(axis=0))
        metrics: dict[str, float] = {}
        input_t = torch.as_tensor((inputs - self.normalizer.in_mean) / self.normalizer.in_std, dtype=torch.float32, device=self.device)
        target_t = torch.as_tensor((targets - self.normalizer.out_mean) / self.normalizer.out_std, dtype=torch.float32, device=self.device)
        count = input_t.shape[0]
        for model, opt in zip(self.models, self.optimizers):
            last_loss = 0.0
            for _ in range(int(epochs)):
                order = torch.randperm(count, device=self.device)
                for start in range(0, count, int(batch_size)):
                    idx = order[start : start + int(batch_size)]
                    pred = model(input_t[idx])
                    loss = nn.functional.mse_loss(pred, target_t[idx])
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    last_loss = float(loss.item())
            metrics["world_model_loss"] = metrics.get("world_model_loss", 0.0) + last_loss / max(1, len(self.models))
        with torch.no_grad():
            pred_delta, pred_reward, pred_done, uncertainty = self.predict(replay.obs[: replay.size], replay.actions[: replay.size])
        metrics.update(
            {
                "model_delta_rmse": float(np.sqrt(np.mean((pred_delta - (replay.next_obs[: replay.size] - replay.obs[: replay.size])) ** 2))),
                "model_reward_rmse": float(np.sqrt(np.mean((pred_reward[:, 0] - replay.rewards[: replay.size, 0]) ** 2))),
                "model_done_accuracy": float(np.mean((pred_done[:, 0] >= 0.5) == (replay.dones[: replay.size, 0] >= 0.5))),
                "model_uncertainty": float(np.mean(uncertainty)),
            }
        )
        return metrics

    def predict(self, obs: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.normalizer is None:
            raise RuntimeError("World model has not been trained yet")
        inputs = np.concatenate([obs, actions], axis=1).astype(np.float32)
        x = torch.as_tensor((inputs - self.normalizer.in_mean) / self.normalizer.in_std, dtype=torch.float32, device=self.device)
        preds = []
        with torch.no_grad():
            for model in self.models:
                pred = model(x).cpu().numpy()
                preds.append(pred * self.normalizer.out_std + self.normalizer.out_mean)
        stacked = np.stack(preds, axis=0)
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)
        pred_delta = mean[:, : self.obs_dim]
        pred_reward = mean[:, self.obs_dim : self.obs_dim + 1]
        pred_done = 1.0 / (1.0 + np.exp(-mean[:, self.obs_dim + 1 : self.obs_dim + 2]))
        uncertainty = np.mean(std[:, :, : self.obs_dim], axis=(0, 2))
        return pred_delta.astype(np.float32), pred_reward.astype(np.float32), pred_done.astype(np.float32), uncertainty.astype(np.float32)


def generate_model_rollouts(
    real_replay: ReplayBuffer,
    model_replay: ReplayBuffer,
    world_model: WorldModelEnsemble,
    agent: Agent,
    recipe: MbpoRecipe,
) -> dict[str, float]:
    starts = real_replay.sample_start_states(recipe.model_rollout_starts, recipe.priority_fraction)
    if starts.shape[0] == 0:
        return {}
    obs = starts
    added = 0
    stopped_uncertain = 0
    stopped_done = 0
    for _ in range(recipe.model_rollout_horizon):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=agent.device)
        with torch.no_grad():
            actions = agent.actor.act(obs_t, deterministic=False).cpu().numpy().astype(np.float32)
        pred_delta, pred_reward, pred_done, uncertainty = world_model.predict(obs, actions)
        next_obs = np.clip(obs + pred_delta, -10.0, 10.0).astype(np.float32)
        accepted = uncertainty <= recipe.uncertainty_threshold
        if np.any(accepted):
            dones = pred_done[accepted, 0] >= 0.5
            model_replay.add_batch(obs[accepted], actions[accepted], pred_reward[accepted, 0], next_obs[accepted], dones)
            added += int(np.sum(accepted))
            stopped_done += int(np.sum(dones))
        stopped_uncertain += int(np.sum(~accepted))
        keep = accepted & (pred_done[:, 0] < 0.5)
        if not np.any(keep):
            break
        obs = next_obs[keep]
    return {
        "synthetic_added": float(added),
        "stopped_uncertain": float(stopped_uncertain),
        "stopped_done": float(stopped_done),
        "model_replay_size": float(model_replay.size),
    }


def save_agent_checkpoint(agent: Agent, checkpoint_path: str | Path, metadata: dict[str, Any] | None = None) -> None:
    torch.save({"agent_state": agent.checkpoint_state(), "metadata": metadata or {}}, checkpoint_path)


def load_agent_checkpoint(agent: Agent, checkpoint_path: str | Path) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location=agent.device)
    agent.load_checkpoint_state(payload["agent_state"])
    return payload.get("metadata", {})


def _train_mbpo_agent(
    *,
    benchmark: Any,
    env_factory: Any,
    candidate: dict[str, Any],
    device: torch.device,
    init_checkpoint: str | Path | None = None,
    live_callback: Any | None = None,
    priority_fn: Callable[[dict[str, Any], np.ndarray], float] | None = None,
    record_extra_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    observation_adapter: Callable[[Any], np.ndarray] | None = None,
) -> tuple[Agent, dict[str, Any]]:
    recipe = MbpoRecipe.from_candidate(candidate)
    np.random.seed(int(benchmark.train_seed))
    torch.manual_seed(int(benchmark.train_seed))
    env: gym.Env[Any, Any] = env_factory(recipe.control_type, recipe.reward_recipe)
    env.action_space.seed(int(benchmark.train_seed))
    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.shape[0])
    agent = _MbpoAgent(obs_dim, np.asarray(env.action_space.high, dtype=np.float32), recipe, device)
    agent.observation_adapter = observation_adapter
    resumed_from = load_agent_checkpoint(agent, init_checkpoint) if init_checkpoint is not None else None
    real_replay = ReplayBuffer(obs_dim, act_dim, recipe.replay_size)
    model_replay = ReplayBuffer(obs_dim, act_dim, recipe.model_replay_size)
    world_model = WorldModelEnsemble(obs_dim, act_dim, recipe, device)
    total_steps = 0
    gradient_updates = 0
    model_updates = 0
    last_metrics: dict[str, float] | None = None
    last_model_metrics: dict[str, float] = {}
    episode_records: list[dict[str, Any]] = []
    started_at = time.time()
    budget_seconds = getattr(benchmark, "train_seconds", None)
    deadline = started_at + float(budget_seconds) if budget_seconds is not None else None

    for episode in range(1, int(benchmark.train_episodes) + 1):
        if deadline is not None and time.time() >= deadline:
            break
        obs, info = env.reset(seed=int(benchmark.train_seed) + episode)
        episode_return = 0.0
        episode_length = 0
        terminated = False
        truncated = False
        while not (terminated or truncated) and (deadline is None or time.time() < deadline):
            total_steps += 1
            action = env.action_space.sample() if total_steps <= recipe.start_steps else agent.act(obs, deterministic=False)
            next_obs, reward, terminated, truncated, info = env.step(action)
            priority = priority_fn(info, next_obs) if priority_fn is not None else 0.0
            real_replay.add(obs, action, float(reward), next_obs, bool(terminated), priority=priority)
            obs = next_obs
            episode_return += float(reward)
            episode_length += 1

            if real_replay.size >= recipe.model_batch_size and total_steps >= recipe.model_warmup_steps:
                if total_steps == recipe.model_warmup_steps or total_steps % recipe.model_train_interval == 0:
                    last_model_metrics = world_model.train_from_replay(real_replay, recipe.model_train_epochs, recipe.model_batch_size)
                    if last_model_metrics:
                        rollout_metrics = generate_model_rollouts(real_replay, model_replay, world_model, agent, recipe)
                        last_model_metrics = {**last_model_metrics, **rollout_metrics}
                        model_updates += 1

            if total_steps >= recipe.update_after and real_replay.size >= recipe.batch_size:
                for _ in range(recipe.gradient_steps):
                    last_metrics = agent.update(real_replay, model_replay, recipe.model_batch_fraction)
                    gradient_updates += 1
                last_metrics = {
                    **(last_metrics or {}),
                    **last_model_metrics,
                    "gradient_updates": float(gradient_updates),
                    "world_model_updates": float(model_updates),
                }

            if live_callback is not None and (total_steps == 1 or total_steps % 10 == 0):
                live_callback(
                    status="running",
                    episode_records=episode_records,
                    total_steps=total_steps,
                    last_metrics=last_metrics,
                    env=env,
                    current_episode=episode,
                    episode_return=episode_return,
                    episode_length=episode_length,
                    agent=agent,
                    elapsed_seconds=elapsed_seconds_since(started_at),
                    diagnostic_series=candidate.get("diagnostic_series"),
                )

        extra = record_extra_fn(info) if record_extra_fn is not None else {}
        episode_records.append(
            make_train_episode_record(
                episode=episode,
                return_value=episode_return,
                length=episode_length,
                success=bool(info.get("is_success", False)),
                step=total_steps,
                elapsed_seconds=elapsed_seconds_since(started_at),
                info_metrics={**extra, **last_model_metrics},
            )
        )
        if live_callback is not None:
            live_callback(
                status="running",
                episode_records=episode_records,
                total_steps=total_steps,
                last_metrics=last_metrics,
                env=env,
                current_episode=episode,
                episode_return=episode_return,
                episode_length=episode_length,
                agent=agent,
                elapsed_seconds=elapsed_seconds_since(started_at),
                diagnostic_series=candidate.get("diagnostic_series"),
            )

    env.close()
    wall_clock = elapsed_seconds_since(started_at)
    if deadline is not None and wall_clock >= float(budget_seconds):
        stop_reason = "time_budget_exhausted"
    elif len(episode_records) >= int(benchmark.train_episodes):
        stop_reason = "episode_cap_reached"
    else:
        stop_reason = "loop_exited"
    successes = [1.0 if record.get("success") else 0.0 for record in episode_records]
    return agent, {
        "episodes": int(benchmark.train_episodes),
        "episodes_completed": len(episode_records),
        "time_budget_seconds": float(budget_seconds) if budget_seconds is not None else None,
        "stop_reason": stop_reason,
        "total_steps": total_steps,
        "env_steps": total_steps,
        "completed_episodes": len(episode_records),
        "episode_batches": len(episode_records),
        "avg_return": float(np.mean([record["return"] for record in episode_records])) if episode_records else 0.0,
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "avg_length": float(np.mean([record["length"] for record in episode_records])) if episode_records else 0.0,
        "last_metrics": {**(last_metrics or {}), **last_model_metrics},
        "gradient_updates": gradient_updates,
        "world_model_updates": model_updates,
        "model_replay_size": model_replay.size,
        "episode_records": episode_records,
        "wall_clock_seconds": wall_clock,
        "init_checkpoint": str(init_checkpoint) if init_checkpoint is not None else None,
        "resumed_from": resumed_from,
        "recipe": candidate.get("recipe", {}),
        "diagnostic_series": candidate.get("diagnostic_series"),
    }


REWARD_RECIPE = "subskill_mbpo"
RECIPE = {
    "style": "cleanrl_sac_mbpo",
    "algorithm": "sac_mbpo",
    "control_type": None,
    "reward_recipe": REWARD_RECIPE,
    "sac": {
        "hidden_dims": (256, 256),
        "batch_size": 128,
        "replay_size": 150000,
        "gamma": 0.98,
        "tau": 0.005,
        "start_steps": 512,
        "update_after": 256,
        "gradient_steps": 1,
        "actor_lr": 3e-4,
        "critic_lr": 3e-4,
        "alpha_lr": 3e-4,
        "init_temperature": 0.1,
    },
    "world_model": {
        "hidden_dims": (256, 256, 256),
        "ensemble_size": 5,
        "lr": 3e-4,
        "warmup_steps": 1500,
        "train_interval": 750,
        "train_epochs": 3,
        "batch_size": 256,
        "rollout_horizon": 1,
        "rollout_starts": 512,
        "model_replay_size": 150000,
        "batch_fraction": 0.20,
        "uncertainty_threshold": 1.0,
        "priority_fraction": 0.75,
        "priority_signal": "near_cube_lift_or_place",
    },
}
DIAGNOSTIC_SERIES = {
    "title": "Panda pick-and-place MBPO diagnostics",
    "series": [
        {"key": "ee_to_cube_distance", "label": "EE to Cube", "color": "#2563eb", "source": "info_metrics", "chart": "normalized_line", "group": "subskills"},
        {"key": "cube_lift_height", "label": "Lift", "color": "#16a34a", "source": "info_metrics", "chart": "normalized_line", "group": "subskills"},
        {"key": "cube_to_goal_distance", "label": "Cube Goal", "color": "#dc2626", "source": "info_metrics", "chart": "normalized_line", "group": "subskills"},
        {"key": "lifted_ever_rate", "label": "Lifted", "color": "#15803d", "source": "info_metrics", "chart": "normalized_line", "group": "subskills"},
        {"key": "placed_success_rate", "label": "Placed", "color": "#7c3aed", "source": "info_metrics", "chart": "normalized_line", "group": "subskills"},
        {"key": "model_delta_rmse", "label": "Model Delta", "color": "#0891b2", "source": "info_metrics", "chart": "normalized_line", "group": "world_model"},
        {"key": "model_replay_size", "label": "Synthetic", "color": "#f97316", "source": "info_metrics", "chart": "normalized_line", "group": "world_model"},
    ],
}

APPROACH_THRESHOLD = 0.065
LIFT_THRESHOLD = 0.055


class Agent(_MbpoAgent):
    def __init__(
        self,
        obs_dim_or_env: int | gym.Env[Any, Any],
        action_dim_or_device: int | torch.device | str,
        recipe: MbpoRecipe | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        if isinstance(obs_dim_or_env, int):
            obs_dim = int(obs_dim_or_env)
            action_dim = int(action_dim_or_device)
            action_high = np.ones(action_dim, dtype=np.float32)
            torch_device = torch.device("cpu" if device is None else device)
        else:
            env = obs_dim_or_env
            obs_dim = int(env.observation_space.shape[0])
            action_high = np.asarray(env.action_space.high, dtype=np.float32)
            torch_device = torch.device(action_dim_or_device)
        super().__init__(obs_dim, action_high, recipe or MbpoRecipe.from_candidate(get_candidate()), torch_device)
        self.observation_adapter = flatten_observation


def get_candidate() -> dict[str, Any]:
    return {
        "description": (
            "CleanRL-style SAC+MBPO Panda pick-and-place seed. It flattens the "
            "state/goal observation, uses train-only subskill shaping, learns an "
            "ensemble dynamics model online, and mixes uncertainty-filtered "
            "horizon-1 synthetic transitions from interaction-heavy replay states."
        ),
        "recipe": RECIPE,
        "diagnostic_series": DIAGNOSTIC_SERIES,
    }


def flatten_observation(obs: Any) -> np.ndarray:
    if isinstance(obs, dict):
        pieces = [np.asarray(obs[key], dtype=np.float32).reshape(-1) for key in ("observation", "achieved_goal", "desired_goal")]
        return np.concatenate(pieces).astype(np.float32, copy=False)
    return np.asarray(obs, dtype=np.float32).reshape(-1)


def flatten_observation_space(space: spaces.Space[Any]) -> spaces.Box:
    if not isinstance(space, spaces.Dict):
        assert isinstance(space, spaces.Box)
        return spaces.Box(
            low=np.asarray(space.low, dtype=np.float32).reshape(-1),
            high=np.asarray(space.high, dtype=np.float32).reshape(-1),
            dtype=np.float32,
        )
    lows = []
    highs = []
    for key in ("observation", "achieved_goal", "desired_goal"):
        subspace = space.spaces[key]
        assert isinstance(subspace, spaces.Box)
        lows.append(np.asarray(subspace.low, dtype=np.float32).reshape(-1))
        highs.append(np.asarray(subspace.high, dtype=np.float32).reshape(-1))
    return spaces.Box(low=np.concatenate(lows), high=np.concatenate(highs), dtype=np.float32)


def _safe_fraction(progress: float, initial_distance: float) -> float:
    if initial_distance <= 1e-6:
        return 0.0
    return float(np.clip(progress / initial_distance, -1.0, 1.0))


def _subskill_reward(raw_reward: float, info: dict[str, Any]) -> float:
    del raw_reward
    ee_to_cube = float(info.get("ee_to_cube_distance", 1.0))
    cube_to_goal = float(info.get("cube_to_goal_distance", 1.0))
    ee_progress = _safe_fraction(float(info.get("ee_to_cube_progress", 0.0)), float(info.get("initial_ee_to_cube_distance", 0.0)))
    goal_progress = _safe_fraction(float(info.get("cube_to_goal_progress", 0.0)), float(info.get("initial_cube_to_goal_distance", 0.0)))
    lift = float(info.get("cube_lift_height", 0.0))
    near = bool(info.get("near_cube", False))
    grasp = bool(info.get("gripper_closed_near_cube", False))
    lifted = bool(info.get("lifted", False))
    lifted_ever = bool(info.get("lifted_ever", lifted))
    placed = bool(info.get("placed_success", False))
    shaped = 0.65 * ee_progress + 0.18 * math.exp(-16.0 * ee_to_cube)
    shaped += 0.25 if near else 0.0
    shaped += 0.35 if grasp else 0.0
    shaped += 1.75 * min(1.0, lift / max(LIFT_THRESHOLD, 1e-6))
    shaped += 0.65 if lifted or lifted_ever else 0.0
    if lifted_ever:
        shaped += 1.25 * goal_progress
        shaped += 0.20 * math.exp(-14.0 * cube_to_goal)
    shaped += 3.0 if placed else 0.0
    return float(np.clip(shaped, -2.0, 5.0))


class RewardRecipeWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env[Any, Any], recipe: str | None = None) -> None:
        super().__init__(env)
        self.recipe = recipe or REWARD_RECIPE
        if self.recipe not in {"task_dense", "subskill_mbpo"}:
            raise ValueError(f"Unknown Panda MBPO reward recipe: {self.recipe}")
        self.observation_space = flatten_observation_space(env.observation_space)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        return flatten_observation(obs), info

    def step(self, action: np.ndarray):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        raw_reward = float(reward)
        shaped = raw_reward if self.recipe == "task_dense" else _subskill_reward(raw_reward, info)
        info["task_reward"] = raw_reward
        info["training_reward"] = shaped
        return flatten_observation(obs), shaped, terminated, truncated, info


def _priority(info: dict[str, Any], next_obs: np.ndarray) -> float:
    del next_obs
    if bool(info.get("near_cube", False)) or bool(info.get("gripper_closed_near_cube", False)):
        return 1.0
    if bool(info.get("lifted", False)) or bool(info.get("lifted_ever", False)) or bool(info.get("placed_success", False)):
        return 1.0
    return 1.0 if float(info.get("ee_to_cube_distance", 1.0)) < APPROACH_THRESHOLD else 0.0


def _record_extra(info: dict[str, Any]) -> dict[str, Any]:
    return {
        "ee_to_cube_distance": float(info.get("ee_to_cube_distance", 0.0)),
        "cube_lift_height": float(info.get("cube_lift_height", 0.0)),
        "cube_to_goal_distance": float(info.get("cube_to_goal_distance", 0.0)),
        "lifted_ever_rate": float(bool(info.get("lifted_ever", False))),
        "placed_success_rate": float(bool(info.get("placed_success", False))),
    }


def train_agent(
    benchmark: Any,
    env_factory: Any,
    candidate: dict[str, Any],
    device: torch.device,
    init_checkpoint: str | Any | None = None,
    live_callback: Any | None = None,
):
    return _train_mbpo_agent(
        benchmark=benchmark,
        env_factory=env_factory,
        candidate=candidate,
        device=device,
        init_checkpoint=init_checkpoint,
        live_callback=live_callback,
        priority_fn=_priority,
        record_extra_fn=_record_extra,
        observation_adapter=flatten_observation,
    )
