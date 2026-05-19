from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import pybullet
import torch
from torch import nn
from torch.distributions import Normal

from autoresearch_gym.runner.curves import elapsed_seconds_since, make_train_episode_record


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


@dataclass
class CandidateSpec:
    description: str
    control_type: str
    algorithm: str
    reward_recipe: str
    hidden_dims: tuple[int, ...]
    curriculum: dict[str, Any] | None
    hyperparameters: dict[str, float | int]


def get_candidate() -> CandidateSpec:
    return CandidateSpec(
        description="cold seed: SAC 512x512x512 with goal-delta-heavy reward.",
        control_type="joints",
        algorithm="sac",
        reward_recipe="goal_delta_heavy",
        hidden_dims=(512, 512, 512),
        curriculum=None,
        hyperparameters={
            "batch_size": 128,
            "replay_size": 100000,
            "gamma": 0.99,
            "tau": 0.005,
            "start_steps": 256,
            "update_after": 64,
            "gradient_steps": 1,
            "actor_lr": 3e-4,
            "critic_lr": 3e-4,
            "alpha_lr": 3e-4,
            "init_temperature": 0.1,
            "policy_noise": 0.2,
            "noise_clip": 0.5,
            "policy_delay": 2,
            "exploration_noise": 0.1,
        },
    )


class RewardRecipeWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env[np.ndarray, np.ndarray], recipe: str) -> None:
        super().__init__(env)
        self.recipe = recipe
        self._prev_ball_goal_distance = 0.0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._prev_ball_goal_distance = float(info.get("ball_goal_distance", 0.0))
        return obs, info

    def _compute_reward(
        self,
        base_reward: float,
        info: dict[str, Any],
        terminated: bool,
        truncated: bool,
    ) -> float:
        success = bool(info.get("is_success", False))
        contacted = bool(info.get("contacted_ball", False))
        first_contact = bool(info.get("first_contact", False))
        ball_goal_distance = float(info.get("ball_goal_distance", self._prev_ball_goal_distance))
        velocity_toward_goal = float(info.get("ball_velocity_toward_goal", 0.0))
        goal_delta = self._prev_ball_goal_distance - ball_goal_distance
        failed_terminal = terminated and not success

        if self.recipe == "task_dense":
            reward = float(base_reward)
        elif self.recipe == "success_only":
            reward = 1.0 if success else 0.0
        elif self.recipe == "contact_then_goal":
            reward = -0.005
            if first_contact:
                reward += 0.5
            if contacted:
                reward += 0.02
            if success:
                reward += 25.0
            if failed_terminal:
                reward -= 1.0
        elif self.recipe == "goal_delta_heavy":
            reward = -0.01
            if first_contact:
                reward += 0.35
            if contacted:
                reward += 0.25 * goal_delta
                reward += 0.10 * np.clip(velocity_toward_goal, -3.0, 3.0)
            if success:
                reward += 25.0
            if failed_terminal:
                reward -= 2.5
        elif self.recipe == "redirect_focus":
            reward = -0.01
            if contacted:
                reward += 0.18 * goal_delta
                reward += 0.08 * np.clip(velocity_toward_goal, -3.0, 3.0)
                if velocity_toward_goal > 0.0:
                    reward += 0.04 * min(velocity_toward_goal**2, 9.0)
            if first_contact:
                reward += 0.45
            if success:
                reward += 25.0
            if failed_terminal:
                reward -= 2.0
        else:
            raise ValueError(f"Unknown reward recipe: {self.recipe}")

        if truncated and not success:
            reward -= 0.25
        self._prev_ball_goal_distance = ball_goal_distance
        return float(reward)

    def step(self, action: np.ndarray):
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        shaped_reward = self._compute_reward(float(base_reward), info, terminated, truncated)
        info = dict(info)
        info["base_reward"] = float(base_reward)
        info["training_reward"] = shaped_reward
        return obs, shaped_reward, terminated, truncated, info


def curriculum_reset_options(
    curriculum: dict[str, Any] | None,
    episode: int,
    total_episodes: int,
) -> tuple[dict[str, Any] | None, str | None]:
    if not curriculum:
        return None, None

    stages = list(curriculum.get("stages", []))
    if not stages:
        return None, None

    progress = float(episode) / float(max(total_episodes, 1))
    chosen_stage = stages[-1]
    for stage in stages:
        until_episode = stage.get("until_episode")
        until_progress = stage.get("until_progress")
        if until_episode is not None and episode <= int(until_episode):
            chosen_stage = stage
            break
        if until_progress is not None and progress <= float(until_progress):
            chosen_stage = stage
            break

    options = dict(chosen_stage.get("env_options", {}) or {})
    if not options:
        return None, str(chosen_stage.get("name") or "curriculum-stage")
    return options, str(chosen_stage.get("name") or "curriculum-stage")


class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, capacity: int) -> None:
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray, done: bool) -> None:
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, ...]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.obs[idx], device=device),
            torch.as_tensor(self.actions[idx], device=device),
            torch.as_tensor(self.rewards[idx], device=device),
            torch.as_tensor(self.next_obs[idx], device=device),
            torch.as_tensor(self.dones[idx], device=device),
        )


def build_mlp(in_dim: int, hidden_dims: tuple[int, ...], out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = in_dim
    for width in hidden_dims:
        layers.append(nn.Linear(last, width))
        layers.append(nn.ReLU())
        last = width
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


class SquashedGaussianActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: tuple[int, ...], action_scale: torch.Tensor) -> None:
        super().__init__()
        trunk_out = hidden_dims[-1]
        self.backbone = build_mlp(obs_dim, hidden_dims[:-1], trunk_out) if len(hidden_dims) > 1 else nn.Sequential(
            nn.Linear(obs_dim, trunk_out),
            nn.ReLU(),
        )
        self.mean = nn.Linear(trunk_out, act_dim)
        self.log_std = nn.Linear(trunk_out, act_dim)
        self.register_buffer("action_scale", action_scale)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(obs)
        mean = self.mean(hidden)
        log_std = self.log_std(hidden)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1.0)
        return mean, log_std

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(obs)
        std = log_std.exp()
        dist = Normal(mean, std)
        pre_tanh = dist.rsample()
        squashed = torch.tanh(pre_tanh)
        action = squashed * self.action_scale
        log_prob = dist.log_prob(pre_tanh) - torch.log(self.action_scale * (1 - squashed.pow(2)) + 1e-6)
        return action, log_prob.sum(dim=-1, keepdim=True)

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        mean, log_std = self(obs)
        if deterministic:
            squashed = torch.tanh(mean)
        else:
            squashed = torch.tanh(Normal(mean, log_std.exp()).sample())
        return squashed * self.action_scale


class DeterministicActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: tuple[int, ...], action_scale: torch.Tensor) -> None:
        super().__init__()
        self.net = build_mlp(obs_dim, hidden_dims, act_dim)
        self.register_buffer("action_scale", action_scale)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(obs)) * self.action_scale


class Critic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: tuple[int, ...]) -> None:
        super().__init__()
        self.q = build_mlp(obs_dim + act_dim, hidden_dims, 1)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.q(torch.cat([obs, act], dim=-1))


class SACAgent:
    def __init__(self, obs_dim: int, act_high: np.ndarray, candidate: CandidateSpec, device: torch.device) -> None:
        hp = candidate.hyperparameters
        self.device = device
        self.act_dim = int(act_high.shape[0])
        self.batch_size = int(hp["batch_size"])
        self.gamma = float(hp["gamma"])
        self.tau = float(hp["tau"])

        action_scale = torch.as_tensor(act_high, dtype=torch.float32, device=device)
        self.actor = SquashedGaussianActor(obs_dim, self.act_dim, candidate.hidden_dims, action_scale).to(device)
        self.q1 = Critic(obs_dim, self.act_dim, candidate.hidden_dims).to(device)
        self.q2 = Critic(obs_dim, self.act_dim, candidate.hidden_dims).to(device)
        self.q1_target = Critic(obs_dim, self.act_dim, candidate.hidden_dims).to(device)
        self.q2_target = Critic(obs_dim, self.act_dim, candidate.hidden_dims).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=float(hp["actor_lr"]))
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=float(hp["critic_lr"]))
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=float(hp["critic_lr"]))

        self.target_entropy = -float(self.act_dim)
        self.log_alpha = torch.tensor(
            math.log(float(hp["init_temperature"])),
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=float(hp["alpha_lr"]))

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor.act(obs_t, deterministic=deterministic)
        return action.squeeze(0).cpu().numpy()

    def update(self, replay: ReplayBuffer) -> dict[str, float]:
        obs, act, rew, next_obs, done = replay.sample(self.batch_size, self.device)

        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_obs)
            target_q1 = self.q1_target(next_obs, next_action)
            target_q2 = self.q2_target(next_obs, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha.detach() * next_log_prob
            target = rew + (1.0 - done) * self.gamma * target_q

        q1_loss = nn.functional.mse_loss(self.q1(obs, act), target)
        q2_loss = nn.functional.mse_loss(self.q2(obs, act), target)

        self.q1_opt.zero_grad()
        q1_loss.backward()
        self.q1_opt.step()

        self.q2_opt.zero_grad()
        q2_loss.backward()
        self.q2_opt.step()

        sampled_action, log_prob = self.actor.sample(obs)
        q_pi = torch.min(self.q1(obs, sampled_action), self.q2(obs, sampled_action))
        actor_loss = (self.alpha.detach() * log_prob - q_pi).mean()

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
        }

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "algorithm": "sac",
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
        if state.get("algorithm") != "sac":
            raise ValueError("Checkpoint algorithm mismatch for SAC agent")
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


class TD3Agent:
    def __init__(self, obs_dim: int, act_high: np.ndarray, candidate: CandidateSpec, device: torch.device) -> None:
        hp = candidate.hyperparameters
        self.device = device
        self.act_dim = int(act_high.shape[0])
        self.batch_size = int(hp["batch_size"])
        self.gamma = float(hp["gamma"])
        self.tau = float(hp["tau"])
        self.policy_noise = float(hp["policy_noise"])
        self.noise_clip = float(hp["noise_clip"])
        self.policy_delay = int(hp["policy_delay"])
        self.exploration_noise = float(hp["exploration_noise"])
        self.update_count = 0

        action_scale = torch.as_tensor(act_high, dtype=torch.float32, device=device)
        self.max_action = action_scale
        self.actor = DeterministicActor(obs_dim, self.act_dim, candidate.hidden_dims, action_scale).to(device)
        self.actor_target = DeterministicActor(obs_dim, self.act_dim, candidate.hidden_dims, action_scale).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.q1 = Critic(obs_dim, self.act_dim, candidate.hidden_dims).to(device)
        self.q2 = Critic(obs_dim, self.act_dim, candidate.hidden_dims).to(device)
        self.q1_target = Critic(obs_dim, self.act_dim, candidate.hidden_dims).to(device)
        self.q2_target = Critic(obs_dim, self.act_dim, candidate.hidden_dims).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=float(hp["actor_lr"]))
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=float(hp["critic_lr"]))
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=float(hp["critic_lr"]))

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(obs_t).squeeze(0).cpu().numpy()
        if deterministic:
            return action
        noise = np.random.normal(0.0, self.exploration_noise, size=action.shape).astype(np.float32)
        return np.clip(action + noise, -self.max_action.cpu().numpy(), self.max_action.cpu().numpy())

    def update(self, replay: ReplayBuffer) -> dict[str, float]:
        self.update_count += 1
        obs, act, rew, next_obs, done = replay.sample(self.batch_size, self.device)

        with torch.no_grad():
            noise = (torch.randn_like(act) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_obs) + noise).clamp(-self.max_action, self.max_action)
            target_q1 = self.q1_target(next_obs, next_action)
            target_q2 = self.q2_target(next_obs, next_action)
            target_q = rew + (1.0 - done) * self.gamma * torch.min(target_q1, target_q2)

        q1_loss = nn.functional.mse_loss(self.q1(obs, act), target_q)
        q2_loss = nn.functional.mse_loss(self.q2(obs, act), target_q)

        self.q1_opt.zero_grad()
        q1_loss.backward()
        self.q1_opt.step()

        self.q2_opt.zero_grad()
        q2_loss.backward()
        self.q2_opt.step()

        actor_loss_value = 0.0
        if self.update_count % self.policy_delay == 0:
            actor_loss = -self.q1(obs, self.actor(obs)).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            actor_loss_value = float(actor_loss.item())

            with torch.no_grad():
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.mul_(1.0 - self.tau).add_(self.tau * param.data)
                for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
                    target_param.data.mul_(1.0 - self.tau).add_(self.tau * param.data)
                for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
                    target_param.data.mul_(1.0 - self.tau).add_(self.tau * param.data)

        return {
            "actor_loss": actor_loss_value,
            "critic_loss": float(((q1_loss + q2_loss) * 0.5).item()),
            "alpha": 0.0,
        }

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "algorithm": "td3",
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "q1_opt": self.q1_opt.state_dict(),
            "q2_opt": self.q2_opt.state_dict(),
            "update_count": self.update_count,
        }

    def load_checkpoint_state(self, state: dict[str, Any]) -> None:
        if state.get("algorithm") != "td3":
            raise ValueError("Checkpoint algorithm mismatch for TD3 agent")
        self.actor.load_state_dict(state["actor"])
        self.actor_target.load_state_dict(state["actor_target"])
        self.q1.load_state_dict(state["q1"])
        self.q2.load_state_dict(state["q2"])
        self.q1_target.load_state_dict(state["q1_target"])
        self.q2_target.load_state_dict(state["q2_target"])
        self.actor_opt.load_state_dict(state["actor_opt"])
        self.q1_opt.load_state_dict(state["q1_opt"])
        self.q2_opt.load_state_dict(state["q2_opt"])
        self.update_count = int(state.get("update_count", 0))


def build_agent(obs_dim: int, act_high: np.ndarray, candidate: CandidateSpec, device: torch.device):
    if candidate.algorithm == "sac":
        return SACAgent(obs_dim, act_high, candidate, device)
    if candidate.algorithm == "td3":
        return TD3Agent(obs_dim, act_high, candidate, device)
    raise ValueError(f"Unsupported algorithm: {candidate.algorithm}")


def save_agent_checkpoint(agent: Any, checkpoint_path: str | Any, metadata: dict[str, Any] | None = None) -> None:
    payload = {
        "agent_state": agent.checkpoint_state(),
        "metadata": metadata or {},
    }
    torch.save(payload, checkpoint_path)


def load_agent_checkpoint(agent: Any, checkpoint_path: str | Any) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location=agent.device)
    state = payload["agent_state"]
    agent.load_checkpoint_state(state)
    return payload.get("metadata", {})


def train_agent(
    benchmark: Any,
    env_factory: Any,
    candidate: CandidateSpec,
    device: torch.device,
    init_checkpoint: str | Any | None = None,
    live_callback: Any | None = None,
) -> tuple[Any, dict[str, Any]]:
    np.random.seed(benchmark.train_seed)
    torch.manual_seed(benchmark.train_seed)

    env = env_factory(candidate.control_type, candidate.reward_recipe)
    obs, _ = env.reset(seed=benchmark.train_seed)
    agent = build_agent(env.observation_space.shape[0], env.action_space.high, candidate, device)
    resumed_from: dict[str, Any] | None = None
    if init_checkpoint is not None:
        resumed_from = load_agent_checkpoint(agent, init_checkpoint)
    replay = ReplayBuffer(
        env.observation_space.shape[0],
        env.action_space.shape[0],
        int(candidate.hyperparameters["replay_size"]),
    )

    total_steps = 0
    last_metrics: dict[str, float] | None = None
    episode_records: list[dict[str, Any]] = []
    started_at = time.time()
    budget_seconds = getattr(benchmark, "train_seconds", None)
    deadline = started_at + float(budget_seconds) if budget_seconds is not None else None
    start_steps = int(candidate.hyperparameters["start_steps"])
    update_after = int(candidate.hyperparameters["update_after"])
    gradient_steps = int(candidate.hyperparameters["gradient_steps"])
    batch_size = int(candidate.hyperparameters["batch_size"])

    for episode in range(1, benchmark.train_episodes + 1):
        if deadline is not None and time.time() >= deadline:
            break
        reset_options, curriculum_stage = curriculum_reset_options(
            candidate.curriculum,
            episode,
            benchmark.train_episodes,
        )
        try:
            obs, info = env.reset(seed=benchmark.train_seed + episode, options=reset_options)
        except (pybullet.error, RuntimeError, TypeError, KeyError, ValueError):
            obs, info = env.reset()
        episode_return = 0.0
        episode_length = 0
        terminated = False
        truncated = False
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
            )

        while not (terminated or truncated) and (deadline is None or time.time() < deadline):
            total_steps += 1
            if total_steps <= start_steps:
                action = env.action_space.sample()
            else:
                action = agent.act(obs, deterministic=False)

            try:
                next_obs, reward, terminated, truncated, info = env.step(action)
            except (pybullet.error, RuntimeError, TypeError, KeyError, ValueError):
                terminated = True
                truncated = False
                reward = -3.0
                next_obs = obs
                info = {"is_success": False, "contacted_ball": False, "ball_goal_distance": 0.0}

            replay.add(obs, action, float(reward), next_obs, bool(terminated))
            obs = next_obs
            episode_return += float(reward)
            episode_length += 1

            if total_steps >= update_after and replay.size >= batch_size:
                for _ in range(gradient_steps):
                    last_metrics = agent.update(replay)

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
            )

        episode_records.append(
            make_train_episode_record(
                episode=episode,
                return_value=episode_return,
                length=episode_length,
                success=bool(info.get("is_success", False)),
                step=total_steps,
                elapsed_seconds=elapsed_seconds_since(started_at),
                contacted_ball=bool(info.get("contacted_ball", False)),
                ball_goal_distance=float(info.get("ball_goal_distance", 0.0)),
                curriculum_stage=curriculum_stage,
                reset_options=reset_options,
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
            )
        if deadline is not None and time.time() >= deadline:
            break

    env.close()
    wall_clock = time.time() - started_at
    if deadline is not None and time.time() >= deadline:
        stop_reason = "time_budget_exhausted"
    elif len(episode_records) >= benchmark.train_episodes:
        stop_reason = "episode_cap_reached"
    else:
        stop_reason = "loop_exited"
    return agent, {
        "episodes": benchmark.train_episodes,
        "episodes_completed": len(episode_records),
        "time_budget_seconds": float(budget_seconds) if budget_seconds is not None else None,
        "stop_reason": stop_reason,
        "total_steps": total_steps,
        "avg_return": float(np.mean([e["return"] for e in episode_records])) if episode_records else 0.0,
        "success_rate": float(np.mean([1.0 if e["success"] else 0.0 for e in episode_records])) if episode_records else 0.0,
        "contacted_ball_rate": float(np.mean([1.0 if e["contacted_ball"] else 0.0 for e in episode_records])) if episode_records else 0.0,
        "avg_length": float(np.mean([e["length"] for e in episode_records])) if episode_records else 0.0,
        "last_metrics": last_metrics,
        "episode_records": episode_records,
        "wall_clock_seconds": wall_clock,
        "init_checkpoint": str(init_checkpoint) if init_checkpoint is not None else None,
        "resumed_from": resumed_from,
        "curriculum": candidate.curriculum,
    }
