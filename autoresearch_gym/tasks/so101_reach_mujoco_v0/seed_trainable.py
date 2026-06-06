from __future__ import annotations

import math
import time
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from autoresearch_gym.runner.curves import elapsed_seconds_since, make_train_episode_record, scalar_info_metrics


EXP_NAME = "so101_mujoco_reach_sac_seed"
ALGORITHM = "sac"
CONTROL_TYPE = None
REWARD_RECIPE = "task_dense"

HIDDEN_DIMS = (256, 256)
REPLAY_SIZE = 200_000
BATCH_SIZE = 64
LEARNING_STARTS = 64
UPDATE_AFTER = 64
GRADIENT_STEPS = 1
GAMMA = 0.98
TAU = 0.005
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
ALPHA_LR = 3e-4
INIT_TEMPERATURE = 0.2
LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0
LIVE_CALLBACK_EVERY_STEPS = 20


def get_candidate() -> dict[str, Any]:
    return {
        "description": (
            "SO-101 MuJoCo reach baseline using a self-contained CleanRL-style SAC "
            "recipe over flattened proprioceptive and goal observations. This is "
            "the cold-start RL baseline for autoresearch; scripted controllers "
            "belong only in smoke tests or demonstrations."
        ),
        "recipe": {
            "algorithm": ALGORITHM,
            "reward_recipe": REWARD_RECIPE,
            "control": "normalized_position_targets",
            "hidden_dims": list(HIDDEN_DIMS),
            "batch_size": BATCH_SIZE,
            "replay_size": REPLAY_SIZE,
            "learning_starts": LEARNING_STARTS,
            "gradient_steps": GRADIENT_STEPS,
        },
    }


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
            raise ValueError(f"Unknown SO-101 reach reward recipe: {self.recipe}")
        self.observation_space = _flatten_observation_space(env.observation_space)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        return flatten_observation(obs), info

    def step(self, action: np.ndarray):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["training_reward"] = float(reward)
        return flatten_observation(obs), float(reward), terminated, truncated, info


class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, capacity: int) -> None:
        self.capacity = int(capacity)
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray, done: bool) -> None:
        self.obs[self.ptr] = flatten_observation(obs)
        self.actions[self.ptr] = np.asarray(action, dtype=np.float32).reshape(-1)
        self.rewards[self.ptr] = float(reward)
        self.next_obs[self.ptr] = flatten_observation(next_obs)
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, ...]:
        idx = np.random.randint(0, self.size, size=int(batch_size))
        return (
            torch.as_tensor(self.obs[idx], dtype=torch.float32, device=device),
            torch.as_tensor(self.actions[idx], dtype=torch.float32, device=device),
            torch.as_tensor(self.rewards[idx], dtype=torch.float32, device=device),
            torch.as_tensor(self.next_obs[idx], dtype=torch.float32, device=device),
            torch.as_tensor(self.dones[idx], dtype=torch.float32, device=device),
        )


def build_mlp(in_dim: int, hidden_dims: tuple[int, ...], out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = int(in_dim)
    for width in hidden_dims:
        layers.append(nn.Linear(last_dim, int(width)))
        layers.append(nn.ReLU())
        last_dim = int(width)
    layers.append(nn.Linear(last_dim, int(out_dim)))
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
        distribution = Normal(mean, log_std.exp())
        pre_tanh = distribution.rsample()
        squashed = torch.tanh(pre_tanh)
        action = squashed * self.action_scale
        log_prob = distribution.log_prob(pre_tanh) - torch.log(
            self.action_scale * (1.0 - squashed.pow(2)) + 1e-6
        )
        return action, log_prob.sum(dim=-1, keepdim=True)

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        mean, log_std = self(obs)
        squashed = torch.tanh(mean) if deterministic else torch.tanh(Normal(mean, log_std.exp()).sample())
        return squashed * self.action_scale


class Critic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: tuple[int, ...]) -> None:
        super().__init__()
        self.q = build_mlp(obs_dim + act_dim, hidden_dims, 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.q(torch.cat([obs, action], dim=-1))


class Agent:
    def __init__(self, env_or_obs_dim: Any, action_dim: int | None = None, device: torch.device | None = None) -> None:
        self.device = device or torch.device("cpu")
        if hasattr(env_or_obs_dim, "observation_space"):
            env = env_or_obs_dim
            obs_dim = int(np.prod(env.observation_space.shape))
            act_dim = int(np.prod(env.action_space.shape))
            action_high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
        else:
            obs_dim = int(env_or_obs_dim)
            act_dim = int(action_dim or 1)
            action_high = np.ones(act_dim, dtype=np.float32)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.batch_size = BATCH_SIZE
        self.gamma = GAMMA
        self.tau = TAU
        action_scale = torch.as_tensor(action_high, dtype=torch.float32, device=self.device)
        self.actor = SquashedGaussianActor(obs_dim, act_dim, HIDDEN_DIMS, action_scale).to(self.device)
        self.q1 = Critic(obs_dim, act_dim, HIDDEN_DIMS).to(self.device)
        self.q2 = Critic(obs_dim, act_dim, HIDDEN_DIMS).to(self.device)
        self.q1_target = Critic(obs_dim, act_dim, HIDDEN_DIMS).to(self.device)
        self.q2_target = Critic(obs_dim, act_dim, HIDDEN_DIMS).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=CRITIC_LR)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=CRITIC_LR)
        self.target_entropy = -float(act_dim)
        self.log_alpha = torch.tensor(math.log(INIT_TEMPERATURE), dtype=torch.float32, device=self.device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=ALPHA_LR)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def act(self, obs: Any, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.as_tensor(flatten_observation(obs), dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor.act(obs_t, deterministic=deterministic)
        return action.squeeze(0).cpu().numpy()

    def act_batch(self, obs: Any, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action = self.actor.act(obs_t, deterministic=deterministic)
        return action.cpu().numpy()

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
            "alpha_loss": float(alpha_loss.item()),
        }

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "algorithm": ALGORITHM,
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
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
        if state.get("algorithm") != ALGORITHM:
            raise ValueError(f"Checkpoint algorithm mismatch: {state.get('algorithm')}")
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


def save_agent_checkpoint(agent: Agent, checkpoint_path: str | Any, metadata: dict[str, Any] | None = None) -> None:
    torch.save({"agent_state": agent.checkpoint_state(), "metadata": metadata or {}}, checkpoint_path)


def load_agent_checkpoint(agent: Agent, checkpoint_path: str | Any) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location=agent.device)
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
    del candidate
    np.random.seed(int(benchmark.train_seed))
    torch.manual_seed(int(benchmark.train_seed))

    env = env_factory(control_type=CONTROL_TYPE, reward_recipe=REWARD_RECIPE)
    env.action_space.seed(int(benchmark.train_seed))
    agent = Agent(env, device=device)
    resumed_from = load_agent_checkpoint(agent, init_checkpoint) if init_checkpoint is not None else None
    replay = ReplayBuffer(agent.obs_dim, agent.act_dim, REPLAY_SIZE)

    total_steps = 0
    gradient_updates = 0
    last_metrics: dict[str, float] | None = None
    episode_records: list[dict[str, Any]] = []
    started_at = time.time()
    budget_seconds = getattr(benchmark, "train_seconds", None)
    deadline = started_at + float(budget_seconds) if budget_seconds is not None else None

    try:
        for episode in range(1, int(benchmark.train_episodes) + 1):
            if deadline is not None and time.time() >= deadline:
                break
            obs, info = env.reset(seed=int(benchmark.train_seed) + episode)
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
                if total_steps <= LEARNING_STARTS:
                    action = env.action_space.sample()
                else:
                    action = agent.act(obs, deterministic=False)
                next_obs, reward, terminated, truncated, info = env.step(action)
                replay.add(obs, action, float(reward), next_obs, bool(terminated))
                obs = next_obs
                episode_return += float(reward)
                episode_length += 1

                if total_steps >= UPDATE_AFTER and replay.size >= BATCH_SIZE:
                    for _ in range(GRADIENT_STEPS):
                        last_metrics = agent.update(replay)
                        gradient_updates += 1
                    if last_metrics is not None:
                        last_metrics = {**last_metrics, "gradient_updates": float(gradient_updates)}

                if live_callback is not None and (
                    total_steps == 1 or total_steps % LIVE_CALLBACK_EVERY_STEPS == 0
                ):
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
                    step=total_steps,
                    return_value=episode_return,
                    length=episode_length,
                    success=bool(info.get("is_success", False)),
                    elapsed_seconds=elapsed_seconds_since(started_at),
                    info_metrics=scalar_info_metrics(info),
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
    finally:
        env.close()

    wall_clock = time.time() - started_at
    stop_reason = (
        "time_budget_exhausted"
        if deadline is not None and time.time() >= deadline
        else "episode_cap_reached"
        if len(episode_records) >= int(benchmark.train_episodes)
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
        "total_steps": total_steps,
        "env_steps": total_steps,
        "completed_episodes": len(episode_records),
        "episode_batches": len(episode_records),
        "avg_return": float(np.mean(returns)) if returns else 0.0,
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "avg_length": float(np.mean([record["length"] for record in episode_records])) if episode_records else 0.0,
        "last_metrics": last_metrics,
        "gradient_updates": gradient_updates,
        "episode_records": episode_records,
        "wall_clock_seconds": wall_clock,
        "init_checkpoint": str(init_checkpoint) if init_checkpoint is not None else None,
        "resumed_from": resumed_from,
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
        low=np.concatenate(lows).astype(np.float32, copy=False),
        high=np.concatenate(highs).astype(np.float32, copy=False),
        dtype=np.float32,
    )
