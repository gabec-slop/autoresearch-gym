from __future__ import annotations

import math
import random
import time
from typing import Any

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gymnasium import spaces
from torch.distributions import Normal

from autoresearch_gym.runner.curves import elapsed_seconds_since, make_train_episode_record, scalar_info_metrics

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:  # pragma: no cover - keeps the seed runnable without tensorboard.
    class SummaryWriter:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def add_scalar(self, *args: Any, **kwargs: Any) -> None:
            pass

        def close(self) -> None:
            pass


gym.register_envs(gymnasium_robotics)

# Harness invariants:
# - Do not modify benchmark.json, eval seeds, or autoresearch_gym/runner/.
# - The runner owns fixed evaluation through evaluate_agent(); train_agent only trains.
# - Keep get_candidate(), RewardRecipeWrapper, train_agent(), and save_agent_checkpoint().
# - Freeform changes belong below this boundary: networks, losses, replay, schedules,
#   reward transforms, normalization, HER/relabeling experiments, logging, and exploration.

EXP_NAME = "fetch_push_dense_v0_cleanrl_sac_seed"
ALGORITHM = "sac"
CONTROL_TYPE = None
REWARD_RECIPE = "task_dense"

LEARNING_RATE = 3e-4
BUFFER_SIZE = 1_000_000
GAMMA = 0.98
TAU = 0.005
BATCH_SIZE = 256
LEARNING_STARTS = 1_000
POLICY_FREQUENCY = 2
TARGET_NETWORK_FREQUENCY = 1
ALPHA = 0.2
AUTOTUNE = True
HIDDEN_SIZE = 256
LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


def get_candidate() -> str:
    return (
        "CleanRL-style single-file SAC seed for FetchPushDense-v4. "
        "The candidate is the code in this file: goal observation flattening, "
        "actor/critic definitions, replay, losses, update cadence, logging, and "
        "future HER/relabeling experiments are intentionally expressed as mutable "
        "Python rather than as a parameter object. Evaluation is owned by the "
        "runner and should not be modified here."
    )


def layer_init(layer: nn.Linear, std: float = math.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def flatten_observation(obs: Any) -> np.ndarray:
    if isinstance(obs, dict):
        pieces = []
        for key in ("observation", "achieved_goal", "desired_goal"):
            if key in obs:
                pieces.append(np.asarray(obs[key], dtype=np.float32).reshape(-1))
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
    return spaces.Box(
        low=np.concatenate(lows).astype(np.float32, copy=False),
        high=np.concatenate(highs).astype(np.float32, copy=False),
        dtype=np.float32,
    )


class RewardRecipeWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env[Any, Any], recipe: str | None = None) -> None:
        super().__init__(env)
        self.recipe = recipe or REWARD_RECIPE
        if self.recipe != "task_dense":
            raise ValueError(f"Unknown FetchPushDense reward recipe: {self.recipe}")
        self.observation_space = flatten_observation_space(env.observation_space)

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
        self.observations = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.next_observations = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)
        self.pos = 0
        self.full = False

    @property
    def size(self) -> int:
        return self.capacity if self.full else self.pos

    def add(self, obs: np.ndarray, next_obs: np.ndarray, action: np.ndarray, reward: float, done: bool) -> None:
        self.observations[self.pos] = flatten_observation(obs)
        self.next_observations[self.pos] = flatten_observation(next_obs)
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = float(done)
        self.pos += 1
        if self.pos == self.capacity:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, ...]:
        batch_inds = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.observations[batch_inds], dtype=torch.float32, device=device),
            torch.as_tensor(self.next_observations[batch_inds], dtype=torch.float32, device=device),
            torch.as_tensor(self.actions[batch_inds], dtype=torch.float32, device=device),
            torch.as_tensor(self.rewards[batch_inds], dtype=torch.float32, device=device),
            torch.as_tensor(self.dones[batch_inds], dtype=torch.float32, device=device),
        )


class SoftQNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int) -> None:
        super().__init__()
        self.fc1 = layer_init(nn.Linear(obs_dim + act_dim, HIDDEN_SIZE))
        self.fc2 = layer_init(nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE))
        self.fc3 = layer_init(nn.Linear(HIDDEN_SIZE, 1), std=1.0)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Actor(nn.Module):
    def __init__(self, obs_dim: int, action_space: gym.Space[Any]) -> None:
        super().__init__()
        assert isinstance(action_space, gym.spaces.Box)
        act_dim = int(np.prod(action_space.shape))
        self.fc1 = layer_init(nn.Linear(obs_dim, HIDDEN_SIZE))
        self.fc2 = layer_init(nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE))
        self.fc_mean = layer_init(nn.Linear(HIDDEN_SIZE, act_dim), std=0.01)
        self.fc_logstd = layer_init(nn.Linear(HIDDEN_SIZE, act_dim), std=0.01)
        self.register_buffer("action_scale", torch.tensor((action_space.high - action_space.low) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor((action_space.high + action_space.low) / 2.0, dtype=torch.float32))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = torch.tanh(self.fc_logstd(x))
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1.0)
        return mean, log_std

    def get_action(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self(obs)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


class Agent:
    def __init__(self, env: gym.Env[Any, Any], device: torch.device) -> None:
        obs_dim = int(np.prod(env.observation_space.shape))
        act_dim = int(np.prod(env.action_space.shape))
        self.device = device
        self.actor = Actor(obs_dim, env.action_space).to(device)
        self.qf1 = SoftQNetwork(obs_dim, act_dim).to(device)
        self.qf2 = SoftQNetwork(obs_dim, act_dim).to(device)
        self.qf1_target = SoftQNetwork(obs_dim, act_dim).to(device)
        self.qf2_target = SoftQNetwork(obs_dim, act_dim).to(device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=LEARNING_RATE)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LEARNING_RATE)
        self.target_entropy = -float(act_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha = ALPHA
        self.a_optimizer = optim.Adam([self.log_alpha], lr=LEARNING_RATE)

    def act(self, obs: Any, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.as_tensor(flatten_observation(obs), dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, _, mean = self.actor.get_action(obs_t)
        selected = mean if deterministic else action
        return selected.squeeze(0).cpu().numpy()

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "algorithm": ALGORITHM,
            "actor": self.actor.state_dict(),
            "qf1": self.qf1.state_dict(),
            "qf2": self.qf2.state_dict(),
            "qf1_target": self.qf1_target.state_dict(),
            "qf2_target": self.qf2_target.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "a_optimizer": self.a_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "alpha": float(self.alpha),
        }

    def load_checkpoint_state(self, state: dict[str, Any]) -> None:
        if state.get("algorithm") != ALGORITHM:
            raise ValueError(f"Checkpoint algorithm mismatch: {state.get('algorithm')}")
        self.actor.load_state_dict(state["actor"])
        self.qf1.load_state_dict(state["qf1"])
        self.qf2.load_state_dict(state["qf2"])
        self.qf1_target.load_state_dict(state["qf1_target"])
        self.qf2_target.load_state_dict(state["qf2_target"])
        self.q_optimizer.load_state_dict(state["q_optimizer"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.a_optimizer.load_state_dict(state["a_optimizer"])
        with torch.no_grad():
            self.log_alpha.copy_(state["log_alpha"].to(self.device))
        self.alpha = float(state.get("alpha", self.log_alpha.exp().item()))


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
    random.seed(benchmark.train_seed)
    np.random.seed(benchmark.train_seed)
    torch.manual_seed(benchmark.train_seed)
    torch.backends.cudnn.deterministic = True

    env = env_factory(CONTROL_TYPE, REWARD_RECIPE)
    obs, _ = env.reset(seed=benchmark.train_seed)
    env.action_space.seed(benchmark.train_seed)
    agent = Agent(env, device)
    resumed_from = load_agent_checkpoint(agent, init_checkpoint) if init_checkpoint is not None else None
    rb = ReplayBuffer(env.observation_space.shape[0], env.action_space.shape[0], BUFFER_SIZE)
    writer = SummaryWriter(f"runs/{EXP_NAME}__{benchmark.train_seed}__{int(time.time())}")

    global_step = 0
    start_time = time.time()
    budget_seconds = getattr(benchmark, "train_seconds", None)
    deadline = start_time + float(budget_seconds) if budget_seconds is not None else None
    last_metrics: dict[str, float] | None = None
    episode_records: list[dict[str, Any]] = []

    for episode in range(1, benchmark.train_episodes + 1):
        if deadline is not None and time.time() >= deadline:
            break
        obs, info = env.reset(seed=benchmark.train_seed + episode)
        episodic_return = 0.0
        episodic_length = 0
        terminated = False
        truncated = False
        if live_callback is not None:
            live_callback(
                status="running",
                episode_records=episode_records,
                total_steps=global_step,
                last_metrics=last_metrics,
                env=env,
                current_episode=episode,
                episode_return=episodic_return,
                episode_length=episodic_length,
                agent=agent,
                elapsed_seconds=elapsed_seconds_since(start_time),
            )

        while not (terminated or truncated) and (deadline is None or time.time() < deadline):
            if global_step < LEARNING_STARTS:
                action = env.action_space.sample()
            else:
                action = agent.act(obs, deterministic=False)

            next_obs, reward, terminated, truncated, info = env.step(action)
            rb.add(obs, next_obs, action, float(reward), bool(terminated))
            obs = next_obs
            episodic_return += float(reward)
            episodic_length += 1
            global_step += 1

            if global_step > LEARNING_STARTS and rb.size >= BATCH_SIZE:
                observations, next_observations, actions, rewards, dones = rb.sample(BATCH_SIZE, device)
                with torch.no_grad():
                    next_state_actions, next_state_log_pi, _ = agent.actor.get_action(next_observations)
                    qf1_next_target = agent.qf1_target(next_observations, next_state_actions)
                    qf2_next_target = agent.qf2_target(next_observations, next_state_actions)
                    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - agent.alpha * next_state_log_pi
                    next_q_value = rewards.flatten() + (1 - dones.flatten()) * GAMMA * min_qf_next_target.view(-1)

                qf1_a_values = agent.qf1(observations, actions).view(-1)
                qf2_a_values = agent.qf2(observations, actions).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss

                agent.q_optimizer.zero_grad()
                qf_loss.backward()
                agent.q_optimizer.step()

                actor_loss_value = 0.0
                alpha_loss_value = 0.0
                if global_step % POLICY_FREQUENCY == 0:
                    for _ in range(POLICY_FREQUENCY):
                        pi, log_pi, _ = agent.actor.get_action(observations)
                        qf1_pi = agent.qf1(observations, pi)
                        qf2_pi = agent.qf2(observations, pi)
                        min_qf_pi = torch.min(qf1_pi, qf2_pi)
                        actor_loss = ((agent.alpha * log_pi) - min_qf_pi).mean()

                        agent.actor_optimizer.zero_grad()
                        actor_loss.backward()
                        agent.actor_optimizer.step()
                        actor_loss_value = float(actor_loss.item())

                        if AUTOTUNE:
                            with torch.no_grad():
                                _, log_pi, _ = agent.actor.get_action(observations)
                            alpha_loss = (-agent.log_alpha.exp() * (log_pi + agent.target_entropy)).mean()
                            agent.a_optimizer.zero_grad()
                            alpha_loss.backward()
                            agent.a_optimizer.step()
                            agent.alpha = float(agent.log_alpha.exp().item())
                            alpha_loss_value = float(alpha_loss.item())

                if global_step % TARGET_NETWORK_FREQUENCY == 0:
                    for param, target_param in zip(agent.qf1.parameters(), agent.qf1_target.parameters()):
                        target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
                    for param, target_param in zip(agent.qf2.parameters(), agent.qf2_target.parameters()):
                        target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

                last_metrics = {
                    "qf1_loss": float(qf1_loss.item()),
                    "qf2_loss": float(qf2_loss.item()),
                    "qf_loss": float(qf_loss.item() / 2.0),
                    "actor_loss": actor_loss_value,
                    "alpha": float(agent.alpha),
                    "alpha_loss": alpha_loss_value,
                }
                if global_step % 100 == 0:
                    writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                    writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                    writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                    writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                    writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                    writer.add_scalar("losses/actor_loss", actor_loss_value, global_step)
                    writer.add_scalar("losses/alpha", agent.alpha, global_step)
                    writer.add_scalar("charts/SPS", int(global_step / max(time.time() - start_time, 1e-6)), global_step)

            if live_callback is not None and (global_step == 1 or global_step % 25 == 0):
                live_callback(
                    status="running",
                    episode_records=episode_records,
                    total_steps=global_step,
                    last_metrics=last_metrics,
                    env=env,
                    current_episode=episode,
                    episode_return=episodic_return,
                    episode_length=episodic_length,
                agent=agent,
                elapsed_seconds=elapsed_seconds_since(start_time),
            )

        episode_record = make_train_episode_record(
            episode=episode,
            return_value=episodic_return,
            length=episodic_length,
            success=bool(info.get("is_success", False)),
            step=global_step,
            elapsed_seconds=elapsed_seconds_since(start_time),
            info_metrics=scalar_info_metrics(info),
        )
        episode_records.append(episode_record)
        writer.add_scalar("charts/episodic_return", episodic_return, global_step)
        writer.add_scalar("charts/episodic_length", episodic_length, global_step)
        writer.add_scalar("charts/success", 1.0 if episode_record["success"] else 0.0, global_step)
        if live_callback is not None:
            live_callback(
                status="running",
                episode_records=episode_records,
                total_steps=global_step,
                last_metrics=last_metrics,
                env=env,
                current_episode=episode,
                episode_return=episodic_return,
                episode_length=episodic_length,
                agent=agent,
                elapsed_seconds=elapsed_seconds_since(start_time),
            )
        if deadline is not None and time.time() >= deadline:
            break

    writer.close()
    env.close()
    wall_clock = time.time() - start_time
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
        "total_steps": global_step,
        "avg_return": float(np.mean([e["return"] for e in episode_records])) if episode_records else 0.0,
        "success_rate": float(np.mean([1.0 if e["success"] else 0.0 for e in episode_records])) if episode_records else 0.0,
        "avg_length": float(np.mean([e["length"] for e in episode_records])) if episode_records else 0.0,
        "last_metrics": last_metrics,
        "episode_records": episode_records,
        "wall_clock_seconds": wall_clock,
        "init_checkpoint": str(init_checkpoint) if init_checkpoint is not None else None,
        "resumed_from": resumed_from,
        "curriculum": None,
    }
