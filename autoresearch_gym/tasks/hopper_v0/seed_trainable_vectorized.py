from __future__ import annotations

import math
import random
import time
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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


# Harness invariants:
# - Do not modify benchmark.json, eval seeds, or autoresearch_gym/runner/.
# - The runner owns fixed evaluation through evaluate_agent(); train_agent only trains.
# - Keep get_candidate(), RewardRecipeWrapper, train_agent(), and save_agent_checkpoint().
# - Freeform changes belong below this boundary: networks, losses, replay, schedules,
#   reward transforms, normalization, logging, and exploration.

EXP_NAME = "hopper_v0_vectorized_sac_seed"
ALGORITHM = "sac"
CONTROL_TYPE = None
REWARD_RECIPE = "task_reward"

# Mutable training-recipe constants. These are intentionally freeform code, not
# a structured config object. The autoresearch agent may edit them to trade off
# data collection, learner updates, GPU utilization, and policy quality inside
# the fixed benchmark time window.
LEARNING_RATE = 3e-4
BUFFER_SIZE = 1_000_000
GAMMA = 0.99
TAU = 0.005
BATCH_SIZE = 1_024
LEARNING_STARTS = 1_000
POLICY_FREQUENCY = 2
TARGET_NETWORK_FREQUENCY = 1
NUM_ENVS = 16
GRADIENT_UPDATES_PER_VECTOR_STEP = 16
VECTOR_ENV_MODE = "async"
ASYNC_CONTEXT = "fork"
RENDER_SIDECAR_ENABLED = True
LIVE_CALLBACK_EVERY_STEPS = 200
MAX_ENV_STEPS_SAFETY_CAP = 1_000_000
NOISE_CLIP = 0.5
ALPHA = 0.2
AUTOTUNE = True
HIDDEN_SIZE = 256
LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


def get_candidate() -> str:
    return (
        "Vectorized Hopper SAC seed for fixed-wall-clock autoresearch. Uses a "
        "canonical 256x256 actor/critic MLP, headless vectorized training envs, "
        "and a single optional rgb_array sidecar env for dashboard frames. The "
        "outer-loop agent is expected to mutate NUM_ENVS, BATCH_SIZE, update "
        "cadence, replay, and rendering cadence in code to improve learning and "
        "hardware utilization under the benchmark time budget."
    )


def layer_init(layer: nn.Linear, std: float = math.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def flatten_observation(obs: Any) -> np.ndarray:
    return np.asarray(obs, dtype=np.float32).reshape(-1)


class RewardRecipeWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env[Any, Any], recipe: str | None = None) -> None:
        super().__init__(env)
        self.recipe = recipe or REWARD_RECIPE
        if self.recipe != "task_reward":
            raise ValueError(f"Unknown Hopper reward recipe: {self.recipe}")

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

    def act_batch(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action, _, mean = self.actor.get_action(obs_t)
        selected = mean if deterministic else action
        return selected.cpu().numpy()

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

    probe_env = env_factory(CONTROL_TYPE, REWARD_RECIPE)
    render_env = probe_env if RENDER_SIDECAR_ENABLED else None
    render_obs = None
    if render_env is not None:
        render_obs, _ = render_env.reset(seed=benchmark.train_seed + 900_000)
        render_env.action_space.seed(benchmark.train_seed + 900_000)
    agent = Agent(probe_env, device)
    resumed_from = load_agent_checkpoint(agent, init_checkpoint) if init_checkpoint is not None else None
    rb = ReplayBuffer(probe_env.observation_space.shape[0], probe_env.action_space.shape[0], BUFFER_SIZE)
    writer = SummaryWriter(f"runs/{EXP_NAME}__{benchmark.train_seed}__{int(time.time())}")

    def make_headless_env(seed_offset: int):
        def thunk():
            env_kwargs = dict(getattr(benchmark, "env_kwargs", {}))
            env_kwargs["render_mode"] = None
            env = gym.make(benchmark.env_id, **env_kwargs)
            wrapped = RewardRecipeWrapper(env, REWARD_RECIPE)
            wrapped.action_space.seed(benchmark.train_seed + seed_offset)
            return wrapped

        return thunk

    env_fns = [make_headless_env(idx) for idx in range(NUM_ENVS)]
    vector_backend = VECTOR_ENV_MODE
    try:
        if VECTOR_ENV_MODE == "async":
            vector_backend = f"async-{ASYNC_CONTEXT or 'default'}"
            envs = gym.vector.AsyncVectorEnv(env_fns, context=ASYNC_CONTEXT)
        else:
            envs = gym.vector.SyncVectorEnv(env_fns)
    except Exception:
        vector_backend = "sync-fallback"
        envs = gym.vector.SyncVectorEnv(env_fns)

    obs, _ = envs.reset(seed=[benchmark.train_seed + idx for idx in range(NUM_ENVS)])

    global_step = 0
    update_step = 0
    start_time = time.time()
    budget_seconds = getattr(benchmark, "train_seconds", None)
    deadline = start_time + float(budget_seconds) if budget_seconds is not None else None
    last_metrics: dict[str, float] | None = None
    episode_records: list[dict[str, Any]] = []
    live_step = 0
    active_returns = np.zeros(NUM_ENVS, dtype=np.float64)
    active_lengths = np.zeros(NUM_ENVS, dtype=np.int64)
    render_episode_return = 0.0
    render_episode_length = 0

    def info_for_env(infos: Any, index: int) -> dict[str, Any]:
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
                if isinstance(value, np.ndarray) and len(value) == NUM_ENVS:
                    env_info[key] = value[index].item() if hasattr(value[index], "item") else value[index]
                elif isinstance(value, (list, tuple)) and len(value) == NUM_ENVS:
                    env_info[key] = value[index]
            except TypeError:
                continue
        return env_info

    def advance_render_env() -> tuple[float, int]:
        nonlocal render_obs, render_episode_return, render_episode_length
        if render_env is None or render_obs is None:
            return render_episode_return, render_episode_length
        if global_step < LEARNING_STARTS:
            render_action = render_env.action_space.sample()
        else:
            render_action = agent.act(render_obs, deterministic=True)
        render_obs, render_reward, render_terminated, render_truncated, _ = render_env.step(render_action)
        render_episode_return += float(render_reward)
        render_episode_length += 1
        if render_terminated or render_truncated:
            finished_return = render_episode_return
            finished_length = render_episode_length
            render_obs, _ = render_env.reset(seed=benchmark.train_seed + 900_000 + len(episode_records))
            render_episode_return = 0.0
            render_episode_length = 0
            return finished_return, finished_length
        return render_episode_return, render_episode_length

    def should_continue_training() -> bool:
        if len(episode_records) >= benchmark.train_episodes:
            return False
        if global_step >= MAX_ENV_STEPS_SAFETY_CAP:
            return False
        if deadline is not None and time.time() >= deadline:
            return False
        return True

    def update_sac() -> dict[str, float]:
        nonlocal update_step
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
        update_step += 1
        if update_step % POLICY_FREQUENCY == 0:
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

        if update_step % TARGET_NETWORK_FREQUENCY == 0:
            for param, target_param in zip(agent.qf1.parameters(), agent.qf1_target.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
            for param, target_param in zip(agent.qf2.parameters(), agent.qf2_target.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

        metrics = {
            "qf1_loss": float(qf1_loss.item()),
            "qf2_loss": float(qf2_loss.item()),
            "qf_loss": float(qf_loss.item() / 2.0),
            "actor_loss": actor_loss_value,
            "alpha": float(agent.alpha),
            "alpha_loss": alpha_loss_value,
            "gradient_updates": float(update_step),
            "num_envs": float(NUM_ENVS),
        }
        if update_step % 100 == 0:
            writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
            writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
            writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
            writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
            writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
            writer.add_scalar("losses/actor_loss", actor_loss_value, global_step)
            writer.add_scalar("losses/alpha", agent.alpha, global_step)
            writer.add_scalar("charts/SPS", int(global_step / max(time.time() - start_time, 1e-6)), global_step)
        return metrics

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
            elapsed_seconds=elapsed_seconds_since(start_time),
        )

    while should_continue_training():
        if global_step < LEARNING_STARTS:
            action = np.stack([envs.single_action_space.sample() for _ in range(NUM_ENVS)]).astype(np.float32)
        else:
            action = agent.act_batch(obs, deterministic=False)

        next_obs, rewards, terminations, truncations, infos = envs.step(action)
        dones = np.logical_or(terminations, truncations)
        for env_index in range(NUM_ENVS):
            rb.add(obs[env_index], next_obs[env_index], action[env_index], float(rewards[env_index]), bool(terminations[env_index]))
        obs = next_obs
        active_returns += rewards.astype(np.float64)
        active_lengths += 1
        global_step += NUM_ENVS
        sidecar_return, sidecar_length = advance_render_env()

        for env_index in np.flatnonzero(dones):
            info = info_for_env(infos, int(env_index))
            episode_record = make_train_episode_record(
                episode=len(episode_records) + 1,
                return_value=float(active_returns[env_index]),
                length=int(active_lengths[env_index]),
                success=bool(info.get("is_success", False)),
                step=global_step,
                elapsed_seconds=elapsed_seconds_since(start_time),
                info_metrics=scalar_info_metrics(info),
                env_index=int(env_index),
            )
            episode_records.append(episode_record)
            writer.add_scalar("charts/episodic_return", episode_record["return"], global_step)
            writer.add_scalar("charts/episodic_length", episode_record["length"], global_step)
            active_returns[env_index] = 0.0
            active_lengths[env_index] = 0
            if len(episode_records) >= benchmark.train_episodes:
                break

        if global_step > LEARNING_STARTS and rb.size >= BATCH_SIZE:
            for _ in range(GRADIENT_UPDATES_PER_VECTOR_STEP):
                if deadline is not None and time.time() >= deadline:
                    break
                last_metrics = update_sac()

        if live_callback is not None and (global_step == NUM_ENVS or global_step - live_step >= LIVE_CALLBACK_EVERY_STEPS):
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
                elapsed_seconds=elapsed_seconds_since(start_time),
            )

    writer.close()
    envs.close()
    if render_env is not None:
        render_env.close()
    elif probe_env is not None:
        probe_env.close()
    wall_clock = time.time() - start_time
    if len(episode_records) >= benchmark.train_episodes:
        stop_reason = "episode_cap_reached"
    elif global_step >= MAX_ENV_STEPS_SAFETY_CAP:
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
        "total_steps": global_step,
        "avg_return": float(np.mean([e["return"] for e in episode_records])) if episode_records else 0.0,
        "success_rate": float(np.mean([1.0 if e["success"] else 0.0 for e in episode_records])) if episode_records else 0.0,
        "avg_length": float(np.mean([e["length"] for e in episode_records])) if episode_records else 0.0,
        "last_metrics": last_metrics,
        "episode_records": episode_records,
        "wall_clock_seconds": wall_clock,
        "vector_envs": NUM_ENVS,
        "vector_backend": vector_backend,
        "gradient_updates": update_step,
        "visual_sampling": (
            "single rgb_array sidecar env stepped by current policy"
            if RENDER_SIDECAR_ENABLED
            else "disabled"
        ),
        "init_checkpoint": str(init_checkpoint) if init_checkpoint is not None else None,
        "resumed_from": resumed_from,
        "curriculum": None,
    }
