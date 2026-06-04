from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import spaces
from torch.distributions import Normal

from autoresearch_gym.runner.curves import elapsed_seconds_since, make_train_collection_window_record


# SB3/RL-Zoo PandaPickAndPlace recipe ported to this repo's MuJoCo/MJWarp
# Menagerie Panda task. The task contract intentionally follows panda-gym:
# dense reward is -distance(achieved_goal, desired_goal), and success is
# distance < 0.05 with no lift gate.
EXP_NAME = "panda_gym_dense_pick_and_place_mjwarp_tqc_her_ee_guided_warmup"
ALGORITHM = "tqc_her"
CONTROL_TYPE = None
REWARD_RECIPE = "panda_gym_dense_tqc_her_ee_guided_warmup"
RECIPE = {
    "algorithm": ALGORITHM,
    "reward_recipe": REWARD_RECIPE,
    "policy": "end_effector_delta_tool",
    "network": {
        "net_arch": [512, 512, 512],
        "n_critics": 2,
        "n_quantiles": 25,
    },
    "replay": {
        "buffer_size": 1_000_000,
        "batch_size": 2048,
        "her_goal_selection_strategy": "future",
        "n_sampled_goal": 4,
        "future_scope": "same_env_same_episode",
        "reward_contract": "panda_gym_dense_distance_plus_training_only_approach_and_guided_warmup",
    },
    "runner": {
        "sample_trajectory_source": "candidate_provided",
    },
}

LEARNING_RATE = 1.0e-3
BUFFER_SIZE = 1_000_000
BATCH_SIZE = 2048
LEARNING_STARTS = 10_000
SCRIPTED_WARMUP_STEPS = 200_000
SCRIPTED_WARMUP_FRACTION = 0.30
SCRIPTED_WARMUP_NOISE = 0.0
GAMMA = 0.95
TAU = 0.05
POLICY_FREQUENCY = 2
TARGET_NETWORK_FREQUENCY = 1
N_CRITICS = 2
N_QUANTILES = 25
TOP_QUANTILES_TO_DROP_PER_NET = 2
HIDDEN_SIZE = 512
LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0
HER_RELABEL_FRACTION = 0.8
SUCCESS_THRESHOLD = 0.05
APPROACH_REWARD_WEIGHT = 0.35
NEAR_CUBE_REWARD = 0.05
NEAR_CUBE_THRESHOLD = 0.055
EE_ACTION_SCALE = 0.080
IK_DAMPING = 1.0e-4
GRIPPER_CLOSE_SIGN = -1.0
GRIPPER_WIDTH_DELTA_SCALE = 1.0
PINCH_OFFSET_FROM_HAND = np.asarray([-0.006, -0.0005, -0.058], dtype=np.float32)
SCRIPTED_PHASES = ("hover_cube", "descend_cube", "close", "lift", "hover_goal", "descend_goal", "open")
SCRIPTED_CONFIG = {
    "hover_z": 0.105,
    "grasp_z": 0.045,
    "lift_z": 0.160,
    "place_clearance": 0.050,
    "close_steps": 55,
    "open_steps": 10,
    "phase_max_steps": {
        "hover_cube": 25,
        "descend_cube": 55,
        "close": 55,
        "lift": 35,
        "hover_goal": 200,
        "descend_goal": 20,
        "open": 10,
    },
    "phase_min_steps": {
        "hover_cube": 15,
        "descend_cube": 45,
        "close": 55,
        "lift": 8,
        "hover_goal": 8,
        "descend_goal": 3,
        "open": 10,
    },
    "x_offset": 0.000,
    "y_offset": 0.000,
    "gain": 1.0,
    "tolerance": 0.025,
}


DIAGNOSTIC_SERIES = {
    "series": [
        {"key": "ee_to_cube_distance", "label": "EE to Cube", "color": "#2563eb", "source": "info_metrics", "chart": "normalized_line", "group": "subskills"},
        {"key": "ee_to_cube_progress", "label": "EE Progress", "color": "#0891b2", "source": "info_metrics", "chart": "normalized_line", "group": "subskills"},
        {"key": "cube_lift_height", "label": "Lift", "color": "#16a34a", "source": "info_metrics", "chart": "normalized_line", "group": "subskills"},
        {"key": "lifted_ever_rate", "label": "Lifted Ever", "color": "#15803d", "source": "info_metrics", "chart": "normalized_line", "group": "subskills"},
        {"key": "cube_to_goal_distance", "label": "Cube to Goal", "color": "#dc2626", "source": "info_metrics", "chart": "normalized_line", "group": "subskills"},
        {"key": "cube_at_goal_rate", "label": "At Goal", "color": "#f97316", "source": "info_metrics", "chart": "normalized_line", "group": "subskills"},
        {"key": "placed_success_rate", "label": "Placed", "color": "#7c3aed", "source": "info_metrics", "chart": "normalized_line", "group": "subskills"},
    ]
}


def get_candidate() -> dict[str, Any]:
    return {
        "description": (
            "Panda-gym dense MuJoCo/MJWarp TQC+HER seed with a guided warmup "
            "collector. For the first 30% of a wall-clock run, or the first "
            "200k env steps for episode-budget runs, collection follows a "
            "scripted hover, descend, close, lift, carry, descend, and release "
            "controller using FK pinch-point feedback. After warmup, TQC+HER "
            "takes over while the replay buffer retains the contact-rich "
            "guided transitions."
        ),
        "recipe": RECIPE,
    }


def _diagnostic_series() -> dict[str, Any]:
    return DIAGNOSTIC_SERIES


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
    """Flatten the raw MJWarp task while keeping the trainable recipe explicit."""

    def __init__(self, env: gym.Env[Any, Any], recipe: str | None = None) -> None:
        super().__init__(env)
        self.recipe = recipe or REWARD_RECIPE
        if self.recipe != REWARD_RECIPE:
            raise ValueError(f"Unknown Panda MJWarp TQC/HER recipe: {self.recipe}")
        self.raw_observation_space = env.observation_space
        self.observation_space = flatten_observation_space(env.observation_space)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        return flatten_observation(obs), info

    def step(self, action: np.ndarray):
        tool = EndEffectorDeltaTool.from_env(self.env.unwrapped)
        raw_action = tool.single_action(flatten_observation(self.env.unwrapped._get_obs()), action)
        obs, reward, terminated, truncated, info = self.env.step(raw_action)
        info = dict(info)
        info["tool_action_dx"] = float(np.asarray(action, dtype=np.float32)[0])
        info["tool_action_dy"] = float(np.asarray(action, dtype=np.float32)[1])
        info["tool_action_dz"] = float(np.asarray(action, dtype=np.float32)[2])
        info["tool_action_gripper"] = float(np.asarray(action, dtype=np.float32)[3])
        info["training_reward"] = float(reward)
        return flatten_observation(obs), float(reward), terminated, truncated, info


class EndEffectorDeltaTool:
    """Map 4D EE delta actions into Menagerie Panda actuator commands.

    The learned action is `[dx, dy, dz, gripper]` in `[-1, 1]`. The first three
    components are interpreted as small end-effector position deltas. A damped
    least-squares Jacobian solve produces seven joint position-control targets;
    the final component uses the tool convention `+1 = close`; the Menagerie
    tendon actuator uses the opposite normalized sign, so it is inverted before
    stepping the raw environment.
    """

    def __init__(self, model: Any, mujoco: Any, ctrl_low: np.ndarray, ctrl_high: np.ndarray, robot_qpos_adrs: np.ndarray, cube_qpos_adr: int, ee_site_id: int, ee_body_id: int, home_qpos: np.ndarray, env_ref: Any | None = None) -> None:
        self.model = model
        self.mujoco = mujoco
        self.env_ref = env_ref
        self.data = mujoco.MjData(model)
        self.ctrl_low = np.asarray(ctrl_low, dtype=np.float32)
        self.ctrl_high = np.asarray(ctrl_high, dtype=np.float32)
        self.robot_qpos_adrs = np.asarray(robot_qpos_adrs[:7], dtype=np.int32)
        self.robot_dof_adrs = self.robot_qpos_adrs.copy()
        self.cube_qpos_adr = int(cube_qpos_adr)
        self.ee_site_id = int(ee_site_id)
        self.ee_body_id = int(ee_body_id)
        self.left_finger_body_id = int(self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, "left_finger"))
        self.right_finger_body_id = int(self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, "right_finger"))
        self.home_qpos = np.asarray(home_qpos, dtype=np.float64).copy()
        self.nv = int(model.nv)

    @classmethod
    def from_env(cls, env: Any) -> "EndEffectorDeltaTool":
        return cls(
            model=env.model,
            mujoco=env.mujoco,
            ctrl_low=env.ctrl_low,
            ctrl_high=env.ctrl_high,
            robot_qpos_adrs=env.robot_qpos_adrs,
            cube_qpos_adr=env.cube_qpos_adr,
            ee_site_id=getattr(env, "ee_site_id", -1),
            ee_body_id=getattr(env, "ee_body_id", -1),
            home_qpos=env.home_qpos,
            env_ref=env,
        )

    def batch_actions(self, observations: np.ndarray, tool_actions: np.ndarray) -> np.ndarray:
        observations = np.asarray(observations, dtype=np.float32)
        tool_actions = np.asarray(tool_actions, dtype=np.float32)
        qpos_batch = self._batch_qpos(observations)
        raw = np.zeros((observations.shape[0], self.ctrl_low.shape[0]), dtype=np.float32)
        for idx in range(observations.shape[0]):
            raw[idx] = self.single_action(observations[idx], tool_actions[idx], qpos=qpos_batch[idx])
        return raw

    def batch_control_positions(self, observations: np.ndarray) -> np.ndarray:
        observations = np.asarray(observations, dtype=np.float32)
        qpos_batch = self._batch_qpos(observations)
        control = np.zeros((observations.shape[0], 3), dtype=np.float32)
        for idx, qpos in enumerate(qpos_batch):
            self.data.qpos[:] = qpos
            self.data.qvel[:] = 0.0
            self.mujoco.mj_forward(self.model, self.data)
            control[idx] = self._current_control_position()
        return control

    def _batch_qpos(self, observations: np.ndarray) -> np.ndarray:
        env = self.env_ref
        if env is not None and hasattr(env, "warp_data"):
            try:
                qpos = np.asarray(env.warp_data.qpos.numpy(), dtype=np.float64)
                if qpos.ndim == 2 and qpos.shape[0] == observations.shape[0]:
                    return qpos.copy()
            except Exception:
                pass
        return np.stack([self._qpos_from_observation(obs) for obs in observations], axis=0)

    def _qpos_from_observation(self, observation: np.ndarray) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32).reshape(-1)
        qpos = self.home_qpos.copy()
        qpos[self.robot_qpos_adrs] = obs[15 : 15 + len(self.robot_qpos_adrs)]
        qpos[self.cube_qpos_adr : self.cube_qpos_adr + 3] = obs[3:6]
        qpos[self.cube_qpos_adr + 3 : self.cube_qpos_adr + 7] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return qpos

    def _current_control_position(self) -> np.ndarray:
        if self.left_finger_body_id >= 0 and self.right_finger_body_id >= 0:
            return (np.asarray(self.data.xpos[self.left_finger_body_id], dtype=np.float32) + np.asarray(self.data.xpos[self.right_finger_body_id], dtype=np.float32)) * 0.5
        if self.ee_site_id >= 0:
            return np.asarray(self.data.site_xpos[self.ee_site_id], dtype=np.float32)
        if self.ee_body_id >= 0:
            return np.asarray(self.data.xpos[self.ee_body_id], dtype=np.float32)
        return np.asarray(self.data.xpos[-1], dtype=np.float32)

    def single_action(self, observation: np.ndarray, tool_action: np.ndarray, qpos: np.ndarray | None = None) -> np.ndarray:
        action = np.clip(np.asarray(tool_action, dtype=np.float32).reshape(-1), -1.0, 1.0)
        if qpos is None:
            qpos = self._qpos_from_observation(observation)
        else:
            qpos = np.asarray(qpos, dtype=np.float64).copy()
        self.data.qpos[:] = qpos
        self.data.qvel[:] = 0.0
        self.mujoco.mj_forward(self.model, self.data)
        jacp = np.zeros((3, self.nv), dtype=np.float64)
        jacr = np.zeros((3, self.nv), dtype=np.float64)
        if self.left_finger_body_id >= 0 and self.right_finger_body_id >= 0:
            left_jacp = np.zeros((3, self.nv), dtype=np.float64)
            right_jacp = np.zeros((3, self.nv), dtype=np.float64)
            self.mujoco.mj_jacBody(self.model, self.data, left_jacp, jacr, self.left_finger_body_id)
            self.mujoco.mj_jacBody(self.model, self.data, right_jacp, jacr, self.right_finger_body_id)
            jacp = 0.5 * (left_jacp + right_jacp)
        elif self.ee_site_id >= 0:
            self.mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)
        elif self.ee_body_id >= 0:
            self.mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.ee_body_id)
        j = jacp[:, self.robot_dof_adrs]
        delta = (action[:3] * EE_ACTION_SCALE).astype(np.float64)
        lhs = j @ j.T + IK_DAMPING * np.eye(3, dtype=np.float64)
        dq = j.T @ np.linalg.solve(lhs, delta)
        target = qpos[self.robot_qpos_adrs] + dq
        raw = np.zeros_like(self.ctrl_low, dtype=np.float32)
        for i, q in enumerate(target[:7]):
            low = float(self.ctrl_low[i])
            high = float(self.ctrl_high[i])
            raw[i] = np.clip(2.0 * (float(q) - low) / max(high - low, 1e-6) - 1.0, -1.0, 1.0)
        if raw.shape[0] > 7:
            # Panda-gym applies action[-1] * 0.2 to the current finger width.
            # Menagerie's tendon actuator is normalized and has the opposite
            # close sign, so keep the small delta but adapt the sign.
            raw[7] = float(np.clip(GRIPPER_CLOSE_SIGN * GRIPPER_WIDTH_DELTA_SCALE * action[3], -1.0, 1.0))
        return raw


class HerReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, capacity: int, relabel_fraction: float) -> None:
        self.capacity = int(capacity)
        self.observations = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.next_observations = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)
        self.env_ids = np.full(self.capacity, -1, dtype=np.int32)
        self.episode_ids = np.full(self.capacity, -1, dtype=np.int64)
        self.episode_steps = np.zeros(self.capacity, dtype=np.int32)
        self.next_lifted_ever = np.zeros(self.capacity, dtype=bool)
        self.completed = np.zeros(self.capacity, dtype=bool)
        self.episode_indices: dict[tuple[int, int], list[int]] = {}
        self.relabel_fraction = float(relabel_fraction)
        self.pos = 0
        self.full = False

    @property
    def size(self) -> int:
        return self.capacity if self.full else self.pos

    @property
    def sample_size(self) -> int:
        return int(self.completed[: self.size].sum()) if not self.full else int(self.completed.sum())

    def add_batch(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        *,
        env_ids: np.ndarray,
        episode_ids: np.ndarray,
        episode_steps: np.ndarray,
        next_lifted_ever: np.ndarray,
    ) -> None:
        for idx in range(obs.shape[0]):
            if self.full:
                self._remove_episode_for_index(self.pos)
            self.observations[self.pos] = obs[idx]
            self.next_observations[self.pos] = next_obs[idx]
            self.actions[self.pos] = actions[idx]
            self.rewards[self.pos, 0] = rewards[idx]
            self.dones[self.pos, 0] = float(dones[idx])
            self.env_ids[self.pos] = int(env_ids[idx])
            self.episode_ids[self.pos] = int(episode_ids[idx])
            self.episode_steps[self.pos] = int(episode_steps[idx])
            self.next_lifted_ever[self.pos] = bool(next_lifted_ever[idx])
            key = (int(env_ids[idx]), int(episode_ids[idx]))
            self.episode_indices.setdefault(key, []).append(self.pos)
            self.completed[self.pos] = False
            if bool(dones[idx]):
                self._mark_episode_complete(key)
            self.pos += 1
            if self.pos == self.capacity:
                self.full = True
                self.pos = 0

    def _remove_episode_for_index(self, index: int) -> None:
        key = (int(self.env_ids[index]), int(self.episode_ids[index]))
        indices = self.episode_indices.pop(key, None)
        if not indices:
            self.completed[index] = False
            return
        self.completed[np.asarray(indices, dtype=np.int64)] = False

    def _mark_episode_complete(self, key: tuple[int, int]) -> None:
        indices = self.episode_indices.get(key)
        if indices:
            self.completed[np.asarray(indices, dtype=np.int64)] = True

    def _valid_indices(self) -> np.ndarray:
        limit = self.size
        if limit <= 0:
            return np.zeros(0, dtype=np.int64)
        return np.flatnonzero(self.completed[:limit]).astype(np.int64)

    def _sample_future_indices(self, batch_inds: np.ndarray) -> np.ndarray:
        future_inds = np.full(len(batch_inds), -1, dtype=np.int64)
        for out_idx, replay_idx in enumerate(np.asarray(batch_inds, dtype=np.int64)):
            if not bool(self.completed[replay_idx]):
                continue
            key = (int(self.env_ids[replay_idx]), int(self.episode_ids[replay_idx]))
            candidates = self.episode_indices.get(key, [])
            if not candidates:
                continue
            current_step = int(self.episode_steps[replay_idx])
            valid = [idx for idx in candidates if bool(self.completed[idx]) and int(self.episode_steps[idx]) >= current_step]
            if valid:
                future_inds[out_idx] = int(random.choice(valid))
        return future_inds

    def sample(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, ...]:
        valid = self._valid_indices()
        if valid.size == 0:
            raise RuntimeError("HER replay has no completed episodes available for sampling.")
        batch_inds = np.random.choice(valid, size=batch_size, replace=True)
        observations = self.observations[batch_inds].copy()
        next_observations = self.next_observations[batch_inds].copy()
        rewards = self.rewards[batch_inds].copy()
        relabel_mask = np.random.random(size=batch_size) < self.relabel_fraction
        if relabel_mask.any():
            relabel_positions = np.flatnonzero(relabel_mask)
            future_inds = self._sample_future_indices(batch_inds[relabel_positions])
            valid_positions = relabel_positions[future_inds >= 0]
            future_inds = future_inds[future_inds >= 0]
            relabel_goals = self.next_observations[future_inds, 3:6]
            observations[valid_positions, 6:9] = relabel_goals
            next_observations[valid_positions, 6:9] = relabel_goals
            observations[valid_positions, 12:15] = relabel_goals - observations[valid_positions, 3:6]
            next_observations[valid_positions, 12:15] = relabel_goals - next_observations[valid_positions, 3:6]
            if observations.shape[1] >= 6:
                observations[valid_positions, -3:] = relabel_goals
                next_observations[valid_positions, -3:] = relabel_goals
            rewards[valid_positions, 0] = _goal_reward(next_observations[valid_positions])
        return (
            torch.as_tensor(observations, dtype=torch.float32, device=device),
            torch.as_tensor(next_observations, dtype=torch.float32, device=device),
            torch.as_tensor(self.actions[batch_inds], dtype=torch.float32, device=device),
            torch.as_tensor(rewards, dtype=torch.float32, device=device),
            torch.as_tensor(self.dones[batch_inds], dtype=torch.float32, device=device),
        )


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, HIDDEN_SIZE)),
            nn.ReLU(),
            layer_init(nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)),
            nn.ReLU(),
            layer_init(nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)),
            nn.ReLU(),
        )
        self.fc_mean = layer_init(nn.Linear(HIDDEN_SIZE, act_dim), std=0.01)
        self.fc_logstd = layer_init(nn.Linear(HIDDEN_SIZE, act_dim), std=0.01)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.net(obs)
        mean = self.fc_mean(x)
        log_std = torch.tanh(self.fc_logstd(x))
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1.0)
        return mean, log_std

    def get_action(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self(obs)
        normal = Normal(mean, log_std.exp())
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t) - torch.log(1.0 - y_t.pow(2) + 1e-6)
        return y_t, log_prob.sum(1, keepdim=True), torch.tanh(mean)


class QuantileCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim + act_dim, HIDDEN_SIZE)),
            nn.ReLU(),
            layer_init(nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)),
            nn.ReLU(),
            layer_init(nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)),
            nn.ReLU(),
            layer_init(nn.Linear(HIDDEN_SIZE, N_QUANTILES), std=1.0),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=1))


class Agent:
    def __init__(self, obs_dim: int, device: torch.device, tool: EndEffectorDeltaTool | None = None) -> None:
        self.device = device
        self.tool = tool
        self.obs_dim = int(obs_dim)
        self.act_dim = 4
        self.actor = Actor(self.obs_dim, self.act_dim).to(device)
        self.critics = nn.ModuleList([QuantileCritic(self.obs_dim, self.act_dim) for _ in range(N_CRITICS)]).to(device)
        self.critic_targets = nn.ModuleList([QuantileCritic(self.obs_dim, self.act_dim) for _ in range(N_CRITICS)]).to(device)
        for target, critic in zip(self.critic_targets, self.critics):
            target.load_state_dict(critic.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LEARNING_RATE)
        self.critic_optimizer = optim.Adam(self.critics.parameters(), lr=LEARNING_RATE)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=LEARNING_RATE)
        self.target_entropy = -float(self.act_dim)
        self.alpha = float(self.log_alpha.exp().item())

    def act(self, obs: Any, deterministic: bool = False) -> np.ndarray:
        flat = flatten_observation(obs)
        obs_t = torch.as_tensor(flat, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, _, mean = self.actor.get_action(obs_t)
        tool_action = (mean if deterministic else action).squeeze(0).cpu().numpy()
        if self.tool is None and isinstance(obs, dict):
            return np.pad(tool_action, (0, 4), constant_values=0.0).astype(np.float32)
        if self.tool is None:
            return tool_action.astype(np.float32)
        return self.tool.single_action(flat, tool_action)

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "algorithm": ALGORITHM,
            "actor": self.actor.state_dict(),
            "critics": self.critics.state_dict(),
            "critic_targets": self.critic_targets.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "alpha": float(self.alpha),
            "recipe": RECIPE,
        }

    def load_checkpoint_state(self, state: dict[str, Any]) -> None:
        if state.get("algorithm") != ALGORITHM:
            raise ValueError(f"Checkpoint algorithm mismatch: {state.get('algorithm')}")
        self.actor.load_state_dict(state["actor"])
        self.critics.load_state_dict(state["critics"])
        self.critic_targets.load_state_dict(state["critic_targets"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.alpha_optimizer.load_state_dict(state["alpha_optimizer"])
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
    device: Any,
    init_checkpoint: str | Path | None = None,
    live_callback: Any | None = None,
) -> tuple[Agent, dict[str, Any]]:
    del candidate
    random.seed(benchmark.train_seed)
    np.random.seed(benchmark.train_seed)
    torch.manual_seed(benchmark.train_seed)
    torch_device = _to_device(device)
    num_envs = int(benchmark.env_kwargs.get("num_envs", 256))
    num_steps = int(benchmark.env_kwargs.get("steps_per_env_per_iteration", 32))
    budget_seconds = getattr(benchmark, "train_seconds", None)
    updates = 1_000_000 if budget_seconds is not None else max(1, int(benchmark.train_episodes))

    probe_env = env_factory(CONTROL_TYPE, REWARD_RECIPE)
    try:
        base_env = probe_env.unwrapped if hasattr(probe_env, "unwrapped") else probe_env
        make_vectorized = getattr(base_env, "make_vectorized", None)
        if make_vectorized is None or not str(benchmark.env_kwargs.get("backend", "")).startswith("mujoco_warp"):
            raise RuntimeError("seed_trainable_tqc_her_ee requires the MJWarp vector backend.")
        vector_env = make_vectorized(num_envs, int(benchmark.train_seed))
    finally:
        try:
            probe_env.close()
        except Exception:
            pass

    obs = vector_env.reset(seed=int(benchmark.train_seed)).astype(np.float32)
    tool = EndEffectorDeltaTool.from_env(vector_env)
    agent = Agent(int(obs.shape[1]), torch_device, tool=tool)
    resumed_from = load_agent_checkpoint(agent, init_checkpoint) if init_checkpoint is not None else None
    rb = HerReplayBuffer(int(obs.shape[1]), agent.act_dim, BUFFER_SIZE, HER_RELABEL_FRACTION)
    env_ids = np.arange(num_envs, dtype=np.int32)
    episode_ids = np.zeros(num_envs, dtype=np.int64)
    episode_steps = np.zeros(num_envs, dtype=np.int32)
    scripted_state = ScriptedE2EWarmupState(num_envs)
    scripted_env_steps = 0
    scripted_success_events = 0
    scripted_near_cube_events = 0
    scripted_lift_events = 0
    scripted_placed_events = 0

    records: list[dict[str, Any]] = []
    global_step = 0
    completed = 0
    gradient_updates = 0
    start_time = time.time()
    deadline = start_time + float(budget_seconds) if budget_seconds is not None else None
    scripted_warmup_deadline = (
        start_time + float(budget_seconds) * SCRIPTED_WARMUP_FRACTION
        if budget_seconds is not None
        else None
    )
    latest_infos: dict[str, Any] = {}
    last_metrics: dict[str, float] | None = None

    for _ in range(updates):
        if deadline is not None and time.time() >= deadline:
            break
        rewards_window: list[float] = []
        done_count = 0
        for _step in range(num_steps):
            scripted_active = (
                time.time() < scripted_warmup_deadline
                if scripted_warmup_deadline is not None
                else global_step < SCRIPTED_WARMUP_STEPS
            )
            if scripted_active:
                control_pos = tool.batch_control_positions(obs)
                tool_actions = scripted_state.actions(obs, noise_scale=SCRIPTED_WARMUP_NOISE, control_pos=control_pos)
            elif global_step < LEARNING_STARTS:
                tool_actions = np.random.uniform(-1.0, 1.0, size=(num_envs, agent.act_dim)).astype(np.float32)
            else:
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=torch_device)
                    action_t, _, _ = agent.actor.get_action(obs_t)
                    tool_actions = action_t.cpu().numpy().astype(np.float32)
            raw_actions = tool.batch_actions(obs, tool_actions)
            next_obs, _raw_rewards, dones, infos = vector_env.step(raw_actions)
            next_obs = next_obs.astype(np.float32)
            rewards = _goal_reward(next_obs)
            next_lifted_ever = _info_bool_array(infos, "lifted_ever", num_envs)
            success_flags = _info_bool_array(infos, "is_success", num_envs)
            near_cube_flags = _info_bool_array(infos, "near_cube", num_envs)
            lifted_flags = _info_bool_array(infos, "lifted_ever", num_envs)
            placed_flags = _info_bool_array(infos, "placed_success", num_envs)
            rb.add_batch(
                obs,
                next_obs,
                tool_actions,
                rewards,
                dones,
                env_ids=env_ids,
                episode_ids=episode_ids,
                episode_steps=episode_steps,
                next_lifted_ever=next_lifted_ever,
            )
            rewards_window.extend([float(v) for v in rewards])
            done_count += int(np.sum(dones))
            completed += int(np.sum(dones))
            episode_steps += 1
            if scripted_active:
                scripted_env_steps += num_envs
                scripted_success_events += int(np.sum(success_flags))
                scripted_near_cube_events += int(np.sum(near_cube_flags))
                scripted_lift_events += int(np.sum(lifted_flags))
                scripted_placed_events += int(np.sum(placed_flags))
                next_control_pos = tool.batch_control_positions(next_obs)
                scripted_state.advance(next_obs, dones, infos, control_pos=next_control_pos)
            if np.any(dones):
                done_mask = np.asarray(dones, dtype=bool)
                next_obs = vector_env.reset_worlds(dones).astype(np.float32)
                episode_ids[done_mask] += 1
                episode_steps[done_mask] = 0
                scripted_state.reset(done_mask)
            obs = next_obs
            latest_infos = dict(infos)
            if scripted_env_steps > 0:
                latest_infos["scripted_warmup_success_rate_cumulative"] = float(scripted_success_events / scripted_env_steps)
                latest_infos["scripted_warmup_near_cube_rate_cumulative"] = float(scripted_near_cube_events / scripted_env_steps)
                latest_infos["scripted_warmup_lifted_rate_cumulative"] = float(scripted_lift_events / scripted_env_steps)
                latest_infos["scripted_warmup_placed_rate_cumulative"] = float(scripted_placed_events / scripted_env_steps)
                latest_infos["scripted_warmup_active"] = float(scripted_active)
                latest_infos["scripted_warmup_avg_phase_index"] = float(np.mean(scripted_state.phase))
            global_step += num_envs
            if rb.sample_size >= BATCH_SIZE and rb.size >= LEARNING_STARTS:
                last_metrics = _update_tqc(agent, rb, torch_device, gradient_updates)
                gradient_updates += 1
        info_metrics = _mean_info(latest_infos)
        callback_metrics = {**(last_metrics or {}), "gradient_updates": float(gradient_updates), **info_metrics}
        records.append(
            make_train_collection_window_record(
                episode=len(records) + 1,
                return_value=float(np.mean(rewards_window)) if rewards_window else 0.0,
                length=float(num_steps),
                episodes_in_window=max(1, done_count),
                success=bool(float(info_metrics.get("is_success_rate", 0.0)) > 0.0),
                step=global_step,
                env_steps_in_window=num_steps * num_envs,
                elapsed_seconds=elapsed_seconds_since(start_time),
                info_metrics=info_metrics,
            )
        )
        response = _live_callback(
            live_callback,
            status="running",
            episode_records=records,
            total_steps=global_step,
            last_metrics=callback_metrics,
            agent=agent,
            env=None,
            elapsed_seconds=elapsed_seconds_since(start_time),
            current_episode=len(records),
            diagnostic_series=_diagnostic_series(),
        )
        _answer_sampled_trajectory_request(
            response,
            live_callback=live_callback,
            agent=agent,
            env_factory=env_factory,
            benchmark=benchmark,
            records=records,
            global_step=global_step,
            last_metrics=callback_metrics,
            elapsed_seconds=elapsed_seconds_since(start_time),
            scripted_active=scripted_active,
        )

    vector_env.close()
    return agent, _summary(benchmark, records, global_step, completed, start_time, gradient_updates, last_metrics, latest_infos, init_checkpoint, resumed_from)


def _update_tqc(agent: Agent, rb: HerReplayBuffer, device: torch.device, update_index: int) -> dict[str, float]:
    observations, next_observations, actions, rewards, dones = rb.sample(BATCH_SIZE, device)
    with torch.no_grad():
        next_actions, next_log_pi, _ = agent.actor.get_action(next_observations)
        target_quantiles = torch.cat([critic(next_observations, next_actions) for critic in agent.critic_targets], dim=1)
        sorted_quantiles, _ = torch.sort(target_quantiles, dim=1)
        drop = TOP_QUANTILES_TO_DROP_PER_NET * N_CRITICS
        kept = sorted_quantiles[:, : sorted_quantiles.shape[1] - drop]
        target = rewards + (1.0 - dones) * GAMMA * (kept - agent.alpha * next_log_pi)

    current_quantiles = torch.stack([critic(observations, actions) for critic in agent.critics], dim=1)
    critic_loss = _quantile_huber_loss(current_quantiles, target)
    agent.critic_optimizer.zero_grad()
    critic_loss.backward()
    agent.critic_optimizer.step()

    actor_loss_value = 0.0
    alpha_loss_value = 0.0
    if update_index % POLICY_FREQUENCY == 0:
        pi, log_pi, _ = agent.actor.get_action(observations)
        q_pi = torch.stack([critic(observations, pi).mean(dim=1) for critic in agent.critics], dim=1)
        min_q_pi = q_pi.min(dim=1, keepdim=True).values
        actor_loss = (agent.alpha * log_pi - min_q_pi).mean()
        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        agent.actor_optimizer.step()
        actor_loss_value = float(actor_loss.item())

        alpha_loss = -(agent.log_alpha.exp() * (log_pi.detach() + agent.target_entropy)).mean()
        agent.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        agent.alpha_optimizer.step()
        agent.alpha = float(agent.log_alpha.exp().item())
        alpha_loss_value = float(alpha_loss.item())

    if update_index % TARGET_NETWORK_FREQUENCY == 0:
        for critic, target_critic in zip(agent.critics, agent.critic_targets):
            for param, target_param in zip(critic.parameters(), target_critic.parameters()):
                target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)

    return {
        "critic_loss": float(critic_loss.item()),
        "actor_loss": actor_loss_value,
        "alpha": float(agent.alpha),
        "alpha_loss": alpha_loss_value,
    }


def _quantile_huber_loss(current: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # current: [batch, critics, quantiles], target: [batch, target_quantiles]
    batch_size = current.shape[0]
    current_flat = current.reshape(batch_size * N_CRITICS, N_QUANTILES)
    target_tiled = target[:, None, :].expand(batch_size, N_CRITICS, target.shape[1]).reshape(batch_size * N_CRITICS, target.shape[1])
    td = target_tiled[:, None, :] - current_flat[:, :, None]
    abs_td = torch.abs(td)
    huber = torch.where(abs_td <= 1.0, 0.5 * td.pow(2), abs_td - 0.5)
    taus = (torch.arange(N_QUANTILES, device=current.device, dtype=torch.float32) + 0.5) / float(N_QUANTILES)
    loss = torch.abs(taus[None, :, None] - (td.detach() < 0).float()) * huber
    return loss.sum(dim=1).mean()


def _goal_reward(next_obs: np.ndarray, lifted_ever: np.ndarray | None = None) -> np.ndarray:
    del lifted_ever
    ee = np.asarray(next_obs[:, 0:3], dtype=np.float32)
    achieved = np.asarray(next_obs[:, 3:6], dtype=np.float32)
    desired = np.asarray(next_obs[:, 6:9], dtype=np.float32)
    cube_to_goal = np.linalg.norm(achieved - desired, axis=1)
    ee_to_cube = np.linalg.norm(ee - achieved, axis=1)
    near_cube_bonus = (ee_to_cube < NEAR_CUBE_THRESHOLD).astype(np.float32) * NEAR_CUBE_REWARD
    reward = -cube_to_goal - APPROACH_REWARD_WEIGHT * ee_to_cube + near_cube_bonus
    return reward.astype(np.float32)


class ScriptedE2EWarmupState:
    def __init__(self, num_envs: int) -> None:
        self.phase = np.zeros(num_envs, dtype=np.int32)
        self.phase_steps = np.zeros(num_envs, dtype=np.int32)
        self.has_grasp_offset = np.zeros(num_envs, dtype=bool)
        self.grasp_offset = np.zeros((num_envs, 3), dtype=np.float32)

    def reset(self, mask: np.ndarray | None = None) -> None:
        if mask is None:
            self.phase[:] = 0
            self.phase_steps[:] = 0
            self.has_grasp_offset[:] = False
            self.grasp_offset[:] = 0.0
            return
        mask = np.asarray(mask, dtype=bool)
        self.phase[mask] = 0
        self.phase_steps[mask] = 0
        self.has_grasp_offset[mask] = False
        self.grasp_offset[mask] = 0.0

    def actions(self, obs: np.ndarray, noise_scale: float = 0.0, control_pos: np.ndarray | None = None) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        target, gripper = _scripted_targets(obs, self.phase, self.phase_steps)
        if control_pos is None:
            control_pos = obs[:, 0:3] + PINCH_OFFSET_FROM_HAND[None, :]
        else:
            control_pos = np.asarray(control_pos, dtype=np.float32)
        carry = ((self.phase == 4) | (self.phase == 5) | (self.phase == 6)) & self.has_grasp_offset
        if np.any(carry):
            desired = obs[:, 6:9]
            cube = obs[:, 3:6]
            target[carry] = desired[carry] + self.grasp_offset[carry]
            hover_carry = carry & (self.phase == 4)
            descend_carry = carry & ((self.phase == 5) | (self.phase == 6))
            if np.any(hover_carry):
                cube_error = desired[hover_carry, :2] - cube[hover_carry, :2]
                target[hover_carry, :2] = control_pos[hover_carry, :2] + cube_error
                target[hover_carry, 2] = np.maximum(
                    control_pos[hover_carry, 2],
                    desired[hover_carry, 2] + self.grasp_offset[hover_carry, 2] + float(SCRIPTED_CONFIG["place_clearance"]),
                )
            if np.any(descend_carry):
                cube_error = desired[descend_carry, :2] - cube[descend_carry, :2]
                target[descend_carry, :2] = control_pos[descend_carry, :2] + cube_error
                target[descend_carry, 2] = desired[descend_carry, 2] + self.grasp_offset[descend_carry, 2]
        delta = target - control_pos
        gain = float(SCRIPTED_CONFIG["gain"])
        tool_actions = np.zeros((obs.shape[0], 4), dtype=np.float32)
        tool_actions[:, :3] = np.clip(gain * delta / max(EE_ACTION_SCALE, 1.0e-6), -1.0, 1.0)
        tool_actions[:, 3] = gripper.astype(np.float32)
        if noise_scale > 0.0:
            tool_actions[:, :3] += np.random.normal(0.0, noise_scale, size=(obs.shape[0], 3)).astype(np.float32)
        return np.clip(tool_actions, -1.0, 1.0).astype(np.float32)

    def advance(self, next_obs: np.ndarray, dones: np.ndarray, infos: dict[str, Any], control_pos: np.ndarray | None = None) -> None:
        self.phase_steps += 1
        next_obs = np.asarray(next_obs, dtype=np.float32)
        target, _gripper = _scripted_targets(next_obs, self.phase, self.phase_steps)
        if control_pos is None:
            control_pos = next_obs[:, 0:3] + PINCH_OFFSET_FROM_HAND[None, :]
        else:
            control_pos = np.asarray(control_pos, dtype=np.float32)
        dist = np.linalg.norm(control_pos - target, axis=1)
        lifted_ever = _info_bool_array(infos, "lifted_ever", next_obs.shape[0])
        cube_at_goal = _info_bool_array(infos, "cube_at_goal", next_obs.shape[0])
        placed_success = _info_bool_array(infos, "placed_success", next_obs.shape[0])
        new_grasp = lifted_ever & ~self.has_grasp_offset
        if np.any(new_grasp):
            self.grasp_offset[new_grasp] = control_pos[new_grasp] - next_obs[new_grasp, 3:6]
            self.has_grasp_offset[new_grasp] = True
        max_steps = np.asarray(
            [int(SCRIPTED_CONFIG["phase_max_steps"][SCRIPTED_PHASES[int(phase)]]) for phase in self.phase],
            dtype=np.int32,
        )
        min_steps = np.asarray(
            [int(SCRIPTED_CONFIG["phase_min_steps"][SCRIPTED_PHASES[int(phase)]]) for phase in self.phase],
            dtype=np.int32,
        )
        min_elapsed = self.phase_steps >= min_steps
        reached = dist < float(SCRIPTED_CONFIG["tolerance"])
        timed = self.phase_steps >= max_steps
        close_done = (self.phase == 2) & (self.phase_steps >= int(SCRIPTED_CONFIG["close_steps"]))
        open_done = (self.phase == 6) & (self.phase_steps >= int(SCRIPTED_CONFIG["open_steps"]))
        dwell_phase = (self.phase == 2) | (self.phase == 6)
        lift_confirmed = (self.phase == 3) & lifted_ever & (self.phase_steps >= 8)
        carry_confirmed = (self.phase == 4) & min_elapsed & cube_at_goal
        descend_confirmed = (self.phase == 5) & min_elapsed & (cube_at_goal | reached)
        release_confirmed = (self.phase == 6) & (placed_success | open_done)
        reached_can_advance = (self.phase != 4) & (reached & min_elapsed) & ~dwell_phase
        base_advance = reached_can_advance | timed | close_done | open_done
        advance = (
            (base_advance | lift_confirmed | carry_confirmed | descend_confirmed | release_confirmed)
            & ~np.asarray(dones, dtype=bool)
        )
        if np.any(advance):
            self.phase[advance] = np.minimum(self.phase[advance] + 1, len(SCRIPTED_PHASES) - 1)
            self.phase_steps[advance] = 0


def _scripted_targets(obs: np.ndarray, phases: np.ndarray, phase_steps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    del phase_steps
    obs = np.asarray(obs, dtype=np.float32)
    phases = np.asarray(phases, dtype=np.int32)
    cube = obs[:, 3:6]
    desired = obs[:, 6:9]
    cube_xy = cube[:, :2] + np.asarray([SCRIPTED_CONFIG["x_offset"], SCRIPTED_CONFIG["y_offset"]], dtype=np.float32)
    goal_xy = desired[:, :2] + np.asarray([SCRIPTED_CONFIG["x_offset"], SCRIPTED_CONFIG["y_offset"]], dtype=np.float32)
    target = np.zeros((obs.shape[0], 3), dtype=np.float32)
    gripper = np.full(obs.shape[0], -1.0, dtype=np.float32)

    hover = phases == 0
    descend = phases == 1
    close = phases == 2
    lift = phases == 3
    hover_goal = phases == 4
    descend_goal = phases == 5
    open_phase = phases == 6

    target[hover, :2] = cube_xy[hover]
    target[hover, 2] = float(SCRIPTED_CONFIG["hover_z"])

    target[descend | close, :2] = cube_xy[descend | close]
    target[descend | close, 2] = float(SCRIPTED_CONFIG["grasp_z"])
    gripper[close] = 1.0

    lift_z = np.maximum(float(SCRIPTED_CONFIG["lift_z"]), desired[:, 2] + 0.08)
    target[lift, :2] = cube_xy[lift]
    target[lift, 2] = lift_z[lift]
    gripper[lift] = 1.0

    target[hover_goal, :2] = goal_xy[hover_goal]
    target[hover_goal, 2] = lift_z[hover_goal]
    gripper[hover_goal] = 1.0

    place_z = np.maximum(
        float(SCRIPTED_CONFIG["grasp_z"]),
        desired[:, 2] + float(SCRIPTED_CONFIG["place_clearance"]),
    )
    target[descend_goal | open_phase, :2] = goal_xy[descend_goal | open_phase]
    target[descend_goal | open_phase, 2] = place_z[descend_goal | open_phase]
    gripper[descend_goal] = 1.0
    gripper[open_phase] = -1.0
    return target, gripper


def _render_policy_frame(env: gym.Env[Any, Any]) -> np.ndarray | None:
    render_env = getattr(env, "unwrapped", env)
    try:
        frame = render_env.render()
    except Exception:
        return None
    if frame is None:
        return None
    return np.asarray(frame, dtype=np.uint8)


def _policy_tool_action(agent: Agent, obs: Any) -> np.ndarray:
    flat = flatten_observation(obs)
    obs_t = torch.as_tensor(flat, dtype=torch.float32, device=agent.device).unsqueeze(0)
    with torch.no_grad():
        _action, _log_prob, mean = agent.actor.get_action(obs_t)
    return mean.squeeze(0).cpu().numpy().astype(np.float32)


def _scripted_tool_action(state: ScriptedE2EWarmupState, obs: Any) -> np.ndarray:
    flat = flatten_observation(obs)
    return state.actions(flat.reshape(1, -1), noise_scale=0.0)[0].astype(np.float32)


def _sample_policy_trajectory(
    agent: Agent,
    env_factory: Any,
    benchmark: Any,
    request: dict[str, Any],
    *,
    scripted: bool = False,
) -> dict[str, Any]:
    env = env_factory(CONTROL_TYPE, REWARD_RECIPE)
    frames: list[np.ndarray] = []
    episode = int(request.get("episode") or 0)
    sample_index = int(request.get("sample_index") or 0)
    stride = max(1, int(request.get("frame_stride") or 2))
    max_steps = int(getattr(benchmark, "max_steps", 50))
    seed = int(getattr(benchmark, "eval_seed_start", getattr(benchmark, "train_seed", 0))) + max(sample_index - 1, 0)
    scripted_state = ScriptedE2EWarmupState(1) if scripted else None
    playback_source = "guided_warmup" if scripted else "policy"
    try:
        obs, _info = env.reset(seed=seed)
        for step in range(max_steps + 1):
            if step % stride == 0:
                frame = _render_policy_frame(env)
                if frame is not None:
                    frames.append(frame)
            if step >= max_steps:
                break
            tool_action = (
                _scripted_tool_action(scripted_state, obs)
                if scripted_state is not None
                else _policy_tool_action(agent, obs)
            )
            obs, _reward, terminated, truncated, info = env.step(tool_action)
            if scripted_state is not None:
                scripted_state.advance(
                    flatten_observation(obs).reshape(1, -1),
                    np.asarray([terminated or truncated], dtype=bool),
                    dict(info),
                )
            if bool(terminated or truncated):
                frame = _render_policy_frame(env)
                if frame is not None:
                    frames.append(frame)
                break
    finally:
        try:
            env.close()
        except Exception:
            pass
    return {
        "episode": episode,
        "sample_index": sample_index,
        "source": RECIPE["runner"]["sample_trajectory_source"],
        "frames": frames,
        "playback_fps": float(request.get("playback_fps") or 20.0),
        "frame_stride": stride,
        "metadata": {"seed": seed, "max_steps": max_steps, "playback_source": playback_source},
    }


def _answer_sampled_trajectory_request(
    response: dict[str, Any] | None,
    *,
    live_callback: Any | None,
    agent: Agent,
    env_factory: Any,
    benchmark: Any,
    records: list[dict[str, Any]],
    global_step: int,
    last_metrics: dict[str, float] | None,
    elapsed_seconds: float,
    scripted_active: bool = False,
) -> None:
    request = response.get("sampled_trajectory_request") if isinstance(response, dict) else None
    if not isinstance(request, dict) or not request.get("requested"):
        return
    sampled_trajectory = _sample_policy_trajectory(agent, env_factory, benchmark, request, scripted=scripted_active)
    _live_callback(
        live_callback,
        status="running",
        episode_records=records,
        total_steps=global_step,
        last_metrics=last_metrics,
        agent=agent,
        env=None,
        elapsed_seconds=elapsed_seconds,
        current_episode=int(request.get("episode") or len(records)),
        sampled_trajectory=sampled_trajectory,
        diagnostic_series=_diagnostic_series(),
    )


def _info_bool_array(infos: dict[str, Any], key: str, count: int) -> np.ndarray:
    if key not in infos:
        return np.zeros(count, dtype=bool)
    value = np.asarray(infos[key])
    if value.shape == ():
        return np.full(count, bool(value), dtype=bool)
    return value.astype(bool, copy=False).reshape(-1)[:count]


def _mean_info(infos: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in infos.items():
        if isinstance(value, str):
            continue
        arr = np.asarray(value)
        if arr.dtype == np.bool_:
            metrics[f"{key}_rate"] = float(arr.astype(np.float32).mean())
        elif np.issubdtype(arr.dtype, np.number):
            metrics[f"avg_{key}"] = float(arr.astype(np.float32).mean())
    return metrics


def _summary(
    benchmark: Any,
    records: list[dict[str, Any]],
    global_step: int,
    completed: int,
    start_time: float,
    gradient_updates: int,
    last_metrics: dict[str, float] | None,
    latest_infos: dict[str, Any],
    init_checkpoint: str | Path | None,
    resumed_from: dict[str, Any] | None,
) -> dict[str, Any]:
    returns = [float(record["return"]) for record in records]
    successes = [1.0 if record.get("success") else 0.0 for record in records]
    wall_clock = elapsed_seconds_since(start_time)
    budget_seconds = getattr(benchmark, "train_seconds", None)
    stop_reason = "time_budget_exhausted" if budget_seconds is not None and wall_clock >= float(budget_seconds) else "episode_cap_reached"
    num_envs = int(getattr(benchmark, "env_kwargs", {}).get("num_envs", 1) or 1)
    return {
        "episodes": int(benchmark.train_episodes),
        "episodes_completed": int(completed),
        "time_budget_seconds": float(budget_seconds) if budget_seconds is not None else None,
        "stop_reason": stop_reason,
        "total_steps": int(global_step),
        "env_steps": int(global_step),
        "num_envs": num_envs,
        "vector_envs": num_envs,
        "completed_episodes": int(completed),
        "episode_batches": len(records),
        "gradient_updates": int(gradient_updates),
        "avg_return": float(np.mean(returns)) if returns else 0.0,
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "avg_length": float(np.mean([record["length"] for record in records])) if records else 0.0,
        "last_metrics": {
            **(last_metrics or {}),
            **_mean_info(latest_infos),
            "gradient_updates": float(gradient_updates),
            "num_envs": float(num_envs),
        },
        "episode_records": records,
        "wall_clock_seconds": wall_clock,
        "init_checkpoint": str(init_checkpoint) if init_checkpoint is not None else None,
        "resumed_from": resumed_from,
        "curriculum": {
            "type": REWARD_RECIPE,
            "ee_action_scale": EE_ACTION_SCALE,
            "ik_damping": IK_DAMPING,
            "gripper_close_sign": GRIPPER_CLOSE_SIGN,
            "gripper_width_delta_scale": GRIPPER_WIDTH_DELTA_SCALE,
            "her_relabel_fraction": HER_RELABEL_FRACTION,
            "her_goal_selection_strategy": "same_env_same_episode_future",
            "reward_contract": "panda_gym_dense_distance_plus_training_only_approach_and_guided_warmup",
            "training_approach_reward_weight": APPROACH_REWARD_WEIGHT,
            "training_near_cube_reward": NEAR_CUBE_REWARD,
            "training_near_cube_threshold": NEAR_CUBE_THRESHOLD,
            "scripted_warmup_steps": SCRIPTED_WARMUP_STEPS,
            "scripted_warmup_fraction": SCRIPTED_WARMUP_FRACTION,
            "scripted_warmup_noise": SCRIPTED_WARMUP_NOISE,
            "scripted_warmup_mode": "fk_pinch_hover_descend_close_lift_until_lifted_carry_descend_release",
            "scripted_phase_max_steps": dict(SCRIPTED_CONFIG["phase_max_steps"]),
            "success_contract": "distance_less_than_0.05",
        },
        "diagnostic_series": _diagnostic_series(),
        "physics_backend": "mujoco_warp_vectorized",
        "vectorized_backend": "mujoco_warp_vectorized",
    }


def _live_callback(live_callback: Any | None, **payload: Any) -> Any:
    if live_callback is None:
        return None
    return live_callback(**payload)


def _to_device(device: Any) -> torch.device:
    if isinstance(device, torch.device):
        return device
    requested = str(device)
    if requested in {"auto", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
