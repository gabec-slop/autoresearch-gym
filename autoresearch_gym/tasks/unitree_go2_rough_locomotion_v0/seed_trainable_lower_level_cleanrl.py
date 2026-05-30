from __future__ import annotations

import math
import shutil
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from autoresearch_gym.runner.curves import make_train_collection_window_record
from autoresearch_gym.tasks.unitree_g1_motion_mirror_v0.seed_trainable_lower_level_cleanrl import (
    RENDER_HEIGHT,
    RENDER_WIDTH,
    _draw_line,
    _maybe_import_mujoco,
    _maybe_import_mujoco_warp,
    _prepare_model_for_mujoco_warp,
    _repo_root,
)


# Lower-level CleanRL seed for Go2/G2 rough locomotion.
# The env adapter, reward terms, vector rollout collection, GAE, PPO losses, and
# checkpointing are deliberately visible in this file so an autoresearch loop can
# mutate the training recipe directly.

EXP_NAME = "unitree_go2_rough_locomotion_lower_level_cleanrl_ppo"
ALGORITHM = "ppo"
CONTROL_TYPE = None
REWARD_RECIPE = "rough_velocity_tracking"
NUM_ENVS = 8
NUM_STEPS = 16
LEARNING_RATE = 3.0e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
UPDATE_EPOCHS = 2
NUM_MINIBATCHES = 2
CLIP_COEF = 0.2
ENT_COEF = 0.005
VF_COEF = 0.5
MAX_GRAD_NORM = 1.0
HIDDEN_SIZE = 128
ACTION_STD_INIT = 0.6
ACTION_SCALE = 0.25
CONTROL_DECIMATION = 4


def _go2_xml_path() -> Path:
    return (
        _repo_root()
        / ".external"
        / "unitree_rl_mjlab"
        / "src"
        / "assets"
        / "robots"
        / "unitree_go2"
        / "xmls"
        / "scene_go2.xml"
    )


def _quat_to_projected_gravity(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quat_wxyz, dtype=np.float64)
    # Third column of R^T * world gravity direction, expressed in base frame.
    return np.asarray(
        [
            -2.0 * (x * z - w * y),
            -2.0 * (y * z + w * x),
            -(1.0 - 2.0 * (x * x + y * y)),
        ],
        dtype=np.float32,
    )


def _terrain_height(x: float, y: float) -> float:
    return 0.035 * math.sin(2.7 * x) + 0.025 * math.sin(3.9 * y + 0.7) + 0.015 * math.sin(5.2 * (x + y))


def get_candidate() -> str:
    return (
        "Lower-level CleanRL-style PPO seed for Unitree Go2/G2 rough locomotion. "
        "The task exposes command tracking, rough terrain disturbance, survival, "
        "energy, slip, and clearance terms directly as code."
    )


class RewardRecipeWrapper:
    def __init__(self, env: Any, recipe: str | None = None) -> None:
        self.env = env
        self.recipe = recipe

    @property
    def unwrapped(self) -> Any:
        return getattr(self.env, "unwrapped", self.env)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)


def layer_init(layer: nn.Linear, std: float = math.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class UnitreeGo2LowerLevelEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, render_mode: str | None = None, max_steps: int = 80) -> None:
        self.render_mode = render_mode
        self.max_steps = int(max_steps)
        self.rng = np.random.default_rng(0)
        self.step_count = 0
        self.command = np.zeros(3, dtype=np.float32)
        self.height_error = 0.0
        self.last_tracking_error = 1.0
        self.physics_backend = "fallback_kinematic"
        self._mujoco = _maybe_import_mujoco()
        self.model = None
        self.data = None
        self.renderer = None
        self.joint_ids = np.asarray([], dtype=np.int32)
        self.qpos_adrs = np.asarray([], dtype=np.int32)
        self.qvel_adrs = np.asarray([], dtype=np.int32)
        self.joint_lower = np.asarray([], dtype=np.float32)
        self.joint_upper = np.asarray([], dtype=np.float32)
        self.default_joint_pos = np.zeros(0, dtype=np.float32)
        self.home_qpos = np.zeros(0, dtype=np.float64)
        self.push_velocity = np.zeros(3, dtype=np.float32)

        xml_path = _go2_xml_path()
        if self._mujoco is not None and xml_path.exists():
            self._init_mujoco(xml_path)
        else:
            self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(27,), dtype=np.float32)
            self.base_velocity = np.zeros(3, dtype=np.float32)
            self.joint_pos = np.zeros(8, dtype=np.float32)
            self.joint_vel = np.zeros(8, dtype=np.float32)
            self.last_action = np.zeros(8, dtype=np.float32)

    def _init_mujoco(self, xml_path: Path) -> None:
        assert self._mujoco is not None
        self.model = self._mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = self._mujoco.MjData(self.model)
        actuator_joint_ids = np.asarray(self.model.actuator_trnid[: self.model.nu, 0], dtype=np.int32)
        self.joint_ids = actuator_joint_ids
        self.qpos_adrs = np.asarray([self.model.jnt_qposadr[int(jid)] for jid in actuator_joint_ids], dtype=np.int32)
        self.qvel_adrs = np.asarray([self.model.jnt_dofadr[int(jid)] for jid in actuator_joint_ids], dtype=np.int32)
        self.joint_lower = self.model.jnt_range[actuator_joint_ids, 0].astype(np.float32)
        self.joint_upper = self.model.jnt_range[actuator_joint_ids, 1].astype(np.float32)
        self.home_qpos = np.asarray(self.model.qpos0, dtype=np.float64).copy()
        if self.model.nkey > 0:
            self._mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
            self.home_qpos = np.asarray(self.data.qpos, dtype=np.float64).copy()
        self.default_joint_pos = np.asarray(self.home_qpos[self.qpos_adrs], dtype=np.float32)
        self.last_action = np.zeros(self.model.nu, dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32)
        height_scan_count = 8
        obs_dim = 3 + 6 + 3 + 2 + 3 * self.model.nu + height_scan_count + self.model.nu
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.physics_backend = "mujoco_unitree_go2_scene"

    def make_vectorized(self, num_envs: int, seed: int = 0):
        if self.model is None:
            return None
        try:
            return UnitreeGo2MujocoWarpVectorEnv(num_envs=num_envs, max_steps=self.max_steps, seed=seed)
        except Exception:
            return None

    def _joint_pos(self) -> np.ndarray:
        assert self.data is not None
        return np.asarray(self.data.qpos[self.qpos_adrs], dtype=np.float32)

    def _joint_vel(self) -> np.ndarray:
        assert self.data is not None
        return np.asarray(self.data.qvel[self.qvel_adrs], dtype=np.float32)

    def _height_scan(self) -> np.ndarray:
        if self.data is None:
            offsets = np.linspace(-0.4, 0.4, 8)
            return np.asarray([_terrain_height(float(offset), 0.0) - self.height_error for offset in offsets], dtype=np.float32)
        x, y, z = [float(v) for v in self.data.qpos[:3]]
        offsets = [(-0.35, -0.25), (-0.15, -0.25), (0.15, -0.25), (0.35, -0.25), (-0.35, 0.25), (-0.15, 0.25), (0.15, 0.25), (0.35, 0.25)]
        return np.asarray([_terrain_height(x + dx, y + dy) - z for dx, dy in offsets], dtype=np.float32)

    def _obs_mujoco(self) -> np.ndarray:
        assert self.data is not None
        gait = np.asarray([math.sin(0.22 * self.step_count), math.cos(0.22 * self.step_count)], dtype=np.float32)
        projected_gravity = _quat_to_projected_gravity(np.asarray(self.data.qpos[3:7], dtype=np.float32))
        return np.concatenate(
            [
                self.command,
                np.asarray(self.data.qvel[:6], dtype=np.float32),
                projected_gravity,
                gait,
                self._joint_pos() - self.default_joint_pos,
                self._joint_vel(),
                self._height_scan(),
                self.last_action,
            ]
        ).astype(np.float32)

    def _obs(self) -> np.ndarray:
        terrain_phase = np.asarray(
            [
                math.sin(0.2 * self.step_count),
                math.cos(0.2 * self.step_count),
                self.height_error,
                self.step_count / max(1, self.max_steps),
                self.last_tracking_error,
            ],
            dtype=np.float32,
        )
        return np.concatenate([self.command, self.base_velocity, self.joint_pos, self.joint_vel, terrain_phase]).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        fixed_case = (options or {}).get("fixed_case", {})
        self.step_count = 0
        self.command = np.asarray(
            fixed_case.get(
                "command",
                [
                    self.rng.uniform(0.4, 1.2),
                    self.rng.uniform(-0.25, 0.25),
                    self.rng.uniform(-0.4, 0.4),
                ],
            ),
            dtype=np.float32,
        )
        if self.model is not None and self.data is not None:
            if self.model.nkey > 0:
                self._mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
            else:
                self.data.qpos[:] = self.home_qpos
                self.data.qvel[:] = 0.0
            self.data.qpos[0] += float(fixed_case.get("x", self.rng.uniform(-0.15, 0.15)))
            self.data.qpos[1] += float(fixed_case.get("y", self.rng.uniform(-0.15, 0.15)))
            self.data.qpos[2] = max(float(self.data.qpos[2]), 0.27 + _terrain_height(float(self.data.qpos[0]), float(self.data.qpos[1])))
            self.data.qpos[self.qpos_adrs] += self.rng.normal(0.0, 0.025, size=self.model.nu)
            self.data.qvel[:] = 0.0
            self.push_velocity = self.rng.normal(0.0, 0.04, size=3).astype(np.float32)
            self.last_action = np.zeros(self.model.nu, dtype=np.float32)
            self._mujoco.mj_forward(self.model, self.data)
            self.height_error = float(self.data.qpos[2] - (0.27 + _terrain_height(float(self.data.qpos[0]), float(self.data.qpos[1]))))
            self.last_tracking_error = float(np.linalg.norm(self.command[:2] - np.asarray(self.data.qvel[:2], dtype=np.float32)))
            return self._obs_mujoco(), {
                "command_tracking_error": self.last_tracking_error,
                "physics_backend": self.physics_backend,
            }
        self.base_velocity = self.rng.normal(0.0, 0.05, size=3).astype(np.float32)
        self.joint_pos = self.rng.normal(0.0, 0.08, size=8).astype(np.float32)
        self.joint_vel = np.zeros(8, dtype=np.float32)
        self.height_error = float(self.rng.normal(0.0, 0.03))
        self.last_action = np.zeros(8, dtype=np.float32)
        self.last_tracking_error = float(np.linalg.norm(self.command - self.base_velocity))
        return self._obs(), {"command_tracking_error": self.last_tracking_error}

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        if self.model is not None and self.data is not None:
            target = np.clip(self.default_joint_pos + ACTION_SCALE * action, self.joint_lower, self.joint_upper)
            kp = np.full(self.model.nu, 35.0, dtype=np.float32)
            kd = np.full(self.model.nu, 1.2, dtype=np.float32)
            torque_limits = np.asarray(self.model.actuator_ctrlrange[: self.model.nu], dtype=np.float32)
            if self.step_count > 0 and self.step_count % 35 == 0:
                self.data.qvel[:3] += self.rng.normal(0.0, 0.16, size=3)
            for _ in range(CONTROL_DECIMATION):
                torque = kp * (target - self._joint_pos()) - kd * self._joint_vel()
                self.data.ctrl[:] = np.clip(torque, torque_limits[:, 0], torque_limits[:, 1])
                self._mujoco.mj_step(self.model, self.data)
            self.step_count += 1
            lin_vel = np.asarray(self.data.qvel[:3], dtype=np.float32)
            ang_vel = np.asarray(self.data.qvel[3:6], dtype=np.float32)
            linear_error = float(np.linalg.norm(self.command[:2] - lin_vel[:2]))
            angular_error = float(abs(float(self.command[2] - ang_vel[2])))
            projected_gravity = _quat_to_projected_gravity(np.asarray(self.data.qpos[3:7], dtype=np.float32))
            orientation_penalty = float(np.linalg.norm(projected_gravity[:2]))
            energy = float(np.mean(np.square(self.data.ctrl / np.maximum(1.0, np.abs(torque_limits[:, 1])))))
            action_rate = float(np.mean(np.square(action - self.last_action)))
            joint_limit = float(np.mean((self._joint_pos() < self.joint_lower + 0.03) | (self._joint_pos() > self.joint_upper - 0.03)))
            terrain_z = _terrain_height(float(self.data.qpos[0]), float(self.data.qpos[1]))
            self.height_error = float(self.data.qpos[2] - (0.27 + terrain_z))
            foot_slip = float(abs(lin_vel[1]) + 0.15 * np.linalg.norm(ang_vel[:2]))
            foot_clearance = float(max(0.0, min(0.12, self.height_error)))
            fall = bool(self.data.qpos[2] < 0.16 or orientation_penalty > 0.9)
            reward = (
                1.8 * math.exp(-2.5 * linear_error)
                + 0.8 * math.exp(-2.0 * angular_error)
                + 0.15
                - 0.45 * orientation_penalty
                - 0.04 * energy
                - 0.05 * action_rate
                - 0.08 * joint_limit
                - 0.12 * foot_slip
                + 0.04 * foot_clearance
                - (1.0 if fall else 0.0)
            )
            tracking_error = float(linear_error + 0.35 * angular_error)
            self.last_action = action
            self.last_tracking_error = tracking_error
            terminated = fall
            truncated = self.step_count >= self.max_steps
            info = {
                "command_tracking_error": tracking_error,
                "tracking_error": tracking_error,
                "mpkpe": tracking_error,
                "linear_velocity_error": linear_error,
                "angular_velocity_error": angular_error,
                "energy_cost": energy,
                "action_rate": action_rate,
                "joint_limit": joint_limit,
                "foot_slip": foot_slip,
                "foot_clearance": foot_clearance,
                "fall_rate": 1.0 if fall else 0.0,
                "rough_height_span": float(np.max(self._height_scan()) - np.min(self._height_scan())),
                "physics_backend": self.physics_backend,
                "is_success": (not fall) and tracking_error < 0.45,
            }
            return self._obs_mujoco(), float(reward), terminated, truncated, info
        terrain = 0.12 * math.sin(0.17 * self.step_count) + float(self.rng.normal(0.0, 0.015))
        drive = np.asarray([action[0] - action[1] + action[2] - action[3], action[4] - action[5], action[6] - action[7]], dtype=np.float32)
        self.base_velocity = 0.88 * self.base_velocity + 0.08 * drive + 0.07 * (self.command - self.base_velocity)
        self.height_error = 0.82 * self.height_error + terrain - 0.025 * float(np.mean(np.abs(action)))
        self.joint_vel = 0.78 * self.joint_vel + 0.16 * action
        self.joint_pos = self.joint_pos + 0.10 * self.joint_vel
        self.step_count += 1
        tracking_error = float(np.linalg.norm(self.command - self.base_velocity))
        energy = float(np.mean(np.square(action)))
        slip = float(abs(self.base_velocity[1]) + 0.2 * max(0.0, abs(self.height_error) - 0.15))
        clearance = float(max(0.0, 0.08 - abs(self.height_error)))
        fall = bool(abs(self.height_error) > 0.55 or np.linalg.norm(self.joint_pos) > 5.0)
        reward = 1.5 - 1.2 * tracking_error - 0.04 * energy - 0.20 * slip + 0.05 * clearance
        self.last_action = action
        self.last_tracking_error = tracking_error
        terminated = fall
        truncated = self.step_count >= self.max_steps
        info = {
            "command_tracking_error": tracking_error,
            "tracking_error": tracking_error,
            "mpkpe": tracking_error,
            "energy_cost": energy,
            "foot_slip": slip,
            "foot_clearance": clearance,
            "fall_rate": 1.0 if fall else 0.0,
            "is_success": (not fall) and tracking_error < 0.35,
        }
        return self._obs(), float(reward), terminated, truncated, info

    def render(self):
        if self.model is not None and self.data is not None:
            try:
                if self.renderer is None:
                    self.renderer = self._mujoco.Renderer(self.model, RENDER_HEIGHT, RENDER_WIDTH)
                self.renderer.update_scene(self.data)
                return np.asarray(self.renderer.render(), dtype=np.uint8)
            except Exception:
                return self._render_fallback()
        return self._render_fallback()

    def _render_fallback(self):
        width, height = 320, 220
        frame = np.full((height, width, 3), 242, dtype=np.uint8)
        base_velocity = np.asarray(self.data.qvel[:3], dtype=np.float32) if self.data is not None else self.base_velocity
        body = np.asarray([80 + int(90 * np.clip(base_velocity[0], -0.5, 1.6)), 100 + int(35 * self.height_error)])
        target = np.asarray([80 + int(90 * np.clip(self.command[0], -0.5, 1.6)), 100])
        frame[max(0, target[1] - 25) : min(height, target[1] + 25), max(0, target[0] - 2) : min(width, target[0] + 2)] = (70, 150, 220)
        legs = [(-24, 18), (-8, 20), (8, 20), (24, 18)]
        for lx, ly in legs:
            hip = body + np.asarray([lx, 8])
            foot = body + np.asarray([lx + int(12 * math.sin(0.25 * self.step_count + lx)), 48 + ly])
            _draw_line(frame, hip, foot, (70, 80, 90))
        frame[max(0, body[1] - 14) : min(height, body[1] + 14), max(0, body[0] - 34) : min(width, body[0] + 34)] = (220, 100, 70)
        bar = int(np.clip(260 * (1.0 - self.last_tracking_error), 0, 260))
        frame[188:198, 30 : 30 + bar] = np.asarray([65, 170, 100], dtype=np.uint8)
        return frame

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


class UnitreeGo2MujocoWarpVectorEnv:
    def __init__(self, num_envs: int, max_steps: int = 80, seed: int = 0) -> None:
        self.num_envs = int(num_envs)
        self.max_steps = int(max_steps)
        self.rng = np.random.default_rng(seed)
        self.mujoco = _maybe_import_mujoco()
        self.mujoco_warp = _maybe_import_mujoco_warp()
        if self.mujoco is None or self.mujoco_warp is None:
            raise RuntimeError("mujoco_warp is not available")
        xml_path = _go2_xml_path()
        if not xml_path.exists():
            raise FileNotFoundError(xml_path)
        self.model = self.mujoco.MjModel.from_xml_path(str(xml_path))
        _prepare_model_for_mujoco_warp(self.mujoco, self.model)
        self.data0 = self.mujoco.MjData(self.model)
        if self.model.nkey > 0:
            self.mujoco.mj_resetDataKeyframe(self.model, self.data0, 0)
        actuator_joint_ids = np.asarray(self.model.actuator_trnid[: self.model.nu, 0], dtype=np.int32)
        self.qpos_adrs = np.asarray([self.model.jnt_qposadr[int(jid)] for jid in actuator_joint_ids], dtype=np.int32)
        self.qvel_adrs = np.asarray([self.model.jnt_dofadr[int(jid)] for jid in actuator_joint_ids], dtype=np.int32)
        self.joint_lower = self.model.jnt_range[actuator_joint_ids, 0].astype(np.float32)
        self.joint_upper = self.model.jnt_range[actuator_joint_ids, 1].astype(np.float32)
        self.home_qpos = np.asarray(self.data0.qpos, dtype=np.float32).copy()
        self.default_joint_pos = np.asarray(self.home_qpos[self.qpos_adrs], dtype=np.float32)
        self.warp_model = self.mujoco_warp.put_model(self.model)
        self.warp_data = self.mujoco_warp.put_data(self.model, self.data0, nworld=self.num_envs)
        self.command = np.zeros((self.num_envs, 3), dtype=np.float32)
        self.step_counts = np.zeros(self.num_envs, dtype=np.int32)
        self.last_actions = np.zeros((self.num_envs, self.model.nu), dtype=np.float32)
        self.height_error = np.zeros(self.num_envs, dtype=np.float32)
        self.last_tracking_error = np.ones(self.num_envs, dtype=np.float32)
        self.physics_backend = "mujoco_warp_unitree_go2_scene"
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32)
        obs_dim = 3 + 6 + 3 + 2 + 3 * self.model.nu + 8 + self.model.nu
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def _height_scan(self, qpos: np.ndarray) -> np.ndarray:
        offsets = [(-0.35, -0.25), (-0.15, -0.25), (0.15, -0.25), (0.35, -0.25), (-0.35, 0.25), (-0.15, 0.25), (0.15, 0.25), (0.35, 0.25)]
        scan = np.zeros((self.num_envs, len(offsets)), dtype=np.float32)
        for idx, (dx, dy) in enumerate(offsets):
            scan[:, idx] = np.asarray([_terrain_height(float(x + dx), float(y + dy)) - float(z) for x, y, z in qpos[:, :3]], dtype=np.float32)
        return scan

    def _projected_gravity_batch(self, qpos: np.ndarray) -> np.ndarray:
        return np.stack([_quat_to_projected_gravity(quat) for quat in qpos[:, 3:7]], axis=0).astype(np.float32)

    def _obs_from_arrays(self, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        gait = np.stack([np.sin(0.22 * self.step_counts), np.cos(0.22 * self.step_counts)], axis=1).astype(np.float32)
        return np.concatenate(
            [
                self.command,
                qvel[:, :6],
                self._projected_gravity_batch(qpos),
                gait,
                qpos[:, self.qpos_adrs] - self.default_joint_pos[None, :],
                qvel[:, self.qvel_adrs],
                self._height_scan(qpos),
                self.last_actions,
            ],
            axis=1,
        ).astype(np.float32)

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.step_counts[:] = 0
        self.command[:, 0] = self.rng.uniform(0.4, 1.2, size=self.num_envs)
        self.command[:, 1] = self.rng.uniform(-0.25, 0.25, size=self.num_envs)
        self.command[:, 2] = self.rng.uniform(-0.4, 0.4, size=self.num_envs)
        qpos = np.repeat(self.home_qpos[None, :], self.num_envs, axis=0).astype(np.float32)
        qvel = np.zeros((self.num_envs, self.model.nv), dtype=np.float32)
        qpos[:, 0] += self.rng.uniform(-0.15, 0.15, size=self.num_envs).astype(np.float32)
        qpos[:, 1] += self.rng.uniform(-0.15, 0.15, size=self.num_envs).astype(np.float32)
        qpos[:, 2] = np.maximum(qpos[:, 2], 0.27 + np.asarray([_terrain_height(float(x), float(y)) for x, y in qpos[:, :2]], dtype=np.float32))
        qpos[:, self.qpos_adrs] += self.rng.normal(0.0, 0.025, size=(self.num_envs, self.model.nu)).astype(np.float32)
        self.last_actions[:] = 0.0
        self.warp_data.qpos.assign(qpos)
        self.warp_data.qvel.assign(qvel)
        self.warp_data.ctrl.assign(np.zeros((self.num_envs, self.model.nu), dtype=np.float32))
        self.mujoco_warp.forward(self.warp_model, self.warp_data)
        return self._obs_from_arrays(self.warp_data.qpos.numpy(), self.warp_data.qvel.numpy())

    def reset_worlds(self, mask: np.ndarray):
        mask = np.asarray(mask, dtype=bool)
        if not np.any(mask):
            return self._obs_from_arrays(self.warp_data.qpos.numpy(), self.warp_data.qvel.numpy())
        qpos = self.warp_data.qpos.numpy()
        qvel = self.warp_data.qvel.numpy()
        rows = np.flatnonzero(mask)
        qpos[mask] = self.home_qpos[None, :]
        qvel[mask] = 0.0
        self.step_counts[mask] = 0
        self.command[mask, 0] = self.rng.uniform(0.4, 1.2, size=len(rows))
        self.command[mask, 1] = self.rng.uniform(-0.25, 0.25, size=len(rows))
        self.command[mask, 2] = self.rng.uniform(-0.4, 0.4, size=len(rows))
        qpos[mask, 0] += self.rng.uniform(-0.15, 0.15, size=len(rows)).astype(np.float32)
        qpos[mask, 1] += self.rng.uniform(-0.15, 0.15, size=len(rows)).astype(np.float32)
        qpos[mask, 2] = np.maximum(qpos[mask, 2], 0.27 + np.asarray([_terrain_height(float(x), float(y)) for x, y in qpos[mask, :2]], dtype=np.float32))
        qpos[np.ix_(rows, self.qpos_adrs)] += self.rng.normal(0.0, 0.025, size=(len(rows), self.model.nu)).astype(np.float32)
        self.last_actions[mask] = 0.0
        self.warp_data.qpos.assign(qpos)
        self.warp_data.qvel.assign(qvel)
        self.mujoco_warp.forward(self.warp_model, self.warp_data)
        return self._obs_from_arrays(self.warp_data.qpos.numpy(), self.warp_data.qvel.numpy())

    def step(self, actions: np.ndarray):
        actions = np.clip(np.asarray(actions, dtype=np.float32), -1.0, 1.0)
        torque_limits = np.asarray(self.model.actuator_ctrlrange[: self.model.nu], dtype=np.float32)
        for _ in range(CONTROL_DECIMATION):
            qpos = self.warp_data.qpos.numpy()
            qvel = self.warp_data.qvel.numpy()
            target = np.clip(self.default_joint_pos[None, :] + ACTION_SCALE * actions, self.joint_lower[None, :], self.joint_upper[None, :])
            torque = 35.0 * (target - qpos[:, self.qpos_adrs]) - 1.2 * qvel[:, self.qvel_adrs]
            self.warp_data.ctrl.assign(np.clip(torque, torque_limits[:, 0], torque_limits[:, 1]).astype(np.float32))
            self.mujoco_warp.step(self.warp_model, self.warp_data)
        self.step_counts += 1
        qpos = self.warp_data.qpos.numpy()
        qvel = self.warp_data.qvel.numpy()
        lin_vel = qvel[:, :3]
        ang_vel = qvel[:, 3:6]
        linear_error = np.linalg.norm(self.command[:, :2] - lin_vel[:, :2], axis=1).astype(np.float32)
        angular_error = np.abs(self.command[:, 2] - ang_vel[:, 2]).astype(np.float32)
        projected_gravity = self._projected_gravity_batch(qpos)
        orientation_penalty = np.linalg.norm(projected_gravity[:, :2], axis=1).astype(np.float32)
        energy = np.mean(np.square(self.warp_data.ctrl.numpy() / np.maximum(1.0, np.abs(torque_limits[:, 1]))), axis=1).astype(np.float32)
        action_rate = np.mean(np.square(actions - self.last_actions), axis=1).astype(np.float32)
        terrain_z = np.asarray([_terrain_height(float(x), float(y)) for x, y in qpos[:, :2]], dtype=np.float32)
        self.height_error = qpos[:, 2] - (0.27 + terrain_z)
        foot_slip = (np.abs(lin_vel[:, 1]) + 0.15 * np.linalg.norm(ang_vel[:, :2], axis=1)).astype(np.float32)
        foot_clearance = np.maximum(0.0, np.minimum(0.12, self.height_error)).astype(np.float32)
        fall = (qpos[:, 2] < 0.16) | (orientation_penalty > 0.9)
        tracking_error = (linear_error + 0.35 * angular_error).astype(np.float32)
        rewards = (
            1.8 * np.exp(-2.5 * linear_error)
            + 0.8 * np.exp(-2.0 * angular_error)
            + 0.15
            - 0.45 * orientation_penalty
            - 0.04 * energy
            - 0.05 * action_rate
            - 0.12 * foot_slip
            + 0.04 * foot_clearance
            - fall.astype(np.float32)
        ).astype(np.float32)
        dones = fall | (self.step_counts >= self.max_steps)
        self.last_actions = actions
        self.last_tracking_error = tracking_error
        infos = {
            "mpkpe": tracking_error,
            "command_tracking_error": tracking_error,
            "tracking_error": tracking_error,
            "energy_cost": energy,
            "fall_rate": fall.astype(np.float32),
            "physics_backend": self.physics_backend,
            "is_success": (tracking_error < 0.45) & ~fall,
        }
        return self._obs_from_arrays(qpos, qvel), rewards, dones.astype(bool), infos

    def close(self) -> None:
        return None


def make_external_env(benchmark: Any, control_type: str | None = None, reward_recipe: str | None = None) -> gym.Env[Any, Any]:
    del control_type
    env = UnitreeGo2LowerLevelEnv(
        render_mode=benchmark.env_kwargs.get("render_mode"),
        max_steps=int(getattr(benchmark, "max_steps", 80)),
    )
    return RewardRecipeWrapper(env, reward_recipe or REWARD_RECIPE)


class Agent(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, HIDDEN_SIZE)),
            nn.Tanh(),
            layer_init(nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)),
            nn.Tanh(),
            layer_init(nn.Linear(HIDDEN_SIZE, action_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, action_dim) * math.log(ACTION_STD_INIT))
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, HIDDEN_SIZE)),
            nn.Tanh(),
            layer_init(nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)),
            nn.Tanh(),
            layer_init(nn.Linear(HIDDEN_SIZE, 1), std=1.0),
        )

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs)

    def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor | None = None):
        mean = self.actor_mean(obs)
        logstd = self.actor_logstd.expand_as(mean)
        probs = Normal(mean, logstd.exp())
        if action is None:
            action = probs.rsample()
        clipped_action = torch.clamp(action, -1.0, 1.0)
        return clipped_action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(obs).squeeze(1)

    def act(self, obs: Any, deterministic: bool = True) -> np.ndarray:
        obs_tensor = torch.as_tensor(np.asarray(obs, dtype=np.float32).reshape(1, -1))
        with torch.no_grad():
            mean = self.actor_mean(obs_tensor)
            action = mean if deterministic else self.get_action_and_value(obs_tensor)[0]
        return torch.clamp(action, -1.0, 1.0).cpu().numpy()[0]


def train_agent(
    benchmark: Any,
    make_env,
    candidate: Any,
    device: str,
    init_checkpoint: Path | None = None,
    live_callback=None,
):
    del candidate, init_checkpoint, live_callback
    torch_device = torch.device("cuda" if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu")
    num_envs = int(benchmark.env_kwargs.get("num_envs", NUM_ENVS))
    num_steps = int(benchmark.env_kwargs.get("steps_per_env_per_iteration", NUM_STEPS))
    budget_seconds = getattr(benchmark, "train_seconds", None)
    updates = max(1, int(getattr(benchmark, "train_episodes", 1)))
    if budget_seconds is not None:
        updates = max(updates, 1_000_000)
    if str(benchmark.env_kwargs.get("backend", "")).startswith("mujoco_warp") and num_envs > 1:
        probe_env = make_env()
        try:
            base_env = probe_env.unwrapped if hasattr(probe_env, "unwrapped") else probe_env
            make_vectorized = getattr(base_env, "make_vectorized", None)
            if make_vectorized is not None:
                vector_env = make_vectorized(num_envs, int(benchmark.train_seed))
                if vector_env is not None:
                    probe_env.close()
                    return _train_agent_vectorized(benchmark, vector_env, torch_device, num_steps, updates, budget_seconds)
        finally:
            try:
                probe_env.close()
            except Exception:
                pass
    envs = [make_env() for _ in range(num_envs)]
    obs_list = [env.reset(seed=int(benchmark.train_seed) + idx)[0] for idx, env in enumerate(envs)]
    obs = np.stack(obs_list).astype(np.float32)
    obs_dim = int(obs.shape[1])
    action_dim = int(envs[0].action_space.shape[0])
    agent = Agent(obs_dim, action_dim).to(torch_device)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)
    records = []
    global_step = 0
    completed = 0
    started_at = time.time()
    stop_reason = "lower_level_cleanrl_complete"
    for update in range(updates):
        obs_buf = torch.zeros((num_steps, num_envs, obs_dim), device=torch_device)
        actions_buf = torch.zeros((num_steps, num_envs, action_dim), device=torch_device)
        logprobs_buf = torch.zeros((num_steps, num_envs), device=torch_device)
        rewards_buf = torch.zeros((num_steps, num_envs), device=torch_device)
        dones_buf = torch.zeros((num_steps, num_envs), device=torch_device)
        values_buf = torch.zeros((num_steps, num_envs), device=torch_device)
        mpkpes = []
        for step in range(num_steps):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=torch_device)
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(obs_tensor)
            action_np = action.cpu().numpy()
            obs_buf[step] = obs_tensor
            actions_buf[step] = action
            logprobs_buf[step] = logprob
            values_buf[step] = value
            next_obs = []
            for env_idx, env in enumerate(envs):
                new_obs, reward, terminated, truncated, info = env.step(action_np[env_idx])
                done = bool(terminated or truncated)
                rewards_buf[step, env_idx] = float(reward)
                dones_buf[step, env_idx] = float(done)
                mpkpes.append(float(info.get("mpkpe", 0.0)))
                if done:
                    completed += 1
                    new_obs, _ = env.reset(seed=int(benchmark.train_seed) + completed + env_idx)
                next_obs.append(new_obs)
            obs = np.stack(next_obs).astype(np.float32)
            global_step += num_envs
        with torch.no_grad():
            next_value = agent.get_value(torch.as_tensor(obs, dtype=torch.float32, device=torch_device)).reshape(1, -1)
            advantages = torch.zeros_like(rewards_buf, device=torch_device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                nextnonterminal = 1.0 - (dones_buf[t] if t == num_steps - 1 else dones_buf[t + 1])
                nextvalues = next_value if t == num_steps - 1 else values_buf[t + 1]
                delta = rewards_buf[t] + GAMMA * nextvalues * nextnonterminal - values_buf[t]
                advantages[t] = lastgaelam = delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
            returns = advantages + values_buf
        b_obs = obs_buf.reshape((-1, obs_dim))
        b_actions = actions_buf.reshape((-1, action_dim))
        b_logprobs = logprobs_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        batch_size = num_envs * num_steps
        minibatch_size = max(1, batch_size // NUM_MINIBATCHES)
        inds = np.arange(batch_size)
        for _ in range(UPDATE_EPOCHS):
            np.random.shuffle(inds)
            for start in range(0, batch_size, minibatch_size):
                mb_inds = inds[start : start + minibatch_size]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                mb_adv = b_advantages[mb_inds]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg_loss = torch.max(-mb_adv * ratio, -mb_adv * torch.clamp(ratio, 1 - CLIP_COEF, 1 + CLIP_COEF)).mean()
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                loss = pg_loss - ENT_COEF * entropy.mean() + VF_COEF * v_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                optimizer.step()
        records.append(
            make_train_collection_window_record(
                episode=update + 1,
                return_value=float(rewards_buf.sum(dim=0).mean().cpu()),
                length=float(num_steps),
                episodes_in_window=num_envs,
                success=bool(np.mean(mpkpes) < 0.45) if mpkpes else False,
                step=global_step,
                elapsed_seconds=time.time() - started_at,
                env_steps_in_window=num_envs * num_steps,
                info_metrics={
                    "command_tracking_error": float(np.mean(mpkpes)) if mpkpes else 0.0,
                    "mpkpe": float(np.mean(mpkpes)) if mpkpes else 0.0,
                    "num_envs": float(num_envs),
                },
            )
        )
        if budget_seconds is not None and time.time() - started_at >= float(budget_seconds):
            stop_reason = "time_budget_exhausted"
            break
    physics_backend = str(getattr(envs[0].unwrapped, "physics_backend", "unknown")) if envs else "unknown"
    for env in envs:
        env.close()
    gradient_updates = int(len(records) * UPDATE_EPOCHS * NUM_MINIBATCHES)
    last_metrics = dict(records[-1]["info_metrics"]) if records else {}
    last_metrics["gradient_updates"] = float(gradient_updates)
    summary = {
        "episode_records": records,
        "total_steps": int(global_step),
        "env_steps": int(global_step),
        "episodes_completed": int(completed + len(records) * num_envs),
        "completed_episodes": int(completed + len(records) * num_envs),
        "episode_batches": len(records),
        "gradient_updates": gradient_updates,
        "last_metrics": last_metrics,
        "physics_backend": physics_backend,
        "stop_reason": stop_reason,
    }
    return agent, summary


def _train_agent_vectorized(
    benchmark: Any,
    vector_env: Any,
    torch_device: torch.device,
    num_steps: int,
    updates: int,
    budget_seconds: float | None,
):
    obs = vector_env.reset(seed=int(benchmark.train_seed))
    num_envs = int(vector_env.num_envs)
    obs_dim = int(obs.shape[1])
    action_dim = int(vector_env.action_space.shape[0])
    agent = Agent(obs_dim, action_dim).to(torch_device)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)
    records = []
    global_step = 0
    completed = 0
    started_at = time.time()
    stop_reason = "lower_level_cleanrl_complete"
    for update in range(updates):
        obs_buf = torch.zeros((num_steps, num_envs, obs_dim), device=torch_device)
        actions_buf = torch.zeros((num_steps, num_envs, action_dim), device=torch_device)
        logprobs_buf = torch.zeros((num_steps, num_envs), device=torch_device)
        rewards_buf = torch.zeros((num_steps, num_envs), device=torch_device)
        dones_buf = torch.zeros((num_steps, num_envs), device=torch_device)
        values_buf = torch.zeros((num_steps, num_envs), device=torch_device)
        mpkpes: list[float] = []
        for step in range(num_steps):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=torch_device)
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(obs_tensor)
            action_np = action.cpu().numpy()
            next_obs, rewards, dones, infos = vector_env.step(action_np)
            obs_buf[step] = obs_tensor
            actions_buf[step] = action
            logprobs_buf[step] = logprob
            values_buf[step] = value
            rewards_buf[step] = torch.as_tensor(rewards, dtype=torch.float32, device=torch_device)
            dones_buf[step] = torch.as_tensor(dones.astype(np.float32), dtype=torch.float32, device=torch_device)
            mpkpes.extend([float(value) for value in np.asarray(infos.get("mpkpe", []), dtype=np.float32).reshape(-1)])
            completed += int(np.sum(dones))
            obs = vector_env.reset_worlds(dones) if np.any(dones) else next_obs
            global_step += num_envs
        with torch.no_grad():
            next_value = agent.get_value(torch.as_tensor(obs, dtype=torch.float32, device=torch_device)).reshape(1, -1)
            advantages = torch.zeros_like(rewards_buf, device=torch_device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                nextnonterminal = 1.0 - (dones_buf[t] if t == num_steps - 1 else dones_buf[t + 1])
                nextvalues = next_value if t == num_steps - 1 else values_buf[t + 1]
                delta = rewards_buf[t] + GAMMA * nextvalues * nextnonterminal - values_buf[t]
                advantages[t] = lastgaelam = delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
            returns = advantages + values_buf
        b_obs = obs_buf.reshape((-1, obs_dim))
        b_actions = actions_buf.reshape((-1, action_dim))
        b_logprobs = logprobs_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        batch_size = num_envs * num_steps
        minibatch_size = max(1, batch_size // NUM_MINIBATCHES)
        inds = np.arange(batch_size)
        for _ in range(UPDATE_EPOCHS):
            np.random.shuffle(inds)
            for start in range(0, batch_size, minibatch_size):
                mb_inds = inds[start : start + minibatch_size]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                mb_adv = b_advantages[mb_inds]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg_loss = torch.max(-mb_adv * ratio, -mb_adv * torch.clamp(ratio, 1 - CLIP_COEF, 1 + CLIP_COEF)).mean()
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                loss = pg_loss - ENT_COEF * entropy.mean() + VF_COEF * v_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                optimizer.step()
        records.append(
            make_train_collection_window_record(
                episode=update + 1,
                return_value=float(rewards_buf.sum(dim=0).mean().cpu()),
                length=float(num_steps),
                episodes_in_window=num_envs,
                success=bool(np.mean(mpkpes) < 0.45) if mpkpes else False,
                step=global_step,
                elapsed_seconds=time.time() - started_at,
                env_steps_in_window=num_envs * num_steps,
                info_metrics={
                    "command_tracking_error": float(np.mean(mpkpes)) if mpkpes else 0.0,
                    "mpkpe": float(np.mean(mpkpes)) if mpkpes else 0.0,
                    "num_envs": float(num_envs),
                    "mujoco_warp_vectorized": 1.0,
                },
            )
        )
        if budget_seconds is not None and time.time() - started_at >= float(budget_seconds):
            stop_reason = "time_budget_exhausted"
            break
    vector_env.close()
    gradient_updates = int(len(records) * UPDATE_EPOCHS * NUM_MINIBATCHES)
    last_metrics = dict(records[-1]["info_metrics"]) if records else {}
    last_metrics["gradient_updates"] = float(gradient_updates)
    summary = {
        "episode_records": records,
        "total_steps": int(global_step),
        "env_steps": int(global_step),
        "episodes_completed": int(completed + len(records) * num_envs),
        "completed_episodes": int(completed + len(records) * num_envs),
        "episode_batches": len(records),
        "gradient_updates": gradient_updates,
        "last_metrics": last_metrics,
        "physics_backend": str(getattr(vector_env, "physics_backend", "mujoco_warp")),
        "vectorized_backend": "mujoco_warp",
        "num_envs": int(num_envs),
        "stop_reason": stop_reason,
    }
    return agent, summary


def evaluate_agent(agent: Agent, benchmark: Any, eval_cases: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    records = []
    cases = eval_cases or []
    for idx in range(int(benchmark.eval_episodes)):
        env = make_external_env(benchmark)
        case = cases[idx] if idx < len(cases) else {}
        obs, _ = env.reset(seed=int(benchmark.eval_seed_start) + idx, options={"fixed_case": case} if case else None)
        total_return = 0.0
        info: dict[str, Any] = {}
        for step in range(int(benchmark.max_steps)):
            obs, reward, terminated, truncated, info = env.step(agent.act(obs, deterministic=True))
            total_return += float(reward)
            if terminated or truncated:
                break
        env.close()
        records.append(
            {
                "episode": idx + 1,
                "seed": int(benchmark.eval_seed_start) + idx,
                "return": float(total_return),
                "length": step + 1,
                "case_label": str(case.get("name", f"case-{idx + 1:02d}")),
                "info_metrics": {
                    "command_tracking_error": float(info.get("command_tracking_error", 0.0)),
                    "energy_cost": float(info.get("energy_cost", 0.0)),
                    "fall_rate": float(info.get("fall_rate", 0.0)),
                },
            }
        )
    return {
        "episodes": len(records),
        "avg_return": float(np.mean([record["return"] for record in records])) if records else 0.0,
        "avg_length": float(np.mean([record["length"] for record in records])) if records else 0.0,
        "avg_command_tracking_error": float(np.mean([record["info_metrics"]["command_tracking_error"] for record in records])) if records else 0.0,
        "physics_backend": str(info.get("physics_backend", "unknown")) if records else "unknown",
        "metric_source": "lower_level_rollout_reward",
        "episode_records": records,
    }


def render_policy(agent: Agent, benchmark: Any, out_dir: Path) -> dict[str, Any]:
    frame_dir = out_dir / "trajectories" / "sample_000001"
    frame_dir.mkdir(parents=True, exist_ok=True)
    env = make_external_env(benchmark)
    obs, _ = env.reset(seed=int(benchmark.eval_seed_start))
    frames = []
    for idx in range(min(24, int(benchmark.max_steps))):
        frame_path = frame_dir / f"frame_{idx:04d}.jpg"
        imageio.imwrite(frame_path, env.render())
        frames.append(str(frame_path))
        obs, _, terminated, truncated, _ = env.step(agent.act(obs, deterministic=True))
        if terminated or truncated:
            break
    env.close()
    gif_path = frame_dir / "rollout.gif"
    imageio.mimsave(gif_path, [imageio.imread(frame) for frame in frames], duration=0.08)
    manifest = {"status": "completed", "sample_index": 1, "frames": frames, "gif_path": str(gif_path), "frame_count": len(frames)}
    (frame_dir / "manifest.json").write_text(__import__("json").dumps(manifest, indent=2), encoding="utf-8")
    live_frame = out_dir / "current_run_frame.jpg"
    if frames:
        shutil.copy2(frames[-1], live_frame)
    return {
        "media_available": bool(frames),
        "live_frame_path": str(live_frame),
        "trajectory_manifest_path": str(frame_dir / "manifest.json"),
        "trajectory_latest_frame_path": str(live_frame),
        "visual": {
            "live_frame_path": str(live_frame),
            "trajectory_manifest_path": str(frame_dir / "manifest.json"),
            "trajectory_latest_frame_path": str(live_frame),
            "sampled_status": "completed",
            "latest_sample_index": 1,
        },
    }


def save_agent_checkpoint(agent: Agent, path: Path, metadata: dict[str, Any] | None = None) -> None:
    obs_dim = int(agent.actor_mean[0].in_features)
    action_dim = int(agent.actor_mean[-1].out_features)
    torch.save({"state_dict": agent.state_dict(), "metadata": metadata or {}, "obs_dim": obs_dim, "action_dim": action_dim}, path)


def load_agent_checkpoint(path: Path, benchmark: Any | None = None) -> Agent:
    del benchmark
    payload = torch.load(path, map_location="cpu")
    agent = Agent(int(payload["obs_dim"]), int(payload["action_dim"]))
    agent.load_state_dict(payload["state_dict"])
    agent.eval()
    return agent
