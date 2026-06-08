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


# Harness invariants:
# - This seed is the mutable lower-level training recipe. Keep env construction,
#   policy architecture, rollout collection, PPO losses, reward terms, and
#   logging visible in this file.
# - The external backend only stages and runs this file on the selected target.
# - UnitreeG1LowerLevelEnv loads the real MJLab/MuJoCo G1 scene when the
#   optional assets are present on the execution target. The compact kinematic
#   fallback exists only so the open-source smoke tests can run without private
#   Unitree assets.

EXP_NAME = "unitree_g1_motion_mirror_lower_level_cleanrl_ppo"
ALGORITHM = "ppo"
CONTROL_TYPE = None
REWARD_RECIPE = "mpkpe_tracking"

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
RENDER_WIDTH = 720
RENDER_HEIGHT = 480


def _repo_root() -> Path:
    # The external backend stages this seed as candidate_trainable.py but runs it
    # from the repository root on the target machine.
    return Path.cwd()


def _g1_xml_path() -> Path:
    return (
        _repo_root()
        / ".external"
        / "unitree_rl_mjlab"
        / "src"
        / "assets"
        / "robots"
        / "unitree_g1"
        / "xmls"
        / "scene_g1.xml"
    )


def _g1_motion_path() -> Path:
    return _repo_root() / "autoresearch_runs" / "source_motions" / "pbhc_side_kick_mjlab_motion.npz"


def _maybe_import_mujoco():
    try:
        import mujoco  # type: ignore

        return mujoco
    except Exception:
        return None


def _maybe_import_mujoco_warp():
    try:
        import mujoco_warp  # type: ignore

        return mujoco_warp
    except Exception:
        return None


def _prepare_model_for_mujoco_warp(mujoco_module: Any, model: Any) -> None:
    # MuJoCo Warp 3.8 rejects non-zero geom margins when MULTICCD is enabled.
    if hasattr(mujoco_module, "mjtDisableBit") and hasattr(mujoco_module.mjtDisableBit, "mjDSBL_MULTICCD"):
        model.opt.disableflags |= int(mujoco_module.mjtDisableBit.mjDSBL_MULTICCD.value)
    if hasattr(model, "geom_margin"):
        model.geom_margin[:] = 0.0


def _quat_distance(q1: np.ndarray, q2: np.ndarray) -> float:
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)
    denom = max(1e-8, float(np.linalg.norm(q1) * np.linalg.norm(q2)))
    return float(1.0 - min(1.0, abs(float(np.dot(q1, q2))) / denom))


def _joint_home_value(name: str) -> float:
    if "hip_pitch" in name:
        return -0.10
    if "knee" in name:
        return 0.30
    if "ankle_pitch" in name:
        return -0.20
    if "shoulder_pitch" in name:
        return 0.35
    if "shoulder_roll" in name:
        return 0.18 if name.startswith("left_") else -0.18
    if "elbow" in name:
        return 0.87
    return 0.0


def _load_motion_clip() -> dict[str, np.ndarray]:
    motion_path = _g1_motion_path()
    if not motion_path.exists():
        return {}
    loaded = np.load(motion_path)
    return {key: np.asarray(loaded[key]) for key in loaded.files}


def get_candidate() -> str:
    return (
        "Lower-level CleanRL-style PPO seed for Unitree G1 motion mirroring. "
        "The env adapter, reward terms, vector rollout collection, GAE, PPO "
        "losses, and checkpointing are executable code in this file."
    )


def layer_init(layer: nn.Linear, std: float = math.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class UnitreeG1LowerLevelEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, render_mode: str | None = None, max_steps: int = 80) -> None:
        self.render_mode = render_mode
        self.max_steps = int(max_steps)
        self.rng = np.random.default_rng(0)
        self.step_count = 0
        self.phase = 0.0
        self.last_mpkpe = 1.0
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
        self.motion: dict[str, np.ndarray] = {}
        self.motion_frame = 0
        self.fixed_case_start_frame: int | None = None
        self.fixed_case_end_frame: int | None = None
        self.fixed_case_max_steps: int | None = None

        xml_path = _g1_xml_path()
        if self._mujoco is not None and xml_path.exists():
            self._init_mujoco(xml_path)
        else:
            self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(21,), dtype=np.float32)
            self.qpos = np.zeros(6, dtype=np.float32)
            self.qvel = np.zeros(6, dtype=np.float32)
            self.last_action = np.zeros(6, dtype=np.float32)

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
        names = [
            self._mujoco.mj_id2name(self.model, self._mujoco.mjtObj.mjOBJ_JOINT, int(jid)) or ""
            for jid in actuator_joint_ids
        ]
        self.default_joint_pos = np.asarray([_joint_home_value(name) for name in names], dtype=np.float32)
        self.home_qpos[self.qpos_adrs] = self.default_joint_pos.astype(np.float64)
        self.last_action = np.zeros(self.model.nu, dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32)
        obs_dim = 2 + 1 + 3 + 4 + 6 + 4 * self.model.nu
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.physics_backend = "mujoco_unitree_g1_scene"
        self.motion = _load_motion_clip()

    def make_vectorized(self, num_envs: int, seed: int = 0):
        if self.model is None:
            return None
        try:
            return UnitreeG1MujocoWarpVectorEnv(num_envs=num_envs, max_steps=self.max_steps, seed=seed)
        except Exception:
            return None

    def _joint_pos(self) -> np.ndarray:
        assert self.data is not None
        return np.asarray(self.data.qpos[self.qpos_adrs], dtype=np.float32)

    def _joint_vel(self) -> np.ndarray:
        assert self.data is not None
        return np.asarray(self.data.qvel[self.qvel_adrs], dtype=np.float32)

    def _target_mujoco(self) -> np.ndarray:
        if "joint_pos" in self.motion and len(self.motion["joint_pos"]):
            return np.asarray(self.motion["joint_pos"][self.motion_frame % len(self.motion["joint_pos"])], dtype=np.float32)
        phase = self.phase
        target = self.default_joint_pos.copy()
        if target.size >= 12:
            target[:12] += np.asarray(
                [
                    0.10 * math.sin(phase),
                    0.05 * math.cos(phase),
                    0.04 * math.sin(phase + 0.4),
                    0.12 + 0.08 * math.sin(phase + 0.6),
                    -0.06 * math.sin(phase),
                    0.02 * math.cos(phase),
                    -0.10 * math.sin(phase),
                    -0.05 * math.cos(phase),
                    -0.04 * math.sin(phase + 0.4),
                    0.12 - 0.08 * math.sin(phase + 0.6),
                    0.06 * math.sin(phase),
                    -0.02 * math.cos(phase),
                ],
                dtype=np.float32,
            )
        return target

    def _mujoco_mpkpe(self) -> tuple[float, float]:
        assert self.data is not None
        joint_error = float(np.sqrt(np.mean(np.square(self._target_mujoco() - self._joint_pos()))))
        if "body_pos_w" not in self.motion or self.model is None:
            return joint_error, joint_error
        target_body = np.asarray(self.motion["body_pos_w"][self.motion_frame % len(self.motion["body_pos_w"])], dtype=np.float32)
        n = min(int(self.model.nbody), int(target_body.shape[0]))
        if n <= 1:
            return joint_error, joint_error
        sim = np.asarray(self.data.xpos[:n], dtype=np.float32)
        sim_rel = sim - sim[0:1]
        target_rel = target_body[:n] - target_body[0:1]
        body_mpkpe = float(np.mean(np.linalg.norm(sim_rel - target_rel, axis=1)))
        return body_mpkpe, joint_error

    def _obs_mujoco(self) -> np.ndarray:
        assert self.data is not None
        joint_pos = self._joint_pos()
        joint_vel = self._joint_vel()
        target = self._target_mujoco()
        root_pos = np.asarray(self.data.qpos[:3], dtype=np.float32)
        root_quat = np.asarray(self.data.qpos[3:7], dtype=np.float32)
        root_vel = np.asarray(self.data.qvel[:6], dtype=np.float32)
        return np.concatenate(
            [
                np.asarray(
                    [
                        math.sin(self.phase),
                        math.cos(self.phase),
                        self.step_count / max(1, self.max_steps),
                    ],
                    dtype=np.float32,
                ),
                root_pos,
                root_quat,
                root_vel,
                joint_pos - self.default_joint_pos,
                joint_vel,
                target - joint_pos,
                self.last_action,
            ]
        ).astype(np.float32)

    def _target(self) -> np.ndarray:
        phase = self.phase
        return np.asarray(
            [
                0.35 * math.sin(phase),
                0.20 * math.cos(phase),
                0.45 * math.sin(phase + 0.8),
                -0.45 * math.sin(phase + 0.8),
                0.55 * math.sin(phase + 1.6),
                -0.55 * math.sin(phase + 1.6),
            ],
            dtype=np.float32,
        )

    def _obs(self) -> np.ndarray:
        target = self._target()
        return np.concatenate(
            [
                np.asarray([math.sin(self.phase), math.cos(self.phase), self.step_count / max(1, self.max_steps)], dtype=np.float32),
                self.qpos,
                self.qvel,
                target - self.qpos,
            ]
        ).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        raw_fixed_case = (options or {}).get("fixed_case", {})
        fixed_case = raw_fixed_case if isinstance(raw_fixed_case, dict) else {}
        start_frame = fixed_case.get("start_frame")
        end_frame = fixed_case.get("end_frame")
        self.fixed_case_start_frame = int(start_frame) if isinstance(start_frame, (int, float)) else None
        self.fixed_case_end_frame = int(end_frame) if isinstance(end_frame, (int, float)) else None
        if self.fixed_case_start_frame is not None and self.fixed_case_end_frame is not None:
            self.fixed_case_max_steps = max(1, self.fixed_case_end_frame - self.fixed_case_start_frame)
        else:
            self.fixed_case_max_steps = None
        if "phase" in fixed_case:
            self.phase = float(fixed_case["phase"])
        elif self.fixed_case_start_frame is not None:
            clip_len = len(self.motion["joint_pos"]) if "joint_pos" in self.motion and len(self.motion["joint_pos"]) else 0
            if clip_len:
                self.phase = 2.0 * math.pi * (self.fixed_case_start_frame % clip_len) / clip_len
            else:
                self.phase = 0.10 * self.fixed_case_start_frame
        else:
            self.phase = float(self.rng.uniform(0.0, 2.0 * math.pi))
        self.step_count = 0
        if self.model is not None and self.data is not None:
            self.data.qpos[:] = self.home_qpos
            self.data.qvel[:] = 0.0
            self.data.qpos[:2] += self.rng.normal(0.0, 0.01, size=2)
            self.data.qpos[2] += self.rng.normal(0.0, 0.005)
            if "joint_pos" in self.motion and len(self.motion["joint_pos"]):
                if self.fixed_case_start_frame is not None:
                    self.motion_frame = self.fixed_case_start_frame % len(self.motion["joint_pos"])
                else:
                    self.motion_frame = int((self.phase / (2.0 * math.pi)) * len(self.motion["joint_pos"])) % len(self.motion["joint_pos"])
                self.data.qpos[self.qpos_adrs] = self._target_mujoco().astype(np.float64)
            self._mujoco.mj_forward(self.model, self.data)
            self.last_action = np.zeros(self.model.nu, dtype=np.float32)
            self.last_mpkpe, joint_error = self._mujoco_mpkpe()
            return self._obs_mujoco(), {
                "mpkpe": self.last_mpkpe,
                "joint_tracking_error": joint_error,
                "physics_backend": self.physics_backend,
            }
        self.qpos = self.rng.normal(0.0, 0.05, size=6).astype(np.float32)
        self.qvel = np.zeros(6, dtype=np.float32)
        self.last_action = np.zeros(6, dtype=np.float32)
        self.last_mpkpe = float(np.linalg.norm(self._target() - self.qpos) / math.sqrt(6))
        return self._obs(), {"mpkpe": self.last_mpkpe, "physics_backend": self.physics_backend}

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        if self.model is not None and self.data is not None:
            target = np.clip(self.default_joint_pos + ACTION_SCALE * action, self.joint_lower, self.joint_upper)
            kp = np.full(self.model.nu, 55.0, dtype=np.float32)
            kd = np.full(self.model.nu, 2.2, dtype=np.float32)
            torque_limits = np.asarray(self.model.actuator_ctrlrange[: self.model.nu], dtype=np.float32)
            for _ in range(CONTROL_DECIMATION):
                torque = kp * (target - self._joint_pos()) - kd * self._joint_vel()
                self.data.ctrl[:] = np.clip(torque, torque_limits[:, 0], torque_limits[:, 1])
                self._mujoco.mj_step(self.model, self.data)
            self.step_count += 1
            if "joint_pos" in self.motion and len(self.motion["joint_pos"]):
                self.motion_frame = (self.motion_frame + 1) % len(self.motion["joint_pos"])
                self.phase = 2.0 * math.pi * self.motion_frame / max(1, len(self.motion["joint_pos"]))
            else:
                self.phase += 0.10
            mpkpe, joint_error = self._mujoco_mpkpe()
            target_root = (
                np.asarray(self.motion["body_pos_w"][self.motion_frame % len(self.motion["body_pos_w"]), 0], dtype=np.float32)
                if "body_pos_w" in self.motion
                else np.asarray([0.0, 0.0, 0.793], dtype=np.float32)
            )
            root_pos_error = float(np.linalg.norm(np.asarray(self.data.qpos[:3], dtype=np.float32) - target_root))
            root_ori_error = _quat_distance(np.asarray(self.data.qpos[3:7]), np.asarray([1.0, 0.0, 0.0, 0.0]))
            action_rate = float(np.mean(np.square(action - self.last_action)))
            limit_margin = np.minimum(self._joint_pos() - self.joint_lower, self.joint_upper - self._joint_pos())
            joint_limit = float(np.mean(limit_margin < 0.04))
            reward = (
                1.20 * math.exp(-8.0 * mpkpe)
                + 0.55 * math.exp(-6.0 * joint_error)
                + 0.30 * math.exp(-5.0 * root_pos_error)
                + 0.15 * math.exp(-4.0 * root_ori_error)
                - 0.03 * action_rate
                - 0.08 * joint_limit
            )
            self.last_action = action
            self.last_mpkpe = mpkpe
            terminated = bool(mpkpe > 1.25 or self.data.qpos[2] < 0.35)
            truncated = self.step_count >= self.max_steps
            if self.fixed_case_max_steps is not None:
                truncated = truncated or self.step_count >= self.fixed_case_max_steps
            info = {
                "mpkpe": mpkpe,
                "tracking_error": mpkpe,
                "joint_tracking_error": joint_error,
                "root_pos_error": root_pos_error,
                "root_ori_error": root_ori_error,
                "action_rate": action_rate,
                "joint_limit": joint_limit,
                "physics_backend": self.physics_backend,
                "is_success": mpkpe < 0.22 and not terminated,
            }
            return self._obs_mujoco(), float(reward), terminated, truncated, info
        target = self._target()
        tracking_accel = 0.18 * (target - self.qpos)
        self.qvel = 0.82 * self.qvel + tracking_accel + 0.08 * action
        self.qpos = self.qpos + 0.12 * self.qvel
        self.phase += 0.10
        self.step_count += 1
        next_target = self._target()
        mpkpe = float(np.linalg.norm(next_target - self.qpos) / math.sqrt(6))
        action_rate = float(np.mean(np.square(action - self.last_action)))
        reward = -mpkpe - 0.03 * action_rate
        self.last_action = action
        self.last_mpkpe = mpkpe
        terminated = bool(mpkpe > 1.75)
        truncated = self.step_count >= self.max_steps
        if self.fixed_case_max_steps is not None:
            truncated = truncated or self.step_count >= self.fixed_case_max_steps
        info = {"mpkpe": mpkpe, "tracking_error": mpkpe, "action_rate": action_rate, "is_success": mpkpe < 0.18}
        return self._obs(), float(reward), terminated, truncated, info

    def render(self):
        if self.model is not None and self.data is not None:
            try:
                if self.renderer is None:
                    self.renderer = self._mujoco.Renderer(self.model, RENDER_HEIGHT, RENDER_WIDTH)
                self.renderer.update_scene(self.data)
                return np.asarray(self.renderer.render(), dtype=np.uint8)
            except Exception:
                return None
        return None

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


class RewardRecipeWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env[Any, Any], recipe: str | None = None) -> None:
        super().__init__(env)
        self.recipe = recipe or REWARD_RECIPE


class UnitreeG1MujocoWarpVectorEnv:
    def __init__(self, num_envs: int, max_steps: int = 80, seed: int = 0) -> None:
        self.num_envs = int(num_envs)
        self.max_steps = int(max_steps)
        self.rng = np.random.default_rng(seed)
        self.mujoco = _maybe_import_mujoco()
        self.mujoco_warp = _maybe_import_mujoco_warp()
        if self.mujoco is None or self.mujoco_warp is None:
            raise RuntimeError("mujoco_warp is not available")
        xml_path = _g1_xml_path()
        if not xml_path.exists():
            raise FileNotFoundError(xml_path)
        self.model = self.mujoco.MjModel.from_xml_path(str(xml_path))
        _prepare_model_for_mujoco_warp(self.mujoco, self.model)
        self.data0 = self.mujoco.MjData(self.model)
        actuator_joint_ids = np.asarray(self.model.actuator_trnid[: self.model.nu, 0], dtype=np.int32)
        self.qpos_adrs = np.asarray([self.model.jnt_qposadr[int(jid)] for jid in actuator_joint_ids], dtype=np.int32)
        self.qvel_adrs = np.asarray([self.model.jnt_dofadr[int(jid)] for jid in actuator_joint_ids], dtype=np.int32)
        self.joint_lower = self.model.jnt_range[actuator_joint_ids, 0].astype(np.float32)
        self.joint_upper = self.model.jnt_range[actuator_joint_ids, 1].astype(np.float32)
        names = [
            self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, int(jid)) or ""
            for jid in actuator_joint_ids
        ]
        self.default_joint_pos = np.asarray([_joint_home_value(name) for name in names], dtype=np.float32)
        self.home_qpos = np.asarray(self.model.qpos0, dtype=np.float32).copy()
        self.home_qpos[self.qpos_adrs] = self.default_joint_pos
        self.motion = _load_motion_clip()
        self.warp_model = self.mujoco_warp.put_model(self.model)
        self.warp_data = self.mujoco_warp.put_data(self.model, self.data0, nworld=self.num_envs)
        self.step_counts = np.zeros(self.num_envs, dtype=np.int32)
        self.phases = np.zeros(self.num_envs, dtype=np.float32)
        self.motion_frames = np.zeros(self.num_envs, dtype=np.int32)
        self.last_actions = np.zeros((self.num_envs, self.model.nu), dtype=np.float32)
        self.last_mpkpes = np.ones(self.num_envs, dtype=np.float32)
        self.physics_backend = "mujoco_warp_unitree_g1_scene"
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32)
        obs_dim = 2 + 1 + 3 + 4 + 6 + 4 * self.model.nu
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def _targets(self) -> np.ndarray:
        if "joint_pos" in self.motion and len(self.motion["joint_pos"]):
            return np.asarray(self.motion["joint_pos"][self.motion_frames % len(self.motion["joint_pos"])], dtype=np.float32)
        targets = np.repeat(self.default_joint_pos[None, :], self.num_envs, axis=0)
        if targets.shape[1] >= 12:
            phase = self.phases
            targets[:, 0] += 0.10 * np.sin(phase)
            targets[:, 1] += 0.05 * np.cos(phase)
            targets[:, 3] += 0.12 + 0.08 * np.sin(phase + 0.6)
            targets[:, 6] += -0.10 * np.sin(phase)
            targets[:, 7] += -0.05 * np.cos(phase)
            targets[:, 9] += 0.12 - 0.08 * np.sin(phase + 0.6)
        return targets.astype(np.float32)

    def _obs_from_arrays(self, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        joint_pos = qpos[:, self.qpos_adrs]
        joint_vel = qvel[:, self.qvel_adrs]
        return np.concatenate(
            [
                np.sin(self.phases)[:, None],
                np.cos(self.phases)[:, None],
                (self.step_counts / max(1, self.max_steps)).astype(np.float32)[:, None],
                qpos[:, :3],
                qpos[:, 3:7],
                qvel[:, :6],
                joint_pos - self.default_joint_pos[None, :],
                joint_vel,
                self._targets() - joint_pos,
                self.last_actions,
            ],
            axis=1,
        ).astype(np.float32)

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.step_counts[:] = 0
        self.phases[:] = self.rng.uniform(0.0, 2.0 * math.pi, size=self.num_envs).astype(np.float32)
        if "joint_pos" in self.motion and len(self.motion["joint_pos"]):
            self.motion_frames[:] = ((self.phases / (2.0 * math.pi)) * len(self.motion["joint_pos"])).astype(np.int32) % len(self.motion["joint_pos"])
        qpos = np.repeat(self.home_qpos[None, :], self.num_envs, axis=0).astype(np.float32)
        qvel = np.zeros((self.num_envs, self.model.nv), dtype=np.float32)
        qpos[:, :2] += self.rng.normal(0.0, 0.01, size=(self.num_envs, 2)).astype(np.float32)
        if "joint_pos" in self.motion and len(self.motion["joint_pos"]):
            qpos[:, self.qpos_adrs] = self._targets()
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
        count = int(np.sum(mask))
        qpos[mask] = self.home_qpos[None, :]
        qvel[mask] = 0.0
        self.step_counts[mask] = 0
        self.phases[mask] = self.rng.uniform(0.0, 2.0 * math.pi, size=count).astype(np.float32)
        if "joint_pos" in self.motion and len(self.motion["joint_pos"]):
            self.motion_frames[mask] = ((self.phases[mask] / (2.0 * math.pi)) * len(self.motion["joint_pos"])).astype(np.int32) % len(self.motion["joint_pos"])
            qpos[np.ix_(np.flatnonzero(mask), self.qpos_adrs)] = self._targets()[mask]
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
            joint_pos = qpos[:, self.qpos_adrs]
            joint_vel = qvel[:, self.qvel_adrs]
            targets = np.clip(self.default_joint_pos[None, :] + ACTION_SCALE * actions, self.joint_lower[None, :], self.joint_upper[None, :])
            torque = 55.0 * (targets - joint_pos) - 2.2 * joint_vel
            self.warp_data.ctrl.assign(np.clip(torque, torque_limits[:, 0], torque_limits[:, 1]).astype(np.float32))
            self.mujoco_warp.step(self.warp_model, self.warp_data)
        self.step_counts += 1
        if "joint_pos" in self.motion and len(self.motion["joint_pos"]):
            self.motion_frames = (self.motion_frames + 1) % len(self.motion["joint_pos"])
            self.phases = (2.0 * math.pi * self.motion_frames / max(1, len(self.motion["joint_pos"]))).astype(np.float32)
        else:
            self.phases += 0.10
        qpos = self.warp_data.qpos.numpy()
        qvel = self.warp_data.qvel.numpy()
        joint_error = np.sqrt(np.mean(np.square(self._targets() - qpos[:, self.qpos_adrs]), axis=1)).astype(np.float32)
        mpkpe = joint_error
        action_rate = np.mean(np.square(actions - self.last_actions), axis=1).astype(np.float32)
        rewards = (1.20 * np.exp(-8.0 * mpkpe) + 0.55 * np.exp(-6.0 * joint_error) - 0.03 * action_rate).astype(np.float32)
        dones = (mpkpe > 1.25) | (qpos[:, 2] < 0.35) | (self.step_counts >= self.max_steps)
        self.last_actions = actions
        self.last_mpkpes = mpkpe
        obs = self._obs_from_arrays(qpos, qvel)
        infos = {
            "mpkpe": mpkpe,
            "tracking_error": mpkpe,
            "action_rate": action_rate,
            "physics_backend": self.physics_backend,
            "is_success": (mpkpe < 0.22) & ~dones,
        }
        return obs, rewards, dones.astype(bool), infos

    def close(self) -> None:
        return None


def make_external_env(benchmark: Any, control_type: str | None = None, reward_recipe: str | None = None) -> gym.Env[Any, Any]:
    del control_type
    env = UnitreeG1LowerLevelEnv(
        render_mode=benchmark.env_kwargs.get("render_mode"),
        max_steps=int(benchmark.max_steps),
    )
    return RewardRecipeWrapper(env, reward_recipe)


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
        window_returns = np.zeros(num_envs, dtype=np.float32)
        window_lengths = np.zeros(num_envs, dtype=np.float32)
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
            done_flags = []
            for env_idx, env in enumerate(envs):
                new_obs, reward, terminated, truncated, info = env.step(action_np[env_idx])
                done = bool(terminated or truncated)
                rewards_buf[step, env_idx] = float(reward)
                dones_buf[step, env_idx] = float(done)
                window_returns[env_idx] += float(reward)
                window_lengths[env_idx] += 1.0
                mpkpes.append(float(info.get("mpkpe", 0.0)))
                if done:
                    completed += 1
                    new_obs, _ = env.reset(seed=int(benchmark.train_seed) + completed + env_idx)
                    window_lengths[env_idx] = 0.0
                    window_returns[env_idx] = 0.0
                next_obs.append(new_obs)
                done_flags.append(done)
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
        b_values = values_buf.reshape(-1)
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
                entropy_loss = entropy.mean()
                loss = pg_loss - ENT_COEF * entropy_loss + VF_COEF * v_loss
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
                success=bool(np.mean(mpkpes) < 0.22) if mpkpes else False,
                step=global_step,
                elapsed_seconds=time.time() - started_at,
                env_steps_in_window=num_envs * num_steps,
                info_metrics={"mpkpe": float(np.mean(mpkpes)) if mpkpes else 0.0, "num_envs": float(num_envs)},
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
                success=bool(np.mean(mpkpes) < 0.22) if mpkpes else False,
                step=global_step,
                elapsed_seconds=time.time() - started_at,
                env_steps_in_window=num_envs * num_steps,
                info_metrics={
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
                "success": bool(info.get("is_success", False)),
                "case_label": str(case.get("name", f"case-{idx + 1:02d}")),
                "info_metrics": {"mpkpe": float(info.get("mpkpe", 0.0)), "tracking_error": float(info.get("tracking_error", 0.0))},
            }
        )
    return {
        "episodes": len(records),
        "avg_return": float(np.mean([record["return"] for record in records])) if records else 0.0,
        "avg_length": float(np.mean([record["length"] for record in records])) if records else 0.0,
        "success_rate": float(np.mean([1.0 if record["success"] else 0.0 for record in records])) if records else 0.0,
        "avg_mpkpe": float(np.mean([record["info_metrics"]["mpkpe"] for record in records])) if records else 0.0,
        "physics_backend": str(info.get("physics_backend", "unknown")) if records else "unknown",
        "episode_records": records,
    }


def render_policy(agent: Agent, benchmark: Any, out_dir: Path) -> dict[str, Any]:
    frame_dir = out_dir / "trajectories" / "sample_000001"
    env = make_external_env(benchmark)
    obs, _ = env.reset(seed=int(benchmark.eval_seed_start))
    frames = []
    for idx in range(min(24, int(benchmark.max_steps))):
        frame = env.render()
        if frame is None:
            break
        frame_path = frame_dir / f"frame_{idx:04d}.jpg"
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(frame_path, frame)
        frames.append(str(frame_path))
        obs, _, terminated, truncated, _ = env.step(agent.act(obs, deterministic=True))
        if terminated or truncated:
            break
    env.close()
    if not frames:
        return {
            "media_available": False,
            "visual": {
                "sampled_status": "unavailable",
                "disabled_reason": "real_unitree_renderer_unavailable",
            },
        }
    gif_path = frame_dir / "rollout.gif"
    imageio.mimsave(gif_path, [imageio.imread(frame) for frame in frames], duration=0.08)
    manifest = {"status": "completed", "sample_index": 1, "frames": frames, "gif_path": str(gif_path), "frame_count": len(frames)}
    (frame_dir / "manifest.json").write_text(__import__("json").dumps(manifest, indent=2), encoding="utf-8")
    live_frame = out_dir / "current_run_frame.jpg"
    if frames:
        shutil.copy2(frames[-1], live_frame)
        frames[-1] = str(live_frame)
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
    agent = Agent(int(payload.get("obs_dim", 21)), int(payload.get("action_dim", 6)))
    agent.load_state_dict(payload["state_dict"])
    agent.eval()
    return agent
