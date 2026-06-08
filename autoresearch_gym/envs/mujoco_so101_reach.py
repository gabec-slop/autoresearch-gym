from __future__ import annotations

import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from autoresearch_gym.envs.vision import _resize_rgb


SUCCESS_THRESHOLD = 0.02
DEFAULT_FRAME_SKIP = 10
TARGET_RADIUS = 0.02
TARGET_CENTER = np.asarray([0.32, 0.0, 0.18], dtype=np.float32)
TARGET_WORKSPACE_HALF_EXTENT = 0.15
TARGET_RANGE = np.full(3, TARGET_WORKSPACE_HALF_EXTENT, dtype=np.float32)
TARGET_LOW = TARGET_CENTER - TARGET_RANGE
TARGET_HIGH = TARGET_CENTER + TARGET_RANGE
RENDER_CAMERA_LOOKAT = np.asarray([0.20, 0.0, 0.18], dtype=np.float64)

JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
SO101_HOME = np.asarray([0.0, -1.35, 1.69, 0.20, 0.0, -0.16], dtype=np.float64)

SO101_FEETECH_ACTUATOR_GUESS_PROFILE: dict[str, Any] = {
    "name": "feetech_sts3215_delta_guess_v0",
    "calibration_source": "engineering_guess",
    "motor_family": "Feetech STS3215-class serial servos",
    "control_mode": "delta_position_target",
    "actuator_lag_alpha": 0.35,
    "joint_deadband_rad": 0.004,
    "backlash_half_width_rad": 0.006,
    "max_arm_delta_rad_per_policy_step": 0.03,
    "max_gripper_delta_rad_per_policy_step": 0.10,
    "notes": [
        "Approximates serial-servo command latency, backlash, and finite target slew for SO-101 follower training.",
        "Values are intentionally conservative guesses until measured from this specific arm.",
    ],
}

SO101_PRINTED_GRIPPER_CONTACT_GUESS_PROFILE: dict[str, Any] = {
    "name": "so101_printed_plastic_gripper_guess_v0",
    "calibration_source": "engineering_guess",
    "material_assumption": "PLA+ low-infill printed fingers with optional TPU/tape contact surface",
    "pla_plus_density_g_cm3": [1.20, 1.27],
    "effective_low_infill_density_g_cm3": [0.25, 0.45],
    "pla_plastic_friction": {
        "sliding": [0.20, 0.35],
        "static": [0.25, 0.40],
        "torsional": [0.005, 0.02],
        "rolling": [0.0001, 0.001],
    },
    "compliant_surface_friction": {
        "sliding": [0.45, 0.90],
        "static": [0.60, 1.20],
    },
    "contact_defaults": {
        "condim": 4,
        "solref": "0.01 1",
        "solimp": "0.90 0.99 0.001",
    },
    "notes": [
        "Used as metadata for the SO-101 MJWarp support boundary.",
        "Reach training has no grasp contact, but manipulation variants should treat these values as sim-to-real gap parameters.",
    ],
}

SO101_MJWARP_GUESSED_PHYSICS_PROFILE: dict[str, Any] = {
    "name": "so101_follower_feetech_plastic_guess_v0",
    "calibration_source": "engineering_guess",
    "robot_asset_source": "mujoco_menagerie_robotstudio_so101",
    "actuator_profile": SO101_FEETECH_ACTUATOR_GUESS_PROFILE,
    "printed_gripper_profile": SO101_PRINTED_GRIPPER_CONTACT_GUESS_PROFILE,
}


class AutoresearchMujocoSO101ReachEnv(gym.Env[dict[str, np.ndarray], np.ndarray]):
    """SO-101 MuJoCo reach task.

    Uses the MuJoCo Menagerie `robotstudio_so101` model.
    """

    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        render_mode: str | None = "rgb_array",
        max_steps: int = 150,
        frame_skip: int = DEFAULT_FRAME_SKIP,
        reward_type: str = "dense",
        render_width: int = 720,
        render_height: int = 480,
        vision_observation: bool = False,
        vision_image_size: int = 84,
        vision_frame_stack: int = 3,
        model_path: str | None = None,
        **_: Any,
    ) -> None:
        try:
            import mujoco  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - optional extra.
            raise ModuleNotFoundError("AutoresearchMujocoSO101ReachEnv requires the `mujoco` extra.") from exc

        self.mujoco = mujoco
        self.render_mode = render_mode
        self.max_steps = int(max_steps)
        self.frame_skip = int(frame_skip)
        self.reward_type = reward_type
        self.render_width = int(render_width)
        self.render_height = int(render_height)
        self.vision_observation = bool(vision_observation)
        self.vision_image_size = int(vision_image_size)
        self.vision_frame_stack = int(vision_frame_stack)
        self.vision_frames: list[np.ndarray] = []
        self.rng = np.random.default_rng(0)
        self.step_count = 0
        self.renderer = None
        self.render_camera = None
        self.feed_renderers: dict[tuple[str, int, int], Any] = {}
        self.viewer = None

        self.so101_xml_path = resolve_so101_xml_path(model_path)
        if self.so101_xml_path is None:
            raise FileNotFoundError(
                "Could not find MuJoCo Menagerie robotstudio_so101/so101.xml. "
                "Set AUTORESEARCH_SO101_MJCF, set MUJOCO_MENAGERIE_PATH, or clone "
                "Menagerie into .external/mujoco_menagerie. SO-101 benchmarks require "
                "the real RobotStudio/Menagerie model and do not provide a substitute model."
            )
        self.scene_xml_path = write_so101_reach_scene_xml(self.so101_xml_path)
        self.model_source = "mujoco_menagerie_robotstudio_so101"
        self.model = self.mujoco.MjModel.from_xml_path(str(self.scene_xml_path))
        self.data = self.mujoco.MjData(self.model)
        self.ctrl_low, self.ctrl_high = _actuator_ctrl_ranges(self.model)
        self.joint_qpos_adrs = _joint_qpos_adrs(self.mujoco, self.model, JOINT_NAMES)
        self.joint_qvel_adrs = _joint_qvel_adrs(self.mujoco, self.model, JOINT_NAMES)
        self.ee_site_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        self.target_site_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_SITE, "target")
        if self.ee_site_id < 0 or self.target_site_id < 0:
            raise RuntimeError("generated SO-101 reach scene is missing gripperframe or target site")

        self.home_qpos = np.asarray(self.model.qpos0, dtype=np.float64).copy()
        self.home_qpos[self.joint_qpos_adrs] = SO101_HOME
        self.target = TARGET_CENTER.copy()
        self.initial_distance = 0.0
        self.last_action = np.zeros(self.model.nu, dtype=np.float32)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32)
        obs_dim = 3 + 3 + 3 + self.joint_qpos_adrs.size + self.joint_qvel_adrs.size + self.model.nu
        if self.vision_observation:
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Box(
                        low=0,
                        high=255,
                        shape=(3 * self.vision_frame_stack, self.vision_image_size, self.vision_image_size),
                        dtype=np.uint8,
                    ),
                    "proprio": spaces.Box(low=-np.inf, high=np.inf, shape=(self.model.nu * 3,), dtype=np.float32),
                    "achieved_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                    "desired_goal": spaces.Box(
                        low=TARGET_LOW,
                        high=TARGET_HIGH,
                        dtype=np.float32,
                    ),
                }
            )
        else:
            self.observation_space = spaces.Dict(
                {
                    "observation": spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32),
                    "achieved_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                    "desired_goal": spaces.Box(
                        low=TARGET_LOW,
                        high=TARGET_HIGH,
                        dtype=np.float32,
                    ),
                }
            )

    def _sample_target(self) -> np.ndarray:
        return self.rng.uniform(TARGET_LOW, TARGET_HIGH).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        fixed_case = (options or {}).get("fixed_case")
        fixed_case = fixed_case if isinstance(fixed_case, dict) else {}
        self.target = np.asarray(fixed_case.get("target_pos", self._sample_target()), dtype=np.float32).reshape(3)
        self.step_count = 0
        self.data.qpos[:] = self.home_qpos
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = self.home_qpos[self.joint_qpos_adrs]
        self._set_target_site(self.target)
        self.mujoco.mj_forward(self.model, self.data)
        self.last_action[:] = 0.0
        self.vision_frames = []
        obs = self._get_obs()
        info = self._info(obs)
        self.initial_distance = float(info["ee_to_target_distance"])
        info["initial_ee_to_target_distance"] = self.initial_distance
        info["ee_to_target_progress"] = 0.0
        return obs, info

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self.data.ctrl[:] = _denormalize_action(action, self.ctrl_low, self.ctrl_high)
        for _ in range(self.frame_skip):
            self.mujoco.mj_step(self.model, self.data)
        self.step_count += 1
        self.last_action = action.astype(np.float32, copy=True)
        obs = self._get_obs()
        info = self._info(obs)
        reward = float(self.compute_reward(obs["achieved_goal"], obs["desired_goal"], info))
        terminated = bool(info["is_success"])
        truncated = self.step_count >= self.max_steps
        return obs, reward, terminated, truncated, info

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict[str, Any] | None = None):
        distance = np.linalg.norm(np.asarray(achieved_goal, dtype=np.float32) - np.asarray(desired_goal, dtype=np.float32), axis=-1)
        success = distance < SUCCESS_THRESHOLD
        if self.reward_type == "sparse":
            return -(~success).astype(np.float32)
        progress = 0.0 if info is None else float(info.get("ee_to_target_progress", 0.0))
        return np.where(success, 1.0, -distance + 0.25 * progress).astype(np.float32)

    def _get_obs(self) -> dict[str, np.ndarray]:
        ee = np.asarray(self.data.site_xpos[self.ee_site_id], dtype=np.float32)
        joint_pos = np.asarray(self.data.qpos[self.joint_qpos_adrs], dtype=np.float32)
        joint_vel = np.asarray(self.data.qvel[self.joint_qvel_adrs], dtype=np.float32)
        observation = np.concatenate(
            [
                ee,
                self.target,
                self.target - ee,
                joint_pos,
                joint_vel,
                self.last_action,
            ]
        ).astype(np.float32)
        obs = {
            "observation": observation,
            "achieved_goal": ee.astype(np.float32),
            "desired_goal": self.target.astype(np.float32),
        }
        if self.vision_observation:
            return self._vision_obs(obs)
        return obs

    def _vision_obs(self, state_obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        wrist = _resize_rgb(self.sample_feeds()["wrist"], (self.vision_image_size, self.vision_image_size))
        if not self.vision_frames:
            self.vision_frames = [wrist.copy() for _ in range(self.vision_frame_stack)]
        else:
            self.vision_frames.append(wrist.copy())
            self.vision_frames = self.vision_frames[-self.vision_frame_stack :]
        pixels = np.concatenate([np.transpose(frame, (2, 0, 1)) for frame in self.vision_frames], axis=0)
        return {
            "pixels": pixels.astype(np.uint8, copy=False),
            "proprio": self.proprio_observation(),
            "achieved_goal": state_obs["achieved_goal"],
            "desired_goal": state_obs["desired_goal"],
        }

    def _info(self, obs: dict[str, np.ndarray]) -> dict[str, Any]:
        distance = float(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"]))
        return {
            "is_success": bool(distance < SUCCESS_THRESHOLD),
            "ee_to_target_distance": distance,
            "initial_ee_to_target_distance": self.initial_distance,
            "ee_to_target_progress": self.initial_distance - distance,
            "success_threshold": SUCCESS_THRESHOLD,
            "physics_backend": self.model_source,
        }

    def _set_target_site(self, target: np.ndarray) -> None:
        self.model.site_pos[self.target_site_id] = np.asarray(target, dtype=np.float64)

    def render(self):
        if self.render_mode == "human":  # pragma: no cover - interactive viewer.
            if self.viewer is None:
                viewer_mod = __import__("mujoco.viewer", fromlist=["launch_passive"])
                self.viewer = viewer_mod.launch_passive(self.model, self.data)
            self.viewer.sync()
            return None
        if self.renderer is None:
            self.renderer = self.mujoco.Renderer(self.model, height=self.render_height, width=self.render_width)
        if self.render_camera is None:
            self.render_camera = self.make_render_camera()
        self.renderer.update_scene(self.data, camera=self.render_camera)
        return np.asarray(self.renderer.render(), dtype=np.uint8)

    def sample_feed_specs(self) -> dict[str, dict[str, Any]]:
        return {
            "world": {
                "type": "rgb",
                "role": "debug",
                "source": "mujoco_free_camera",
                "shape": [self.render_height, self.render_width, 3],
            },
            "wrist": {
                "type": "rgb",
                "role": "policy_input",
                "source": "so101_nexus_wrist_cam_mjcf",
                "shape": [128, 128, 3],
            },
        }

    def sample_feeds(self) -> dict[str, np.ndarray]:
        return {
            "world": self._render_feed("world", self.render_width, self.render_height),
            "wrist": self._render_feed("wrist", 128, 128),
        }

    def proprio_observation(self) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray(self.data.qpos[self.joint_qpos_adrs], dtype=np.float32),
                np.asarray(self.data.qvel[self.joint_qvel_adrs], dtype=np.float32),
                self.last_action.astype(np.float32, copy=False),
            ]
        ).astype(np.float32, copy=False)

    def _render_feed(self, name: str, width: int, height: int) -> np.ndarray:
        key = (name, int(width), int(height))
        renderer = self.feed_renderers.get(key)
        if renderer is None:
            renderer = self.mujoco.Renderer(self.model, height=int(height), width=int(width))
            self.feed_renderers[key] = renderer
        self.mujoco.mj_forward(self.model, self.data)
        if name == "wrist":
            renderer.update_scene(self.data, camera="wrist")
        else:
            renderer.update_scene(self.data, camera=self.make_render_camera())
        return np.asarray(renderer.render(), dtype=np.uint8)

    def make_render_camera(self) -> Any:
        camera = self.mujoco.MjvCamera()
        camera.type = self.mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = RENDER_CAMERA_LOOKAT
        camera.distance = 0.72
        camera.azimuth = 55.0
        camera.elevation = -24.0
        return camera

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        for renderer in self.feed_renderers.values():
            renderer.close()
        self.feed_renderers = {}
        self.render_camera = None
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


class AutoresearchMujocoSO101ReachWarpVectorEnv:
    """Batched SO-101 reach collector backed by MuJoCo Warp.

    The source model is still the real RobotStudio/Menagerie SO-101 MJCF. The
    additional servo and printed-gripper physics are explicit engineering
    guesses so downstream runs can track this sim-to-real gap.
    """

    def __init__(
        self,
        num_envs: int,
        seed: int,
        model_path: str | None = None,
        max_steps: int = 150,
        frame_skip: int = DEFAULT_FRAME_SKIP,
        reward_type: str = "dense",
        actuator_lag_alpha: float | None = None,
        max_arm_delta: float | None = None,
        max_gripper_delta: float | None = None,
        joint_deadband: float | None = None,
        backlash_half_width: float | None = None,
        warp_nconmax: int = 64,
        warp_njmax: int = 256,
        **_: Any,
    ) -> None:
        try:
            import mujoco  # type: ignore
            import mujoco_warp  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - optional extra.
            raise ModuleNotFoundError("SO-101 MuJoCo Warp vector training requires `mujoco` and `mujoco_warp`.") from exc

        self.mujoco = mujoco
        self.mujoco_warp = mujoco_warp
        self.num_envs = int(num_envs)
        self.max_steps = int(max_steps)
        self.frame_skip = int(frame_skip)
        self.reward_type = reward_type
        self.warp_nconmax = int(warp_nconmax)
        self.warp_njmax = int(warp_njmax)
        actuator_profile = SO101_FEETECH_ACTUATOR_GUESS_PROFILE
        self.actuator_lag_alpha = float(
            actuator_profile["actuator_lag_alpha"] if actuator_lag_alpha is None else actuator_lag_alpha
        )
        self.max_arm_delta = float(
            actuator_profile["max_arm_delta_rad_per_policy_step"] if max_arm_delta is None else max_arm_delta
        )
        self.max_gripper_delta = float(
            actuator_profile["max_gripper_delta_rad_per_policy_step"] if max_gripper_delta is None else max_gripper_delta
        )
        self.joint_deadband = float(actuator_profile["joint_deadband_rad"] if joint_deadband is None else joint_deadband)
        self.backlash_half_width = float(
            actuator_profile["backlash_half_width_rad"] if backlash_half_width is None else backlash_half_width
        )
        self.rng = np.random.default_rng(seed)

        self.so101_xml_path = resolve_so101_xml_path(model_path)
        if self.so101_xml_path is None:
            raise FileNotFoundError(
                "Could not find MuJoCo Menagerie robotstudio_so101/so101.xml. "
                "SO-101 MuJoCo Warp support requires the real RobotStudio/Menagerie model."
            )
        self.scene_xml_path = write_so101_reach_scene_xml(self.so101_xml_path)
        self.model_source = "mujoco_menagerie_robotstudio_so101"
        self.model = self.mujoco.MjModel.from_xml_path(str(self.scene_xml_path))
        _prepare_model_for_mujoco_warp(self.mujoco, self.model)
        self.data0 = self.mujoco.MjData(self.model)
        self.ctrl_low, self.ctrl_high = _actuator_ctrl_ranges(self.model)
        self.joint_qpos_adrs = _joint_qpos_adrs(self.mujoco, self.model, JOINT_NAMES)
        self.joint_qvel_adrs = _joint_qvel_adrs(self.mujoco, self.model, JOINT_NAMES)
        self.ee_site_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        self.ee_body_id = _find_body_id(
            self.mujoco,
            self.model,
            ("gripper", "camera_mount", "moving_jaw_so101_v1", "wrist", "lower_arm"),
        )
        if self.ee_site_id < 0 and self.ee_body_id < 0:
            raise RuntimeError("SO-101 scene is missing gripperframe site and gripper body fallback")

        self.home_qpos = np.asarray(self.model.qpos0, dtype=np.float32).copy()
        self.home_qpos[self.joint_qpos_adrs] = SO101_HOME.astype(np.float32)
        self.home_ctrl = np.clip(self.home_qpos[self.joint_qpos_adrs], self.ctrl_low, self.ctrl_high).astype(np.float32)
        self.delta_limits = np.full(self.model.nu, self.max_arm_delta, dtype=np.float32)
        if self.delta_limits.size:
            self.delta_limits[-1] = self.max_gripper_delta
        self.last_actions = np.zeros((self.num_envs, self.model.nu), dtype=np.float32)
        self.ctrl_targets = np.repeat(self.home_ctrl[None, :], self.num_envs, axis=0).astype(np.float32)
        self.applied_ctrl = self.ctrl_targets.copy()
        self.control_error_sign = np.zeros_like(self.applied_ctrl, dtype=np.float32)
        self.targets = np.zeros((self.num_envs, 3), dtype=np.float32)
        self.initial_distances = np.zeros(self.num_envs, dtype=np.float32)
        self.step_counts = np.zeros(self.num_envs, dtype=np.int32)

        self.warp_model = self.mujoco_warp.put_model(self.model)
        self.warp_data = self.mujoco_warp.put_data(
            self.model,
            self.data0,
            nworld=self.num_envs,
            nconmax=self.warp_nconmax,
            njmax=self.warp_njmax,
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32)
        obs_dim = 3 + 3 + 3 + self.joint_qpos_adrs.size + self.joint_qvel_adrs.size + self.model.nu
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim + 6,), dtype=np.float32)
        self.physics_backend = "mujoco_warp_so101_follower_guess"

    def physics_profile_metadata(self) -> dict[str, Any]:
        return {
            **SO101_MJWARP_GUESSED_PHYSICS_PROFILE,
            "physics_backend": self.physics_backend,
            "num_envs": self.num_envs,
            "frame_skip": self.frame_skip,
            "warp_nconmax": self.warp_nconmax,
            "warp_njmax": self.warp_njmax,
        }

    def _sample_targets(self, count: int) -> np.ndarray:
        return self.rng.uniform(TARGET_LOW, TARGET_HIGH, size=(int(count), 3)).astype(np.float32)

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mask = np.ones(self.num_envs, dtype=bool)
        obs = self._reset_mask(mask)
        self._set_initial_distance_metrics(obs, mask)
        return obs

    def reset_worlds(self, mask: np.ndarray) -> np.ndarray:
        mask = np.asarray(mask, dtype=bool)
        if not np.any(mask):
            return self._obs_from_arrays(self.warp_data.qpos.numpy(), self.warp_data.qvel.numpy())
        obs = self._reset_mask(mask)
        self._set_initial_distance_metrics(obs, mask)
        return obs

    def _reset_mask(self, mask: np.ndarray) -> np.ndarray:
        qpos = self.warp_data.qpos.numpy()
        qvel = self.warp_data.qvel.numpy()
        qpos[mask] = self.home_qpos[None, :]
        qvel[mask] = 0.0
        self.targets[mask] = self._sample_targets(int(np.sum(mask)))
        self.ctrl_targets[mask] = self.home_ctrl[None, :]
        self.applied_ctrl[mask] = self.home_ctrl[None, :]
        self.control_error_sign[mask] = 0.0
        self.last_actions[mask] = 0.0
        self.step_counts[mask] = 0
        self.warp_data.qpos.assign(qpos.astype(np.float32))
        self.warp_data.qvel.assign(qvel.astype(np.float32))
        self.warp_data.ctrl.assign(self.applied_ctrl.astype(np.float32))
        self.mujoco_warp.forward(self.warp_model, self.warp_data)
        return self._obs_from_arrays(self.warp_data.qpos.numpy(), self.warp_data.qvel.numpy())

    def step(self, actions: np.ndarray):
        actions = np.clip(np.asarray(actions, dtype=np.float32), -1.0, 1.0)
        self._advance_servo_targets(actions)
        for _ in range(self.frame_skip):
            self.warp_data.ctrl.assign(self.applied_ctrl.astype(np.float32))
            self.mujoco_warp.step(self.warp_model, self.warp_data)
        self.step_counts += 1
        self.last_actions = actions.astype(np.float32, copy=True)
        obs = self._obs_from_arrays(self.warp_data.qpos.numpy(), self.warp_data.qvel.numpy())
        ee = obs[:, 0:3]
        distance = np.linalg.norm(ee - self.targets, axis=1).astype(np.float32)
        success = distance < SUCCESS_THRESHOLD
        progress = self.initial_distances - distance
        if self.reward_type == "sparse":
            reward = np.where(success, 0.0, -1.0).astype(np.float32)
        else:
            reward = np.where(success, 1.0, -distance + 0.25 * progress).astype(np.float32)
        truncated = self.step_counts >= self.max_steps
        infos = {
            "is_success": success.copy(),
            "ee_to_target_distance": distance.copy(),
            "initial_ee_to_target_distance": self.initial_distances.copy(),
            "ee_to_target_progress": progress.astype(np.float32),
            "success_threshold": np.full(self.num_envs, SUCCESS_THRESHOLD, dtype=np.float32),
            "physics_backend": np.asarray([self.physics_backend] * self.num_envs, dtype=object),
            "actuator_profile": np.asarray([SO101_FEETECH_ACTUATOR_GUESS_PROFILE["name"]] * self.num_envs, dtype=object),
        }
        return obs, reward, success.astype(bool), truncated.astype(bool), infos

    def _advance_servo_targets(self, actions: np.ndarray) -> None:
        desired = self.ctrl_targets + actions * self.delta_limits[None, :]
        self.ctrl_targets = np.clip(desired, self.ctrl_low[None, :], self.ctrl_high[None, :]).astype(np.float32)
        error = self.ctrl_targets - self.applied_ctrl
        sign = np.sign(error).astype(np.float32)
        reversing = (self.control_error_sign != 0.0) & (sign != 0.0) & (sign != self.control_error_sign)
        stalled_by_backlash = reversing & (np.abs(error) < self.backlash_half_width)
        active = (np.abs(error) >= self.joint_deadband) & ~stalled_by_backlash
        self.applied_ctrl[active] += self.actuator_lag_alpha * error[active]
        self.applied_ctrl = np.clip(self.applied_ctrl, self.ctrl_low[None, :], self.ctrl_high[None, :]).astype(np.float32)
        self.control_error_sign[active] = sign[active]

    def _set_initial_distance_metrics(self, obs: np.ndarray, mask: np.ndarray) -> None:
        ee = obs[:, 0:3]
        self.initial_distances[mask] = np.linalg.norm(ee[mask] - self.targets[mask], axis=1).astype(np.float32)

    def _obs_from_arrays(self, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        ee = self._ee_positions()
        joint_pos = qpos[:, self.joint_qpos_adrs].astype(np.float32)
        joint_vel = qvel[:, self.joint_qvel_adrs].astype(np.float32)
        observation = np.concatenate(
            [
                ee,
                self.targets,
                self.targets - ee,
                joint_pos,
                joint_vel,
                self.last_actions,
            ],
            axis=1,
        ).astype(np.float32)
        return np.concatenate([observation, ee, self.targets], axis=1).astype(np.float32)

    def _ee_positions(self) -> np.ndarray:
        try:
            site_xpos = getattr(self.warp_data, "site_xpos")
            return site_xpos.numpy()[:, self.ee_site_id, :].astype(np.float32)
        except Exception:
            pass
        try:
            xpos = self.warp_data.xpos.numpy()
            body_id = self.ee_body_id if self.ee_body_id >= 0 else -1
            return xpos[:, body_id, :].astype(np.float32)
        except Exception:
            qpos = self.warp_data.qpos.numpy()
            return qpos[:, self.joint_qpos_adrs[:3]].astype(np.float32)

    def close(self) -> None:
        return None


def resolve_so101_xml_path(model_path: str | None = None) -> Path | None:
    candidates: list[Path] = []
    if model_path:
        candidates.append(Path(model_path).expanduser())
    env_path = os.environ.get("AUTORESEARCH_SO101_MJCF")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    menagerie_root = os.environ.get("MUJOCO_MENAGERIE_PATH")
    if menagerie_root:
        candidates.append(Path(menagerie_root).expanduser() / "robotstudio_so101" / "so101.xml")
    candidates.append(Path.cwd() / ".external" / "mujoco_menagerie" / "robotstudio_so101" / "so101.xml")
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def write_so101_reach_scene_xml(so101_xml_path: Path) -> Path:
    scene_dir = Path(tempfile.gettempdir()) / "autoresearch_gym_so101"
    scene_dir.mkdir(parents=True, exist_ok=True)
    scene_path = scene_dir / f"so101_reach_menagerie_{abs(hash(str(so101_xml_path)))}.xml"
    tree = ET.parse(so101_xml_path)
    root = tree.getroot()
    root.set("model", "autoresearch_so101_reach")
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    meshdir = compiler.get("meshdir")
    if meshdir and not Path(meshdir).is_absolute():
        compiler.set("meshdir", str((so101_xml_path.parent / meshdir).resolve()))
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    global_visual = visual.find("global")
    if global_visual is None:
        global_visual = ET.SubElement(visual, "global")
    global_visual.set("offwidth", "1024")
    global_visual.set("offheight", "768")
    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(root, "worldbody")
    if worldbody.find("./site[@name='target']") is None:
        ET.SubElement(
            worldbody,
            "site",
            {
                "name": "target",
                "type": "sphere",
                "pos": f"{TARGET_CENTER[0]:.6f} {TARGET_CENTER[1]:.6f} {TARGET_CENTER[2]:.6f}",
                "size": f"{TARGET_RADIUS:.6f}",
                "rgba": "0.1 0.8 0.25 0.6",
            },
        )
    if worldbody.find("./geom[@name='floor']") is None:
        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": "floor",
                "type": "plane",
                "pos": "0 0 -0.005",
                "size": "0.8 0.8 0.02",
                "rgba": "0.55 0.58 0.62 1",
            },
        )
    if worldbody.find("./light[@name='key']") is None:
        ET.SubElement(
            worldbody,
            "light",
            {
                "name": "key",
                "pos": "0 -0.8 1.2",
                "dir": "0 0.7 -1",
                "diffuse": "0.9 0.9 0.9",
            },
        )
    if worldbody.find("./camera[@name='world']") is None:
        ET.SubElement(
            worldbody,
            "camera",
            {
                "name": "world",
                "pos": "0.55 -0.50 0.43",
                "xyaxes": "0.673 0.740 0 -0.329 0.299 0.896",
                "fovy": "46",
            },
        )
    if worldbody.find(".//camera[@name='wrist']") is None:
        gripper_site = worldbody.find(".//site[@name='gripperframe']")
        if gripper_site is not None:
            parent = _find_parent(worldbody, gripper_site)
            if parent is not None:
                ET.SubElement(
                    parent,
                    "camera",
                    {
                        "name": "wrist",
                        "pos": "0 0.04 -0.04",
                        "euler": "-0.5 0.0 6.28",
                        "fovy": "75",
                    },
                )
    tree.write(scene_path, encoding="utf-8", xml_declaration=False)
    return scene_path


def _actuator_ctrl_ranges(model: Any) -> tuple[np.ndarray, np.ndarray]:
    ranges = np.asarray(model.actuator_ctrlrange, dtype=np.float32)
    return ranges[:, 0], ranges[:, 1]


def _prepare_model_for_mujoco_warp(mujoco_module: Any, model: Any) -> None:
    if hasattr(mujoco_module, "mjtDisableBit") and hasattr(mujoco_module.mjtDisableBit, "mjDSBL_MULTICCD"):
        model.opt.disableflags |= int(mujoco_module.mjtDisableBit.mjDSBL_MULTICCD.value)
    if hasattr(model, "geom_margin"):
        model.geom_margin[:] = 0.0


def _find_body_id(mujoco: Any, model: Any, names: tuple[str, ...]) -> int:
    for name in names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id >= 0:
            return int(body_id)
    return -1


def _joint_qpos_adrs(mujoco: Any, model: Any, names: tuple[str, ...]) -> np.ndarray:
    adrs = []
    for name in names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise RuntimeError(f"SO-101 scene is missing joint {name}")
        adrs.append(int(model.jnt_qposadr[joint_id]))
    return np.asarray(adrs, dtype=np.int64)


def _joint_qvel_adrs(mujoco: Any, model: Any, names: tuple[str, ...]) -> np.ndarray:
    adrs = []
    for name in names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise RuntimeError(f"SO-101 scene is missing joint {name}")
        adrs.append(int(model.jnt_dofadr[joint_id]))
    return np.asarray(adrs, dtype=np.int64)


def _denormalize_action(action: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return low + (np.asarray(action, dtype=np.float32) + 1.0) * 0.5 * (high - low)


def _find_parent(root: ET.Element, child: ET.Element) -> ET.Element | None:
    for parent in root.iter():
        if child in list(parent):
            return parent
    return None
