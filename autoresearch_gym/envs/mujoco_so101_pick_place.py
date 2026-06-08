from __future__ import annotations

import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from autoresearch_gym.envs.vision import _resize_rgb
from autoresearch_gym.envs.mujoco_so101_reach import (
    DEFAULT_FRAME_SKIP,
    JOINT_NAMES,
    RENDER_CAMERA_LOOKAT,
    SO101_FEETECH_ACTUATOR_GUESS_PROFILE,
    SO101_HOME,
    SO101_MJWARP_GUESSED_PHYSICS_PROFILE,
    SO101_PRINTED_GRIPPER_CONTACT_GUESS_PROFILE,
    _actuator_ctrl_ranges,
    _denormalize_action,
    _find_body_id,
    _joint_qpos_adrs,
    _joint_qvel_adrs,
    _prepare_model_for_mujoco_warp,
    resolve_so101_xml_path,
)


TaskKind = Literal["cube_to_bin", "vial_to_rack"]

SO101_MANIP_MAX_STEPS = 512
CUBE_SUCCESS_THRESHOLD = 0.055
VIAL_SUCCESS_THRESHOLD = 0.035
VIAL_UPRIGHTNESS_THRESHOLD = 0.90
VIAL_HEIGHT_THRESHOLD = 0.026
NEXUS_REWARD_SOURCE = "SO101-Nexus v0.3.12 PickAndPlaceEnv/RewardConfig"
NEXUS_REWARD_SOURCE_URL = "https://pypi.org/project/so101-nexus-mujoco/"
NEXUS_REWARD_REACHING = 0.25
NEXUS_REWARD_GRASPING = 0.25
NEXUS_REWARD_TASK_OBJECTIVE = 0.40
NEXUS_REWARD_COMPLETION_BONUS = 0.10
NEXUS_REWARD_TANH_SCALE = 5.0
NEXUS_GRASP_FORCE_THRESHOLD = 0.5
NEXUS_STATIC_VEL_THRESHOLD = 0.2
MJWARP_GRASP_PROXY_DISTANCE = 0.035
CUBE_HALF_EXTENTS = np.asarray([0.0125, 0.0125, 0.0125], dtype=np.float32)
CUBE_MASS = 0.035 / 8.0
CUBE_OBJECT_CENTER = np.asarray([0.27, -0.06, float(CUBE_HALF_EXTENTS[2])], dtype=np.float32)
CUBE_TARGET_CENTER = np.asarray([0.36, 0.065, 0.03], dtype=np.float32)
VIAL_OBJECT_CENTER = np.asarray([0.28, -0.055, 0.045], dtype=np.float32)
VIAL_TARGET_CENTER = np.asarray([0.35, 0.055, 0.104], dtype=np.float32)
OBJECT_RANGE = np.asarray([0.035, 0.025, 0.0], dtype=np.float32)
TARGET_RANGE = np.asarray([0.025, 0.020, 0.0], dtype=np.float32)


def _nexus_reach_progress(distance: np.ndarray | float) -> np.ndarray:
    return 1.0 - np.tanh(NEXUS_REWARD_TANH_SCALE * np.maximum(0.0, np.asarray(distance, dtype=np.float32)))


def _nexus_pick_place_reward(
    *,
    tcp_to_obj_dist: np.ndarray | float,
    obj_to_target_dist: np.ndarray | float,
    is_grasped: np.ndarray | bool | float,
    is_complete: np.ndarray | bool | float,
) -> np.ndarray:
    """SO101-Nexus-style normalized pick-and-place reward.

    Mirrors the SO101-Nexus v0.3.12 pick-and-place reward budget:
    reach-to-object progress, binary grasp, grasp-gated placement progress, and
    completion bonus. See ``NEXUS_REWARD_SOURCE_URL`` for the upstream project.
    """

    grasp = np.asarray(is_grasped, dtype=np.float32)
    complete = np.asarray(is_complete, dtype=np.float32)
    reach_progress = _nexus_reach_progress(tcp_to_obj_dist)
    task_progress = _nexus_reach_progress(obj_to_target_dist) * grasp
    reward = (
        NEXUS_REWARD_REACHING * reach_progress
        + NEXUS_REWARD_GRASPING * grasp
        + NEXUS_REWARD_TASK_OBJECTIVE * task_progress
        + NEXUS_REWARD_COMPLETION_BONUS * complete
    )
    return reward.astype(np.float32, copy=False)


class _AutoresearchMujocoSO101PickPlaceEnv(gym.Env[dict[str, np.ndarray], np.ndarray]):
    metadata = {"render_modes": ["rgb_array", "human"]}

    task_kind: TaskKind = "cube_to_bin"
    object_name = "object"
    object_metric_name = "object_to_target"
    ee_metric_name = "ee_to_object"
    success_threshold = CUBE_SUCCESS_THRESHOLD
    object_center = CUBE_OBJECT_CENTER
    target_center = CUBE_TARGET_CENTER

    def __init__(
        self,
        render_mode: str | None = "rgb_array",
        max_steps: int = SO101_MANIP_MAX_STEPS,
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
            raise ModuleNotFoundError("SO-101 MuJoCo manipulation tasks require the `mujoco` extra.") from exc

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
        self.scene_xml_path = write_so101_pick_place_scene_xml(self.so101_xml_path, self.task_kind)
        self.model_source = "mujoco_menagerie_robotstudio_so101"

        self.model = self.mujoco.MjModel.from_xml_path(str(self.scene_xml_path))
        self.data = self.mujoco.MjData(self.model)
        self.ctrl_low, self.ctrl_high = _actuator_ctrl_ranges(self.model)
        self.joint_qpos_adrs = _joint_qpos_adrs(self.mujoco, self.model, JOINT_NAMES)
        self.joint_qvel_adrs = _joint_qvel_adrs(self.mujoco, self.model, JOINT_NAMES)
        self.ee_site_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        self.target_site_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_SITE, "target")
        self.object_site_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_SITE, "object_site")
        self.object_body_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, "object")
        object_joint_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, "object_freejoint")
        if min(self.ee_site_id, self.target_site_id, self.object_site_id, self.object_body_id, object_joint_id) < 0:
            raise RuntimeError("generated SO-101 manipulation scene is missing required sites or freejoint")
        self.object_qpos_adr = int(self.model.jnt_qposadr[object_joint_id])
        self.object_qvel_adr = int(self.model.jnt_dofadr[object_joint_id])
        self.object_geom_ids = self._collision_geoms_for_body(self.object_body_id)
        static_body_id = _find_body_id(self.mujoco, self.model, ("gripper", "fixed_jaw_so101_v1", "static_jaw"))
        moving_body_id = _find_body_id(self.mujoco, self.model, ("moving_jaw_so101_v1", "moving_jaw", "jaw"))
        self.static_finger_geom_ids = self._collision_geoms_for_body(static_body_id) | self._matching_geom_ids(
            ("static_finger_pad", "static")
        )
        self.moving_finger_geom_ids = self._collision_geoms_for_body(moving_body_id) | self._matching_geom_ids(
            ("moving_finger_pad", "moving_jaw", "moving")
        )

        self.home_qpos = np.asarray(self.model.qpos0, dtype=np.float64).copy()
        self.home_qpos[self.joint_qpos_adrs] = SO101_HOME
        self.object_pos = self.object_center.copy()
        self.target = self.target_center.copy()
        self.initial_distance = 0.0
        self.initial_ee_to_object = 0.0
        self.initial_object_z = float(self.object_center[2])
        self.last_action = np.zeros(self.model.nu, dtype=np.float32)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32)
        obs_dim = 3 + 3 + 3 + 3 + 3 + self.joint_qpos_adrs.size + self.joint_qvel_adrs.size + self.model.nu
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
                        low=self.target_center - TARGET_RANGE,
                        high=self.target_center + TARGET_RANGE,
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
                        low=self.target_center - TARGET_RANGE,
                        high=self.target_center + TARGET_RANGE,
                        dtype=np.float32,
                    ),
                }
            )

    def _sample_object_pos(self) -> np.ndarray:
        return (self.object_center + self.rng.uniform(-OBJECT_RANGE, OBJECT_RANGE)).astype(np.float32)

    def _sample_target_pos(self) -> np.ndarray:
        return (self.target_center + self.rng.uniform(-TARGET_RANGE, TARGET_RANGE)).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        fixed_case = (options or {}).get("fixed_case")
        fixed_case = fixed_case if isinstance(fixed_case, dict) else {}
        self.object_pos = np.asarray(fixed_case.get("object_pos", self._sample_object_pos()), dtype=np.float32).reshape(3)
        self.target = np.asarray(fixed_case.get("target_pos", self._sample_target_pos()), dtype=np.float32).reshape(3)
        self.step_count = 0
        self.data.qpos[:] = self.home_qpos
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = self.home_qpos[self.joint_qpos_adrs]
        self._set_object_pose(self.object_pos)
        self._set_target_site(self.target)
        self.mujoco.mj_forward(self.model, self.data)
        self.last_action[:] = 0.0
        self.vision_frames = []
        obs = self._get_obs()
        info = self._info(obs)
        self.initial_distance = float(info[f"{self.object_metric_name}_distance"])
        self.initial_ee_to_object = float(info[f"{self.ee_metric_name}_distance"])
        self.initial_object_z = float(obs["achieved_goal"][2])
        info[f"initial_{self.object_metric_name}_distance"] = self.initial_distance
        info[f"{self.object_metric_name}_progress"] = 0.0
        info[f"initial_{self.ee_metric_name}_distance"] = self.initial_ee_to_object
        info[f"{self.ee_metric_name}_progress"] = 0.0
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
        object_distance = np.linalg.norm(
            np.asarray(achieved_goal, dtype=np.float32) - np.asarray(desired_goal, dtype=np.float32),
            axis=-1,
        )
        success = False
        if info is not None and self.task_kind == "vial_to_rack":
            success = bool(info.get("is_success", False))
        else:
            success = object_distance < self.success_threshold
        success_array = np.asarray(success, dtype=bool)
        if self.reward_type == "sparse":
            return -(~success_array).astype(np.float32)
        ee_distance = 0.0 if info is None else info.get(f"{self.ee_metric_name}_distance", 0.0)
        is_grasped = False if info is None else info.get("is_grasped", False)
        return _nexus_pick_place_reward(
            tcp_to_obj_dist=ee_distance,
            obj_to_target_dist=object_distance,
            is_grasped=is_grasped,
            is_complete=success_array,
        )

    def _get_obs(self) -> dict[str, np.ndarray]:
        ee = np.asarray(self.data.site_xpos[self.ee_site_id], dtype=np.float32)
        object_pos = np.asarray(self.data.site_xpos[self.object_site_id], dtype=np.float32)
        joint_pos = np.asarray(self.data.qpos[self.joint_qpos_adrs], dtype=np.float32)
        joint_vel = np.asarray(self.data.qvel[self.joint_qvel_adrs], dtype=np.float32)
        observation = np.concatenate(
            [
                ee,
                object_pos,
                self.target,
                object_pos - ee,
                self.target - object_pos,
                joint_pos,
                joint_vel,
                self.last_action,
            ]
        ).astype(np.float32)
        obs = {
            "observation": observation,
            "achieved_goal": object_pos.astype(np.float32),
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
        object_distance = float(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"]))
        ee = np.asarray(self.data.site_xpos[self.ee_site_id], dtype=np.float32)
        ee_distance = float(np.linalg.norm(obs["achieved_goal"] - ee))
        is_success = bool(object_distance < self.success_threshold)
        is_grasped = self._is_grasped()
        is_robot_static = self._is_robot_static()
        lift_height = float(obs["achieved_goal"][2] - self.initial_object_z)
        info = {
            "is_success": is_success,
            "success": is_success,
            "is_grasped": is_grasped,
            "is_robot_static": is_robot_static,
            "lift_height": lift_height,
            f"{self.object_metric_name}_distance": object_distance,
            f"initial_{self.object_metric_name}_distance": self.initial_distance,
            f"{self.object_metric_name}_progress": self.initial_distance - object_distance,
            f"{self.ee_metric_name}_distance": ee_distance,
            f"initial_{self.ee_metric_name}_distance": self.initial_ee_to_object,
            f"{self.ee_metric_name}_progress": self.initial_ee_to_object - ee_distance,
            "object_to_target_distance": object_distance,
            "ee_to_object_distance": ee_distance,
            "tcp_to_obj_dist": ee_distance,
            "obj_to_target_dist": object_distance,
            "success_threshold": self.success_threshold,
            "reward_source": NEXUS_REWARD_SOURCE,
            "reward_source_url": NEXUS_REWARD_SOURCE_URL,
            "physics_backend": self.model_source,
            "task_kind": self.task_kind,
        }
        if self.task_kind == "vial_to_rack":
            uprightness = self._object_uprightness()
            height_error = abs(float(obs["achieved_goal"][2] - obs["desired_goal"][2]))
            is_success = bool(
                object_distance < self.success_threshold
                and uprightness >= VIAL_UPRIGHTNESS_THRESHOLD
                and height_error <= VIAL_HEIGHT_THRESHOLD
            )
            info.update(
                {
                    "is_success": is_success,
                    "success": is_success,
                    "vial_uprightness": uprightness,
                    "vial_height_error": height_error,
                    "vial_uprightness_threshold": VIAL_UPRIGHTNESS_THRESHOLD,
                    "vial_height_threshold": VIAL_HEIGHT_THRESHOLD,
                }
            )
        return info

    def _collision_geoms_for_body(self, body_id: int) -> set[int]:
        if body_id < 0:
            return set()
        return {
            geom_id
            for geom_id in range(int(self.model.ngeom))
            if int(self.model.geom_bodyid[geom_id]) == int(body_id) and int(self.model.geom_contype[geom_id]) != 0
        }

    def _matching_geom_ids(self, name_terms: tuple[str, ...]) -> set[int]:
        matches: set[int] = set()
        for geom_id in range(int(self.model.ngeom)):
            name = self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            if any(term in name for term in name_terms) and int(self.model.geom_contype[geom_id]) != 0:
                matches.add(geom_id)
        return matches

    def _is_grasped(self) -> float:
        if not self.object_geom_ids or not self.static_finger_geom_ids or not self.moving_finger_geom_ids:
            return 0.0
        static_contact = False
        moving_contact = False
        force_buf = np.zeros(6, dtype=np.float64)
        for contact_index in range(int(self.data.ncon)):
            contact = self.data.contact[contact_index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            object_involved = geom1 in self.object_geom_ids or geom2 in self.object_geom_ids
            if not object_involved:
                continue
            other = geom2 if geom1 in self.object_geom_ids else geom1
            self.mujoco.mj_contactForce(self.model, self.data, contact_index, force_buf)
            if abs(float(force_buf[0])) < NEXUS_GRASP_FORCE_THRESHOLD:
                continue
            static_contact = static_contact or other in self.static_finger_geom_ids
            moving_contact = moving_contact or other in self.moving_finger_geom_ids
            if static_contact and moving_contact:
                return 1.0
        return 0.0

    def _is_robot_static(self) -> bool:
        joint_vel = np.asarray(self.data.qvel[self.joint_qvel_adrs], dtype=np.float32)
        return bool(np.all(np.abs(joint_vel) < NEXUS_STATIC_VEL_THRESHOLD))

    def _object_uprightness(self) -> float:
        body_xmat = np.asarray(self.data.xmat[self.object_body_id], dtype=np.float64).reshape(3, 3)
        local_z_axis = body_xmat[:, 2]
        return float(np.clip(local_z_axis[2], -1.0, 1.0))

    def _set_object_pose(self, object_pos: np.ndarray) -> None:
        qpos = self.data.qpos[self.object_qpos_adr : self.object_qpos_adr + 7]
        qpos[:3] = np.asarray(object_pos, dtype=np.float64)
        qpos[3:] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.data.qvel[self.object_qvel_adr : self.object_qvel_adr + 6] = 0.0

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
        camera.distance = 0.76
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


class AutoresearchMujocoSO101CubeToBinEnv(_AutoresearchMujocoSO101PickPlaceEnv):
    """SO-101 cube-in-bin manipulation task."""

    task_kind = "cube_to_bin"
    object_name = "cube"
    object_metric_name = "cube_to_bin"
    ee_metric_name = "ee_to_cube"
    success_threshold = CUBE_SUCCESS_THRESHOLD
    object_center = CUBE_OBJECT_CENTER
    target_center = CUBE_TARGET_CENTER


class AutoresearchMujocoSO101VialToRackEnv(_AutoresearchMujocoSO101PickPlaceEnv):
    """SO-101 vial-to-rack manipulation task."""

    task_kind = "vial_to_rack"
    object_name = "vial"
    object_metric_name = "vial_to_slot"
    ee_metric_name = "ee_to_vial"
    success_threshold = VIAL_SUCCESS_THRESHOLD
    object_center = VIAL_OBJECT_CENTER
    target_center = VIAL_TARGET_CENTER


class AutoresearchMujocoSO101PickPlaceWarpVectorEnv:
    """Batched SO-101 manipulation collector backed by MuJoCo Warp.

    The robot asset is still the real RobotStudio/Menagerie SO-101 MJCF. Servo
    response, backlash, and printed-plastic gripper contact values are explicit
    engineering guesses for training throughput experiments and should be
    treated as sim-to-real gap parameters.
    """

    def __init__(
        self,
        *,
        task_kind: TaskKind,
        num_envs: int,
        seed: int,
        model_path: str | None = None,
        max_steps: int = SO101_MANIP_MAX_STEPS,
        frame_skip: int = DEFAULT_FRAME_SKIP,
        reward_type: str = "dense",
        actuator_lag_alpha: float | None = None,
        max_arm_delta: float | None = None,
        max_gripper_delta: float | None = None,
        joint_deadband: float | None = None,
        backlash_half_width: float | None = None,
        warp_nconmax: int = 128,
        warp_njmax: int = 512,
        **_: Any,
    ) -> None:
        try:
            import mujoco  # type: ignore
            import mujoco_warp  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - optional extra.
            raise ModuleNotFoundError("SO-101 MuJoCo Warp manipulation training requires `mujoco` and `mujoco_warp`.") from exc

        if task_kind not in ("cube_to_bin", "vial_to_rack"):
            raise ValueError(f"unknown SO-101 MJWarp manipulation task_kind: {task_kind}")
        self.mujoco = mujoco
        self.mujoco_warp = mujoco_warp
        self.task_kind: TaskKind = task_kind
        self.num_envs = int(num_envs)
        self.max_steps = int(max_steps)
        self.frame_skip = int(frame_skip)
        self.reward_type = reward_type
        self.warp_nconmax = int(warp_nconmax)
        self.warp_njmax = int(warp_njmax)
        if task_kind == "cube_to_bin":
            self.object_metric_name = "cube_to_bin"
            self.ee_metric_name = "ee_to_cube"
            self.success_threshold = CUBE_SUCCESS_THRESHOLD
            self.object_center = CUBE_OBJECT_CENTER
            self.target_center = CUBE_TARGET_CENTER
        else:
            self.object_metric_name = "vial_to_slot"
            self.ee_metric_name = "ee_to_vial"
            self.success_threshold = VIAL_SUCCESS_THRESHOLD
            self.object_center = VIAL_OBJECT_CENTER
            self.target_center = VIAL_TARGET_CENTER

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
        self.scene_xml_path = write_so101_pick_place_scene_xml(self.so101_xml_path, task_kind)
        self.model_source = "mujoco_menagerie_robotstudio_so101"
        self.model = self.mujoco.MjModel.from_xml_path(str(self.scene_xml_path))
        _apply_printed_gripper_contact_guess(self.mujoco, self.model)
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
        self.object_site_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_SITE, "object_site")
        self.object_body_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, "object")
        object_joint_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, "object_freejoint")
        if min(self.object_site_id, self.object_body_id, object_joint_id) < 0:
            raise RuntimeError("generated SO-101 MJWarp manipulation scene is missing the object freejoint")
        if self.ee_site_id < 0 and self.ee_body_id < 0:
            raise RuntimeError("SO-101 scene is missing gripperframe site and gripper body fallback")
        self.object_qpos_adr = int(self.model.jnt_qposadr[object_joint_id])
        self.object_qvel_adr = int(self.model.jnt_dofadr[object_joint_id])

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
        self.objects = np.zeros((self.num_envs, 3), dtype=np.float32)
        self.targets = np.zeros((self.num_envs, 3), dtype=np.float32)
        self.initial_object_distances = np.zeros(self.num_envs, dtype=np.float32)
        self.initial_ee_distances = np.zeros(self.num_envs, dtype=np.float32)
        self.initial_object_z = np.zeros(self.num_envs, dtype=np.float32)
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
        obs_dim = 3 + 3 + 3 + 3 + 3 + self.joint_qpos_adrs.size + self.joint_qvel_adrs.size + self.model.nu
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim + 6,), dtype=np.float32)
        self.physics_backend = f"mujoco_warp_so101_{task_kind}_feetech_plastic_guess"

    def physics_profile_metadata(self) -> dict[str, Any]:
        return {
            **SO101_MJWARP_GUESSED_PHYSICS_PROFILE,
            "physics_backend": self.physics_backend,
            "task_kind": self.task_kind,
            "num_envs": self.num_envs,
            "frame_skip": self.frame_skip,
            "warp_nconmax": self.warp_nconmax,
            "warp_njmax": self.warp_njmax,
            "printed_gripper_profile": SO101_PRINTED_GRIPPER_CONTACT_GUESS_PROFILE,
        }

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

    def _sample_object_pos(self, count: int) -> np.ndarray:
        return (self.object_center + self.rng.uniform(-OBJECT_RANGE, OBJECT_RANGE, size=(int(count), 3))).astype(np.float32)

    def _sample_target_pos(self, count: int) -> np.ndarray:
        return (self.target_center + self.rng.uniform(-TARGET_RANGE, TARGET_RANGE, size=(int(count), 3))).astype(np.float32)

    def _reset_mask(self, mask: np.ndarray) -> np.ndarray:
        qpos = self.warp_data.qpos.numpy()
        qvel = self.warp_data.qvel.numpy()
        qpos[mask] = self.home_qpos[None, :]
        qvel[mask] = 0.0
        self.objects[mask] = self._sample_object_pos(int(np.sum(mask)))
        self.targets[mask] = self._sample_target_pos(int(np.sum(mask)))
        qpos[mask, self.object_qpos_adr : self.object_qpos_adr + 3] = self.objects[mask]
        qpos[mask, self.object_qpos_adr + 3 : self.object_qpos_adr + 7] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        qvel[mask, self.object_qvel_adr : self.object_qvel_adr + 6] = 0.0
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
        qpos = self.warp_data.qpos.numpy()
        qvel = self.warp_data.qvel.numpy()
        obs = self._obs_from_arrays(qpos, qvel)
        object_pos = obs[:, 3:6]
        ee = obs[:, 0:3]
        object_distance = np.linalg.norm(object_pos - self.targets, axis=1).astype(np.float32)
        ee_distance = np.linalg.norm(object_pos - ee, axis=1).astype(np.float32)
        uprightness = self._object_uprightness(qpos)
        height_error = np.abs(object_pos[:, 2] - self.targets[:, 2]).astype(np.float32)
        success = object_distance < self.success_threshold
        if self.task_kind == "vial_to_rack":
            success = success & (uprightness >= VIAL_UPRIGHTNESS_THRESHOLD) & (height_error <= VIAL_HEIGHT_THRESHOLD)
        object_progress = self.initial_object_distances - object_distance
        ee_progress = self.initial_ee_distances - ee_distance
        is_grasped = ee_distance < MJWARP_GRASP_PROXY_DISTANCE
        lift_height = (object_pos[:, 2] - self.initial_object_z).astype(np.float32)
        if self.reward_type == "sparse":
            reward = np.where(success, 0.0, -1.0).astype(np.float32)
        else:
            reward = _nexus_pick_place_reward(
                tcp_to_obj_dist=ee_distance,
                obj_to_target_dist=object_distance,
                is_grasped=is_grasped,
                is_complete=success,
            )
        truncated = self.step_counts >= self.max_steps
        infos: dict[str, np.ndarray] = {
            "is_success": success.astype(bool),
            "success": success.astype(bool),
            "is_grasped": is_grasped.astype(bool),
            "is_grasped_proxy": is_grasped.astype(bool),
            "is_robot_static": np.zeros(self.num_envs, dtype=bool),
            "lift_height": lift_height.astype(np.float32),
            f"{self.object_metric_name}_distance": object_distance.copy(),
            f"initial_{self.object_metric_name}_distance": self.initial_object_distances.copy(),
            f"{self.object_metric_name}_progress": object_progress.astype(np.float32),
            f"{self.ee_metric_name}_distance": ee_distance.copy(),
            f"initial_{self.ee_metric_name}_distance": self.initial_ee_distances.copy(),
            f"{self.ee_metric_name}_progress": ee_progress.astype(np.float32),
            "object_to_target_distance": object_distance.copy(),
            "ee_to_object_distance": ee_distance.copy(),
            "tcp_to_obj_dist": ee_distance.copy(),
            "obj_to_target_dist": object_distance.copy(),
            "success_threshold": np.full(self.num_envs, self.success_threshold, dtype=np.float32),
            "reward_source": np.asarray([NEXUS_REWARD_SOURCE] * self.num_envs, dtype=object),
            "reward_source_url": np.asarray([NEXUS_REWARD_SOURCE_URL] * self.num_envs, dtype=object),
            "physics_backend": np.asarray([self.physics_backend] * self.num_envs, dtype=object),
            "actuator_profile": np.asarray([SO101_FEETECH_ACTUATOR_GUESS_PROFILE["name"]] * self.num_envs, dtype=object),
            "printed_gripper_profile": np.asarray(
                [SO101_PRINTED_GRIPPER_CONTACT_GUESS_PROFILE["name"]] * self.num_envs,
                dtype=object,
            ),
            "task_kind": np.asarray([self.task_kind] * self.num_envs, dtype=object),
        }
        if self.task_kind == "vial_to_rack":
            infos.update(
                {
                    "vial_uprightness": uprightness.astype(np.float32),
                    "vial_height_error": height_error.astype(np.float32),
                    "vial_uprightness_threshold": np.full(self.num_envs, VIAL_UPRIGHTNESS_THRESHOLD, dtype=np.float32),
                    "vial_height_threshold": np.full(self.num_envs, VIAL_HEIGHT_THRESHOLD, dtype=np.float32),
                }
            )
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
        object_pos = obs[:, 3:6]
        self.initial_object_distances[mask] = np.linalg.norm(object_pos[mask] - self.targets[mask], axis=1).astype(np.float32)
        self.initial_ee_distances[mask] = np.linalg.norm(object_pos[mask] - ee[mask], axis=1).astype(np.float32)
        self.initial_object_z[mask] = object_pos[mask, 2].astype(np.float32)

    def _obs_from_arrays(self, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        ee = self._ee_positions()
        object_pos = qpos[:, self.object_qpos_adr : self.object_qpos_adr + 3].astype(np.float32)
        joint_pos = qpos[:, self.joint_qpos_adrs].astype(np.float32)
        joint_vel = qvel[:, self.joint_qvel_adrs].astype(np.float32)
        observation = np.concatenate(
            [
                ee,
                object_pos,
                self.targets,
                object_pos - ee,
                self.targets - object_pos,
                joint_pos,
                joint_vel,
                self.last_actions,
            ],
            axis=1,
        ).astype(np.float32)
        return np.concatenate([observation, object_pos, self.targets], axis=1).astype(np.float32)

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

    def _object_uprightness(self, qpos: np.ndarray) -> np.ndarray:
        quat = qpos[:, self.object_qpos_adr + 3 : self.object_qpos_adr + 7].astype(np.float32)
        norms = np.linalg.norm(quat, axis=1, keepdims=True)
        quat = np.divide(quat, np.maximum(norms, 1e-6), out=np.zeros_like(quat), where=norms > 0)
        _w, x, y, _z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        return np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0).astype(np.float32)

    def close(self) -> None:
        return None


def write_so101_pick_place_scene_xml(so101_xml_path: Path, task_kind: TaskKind) -> Path:
    scene_dir = Path(tempfile.gettempdir()) / "autoresearch_gym_so101"
    scene_dir.mkdir(parents=True, exist_ok=True)
    scene_path = scene_dir / f"so101_{task_kind}_{abs(hash(str(so101_xml_path)))}.xml"
    tree = ET.parse(so101_xml_path)
    root = tree.getroot()
    root.set("model", f"autoresearch_so101_{task_kind}")
    _configure_compiler(root, so101_xml_path)
    _ensure_visual(root)
    asset = _ensure_child(root, "asset")
    _ensure_material(asset, "task_blue", "0.16 0.36 0.95 1")
    _ensure_material(asset, "task_green", "0.1 0.8 0.25 0.65")
    _ensure_material(asset, "task_orange", "1.0 0.45 0.08 1")
    _ensure_material(asset, "task_clear", "0.75 0.95 1.0 0.38")
    _ensure_material(asset, "task_rack", "0.12 0.13 0.16 1")
    _ensure_material(asset, "task_floor", "0.55 0.58 0.62 1")
    worldbody = _ensure_child(root, "worldbody")
    _ensure_common_world(worldbody)
    if task_kind == "cube_to_bin":
        _add_cube_to_bin_world(worldbody)
    else:
        _add_vial_to_rack_world(worldbody)
    tree.write(scene_path, encoding="utf-8", xml_declaration=False)
    return scene_path


def _configure_compiler(root: ET.Element, so101_xml_path: Path) -> None:
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    meshdir = compiler.get("meshdir")
    if meshdir and not Path(meshdir).is_absolute():
        compiler.set("meshdir", str((so101_xml_path.parent / meshdir).resolve()))


def _ensure_visual(root: ET.Element) -> None:
    visual = _ensure_child(root, "visual")
    global_visual = _ensure_child(visual, "global")
    global_visual.set("offwidth", "1024")
    global_visual.set("offheight", "768")


def _ensure_common_world(worldbody: ET.Element) -> None:
    if worldbody.find("./geom[@name='floor']") is None:
        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": "floor",
                "type": "plane",
                "pos": "0 0 -0.005",
                "size": "0.8 0.8 0.02",
                "material": "task_floor",
            },
        )
    if worldbody.find("./camera[@name='world']") is None:
        ET.SubElement(
            worldbody,
            "camera",
            {
                "name": "world",
                "pos": "0.58 -0.52 0.42",
                "xyaxes": "0.669 0.743 0 -0.338 0.305 0.891",
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


def _add_cube_to_bin_world(worldbody: ET.Element) -> None:
    _remove_existing_task_nodes(worldbody)
    ET.SubElement(
        worldbody,
        "site",
        {
            "name": "target",
            "type": "sphere",
            "pos": _vec(CUBE_TARGET_CENTER),
            "size": "0.020",
            "material": "task_green",
        },
    )
    _add_bin(worldbody, CUBE_TARGET_CENTER)
    cube_body = ET.SubElement(worldbody, "body", {"name": "object", "pos": _vec(CUBE_OBJECT_CENTER)})
    ET.SubElement(cube_body, "freejoint", {"name": "object_freejoint"})
    ET.SubElement(
        cube_body,
        "geom",
        {
            "name": "cube",
            "type": "box",
            "size": _vec(CUBE_HALF_EXTENTS),
            "mass": f"{CUBE_MASS:.8f}",
            "material": "task_blue",
            "friction": "1.2 0.01 0.0001",
        },
    )
    ET.SubElement(cube_body, "site", {"name": "object_site", "pos": "0 0 0", "size": "0.008", "rgba": "1 1 1 0"})


def _add_vial_to_rack_world(worldbody: ET.Element) -> None:
    _remove_existing_task_nodes(worldbody)
    ET.SubElement(
        worldbody,
        "site",
        {
            "name": "target",
            "type": "sphere",
            "pos": _vec(VIAL_TARGET_CENTER),
            "size": "0.016",
            "material": "task_green",
        },
    )
    _add_rack(worldbody, VIAL_TARGET_CENTER)
    vial_body = ET.SubElement(worldbody, "body", {"name": "object", "pos": _vec(VIAL_OBJECT_CENTER)})
    ET.SubElement(vial_body, "freejoint", {"name": "object_freejoint"})
    ET.SubElement(
        vial_body,
        "geom",
        {
            "name": "vial",
            "type": "cylinder",
            "size": "0.012 0.045",
            "mass": "0.025",
            "material": "task_clear",
            "friction": "1.0 0.01 0.0001",
        },
    )
    ET.SubElement(
        vial_body,
        "geom",
        {
            "name": "vial_cap",
            "type": "cylinder",
            "pos": "0 0 0.049",
            "size": "0.013 0.006",
            "mass": "0.003",
            "material": "task_orange",
            "contype": "0",
            "conaffinity": "0",
        },
    )
    ET.SubElement(vial_body, "site", {"name": "object_site", "pos": "0 0 0", "size": "0.008", "rgba": "1 1 1 0"})


def _add_bin(worldbody: ET.Element, center: np.ndarray) -> None:
    cx, cy, _cz = center.tolist()
    wall_height = 0.035
    wall_z = wall_height / 2.0
    wall_specs = [
        ("bin_back", cx + 0.045, cy, wall_z, "0.006 0.055 0.035"),
        ("bin_front", cx - 0.045, cy, wall_z, "0.006 0.055 0.035"),
        ("bin_left", cx, cy + 0.055, wall_z, "0.045 0.006 0.035"),
        ("bin_right", cx, cy - 0.055, wall_z, "0.045 0.006 0.035"),
    ]
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "bin_floor",
            "type": "box",
            "pos": f"{cx:.6f} {cy:.6f} 0.002000",
            "size": "0.052 0.062 0.004",
            "material": "task_rack",
            "contype": "0",
            "conaffinity": "0",
        },
    )
    for name, x, y, z, size in wall_specs:
        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": name,
                "type": "box",
                "pos": f"{x:.6f} {y:.6f} {z:.6f}",
                "size": size,
                "material": "task_rack",
            },
        )


def _add_rack(worldbody: ET.Element, center: np.ndarray) -> None:
    cx, cy, _cz = center.tolist()
    rail_z = 0.030
    rail_half_z = 0.024
    rack_geoms = [
        ("rack_back_rail", cx, cy + 0.042, rail_z, "0.080 0.006 0.024"),
        ("rack_front_rail", cx, cy - 0.042, rail_z, "0.080 0.006 0.024"),
        ("rack_left_rail", cx - 0.078, cy, rail_z, "0.006 0.042 0.024"),
        ("rack_right_rail", cx + 0.078, cy, rail_z, "0.006 0.042 0.024"),
        ("rack_center_slot_back", cx, cy + 0.020, rail_z, "0.028 0.004 0.024"),
        ("rack_center_slot_front", cx, cy - 0.020, rail_z, "0.028 0.004 0.024"),
        ("rack_center_slot_left", cx - 0.020, cy, rail_z, "0.004 0.028 0.024"),
        ("rack_center_slot_right", cx + 0.020, cy, rail_z, "0.004 0.028 0.024"),
    ]
    for name, x, y, z, size in rack_geoms:
        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": name,
                "type": "box",
                "pos": f"{x:.6f} {y:.6f} {z:.6f}",
                "size": size,
                "material": "task_rack",
                "friction": "1.1 0.01 0.0001",
            },
        )
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "rack_slot_floor",
            "type": "cylinder",
            "pos": f"{cx:.6f} {cy:.6f} 0.003000",
            "size": "0.014 0.003",
            "material": "task_green",
            "contype": "0",
            "conaffinity": "0",
        },
    )
    for index, xoff in enumerate((-0.035, 0.0, 0.035), start=1):
        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": f"rack_slot_{index}",
                "type": "cylinder",
                "pos": f"{cx + xoff:.6f} {cy:.6f} {rail_z + rail_half_z + 0.002:.6f}",
                "size": "0.016 0.003",
                "material": "task_green" if xoff == 0.0 else "task_floor",
                "contype": "0",
                "conaffinity": "0",
            },
        )
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "rack_backstop",
            "type": "box",
            "pos": f"{cx + 0.090:.6f} {cy:.6f} 0.038000",
            "size": "0.006 0.050 0.030",
            "material": "task_rack",
        },
    )


def _remove_existing_task_nodes(worldbody: ET.Element) -> None:
    names = {
        "target",
        "object",
        "bin_floor",
        "bin_back",
        "bin_front",
        "bin_left",
        "bin_right",
        "rack_plate",
        "rack_back_rail",
        "rack_front_rail",
        "rack_left_rail",
        "rack_right_rail",
        "rack_center_slot_back",
        "rack_center_slot_front",
        "rack_center_slot_left",
        "rack_center_slot_right",
        "rack_slot_floor",
        "rack_slot_1",
        "rack_slot_2",
        "rack_slot_3",
        "rack_backstop",
    }
    for child in list(worldbody):
        if child.get("name") in names:
            worldbody.remove(child)


def _apply_printed_gripper_contact_guess(mujoco: Any, model: Any) -> None:
    profile = SO101_PRINTED_GRIPPER_CONTACT_GUESS_PROFILE
    contact_defaults = profile["contact_defaults"]
    compliant = profile["compliant_surface_friction"]
    sliding = float(np.mean(compliant["sliding"]))
    torsional = float(np.mean(profile["pla_plastic_friction"]["torsional"]))
    rolling = float(np.mean(profile["pla_plastic_friction"]["rolling"]))
    condim = int(contact_defaults["condim"])
    gripper_terms = ("jaw", "gripper", "camera_box")
    for geom_id in range(int(model.ngeom)):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if not any(term in name for term in gripper_terms):
            continue
        if hasattr(model, "geom_friction"):
            model.geom_friction[geom_id] = np.asarray([sliding, torsional, rolling], dtype=np.float64)
        if hasattr(model, "geom_condim"):
            model.geom_condim[geom_id] = condim
        if hasattr(model, "geom_solref"):
            solref = np.fromstring(str(contact_defaults["solref"]), sep=" ")
            model.geom_solref[geom_id, : solref.size] = solref
        if hasattr(model, "geom_solimp"):
            solimp = np.fromstring(str(contact_defaults["solimp"]), sep=" ")
            model.geom_solimp[geom_id, : solimp.size] = solimp


def _ensure_child(parent: ET.Element, tag: str) -> ET.Element:
    child = parent.find(tag)
    if child is None:
        child = ET.SubElement(parent, tag)
    return child


def _ensure_material(asset: ET.Element, name: str, rgba: str) -> None:
    if asset.find(f"./material[@name='{name}']") is None:
        ET.SubElement(asset, "material", {"name": name, "rgba": rgba})


def _find_parent(root: ET.Element, child: ET.Element) -> ET.Element | None:
    for parent in root.iter():
        if child in list(parent):
            return parent
    return None


def _vec(values: np.ndarray) -> str:
    arr = np.asarray(values, dtype=np.float32).reshape(3)
    return f"{arr[0]:.6f} {arr[1]:.6f} {arr[2]:.6f}"
