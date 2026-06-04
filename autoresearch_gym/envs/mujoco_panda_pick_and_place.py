from __future__ import annotations

import math
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


TABLE_HEIGHT = 0.40
TABLE_TOP_Z = 0.0
TABLE_CENTER_X = -0.30
TABLE_LENGTH = 1.10
TABLE_WIDTH = 0.70
ROBOT_BASE_X = -0.60
CUBE_SIZE = 0.02
CUBE_Z = TABLE_TOP_Z + CUBE_SIZE
XY_RANGE = 0.15
GOAL_Z_RANGE = 0.0
TABLETOP_GOAL_PROBABILITY = 1.00
SUCCESS_THRESHOLD = 0.05
LIFT_THRESHOLD = 0.055
MAX_INITIAL_SAMPLE_ATTEMPTS = 64
DEFAULT_FRAME_SKIP = 20
PANDAGYM_GOAL_XY_RANGE = 0.30
PANDAGYM_GOAL_Z_RANGE = 0.20
PANDAGYM_OBJ_XY_RANGE = 0.30
PANDAGYM_TABLETOP_GOAL_PROBABILITY = 0.30
PANDA_HOME = np.asarray([-0.00595, 0.59970, 0.00694, -1.81345, -0.01718, 2.30801, 0.79292], dtype=np.float64)
RENDER_CAMERA_LOOKAT = np.asarray([-0.25, 0.0, 0.08], dtype=np.float64)
RENDER_CAMERA_DISTANCE = 1.85
RENDER_CAMERA_AZIMUTH = 45.0
RENDER_CAMERA_ELEVATION = -28.0


def _pick_place_success(cube_to_goal: np.ndarray | float, lifted_ever: np.ndarray | bool) -> np.ndarray | np.bool_:
    return (np.asarray(cube_to_goal) < SUCCESS_THRESHOLD) & np.asarray(lifted_ever, dtype=bool)


class AutoresearchMujocoPandaPickAndPlaceEnv(gym.Env[dict[str, np.ndarray], np.ndarray]):
    """Menagerie Franka Panda pick-and-place task with MuJoCo physics.

    This is the CPU/reference environment for the MJWarp task. It keeps a
    Gymnasium-Robotics-style observation contract while sourcing the robot from
    MuJoCo Menagerie via `robot_descriptions`, `MUJOCO_MENAGERIE_PATH`, or a
    caller-supplied `model_path`.
    """

    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        render_mode: str | None = "rgb_array",
        model_path: str | None = None,
        backend: str = "mujoco",
        max_steps: int = 50,
        frame_skip: int = DEFAULT_FRAME_SKIP,
        reward_type: str = "dense",
        success_requires_lift: bool = True,
        goal_xy_range: float | None = None,
        goal_z_range: float | None = None,
        obj_xy_range: float | None = None,
        tabletop_goal_probability: float | None = None,
        reject_initial_success: bool = True,
        render_width: int = 720,
        render_height: int = 480,
        warp_nconmax: int = 128,
        warp_njmax: int = 512,
        **_: Any,
    ) -> None:
        del backend
        try:
            import mujoco  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - optional extra.
            raise ModuleNotFoundError(
                "AutoresearchMujocoPandaPickAndPlaceEnv requires the `mujoco-warp` extra "
                "or an environment with `mujoco` and Menagerie Panda assets installed."
            ) from exc

        self.mujoco = mujoco
        self.render_mode = render_mode
        self.reward_type = reward_type
        self.success_requires_lift = bool(success_requires_lift)
        self.goal_xy_range = float(XY_RANGE * 2.0 if goal_xy_range is None else goal_xy_range)
        self.goal_z_range = float(GOAL_Z_RANGE if goal_z_range is None else goal_z_range)
        self.obj_xy_range = float(XY_RANGE * 2.0 if obj_xy_range is None else obj_xy_range)
        self.tabletop_goal_probability = float(TABLETOP_GOAL_PROBABILITY if tabletop_goal_probability is None else tabletop_goal_probability)
        self.reject_initial_success = bool(reject_initial_success)
        self.goal_range_low = np.asarray([-self.goal_xy_range / 2.0, -self.goal_xy_range / 2.0, 0.0], dtype=np.float32)
        self.goal_range_high = np.asarray([self.goal_xy_range / 2.0, self.goal_xy_range / 2.0, self.goal_z_range], dtype=np.float32)
        self.obj_range_low = np.asarray([-self.obj_xy_range / 2.0, -self.obj_xy_range / 2.0, 0.0], dtype=np.float32)
        self.obj_range_high = np.asarray([self.obj_xy_range / 2.0, self.obj_xy_range / 2.0, 0.0], dtype=np.float32)
        self.max_steps = int(max_steps)
        self.frame_skip = int(frame_skip)
        self.render_width = int(render_width)
        self.render_height = int(render_height)
        self.warp_nconmax = int(warp_nconmax)
        self.warp_njmax = int(warp_njmax)
        self.rng = np.random.default_rng(0)
        self.step_count = 0
        self.renderer = None
        self.render_camera = None
        self.viewer = None

        self.panda_xml_path = resolve_panda_xml_path(model_path)
        self.scene_xml_path = _write_scene_xml(self.panda_xml_path)
        self.model = self.mujoco.MjModel.from_xml_path(str(self.scene_xml_path))
        _prepare_model_for_mujoco_warp(self.mujoco, self.model)
        self.data = self.mujoco.MjData(self.model)

        self.ctrl_low, self.ctrl_high = _actuator_ctrl_ranges(self.model)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32)
        self.robot_qpos_adrs = _actuated_qpos_adrs(self.model)
        self.robot_qvel_adrs = _actuated_qvel_adrs(self.model)
        self.finger_qpos_adrs = _finger_qpos_adrs(self.mujoco, self.model)
        self.cube_joint_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, "cube_freejoint")
        if self.cube_joint_id < 0:
            raise RuntimeError("generated Panda pick-place scene is missing cube_freejoint")
        self.cube_qpos_adr = int(self.model.jnt_qposadr[self.cube_joint_id])
        self.cube_qvel_adr = int(self.model.jnt_dofadr[self.cube_joint_id])
        self.cube_body_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, "object")
        self.target_site_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_SITE, "target")
        self.ee_site_id = _find_site(self.mujoco, self.model, ["gripper", "pinch", "attachment_site", "end_effector"])
        self.ee_body_id = _find_body(self.mujoco, self.model, ["hand", "panda_hand", "link7", "panda_link7"])
        self.home_qpos = np.asarray(self.model.qpos0, dtype=np.float64).copy()
        if self.robot_qpos_adrs.size:
            count = min(7, self.robot_qpos_adrs.size)
            self.home_qpos[self.robot_qpos_adrs[:count]] = PANDA_HOME[:count]
        if self.finger_qpos_adrs.size:
            self.home_qpos[self.finger_qpos_adrs] = 0.04
        self.last_action = np.zeros(self.model.nu, dtype=np.float32)
        self.goal = np.zeros(3, dtype=np.float32)
        self.initial_ee_to_cube_distance = 0.0
        self.initial_cube_to_goal_distance = 0.0
        self.lifted_ever = False

        obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self._observation_dim(),), dtype=np.float32)
        goal_space = spaces.Box(
            low=np.asarray([self.goal_range_low[0], self.goal_range_low[1], CUBE_Z], dtype=np.float32),
            high=np.asarray([self.goal_range_high[0], self.goal_range_high[1], CUBE_Z + self.goal_z_range], dtype=np.float32),
        )
        self.observation_space = spaces.Dict(
            {
                "observation": obs_space,
                "achieved_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                "desired_goal": goal_space,
            }
        )

    def _observation_dim(self) -> int:
        return 3 + 3 + 3 + 3 + 3 + self.robot_qpos_adrs.size + self.robot_qvel_adrs.size + self.model.nu

    def _sample_cube_pos(self) -> np.ndarray:
        object_position = np.asarray([0.0, 0.0, CUBE_Z], dtype=np.float32)
        obj_range_low = getattr(self, "obj_range_low", np.asarray([-XY_RANGE, -XY_RANGE, 0.0], dtype=np.float32))
        obj_range_high = getattr(self, "obj_range_high", np.asarray([XY_RANGE, XY_RANGE, 0.0], dtype=np.float32))
        if bool(getattr(self, "success_requires_lift", True)) and obj_range_low[2] == obj_range_high[2] == 0.0:
            xy_noise = self.rng.uniform(obj_range_low[:2], obj_range_high[:2]).astype(np.float32)
            noise = np.asarray([xy_noise[0], xy_noise[1], 0.0], dtype=np.float32)
        else:
            noise = self.rng.uniform(obj_range_low, obj_range_high).astype(np.float32)
        return (object_position + noise).astype(np.float32)

    def _sample_goal_pos(self) -> np.ndarray:
        goal = np.asarray([0.0, 0.0, CUBE_Z], dtype=np.float32)
        goal_range_low = getattr(self, "goal_range_low", np.asarray([-XY_RANGE, -XY_RANGE, 0.0], dtype=np.float32))
        goal_range_high = getattr(self, "goal_range_high", np.asarray([XY_RANGE, XY_RANGE, GOAL_Z_RANGE], dtype=np.float32))
        tabletop_goal_probability = float(getattr(self, "tabletop_goal_probability", TABLETOP_GOAL_PROBABILITY))
        legacy_fixed_z = (
            bool(getattr(self, "success_requires_lift", True))
            and goal_range_low[2] == goal_range_high[2] == 0.0
            and tabletop_goal_probability >= 1.0
        )
        if legacy_fixed_z:
            xy_noise = self.rng.uniform(goal_range_low[:2], goal_range_high[:2]).astype(np.float32)
            noise = np.asarray([xy_noise[0], xy_noise[1], 0.0], dtype=np.float32)
        else:
            noise = self.rng.uniform(goal_range_low, goal_range_high).astype(np.float32)
        if tabletop_goal_probability >= 1.0 or (tabletop_goal_probability > 0.0 and self.rng.random() < tabletop_goal_probability):
            noise[2] = 0.0
        return (goal + noise).astype(np.float32)

    def _sample_goal_and_cube_pos(self) -> tuple[np.ndarray, np.ndarray, int]:
        goal = self._sample_goal_pos()
        cube_pos = self._sample_cube_pos()
        if not bool(getattr(self, "reject_initial_success", True)):
            return goal, cube_pos, 0
        for attempt in range(MAX_INITIAL_SAMPLE_ATTEMPTS):
            if _goal_distance(cube_pos, goal) >= SUCCESS_THRESHOLD:
                return goal, cube_pos, attempt
            goal = self._sample_goal_pos()
            cube_pos = self._sample_cube_pos()
        return goal, _move_cube_away_from_goal(cube_pos, goal), MAX_INITIAL_SAMPLE_ATTEMPTS

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        fixed_case = (options or {}).get("fixed_case")
        fixed_case = fixed_case if isinstance(fixed_case, dict) else {}
        if "goal_pos" in fixed_case or "cube_pos" in fixed_case:
            goal = np.asarray(fixed_case.get("goal_pos", self._sample_goal_pos()), dtype=np.float32)
            cube_pos = np.asarray(fixed_case.get("cube_pos", self._sample_cube_pos()), dtype=np.float32)
            initial_resample_attempts = 0
        else:
            goal, cube_pos, initial_resample_attempts = self._sample_goal_and_cube_pos()
        cube_pos = cube_pos.reshape(3)
        goal = goal.reshape(3)
        initial_goal_distance = _goal_distance(cube_pos, goal)
        if self.reject_initial_success and initial_goal_distance < SUCCESS_THRESHOLD:
            raise ValueError(
                f"fixed Panda pick-and-place case starts solved: cube-goal distance "
                f"{initial_goal_distance:.4f} is below success threshold {SUCCESS_THRESHOLD:.4f}"
            )
        self.step_count = 0
        self.lifted_ever = False
        self.data.qpos[:] = self.home_qpos
        self.data.qvel[:] = 0.0
        self.data.qpos[self.cube_qpos_adr : self.cube_qpos_adr + 3] = cube_pos
        self.data.qpos[self.cube_qpos_adr + 3 : self.cube_qpos_adr + 7] = np.asarray([1.0, 0.0, 0.0, 0.0])
        self.goal = goal.astype(np.float32, copy=True)
        self._set_target_site(self.goal)
        self.data.ctrl[:] = _denormalize_action(_open_gripper_action(self.model.nu), self.ctrl_low, self.ctrl_high)
        self.mujoco.mj_forward(self.model, self.data)
        self.last_action = _open_gripper_action(self.model.nu)
        obs = self._get_obs()
        info = self._info(obs)
        self.initial_ee_to_cube_distance = float(info["ee_to_cube_distance"])
        self.initial_cube_to_goal_distance = initial_goal_distance
        info["initial_goal_distance"] = initial_goal_distance
        info["initial_cube_to_goal_distance"] = initial_goal_distance
        info["initial_ee_to_cube_distance"] = self.initial_ee_to_cube_distance
        info["ee_to_cube_progress"] = 0.0
        info["cube_to_goal_progress"] = 0.0
        info["initial_resample_attempts"] = initial_resample_attempts
        return obs, info

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self.data.ctrl[:] = _denormalize_action(action, self.ctrl_low, self.ctrl_high)
        for _ in range(self.frame_skip):
            self.mujoco.mj_step(self.model, self.data)
        self.step_count += 1
        self.last_action = action
        obs = self._get_obs()
        info = self._info(obs)
        self.lifted_ever = bool(info["lifted_ever"])
        reward = float(self.compute_reward(obs["achieved_goal"], obs["desired_goal"], info))
        terminated = bool(info["is_success"])
        truncated = self.step_count >= self.max_steps
        return obs, reward, terminated, truncated, info

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict[str, Any] | None = None):
        achieved = np.asarray(achieved_goal, dtype=np.float32)
        desired = np.asarray(desired_goal, dtype=np.float32)
        distance = np.linalg.norm(achieved - desired, axis=-1)
        lifted_ever = np.asarray((info or {}).get("lifted_ever", False), dtype=bool)
        success_requires_lift = bool(getattr(self, "success_requires_lift", True))
        success = _pick_place_success(distance, lifted_ever) if success_requires_lift else distance < SUCCESS_THRESHOLD
        if getattr(self, "reward_type", "dense") == "sparse":
            return -(~success).astype(np.float32)
        if success_requires_lift and info is not None and "lifted_ever" in info:
            return np.where(success, 0.0, np.where(lifted_ever, -distance, -1.0)).astype(np.float32)
        return -distance.astype(np.float32)

    def _get_obs(self) -> dict[str, np.ndarray]:
        cube_pos = np.asarray(self.data.xpos[self.cube_body_id], dtype=np.float32)
        cube_vel = np.asarray(self.data.qvel[self.cube_qvel_adr : self.cube_qvel_adr + 3], dtype=np.float32)
        ee_pos = self._ee_pos()
        joint_pos = np.asarray(self.data.qpos[self.robot_qpos_adrs], dtype=np.float32)
        joint_vel = np.asarray(self.data.qvel[self.robot_qvel_adrs], dtype=np.float32)
        observation = np.concatenate(
            [
                ee_pos,
                cube_pos,
                self.goal,
                cube_pos - ee_pos,
                self.goal - cube_pos,
                joint_pos,
                joint_vel,
                self.last_action,
            ]
        ).astype(np.float32)
        return {
            "observation": observation,
            "achieved_goal": cube_pos.astype(np.float32),
            "desired_goal": self.goal.astype(np.float32),
        }

    def _ee_pos(self) -> np.ndarray:
        if self.ee_site_id >= 0:
            return np.asarray(self.data.site_xpos[self.ee_site_id], dtype=np.float32)
        if self.ee_body_id >= 0:
            return np.asarray(self.data.xpos[self.ee_body_id], dtype=np.float32)
        return np.asarray(self.data.xpos[-1], dtype=np.float32)

    def _set_target_site(self, goal: np.ndarray) -> None:
        if self.target_site_id >= 0:
            self.model.site_pos[self.target_site_id] = np.asarray(goal, dtype=np.float64)

    def _info(self, obs: dict[str, np.ndarray]) -> dict[str, Any]:
        cube = obs["achieved_goal"]
        goal = obs["desired_goal"]
        ee = obs["observation"][:3]
        ee_to_cube = float(np.linalg.norm(ee - cube))
        cube_to_goal = float(np.linalg.norm(cube - goal))
        cube_lift = float(max(0.0, cube[2] - CUBE_Z))
        near_cube = ee_to_cube < 0.055
        lifted = cube_lift > LIFT_THRESHOLD
        lifted_ever = bool(self.lifted_ever or lifted)
        cube_at_goal = cube_to_goal < SUCCESS_THRESHOLD
        placed = bool(_pick_place_success(cube_to_goal, lifted_ever)) if self.success_requires_lift else bool(cube_at_goal)
        gripper_closed = bool(_finger_width(self.data, self.finger_qpos_adrs) < 0.035)
        return {
            "is_success": bool(placed),
            "cube_at_goal": bool(cube_at_goal),
            "ee_to_cube_distance": ee_to_cube,
            "cube_to_goal_distance": cube_to_goal,
            "initial_ee_to_cube_distance": self.initial_ee_to_cube_distance,
            "initial_cube_to_goal_distance": self.initial_cube_to_goal_distance,
            "ee_to_cube_progress": self.initial_ee_to_cube_distance - ee_to_cube,
            "cube_to_goal_progress": self.initial_cube_to_goal_distance - cube_to_goal,
            "cube_lift_height": cube_lift,
            "near_cube": near_cube,
            "gripper_closed_near_cube": bool(near_cube and gripper_closed),
            "lifted": bool(lifted),
            "lifted_ever": bool(lifted_ever),
            "placed_success": bool(placed),
            "success_threshold": SUCCESS_THRESHOLD,
            "physics_backend": "mujoco_menagerie_panda",
        }

    def make_vectorized(self, num_envs: int, seed: int = 0):
        return MujocoWarpPandaPickAndPlaceVectorEnv(
            num_envs=num_envs,
            seed=seed,
            model_path=str(self.panda_xml_path),
            max_steps=self.max_steps,
            frame_skip=self.frame_skip,
            reward_type=self.reward_type,
            success_requires_lift=self.success_requires_lift,
            goal_xy_range=self.goal_xy_range,
            goal_z_range=self.goal_z_range,
            obj_xy_range=self.obj_xy_range,
            tabletop_goal_probability=self.tabletop_goal_probability,
            reject_initial_success=self.reject_initial_success,
            warp_nconmax=self.warp_nconmax,
            warp_njmax=self.warp_njmax,
        )

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

    def make_render_camera(self) -> Any:
        camera = self.mujoco.MjvCamera()
        camera.type = self.mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = RENDER_CAMERA_LOOKAT
        camera.distance = RENDER_CAMERA_DISTANCE
        camera.azimuth = RENDER_CAMERA_AZIMUTH
        camera.elevation = RENDER_CAMERA_ELEVATION
        return camera

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        self.render_camera = None
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


class MujocoWarpPandaPickAndPlaceVectorEnv:
    def __init__(
        self,
        num_envs: int,
        seed: int,
        model_path: str | None = None,
        max_steps: int = 50,
        frame_skip: int = DEFAULT_FRAME_SKIP,
        reward_type: str = "dense",
        success_requires_lift: bool = True,
        goal_xy_range: float | None = None,
        goal_z_range: float | None = None,
        obj_xy_range: float | None = None,
        tabletop_goal_probability: float | None = None,
        reject_initial_success: bool = True,
        warp_nconmax: int = 128,
        warp_njmax: int = 512,
    ) -> None:
        try:
            import mujoco  # type: ignore
            import mujoco_warp  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - optional extra.
            raise ModuleNotFoundError("MuJoCo Warp vector training requires `mujoco` and `mujoco_warp`.") from exc
        self.mujoco = mujoco
        self.mujoco_warp = mujoco_warp
        self.num_envs = int(num_envs)
        self.max_steps = int(max_steps)
        self.frame_skip = int(frame_skip)
        self.reward_type = reward_type
        self.success_requires_lift = bool(success_requires_lift)
        self.goal_xy_range = float(XY_RANGE * 2.0 if goal_xy_range is None else goal_xy_range)
        self.goal_z_range = float(GOAL_Z_RANGE if goal_z_range is None else goal_z_range)
        self.obj_xy_range = float(XY_RANGE * 2.0 if obj_xy_range is None else obj_xy_range)
        self.tabletop_goal_probability = float(TABLETOP_GOAL_PROBABILITY if tabletop_goal_probability is None else tabletop_goal_probability)
        self.reject_initial_success = bool(reject_initial_success)
        self.goal_range_low = np.asarray([-self.goal_xy_range / 2.0, -self.goal_xy_range / 2.0, 0.0], dtype=np.float32)
        self.goal_range_high = np.asarray([self.goal_xy_range / 2.0, self.goal_xy_range / 2.0, self.goal_z_range], dtype=np.float32)
        self.obj_range_low = np.asarray([-self.obj_xy_range / 2.0, -self.obj_xy_range / 2.0, 0.0], dtype=np.float32)
        self.obj_range_high = np.asarray([self.obj_xy_range / 2.0, self.obj_xy_range / 2.0, 0.0], dtype=np.float32)
        self.warp_nconmax = int(warp_nconmax)
        self.warp_njmax = int(warp_njmax)
        self.rng = np.random.default_rng(seed)
        panda_xml_path = resolve_panda_xml_path(model_path)
        self.scene_xml_path = _write_scene_xml(panda_xml_path)
        self.model = self.mujoco.MjModel.from_xml_path(str(self.scene_xml_path))
        _prepare_model_for_mujoco_warp(self.mujoco, self.model)
        self.data0 = self.mujoco.MjData(self.model)
        self.ctrl_low, self.ctrl_high = _actuator_ctrl_ranges(self.model)
        self.robot_qpos_adrs = _actuated_qpos_adrs(self.model)
        self.robot_qvel_adrs = _actuated_qvel_adrs(self.model)
        self.finger_qpos_adrs = _finger_qpos_adrs(self.mujoco, self.model)
        self.cube_joint_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, "cube_freejoint")
        self.cube_qpos_adr = int(self.model.jnt_qposadr[self.cube_joint_id])
        self.cube_qvel_adr = int(self.model.jnt_dofadr[self.cube_joint_id])
        self.cube_body_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, "object")
        self.ee_site_id = _find_site(self.mujoco, self.model, ["gripper", "pinch", "attachment_site", "end_effector"])
        self.ee_body_id = _find_body(self.mujoco, self.model, ["hand", "panda_hand", "link7", "panda_link7"])
        self.home_qpos = np.asarray(self.model.qpos0, dtype=np.float32).copy()
        if self.robot_qpos_adrs.size:
            count = min(7, self.robot_qpos_adrs.size)
            self.home_qpos[self.robot_qpos_adrs[:count]] = PANDA_HOME[:count].astype(np.float32)
        if self.finger_qpos_adrs.size:
            self.home_qpos[self.finger_qpos_adrs] = 0.04
        self.last_actions = np.zeros((self.num_envs, self.model.nu), dtype=np.float32)
        self.goals = np.zeros((self.num_envs, 3), dtype=np.float32)
        self.initial_ee_to_cube_distances = np.zeros(self.num_envs, dtype=np.float32)
        self.initial_cube_to_goal_distances = np.zeros(self.num_envs, dtype=np.float32)
        self.lifted_ever = np.zeros(self.num_envs, dtype=bool)
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
        obs_dim = 3 + 3 + 3 + 3 + 3 + self.robot_qpos_adrs.size + self.robot_qvel_adrs.size + self.model.nu
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim + 6,), dtype=np.float32)
        self.physics_backend = "mujoco_warp_menagerie_panda"

    def _sample_cube_positions(self, count: int) -> np.ndarray:
        base = np.repeat(np.asarray([[0.0, 0.0, CUBE_Z]], dtype=np.float32), count, axis=0)
        obj_range_low = getattr(self, "obj_range_low", np.asarray([-XY_RANGE, -XY_RANGE, 0.0], dtype=np.float32))
        obj_range_high = getattr(self, "obj_range_high", np.asarray([XY_RANGE, XY_RANGE, 0.0], dtype=np.float32))
        if bool(getattr(self, "success_requires_lift", True)) and obj_range_low[2] == obj_range_high[2] == 0.0:
            xy_noise = self.rng.uniform(obj_range_low[:2], obj_range_high[:2], size=(count, 2)).astype(np.float32)
            noise = np.zeros((count, 3), dtype=np.float32)
            noise[:, :2] = xy_noise
        else:
            noise = self.rng.uniform(obj_range_low, obj_range_high, size=(count, 3)).astype(np.float32)
        return (base + noise).astype(np.float32)

    def _sample_goals(self, count: int) -> np.ndarray:
        base = np.repeat(np.asarray([[0.0, 0.0, CUBE_Z]], dtype=np.float32), count, axis=0)
        goal_range_low = getattr(self, "goal_range_low", np.asarray([-XY_RANGE, -XY_RANGE, 0.0], dtype=np.float32))
        goal_range_high = getattr(self, "goal_range_high", np.asarray([XY_RANGE, XY_RANGE, GOAL_Z_RANGE], dtype=np.float32))
        tabletop_goal_probability = float(getattr(self, "tabletop_goal_probability", TABLETOP_GOAL_PROBABILITY))
        legacy_fixed_z = (
            bool(getattr(self, "success_requires_lift", True))
            and goal_range_low[2] == goal_range_high[2] == 0.0
            and tabletop_goal_probability >= 1.0
        )
        if legacy_fixed_z:
            xy_noise = self.rng.uniform(goal_range_low[:2], goal_range_high[:2], size=(count, 2)).astype(np.float32)
            noise = np.zeros((count, 3), dtype=np.float32)
            noise[:, :2] = xy_noise
        else:
            noise = self.rng.uniform(goal_range_low, goal_range_high, size=(count, 3)).astype(np.float32)
        if tabletop_goal_probability >= 1.0:
            tabletop = np.ones(count, dtype=bool)
        elif tabletop_goal_probability <= 0.0:
            tabletop = np.zeros(count, dtype=bool)
        else:
            tabletop = self.rng.random(count) < tabletop_goal_probability
        noise[tabletop, 2] = 0.0
        return (base + noise).astype(np.float32)

    def _sample_goal_and_cube_positions(self, count: int) -> tuple[np.ndarray, np.ndarray]:
        goals = self._sample_goals(count)
        cube = self._sample_cube_positions(count)
        if not bool(getattr(self, "reject_initial_success", True)):
            return goals, cube
        invalid = np.linalg.norm(cube - goals, axis=1) < SUCCESS_THRESHOLD
        for _ in range(MAX_INITIAL_SAMPLE_ATTEMPTS):
            if not np.any(invalid):
                return goals, cube
            replace_count = int(np.sum(invalid))
            goals[invalid] = self._sample_goals(replace_count)
            cube[invalid] = self._sample_cube_positions(replace_count)
            invalid = np.linalg.norm(cube - goals, axis=1) < SUCCESS_THRESHOLD
        if np.any(invalid):
            cube[invalid] = _move_cubes_away_from_goals(cube[invalid], goals[invalid])
        return goals, cube

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.step_counts[:] = 0
        self.lifted_ever[:] = False
        self.goals[:], cube = self._sample_goal_and_cube_positions(self.num_envs)
        qpos = np.repeat(self.home_qpos[None, :], self.num_envs, axis=0).astype(np.float32)
        qvel = np.zeros((self.num_envs, self.model.nv), dtype=np.float32)
        qpos[:, self.cube_qpos_adr : self.cube_qpos_adr + 3] = cube
        qpos[:, self.cube_qpos_adr + 3] = 1.0
        self.last_actions[:] = _open_gripper_action(self.model.nu)
        self.warp_data.qpos.assign(qpos)
        self.warp_data.qvel.assign(qvel)
        self.warp_data.ctrl.assign(np.repeat(_denormalize_action(_open_gripper_action(self.model.nu), self.ctrl_low, self.ctrl_high)[None, :], self.num_envs, axis=0).astype(np.float32))
        self.mujoco_warp.forward(self.warp_model, self.warp_data)
        obs = self._obs_from_arrays(self.warp_data.qpos.numpy(), self.warp_data.qvel.numpy())
        self._set_initial_distance_metrics(obs, np.ones(self.num_envs, dtype=bool))
        return obs

    def reset_worlds(self, mask: np.ndarray) -> np.ndarray:
        mask = np.asarray(mask, dtype=bool)
        if not np.any(mask):
            return self._obs_from_arrays(self.warp_data.qpos.numpy(), self.warp_data.qvel.numpy())
        qpos = self.warp_data.qpos.numpy()
        qvel = self.warp_data.qvel.numpy()
        count = int(np.sum(mask))
        self.goals[mask], cube = self._sample_goal_and_cube_positions(count)
        qpos[mask] = self.home_qpos[None, :]
        qvel[mask] = 0.0
        qpos[mask, self.cube_qpos_adr : self.cube_qpos_adr + 3] = cube
        qpos[mask, self.cube_qpos_adr + 3] = 1.0
        self.step_counts[mask] = 0
        self.lifted_ever[mask] = False
        self.last_actions[mask] = 0.0
        self.warp_data.qpos.assign(qpos)
        self.warp_data.qvel.assign(qvel)
        self.mujoco_warp.forward(self.warp_model, self.warp_data)
        obs = self._obs_from_arrays(qpos, qvel)
        self._set_initial_distance_metrics(obs, mask)
        return obs

    def step(self, actions: np.ndarray):
        actions = np.clip(np.asarray(actions, dtype=np.float32), -1.0, 1.0)
        ctrl = _denormalize_action(actions, self.ctrl_low[None, :], self.ctrl_high[None, :])
        for _ in range(self.frame_skip):
            self.warp_data.ctrl.assign(ctrl.astype(np.float32))
            self.mujoco_warp.step(self.warp_model, self.warp_data)
        self.step_counts += 1
        qpos = self.warp_data.qpos.numpy()
        qvel = self.warp_data.qvel.numpy()
        self.last_actions = actions
        obs = self._obs_from_arrays(qpos, qvel)
        cube = obs[:, 3:6]
        ee = obs[:, 0:3]
        cube_to_goal = np.linalg.norm(cube - self.goals, axis=1).astype(np.float32)
        ee_to_cube = np.linalg.norm(ee - cube, axis=1).astype(np.float32)
        finger_width = _finger_width_from_qpos(qpos, self.finger_qpos_adrs)
        gripper_closed = finger_width < 0.035
        cube_lift = np.maximum(0.0, cube[:, 2] - CUBE_Z).astype(np.float32)
        lifted = cube_lift > LIFT_THRESHOLD
        self.lifted_ever |= lifted
        cube_at_goal = cube_to_goal < SUCCESS_THRESHOLD
        placed = (cube_at_goal & self.lifted_ever) if self.success_requires_lift else cube_at_goal
        reward = -cube_to_goal.astype(np.float32)
        if self.success_requires_lift:
            reward = np.where(placed, 0.0, np.where(self.lifted_ever, -cube_to_goal, -1.0)).astype(np.float32)
        if self.reward_type == "sparse":
            reward = (~placed).astype(np.float32) * -1.0
        done = placed | (self.step_counts >= self.max_steps)
        infos = {
            "ee_to_cube_distance": ee_to_cube,
            "cube_to_goal_distance": cube_to_goal,
            "cube_at_goal": cube_at_goal,
            "initial_ee_to_cube_distance": self.initial_ee_to_cube_distances.copy(),
            "initial_cube_to_goal_distance": self.initial_cube_to_goal_distances.copy(),
            "ee_to_cube_progress": self.initial_ee_to_cube_distances - ee_to_cube,
            "cube_to_goal_progress": self.initial_cube_to_goal_distances - cube_to_goal,
            "cube_lift_height": cube_lift,
            "near_cube": ee_to_cube < 0.055,
            "gripper_closed_near_cube": (ee_to_cube < 0.055) & gripper_closed,
            "lifted": lifted,
            "lifted_ever": self.lifted_ever.copy(),
            "placed_success": placed,
            "is_success": placed,
            "physics_backend": self.physics_backend,
        }
        return obs, reward, done.astype(bool), infos

    def _set_initial_distance_metrics(self, obs: np.ndarray, mask: np.ndarray) -> None:
        cube = obs[:, 3:6]
        ee = obs[:, 0:3]
        self.initial_ee_to_cube_distances[mask] = np.linalg.norm(ee[mask] - cube[mask], axis=1).astype(np.float32)
        self.initial_cube_to_goal_distances[mask] = np.linalg.norm(cube[mask] - self.goals[mask], axis=1).astype(np.float32)

    def _obs_from_arrays(self, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        cube = qpos[:, self.cube_qpos_adr : self.cube_qpos_adr + 3].astype(np.float32)
        cube_vel = qvel[:, self.cube_qvel_adr : self.cube_qvel_adr + 3].astype(np.float32)
        # Warp does not expose site_xpos through this thin wrapper, so use the
        # terminal link/body position only when available in xipos/xpos-like
        # fields; otherwise fall back to a geometric proxy from cube direction.
        try:
            xpos = self.warp_data.xpos.numpy()
            ee = xpos[:, self.ee_body_id if self.ee_body_id >= 0 else -1, :].astype(np.float32)
        except Exception:
            ee = cube + np.asarray([0.0, 0.0, 0.12], dtype=np.float32)
        joint_pos = qpos[:, self.robot_qpos_adrs].astype(np.float32)
        joint_vel = qvel[:, self.robot_qvel_adrs].astype(np.float32)
        return np.concatenate(
            [
                ee,
                cube,
                self.goals,
                cube - ee,
                self.goals - cube,
                joint_pos,
                joint_vel,
                self.last_actions,
                cube,
                self.goals,
            ],
            axis=1,
        ).astype(np.float32)

    def close(self) -> None:
        return None


def resolve_panda_xml_path(model_path: str | None = None) -> Path:
    candidates: list[Path] = []
    if model_path:
        candidates.append(Path(model_path).expanduser())
    env_path = os.environ.get("AUTORESEARCH_PANDA_MJCF")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    menagerie_root = os.environ.get("MUJOCO_MENAGERIE_PATH")
    if menagerie_root:
        candidates.append(Path(menagerie_root).expanduser() / "franka_emika_panda" / "panda.xml")
    candidates.append(Path.cwd() / ".external" / "mujoco_menagerie" / "franka_emika_panda" / "panda.xml")
    try:
        from robot_descriptions import panda_mj_description  # type: ignore

        candidates.append(Path(panda_mj_description.MJCF_PATH))
    except Exception:
        pass
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not find MuJoCo Menagerie Panda MJCF. Install `robot_descriptions`, set "
        "AUTORESEARCH_PANDA_MJCF, set MUJOCO_MENAGERIE_PATH, or clone Menagerie into "
        ".external/mujoco_menagerie."
    )


def _write_scene_xml(panda_xml_path: Path) -> Path:
    scene_dir = Path(tempfile.gettempdir()) / "autoresearch_gym_mjwarp_panda"
    scene_dir.mkdir(parents=True, exist_ok=True)
    scene_path = scene_dir / f"scene_{abs(hash(str(panda_xml_path)))}.xml"
    tree = ET.parse(panda_xml_path)
    root = tree.getroot()
    root.set("model", "autoresearch_panda_pick_place")
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    for attr in ("meshdir", "texturedir"):
        value = compiler.get(attr)
        if value and not Path(value).is_absolute():
            compiler.set(attr, str((panda_xml_path.parent / value).resolve()))
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", "0.002")
    option.set("integrator", "implicitfast")
    option.set("cone", "elliptic")
    size = root.find("size")
    if size is None:
        size = ET.SubElement(root, "size")
    size.set("njmax", str(max(int(size.get("njmax", "0")), 512)))
    size.set("nconmax", str(max(int(size.get("nconmax", "0")), 256)))
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    global_visual = visual.find("global")
    if global_visual is None:
        global_visual = ET.SubElement(visual, "global")
    global_visual.set("offwidth", str(max(int(global_visual.get("offwidth", "0")), 1024)))
    global_visual.set("offheight", str(max(int(global_visual.get("offheight", "0")), 768)))
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "material", name="ar_table", rgba="0.62 0.62 0.58 1")
    ET.SubElement(asset, "material", name="ar_cube", rgba="0.86 0.22 0.12 1")
    ET.SubElement(asset, "material", name="ar_goal", rgba="0.05 0.25 1.0 0.45")
    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(root, "worldbody")
    robot_base = worldbody.find("./body[@name='link0']")
    if robot_base is not None:
        robot_base.set("pos", f"{ROBOT_BASE_X:.3f} 0 0")
    ET.SubElement(worldbody, "light", name="ar_key", pos="0 0 2.8", dir="0 0 -1", diffuse="0.8 0.8 0.8")
    ET.SubElement(
        worldbody,
        "geom",
        name="floor",
        type="plane",
        pos=f"0 0 {-TABLE_HEIGHT:.3f}",
        size="2 2 0.01",
        rgba="0.18 0.18 0.18 1",
        contype="1",
        conaffinity="1",
    )
    ET.SubElement(
        worldbody,
        "geom",
        name="table",
        type="box",
        pos=f"{TABLE_CENTER_X:.3f} 0 {TABLE_TOP_Z - TABLE_HEIGHT / 2:.3f}",
        size=f"{TABLE_LENGTH / 2:.3f} {TABLE_WIDTH / 2:.3f} {TABLE_HEIGHT / 2:.3f}",
        material="ar_table",
        contype="1",
        conaffinity="1",
    )
    cube = ET.SubElement(worldbody, "body", name="object", pos=f"0 0 {CUBE_Z:.3f}")
    ET.SubElement(cube, "freejoint", name="cube_freejoint")
    ET.SubElement(
        cube,
        "geom",
        name="cube",
        type="box",
        size=f"{CUBE_SIZE:.3f} {CUBE_SIZE:.3f} {CUBE_SIZE:.3f}",
        mass="0.08",
        material="ar_cube",
        friction="1.0 0.005 0.0001",
        contype="1",
        conaffinity="1",
    )
    ET.SubElement(cube, "site", name="cube_center", size="0.008", rgba="1 1 1 0")
    ET.SubElement(
        worldbody,
        "site",
        name="target",
        pos=f"0.12 0 {CUBE_Z:.3f}",
        type="box",
        size=f"{CUBE_SIZE:.3f} {CUBE_SIZE:.3f} {CUBE_SIZE:.3f}",
        rgba="0.1 0.9 0.1 0.3",
    )
    ET.indent(tree, space="  ")
    tree.write(scene_path, encoding="utf-8", xml_declaration=False)
    return scene_path


def _prepare_model_for_mujoco_warp(mujoco_module: Any, model: Any) -> None:
    if hasattr(mujoco_module, "mjtDisableBit") and hasattr(mujoco_module.mjtDisableBit, "mjDSBL_MULTICCD"):
        model.opt.disableflags |= int(mujoco_module.mjtDisableBit.mjDSBL_MULTICCD.value)
    if hasattr(model, "geom_margin"):
        model.geom_margin[:] = 0.0


def _actuator_ctrl_ranges(model: Any) -> tuple[np.ndarray, np.ndarray]:
    ranges = np.asarray(model.actuator_ctrlrange[: model.nu], dtype=np.float32)
    low = ranges[:, 0].copy()
    high = ranges[:, 1].copy()
    invalid = ~np.isfinite(low) | ~np.isfinite(high) | (high <= low)
    low[invalid] = -1.0
    high[invalid] = 1.0
    return low.astype(np.float32), high.astype(np.float32)


def _goal_distance(cube_pos: np.ndarray, goal: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(cube_pos, dtype=np.float32) - np.asarray(goal, dtype=np.float32)))


def _move_cube_away_from_goal(cube_pos: np.ndarray, goal: np.ndarray) -> np.ndarray:
    cube = np.asarray(cube_pos, dtype=np.float32).copy()
    goal_arr = np.asarray(goal, dtype=np.float32)
    cube[0] = np.clip(goal_arr[0] + 0.08, -XY_RANGE, XY_RANGE)
    if _goal_distance(cube, goal_arr) < SUCCESS_THRESHOLD:
        cube[0] = np.clip(goal_arr[0] - 0.08, -XY_RANGE, XY_RANGE)
    if _goal_distance(cube, goal_arr) < SUCCESS_THRESHOLD:
        cube[1] = np.clip(goal_arr[1] + 0.08, -XY_RANGE, XY_RANGE)
    cube[2] = CUBE_Z
    return cube


def _move_cubes_away_from_goals(cube_pos: np.ndarray, goals: np.ndarray) -> np.ndarray:
    cubes = np.asarray(cube_pos, dtype=np.float32).copy()
    for idx in range(len(cubes)):
        cubes[idx] = _move_cube_away_from_goal(cubes[idx], goals[idx])
    return cubes


def _denormalize_action(action: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return (low + 0.5 * (np.asarray(action, dtype=np.float32) + 1.0) * (high - low)).astype(np.float32)


def _actuated_joint_ids(model: Any) -> np.ndarray:
    joint_type = 0
    try:
        import mujoco  # type: ignore

        joint_type = int(mujoco.mjtTrn.mjTRN_JOINT)
    except Exception:
        pass
    ids = []
    for actuator_id in range(int(model.nu)):
        if int(model.actuator_trntype[actuator_id]) == joint_type:
            ids.append(int(model.actuator_trnid[actuator_id, 0]))
    return np.asarray(ids, dtype=np.int32)


def _actuated_qpos_adrs(model: Any) -> np.ndarray:
    joint_ids = _actuated_joint_ids(model)
    adrs = []
    for joint_id in joint_ids:
        if joint_id >= 0:
            adrs.append(int(model.jnt_qposadr[int(joint_id)]))
    return np.asarray(adrs, dtype=np.int32)


def _actuated_qvel_adrs(model: Any) -> np.ndarray:
    joint_ids = _actuated_joint_ids(model)
    adrs = []
    for joint_id in joint_ids:
        if joint_id >= 0:
            adrs.append(int(model.jnt_dofadr[int(joint_id)]))
    return np.asarray(adrs, dtype=np.int32)


def _finger_qpos_adrs(mujoco_module: Any, model: Any) -> np.ndarray:
    adrs = []
    for name in ("finger_joint1", "finger_joint2"):
        joint_id = mujoco_module.mj_name2id(model, mujoco_module.mjtObj.mjOBJ_JOINT, name)
        if joint_id >= 0:
            adrs.append(int(model.jnt_qposadr[int(joint_id)]))
    return np.asarray(adrs, dtype=np.int32)


def _find_site(mujoco_module: Any, model: Any, names: list[str]) -> int:
    for name in names:
        site_id = mujoco_module.mj_name2id(model, mujoco_module.mjtObj.mjOBJ_SITE, name)
        if site_id >= 0:
            return int(site_id)
    return -1


def _find_body(mujoco_module: Any, model: Any, names: list[str]) -> int:
    for name in names:
        body_id = mujoco_module.mj_name2id(model, mujoco_module.mjtObj.mjOBJ_BODY, name)
        if body_id >= 0:
            return int(body_id)
    return -1


def _finger_width(data: Any, finger_qpos_adrs: np.ndarray) -> float:
    if finger_qpos_adrs.size < 2:
        return 1.0
    return float(np.sum(np.maximum(0.0, np.asarray(data.qpos[finger_qpos_adrs], dtype=np.float32))))


def _finger_width_from_qpos(qpos: np.ndarray, finger_qpos_adrs: np.ndarray) -> np.ndarray:
    if finger_qpos_adrs.size < 2:
        return np.ones(qpos.shape[0], dtype=np.float32)
    return np.sum(np.maximum(0.0, qpos[:, finger_qpos_adrs].astype(np.float32)), axis=1)


def _open_gripper_action(action_dim: int) -> np.ndarray:
    action = np.zeros(int(action_dim), dtype=np.float32)
    if action.size > 7:
        action[7] = 1.0
    return action
