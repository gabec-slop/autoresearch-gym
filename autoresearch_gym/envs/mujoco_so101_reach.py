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


SUCCESS_THRESHOLD = 0.035
DEFAULT_FRAME_SKIP = 10
TARGET_CENTER = np.asarray([0.32, 0.0, 0.18], dtype=np.float32)
TARGET_RANGE = np.asarray([0.07, 0.07, 0.05], dtype=np.float32)
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


class AutoresearchMujocoSO101ReachEnv(gym.Env[dict[str, np.ndarray], np.ndarray]):
    """SO-101 MuJoCo reach task.

    Uses the MuJoCo Menagerie `robotstudio_so101` model when available, with a
    mesh-free primitive fallback for minimal installs.
    """

    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        render_mode: str | None = "rgb_array",
        max_steps: int = 80,
        frame_skip: int = DEFAULT_FRAME_SKIP,
        reward_type: str = "dense",
        render_width: int = 720,
        render_height: int = 480,
        vision_observation: bool = False,
        vision_image_size: int = 84,
        vision_frame_stack: int = 3,
        model_path: str | None = None,
        allow_primitive_fallback: bool = True,
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
        if self.so101_xml_path is not None:
            self.scene_xml_path = write_so101_reach_scene_xml(self.so101_xml_path)
            self.model_source = "mujoco_menagerie_robotstudio_so101"
        elif allow_primitive_fallback:
            self.scene_xml_path = write_so101_primitive_reach_scene_xml()
            self.model_source = "mujoco_primitive_so101"
        else:
            raise FileNotFoundError(
                "Could not find MuJoCo Menagerie robotstudio_so101/so101.xml. "
                "Set AUTORESEARCH_SO101_MJCF, set MUJOCO_MENAGERIE_PATH, or clone "
                "Menagerie into .external/mujoco_menagerie."
            )
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
                        low=TARGET_CENTER - TARGET_RANGE,
                        high=TARGET_CENTER + TARGET_RANGE,
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
                        low=TARGET_CENTER - TARGET_RANGE,
                        high=TARGET_CENTER + TARGET_RANGE,
                        dtype=np.float32,
                    ),
                }
            )

    def _sample_target(self) -> np.ndarray:
        return (TARGET_CENTER + self.rng.uniform(-TARGET_RANGE, TARGET_RANGE)).astype(np.float32)

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
                "size": "0.018",
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


def write_so101_primitive_reach_scene_xml() -> Path:
    scene_dir = Path(tempfile.gettempdir()) / "autoresearch_gym_so101"
    scene_dir.mkdir(parents=True, exist_ok=True)
    scene_path = scene_dir / "so101_reach_primitive.xml"
    scene_path.write_text(_SO101_REACH_XML, encoding="utf-8")
    return scene_path


def _actuator_ctrl_ranges(model: Any) -> tuple[np.ndarray, np.ndarray]:
    ranges = np.asarray(model.actuator_ctrlrange, dtype=np.float32)
    return ranges[:, 0], ranges[:, 1]


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


_SO101_REACH_XML = """
<mujoco model="autoresearch_so101_reach">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast"/>
  <visual>
    <global offwidth="1024" offheight="768"/>
  </visual>
  <asset>
    <material name="printed_yellow" rgba="1.0 0.82 0.12 1"/>
    <material name="servo_black" rgba="0.08 0.08 0.08 1"/>
    <material name="target_green" rgba="0.1 0.8 0.25 0.55"/>
    <material name="table_mat" rgba="0.55 0.58 0.62 1"/>
  </asset>
  <default>
    <joint damping="0.60" frictionloss="0.052" armature="0.028"/>
    <position kp="60" kv="2.0" forcerange="-3.35 3.35"/>
    <geom condim="3" friction="1.0 0.005 0.0001"/>
  </default>
  <worldbody>
    <light name="key" pos="0 -0.8 1.2" dir="0 0.7 -1" diffuse="0.9 0.9 0.9"/>
    <camera name="world" pos="0.55 -0.50 0.43" xyaxes="0.673 0.740 0 -0.329 0.299 0.896" fovy="46"/>
    <geom name="floor" type="plane" pos="0 0 -0.005" size="0.8 0.8 0.02" material="table_mat"/>
    <site name="target" type="sphere" pos="-0.18 0 0.11" size="0.018" material="target_green"/>
    <body name="base" pos="0 0 0">
      <inertial pos="0.0137179 -0.000052 0.0334843" mass="0.147" fullinertia="0.000114686 0.000136117 0.000130364 -4.59787e-07 4.97151e-06 9.75275e-08"/>
      <geom name="base_plate" type="cylinder" pos="0 0 0.012" size="0.055 0.012" material="printed_yellow"/>
      <geom name="base_servo" type="box" pos="0.0263 0 0.0437" size="0.022 0.018 0.022" material="servo_black"/>
      <body name="shoulder" pos="0.0388353 0 0.0624" quat="0 0 -1 0">
        <joint axis="0 0 1" name="shoulder_pan" type="hinge" range="-1.9198621771937616 1.9198621771937634"/>
        <inertial pos="-0.0307604 -0.0000167 -0.0252713" mass="0.100006" fullinertia="8.3759e-05 8.10403e-05 2.39783e-05 7.55525e-08 -1.16342e-06 1.54663e-07"/>
        <geom name="shoulder_servo" type="box" pos="-0.0303992 0.000422241 -0.0417" size="0.021 0.018 0.022" material="servo_black"/>
        <geom name="shoulder_bracket" type="capsule" fromto="-0.02 0 0.0 -0.075 0 0.018" size="0.014" material="printed_yellow"/>
        <body name="upper_arm" pos="-0.0303992 -0.0182778 -0.0542" quat="0.5 -0.5 -0.5 -0.5">
          <joint axis="0 0 1" name="shoulder_lift" type="hinge" range="-1.7453292519943224 1.7453292519943366"/>
          <inertial pos="-0.0898471 -0.00838224 0.0184089" mass="0.103" fullinertia="4.08002e-05 0.000147318 0.000142487 -1.97819e-05 -4.03016e-08 8.97326e-09"/>
          <geom name="upper_servo" type="box" pos="-0.11257 -0.0155 0.0187" size="0.021 0.018 0.022" material="servo_black"/>
          <geom name="upper_link" type="capsule" fromto="-0.010 0.012 0.0182 -0.112 0.012 0.0182" size="0.012" material="printed_yellow"/>
          <body name="lower_arm" pos="-0.11257 -0.028 0" quat="0.707107 0 0 0.707107">
            <joint axis="0 0 1" name="elbow_flex" type="hinge" range="-1.69 1.69"/>
            <inertial pos="-0.0980701 0.00324376 0.0182831" mass="0.104" fullinertia="2.87438e-05 0.000159844 0.00014529 7.41152e-06 1.26409e-06 -4.90188e-08"/>
            <geom name="lower_servo" type="box" pos="-0.1224 0.0052 0.0187" size="0.021 0.018 0.022" material="servo_black"/>
            <geom name="lower_link" type="capsule" fromto="-0.010 -0.032 0.0182 -0.130 -0.032 0.0182" size="0.012" material="printed_yellow"/>
            <body name="wrist" pos="-0.1349 0.0052 0" quat="0.707107 0 0 -0.707107">
              <joint axis="0 0 1" name="wrist_flex" type="hinge" range="-1.6580628494556928 1.6580627293335335"/>
              <inertial pos="-0.000103312 -0.0386143 0.0281156" mass="0.079" fullinertia="3.68263e-05 2.5391e-05 2.1e-05 1.7893e-08 -5.28128e-08 3.6412e-06"/>
              <geom name="wrist_servo" type="box" pos="0 -0.0424 0.0306" size="0.020 0.018 0.020" material="servo_black"/>
              <geom name="wrist_link" type="capsule" fromto="0 -0.005 0.018 0 -0.061 0.018" size="0.010" material="printed_yellow"/>
              <body name="gripper" pos="0 -0.0611 0.0181" quat="0.0172091 -0.0172091 0.706897 0.706897">
                <joint axis="0 0 1" name="wrist_roll" type="hinge" range="-2.7438472969992493 2.841206309382605"/>
                <inertial pos="0.000213627 0.000245138 -0.025187" mass="0.087" fullinertia="2.75087e-05 4.33657e-05 3.45059e-05 -3.35241e-07 -5.7352e-06 -5.17847e-08"/>
                <geom name="gripper_body" type="box" pos="0 0 -0.023" size="0.018 0.018 0.028" material="servo_black"/>
                <geom name="fixed_finger" type="capsule" fromto="-0.018 0.010 -0.045 -0.018 0.010 -0.095" size="0.004" material="printed_yellow"/>
                <camera name="wrist" pos="0 0.04 -0.04" euler="-0.5 0.0 6.28" fovy="75"/>
                <site group="3" name="gripperframe" pos="-0.0079 -0.000218121 -0.0981274" quat="0.707107 0 0.707107 0" size="0.009" rgba="0.1 0.4 1.0 1"/>
                <body name="moving_jaw_so101_v1" pos="0.0202 0.0188 -0.0234" quat="0.707107 0.707107 0 0">
                  <joint axis="0 0 1" name="gripper" type="hinge" range="-0.17453297762778586 1.7453291995659765"/>
                  <inertial pos="-0.00157495 -0.0300244 0.0192755" mass="0.012" fullinertia="6.61427e-06 1.89032e-06 5.28738e-06 -3.19807e-07 -5.90717e-09 -1.09945e-07"/>
                  <geom name="moving_finger" type="capsule" fromto="0 0 0.018 0 -0.050 0.018" size="0.004" material="printed_yellow"/>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="shoulder_pan" joint="shoulder_pan" ctrlrange="-1.91986 1.91986"/>
    <position name="shoulder_lift" joint="shoulder_lift" ctrlrange="-1.74533 1.74533"/>
    <position name="elbow_flex" joint="elbow_flex" ctrlrange="-1.69 1.69"/>
    <position name="wrist_flex" joint="wrist_flex" ctrlrange="-1.65806 1.65806"/>
    <position name="wrist_roll" joint="wrist_roll" ctrlrange="-2.74385 2.84121"/>
    <position name="gripper" joint="gripper" ctrlrange="-0.17453 1.74533"/>
  </actuator>
</mujoco>
""".strip()
