from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from panda_gym.envs.robots.panda import Panda
from panda_gym.pybullet import PyBullet


@dataclass
class BatToGoalConfig:
    render_mode: str = "rgb_array"
    renderer: str = "Tiny"
    control_type: str = "ee"
    reward_type: str = "dense"
    max_steps: int = 240
    ball_radius: float = 0.03
    goal_radius: float = 0.09
    goal_height: float = 0.16
    table_half_extents: tuple[float, float, float] = (1.1 / 2, 0.7 / 2, 0.4 / 2)
    # Keep the task centered in front of the robot rather than drifting wide in y.
    ball_launch_low: tuple[float, float, float] = (0.38, -0.08, 0.22)
    ball_launch_high: tuple[float, float, float] = (0.48, 0.08, 0.32)
    strike_zone_low: tuple[float, float, float] = (0.00, -0.06, 0.18)
    strike_zone_high: tuple[float, float, float] = (0.14, 0.06, 0.30)
    # Place the goal beyond the screen-right (+x) table edge in the current camera view.
    goal_low: tuple[float, float, float] = (0.34, -0.06, 0.14)
    goal_high: tuple[float, float, float] = (0.44, 0.06, 0.26)
    launch_time_low: float = 0.40
    launch_time_high: float = 0.62
    launch_noise_scale: float = 0.02
    paddle_half_extents: tuple[float, float, float] = (0.012, 0.075, 0.065)
    paddle_offset: tuple[float, float, float] = (0.07, 0.0, 0.0)
    paddle_mass: float = 0.04
    paddle_restitution: float = 0.92
    floor_z: float = -0.4


class BatToGoalEnv(gym.Env[np.ndarray, np.ndarray]):
    """Continuous-control Panda task: bat an incoming ball into a goal volume."""

    metadata = {"render_modes": ["human", "rgb_array"]}
    gravity = np.array([0.0, 0.0, -9.81], dtype=np.float32)

    def __init__(
        self,
        render_mode: str = "rgb_array",
        renderer: str = "Tiny",
        control_type: str = "ee",
        reward_type: str = "dense",
        max_steps: int = 240,
    ) -> None:
        super().__init__()
        self.config = BatToGoalConfig(
            render_mode=render_mode,
            renderer=renderer,
            control_type=control_type,
            reward_type=reward_type,
            max_steps=max_steps,
        )
        self.sim = PyBullet(render_mode=render_mode, renderer=renderer)
        self.robot = Panda(
            self.sim,
            block_gripper=True,
            base_position=np.array([-0.6, 0.0, 0.0]),
            control_type=control_type,
        )
        self.render_mode = render_mode
        self.action_space = self.robot.action_space
        self.np_random = np.random.default_rng()

        self.ball_name = "ball"
        self.goal_name = "goal"
        self.paddle_name = "paddle"
        self._paddle_constraint: int | None = None
        self._goal = np.zeros(3, dtype=np.float32)
        self._ball_launch = np.zeros(3, dtype=np.float32)
        self._ball_velocity = np.zeros(3, dtype=np.float32)
        self._previous_ball_goal_distance = 0.0
        self._best_ball_goal_distance_after_contact = np.inf
        self._has_touched_ball = False
        self._ball_to_goal_after_contact = False
        self._goal_eligible = False
        self._step_count = 0
        self._prev_ee_pos = np.zeros(3, dtype=np.float32)
        self._prev_paddle_pos = np.zeros(3, dtype=np.float32)
        self._prev_ball_pos = np.zeros(3, dtype=np.float32)
        self._prev_ball_vel = np.zeros(3, dtype=np.float32)
        self._prev_action = np.zeros(7, dtype=np.float32)

        with self.sim.no_rendering():
            self._create_scene()
            self.sim.place_visualizer(
                target_position=np.array([0.05, 0.0, 0.15]),
                distance=1.45,
                yaw=35.0,
                pitch=-28.0,
            )

        obs = self._get_obs()
        obs_shape = obs.shape
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=obs_shape,
            dtype=np.float32,
        )

    def _create_scene(self) -> None:
        cfg = self.config
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(
            length=cfg.table_half_extents[0] * 2,
            width=cfg.table_half_extents[1] * 2,
            height=cfg.table_half_extents[2] * 2,
            x_offset=-0.3,
            lateral_friction=1.0,
            spinning_friction=0.01,
        )
        self.sim.create_sphere(
            body_name=self.ball_name,
            radius=cfg.ball_radius,
            mass=0.08,
            position=np.array([0.45, 0.0, 0.2]),
            rgba_color=np.array([0.95, 0.35, 0.1, 1.0]),
            lateral_friction=0.4,
            spinning_friction=0.001,
        )
        self.sim.create_cylinder(
            body_name=self.goal_name,
            radius=cfg.goal_radius,
            height=cfg.goal_height,
            mass=0.0,
            ghost=True,
            position=np.array([0.28, 0.0, 0.18]),
            rgba_color=np.array([0.1, 0.9, 0.2, 0.25]),
        )
        self._create_paddle()

    def _ee_orientation(self) -> np.ndarray:
        return np.array(
            self._require_valid(
                self.sim.get_link_orientation(self.robot.body_name, self.robot.ee_link),
                "ee_orientation",
            ),
            dtype=np.float32,
        )

    def _rotate_local_vector(self, quat: np.ndarray, local_vec: np.ndarray) -> np.ndarray:
        rotation = np.array(self.sim.physics_client.getMatrixFromQuaternion(quat.tolist()), dtype=np.float32).reshape(3, 3)
        return rotation @ local_vec

    def _paddle_pose_from_ee(self) -> tuple[np.ndarray, np.ndarray]:
        ee_pos = self._ee_position()
        ee_orn = self._ee_orientation()
        offset_world = self._rotate_local_vector(ee_orn, np.array(self.config.paddle_offset, dtype=np.float32))
        return ee_pos + offset_world, ee_orn

    def _settle_paddle(self, steps: int = 5) -> None:
        for _ in range(steps):
            self.sim.step()

    def _hard_reset_robot(self) -> None:
        neutral = self.robot.neutral_joint_values.astype(np.float32)
        robot_id = self.sim._bodies_idx[self.robot.body_name]
        for joint_index, angle in zip(self.robot.joint_indices, neutral):
            self.sim.physics_client.resetJointState(
                bodyUniqueId=robot_id,
                jointIndex=int(joint_index),
                targetValue=float(angle),
                targetVelocity=0.0,
            )
        self.robot.control_joints(target_angles=neutral)

    def _sync_paddle_pose(self) -> None:
        paddle_position, paddle_orientation = self._paddle_pose_from_ee()
        self.sim.set_base_pose(self.paddle_name, paddle_position, paddle_orientation)
        paddle_id = self.sim._bodies_idx[self.paddle_name]
        self.sim.physics_client.resetBaseVelocity(
            objectUniqueId=paddle_id,
            linearVelocity=[0.0, 0.0, 0.0],
            angularVelocity=[0.0, 0.0, 0.0],
        )

    def _create_paddle(self) -> None:
        paddle_position, paddle_orientation = self._paddle_pose_from_ee()
        self.sim.create_box(
            body_name=self.paddle_name,
            half_extents=np.array(self.config.paddle_half_extents, dtype=np.float32),
            mass=self.config.paddle_mass,
            position=paddle_position,
            rgba_color=np.array([0.2, 0.72, 0.95, 0.95]),
            lateral_friction=0.9,
            spinning_friction=0.02,
        )
        self.sim.set_base_pose(self.paddle_name, paddle_position, paddle_orientation)

        paddle_id = self.sim._bodies_idx[self.paddle_name]
        robot_id = self.sim._bodies_idx[self.robot.body_name]
        self.sim.physics_client.changeDynamics(
            paddle_id,
            -1,
            restitution=self.config.paddle_restitution,
            lateralFriction=0.9,
            spinningFriction=0.02,
        )
        self._paddle_constraint = self.sim.physics_client.createConstraint(
            parentBodyUniqueId=robot_id,
            parentLinkIndex=self.robot.ee_link,
            childBodyUniqueId=paddle_id,
            childLinkIndex=-1,
            jointType=self.sim.physics_client.JOINT_FIXED,
            jointAxis=[0.0, 0.0, 0.0],
            parentFramePosition=[0.0, 0.0, 0.0],
            childFramePosition=[-value for value in self.config.paddle_offset],
            parentFrameOrientation=[0.0, 0.0, 0.0, 1.0],
            childFrameOrientation=[0.0, 0.0, 0.0, 1.0],
        )
        for link_idx in range(-1, self.sim.physics_client.getNumJoints(robot_id)):
            self.sim.physics_client.setCollisionFilterPair(paddle_id, robot_id, -1, link_idx, 0)

    def _sample_ball_launch(self) -> np.ndarray:
        cfg = self.config
        return self.np_random.uniform(cfg.ball_launch_low, cfg.ball_launch_high).astype(np.float32)

    def _sample_goal(self) -> np.ndarray:
        cfg = self.config
        return self.np_random.uniform(cfg.goal_low, cfg.goal_high).astype(np.float32)

    def _sample_strike_point(self) -> np.ndarray:
        cfg = self.config
        return self.np_random.uniform(cfg.strike_zone_low, cfg.strike_zone_high).astype(np.float32)

    def _option_vec3(self, options: dict[str, Any], key: str, default: tuple[float, float, float]) -> np.ndarray:
        return np.array(options.get(key, default), dtype=np.float32)

    def _range_from_options(
        self,
        options: dict[str, Any],
        prefix: str,
        default_low: tuple[float, float, float],
        default_high: tuple[float, float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        low = self._option_vec3(options, f"{prefix}_low", default_low)
        high = self._option_vec3(options, f"{prefix}_high", default_high)
        return low, high

    def _sample_box_point(self, low: np.ndarray, high: np.ndarray) -> np.ndarray:
        return self.np_random.uniform(low, high).astype(np.float32)

    def _sample_strike_point_from_range(self, low: np.ndarray, high: np.ndarray) -> np.ndarray:
        return self._sample_box_point(low, high)

    def _sample_launch_velocity(
        self,
        launch_position: np.ndarray,
        strike_zone_low: np.ndarray | None = None,
        strike_zone_high: np.ndarray | None = None,
        launch_time_low: float | None = None,
        launch_time_high: float | None = None,
        launch_noise_scale: float | None = None,
    ) -> np.ndarray:
        strike_zone_low = (
            np.array(self.config.strike_zone_low, dtype=np.float32)
            if strike_zone_low is None
            else np.array(strike_zone_low, dtype=np.float32)
        )
        strike_zone_high = (
            np.array(self.config.strike_zone_high, dtype=np.float32)
            if strike_zone_high is None
            else np.array(strike_zone_high, dtype=np.float32)
        )
        launch_time_low = self.config.launch_time_low if launch_time_low is None else float(launch_time_low)
        launch_time_high = self.config.launch_time_high if launch_time_high is None else float(launch_time_high)
        launch_noise_scale = self.config.launch_noise_scale if launch_noise_scale is None else float(launch_noise_scale)
        strike_point = self._sample_strike_point_from_range(strike_zone_low, strike_zone_high)
        strike_point += self.np_random.normal(0.0, launch_noise_scale, size=3)
        strike_point[2] = np.clip(strike_point[2], strike_zone_low[2], strike_zone_high[2])
        flight_time = float(self.np_random.uniform(launch_time_low, launch_time_high))
        velocity = (strike_point - launch_position - 0.5 * self.gravity * (flight_time**2)) / flight_time
        velocity[0] = min(velocity[0], -0.35)
        return velocity.astype(np.float32)

    def _set_ball_state(self, position: np.ndarray, velocity: np.ndarray) -> None:
        self.sim.set_base_pose(self.ball_name, position, np.array([0.0, 0.0, 0.0, 1.0]))
        ball_id = self.sim._bodies_idx[self.ball_name]
        self.sim.physics_client.resetBaseVelocity(
            objectUniqueId=ball_id,
            linearVelocity=velocity.tolist(),
            angularVelocity=[0.0, 0.0, 0.0],
        )

    def _require_valid(self, value: Any, what: str) -> Any:
        if value is None:
            raise RuntimeError(f"PyBullet returned invalid state for {what}")
        return value

    def _ball_position(self) -> np.ndarray:
        return np.array(self._require_valid(self.sim.get_base_position(self.ball_name), "ball_position"), dtype=np.float32)

    def _ball_velocity_vec(self) -> np.ndarray:
        return np.array(self._require_valid(self.sim.get_base_velocity(self.ball_name), "ball_velocity"), dtype=np.float32)

    def _ee_position(self) -> np.ndarray:
        return np.array(self._require_valid(self.robot.get_ee_position(), "ee_position"), dtype=np.float32)

    def _ee_velocity(self) -> np.ndarray:
        return np.array(self._require_valid(self.robot.get_ee_velocity(), "ee_velocity"), dtype=np.float32)

    def _paddle_position(self) -> np.ndarray:
        return np.array(
            self._require_valid(self.sim.get_base_position(self.paddle_name), "paddle_position"),
            dtype=np.float32,
        )

    def _arm_joint_angles(self) -> np.ndarray:
        return np.array([self.robot.get_joint_angle(joint=i) for i in range(7)], dtype=np.float32)

    def _arm_joint_velocities(self) -> np.ndarray:
        return np.array([self.robot.get_joint_velocity(joint=i) for i in range(7)], dtype=np.float32)

    def _format_prev_action(self, action: np.ndarray | None = None) -> np.ndarray:
        padded = np.zeros(7, dtype=np.float32)
        if action is not None:
            count = min(len(action), 7)
            padded[:count] = action[:count]
        return padded

    def _goal_distance(self) -> float:
        return float(np.linalg.norm(self._ball_position() - self._goal))

    def _ball_in_goal(self) -> bool:
        ball_pos = self._ball_position()
        radial_distance = np.linalg.norm((ball_pos - self._goal)[:2])
        vertical_distance = abs(ball_pos[2] - self._goal[2])
        return radial_distance <= self.config.goal_radius and vertical_distance <= self.config.goal_height / 2

    def _ball_hit_floor(self) -> bool:
        ball_id = self.sim._bodies_idx[self.ball_name]
        floor_id = self.sim._bodies_idx["plane"]
        floor_contacts = self.sim.physics_client.getContactPoints(bodyA=ball_id, bodyB=floor_id)
        if floor_contacts:
            return True
        # Fallback for edge cases where contact points lag a frame.
        return bool(self._ball_position()[2] <= self.config.floor_z + self.config.ball_radius)

    def _ball_out_of_bounds(self) -> bool:
        ball_pos = self._ball_position()
        return bool(
            ball_pos[0] < -0.3
            or ball_pos[0] > 0.82
            or abs(ball_pos[1]) > 0.6
            # End the episode as soon as the return reaches the floor.
            or self._ball_hit_floor()
            or ball_pos[2] > 0.8
        )

    def _ball_robot_contact(self) -> bool:
        ball_id = self.sim._bodies_idx[self.ball_name]
        robot_id = self.sim._bodies_idx[self.robot.body_name]
        paddle_id = self.sim._bodies_idx[self.paddle_name]
        robot_contacts = self.sim.physics_client.getContactPoints(bodyA=ball_id, bodyB=robot_id)
        paddle_contacts = self.sim.physics_client.getContactPoints(bodyA=ball_id, bodyB=paddle_id)
        return len(robot_contacts) > 0 or len(paddle_contacts) > 0

    def _ball_velocity_toward_goal(self) -> float:
        ball_pos = self._ball_position()
        ball_vel = self._ball_velocity_vec()
        goal_delta = self._goal - ball_pos
        norm = np.linalg.norm(goal_delta)
        if norm < 1e-6:
            return 0.0
        return float(np.dot(ball_vel, goal_delta / norm))

    def _get_obs(self) -> np.ndarray:
        ee_pos = self._ee_position()
        ee_vel = self._ee_velocity()
        ee_orientation = self._ee_orientation()
        paddle_pos = self._paddle_position()
        ball_pos = self._ball_position()
        ball_vel = self._ball_velocity_vec()
        joint_angles = self._arm_joint_angles()
        joint_velocities = self._arm_joint_velocities()

        obs = np.concatenate(
            [
                ee_pos,
                ee_vel,
                ee_orientation,
                paddle_pos,
                ball_pos,
                ball_vel,
                self._goal,
                self._ball_launch,
                ball_pos - ee_pos,
                self._goal - ball_pos,
                ball_pos - paddle_pos,
                joint_angles,
                joint_velocities,
                ee_pos - self._prev_ee_pos,
                paddle_pos - self._prev_paddle_pos,
                ball_pos - self._prev_ball_pos,
                ball_vel - self._prev_ball_vel,
                self._prev_action,
                np.array(
                    [
                        float(self._has_touched_ball),
                        float(self._ball_to_goal_after_contact),
                        float(self._goal_eligible),
                        float(self._step_count) / float(self.config.max_steps),
                    ],
                    dtype=np.float32,
                ),
            ]
        )
        return obs.astype(np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        options = dict(options or {})
        fixed_case = dict(options.get("fixed_case", {}) or {})

        self._step_count = 0
        self._has_touched_ball = False
        self._ball_to_goal_after_contact = False
        self._goal_eligible = False
        self._best_ball_goal_distance_after_contact = np.inf

        with self.sim.no_rendering():
            self._hard_reset_robot()
            self._settle_paddle()
            self._sync_paddle_pose()
            goal_low, goal_high = self._range_from_options(
                options,
                "goal",
                self.config.goal_low,
                self.config.goal_high,
            )
            launch_low, launch_high = self._range_from_options(
                options,
                "ball_launch",
                self.config.ball_launch_low,
                self.config.ball_launch_high,
            )
            strike_low, strike_high = self._range_from_options(
                options,
                "strike_zone",
                self.config.strike_zone_low,
                self.config.strike_zone_high,
            )
            launch_time_low = float(options.get("launch_time_low", self.config.launch_time_low))
            launch_time_high = float(options.get("launch_time_high", self.config.launch_time_high))
            launch_noise_scale = float(options.get("launch_noise_scale", self.config.launch_noise_scale))

            if "goal" in fixed_case:
                self._goal = np.array(fixed_case["goal"], dtype=np.float32)
            else:
                self._goal = self._sample_box_point(goal_low, goal_high)

            if "ball_launch" in fixed_case:
                self._ball_launch = np.array(fixed_case["ball_launch"], dtype=np.float32)
            else:
                self._ball_launch = self._sample_box_point(launch_low, launch_high)

            if "ball_velocity" in fixed_case:
                self._ball_velocity = np.array(fixed_case["ball_velocity"], dtype=np.float32)
            else:
                self._ball_velocity = self._sample_launch_velocity(
                    self._ball_launch,
                    strike_zone_low=strike_low,
                    strike_zone_high=strike_high,
                    launch_time_low=launch_time_low,
                    launch_time_high=launch_time_high,
                    launch_noise_scale=launch_noise_scale,
                )
            self.sim.set_base_pose(self.goal_name, self._goal, np.array([0.0, 0.0, 0.0, 1.0]))
            self._set_ball_state(self._ball_launch, self._ball_velocity)

        if self.render_mode == "human":
            self._sync_paddle_pose()
            time.sleep(0.03)

        self._prev_ee_pos = self._ee_position().copy()
        self._prev_paddle_pos = self._paddle_position().copy()
        self._prev_ball_pos = self._ball_position().copy()
        self._prev_ball_vel = self._ball_velocity_vec().copy()
        self._prev_action = self._format_prev_action()
        self._previous_ball_goal_distance = self._goal_distance()
        info = {
            "is_success": False,
            "contacted_ball": False,
            "ball_velocity_toward_goal": self._ball_velocity_toward_goal(),
        }
        return self._get_obs(), info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._step_count += 1
        prev_ee_pos = self._ee_position()
        prev_paddle_pos = self._paddle_position()
        prev_ball_pos = self._ball_position()
        prev_ball_vel = self._ball_velocity_vec()
        self.robot.set_action(action)
        self._sync_paddle_pose()
        self.sim.step()
        self._sync_paddle_pose()

        contact_now = self._ball_robot_contact()
        first_contact = contact_now and not self._has_touched_ball
        self._has_touched_ball = self._has_touched_ball or contact_now

        ball_goal_distance = self._goal_distance()
        velocity_toward_goal = self._ball_velocity_toward_goal()
        if self._has_touched_ball and velocity_toward_goal > 0.0:
            self._ball_to_goal_after_contact = True
            self._goal_eligible = True

        reward = -0.01
        if self.config.reward_type == "dense":
            ee_ball_distance = float(np.linalg.norm(self._ball_position() - self._ee_position()))
            if not self._has_touched_ball:
                # Before contact, only reward getting into striking position.
                reward += 0.035 * np.exp(-4.0 * ee_ball_distance)
            else:
                # After contact, reward making the ball meaningfully more goal-directed.
                reward += 0.16 * (self._previous_ball_goal_distance - ball_goal_distance)
                reward += 0.05 * np.clip(velocity_toward_goal, -3.0, 3.0)
                reward += 0.02 * np.exp(-3.0 * ball_goal_distance)
                # Make directional outgoing velocity matter more than just touching the ball.
                if velocity_toward_goal > 0.0:
                    reward += 0.12 * min(velocity_toward_goal**2, 9.0)
                else:
                    reward -= 0.08 * min(abs(velocity_toward_goal), 3.0)

                if np.isfinite(self._best_ball_goal_distance_after_contact) and ball_goal_distance < self._best_ball_goal_distance_after_contact:
                    reward += 0.45 * (self._best_ball_goal_distance_after_contact - ball_goal_distance)
                    self._best_ball_goal_distance_after_contact = ball_goal_distance

                if velocity_toward_goal < 0.0:
                    reward += 0.03 * velocity_toward_goal

        if first_contact:
            reward += 0.75
            self._best_ball_goal_distance_after_contact = ball_goal_distance
            if velocity_toward_goal > 0.0:
                reward += 0.5
            else:
                reward -= 0.25
        if self._ball_to_goal_after_contact:
            reward += 0.08

        terminated = False
        truncated = False

        success = self._goal_eligible and self._ball_in_goal()
        if success:
            reward += 25.0
            terminated = True
        elif self._ball_out_of_bounds():
            reward -= 3.0
            terminated = True
        elif self._step_count >= self.config.max_steps:
            truncated = True

        info = {
            "is_success": success,
            "contacted_ball": self._has_touched_ball,
            "first_contact": first_contact,
            "ball_goal_distance": ball_goal_distance,
            "ball_velocity_toward_goal": velocity_toward_goal,
            "ball_to_goal_after_contact": self._ball_to_goal_after_contact,
            "goal_eligible": self._goal_eligible,
        }
        self._prev_ee_pos = prev_ee_pos
        self._prev_paddle_pos = prev_paddle_pos
        self._prev_ball_pos = prev_ball_pos
        self._prev_ball_vel = prev_ball_vel
        self._prev_action = self._format_prev_action(action)
        self._previous_ball_goal_distance = ball_goal_distance
        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self, width: int = 720, height: int = 480) -> np.ndarray | None:
        return self.sim.render(
            width=width,
            height=height,
            target_position=np.array([0.05, 0.0, 0.16]),
            distance=1.45,
            yaw=35.0,
            pitch=-28.0,
            roll=0.0,
        )

    def close(self) -> None:
        self.sim.close()
