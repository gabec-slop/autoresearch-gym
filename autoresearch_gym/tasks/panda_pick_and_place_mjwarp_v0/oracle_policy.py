"""Task-owned scripted oracle for the MuJoCo/MJWarp Panda pick-and-place task.

This is the FK pinch-point hover/descend/close/lift/carry/release controller
originally developed inside the guided-warmup seed. It lives with the task
(not a seed) so it can serve as the learnability oracle: if this controller
cannot lift and place the cube under a benchmark's env_kwargs, the task or
environment contract is broken and no training run should be launched against
it. Seeds remain self-contained and must not import this module; sessions and
tests may.
"""

from __future__ import annotations

from typing import Any

import numpy as np


EE_ACTION_SCALE = 0.080
IK_DAMPING = 1.0e-4
GRIPPER_CLOSE_SIGN = -1.0
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


def flatten_observation(obs: Any) -> np.ndarray:
    if isinstance(obs, dict):
        pieces = []
        for key in ("observation", "achieved_goal", "desired_goal"):
            if key in obs:
                pieces.append(np.asarray(obs[key], dtype=np.float32).reshape(-1))
        return np.concatenate(pieces).astype(np.float32, copy=False)
    return np.asarray(obs, dtype=np.float32).reshape(-1)


class EndEffectorDeltaTool:
    """Map 4D EE delta actions into Menagerie Panda actuator commands.

    The action is `[dx, dy, dz, gripper]` in `[-1, 1]`. The first three
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
            raw[7] = float(np.clip(GRIPPER_CLOSE_SIGN * action[3], -1.0, 1.0))
        return raw


class ScriptedPickPlaceOracle:
    """Vectorized FK pinch-point pick-and-place state machine.

    Phases: hover over the cube, descend, close, lift until `lifted_ever`,
    carry to the goal, descend, release. `actions` emits 4D EE-delta tool
    actions; `advance` updates per-env phases from the post-step observation
    and info flags.
    """

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
            control_pos = obs[:, 0:3]
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
            control_pos = next_obs[:, 0:3]
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


def _info_bool_array(infos: dict[str, Any], key: str, count: int) -> np.ndarray:
    if key not in infos:
        return np.zeros(count, dtype=bool)
    value = np.asarray(infos[key])
    if value.shape == ():
        return np.full(count, bool(value), dtype=bool)
    return value.astype(bool, copy=False).reshape(-1)[:count]


def evaluate_oracle(env: Any, episodes: int = 5, seed: int = 0, max_steps: int | None = None) -> dict[str, float]:
    """Run the scripted oracle on a single (CPU) env and report outcome rates.

    `env` is the raw task env (dict observations, 8D raw action space). Returns
    success/lift rates over the requested episodes. Intended for task admission
    checks: a healthy task/env contract should give a clearly nonzero
    `lifted_ever_rate` and `success_rate` here before any RL run is launched.
    """

    base_env = env.unwrapped if hasattr(env, "unwrapped") else env
    tool = EndEffectorDeltaTool.from_env(base_env)
    oracle = ScriptedPickPlaceOracle(1)
    successes = 0
    lifted = 0
    horizon = int(max_steps if max_steps is not None else getattr(base_env, "max_steps", 400))
    for episode in range(int(episodes)):
        obs, info = env.reset(seed=int(seed) + episode)
        oracle.reset()
        episode_success = False
        episode_lifted = False
        for _ in range(horizon):
            flat = flatten_observation(obs).reshape(1, -1)
            control_pos = tool.batch_control_positions(flat)
            tool_action = oracle.actions(flat, noise_scale=0.0, control_pos=control_pos)[0]
            raw_action = tool.single_action(flat[0], tool_action)
            obs, _reward, terminated, truncated, info = env.step(raw_action)
            flat_next = flatten_observation(obs).reshape(1, -1)
            next_control_pos = tool.batch_control_positions(flat_next)
            oracle.advance(flat_next, np.asarray([terminated or truncated], dtype=bool), dict(info), control_pos=next_control_pos)
            episode_success = episode_success or bool(info.get("is_success", False))
            episode_lifted = episode_lifted or bool(info.get("lifted_ever", False))
            if terminated or truncated:
                break
        successes += int(episode_success)
        lifted += int(episode_lifted)
    return {
        "episodes": float(episodes),
        "success_rate": successes / float(episodes),
        "lifted_ever_rate": lifted / float(episodes),
    }


__all__ = [
    "EE_ACTION_SCALE",
    "IK_DAMPING",
    "GRIPPER_CLOSE_SIGN",
    "SCRIPTED_PHASES",
    "SCRIPTED_CONFIG",
    "EndEffectorDeltaTool",
    "ScriptedPickPlaceOracle",
    "evaluate_oracle",
    "flatten_observation",
]
