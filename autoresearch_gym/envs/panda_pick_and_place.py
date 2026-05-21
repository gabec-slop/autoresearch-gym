from __future__ import annotations

import ctypes
import os
import sys
from contextlib import contextmanager
from typing import Optional

import gymnasium as gym
import numpy as np


class AutoresearchPandaPickAndPlaceEnv:
    """PandaGym pick-and-place with a visually distinct goal marker.

    Dynamics stay delegated to PandaGym. This wrapper only recolors the ghost
    target so live dashboard frames do not make it look like a second solid cube.
    """

    def __new__(
        cls,
        render_mode: str = "rgb_array",
        reward_type: str = "dense",
        control_type: str = "ee",
        renderer: str = "Tiny",
        render_width: int = 720,
        render_height: int = 480,
        render_target_position: Optional[np.ndarray] = None,
        render_distance: float = 1.4,
        render_yaw: float = 45,
        render_pitch: float = -30,
        render_roll: float = 0,
    ):
        try:
            with _suppress_native_output():
                from panda_gym.envs.panda_tasks import PandaPickAndPlaceEnv

                env = PandaPickAndPlaceEnv(
                    render_mode=render_mode,
                    reward_type=reward_type,
                    control_type=control_type,
                    renderer=renderer,
                    render_width=render_width,
                    render_height=render_height,
                    render_target_position=render_target_position,
                    render_distance=render_distance,
                    render_yaw=render_yaw,
                    render_pitch=render_pitch,
                    render_roll=render_roll,
                )
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("AutoresearchPandaPickAndPlaceEnv requires the `panda` extra.") from exc
        _recolor_goal_marker(env)
        return _RejectInitialSuccessWrapper(env)


def _recolor_goal_marker(env) -> None:
    target_id = env.sim._bodies_idx.get("target")
    if target_id is None:
        return
    env.sim.physics_client.changeVisualShape(
        target_id,
        -1,
        rgbaColor=np.array([0.05, 0.25, 1.0, 0.65]),
        specularColor=np.zeros(3),
    )


class _RejectInitialSuccessWrapper(gym.Wrapper):
    """Reject reset samples that are already solved before any action."""

    def __init__(self, env: gym.Env, max_attempts: int = 64) -> None:
        super().__init__(env)
        self.max_attempts = int(max_attempts)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        last_obs = None
        last_info = None
        for attempt in range(self.max_attempts):
            attempt_seed = None if seed is None else int(seed) + attempt * 1009
            obs, info = self.env.reset(seed=attempt_seed, options=options)
            last_obs = obs
            last_info = dict(info)
            initial_distance = _goal_distance(obs)
            if initial_distance >= _success_threshold(self.env):
                last_info["initial_goal_distance"] = initial_distance
                last_info["initial_resample_attempts"] = attempt
                return obs, last_info

        assert last_obs is not None and last_info is not None
        last_info["initial_goal_distance"] = _goal_distance(last_obs)
        last_info["initial_resample_attempts"] = self.max_attempts
        return last_obs, last_info


def _goal_distance(obs) -> float:
    achieved = np.asarray(obs["achieved_goal"], dtype=np.float32)
    desired = np.asarray(obs["desired_goal"], dtype=np.float32)
    return float(np.linalg.norm(achieved - desired))


def _success_threshold(env) -> float:
    return float(getattr(env.unwrapped.task, "distance_threshold", 0.05))


@contextmanager
def _suppress_native_output():
    """Silence PyBullet's native argv echo during client construction."""

    sys.stdout.flush()
    sys.stderr.flush()
    _flush_native_stdio()
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)
    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        _flush_native_stdio()
        os.dup2(saved_stdout_fd, 1)
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
        os.close(devnull_fd)


def _flush_native_stdio() -> None:
    try:
        ctypes.CDLL(None).fflush(None)
    except Exception:
        pass
