from __future__ import annotations

from typing import Optional

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
            from panda_gym.envs.panda_tasks import PandaPickAndPlaceEnv
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("AutoresearchPandaPickAndPlaceEnv requires the `panda` extra.") from exc

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
        _recolor_goal_marker(env)
        return env


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
