from __future__ import annotations

from gymnasium.envs.registration import register, registry


def register_envs() -> None:
    if "PandaBatToGoal-v0" not in registry:
        register(
            id="PandaBatToGoal-v0",
            entry_point="autoresearch_gym.envs.bat_to_goal:BatToGoalEnv",
        )
    if "AutoresearchPandaPickAndPlaceDense-v0" not in registry:
        register(
            id="AutoresearchPandaPickAndPlaceDense-v0",
            entry_point="autoresearch_gym.envs.panda_pick_and_place:AutoresearchPandaPickAndPlaceEnv",
            max_episode_steps=50,
        )
    if "AutoresearchMujocoPandaPickAndPlaceDense-v0" not in registry:
        register(
            id="AutoresearchMujocoPandaPickAndPlaceDense-v0",
            entry_point="autoresearch_gym.envs.mujoco_panda_pick_and_place:AutoresearchMujocoPandaPickAndPlaceEnv",
            max_episode_steps=50,
        )
    if "AutoresearchMujocoPandaGymPickAndPlaceDense-v0" not in registry:
        register(
            id="AutoresearchMujocoPandaGymPickAndPlaceDense-v0",
            entry_point="autoresearch_gym.envs.mujoco_panda_pick_and_place:AutoresearchMujocoPandaPickAndPlaceEnv",
            max_episode_steps=50,
            kwargs={
                "reward_type": "dense",
                "success_requires_lift": False,
                "goal_xy_range": 0.30,
                "goal_z_range": 0.20,
                "obj_xy_range": 0.30,
                "tabletop_goal_probability": 0.30,
                "reject_initial_success": False,
            },
        )


def __getattr__(name: str):
    if name == "BatToGoalEnv":
        from autoresearch_gym.envs.bat_to_goal import BatToGoalEnv

        return BatToGoalEnv
    if name == "AutoresearchPandaPickAndPlaceEnv":
        from autoresearch_gym.envs.panda_pick_and_place import AutoresearchPandaPickAndPlaceEnv

        return AutoresearchPandaPickAndPlaceEnv
    if name == "AutoresearchMujocoPandaPickAndPlaceEnv":
        from autoresearch_gym.envs.mujoco_panda_pick_and_place import AutoresearchMujocoPandaPickAndPlaceEnv

        return AutoresearchMujocoPandaPickAndPlaceEnv
    raise AttributeError(name)


__all__ = ["AutoresearchMujocoPandaPickAndPlaceEnv", "AutoresearchPandaPickAndPlaceEnv", "BatToGoalEnv", "register_envs"]
