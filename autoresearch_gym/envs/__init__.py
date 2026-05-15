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


def __getattr__(name: str):
    if name == "BatToGoalEnv":
        from autoresearch_gym.envs.bat_to_goal import BatToGoalEnv

        return BatToGoalEnv
    if name == "AutoresearchPandaPickAndPlaceEnv":
        from autoresearch_gym.envs.panda_pick_and_place import AutoresearchPandaPickAndPlaceEnv

        return AutoresearchPandaPickAndPlaceEnv
    raise AttributeError(name)


__all__ = ["AutoresearchPandaPickAndPlaceEnv", "BatToGoalEnv", "register_envs"]
