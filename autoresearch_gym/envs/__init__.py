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
    if "AutoresearchMujocoSO101Reach-v0" not in registry:
        register(
            id="AutoresearchMujocoSO101Reach-v0",
            entry_point="autoresearch_gym.envs.mujoco_so101_reach:AutoresearchMujocoSO101ReachEnv",
            max_episode_steps=150,
        )
    if "AutoresearchMujocoSO101CubeToBin-v0" not in registry:
        register(
            id="AutoresearchMujocoSO101CubeToBin-v0",
            entry_point="autoresearch_gym.envs.mujoco_so101_pick_place:AutoresearchMujocoSO101CubeToBinEnv",
            max_episode_steps=512,
        )
    if "AutoresearchMujocoSO101VialToRack-v0" not in registry:
        register(
            id="AutoresearchMujocoSO101VialToRack-v0",
            entry_point="autoresearch_gym.envs.mujoco_so101_pick_place:AutoresearchMujocoSO101VialToRackEnv",
            max_episode_steps=512,
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
    if name == "AutoresearchMujocoSO101ReachEnv":
        from autoresearch_gym.envs.mujoco_so101_reach import AutoresearchMujocoSO101ReachEnv

        return AutoresearchMujocoSO101ReachEnv
    if name == "AutoresearchMujocoSO101CubeToBinEnv":
        from autoresearch_gym.envs.mujoco_so101_pick_place import AutoresearchMujocoSO101CubeToBinEnv

        return AutoresearchMujocoSO101CubeToBinEnv
    if name == "AutoresearchMujocoSO101VialToRackEnv":
        from autoresearch_gym.envs.mujoco_so101_pick_place import AutoresearchMujocoSO101VialToRackEnv

        return AutoresearchMujocoSO101VialToRackEnv
    raise AttributeError(name)


__all__ = [
    "AutoresearchMujocoPandaPickAndPlaceEnv",
    "AutoresearchMujocoSO101CubeToBinEnv",
    "AutoresearchMujocoSO101ReachEnv",
    "AutoresearchMujocoSO101VialToRackEnv",
    "AutoresearchPandaPickAndPlaceEnv",
    "BatToGoalEnv",
    "register_envs",
]
