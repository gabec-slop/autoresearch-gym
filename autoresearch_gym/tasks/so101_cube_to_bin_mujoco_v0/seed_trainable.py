from __future__ import annotations

from typing import Any

from autoresearch_gym.tasks.so101_reach_mujoco_v0.seed_trainable import (
    ALGORITHM,
    Agent,
    ReplayBuffer,
    RewardRecipeWrapper,
    load_agent_checkpoint,
    save_agent_checkpoint,
    train_agent,
)


def get_candidate() -> dict[str, Any]:
    return {
        "description": (
            "SO-101 MuJoCo cube-to-bin baseline using the shared CleanRL-style SAC "
            "recipe over flattened proprioceptive, object, and goal observations. "
            "This is a cold-start RL seed; task progress comes from replay-backed "
            "actor/critic updates, not a scripted pick-place controller."
        ),
        "recipe": {
            "algorithm": ALGORITHM,
            "reward_recipe": "task_dense",
            "control": "normalized_position_targets",
            "task": "cube_to_bin",
        },
    }


__all__ = [
    "ALGORITHM",
    "Agent",
    "ReplayBuffer",
    "RewardRecipeWrapper",
    "get_candidate",
    "load_agent_checkpoint",
    "save_agent_checkpoint",
    "train_agent",
]
