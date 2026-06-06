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
            "SO-101 MuJoCo vial-to-rack baseline using the shared CleanRL-style SAC "
            "recipe over flattened proprioceptive, object, and goal observations. "
            "This is a cold-start RL seed; the baseline performs replay-backed "
            "actor/critic updates instead of deterministic insertion scripting."
        ),
        "recipe": {
            "algorithm": ALGORITHM,
            "reward_recipe": "task_dense",
            "control": "normalized_position_targets",
            "task": "vial_to_rack",
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
