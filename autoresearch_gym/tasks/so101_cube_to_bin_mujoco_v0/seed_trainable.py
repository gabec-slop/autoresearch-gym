from __future__ import annotations

from typing import Any

from autoresearch_gym.tasks.so101_scripted_pick_place_seed import (
    Agent,
    RewardRecipeWrapper,
    load_agent_checkpoint,
    save_agent_checkpoint,
    train_agent,
)


def get_candidate() -> dict[str, Any]:
    return {
        "description": (
            "SO-101 MuJoCo cube-in-bin seed with a deterministic joint-space "
            "approach-and-push baseline. It is intended as a runnable control "
            "surface for manipulation research; stronger candidates should add "
            "SAC+HER, demonstrations, or IK-assisted grasp staging."
        ),
        "recipe": {
            "algorithm": "so101_scripted_pick_place_baseline",
            "reward_recipe": "task_dense",
            "control": "normalized_position_targets",
            "task": "cube_to_bin",
        },
    }
