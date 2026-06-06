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
            "SO-101 MuJoCo vial-to-rack seed with a deterministic joint-space "
            "approach-and-place baseline. It keeps the benchmark runnable while "
            "leaving room for stronger SAC+HER, teleop-imitation, or staged "
            "grasp-and-insert candidates."
        ),
        "recipe": {
            "algorithm": "so101_scripted_pick_place_baseline",
            "reward_recipe": "task_dense",
            "control": "normalized_position_targets",
            "task": "vial_to_rack",
        },
    }
