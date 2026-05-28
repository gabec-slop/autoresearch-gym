from __future__ import annotations

from pathlib import Path
from typing import Any


RECIPE = {
    "style": "cleanrl_mjlab_ppo",
    "algorithm": "ppo",
    "notes": (
        "External MJLab G1 motion-tracking seed. The autoresearch loop should "
        "mutate this plain recipe; the Unitree backend post-processes supported "
        "fields into MJLab TrainConfig overrides."
    ),
    "runner": {
        "num_envs": 4096,
        "eval_num_envs": 1024,
        "num_steps_per_env": 24,
        "max_iterations": None,
        "seed": 42,
        "save_interval": 500,
        "logger": "tensorboard",
        "clip_actions": None,
        "obs_groups": None,
        "run_name_suffix": "cleanrl_seed",
    },
    "actor": {
        "hidden_dims": [512, 256, 128],
        "activation": "elu",
        "obs_normalization": True,
        "init_std": 1.0,
        "std_type": "scalar",
        "rnn_type": None,
        "rnn_hidden_dim": None,
        "rnn_num_layers": None,
    },
    "critic": {
        "hidden_dims": [512, 256, 128],
        "activation": "elu",
        "obs_normalization": True,
        "rnn_type": None,
        "rnn_hidden_dim": None,
        "rnn_num_layers": None,
    },
    "ppo": {
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "learning_rate": 1.0e-3,
        "schedule": "adaptive",
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "entropy_coef": 0.005,
        "desired_kl": 0.01,
        "max_grad_norm": 1.0,
        "value_loss_coef": 1.0,
        "use_clipped_value_loss": True,
        "clip_param": 0.2,
        "normalize_advantage_per_mini_batch": False,
        "optimizer": "adam",
        "share_cnn_encoders": False,
    },
    "environment": {
        "episode_length_s": 10.0,
        "decimation": 4,
        "action_scale": 0.25,
        "enable_nan_guard": True,
        "video": False,
        "video_length": 120,
        "video_interval": 2000,
        "sim": {
            "timestep": 0.005,
            "iterations": 10,
            "ls_iterations": 20,
            "nconmax": 35,
            "njmax": 250,
        },
    },
    "motion_command": {
        "resampling_time_range": [1.0e9, 1.0e9],
        "debug_vis": True,
        "pose_range": None,
        "velocity_range": None,
        "joint_position_range": [-0.1, 0.1],
    },
    "reward_weights": {
        "motion_global_root_pos": 1.0,
        "motion_global_root_ori": 1.0,
        "motion_body_pos": 1.0,
        "motion_body_ori": 1.0,
        "motion_body_lin_vel": 1.0,
        "motion_body_ang_vel": 1.0,
        "action_rate_l2": -0.005,
        "joint_limit": -10.0,
        "self_collisions": -10.0,
    },
    "reward_params": {
        "motion_global_root_pos": {"std": 0.2},
        "motion_global_root_ori": {"std": 0.4},
        "motion_body_pos": {"std": 0.2},
        "motion_body_ori": {"std": 0.4},
        "motion_body_lin_vel": {"std": 1.0},
        "motion_body_ang_vel": {"std": 3.14},
    },
    "event_overrides": {
        "push_robot": {
            "enabled": True,
            "interval_range_s": [1.0, 3.0],
            "params": {"velocity_range": {"x": [-0.5, 0.5], "y": [-0.5, 0.5]}},
        },
        "base_com": {
            "enabled": True,
            "params": {
                "com_range": {"x": [-0.05, 0.05], "y": [-0.03, 0.03], "z": [-0.05, 0.05]}
            },
        },
        "encoder_bias": {"enabled": True, "params": {"bias_range": [-0.01, 0.01]}},
        "foot_friction": {"enabled": True, "params": {"ranges": [0.3, 1.2]}},
    },
    "termination_overrides": {
        "anchor_pos": {"enabled": True, "params": {"threshold": 0.25}},
        "anchor_ori": {"enabled": True, "params": {"threshold": 0.8}},
        "ee_body_pos": {"enabled": True, "params": {"threshold": 0.25}},
    },
}


def get_candidate() -> dict[str, Any]:
    return {
        "description": "External Unitree G1 motion mirroring seed with explicit MJLab PPO and task levers.",
        "recipe": RECIPE,
    }


class RewardRecipeWrapper:
    def __init__(self, env: Any, recipe: str | None = None) -> None:
        self.env = env
        self.recipe = recipe


def train_agent(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError("unitree_g1_motion_mirror_v0 must run through execution_backend")


def save_agent_checkpoint(agent: Any, path: Path, metadata: dict[str, Any] | None = None) -> None:
    path.write_text("unitree g1 external checkpoint placeholder\n", encoding="utf-8")
