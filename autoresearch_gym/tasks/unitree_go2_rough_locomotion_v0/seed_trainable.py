from __future__ import annotations

from pathlib import Path
from typing import Any


RECIPE = {
    "style": "cleanrl_mjlab_ppo",
    "algorithm": "ppo",
    "notes": (
        "External MJLab Go2/G2 rough-terrain locomotion seed. Keep the recipe "
        "plain and mutable; the backend compiles supported values into MJLab "
        "runner, env, reward, command, event, and curriculum config."
    ),
    "runner": {
        "num_envs": 1024,
        "eval_num_envs": 256,
        "num_steps_per_env": 24,
        "max_iterations": None,
        "seed": 52,
        "save_interval": 100,
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
        "entropy_coef": 0.01,
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
        "episode_length_s": 20.0,
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
            "njmax": 1500,
        },
    },
    "terrain": {
        "max_init_terrain_level": 5,
        "terrain_extent": 2.0,
    },
    "twist_command": {
        "resampling_time_range": [3.0, 8.0],
        "rel_standing_envs": 0.05,
        "heading_command": True,
        "heading_control_stiffness": 0.5,
        "ranges": {
            "lin_vel_x": [-1.0, 2.0],
            "lin_vel_y": [-1.0, 1.0],
            "ang_vel_z": [-1.0, 1.0],
            "heading": [-3.14159, 3.14159],
        },
    },
    "reward_weights": {
        "track_linear_velocity": 1.5,
        "track_angular_velocity": 0.8,
        "body_orientation_l2": -2.0,
        "pose": -0.5,
        "body_ang_vel": -0.05,
        "angular_momentum": -0.001,
        "is_terminated": -200.0,
        "joint_acc_l2": -2.5e-7,
        "joint_pos_limits": -5.0,
        "action_rate_l2": -0.01,
        "foot_gait": 0.5,
        "foot_clearance": 0.5,
        "foot_slip": -0.1,
        "soft_landing": -0.2,
        "stand_still": -0.5,
    },
    "reward_params": {
        "track_linear_velocity": {"std": 0.25},
        "track_angular_velocity": {"std": 0.25},
        "foot_gait": {"period": 0.8, "offset": [0.0, 0.5, 0.5, 0.0], "threshold": 0.55},
        "foot_clearance": {"target_height": 0.08},
        "stand_still": {"command_threshold": 0.1},
    },
    "event_overrides": {
        "reset_base": {
            "enabled": True,
            "params": {
                "pose_range": {"x": [-0.5, 0.5], "y": [-0.5, 0.5], "yaw": [-3.14, 3.14]},
                "velocity_range": {
                    "x": [-0.5, 0.5],
                    "y": [-0.5, 0.5],
                    "z": [-0.2, 0.2],
                    "roll": [-0.5, 0.5],
                    "pitch": [-0.5, 0.5],
                    "yaw": [-0.5, 0.5],
                },
            },
        },
        "reset_robot_joints": {
            "enabled": True,
            "params": {"position_range": [0.5, 1.5], "velocity_range": [-0.5, 0.5]},
        },
        "push_robot": {
            "enabled": True,
            "interval_range_s": [5.0, 6.0],
            "params": {"velocity_range": {"x": [-0.5, 0.5], "y": [-0.5, 0.5]}},
        },
        "foot_friction": {"enabled": True, "params": {"ranges": [0.3, 1.6]}},
        "encoder_bias": {"enabled": True, "params": {"bias_range": [-0.015, 0.015]}},
        "base_com": {
            "enabled": True,
            "params": {
                "com_range": {"x": [-0.05, 0.05], "y": [-0.03, 0.03], "z": [-0.05, 0.05]}
            },
        },
    },
    "termination_overrides": {
        "fell_over": {"enabled": True, "params": {"limit_angle": 70.0}},
    },
    "curriculum_overrides": {
        "terrain_levels": {"enabled": True, "params": {}},
        "command_vel": {"enabled": True, "params": {"velocity_stages": None}},
    },
}


def get_candidate() -> dict[str, Any]:
    return {
        "description": "External Unitree Go2/G2 rough-terrain locomotion seed with explicit MJLab PPO and task levers.",
        "recipe": RECIPE,
    }


class RewardRecipeWrapper:
    def __init__(self, env: Any, recipe: str | None = None) -> None:
        self.env = env
        self.recipe = recipe


def train_agent(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError("unitree_go2_rough_locomotion_v0 must run through execution_backend")


def save_agent_checkpoint(agent: Any, path: Path, metadata: dict[str, Any] | None = None) -> None:
    path.write_text("unitree go2 external checkpoint placeholder\n", encoding="utf-8")
