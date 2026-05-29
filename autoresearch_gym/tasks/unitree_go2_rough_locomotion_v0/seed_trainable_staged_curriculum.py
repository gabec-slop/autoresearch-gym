from __future__ import annotations

from pathlib import Path
from typing import Any


RECIPE = {
    "style": "cleanrl_mjlab_ppo",
    "algorithm": "ppo",
    "notes": (
        "External MJLab Go2/G2 single-pass staged-curriculum seed. One long "
        "training run should begin with upright/simple-terrain behavior and "
        "advance toward full rough-terrain command tracking."
    ),
    "single_pass_curriculum": True,
    "curriculum_plan": [
        {
            "name": "stand_and_creep",
            "goal": "Survive full episodes on simple terrain with nearly zero or tiny forward command.",
            "target_iteration_fraction": [0.0, 0.3],
            "promotion_gate": {
                "train_episode_length_fraction_min": 0.8,
                "termination_fraction_max": 0.05,
                "upright_video_required": True,
            },
        },
        {
            "name": "flat_forward_hold",
            "goal": "Track small forward commands on simple terrain with no pushes.",
            "command": [0.2, 0.0, 0.0],
            "target_iteration_fraction": [0.3, 0.5],
            "promotion_gate": {
                "eval_done_fraction_max": 0.05,
                "eval_avg_step_reward_min": -0.15,
                "upright_video_required": True,
            },
        },
        {
            "name": "flat_command_range",
            "goal": "Add mild lateral and yaw commands while staying on easy terrain.",
            "target_iteration_fraction": [0.5, 0.7],
            "promotion_gate": {
                "eval_done_fraction_max": 0.08,
                "train_episode_length_fraction_min": 0.75,
            },
        },
        {
            "name": "simple_rough",
            "goal": "Increase terrain level while keeping command ranges narrow.",
            "target_iteration_fraction": [0.7, 0.9],
            "promotion_gate": {
                "curriculum_terrain_levels_min": 1.0,
                "eval_done_fraction_max": 0.08,
            },
        },
        {
            "name": "rough_commands",
            "goal": "Add wider velocity/yaw commands after stable locomotion exists.",
            "target_iteration_fraction": [0.9, 1.0],
        },
    ],
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
        "run_name_suffix": "staged_curriculum",
        "sample_trajectory_source": "train_context",
        "sample_rollout_frame_count": 24,
    },
    "actor": {
        "hidden_dims": [512, 256, 128],
        "activation": "elu",
        "obs_normalization": True,
        "init_std": 0.6,
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
        "entropy_coef": 0.004,
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
        "action_scale": 0.18,
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
        "resampling_time_range": [4.0, 8.0],
        "rel_standing_envs": 0.25,
        "heading_command": True,
        "heading_control_stiffness": 0.35,
        "ranges": {
            "lin_vel_x": [-0.3, 1.2],
            "lin_vel_y": [-0.4, 0.4],
            "ang_vel_z": [-0.6, 0.6],
            "heading": [-1.0, 1.0],
        },
    },
    "reward_weights": {
        "track_linear_velocity": 0.8,
        "track_angular_velocity": 0.25,
        "body_orientation_l2": -3.0,
        "pose": -0.25,
        "body_ang_vel": -0.1,
        "angular_momentum": -0.005,
        "is_terminated": -250.0,
        "joint_acc_l2": -2.5e-7,
        "joint_pos_limits": -6.0,
        "action_rate_l2": -0.025,
        "foot_gait": 0.15,
        "foot_clearance": 0.15,
        "foot_slip": -0.2,
        "soft_landing": -0.1,
        "stand_still": -0.05,
    },
    "reward_params": {
        "track_linear_velocity": {"std": 0.35},
        "track_angular_velocity": {"std": 0.4},
        "foot_gait": {"period": 0.8, "offset": [0.0, 0.5, 0.5, 0.0], "threshold": 0.55},
        "foot_clearance": {"target_height": 0.05},
        "stand_still": {"command_threshold": 0.12},
    },
    "event_overrides": {
        "reset_base": {
            "enabled": True,
            "params": {
                "pose_range": {"x": [-0.05, 0.05], "y": [-0.05, 0.05], "yaw": [-0.15, 0.15]},
                "velocity_range": {
                    "x": [-0.05, 0.05],
                    "y": [-0.05, 0.05],
                    "z": [-0.02, 0.02],
                    "roll": [-0.05, 0.05],
                    "pitch": [-0.05, 0.05],
                    "yaw": [-0.05, 0.05],
                },
            },
        },
        "reset_robot_joints": {
            "enabled": True,
            "params": {"position_range": [0.9, 1.1], "velocity_range": [-0.05, 0.05]},
        },
        "push_robot": {"enabled": False},
        "foot_friction": {"enabled": True, "params": {"ranges": [0.8, 1.2]}},
        "encoder_bias": {"enabled": False},
        "base_com": {"enabled": False},
    },
    "termination_overrides": {
        "fell_over": {"enabled": True, "params": {"limit_angle": 45.0}},
    },
    "curriculum_overrides": {
        "terrain_levels": {"enabled": True, "params": {}},
        "command_vel": {
            "enabled": True,
            "params": {
                "velocity_stages": [
                    {
                        "step": -1,
                        "lin_vel_x": [0.0, 0.15],
                        "lin_vel_y": [0.0, 0.0],
                        "ang_vel_z": [0.0, 0.0],
                    },
                    {
                        "step": 7200,
                        "lin_vel_x": [0.05, 0.35],
                        "lin_vel_y": [-0.04, 0.04],
                        "ang_vel_z": [-0.08, 0.08],
                    },
                    {
                        "step": 12000,
                        "lin_vel_x": [0.0, 0.55],
                        "lin_vel_y": [-0.1, 0.1],
                        "ang_vel_z": [-0.2, 0.2],
                    },
                    {
                        "step": 16800,
                        "lin_vel_x": [0.0, 0.75],
                        "lin_vel_y": [-0.2, 0.2],
                        "ang_vel_z": [-0.3, 0.3],
                    },
                    {
                        "step": 21600,
                        "lin_vel_x": [-0.3, 1.2],
                        "lin_vel_y": [-0.4, 0.4],
                        "ang_vel_z": [-0.6, 0.6],
                    },
                ]
            },
        },
    },
}


def get_candidate() -> dict[str, Any]:
    return {
        "description": (
            "External Unitree Go2/G2 single-pass staged curriculum: train through "
            "upright survival, flat forward locomotion, simple rough terrain, and "
            "rough command tracking in one long MJLab run."
        ),
        "recipe": RECIPE,
    }


class RewardRecipeWrapper:
    def __init__(self, env: Any, recipe: str | None = None) -> None:
        self.env = env
        self.recipe = recipe


def train_agent(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError("unitree_go2_rough_locomotion_v0 must run through execution_backend")


def save_agent_checkpoint(agent: Any, path: Path, metadata: dict[str, Any] | None = None) -> None:
    path.write_text("unitree go2 staged curriculum external checkpoint placeholder\n", encoding="utf-8")
