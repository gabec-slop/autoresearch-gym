from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import spaces
from torch.distributions import Normal

from autoresearch_gym.runner.curves import elapsed_seconds_since, make_train_collection_window_record, scalar_info_metrics


# Harness invariants:
# - Fixed eval uses the raw MuJoCo Menagerie Panda task.
# - This file owns the mutable training recipe: subskill curriculum, vector
#   collection, PPO loss, entropy, and diagnostic logging.
# - When env_kwargs.backend starts with "mujoco_warp", train_agent attempts a
#   batched MuJoCo Warp collector through env.unwrapped.make_vectorized().

EXP_NAME = "panda_pick_and_place_mjwarp_curriculum_ppo"
ALGORITHM = "ppo"
CONTROL_TYPE = None
REWARD_RECIPE = "subskill_curriculum"
RECIPE = {
    "algorithm": ALGORITHM,
    "reward_recipe": REWARD_RECIPE,
    "runner": {
        "sample_trajectory_source": "candidate_provided",
    },
}
DIAGNOSTIC_SERIES = {
    "series": [
        {"key": "ee_to_cube_distance", "label": "EE to Cube", "color": "#2563eb", "source": "info_metrics", "chart": "normalized_line", "group": "subskills"},
        {"key": "ee_to_cube_progress", "label": "EE Progress", "color": "#0891b2", "source": "info_metrics", "chart": "normalized_line", "group": "subskills"},
        {"key": "cube_lift_height", "label": "Lift", "color": "#16a34a", "source": "info_metrics", "chart": "normalized_line", "group": "subskills"},
        {"key": "lifted_ever_rate", "label": "Lifted Ever", "color": "#15803d", "source": "info_metrics", "chart": "normalized_line", "group": "subskills"},
        {"key": "cube_to_goal_distance", "label": "Cube to Goal", "color": "#dc2626", "source": "info_metrics", "chart": "normalized_line", "group": "subskills"},
        {"key": "cube_at_goal_rate", "label": "At Goal", "color": "#f97316", "source": "info_metrics", "chart": "normalized_line", "group": "subskills"},
        {"key": "placed_success_rate", "label": "Placed", "color": "#7c3aed", "source": "info_metrics", "chart": "normalized_line", "group": "subskills"},
    ]
}

NUM_ENVS = 16
NUM_STEPS = 32
LEARNING_RATE = 3.0e-4
GAMMA = 0.98
GAE_LAMBDA = 0.95
UPDATE_EPOCHS = 3
NUM_MINIBATCHES = 4
CLIP_COEF = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 1.0
HIDDEN_SIZE = 192
ACTION_STD_INIT = 0.55

APPROACH_THRESHOLD = 0.065
LIFT_THRESHOLD = 0.055
PLACE_THRESHOLD = 0.05
APPROACH_STEPS = 20_000
GRASP_LIFT_STEPS = 70_000


def get_candidate() -> dict[str, Any]:
    return {
        "description": (
            "MuJoCo Menagerie Panda pick-and-place PPO seed with train-only subskill "
            "curriculum. It scaffolds from end-effector proximity to cube, to closed "
            "gripper/contact and lift, to cube-goal proximity, and finally successful "
            "place. If MuJoCo Warp is available and requested, collection uses a "
            "batched MJWarp vector environment while fixed eval remains the raw task."
        ),
        "recipe": RECIPE,
    }


def _diagnostic_series() -> dict[str, Any]:
    return DIAGNOSTIC_SERIES


def flatten_observation(obs: Any) -> np.ndarray:
    if isinstance(obs, dict):
        pieces = []
        for key in ("observation", "achieved_goal", "desired_goal"):
            if key in obs:
                pieces.append(np.asarray(obs[key], dtype=np.float32).reshape(-1))
        return np.concatenate(pieces).astype(np.float32, copy=False)
    return np.asarray(obs, dtype=np.float32).reshape(-1)


def flatten_observation_space(space: spaces.Space[Any]) -> spaces.Box:
    if not isinstance(space, spaces.Dict):
        assert isinstance(space, spaces.Box)
        return spaces.Box(
            low=np.asarray(space.low, dtype=np.float32).reshape(-1),
            high=np.asarray(space.high, dtype=np.float32).reshape(-1),
            dtype=np.float32,
        )
    lows = []
    highs = []
    for key in ("observation", "achieved_goal", "desired_goal"):
        subspace = space.spaces[key]
        assert isinstance(subspace, spaces.Box)
        lows.append(np.asarray(subspace.low, dtype=np.float32).reshape(-1))
        highs.append(np.asarray(subspace.high, dtype=np.float32).reshape(-1))
    return spaces.Box(
        low=np.concatenate(lows).astype(np.float32, copy=False),
        high=np.concatenate(highs).astype(np.float32, copy=False),
        dtype=np.float32,
    )


def _phase(global_step: int) -> str:
    if global_step < APPROACH_STEPS:
        return "approach"
    if global_step < GRASP_LIFT_STEPS:
        return "grasp_lift"
    return "place"


def _safe_progress_fraction(progress: float, initial_distance: float) -> float:
    if initial_distance <= 1e-6:
        return 0.0
    return float(np.clip(progress / initial_distance, -1.0, 1.0))


def _curriculum_reward(raw_reward: float, info: dict[str, Any], global_step: int) -> float:
    del raw_reward
    ee_to_cube = float(info.get("ee_to_cube_distance", 1.0))
    cube_to_goal = float(info.get("cube_to_goal_distance", 1.0))
    ee_progress = float(info.get("ee_to_cube_progress", 0.0))
    goal_progress = float(info.get("cube_to_goal_progress", 0.0))
    initial_ee = float(info.get("initial_ee_to_cube_distance", 0.0))
    initial_goal = float(info.get("initial_cube_to_goal_distance", 0.0))
    lift = float(info.get("cube_lift_height", 0.0))
    near = bool(info.get("near_cube", False))
    grasp = bool(info.get("gripper_closed_near_cube", False))
    lifted = bool(info.get("lifted", False))
    lifted_ever = bool(info.get("lifted_ever", lifted))
    placed = bool(info.get("placed_success", False))
    ee_progress_frac = _safe_progress_fraction(ee_progress, initial_ee)
    goal_progress_frac = _safe_progress_fraction(goal_progress, initial_goal)
    phase = _phase(global_step)
    shaped = 0.0
    if phase == "approach":
        shaped += 1.25 * ee_progress_frac
        shaped += 0.20 * math.exp(-18.0 * ee_to_cube)
        shaped += 0.35 if near else 0.0
    elif phase == "grasp_lift":
        shaped += 0.55 * ee_progress_frac
        shaped += 0.15 * math.exp(-16.0 * ee_to_cube)
        shaped += 0.25 if near else 0.0
        shaped += 0.35 if grasp else 0.0
        shaped += 2.25 * min(1.0, lift / max(LIFT_THRESHOLD, 1e-6))
        shaped += 0.70 if lifted else 0.0
    else:
        shaped += 0.25 if near else 0.0
        shaped += 0.60 if lifted_ever else 0.0
        shaped += 1.50 * goal_progress_frac if lifted_ever else 0.0
        shaped += 0.20 * math.exp(-14.0 * cube_to_goal) if lifted_ever else 0.0
        shaped += 3.00 if placed else 0.0
    return float(np.clip(shaped, -2.0, 5.0))


def _safe_progress_fraction_vector(progress: np.ndarray, initial_distance: np.ndarray) -> np.ndarray:
    progress = np.asarray(progress, dtype=np.float32)
    initial_distance = np.asarray(initial_distance, dtype=np.float32)
    return np.divide(
        progress,
        initial_distance,
        out=np.zeros_like(progress, dtype=np.float32),
        where=initial_distance > 1e-6,
    ).clip(-1.0, 1.0)


def _curriculum_reward_vector(raw: np.ndarray, infos: dict[str, Any], global_step: int) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float32)
    ee = np.asarray(infos.get("ee_to_cube_distance", np.ones_like(raw)), dtype=np.float32)
    cube_goal = np.asarray(infos.get("cube_to_goal_distance", np.ones_like(raw)), dtype=np.float32)
    ee_progress = np.asarray(infos.get("ee_to_cube_progress", np.zeros_like(raw)), dtype=np.float32)
    goal_progress = np.asarray(infos.get("cube_to_goal_progress", np.zeros_like(raw)), dtype=np.float32)
    initial_ee = np.asarray(infos.get("initial_ee_to_cube_distance", np.zeros_like(raw)), dtype=np.float32)
    initial_goal = np.asarray(infos.get("initial_cube_to_goal_distance", np.zeros_like(raw)), dtype=np.float32)
    lift = np.asarray(infos.get("cube_lift_height", np.zeros_like(raw)), dtype=np.float32)
    near = np.asarray(infos.get("near_cube", np.zeros_like(raw, dtype=bool)), dtype=bool)
    grasp = np.asarray(infos.get("gripper_closed_near_cube", np.zeros_like(raw, dtype=bool)), dtype=bool)
    lifted = np.asarray(infos.get("lifted", np.zeros_like(raw, dtype=bool)), dtype=bool)
    lifted_ever = np.asarray(infos.get("lifted_ever", lifted), dtype=bool)
    placed = np.asarray(infos.get("placed_success", np.zeros_like(raw, dtype=bool)), dtype=bool)
    ee_progress_frac = _safe_progress_fraction_vector(ee_progress, initial_ee)
    goal_progress_frac = _safe_progress_fraction_vector(goal_progress, initial_goal)
    shaped = np.zeros_like(raw, dtype=np.float32)
    phase = _phase(global_step)
    if phase == "approach":
        shaped += 1.25 * ee_progress_frac
        shaped += 0.20 * np.exp(-18.0 * ee)
        shaped += 0.35 * near.astype(np.float32)
    elif phase == "grasp_lift":
        shaped += 0.55 * ee_progress_frac
        shaped += 0.15 * np.exp(-16.0 * ee)
        shaped += 0.25 * near.astype(np.float32)
        shaped += 0.35 * grasp.astype(np.float32)
        shaped += 2.25 * np.minimum(1.0, lift / max(LIFT_THRESHOLD, 1e-6))
        shaped += 0.70 * lifted.astype(np.float32)
    else:
        shaped += 0.25 * near.astype(np.float32)
        lifted_gate = lifted_ever.astype(np.float32)
        shaped += 0.60 * lifted_gate
        shaped += 1.50 * goal_progress_frac * lifted_gate
        shaped += 0.20 * np.exp(-14.0 * cube_goal) * lifted_gate
        shaped += 3.00 * placed.astype(np.float32)
    return np.clip(shaped, -2.0, 5.0).astype(np.float32)


class RewardRecipeWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env[Any, Any], recipe: str | None = None) -> None:
        super().__init__(env)
        self.recipe = recipe or REWARD_RECIPE
        if self.recipe not in {"task_dense", "subskill_curriculum"}:
            raise ValueError(f"Unknown MuJoCo Panda reward recipe: {self.recipe}")
        self.observation_space = flatten_observation_space(env.observation_space)
        self.global_step = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        return flatten_observation(obs), info

    def step(self, action: np.ndarray):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        raw_reward = float(reward)
        if self.recipe == "subskill_curriculum":
            reward = _curriculum_reward(raw_reward, info, self.global_step)
            info["task_reward"] = raw_reward
            info["training_reward"] = float(reward)
            info["curriculum_phase_index"] = float({"approach": 0, "grasp_lift": 1, "place": 2}[_phase(self.global_step)])
        else:
            info["training_reward"] = raw_reward
        self.global_step += 1
        return flatten_observation(obs), float(reward), terminated, truncated, info


def layer_init(layer: nn.Linear, std: float = math.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, HIDDEN_SIZE)),
            nn.Tanh(),
            layer_init(nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)),
            nn.Tanh(),
            layer_init(nn.Linear(HIDDEN_SIZE, action_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, action_dim) * math.log(ACTION_STD_INIT))
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, HIDDEN_SIZE)),
            nn.Tanh(),
            layer_init(nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)),
            nn.Tanh(),
            layer_init(nn.Linear(HIDDEN_SIZE, 1), std=1.0),
        )

    def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor | None = None):
        mean = self.actor_mean(obs)
        logstd = self.actor_logstd.expand_as(mean)
        probs = Normal(mean, logstd.exp())
        if action is None:
            action = probs.rsample()
        clipped_action = torch.clamp(action, -1.0, 1.0)
        return clipped_action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(obs).squeeze(1)

    def act(self, obs: Any, deterministic: bool = True) -> np.ndarray:
        device = next(self.parameters()).device
        obs_tensor = torch.as_tensor(flatten_observation(obs), dtype=torch.float32, device=device).reshape(1, -1)
        with torch.no_grad():
            action = self.actor_mean(obs_tensor) if deterministic else self.get_action_and_value(obs_tensor)[0]
        return torch.clamp(action, -1.0, 1.0).cpu().numpy()[0]


def save_agent_checkpoint(agent: Agent, checkpoint_path: str | Path, metadata: dict[str, Any] | None = None) -> None:
    torch.save({"agent_state": agent.state_dict(), "metadata": metadata or {}, "algorithm": ALGORITHM}, checkpoint_path)


def load_agent_checkpoint(agent: Agent, checkpoint_path: str | Path) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    agent.load_state_dict(payload["agent_state"])
    return payload.get("metadata", {})


def _to_device(device: Any) -> torch.device:
    if isinstance(device, torch.device):
        return device
    requested = str(device)
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    return torch.device("cpu")


def _live_callback(callback: Any | None, **kwargs: Any) -> dict[str, Any]:
    if callback is None:
        return {}
    try:
        result = callback(**kwargs)
        return result if isinstance(result, dict) else {}
    except TypeError as exc:
        if "unexpected keyword argument" not in str(exc):
            raise
        fallback = {
            key: kwargs[key]
            for key in (
                "status",
                "episode_records",
                "total_steps",
                "last_metrics",
                "current_episode",
                "episode_return",
                "episode_length",
            )
            if key in kwargs
        }
        try:
            result = callback(**fallback)
            return result if isinstance(result, dict) else {}
        except TypeError as fallback_exc:
            if "unexpected keyword argument" not in str(fallback_exc):
                raise
            minimal = {
                key: kwargs[key]
                for key in ("status", "episode_records", "total_steps", "last_metrics")
                if key in kwargs
            }
            try:
                result = callback(**minimal)
                return result if isinstance(result, dict) else {}
            except TypeError as minimal_exc:
                if "unexpected keyword argument" not in str(minimal_exc):
                    raise
                return {}


def _render_policy_frame(env: gym.Env[Any, Any]) -> np.ndarray | None:
    render_env = getattr(env, "unwrapped", env)
    try:
        frame = render_env.render(width=720, height=480)
    except TypeError:
        try:
            frame = render_env.render()
        except Exception:
            return None
    except Exception:
        return None
    if frame is None:
        return None
    return np.asarray(frame, dtype=np.uint8)


def _sample_policy_trajectory(
    agent: Agent,
    env_factory: Any,
    benchmark: Any,
    request: dict[str, Any],
) -> dict[str, Any]:
    env = env_factory(CONTROL_TYPE, REWARD_RECIPE)
    frames: list[np.ndarray] = []
    episode = int(request.get("episode") or 0)
    sample_index = int(request.get("sample_index") or 0)
    stride = max(1, int(request.get("frame_stride") or 2))
    max_steps = int(benchmark.max_steps)
    seed = int(getattr(benchmark, "eval_seed_start", getattr(benchmark, "train_seed", 0))) + max(sample_index - 1, 0)
    try:
        obs, _ = env.reset(seed=seed)
        for step in range(max_steps + 1):
            if step % stride == 0:
                frame = _render_policy_frame(env)
                if frame is not None:
                    frames.append(frame)
            if step >= max_steps:
                break
            action = agent.act(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            if bool(terminated or truncated):
                frame = _render_policy_frame(env)
                if frame is not None:
                    frames.append(frame)
                break
    finally:
        try:
            env.close()
        except Exception:
            pass
    return {
        "episode": episode,
        "sample_index": sample_index,
        "source": RECIPE["runner"]["sample_trajectory_source"],
        "frames": frames,
        "playback_fps": float(request.get("playback_fps") or 20.0),
        "frame_stride": stride,
        "metadata": {"seed": seed, "max_steps": max_steps},
    }


def _answer_sampled_trajectory_request(
    response: dict[str, Any],
    *,
    live_callback: Any | None,
    agent: Agent,
    env_factory: Any,
    benchmark: Any,
    records: list[dict[str, Any]],
    global_step: int,
    last_metrics: dict[str, float] | None,
) -> None:
    request = response.get("sampled_trajectory_request") if isinstance(response, dict) else None
    if not isinstance(request, dict) or not request.get("requested"):
        return
    sampled_trajectory = _sample_policy_trajectory(agent, env_factory, benchmark, request)
    _live_callback(
        live_callback,
        status="running",
        episode_records=records,
        total_steps=global_step,
        last_metrics=last_metrics,
        agent=agent,
        elapsed_seconds=0.0,
        current_episode=int(request.get("episode") or len(records)),
        sampled_trajectory=sampled_trajectory,
        diagnostic_series=_diagnostic_series(),
    )


def _mean_info(infos: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in infos.items():
        if key == "physics_backend":
            continue
        arr = np.asarray(value)
        if arr.dtype == bool:
            out[f"{key}_rate"] = float(np.mean(arr.astype(np.float32)))
        elif np.issubdtype(arr.dtype, np.number):
            out[key] = float(np.mean(arr.astype(np.float32)))
    return out


def train_agent(
    benchmark: Any,
    env_factory: Any,
    candidate: Any,
    device: Any,
    init_checkpoint: str | Path | None = None,
    live_callback: Any | None = None,
) -> tuple[Agent, dict[str, Any]]:
    del candidate
    random.seed(benchmark.train_seed)
    np.random.seed(benchmark.train_seed)
    torch.manual_seed(benchmark.train_seed)
    torch_device = _to_device(device)
    num_envs = int(benchmark.env_kwargs.get("num_envs", NUM_ENVS))
    num_steps = int(benchmark.env_kwargs.get("steps_per_env_per_iteration", NUM_STEPS))
    budget_seconds = getattr(benchmark, "train_seconds", None)
    updates = max(1, int(benchmark.train_episodes))
    if budget_seconds is not None:
        updates = 1_000_000

    vector_env = _try_make_vector_env(benchmark, env_factory, num_envs)
    if vector_env is not None:
        return _train_vectorized(
            benchmark,
            vector_env,
            env_factory,
            torch_device,
            num_steps,
            updates,
            budget_seconds,
            init_checkpoint,
            live_callback,
        )
    return _train_subproc_fallback(benchmark, env_factory, torch_device, num_envs, num_steps, updates, budget_seconds, init_checkpoint, live_callback)


def _try_make_vector_env(benchmark: Any, env_factory: Any, num_envs: int):
    if num_envs <= 1 or not str(benchmark.env_kwargs.get("backend", "")).startswith("mujoco_warp"):
        return None
    probe_env = env_factory(CONTROL_TYPE, REWARD_RECIPE)
    try:
        base_env = probe_env.unwrapped if hasattr(probe_env, "unwrapped") else probe_env
        make_vectorized = getattr(base_env, "make_vectorized", None)
        if make_vectorized is None:
            return None
        return make_vectorized(num_envs, int(benchmark.train_seed))
    finally:
        try:
            probe_env.close()
        except Exception:
            pass


def _train_subproc_fallback(
    benchmark: Any,
    env_factory: Any,
    device: torch.device,
    num_envs: int,
    num_steps: int,
    updates: int,
    budget_seconds: float | None,
    init_checkpoint: str | Path | None,
    live_callback: Any | None,
) -> tuple[Agent, dict[str, Any]]:
    envs = [env_factory(CONTROL_TYPE, REWARD_RECIPE) for _ in range(num_envs)]
    obs = np.stack([env.reset(seed=int(benchmark.train_seed) + idx)[0] for idx, env in enumerate(envs)]).astype(np.float32)
    agent = Agent(int(obs.shape[1]), int(envs[0].action_space.shape[0])).to(device)
    resumed_from = load_agent_checkpoint(agent, init_checkpoint) if init_checkpoint is not None else None
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)
    records: list[dict[str, Any]] = []
    global_step = 0
    completed = 0
    start_time = time.time()
    last_metrics: dict[str, float] | None = None
    episode_returns = np.zeros(num_envs, dtype=np.float32)
    episode_lengths = np.zeros(num_envs, dtype=np.float32)
    latest_infos: dict[str, Any] = {}
    for update in range(updates):
        if budget_seconds is not None and elapsed_seconds_since(start_time) >= budget_seconds:
            break
        obs, global_step, completed, last_metrics, latest_infos = _ppo_update(
            agent,
            optimizer,
            obs,
            lambda actions: _step_env_list(envs, actions),
            device,
            num_steps,
            global_step,
            completed,
            episode_returns,
            episode_lengths,
            records,
            start_time,
            recipe_is_vector=False,
        )
        response = _live_callback(
            live_callback,
            status="running",
            episode_records=records,
            total_steps=global_step,
            last_metrics=_with_gradient_update_metrics(last_metrics, update + 1),
            agent=agent,
            env=envs[0] if envs else None,
            elapsed_seconds=elapsed_seconds_since(start_time),
            current_episode=len(records),
            diagnostic_series=_diagnostic_series(),
        )
        _answer_sampled_trajectory_request(
            response,
            live_callback=live_callback,
            agent=agent,
            env_factory=env_factory,
            benchmark=benchmark,
            records=records,
            global_step=global_step,
            last_metrics=_with_gradient_update_metrics(last_metrics, update + 1),
        )
    for env in envs:
        env.close()
    return agent, _summary(benchmark, records, global_step, completed, start_time, last_metrics, latest_infos, "python_mujoco_envs", init_checkpoint, resumed_from)


def _train_vectorized(
    benchmark: Any,
    vector_env: Any,
    env_factory: Any,
    device: torch.device,
    num_steps: int,
    updates: int,
    budget_seconds: float | None,
    init_checkpoint: str | Path | None,
    live_callback: Any | None,
) -> tuple[Agent, dict[str, Any]]:
    obs = vector_env.reset(seed=int(benchmark.train_seed)).astype(np.float32)
    agent = Agent(int(obs.shape[1]), int(vector_env.action_space.shape[0])).to(device)
    resumed_from = load_agent_checkpoint(agent, init_checkpoint) if init_checkpoint is not None else None
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)
    records: list[dict[str, Any]] = []
    global_step = 0
    completed = 0
    start_time = time.time()
    last_metrics: dict[str, float] | None = None
    episode_returns = np.zeros(vector_env.num_envs, dtype=np.float32)
    episode_lengths = np.zeros(vector_env.num_envs, dtype=np.float32)
    latest_infos: dict[str, Any] = {}
    for update in range(updates):
        if budget_seconds is not None and elapsed_seconds_since(start_time) >= budget_seconds:
            break
        obs, global_step, completed, last_metrics, latest_infos = _ppo_update(
            agent,
            optimizer,
            obs,
            lambda actions: _step_vector_env(vector_env, actions, global_step),
            device,
            num_steps,
            global_step,
            completed,
            episode_returns,
            episode_lengths,
            records,
            start_time,
            recipe_is_vector=True,
        )
        response = _live_callback(
            live_callback,
            status="running",
            episode_records=records,
            total_steps=global_step,
            last_metrics=_with_gradient_update_metrics(last_metrics, update + 1),
            agent=agent,
            elapsed_seconds=elapsed_seconds_since(start_time),
            current_episode=len(records),
            diagnostic_series=_diagnostic_series(),
        )
        _answer_sampled_trajectory_request(
            response,
            live_callback=live_callback,
            agent=agent,
            env_factory=env_factory,
            benchmark=benchmark,
            records=records,
            global_step=global_step,
            last_metrics=_with_gradient_update_metrics(last_metrics, update + 1),
        )
    vector_env.close()
    return agent, _summary(benchmark, records, global_step, completed, start_time, last_metrics, latest_infos, "mujoco_warp_vectorized", init_checkpoint, resumed_from)


def _step_env_list(envs: list[gym.Env[Any, Any]], actions: np.ndarray):
    next_obs = []
    rewards = np.zeros(len(envs), dtype=np.float32)
    dones = np.zeros(len(envs), dtype=bool)
    info_values: dict[str, list[Any]] = {}
    for idx, env in enumerate(envs):
        obs, reward, terminated, truncated, info = env.step(actions[idx])
        done = bool(terminated or truncated)
        if done:
            obs, reset_info = env.reset()
            info = {**dict(info), **{f"reset_{k}": v for k, v in reset_info.items()}}
        next_obs.append(obs)
        rewards[idx] = float(reward)
        dones[idx] = done
        for key, value in dict(info).items():
            info_values.setdefault(key, []).append(value)
    return np.stack(next_obs).astype(np.float32), rewards, dones, {k: np.asarray(v) for k, v in info_values.items()}


def _step_vector_env(vector_env: Any, actions: np.ndarray, global_step: int):
    obs, raw_rewards, dones, infos = vector_env.step(actions)
    rewards = _curriculum_reward_vector(raw_rewards, infos, global_step)
    if np.any(dones):
        obs = vector_env.reset_worlds(dones)
    infos = dict(infos)
    infos["task_reward"] = raw_rewards
    infos["training_reward"] = rewards
    infos["curriculum_phase_index"] = np.full_like(raw_rewards, {"approach": 0, "grasp_lift": 1, "place": 2}[_phase(global_step)], dtype=np.float32)
    return obs.astype(np.float32), rewards, dones, infos


def _ppo_update(
    agent: Agent,
    optimizer: optim.Optimizer,
    obs: np.ndarray,
    step_fn: Any,
    device: torch.device,
    num_steps: int,
    global_step: int,
    completed: int,
    episode_returns: np.ndarray,
    episode_lengths: np.ndarray,
    records: list[dict[str, Any]],
    start_time: float,
    recipe_is_vector: bool,
):
    del recipe_is_vector
    num_envs, obs_dim = obs.shape
    action_dim = int(agent.actor_logstd.shape[1])
    obs_buf = torch.zeros((num_steps, num_envs, obs_dim), device=device)
    actions_buf = torch.zeros((num_steps, num_envs, action_dim), device=device)
    logprobs_buf = torch.zeros((num_steps, num_envs), device=device)
    rewards_buf = torch.zeros((num_steps, num_envs), device=device)
    dones_buf = torch.zeros((num_steps, num_envs), device=device)
    values_buf = torch.zeros((num_steps, num_envs), device=device)
    latest_infos: dict[str, Any] = {}
    for step in range(num_steps):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(obs_tensor)
        action_np = action.cpu().numpy()
        next_obs, rewards, dones, infos = step_fn(action_np)
        obs_buf[step] = obs_tensor
        actions_buf[step] = action
        logprobs_buf[step] = logprob
        rewards_buf[step] = torch.as_tensor(rewards, dtype=torch.float32, device=device)
        dones_buf[step] = torch.as_tensor(dones.astype(np.float32), dtype=torch.float32, device=device)
        values_buf[step] = value
        episode_returns += rewards
        episode_lengths += 1.0
        completed_now = int(np.sum(dones))
        if completed_now:
            completed += completed_now
            episode_returns[dones] = 0.0
            episode_lengths[dones] = 0.0
        global_step += num_envs
        obs = next_obs
        latest_infos = infos
    with torch.no_grad():
        next_value = agent.critic(torch.as_tensor(obs, dtype=torch.float32, device=device)).squeeze(1)
        advantages = torch.zeros_like(rewards_buf)
        lastgaelam = torch.zeros(num_envs, device=device)
        for t in reversed(range(num_steps)):
            next_nonterminal = 1.0 - dones_buf[t]
            next_values = next_value if t == num_steps - 1 else values_buf[t + 1]
            delta = rewards_buf[t] + GAMMA * next_values * next_nonterminal - values_buf[t]
            advantages[t] = lastgaelam = delta + GAMMA * GAE_LAMBDA * next_nonterminal * lastgaelam
        returns = advantages + values_buf
    b_obs = obs_buf.reshape((-1, obs_dim))
    b_actions = actions_buf.reshape((-1, action_dim))
    b_logprobs = logprobs_buf.reshape(-1)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = values_buf.reshape(-1)
    batch_size = b_obs.shape[0]
    minibatch_size = max(1, batch_size // NUM_MINIBATCHES)
    inds = np.arange(batch_size)
    last_metrics: dict[str, float] = {}
    for _ in range(UPDATE_EPOCHS):
        np.random.shuffle(inds)
        for start in range(0, batch_size, minibatch_size):
            mb_inds = inds[start : start + minibatch_size]
            _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
            logratio = newlogprob - b_logprobs[mb_inds]
            ratio = logratio.exp()
            mb_adv = b_advantages[mb_inds]
            mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std(unbiased=False) + 1e-8)
            pg_loss1 = -mb_adv * ratio
            pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - CLIP_COEF, 1.0 + CLIP_COEF)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
            v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
            entropy_loss = entropy.mean()
            loss = pg_loss - ENT_COEF * entropy_loss + VF_COEF * v_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            last_metrics = {
                "policy_loss": float(pg_loss.item()),
                "value_loss": float(v_loss.item()),
                "entropy": float(entropy_loss.item()),
                "approx_kl": float(((ratio - 1.0) - logratio).mean().item()),
            }
    info_metrics = _mean_info(latest_infos)
    records.append(
        make_train_collection_window_record(
            episode=len(records) + 1,
            return_value=float(rewards_buf.sum(dim=0).mean().item()),
            length=float(num_steps),
            episodes_in_window=max(1, int(np.sum(dones_buf.cpu().numpy()))),
            success=bool(float(info_metrics.get("is_success_rate", 0.0)) > 0.0),
            step=global_step,
            env_steps_in_window=num_steps * num_envs,
            elapsed_seconds=elapsed_seconds_since(start_time),
            info_metrics=info_metrics,
        )
    )
    return obs, global_step, completed, last_metrics, latest_infos


def _gradient_update_count(update_count: int) -> int:
    return max(0, int(update_count)) * UPDATE_EPOCHS * NUM_MINIBATCHES


def _with_gradient_update_metrics(last_metrics: dict[str, float] | None, update_count: int) -> dict[str, float]:
    metrics = dict(last_metrics or {})
    metrics["gradient_updates"] = float(_gradient_update_count(update_count))
    return metrics


def _summary(
    benchmark: Any,
    records: list[dict[str, Any]],
    global_step: int,
    completed: int,
    start_time: float,
    last_metrics: dict[str, float] | None,
    latest_infos: dict[str, Any],
    backend: str,
    init_checkpoint: str | Path | None,
    resumed_from: dict[str, Any] | None,
) -> dict[str, Any]:
    returns = [float(record["return"]) for record in records]
    successes = [1.0 if record.get("success") else 0.0 for record in records]
    wall_clock = elapsed_seconds_since(start_time)
    budget_seconds = getattr(benchmark, "train_seconds", None)
    stop_reason = "time_budget_exhausted" if budget_seconds is not None and wall_clock >= float(budget_seconds) else "episode_cap_reached"
    gradient_updates = _gradient_update_count(len(records))
    num_envs = int(getattr(benchmark, "env_kwargs", {}).get("num_envs", 1) or 1)
    return {
        "episodes": int(benchmark.train_episodes),
        "episodes_completed": int(completed),
        "time_budget_seconds": float(budget_seconds) if budget_seconds is not None else None,
        "stop_reason": stop_reason,
        "total_steps": int(global_step),
        "env_steps": int(global_step),
        "num_envs": num_envs,
        "vector_envs": num_envs,
        "completed_episodes": int(completed),
        "episode_batches": len(records),
        "gradient_updates": gradient_updates,
        "avg_return": float(np.mean(returns)) if returns else 0.0,
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "avg_length": float(np.mean([record["length"] for record in records])) if records else 0.0,
        "last_metrics": {
            **(last_metrics or {}),
            **_mean_info(latest_infos),
            "gradient_updates": float(gradient_updates),
            "num_envs": float(num_envs),
        },
        "episode_records": records,
        "wall_clock_seconds": wall_clock,
        "init_checkpoint": str(init_checkpoint) if init_checkpoint is not None else None,
        "resumed_from": resumed_from,
        "curriculum": {
            "type": REWARD_RECIPE,
            "approach_steps": APPROACH_STEPS,
            "grasp_lift_steps": GRASP_LIFT_STEPS,
            "approach_threshold": APPROACH_THRESHOLD,
            "lift_threshold": LIFT_THRESHOLD,
            "place_threshold": PLACE_THRESHOLD,
        },
        "diagnostic_series": _diagnostic_series(),
        "physics_backend": backend,
        "vectorized_backend": backend,
    }
