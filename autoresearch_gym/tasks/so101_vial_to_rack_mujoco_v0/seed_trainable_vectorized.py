from __future__ import annotations

import time
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from autoresearch_gym.envs.mujoco_so101_reach import (
    SO101_FEETECH_ACTUATOR_GUESS_PROFILE,
    SO101_MJWARP_GUESSED_PHYSICS_PROFILE,
)
from autoresearch_gym.runner.curves import elapsed_seconds_since, make_train_episode_record, scalar_info_metrics
from autoresearch_gym.runner.experiment import SAMPLE_TRAJECTORY_SOURCE_CANDIDATE_PROVIDED
from autoresearch_gym.tasks.so101_reach_mujoco_v0.seed_trainable import (
    ALGORITHM,
    CONTROL_TYPE,
    REPLAY_SIZE,
    REWARD_RECIPE,
    Agent,
    ReplayBuffer,
    RewardRecipeWrapper,
    flatten_observation,
    load_agent_checkpoint,
    save_agent_checkpoint,
)


EXP_NAME = "so101_mujoco_vial_to_rack_vectorized_sac_seed"
MJWARP_PHYSICS_PROFILE_NAME = "so101_follower_feetech_plastic_guess_v0"
NUM_ENVS = 8
MJWARP_NUM_ENVS = 64
BATCH_SIZE = 512
LEARNING_STARTS = 1_024
UPDATE_AFTER = 1_024
GRADIENT_UPDATES_PER_VECTOR_STEP = 4
LIVE_CALLBACK_EVERY_STEPS = 512
RENDER_SIDECAR_ENABLED = True
VECTOR_ENV_MODE = "sync_or_mujoco_warp"
MAX_ENV_STEPS_SAFETY_CAP = 2_000_000
RECIPE = {
    "algorithm": ALGORITHM,
    "reward_recipe": REWARD_RECIPE,
    "control": "sync_normalized_position_targets_or_mjwarp_feetech_delta_targets",
    "vector_envs": NUM_ENVS,
    "mjwarp_vector_envs": MJWARP_NUM_ENVS,
    "batch_size": BATCH_SIZE,
    "gradient_updates_per_vector_step": GRADIENT_UPDATES_PER_VECTOR_STEP,
    "mjwarp_physics_profile": MJWARP_PHYSICS_PROFILE_NAME,
    "runner": {
        "sample_trajectory_source": SAMPLE_TRAJECTORY_SOURCE_CANDIDATE_PROVIDED,
    },
}


def _benchmark_max_steps(benchmark: Any) -> int:
    max_steps = int(getattr(benchmark, "max_steps"))
    if max_steps <= 0:
        raise ValueError("SO-101 vectorized seed requires benchmark.max_steps > 0")
    return max_steps


def get_candidate() -> dict[str, Any]:
    return {
        "description": (
            "SO-101 MuJoCo vial-to-rack vectorized SAC baseline. Uses headless "
            "SyncVectorEnv workers by default and switches to a MuJoCo Warp batched "
            "collector when the benchmark requests backend=mujoco_warp. Sampled "
            "dashboard trajectories are candidate-rendered so MJWarp Feetech "
            "delta-control runs are visualized with the same guessed joint and "
            "printed-plastic gripper contact dynamics used for collection."
        ),
        "recipe": {
            **RECIPE,
            "task": "vial_to_rack",
            "control": "mjwarp_feetech_delta_targets",
            "printed_gripper_profile": "so101_printed_plastic_gripper_guess_v0",
            "success_requires": ["slot_center", "upright", "rack_height"],
        },
    }


def _make_headless_env(benchmark: Any, seed_offset: int):
    def thunk():
        env_kwargs = dict(getattr(benchmark, "env_kwargs", {}) or {})
        env_kwargs["render_mode"] = None
        horizon = _benchmark_max_steps(benchmark)
        env_kwargs["max_steps"] = horizon
        env_kwargs["max_episode_steps"] = horizon
        env = gym.make(benchmark.env_id, **env_kwargs)
        wrapped = RewardRecipeWrapper(env, REWARD_RECIPE)
        wrapped.action_space.seed(int(benchmark.train_seed) + seed_offset)
        return wrapped

    return thunk


def _wants_mjwarp(benchmark: Any) -> bool:
    env_kwargs = dict(getattr(benchmark, "env_kwargs", {}) or {})
    return str(env_kwargs.get("backend", "")).startswith("mujoco_warp")


def _requested_num_envs(benchmark: Any, use_mjwarp: bool) -> int:
    env_kwargs = dict(getattr(benchmark, "env_kwargs", {}) or {})
    default = MJWARP_NUM_ENVS if use_mjwarp else NUM_ENVS
    return max(1, int(env_kwargs.get("num_envs", default)))


def _mjwarp_task_kind(benchmark: Any) -> str:
    env_id = str(getattr(benchmark, "env_id", ""))
    if "CubeToBin" in env_id:
        return "cube_to_bin"
    if "VialToRack" in env_id:
        return "vial_to_rack"
    return "reach"


def _make_mjwarp_vector_env(benchmark: Any, num_envs: int):
    env_kwargs = dict(getattr(benchmark, "env_kwargs", {}) or {})
    task_kind = _mjwarp_task_kind(benchmark)
    common_kwargs = {
        "num_envs": num_envs,
        "seed": int(benchmark.train_seed),
        "max_steps": _benchmark_max_steps(benchmark),
        "frame_skip": int(env_kwargs.get("frame_skip", 10)),
        "reward_type": str(env_kwargs.get("reward_type", "dense")),
        "model_path": env_kwargs.get("model_path"),
        "actuator_lag_alpha": env_kwargs.get("actuator_lag_alpha"),
        "max_arm_delta": env_kwargs.get("max_arm_delta"),
        "max_gripper_delta": env_kwargs.get("max_gripper_delta"),
        "joint_deadband": env_kwargs.get("joint_deadband"),
        "backlash_half_width": env_kwargs.get("backlash_half_width"),
        "warp_nconmax": int(env_kwargs.get("warp_nconmax", 64)),
        "warp_njmax": int(env_kwargs.get("warp_njmax", 256)),
    }
    if task_kind == "reach":
        from autoresearch_gym.envs.mujoco_so101_reach import AutoresearchMujocoSO101ReachWarpVectorEnv

        return AutoresearchMujocoSO101ReachWarpVectorEnv(**common_kwargs)

    from autoresearch_gym.envs.mujoco_so101_pick_place import AutoresearchMujocoSO101PickPlaceWarpVectorEnv

    if "warp_nconmax" not in env_kwargs:
        common_kwargs["warp_nconmax"] = 128
    if "warp_njmax" not in env_kwargs:
        common_kwargs["warp_njmax"] = 512
    return AutoresearchMujocoSO101PickPlaceWarpVectorEnv(task_kind=task_kind, **common_kwargs)


def _info_for_env(infos: Any, index: int, num_envs: int) -> dict[str, Any]:
    if not isinstance(infos, dict):
        return {}
    final_infos = infos.get("final_info")
    if final_infos is not None and index < len(final_infos) and final_infos[index] is not None:
        return dict(final_infos[index])
    env_info: dict[str, Any] = {}
    for key, value in infos.items():
        if key.startswith("_") or key == "final_info":
            continue
        try:
            if isinstance(value, np.ndarray) and len(value) == num_envs:
                item = value[index]
                env_info[key] = item.item() if hasattr(item, "item") else item
            elif isinstance(value, (list, tuple)) and len(value) == num_envs:
                env_info[key] = value[index]
        except TypeError:
            continue
    return env_info


def _render_env_kwargs(benchmark: Any) -> dict[str, Any]:
    env_kwargs = dict(getattr(benchmark, "env_kwargs", {}) or {})
    env_kwargs["render_mode"] = "rgb_array"
    env_kwargs["max_steps"] = _benchmark_max_steps(benchmark)
    return env_kwargs


def _copy_render_feeds(env: gym.Env[Any, Any]) -> tuple[np.ndarray | None, dict[str, np.ndarray]]:
    raw_env = env.unwrapped
    feeds: dict[str, np.ndarray] = {}
    if hasattr(raw_env, "sample_feeds"):
        try:
            raw_feeds = raw_env.sample_feeds()
            feeds = {
                str(name): np.asarray(frame, dtype=np.uint8).copy()
                for name, frame in raw_feeds.items()
            }
        except Exception:
            feeds = {}
    frame = feeds.get("world")
    if frame is None:
        try:
            rendered = env.render()
            if rendered is not None:
                frame = np.asarray(rendered, dtype=np.uint8).copy()
        except Exception:
            frame = None
    return frame, feeds


def _feed_specs(env: gym.Env[Any, Any]) -> dict[str, dict[str, Any]]:
    raw_env = env.unwrapped
    if hasattr(raw_env, "sample_feed_specs"):
        try:
            specs = raw_env.sample_feed_specs()
            if isinstance(specs, dict):
                return {str(name): dict(spec) for name, spec in specs.items() if isinstance(spec, dict)}
        except Exception:
            return {}
    return {}


class _FeetechDeltaRenderController:
    def __init__(self, env: gym.Env[Any, Any], benchmark: Any) -> None:
        self.env = env
        self.raw_env = env.unwrapped
        env_kwargs = dict(getattr(benchmark, "env_kwargs", {}) or {})
        actuator_profile = SO101_FEETECH_ACTUATOR_GUESS_PROFILE
        self.actuator_lag_alpha = float(
            actuator_profile["actuator_lag_alpha"]
            if env_kwargs.get("actuator_lag_alpha") is None
            else env_kwargs["actuator_lag_alpha"]
        )
        max_arm_delta = float(
            actuator_profile["max_arm_delta_rad_per_policy_step"]
            if env_kwargs.get("max_arm_delta") is None
            else env_kwargs["max_arm_delta"]
        )
        max_gripper_delta = float(
            actuator_profile["max_gripper_delta_rad_per_policy_step"]
            if env_kwargs.get("max_gripper_delta") is None
            else env_kwargs["max_gripper_delta"]
        )
        self.joint_deadband = float(
            actuator_profile["joint_deadband_rad"]
            if env_kwargs.get("joint_deadband") is None
            else env_kwargs["joint_deadband"]
        )
        self.backlash_half_width = float(
            actuator_profile["backlash_half_width_rad"]
            if env_kwargs.get("backlash_half_width") is None
            else env_kwargs["backlash_half_width"]
        )
        self.delta_limits = np.full(self.raw_env.model.nu, max_arm_delta, dtype=np.float32)
        if self.delta_limits.size:
            self.delta_limits[-1] = max_gripper_delta
        self.ctrl_targets = np.zeros(self.raw_env.model.nu, dtype=np.float32)
        self.applied_ctrl = np.zeros(self.raw_env.model.nu, dtype=np.float32)
        self.control_error_sign = np.zeros(self.raw_env.model.nu, dtype=np.float32)

    def reset(self, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed)
        ctrl = np.asarray(self.raw_env.data.ctrl, dtype=np.float32).copy()
        self.ctrl_targets = ctrl.copy()
        self.applied_ctrl = ctrl.copy()
        self.control_error_sign[:] = 0.0
        return flatten_observation(obs), dict(info)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0)
        desired = self.ctrl_targets + action * self.delta_limits
        self.ctrl_targets = np.clip(desired, self.raw_env.ctrl_low, self.raw_env.ctrl_high).astype(np.float32)
        error = self.ctrl_targets - self.applied_ctrl
        sign = np.sign(error).astype(np.float32)
        reversing = (self.control_error_sign != 0.0) & (sign != 0.0) & (sign != self.control_error_sign)
        stalled_by_backlash = reversing & (np.abs(error) < self.backlash_half_width)
        active = (np.abs(error) >= self.joint_deadband) & ~stalled_by_backlash
        self.applied_ctrl[active] += self.actuator_lag_alpha * error[active]
        self.applied_ctrl = np.clip(
            self.applied_ctrl,
            self.raw_env.ctrl_low,
            self.raw_env.ctrl_high,
        ).astype(np.float32)
        self.control_error_sign[active] = sign[active]

        self.raw_env.data.ctrl[:] = self.applied_ctrl
        for _ in range(self.raw_env.frame_skip):
            self.raw_env.mujoco.mj_step(self.raw_env.model, self.raw_env.data)
        self.raw_env.step_count += 1
        self.raw_env.last_action = action.astype(np.float32, copy=True)
        obs_dict = self.raw_env._get_obs()
        info = dict(self.raw_env._info(obs_dict))
        reward = float(self.raw_env.compute_reward(obs_dict["achieved_goal"], obs_dict["desired_goal"], info))
        terminated = bool(info.get("is_success", False))
        truncated = bool(self.raw_env.step_count >= self.raw_env.max_steps)
        info["training_reward"] = reward
        info["actuator_profile"] = SO101_FEETECH_ACTUATOR_GUESS_PROFILE["name"]
        info["physics_profile"] = SO101_MJWARP_GUESSED_PHYSICS_PROFILE["name"]
        info["sampled_trajectory_render_backend"] = "cpu_mujoco_feetech_delta_replay"
        return flatten_observation(obs_dict), reward, terminated, truncated, info


def _sample_policy_trajectory(
    *,
    request: dict[str, Any],
    agent: Agent,
    env: gym.Env[Any, Any],
    obs: np.ndarray,
    benchmark: Any,
    source: str,
    stepper: Any | None = None,
) -> dict[str, Any]:
    frame_stride = max(1, int(request.get("frame_stride", 2)))
    frames: list[np.ndarray] = []
    steps: list[dict[str, Any]] = []
    total_return = 0.0
    terminal_info: dict[str, Any] = {}

    def capture(step_index: int, reward: float | None = None, info: dict[str, Any] | None = None) -> None:
        frame, feeds = _copy_render_feeds(env)
        if frame is None:
            return
        frames.append(frame)
        step_payload: dict[str, Any] = {
            "index": len(frames) - 1,
            "step": int(step_index),
        }
        if reward is not None:
            step_payload["reward"] = float(reward)
        if info:
            step_payload["info"] = scalar_info_metrics(info)
        if feeds:
            step_payload["feeds"] = feeds
        steps.append(step_payload)

    capture(0)
    max_steps = _benchmark_max_steps(benchmark)
    terminated = False
    truncated = False
    for step_index in range(1, max_steps + 1):
        action = agent.act(obs, deterministic=True)
        if stepper is None:
            obs, reward, terminated, truncated, info = env.step(action)
        else:
            obs, reward, terminated, truncated, info = stepper.step(action)
        total_return += float(reward)
        terminal_info = dict(info)
        if step_index % frame_stride == 0 or terminated or truncated:
            capture(step_index, float(reward), terminal_info)
        if terminated or truncated:
            break

    return {
        "episode": int(request["episode"]),
        "sample_index": int(request["sample_index"]),
        "source": source,
        "frames": frames,
        "steps": steps,
        "feed_specs": _feed_specs(env),
        "frame_stride": frame_stride,
        "playback_fps": float(request.get("playback_fps", 20.0)),
        "metadata": {
            "env_id": str(getattr(benchmark, "env_id", "")),
            "return": float(total_return),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "terminal_info": scalar_info_metrics(terminal_info),
        },
    }


def _make_custom_sampled_trajectory(
    *,
    request: dict[str, Any],
    agent: Agent,
    env_factory: Any,
    benchmark: Any,
) -> dict[str, Any]:
    source = str(request.get("source") or SAMPLE_TRAJECTORY_SOURCE_CANDIDATE_PROVIDED)
    seed = int(getattr(benchmark, "train_seed", 0)) + 850_000 + int(request.get("sample_index", 0))
    if _wants_mjwarp(benchmark):
        env = gym.make(str(benchmark.env_id), **_render_env_kwargs(benchmark))
        try:
            controller = _FeetechDeltaRenderController(env, benchmark)
            obs, _ = controller.reset(seed=seed)
            payload = _sample_policy_trajectory(
                request=request,
                agent=agent,
                env=env,
                obs=obs,
                benchmark=benchmark,
                source=source,
                stepper=controller,
            )
            payload["metadata"] = {
                **dict(payload.get("metadata", {})),
                "sampled_trajectory_render_backend": "cpu_mujoco_feetech_delta_replay",
                "actuator_profile": SO101_FEETECH_ACTUATOR_GUESS_PROFILE["name"],
                "physics_profile": SO101_MJWARP_GUESSED_PHYSICS_PROFILE["name"],
                "control": "mjwarp_feetech_delta_targets",
            }
            return payload
        finally:
            env.close()

    env = env_factory(control_type=CONTROL_TYPE, reward_recipe=REWARD_RECIPE)
    try:
        obs, _ = env.reset(seed=seed)
        payload = _sample_policy_trajectory(
            request=request,
            agent=agent,
            env=env,
            obs=flatten_observation(obs),
            benchmark=benchmark,
            source=source,
        )
        payload["metadata"] = {
            **dict(payload.get("metadata", {})),
            "sampled_trajectory_render_backend": "cpu_mujoco_policy_sidecar",
            "control": "normalized_position_targets",
        }
        return payload
    finally:
        env.close()


def _answer_sampled_trajectory_request(
    response: Any,
    *,
    live_callback: Any | None,
    agent: Agent,
    env_factory: Any,
    benchmark: Any,
    records: list[dict[str, Any]],
    global_step: int,
    last_metrics: dict[str, float] | None,
    elapsed_seconds: float,
) -> None:
    if live_callback is None or not isinstance(response, dict):
        return
    request = response.get("sampled_trajectory_request")
    if not isinstance(request, dict) or not request.get("requested"):
        return
    payload = _make_custom_sampled_trajectory(
        request=request,
        agent=agent,
        env_factory=env_factory,
        benchmark=benchmark,
    )
    live_callback(
        status="running",
        episode_records=records,
        total_steps=int(global_step),
        last_metrics=last_metrics,
        current_episode=int(request.get("episode", len(records) + 1)),
        sampled_trajectory=payload,
        elapsed_seconds=float(elapsed_seconds),
    )


def train_agent(
    benchmark: Any,
    env_factory: Any,
    candidate: Any,
    device: torch.device,
    init_checkpoint: str | Any | None = None,
    live_callback: Any | None = None,
) -> tuple[Agent, dict[str, Any]]:
    del candidate
    np.random.seed(int(benchmark.train_seed))
    torch.manual_seed(int(benchmark.train_seed))

    use_mjwarp = _wants_mjwarp(benchmark)
    probe_env = env_factory(control_type=CONTROL_TYPE, reward_recipe=REWARD_RECIPE)
    render_env = probe_env if RENDER_SIDECAR_ENABLED else None
    render_stepper = _FeetechDeltaRenderController(render_env, benchmark) if render_env is not None and use_mjwarp else None
    render_obs = None
    if render_stepper is not None:
        render_obs, _ = render_stepper.reset(seed=int(benchmark.train_seed) + 900_000)
        render_env.action_space.seed(int(benchmark.train_seed) + 900_000)
    elif render_env is not None:
        render_obs, _ = render_env.reset(seed=int(benchmark.train_seed) + 900_000)
        render_env.action_space.seed(int(benchmark.train_seed) + 900_000)
    agent = Agent(probe_env, device=device)
    agent.batch_size = BATCH_SIZE
    resumed_from = load_agent_checkpoint(agent, init_checkpoint) if init_checkpoint is not None else None
    replay = ReplayBuffer(agent.obs_dim, agent.act_dim, REPLAY_SIZE)

    num_envs = _requested_num_envs(benchmark, use_mjwarp)
    physics_profile: dict[str, Any] | None = None
    if use_mjwarp:
        envs = _make_mjwarp_vector_env(benchmark, num_envs)
        obs = envs.reset(seed=int(benchmark.train_seed))
        single_action_space = envs.action_space
        vector_backend = envs.physics_backend
        physics_profile = envs.physics_profile_metadata()
    else:
        env_fns = [_make_headless_env(benchmark, idx) for idx in range(num_envs)]
        envs = gym.vector.SyncVectorEnv(env_fns)
        obs, _ = envs.reset(seed=[int(benchmark.train_seed) + idx for idx in range(num_envs)])
        single_action_space = envs.single_action_space
        vector_backend = "sync"

    global_step = 0
    gradient_updates = 0
    started_at = time.time()
    budget_seconds = getattr(benchmark, "train_seconds", None)
    deadline = started_at + float(budget_seconds) if budget_seconds is not None else None
    episode_records: list[dict[str, Any]] = []
    last_metrics: dict[str, float] | None = None
    active_returns = np.zeros(num_envs, dtype=np.float64)
    active_lengths = np.zeros(num_envs, dtype=np.int64)
    live_step = 0
    render_episode_return = 0.0
    render_episode_length = 0

    def should_continue() -> bool:
        if len(episode_records) >= int(benchmark.train_episodes):
            return False
        if global_step >= MAX_ENV_STEPS_SAFETY_CAP:
            return False
        if deadline is not None and time.time() >= deadline:
            return False
        return True

    def advance_render_env() -> tuple[float, int]:
        nonlocal render_obs, render_episode_return, render_episode_length
        if render_env is None or render_obs is None:
            return render_episode_return, render_episode_length
        action = (
            render_env.action_space.sample()
            if global_step < LEARNING_STARTS
            else agent.act(render_obs, deterministic=True)
        )
        if render_stepper is None:
            render_obs, reward, terminated, truncated, _ = render_env.step(action)
        else:
            render_obs, reward, terminated, truncated, _ = render_stepper.step(action)
        render_episode_return += float(reward)
        render_episode_length += 1
        if terminated or truncated:
            finished_return = render_episode_return
            finished_length = render_episode_length
            if render_stepper is None:
                render_obs, _ = render_env.reset(seed=int(benchmark.train_seed) + 900_000 + len(episode_records))
            else:
                render_obs, _ = render_stepper.reset(seed=int(benchmark.train_seed) + 900_000 + len(episode_records))
            render_episode_return = 0.0
            render_episode_length = 0
            return finished_return, finished_length
        return render_episode_return, render_episode_length

    try:
        if live_callback is not None:
            response = live_callback(
                status="running",
                episode_records=episode_records,
                total_steps=global_step,
                last_metrics=last_metrics,
                env=render_env,
                current_episode=1,
                episode_return=render_episode_return,
                episode_length=render_episode_length,
                agent=agent,
                elapsed_seconds=elapsed_seconds_since(started_at),
            )
            _answer_sampled_trajectory_request(
                response,
                live_callback=live_callback,
                agent=agent,
                env_factory=env_factory,
                benchmark=benchmark,
                records=episode_records,
                global_step=global_step,
                last_metrics=last_metrics,
                elapsed_seconds=elapsed_seconds_since(started_at),
            )

        while should_continue():
            if global_step < LEARNING_STARTS:
                action = np.stack([single_action_space.sample() for _ in range(num_envs)]).astype(np.float32)
            else:
                action = agent.act_batch(obs, deterministic=False)

            next_obs, rewards, terminations, truncations, infos = envs.step(action)
            dones = np.logical_or(terminations, truncations)
            for env_index in range(num_envs):
                replay.add(
                    obs[env_index],
                    action[env_index],
                    float(rewards[env_index]),
                    next_obs[env_index],
                    bool(terminations[env_index]),
                )
            if use_mjwarp and np.any(dones):
                obs = next_obs.copy()
                reset_obs = envs.reset_worlds(dones)
                obs[dones] = reset_obs[dones]
            else:
                obs = next_obs
            active_returns += rewards.astype(np.float64)
            active_lengths += 1
            global_step += num_envs
            sidecar_return, sidecar_length = advance_render_env()

            for env_index in np.flatnonzero(dones):
                info = _info_for_env(infos, int(env_index), num_envs)
                record = make_train_episode_record(
                    episode=len(episode_records) + 1,
                    step=global_step,
                    return_value=float(active_returns[env_index]),
                    length=int(active_lengths[env_index]),
                    success=bool(info.get("is_success", False)),
                    elapsed_seconds=elapsed_seconds_since(started_at),
                    info_metrics=scalar_info_metrics(info),
                    env_index=int(env_index),
                )
                episode_records.append(record)
                active_returns[env_index] = 0.0
                active_lengths[env_index] = 0
                if len(episode_records) >= int(benchmark.train_episodes):
                    break

            if global_step >= UPDATE_AFTER and replay.size >= BATCH_SIZE:
                for _ in range(GRADIENT_UPDATES_PER_VECTOR_STEP):
                    if deadline is not None and time.time() >= deadline:
                        break
                    last_metrics = agent.update(replay)
                    gradient_updates += 1
                if last_metrics is not None:
                    last_metrics = {
                        **last_metrics,
                        "gradient_updates": float(gradient_updates),
                        "num_envs": float(num_envs),
                        "mujoco_warp": float(use_mjwarp),
                    }

            if live_callback is not None and (
                global_step == num_envs or global_step - live_step >= LIVE_CALLBACK_EVERY_STEPS
            ):
                live_step = global_step
                response = live_callback(
                    status="running",
                    episode_records=episode_records,
                    total_steps=global_step,
                    last_metrics=last_metrics,
                    env=render_env,
                    current_episode=len(episode_records) + 1,
                    episode_return=sidecar_return,
                    episode_length=sidecar_length,
                    agent=agent,
                    elapsed_seconds=elapsed_seconds_since(started_at),
                )
                _answer_sampled_trajectory_request(
                    response,
                    live_callback=live_callback,
                    agent=agent,
                    env_factory=env_factory,
                    benchmark=benchmark,
                    records=episode_records,
                    global_step=global_step,
                    last_metrics=last_metrics,
                    elapsed_seconds=elapsed_seconds_since(started_at),
                )
    finally:
        envs.close()
        if render_env is not None:
            render_env.close()

    wall_clock = time.time() - started_at
    stop_reason = (
        "episode_cap_reached"
        if len(episode_records) >= int(benchmark.train_episodes)
        else "max_env_steps_safety_cap_reached"
        if global_step >= MAX_ENV_STEPS_SAFETY_CAP
        else "time_budget_exhausted"
        if deadline is not None and time.time() >= deadline
        else "loop_exited"
    )
    returns = [float(record["return"]) for record in episode_records]
    successes = [1.0 if record.get("success") else 0.0 for record in episode_records]
    return agent, {
        "algorithm": ALGORITHM,
        "episodes": int(benchmark.train_episodes),
        "episodes_completed": len(episode_records),
        "time_budget_seconds": float(budget_seconds) if budget_seconds is not None else None,
        "stop_reason": stop_reason,
        "total_steps": global_step,
        "env_steps": global_step,
        "completed_episodes": len(episode_records),
        "episode_batches": len(episode_records),
        "avg_return": float(np.mean(returns)) if returns else 0.0,
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "avg_length": float(np.mean([record["length"] for record in episode_records])) if episode_records else 0.0,
        "last_metrics": last_metrics,
        "gradient_updates": gradient_updates,
        "episode_records": episode_records,
        "wall_clock_seconds": wall_clock,
        "vector_envs": num_envs,
        "vector_backend": vector_backend,
        "physics_backend": vector_backend,
        "physics_profile": physics_profile,
        "visual_sampling": (
            "candidate_provided cpu_mujoco_feetech_delta_replay"
            if use_mjwarp
            else "candidate_provided cpu_mujoco_policy_sidecar"
        ),
        "init_checkpoint": str(init_checkpoint) if init_checkpoint is not None else None,
        "resumed_from": resumed_from,
    }
