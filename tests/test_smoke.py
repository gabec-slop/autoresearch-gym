from __future__ import annotations

import argparse
import json

import gymnasium as gym
import pytest

import autoresearch_gym  # noqa: F401
from autoresearch_gym.runner.experiment import (
    BenchmarkSpec,
    apply_headless_env_override,
    compact_status_line,
    make_compact_status_writer,
    normalize_run_tag,
    utilization_notes,
)
from autoresearch_gym.runner.session_run import parse_args, write_live_session_pointer


def test_run_tag_normalization_collapses_duplicate_pass_prefix() -> None:
    assert normalize_run_tag("pass01-pass01-baseline") == "pass01-baseline"
    assert normalize_run_tag("pass02-pass02-earlier-learning") == "pass02-earlier-learning"
    assert normalize_run_tag("pass03-amplify-winner") == "pass03-amplify-winner"


def test_utilization_notes_distinguish_unreported_gradient_updates() -> None:
    notes = utilization_notes(
        {"device": "cpu", "steps_per_second": 3.0, "updates_per_second": None},
        {"total_steps": 900},
    )

    assert "updates_per_second is unavailable rather than a measured zero" in notes


def test_utilization_notes_show_reported_zero_gradient_updates() -> None:
    notes = utilization_notes(
        {"device": "cpu", "steps_per_second": 3.0, "updates_per_second": 0.0},
        {"total_steps": 900, "gradient_updates": 0},
    )

    assert "0.0 reported gradient updates/sec" in notes
    assert "unavailable" not in notes


def test_headless_env_override_disables_render_mode_when_supported() -> None:
    benchmark = BenchmarkSpec(
        name="test",
        env_id="Hopper-v5",
        env_kwargs={"render_mode": "rgb_array", "max_episode_steps": 4},
        train_episodes=1,
        train_seconds=None,
        eval_episodes=1,
        max_steps=4,
        reward_type=None,
        render_mode="rgb_array",
        primary_metric="eval_avg_return",
        primary_metric_mode="maximize",
        train_seed=1,
        eval_seed_start=2,
        device="cpu",
        eval_case_bank=None,
    )

    state = apply_headless_env_override(benchmark)

    assert state["requested"] is True
    assert state["effective"] is True
    assert benchmark.render_mode is None
    assert benchmark.env_kwargs["render_mode"] is None


def test_headless_env_override_keeps_panda_pybullet_render_mode() -> None:
    benchmark = BenchmarkSpec(
        name="test",
        env_id="AutoresearchPandaPickAndPlaceDense-v0",
        env_kwargs={"render_mode": "rgb_array", "renderer": "Tiny"},
        train_episodes=1,
        train_seconds=None,
        eval_episodes=1,
        max_steps=4,
        reward_type=None,
        render_mode="rgb_array",
        primary_metric="eval_avg_return",
        primary_metric_mode="maximize",
        train_seed=1,
        eval_seed_start=2,
        device="cpu",
        eval_case_bank=None,
    )

    state = apply_headless_env_override(benchmark)

    assert state["requested"] is True
    assert state["effective"] is False
    assert state["reason"] == "env_requires_render_mode"
    assert benchmark.render_mode == "rgb_array"
    assert benchmark.env_kwargs["render_mode"] == "rgb_array"
    assert benchmark.env_kwargs["renderer"] == "Tiny"


def test_compact_status_writer_uses_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    writer = make_compact_status_writer(10.0)

    writer(
        status="running",
        episode_records=[],
        total_steps=12,
        last_metrics=None,
        current_episode=1,
        episode_return=-1.5,
        episode_length=3,
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "st=run" in captured.err
    assert "step=12" in captured.err
    assert "upd=?" in captured.err


def test_compact_status_line_is_token_efficient() -> None:
    line = compact_status_line(
        elapsed_seconds=65.2,
        train_seconds=300.0,
        train_episodes=100000,
        status="running",
        episode_records=[{"return": 1.0, "success": False}, {"return": 3.0, "success": True}],
        total_steps=1234,
        last_metrics={"qf_loss": 0.5},
        current_episode=3,
        episode_return=4.25,
        episode_length=17,
    )

    assert (
        line
        == "t=00:01:05 pct=21.7 time=65/300s st=run step=1234 ep=3 done=2 avg=2.000 succ=0.500 cur=4.250 len=17 upd=Y"
    )


def test_compact_status_line_uses_episode_progress_without_time_budget() -> None:
    line = compact_status_line(
        elapsed_seconds=65.2,
        train_seconds=None,
        train_episodes=4,
        status="running",
        episode_records=[{"return": 1.0, "success": False}, {"return": 3.0, "success": True}],
        total_steps=1234,
        last_metrics=None,
        current_episode=3,
        episode_return=4.25,
        episode_length=17,
    )

    assert (
        line
        == "t=00:01:05 pct=50.0 eps=2/4 st=run step=1234 ep=3 done=2 avg=2.000 succ=0.500 cur=4.250 len=17 upd=?"
    )


def test_compact_status_writer_can_write_compact_file(tmp_path) -> None:
    status_file = tmp_path / "status.log"
    writer = make_compact_status_writer(
        10.0,
        train_seconds=300.0,
        train_episodes=100000,
        emit_stderr=False,
        compact_status_file=status_file,
    )

    writer(
        status="running",
        episode_records=[],
        total_steps=12,
        last_metrics=None,
        current_episode=1,
        episode_return=-1.5,
        episode_length=3,
    )

    assert status_file.read_text(encoding="utf-8").strip().endswith(
        "time=0/300s st=run step=12 ep=1 done=0 avg=0.000 succ=0.000 cur=-1.500 len=3 upd=?"
    )


def test_run_parser_accepts_headless_and_compact_status_flags() -> None:
    args = parse_args(
        [
            "--candidate",
            "candidate.py",
            "--headless-env",
            "--compact-status",
            "--compact-status-file",
            "status.log",
            "--status-interval-seconds",
            "2.5",
        ]
    )

    assert args.headless_env is True
    assert args.compact_status is True
    assert str(args.compact_status_file) == "status.log"
    assert args.status_interval_seconds == 2.5


def test_live_session_pointer_records_unresolved_latest_alias(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    base_dir = tmp_path / "autoresearch_runs"
    session_dir = base_dir / "sessions" / "example-session"
    session_dir.mkdir(parents=True)
    args = argparse.Namespace(tag="pass01-baseline", search_mode="linear")

    write_live_session_pointer(base_dir, session_dir, args)

    payload = json.loads((base_dir / "live_session.json").read_text(encoding="utf-8"))
    assert payload["session_path"] == "autoresearch_runs/sessions/example-session"
    assert payload["latest_alias_path"] == "autoresearch_runs/sessions/latest"


def test_all_bundled_benchmarks_have_expected_budget_shape() -> None:
    from pathlib import Path

    task_root = Path(__file__).resolve().parents[1] / "autoresearch_gym" / "tasks"
    task_dirs = [path for path in task_root.iterdir() if path.is_dir() and not path.name.startswith("__")]
    assert task_dirs
    for task_dir in task_dirs:
        episode_benchmark = task_dir / "benchmark.json"
        wall_clock_benchmark = task_dir / "benchmark_wall_clock.json"
        assert episode_benchmark.exists(), task_dir
        assert wall_clock_benchmark.exists(), task_dir
        episode_payload = json.loads(episode_benchmark.read_text(encoding="utf-8"))
        wall_clock_payload = json.loads(wall_clock_benchmark.read_text(encoding="utf-8"))
        assert "train_seconds" not in episode_payload
        assert wall_clock_payload["train_seconds"] == 300
        for payload in (episode_payload, wall_clock_payload):
            assert payload["env_id"]
            assert payload["render_mode"] == "rgb_array"
            assert payload["env_kwargs"]["render_mode"] == "rgb_array"
            assert payload["train_episodes"] > 0
            assert payload["eval_episodes"] > 0
            assert payload["primary_metric"]


def test_registered_env_resets_and_steps() -> None:
    pytest.importorskip("panda_gym")
    pytest.importorskip("pybullet")

    env = gym.make("PandaBatToGoal-v0", render_mode="rgb_array", max_steps=4)
    obs, info = env.reset(seed=123)
    assert obs.shape == env.observation_space.shape
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)
    assert next_obs.shape == env.observation_space.shape
    assert isinstance(float(reward), float)
    assert isinstance(bool(terminated), bool)
    assert isinstance(bool(truncated), bool)
    env.close()


def test_panda_pick_and_place_goal_marker_is_visually_distinct_when_panda_gym_is_installed() -> None:
    pytest.importorskip("panda_gym")
    pytest.importorskip("pybullet")

    env = gym.make("AutoresearchPandaPickAndPlaceDense-v0", render_mode="rgb_array", renderer="Tiny")
    try:
        assert env.spec is not None
        assert env.spec.max_episode_steps == 50
        sim = env.unwrapped.sim
        object_id = sim._bodies_idx["object"]
        target_id = sim._bodies_idx["target"]
        object_rgba = sim.physics_client.getVisualShapeData(object_id)[0][7]
        target_rgba = sim.physics_client.getVisualShapeData(target_id)[0][7]

        assert object_rgba[:3] != target_rgba[:3]
        assert len(sim.physics_client.getCollisionShapeData(object_id, -1)) == 1
        assert len(sim.physics_client.getCollisionShapeData(target_id, -1)) == 0
    finally:
        env.close()


def test_hopper_seed_task_resets_and_steps_when_mujoco_is_installed() -> None:
    pytest.importorskip("mujoco")
    from autoresearch_gym.tasks.hopper_v0.seed_trainable import RewardRecipeWrapper

    env = RewardRecipeWrapper(gym.make("Hopper-v5", render_mode="rgb_array", max_episode_steps=4), "task_reward")
    obs, info = env.reset(seed=123)
    assert obs.shape == env.observation_space.shape
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)
    assert next_obs.shape == env.observation_space.shape
    assert isinstance(float(reward), float)
    assert isinstance(bool(terminated), bool)
    assert isinstance(bool(truncated), bool)
    env.close()


def test_inverted_pendulum_seed_task_resets_and_steps_when_mujoco_is_installed() -> None:
    pytest.importorskip("mujoco")
    from autoresearch_gym.tasks.inverted_pendulum_v5.seed_trainable import RewardRecipeWrapper

    env = RewardRecipeWrapper(gym.make("InvertedPendulum-v5", render_mode="rgb_array", max_episode_steps=4), "task_reward")
    obs, info = env.reset(seed=123)
    assert obs.shape == env.observation_space.shape
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)
    assert next_obs.shape == env.observation_space.shape
    assert isinstance(float(reward), float)
    assert isinstance(bool(terminated), bool)
    assert isinstance(bool(truncated), bool)
    env.close()


def test_fetch_push_seed_task_resets_and_steps_when_robotics_is_installed() -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("gymnasium_robotics")
    from autoresearch_gym.tasks.fetch_push_dense_v0.seed_trainable import RewardRecipeWrapper

    env = RewardRecipeWrapper(gym.make("FetchPushDense-v4", render_mode="rgb_array", max_episode_steps=4), "task_dense")
    obs, info = env.reset(seed=123)
    assert obs.shape == env.observation_space.shape
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)
    assert next_obs.shape == env.observation_space.shape
    assert isinstance(float(reward), float)
    assert isinstance(bool(terminated), bool)
    assert isinstance(bool(truncated), bool)
    env.close()


def test_fetch_push_her_seed_task_resets_and_steps_when_robotics_is_installed() -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("gymnasium_robotics")
    from autoresearch_gym.tasks.fetch_push_dense_v0.seed_trainable_her import RewardRecipeWrapper

    env = RewardRecipeWrapper(gym.make("FetchPushDense-v4", render_mode="rgb_array", max_episode_steps=4), "task_dense")
    obs, info = env.reset(seed=123)
    assert obs.shape == env.observation_space.shape
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)
    assert next_obs.shape == env.observation_space.shape
    assert isinstance(float(reward), float)
    assert isinstance(bool(terminated), bool)
    assert isinstance(bool(truncated), bool)
    env.close()


def test_panda_pick_and_place_seed_task_resets_and_steps_when_panda_gym_is_installed() -> None:
    pytest.importorskip("panda_gym")
    pytest.importorskip("pybullet")
    from autoresearch_gym.tasks.panda_pick_and_place_v0.seed_trainable import RewardRecipeWrapper

    env = RewardRecipeWrapper(
        gym.make("AutoresearchPandaPickAndPlaceDense-v0", render_mode="rgb_array", renderer="Tiny"),
        "task_dense",
    )
    obs, info = env.reset(seed=123)
    assert obs.shape == env.observation_space.shape
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)
    assert next_obs.shape == env.observation_space.shape
    assert isinstance(float(reward), float)
    assert isinstance(bool(terminated), bool)
    assert isinstance(bool(truncated), bool)
    env.close()


def test_panda_pick_and_place_her_seed_task_resets_and_steps_when_panda_gym_is_installed() -> None:
    pytest.importorskip("panda_gym")
    pytest.importorskip("pybullet")
    from autoresearch_gym.tasks.panda_pick_and_place_v0.seed_trainable_her import RewardRecipeWrapper

    env = RewardRecipeWrapper(
        gym.make("AutoresearchPandaPickAndPlaceDense-v0", render_mode="rgb_array", renderer="Tiny"),
        "task_dense",
    )
    obs, info = env.reset(seed=123)
    assert obs.shape == env.observation_space.shape
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)
    assert next_obs.shape == env.observation_space.shape
    assert isinstance(float(reward), float)
    assert isinstance(bool(terminated), bool)
    assert isinstance(bool(truncated), bool)
    env.close()
