from __future__ import annotations

import argparse
import json
import sys
import types

import gymnasium as gym
import pytest

import autoresearch_gym  # noqa: F401
from autoresearch_gym import cli
from autoresearch_gym.runner.experiment import (
    BenchmarkSpec,
    TrainProbeSpec,
    apply_headless_env_override,
    compact_status_line,
    make_compact_status_writer,
    make_policy_probe_callback,
    normalize_run_tag,
    utilization_notes,
    validate_train_curve_contract,
)
from autoresearch_gym.runner.curves import (
    make_policy_probe_record,
    make_train_episode_record,
)
from autoresearch_gym.runner.session_run import parse_args, write_live_session_pointer


def test_doctor_warns_when_nvidia_gpu_is_visible_but_torch_cuda_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_torch = types.SimpleNamespace(
        __version__="9.9.9+cpu",
        version=types.SimpleNamespace(cuda=None),
        cuda=types.SimpleNamespace(
            is_available=lambda: False,
            device_count=lambda: 0,
            get_device_name=lambda index: f"fake cuda {index}",
        ),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)),
        device=lambda value: value,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        cli,
        "_nvidia_smi_gpus",
        lambda: [{"name": "NVIDIA GeForce RTX 3060", "memory_total_mb": 12288, "driver_version": "596.21"}],
    )

    status = cli.cmd_doctor(argparse.Namespace(device="auto", strict=True))
    payload = json.loads(capsys.readouterr().out)

    assert status == 1
    assert payload["ok"] is False
    assert payload["selected_device"] == "cpu"
    assert payload["checks"][0]["status"] == "warn"
    assert "cannot use CUDA through PyTorch" in payload["checks"][0]["message"]


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


def test_utilization_notes_warn_when_nvidia_gpu_is_visible_but_cpu_selected() -> None:
    notes = utilization_notes(
        {
            "device": "cpu",
            "steps_per_second": 3.0,
            "updates_per_second": None,
            "visible_nvidia_device_name": "NVIDIA GeForce RTX 3060",
        },
        {"total_steps": 900},
    )

    assert "can see NVIDIA GeForce RTX 3060" in notes
    assert "CPU-only Torch wheel" in notes


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


def test_train_curve_contract_rejects_missing_completed_episode_records() -> None:
    with pytest.raises(ValueError, match="completed episodes"):
        validate_train_curve_contract(
            {
                "episodes_completed": 12,
                "total_steps": 345,
                "episode_records": [],
            }
        )


def test_train_curve_contract_accepts_windowed_episode_records() -> None:
    validate_train_curve_contract(
        {
            "episodes_completed": 1200,
            "total_steps": 120000,
            "episode_records": [
                {
                    "episode": 1200,
                    "return": 42.5,
                    "length": 43.5,
                    "success": False,
                    "step": 120000,
                    "sampled": True,
                    "episodes_in_window": 1200,
                }
            ],
        }
    )


def test_train_curve_contract_rejects_steps_without_curve_or_unsupported_status() -> None:
    with pytest.raises(ValueError, match="training steps"):
        validate_train_curve_contract(
            {
                "episodes_completed": 0,
                "total_steps": 12,
                "episode_records": [],
            }
        )

    validate_train_curve_contract(
        {
            "episodes_completed": 0,
            "total_steps": 12,
            "episode_records": [],
            "curve_status": "unsupported",
            "curve_status_reason": "external trainer",
        }
    )


def test_train_curve_contract_accepts_typed_episode_and_probe_records() -> None:
    validate_train_curve_contract(
        {
            "episodes_completed": 1,
            "total_steps": 10,
            "episode_records": [
                make_train_episode_record(
                    episode=1,
                    return_value=2.0,
                    length=3,
                    success=False,
                    step=3,
                    elapsed_seconds=0.1,
                ),
                make_policy_probe_record(
                    episode=2,
                    return_value=4.0,
                    length=5.0,
                    step=10,
                    elapsed_seconds=5.0,
                    probe_episodes=2,
                    probe_seed_start=900_000,
                ),
            ],
        }
    )


def test_policy_probe_callback_keeps_seed_episode_records_unmutated() -> None:
    module = types.SimpleNamespace(
        probe_policy=lambda *args, **kwargs: {
            "episodes": kwargs["episodes"],
            "avg_return": 12.0,
            "avg_length": 13.0,
            "success_rate": 0.5,
        }
    )
    benchmark = BenchmarkSpec(
        name="probe-test",
        env_id="CartPole-v1",
        env_kwargs={},
        train_episodes=10,
        train_seconds=None,
        eval_episodes=1,
        max_steps=100,
        reward_type=None,
        render_mode=None,
        primary_metric="eval_avg_return",
        primary_metric_mode="maximize",
        train_seed=1,
        eval_seed_start=2,
        device="cpu",
        eval_case_bank=None,
        train_probe=TrainProbeSpec(enabled=True, interval_seconds=0.0, episodes=2, seed_start=123),
    )
    callback = make_policy_probe_callback(module, benchmark, object(), "cpu")
    seed_records = [
        make_train_episode_record(
            episode=1,
            return_value=1.0,
            length=2,
            step=2,
            elapsed_seconds=0.1,
        )
    ]

    payload = callback(
        status="running",
        agent=object(),
        episode_records=seed_records,
        total_steps=2,
        elapsed_seconds=1.0,
    )

    assert len(seed_records) == 1
    assert len(payload["episode_records"]) == 2
    assert payload["episode_records"][-1]["record_type"] == "policy_probe"
    assert callback.probe_records[-1]["return"] == 12.0


def test_evaluate_agent_uses_candidate_owned_evaluator() -> None:
    from autoresearch_gym.runner.experiment import evaluate_agent

    class CustomAgent:
        def evaluate(self, benchmark, candidate):
            return {
                "episodes": benchmark.eval_episodes,
                "success_rate": 1.0,
                "avg_return": -0.25,
                "avg_length": 0.0,
                "mpkpe": 0.25,
                "episode_records": [],
            }

    benchmark = BenchmarkSpec(
        name="custom",
        env_id="NotARegisteredGymEnv-v0",
        env_kwargs={},
        train_episodes=1,
        train_seconds=None,
        eval_episodes=3,
        max_steps=1,
        reward_type=None,
        render_mode=None,
        primary_metric="eval_mpkpe",
        primary_metric_mode="minimize",
        train_seed=1,
        eval_seed_start=2,
        device="cpu",
        eval_case_bank=None,
    )

    summary = evaluate_agent(CustomAgent(), benchmark, object())

    assert summary["episodes"] == 3
    assert summary["mpkpe"] == 0.25


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
            "--probe-interval-seconds",
            "3.5",
            "--probe-episodes",
            "2",
            "--no-train-probe",
        ]
    )

    assert args.headless_env is True
    assert args.compact_status is True
    assert str(args.compact_status_file) == "status.log"
    assert args.status_interval_seconds == 2.5
    assert args.probe_interval_seconds == 3.5
    assert args.probe_episodes == 2
    assert args.no_train_probe is True


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
