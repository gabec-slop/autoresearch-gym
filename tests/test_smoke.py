from __future__ import annotations

import argparse
import json
import sys
import types

import gymnasium as gym
import numpy as np
import pytest

import autoresearch_gym  # noqa: F401
from autoresearch_gym import cli
from autoresearch_gym.runner.experiment import (
    BenchmarkSpec,
    TrainProbeSpec,
    apply_headless_env_override,
    compact_status_line,
    make_compact_status_writer,
    make_live_writer,
    make_policy_probe_callback,
    normalize_train_summary_curve,
    normalize_run_tag,
    utilization_flags,
    validate_train_curve_contract,
)
from autoresearch_gym.runner.curves import (
    make_policy_probe_record,
    make_train_collection_window_record,
    make_train_episode_record,
)
from autoresearch_gym.runner.session_run import parse_args, write_live_session_pointer
from scripts.check_trainable_contract import validate_records, validate_summary


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


def test_utilization_flags_distinguish_unreported_gradient_updates() -> None:
    flags = utilization_flags(
        {"device": "cpu", "steps_per_second": 3.0, "updates_per_second": None},
        {"total_steps": 900},
    )

    assert flags["gradient_updates_reported"] is False


def test_utilization_flags_show_reported_zero_gradient_updates() -> None:
    flags = utilization_flags(
        {"device": "cpu", "steps_per_second": 3.0, "updates_per_second": 0.0},
        {"total_steps": 900, "gradient_updates": 0},
    )

    assert flags["gradient_updates_reported"] is True


def test_trainable_contract_checker_requires_gradient_update_counter() -> None:
    errors = validate_summary(
        {
            "train": {
                "episodes_completed": 10,
                "completed_episodes": 10,
                "episode_batches": 10,
                "total_steps": 300,
                "env_steps": 300,
                "last_metrics": {"actor_loss": 1.0},
            }
        },
        require_gradient_updates=True,
    )

    assert "train.gradient_updates is required" in "\n".join(errors)
    assert "train.last_metrics.gradient_updates is required" in "\n".join(errors)


def test_trainable_contract_checker_accepts_reported_zero_gradient_updates() -> None:
    errors = validate_summary(
        {
            "train": {
                "episodes_completed": 0,
                "completed_episodes": 0,
                "episode_batches": 0,
                "total_steps": 12,
                "env_steps": 12,
                "gradient_updates": 0,
                "last_metrics": {"gradient_updates": 0},
            }
        },
        require_gradient_updates=True,
    )

    assert errors == []


def test_trainable_contract_checker_rejects_probe_axis_after_collection_count() -> None:
    errors = validate_records(
        [
            make_train_episode_record(episode=1, return_value=1.0, length=2),
            make_policy_probe_record(
                episode=3,
                return_value=2.0,
                length=2.0,
                step=2,
                elapsed_seconds=1.0,
                probe_episodes=1,
                probe_seed_start=123,
            ),
        ]
    )

    assert "policy_probe episode axis 3 exceeds completed collection rollouts 1" in "\n".join(errors)


def test_utilization_flags_detect_when_nvidia_gpu_is_visible_but_cpu_selected() -> None:
    flags = utilization_flags(
        {
            "device": "cpu",
            "steps_per_second": 3.0,
            "updates_per_second": None,
            "visible_nvidia_device_name": "NVIDIA GeForce RTX 3060",
        },
        {"total_steps": 900},
    )

    assert flags["torch_selected_cpu_with_visible_nvidia"] is True


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


def test_compact_status_writer_ignores_enriched_live_callback_fields(capsys: pytest.CaptureFixture[str]) -> None:
    writer = make_compact_status_writer(10.0)

    writer(
        status="running",
        episode_records=[],
        total_steps=12,
        last_metrics=None,
        current_episode=1,
        episode_return=-1.5,
        episode_length=3,
        agent=object(),
        elapsed_seconds=0.1,
    )

    assert "step=12" in capsys.readouterr().err


def test_live_writer_ignores_enriched_live_callback_fields(tmp_path) -> None:
    benchmark = BenchmarkSpec(
        name="test",
        env_id="CartPole-v1",
        env_kwargs={"render_mode": "rgb_array"},
        train_episodes=10,
        train_seconds=30.0,
        eval_episodes=1,
        max_steps=50,
        reward_type=None,
        render_mode="rgb_array",
        primary_metric="eval_avg_return",
        primary_metric_mode="maximize",
        train_seed=1,
        eval_seed_start=2,
        device="cpu",
        eval_case_bank=None,
        train_probe=TrainProbeSpec(enabled=False),
    )
    writer = make_live_writer(tmp_path / "session", "run-1", "tag-1", benchmark, {"description": "candidate"})
    assert writer is not None

    writer(
        status="running",
        episode_records=[],
        total_steps=12,
        last_metrics=None,
        current_episode=1,
        episode_return=-1.5,
        episode_length=3,
        agent=object(),
        elapsed_seconds=0.1,
    )

    payload = json.loads((tmp_path / "session" / "live" / "current_run_metrics.json").read_text(encoding="utf-8"))
    assert payload["current"]["step"] == 12
    assert payload["current"]["env_steps"] == 12
    assert payload["current"]["episode_batch"] == 0
    assert payload["current"]["active_episode_batch"] == 1
    assert payload["current"]["completed_episodes"] == 0


def test_live_writer_keeps_full_episode_history(tmp_path) -> None:
    benchmark = BenchmarkSpec(
        name="test",
        env_id="CartPole-v1",
        env_kwargs={"render_mode": "rgb_array"},
        train_episodes=1000,
        train_seconds=30.0,
        eval_episodes=1,
        max_steps=50,
        reward_type=None,
        render_mode="rgb_array",
        primary_metric="eval_avg_return",
        primary_metric_mode="maximize",
        train_seed=1,
        eval_seed_start=2,
        device="cpu",
        eval_case_bank=None,
        train_probe=TrainProbeSpec(enabled=False),
    )
    writer = make_live_writer(tmp_path / "session", "run-1", "tag-1", benchmark, {"description": "candidate"})
    assert writer is not None
    episode_records = [
        make_train_episode_record(
            episode=index + 1,
            return_value=float(index),
            length=1,
            step=index + 1,
            elapsed_seconds=float(index) * 0.1,
        )
        for index in range(405)
    ]

    writer(
        status="running",
        episode_records=episode_records,
        total_steps=405,
        last_metrics=None,
        current_episode=406,
        episode_return=0.0,
        episode_length=0,
    )

    payload = json.loads((tmp_path / "session" / "live" / "current_run_metrics.json").read_text(encoding="utf-8"))
    assert len(payload["episodes"]) == 405
    assert payload["episodes"][0]["episode"] == 1
    assert payload["episodes"][-1]["episode"] == 405


def test_live_writer_sampled_trajectory_records_full_episode(tmp_path) -> None:
    class DummyVisualEnv(gym.Env):
        metadata = {"render_modes": ["rgb_array"]}

        def __init__(self) -> None:
            self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
            self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
            self.steps = 0

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.steps = 0
            return np.zeros(1, dtype=np.float32), {}

        def step(self, action):
            self.steps += 1
            return np.zeros(1, dtype=np.float32), 0.0, self.steps >= 5, False, {}

        def render(self, *args, **kwargs):
            return np.full((8, 8, 3), self.steps, dtype=np.uint8)

    benchmark = BenchmarkSpec(
        name="test",
        env_id="DummyVisual-v0",
        env_kwargs={"render_mode": "rgb_array"},
        train_episodes=10,
        train_seconds=30.0,
        eval_episodes=1,
        max_steps=50,
        reward_type=None,
        render_mode="rgb_array",
        primary_metric="eval_avg_return",
        primary_metric_mode="maximize",
        train_seed=1,
        eval_seed_start=2,
        device="cpu",
        eval_case_bank=None,
        train_probe=TrainProbeSpec(enabled=False),
    )
    writer = make_live_writer(tmp_path / "session", "run-1", "tag-1", benchmark, {"description": "candidate"})
    assert writer is not None
    env = writer.wrap_env(DummyVisualEnv())  # type: ignore[attr-defined]

    env.reset()
    terminated = False
    while not terminated:
        _, _, terminated, _, _ = env.step(np.zeros(1, dtype=np.float32))

    manifest = json.loads(
        (tmp_path / "session" / "live" / "trajectories" / "run-1" / "episode_000001" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["status"] == "completed"
    assert manifest["frame_count"] >= 3


def test_live_writer_sampled_trajectory_is_pinned_to_one_env(tmp_path) -> None:
    class DummyVisualEnv(gym.Env):
        metadata = {"render_modes": ["rgb_array"]}

        def __init__(self, value: int) -> None:
            self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
            self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
            self.value = value
            self.steps = 0

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.steps = 0
            return np.zeros(1, dtype=np.float32), {}

        def step(self, action):
            self.steps += 1
            return np.zeros(1, dtype=np.float32), 0.0, self.steps >= 4, False, {}

        def render(self, *args, **kwargs):
            return np.full((8, 8, 3), self.value + self.steps, dtype=np.uint8)

    benchmark = BenchmarkSpec(
        name="test",
        env_id="DummyVisual-v0",
        env_kwargs={"render_mode": "rgb_array"},
        train_episodes=10,
        train_seconds=30.0,
        eval_episodes=1,
        max_steps=50,
        reward_type=None,
        render_mode="rgb_array",
        primary_metric="eval_avg_return",
        primary_metric_mode="maximize",
        train_seed=1,
        eval_seed_start=2,
        device="cpu",
        eval_case_bank=None,
        train_probe=TrainProbeSpec(enabled=False),
    )
    writer = make_live_writer(tmp_path / "session", "run-1", "tag-1", benchmark, {"description": "candidate"})
    assert writer is not None
    env_a = writer.wrap_env(DummyVisualEnv(10))  # type: ignore[attr-defined]
    env_b = writer.wrap_env(DummyVisualEnv(100))  # type: ignore[attr-defined]

    env_a.reset()
    manifest_path = tmp_path / "session" / "live" / "trajectories" / "run-1" / "episode_000001" / "manifest.json"
    initial_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    active_env_id = initial_manifest["env_id"]

    env_b.reset()
    env_b.step(np.zeros(1, dtype=np.float32))
    after_other_env_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert after_other_env_manifest["status"] == "recording"
    assert after_other_env_manifest["env_id"] == active_env_id
    assert after_other_env_manifest["frame_count"] == initial_manifest["frame_count"]

    env_a.step(np.zeros(1, dtype=np.float32))
    env_a.step(np.zeros(1, dtype=np.float32))
    after_owner_step_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert after_owner_step_manifest["env_id"] == active_env_id
    assert after_owner_step_manifest["frame_count"] == initial_manifest["frame_count"] + 1


def test_live_writer_reports_observed_env_count(tmp_path) -> None:
    class DummyVisualEnv(gym.Env):
        metadata = {"render_modes": ["rgb_array"]}

        def __init__(self) -> None:
            self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
            self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return np.zeros(1, dtype=np.float32), {}

        def step(self, action):
            return np.zeros(1, dtype=np.float32), 0.0, False, False, {}

        def render(self, *args, **kwargs):
            return np.zeros((8, 8, 3), dtype=np.uint8)

    benchmark = BenchmarkSpec(
        name="test",
        env_id="DummyVisual-v0",
        env_kwargs={"render_mode": "rgb_array"},
        train_episodes=10,
        train_seconds=30.0,
        eval_episodes=1,
        max_steps=50,
        reward_type=None,
        render_mode="rgb_array",
        primary_metric="eval_avg_return",
        primary_metric_mode="maximize",
        train_seed=1,
        eval_seed_start=2,
        device="cpu",
        eval_case_bank=None,
        train_probe=TrainProbeSpec(enabled=False),
    )
    writer = make_live_writer(tmp_path / "session", "run-1", "tag-1", benchmark, {"description": "candidate"})
    assert writer is not None
    writer.wrap_env(DummyVisualEnv())  # type: ignore[attr-defined]
    writer.wrap_env(DummyVisualEnv())  # type: ignore[attr-defined]
    writer(
        status="running",
        episode_records=[],
        total_steps=0,
        last_metrics=None,
        current_episode=1,
        episode_return=0.0,
        episode_length=0,
    )

    metrics = json.loads((tmp_path / "session" / "live" / "current_run_metrics.json").read_text(encoding="utf-8"))
    assert metrics["observed_env_count"] == 2
    assert metrics["run"]["observed_env_count"] == 2
    assert metrics["visual"]["observed_env_count"] == 2


def test_bat_to_goal_seed_trainable_samples_first_real_episode_rollout(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    from autoresearch_gym.tasks.bat_to_goal_v0 import seed_trainable

    class TinyVisualEnv(gym.Env):
        metadata = {"render_modes": ["rgb_array"]}

        def __init__(self) -> None:
            self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
            self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
            self.steps = 0
            self.reset_count = 0

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.steps = 0
            self.reset_count += 1
            return np.zeros(2, dtype=np.float32), {}

        def step(self, action):
            self.steps += 1
            terminated = self.steps >= 5
            info = {"is_success": False, "contacted_ball": False, "ball_goal_distance": 0.5}
            return np.zeros(2, dtype=np.float32), -0.1, terminated, False, info

        def render(self, *args, **kwargs):
            return np.full((8, 8, 3), self.steps, dtype=np.uint8)

    benchmark = types.SimpleNamespace(
        train_seed=1,
        train_episodes=1,
        train_seconds=None,
    )
    live_benchmark = BenchmarkSpec(
        name="test",
        env_id="TinyVisual-v0",
        env_kwargs={"render_mode": "rgb_array"},
        train_episodes=1,
        train_seconds=None,
        eval_episodes=1,
        max_steps=5,
        reward_type=None,
        render_mode="rgb_array",
        primary_metric="eval_avg_return",
        primary_metric_mode="maximize",
        train_seed=1,
        eval_seed_start=2,
        device="cpu",
        eval_case_bank=None,
        train_probe=TrainProbeSpec(enabled=False),
    )
    candidate = seed_trainable.get_candidate()
    writer = make_live_writer(tmp_path / "session", "run-1", "tag-1", live_benchmark, candidate)
    assert writer is not None
    base_env = TinyVisualEnv()

    def env_factory(control_type=None, reward_recipe=None):
        return writer.wrap_env(base_env)  # type: ignore[attr-defined]

    _, summary = seed_trainable.train_agent(
        benchmark,
        env_factory,
        candidate,
        torch.device("cpu"),
        live_callback=writer,
    )

    manifest = json.loads(
        (tmp_path / "session" / "live" / "trajectories" / "run-1" / "episode_000001" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["episodes_completed"] == 1
    assert base_env.reset_count == 1
    assert manifest["status"] == "completed"
    assert manifest["frame_count"] >= 3


def test_live_writer_finalizes_active_sampled_trajectory_on_finish(tmp_path) -> None:
    class DummyVisualEnv(gym.Env):
        metadata = {"render_modes": ["rgb_array"]}

        def __init__(self) -> None:
            self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
            self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return np.zeros(1, dtype=np.float32), {}

        def step(self, action):
            return np.zeros(1, dtype=np.float32), 0.0, False, False, {}

        def render(self, *args, **kwargs):
            return np.zeros((8, 8, 3), dtype=np.uint8)

    benchmark = BenchmarkSpec(
        name="test",
        env_id="DummyVisual-v0",
        env_kwargs={"render_mode": "rgb_array"},
        train_episodes=10,
        train_seconds=30.0,
        eval_episodes=1,
        max_steps=50,
        reward_type=None,
        render_mode="rgb_array",
        primary_metric="eval_avg_return",
        primary_metric_mode="maximize",
        train_seed=1,
        eval_seed_start=2,
        device="cpu",
        eval_case_bank=None,
        train_probe=TrainProbeSpec(enabled=False),
    )
    writer = make_live_writer(tmp_path / "session", "run-1", "tag-1", benchmark, {"description": "candidate"})
    assert writer is not None
    env = writer.wrap_env(DummyVisualEnv())  # type: ignore[attr-defined]

    env.reset()
    env.step(np.zeros(1, dtype=np.float32))
    writer(
        status="finished",
        episode_records=[],
        total_steps=1,
        last_metrics=None,
        env=env,
        current_episode=1,
        episode_return=0.0,
        episode_length=1,
    )

    metrics = json.loads((tmp_path / "session" / "live" / "current_run_metrics.json").read_text(encoding="utf-8"))
    manifest_path = metrics["visual"]["trajectory_manifest_path"]
    manifest = json.loads((tmp_path / "session" / "live" / "trajectories" / "run-1" / "episode_000001" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest_path.endswith("manifest.json")
    assert manifest["status"] == "interrupted"
    assert manifest["reason"] == "training_finished"
    assert metrics["visual"]["active_sampled_episode"] is None


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
            "env_steps": 120000,
            "episode_records": [
                make_train_collection_window_record(
                    episode=1,
                    return_value=42.5,
                    length=43.5,
                    episodes_in_window=1200,
                    success=False,
                    step=120000,
                    env_steps_in_window=120000,
                    sampled=True,
                )
            ],
        }
    )


def test_train_curve_contract_rejects_env_step_alias_mismatch() -> None:
    with pytest.raises(ValueError, match="env_steps must match total_steps"):
        validate_train_curve_contract(
            {
                "episodes_completed": 1,
                "total_steps": 12,
                "env_steps": 11,
                "episode_records": [
                    make_train_episode_record(
                        episode=1,
                        return_value=0.0,
                        length=12,
                        step=12,
                    )
                ],
            }
        )


def test_train_summary_normalization_counts_collection_window_episodes() -> None:
    summary = {
        "episodes_completed": 0,
        "total_steps": 120,
        "episode_records": [
            make_train_collection_window_record(
                episode=1,
                return_value=10.0,
                length=12.0,
                episodes_in_window=5,
                success=False,
                step=60,
                env_steps_in_window=60,
            ),
            make_train_collection_window_record(
                episode=2,
                return_value=20.0,
                length=18.0,
                episodes_in_window=15,
                success=True,
                step=120,
                env_steps_in_window=60,
            ),
        ],
    }

    normalize_train_summary_curve(summary)

    assert summary["env_steps"] == 120
    assert summary["episodes_completed"] == 20
    assert summary["completed_episodes"] == 20
    assert summary["episode_batches"] == 2
    assert summary["avg_return"] == pytest.approx(17.5)
    assert summary["avg_length"] == pytest.approx(16.5)
    assert summary["success_rate"] == pytest.approx(0.75)


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
    assert payload["episode_records"][-1]["episode"] == 1
    assert callback.probe_records[-1]["return"] == 12.0


def test_evaluate_agent_ignores_candidate_owned_evaluator(monkeypatch: pytest.MonkeyPatch) -> None:
    from autoresearch_gym.runner import experiment

    class CustomAgent:
        def evaluate(self, benchmark, candidate):
            raise AssertionError("candidate-owned final evaluation must not be used")

        def act(self, obs, deterministic: bool = False):
            return np.array([0.0], dtype=np.float32)

    class FixedEvalEnv:
        def __init__(self) -> None:
            self.step_count = 0

        def reset(self, *, seed=None, options=None):
            self.step_count = 0
            return np.array([0.0], dtype=np.float32), {}

        def step(self, action):
            self.step_count += 1
            return (
                np.array([0.0], dtype=np.float32),
                2.0,
                True,
                False,
                {"is_success": True, "fixed_eval_path": 1.0},
            )

        def close(self) -> None:
            pass

    benchmark = BenchmarkSpec(
        name="custom",
        env_id="RunnerOwnedEval-v0",
        env_kwargs={},
        train_episodes=1,
        train_seconds=None,
        eval_episodes=3,
        max_steps=1,
        reward_type=None,
        render_mode=None,
        primary_metric="eval_avg_return",
        primary_metric_mode="maximize",
        train_seed=1,
        eval_seed_start=2,
        device="cpu",
        eval_case_bank=None,
    )
    monkeypatch.setattr(experiment, "make_eval_env", lambda benchmark, control_type: FixedEvalEnv())

    summary = experiment.evaluate_agent(CustomAgent(), benchmark, object())

    assert summary["episodes"] == 3
    assert summary["success_rate"] == 1.0
    assert summary["avg_return"] == 2.0
    assert summary["avg_fixed_eval_path"] == 1.0


def test_eval_case_bank_must_cover_requested_eval_episodes(tmp_path) -> None:
    from autoresearch_gym.runner.experiment import load_eval_cases

    case_bank = tmp_path / "eval_cases.json"
    case_bank.write_text(json.dumps({"cases": [{"name": "case-one"}]}), encoding="utf-8")
    benchmark = BenchmarkSpec(
        name="custom",
        env_id="RunnerOwnedEval-v0",
        env_kwargs={},
        train_episodes=1,
        train_seconds=None,
        eval_episodes=2,
        max_steps=1,
        reward_type=None,
        render_mode=None,
        primary_metric="eval_avg_return",
        primary_metric_mode="maximize",
        train_seed=1,
        eval_seed_start=2,
        device="cpu",
        eval_case_bank=case_bank,
    )

    with pytest.raises(ValueError, match="has 1 cases but benchmark requests 2 eval episodes"):
        load_eval_cases(benchmark)


def test_fixed_eval_reset_failure_invalidates_run(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from autoresearch_gym.runner import experiment

    class Agent:
        def act(self, obs, deterministic: bool = False):
            return np.array([0.0], dtype=np.float32)

    class BrokenFixedCaseEnv:
        def reset(self, *, seed=None, options=None):
            if options is not None:
                raise ValueError("fixed case unsupported")
            return np.array([0.0], dtype=np.float32), {}

        def step(self, action):
            return np.array([0.0], dtype=np.float32), 0.0, True, False, {"is_success": False}

        def close(self) -> None:
            pass

    case_bank = tmp_path / "eval_cases.json"
    case_bank.write_text(json.dumps({"cases": [{"name": "fixed-case"}]}), encoding="utf-8")
    benchmark = BenchmarkSpec(
        name="custom",
        env_id="RunnerOwnedEval-v0",
        env_kwargs={},
        train_episodes=1,
        train_seconds=None,
        eval_episodes=1,
        max_steps=1,
        reward_type=None,
        render_mode=None,
        primary_metric="eval_avg_return",
        primary_metric_mode="maximize",
        train_seed=1,
        eval_seed_start=2,
        device="cpu",
        eval_case_bank=case_bank,
    )
    monkeypatch.setattr(experiment, "make_eval_env", lambda benchmark, control_type: BrokenFixedCaseEnv())

    with pytest.raises(RuntimeError, match="fixed eval reset failed for fixed-case"):
        experiment.evaluate_agent(Agent(), benchmark, object())


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


def test_panda_pick_and_place_rejects_initial_goal_overlap_when_panda_gym_is_installed() -> None:
    pytest.importorskip("panda_gym")
    pytest.importorskip("pybullet")

    env = gym.make("AutoresearchPandaPickAndPlaceDense-v0", render_mode="rgb_array", renderer="Tiny")
    try:
        threshold = env.unwrapped.task.distance_threshold
        for seed in (665, 4520):
            obs, info = env.reset(seed=seed)
            distance = float(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"]))

            assert distance >= threshold
            assert not bool(info["is_success"])
            assert info["initial_goal_distance"] == pytest.approx(distance)
            assert info["initial_resample_attempts"] > 0
    finally:
        env.close()


def test_panda_pick_and_place_suppresses_pybullet_background_argv_echo(capfd: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("panda_gym")
    pytest.importorskip("pybullet")
    capfd.readouterr()

    env = gym.make("AutoresearchPandaPickAndPlaceDense-v0", render_mode="rgb_array", renderer="Tiny")
    env.close()
    captured = capfd.readouterr()

    assert "argv[" not in captured.out
    assert "argv[" not in captured.err
    assert "background_color" not in captured.out
    assert "background_color" not in captured.err


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
