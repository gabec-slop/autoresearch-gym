from __future__ import annotations

import argparse
import ast
import importlib.util
import io
import json
import shutil
import sys
import tarfile
import types
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

import autoresearch_gym  # noqa: F401
from autoresearch_gym import cli
from autoresearch_gym.external.base import ArtifactSet, RunBundle
from autoresearch_gym.runner.experiment import (
    BenchmarkSpec,
    TrainProbeSpec,
    apply_headless_env_override,
    candidate_metadata,
    compact_status_line,
    make_compact_status_writer,
    make_live_writer,
    make_policy_probe_callback,
    normalize_train_summary_curve,
    normalize_run_tag,
    render_mujoco_kinematic_frame,
    utilization_flags,
    validate_train_curve_contract,
)
from autoresearch_gym.external.targets import load_target_config
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


def test_external_target_config_redacts_private_ssh_fields(tmp_path) -> None:
    config = tmp_path / "targets.toml"
    config.write_text(
        """
[targets.windows_gpu]
kind = "ssh"
host = "user@windows-gpu.example.invalid"
remote_root = "C:/code/autoresearch-gym"
path_style = "windows"
python = ".venv/Scripts/python.exe"
""".strip(),
        encoding="utf-8",
    )

    target = load_target_config("windows_gpu", config_path=config)
    redacted = target.redacted_summary()

    assert target.kind == "ssh"
    assert target.host == "user@windows-gpu.example.invalid"
    assert target.artifact_sync == "scp"
    assert redacted == {
        "target": "windows_gpu",
        "target_kind": "ssh",
        "host_redacted": True,
        "remote_root_redacted": True,
        "path_style": "windows",
    }

    config.write_text(config.read_text(encoding="utf-8") + '\nartifact_sync = "sftp"\n', encoding="utf-8")
    legacy_target = load_target_config("windows_gpu", config_path=config)
    assert legacy_target.artifact_sync == "scp"


def test_fake_external_run_writes_normalized_artifacts(tmp_path) -> None:
    task_dir = tmp_path / "fake_external_task"
    task_dir.mkdir()
    (task_dir / "eval_cases.json").write_text(
        json.dumps(
            {
                "name": "fake_external_eval_cases_v0",
                "cases": [
                    {"name": "fake-case-01", "difficulty": 0.1},
                    {"name": "fake-case-02", "difficulty": 0.2},
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (task_dir / "benchmark.json").write_text(
        json.dumps(
            {
                "name": "fake_external_v0",
                "env_id": "external:fake:FakeExternal-v0",
                "env_kwargs": {"render_mode": "rgb_array"},
                "train_episodes": 3,
                "eval_episodes": 2,
                "max_steps": 12,
                "render_mode": "rgb_array",
                "primary_metric": "eval_avg_return",
                "primary_metric_mode": "maximize",
                "train_seed": 1,
                "eval_seed_start": 7000,
                "device": "external",
                "eval_case_bank": "eval_cases.json",
                "execution_backend": {
                    "kind": "external",
                    "name": "fake",
                    "adapter": "autoresearch_gym.external.fake_backend:FakeExternalBackend",
                    "artifact_schema_version": 1,
                    "supports_live_frame": True,
                    "supports_sampled_trajectory": False,
                    "policy_artifact": "fake_checkpoint",
                    "execution_target": "fake",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (task_dir / "seed_trainable.py").write_text(
        """
from __future__ import annotations

from pathlib import Path
from typing import Any


def get_candidate() -> dict[str, Any]:
    return {"description": "Fake external backend candidate.", "recipe": {"algorithm": "fake"}}


class RewardRecipeWrapper:
    def __init__(self, env: Any, recipe: str | None = None) -> None:
        self.env = env
        self.recipe = recipe


def train_agent(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError("fake external tests must run through execution_backend")


def save_agent_checkpoint(agent: Any, path: Path, metadata: dict[str, Any] | None = None) -> None:
    path.write_text("fake checkpoint placeholder\\n", encoding="utf-8")
""".strip(),
        encoding="utf-8",
    )
    summary = cli.session_run.run_experiment(
        benchmark_path=task_dir / "benchmark.json",
        candidate_path=task_dir / "seed_trainable.py",
        tag="pytest-fake-external",
        out_dir=tmp_path / "runs",
        results_path=tmp_path / "results.jsonl",
    )

    run_dir = Path(tmp_path / "runs" / summary["run_id"])
    assert summary["execution"]["target_kind"] == "fake"
    assert summary["objective"]["value"] == summary["eval"]["avg_return"]
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "train_episodes.json").exists()
    assert (run_dir / "eval_episodes.json").exists()
    assert Path(summary["artifacts"]["checkpoint_path"]).exists()
    assert Path(summary["media"]["live_frame_path"]).exists()


def test_unitree_cleanrl_style_seeds_expose_mjlab_levers() -> None:
    from autoresearch_gym.tasks.unitree_g1_motion_mirror_v0 import seed_trainable as g1_seed
    from autoresearch_gym.tasks.unitree_go2_rough_locomotion_v0 import seed_trainable as go2_seed
    from autoresearch_gym.tasks.unitree_go2_rough_locomotion_v0 import (
        seed_trainable_staged_curriculum as go2_staged_seed,
    )

    g1_recipe = g1_seed.get_candidate()["recipe"]
    go2_recipe = go2_seed.get_candidate()["recipe"]
    go2_staged_recipe = go2_staged_seed.get_candidate()["recipe"]

    for recipe in (g1_recipe, go2_recipe, go2_staged_recipe):
        assert recipe["style"] == "cleanrl_mjlab_ppo"
        assert recipe["runner"]["num_envs"] >= 1024
        assert recipe["runner"]["num_steps_per_env"] > 0
        assert recipe["actor"]["hidden_dims"]
        assert recipe["critic"]["hidden_dims"]
        assert "learning_rate" in recipe["ppo"]
        assert "clip_param" in recipe["ppo"]
        assert "action_scale" in recipe["environment"]
        assert recipe["reward_weights"]
        assert recipe["event_overrides"]
        assert recipe["termination_overrides"]

    assert "motion_command" in g1_recipe
    assert g1_recipe["runner"]["save_interval"] == 100
    assert g1_recipe["runner"]["probe_interval_iterations"] == 100
    assert g1_recipe["runner"]["sample_rollout_frame_count"] == 24
    assert "motion_global_root_pos" in g1_recipe["reward_weights"]
    assert "motion_body_ang_vel" in g1_recipe["reward_params"]
    assert "anchor_pos" in g1_recipe["termination_overrides"]
    g1_diagnostic = g1_recipe["diagnostic_series"]
    g1_diagnostic_keys = [item["key"] for item in g1_diagnostic["series"]]
    assert g1_diagnostic["title"] == "G1 motion mirroring diagnostics"
    assert "episode_reward_motion_global_root_pos" in g1_diagnostic_keys
    assert "episode_reward_motion_body_ang_vel" in g1_diagnostic_keys
    assert "metrics_mpkpe" in g1_diagnostic_keys
    assert "episode_termination_anchor_pos" in g1_diagnostic_keys

    assert "twist_command" in go2_recipe
    assert "terrain" in go2_recipe
    assert "track_linear_velocity" in go2_recipe["reward_weights"]
    assert "foot_gait" in go2_recipe["reward_params"]
    assert "command_vel" in go2_recipe["curriculum_overrides"]
    go2_diagnostic = go2_recipe["diagnostic_series"]
    go2_diagnostic_keys = [item["key"] for item in go2_diagnostic["series"]]
    assert go2_diagnostic["title"] == "Go2 locomotion diagnostics"
    assert "episode_reward_track_linear_velocity" in go2_diagnostic_keys
    assert "episode_reward_body_orientation_l2" in go2_diagnostic_keys
    assert "curriculum_terrain_levels" in go2_diagnostic_keys
    assert "episode_reward_pose" not in go2_diagnostic_keys
    assert "episode_reward_soft_landing" not in go2_diagnostic_keys
    assert go2_staged_recipe["single_pass_curriculum"] is True
    assert go2_staged_recipe["diagnostic_series"] == go2_diagnostic
    assert go2_staged_recipe["terrain"]["max_init_terrain_level"] == 5
    assert go2_staged_recipe["runner"]["sample_trajectory_source"] == "train_context"
    assert go2_staged_recipe["event_overrides"]["push_robot"]["enabled"] is False
    assert go2_staged_recipe["twist_command"]["ranges"]["lin_vel_x"][1] >= 1.0
    assert go2_staged_recipe["curriculum_plan"][0]["name"] == "stand_and_creep"
    velocity_stages = go2_staged_recipe["curriculum_overrides"]["command_vel"]["params"]["velocity_stages"]
    assert len(velocity_stages) == 5
    assert [stage["step"] for stage in velocity_stages] == [-1, 7200, 12000, 16800, 21600]


def test_unitree_backend_reads_nested_recipe_budget_fields() -> None:
    from autoresearch_gym.external.unitree_backend import (
        _learning_iterations,
        _parallel_env_count,
        _seed,
        _steps_per_env,
    )

    bundle = {
        "candidate": {
            "recipe": {
                "runner": {
                    "num_envs": 128,
                    "eval_num_envs": 16,
                    "num_steps_per_env": 12,
                    "max_iterations": 7,
                    "seed": 1234,
                }
            }
        },
        "benchmark": {"env_kwargs": {"num_envs": 4096, "eval_num_envs": 1024}, "train_episodes": 2},
    }

    assert _parallel_env_count(bundle) == 128
    assert _parallel_env_count(bundle, for_eval=True) == 16
    assert _steps_per_env(bundle) == 12
    assert _learning_iterations(bundle) == 7
    assert _seed(bundle, 42) == 1234


def test_unitree_backend_scales_mjlab_probe_interval_to_run_length(monkeypatch: pytest.MonkeyPatch) -> None:
    from autoresearch_gym.external.unitree_backend import _mjlab_probe_interval_iterations

    bundle = {"benchmark": {"train_seconds": 1800.0}}
    recipe = {"runner": {"save_interval": 500}}

    assert _mjlab_probe_interval_iterations(recipe, bundle, 692) == 139
    assert _mjlab_probe_interval_iterations({"runner": {"probe_interval_iterations": 100}}, bundle, 692) == 100
    assert _mjlab_probe_interval_iterations({"runner": {"probe_interval_iterations": 0}}, bundle, 692) == 0

    monkeypatch.setenv("UNITREE_MJLAB_TARGET_POLICY_PROBES", "10")
    assert _mjlab_probe_interval_iterations(recipe, bundle, 692) == 70


def test_ssh_live_sync_merges_policy_probe_records_for_dashboard(tmp_path) -> None:
    from autoresearch_gym.external.targets import SshTarget, TargetConfig

    external_dir = tmp_path / "external"
    live_dir = external_dir / "live"
    live_dir.mkdir(parents=True)
    (live_dir / "current_run_metrics.json").write_text(
        json.dumps(
            {
                "current": {"info_metrics": {"train_mean_reward": 1.0}},
                "episodes": [
                    {
                        "record_type": "train_collection_window",
                        "episode": 1,
                        "return": 1.0,
                        "step": 100,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (external_dir / "policy_probe_records.jsonl").write_text(
        json.dumps(
            {
                "record_type": "policy_probe",
                "episode": 100,
                "return": -2.5,
                "length": 120.0,
                "step": 100,
                "probe_seed_start": 900100,
                "elapsed_seconds": 30.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    bundle = RunBundle(
        run_id="pytest-run",
        tag="pytest",
        benchmark_path=tmp_path / "benchmark.json",
        candidate_path=tmp_path / "candidate.py",
        local_run_dir=tmp_path,
        external_dir=external_dir,
        benchmark={},
        candidate={},
        candidate_metadata={},
        execution_backend={},
        eval_cases=None,
        train_episodes=1,
        train_seconds=None,
        eval_episodes=1,
        max_steps=120,
        compact_status_file=None,
        session_dir=None,
        target_name="ssh",
    )
    target = SshTarget(TargetConfig(name="ssh", kind="ssh", host="example.invalid", remote_root="/tmp/repo"))

    target._merge_live_policy_probe_records(bundle)
    payload = json.loads((live_dir / "current_run_metrics.json").read_text(encoding="utf-8"))

    assert [record["record_type"] for record in payload["episodes"]] == [
        "train_collection_window",
        "policy_probe",
    ]
    assert payload["current"]["info_metrics"]["policy_probe_count"] == 1.0
    assert payload["current"]["info_metrics"]["policy_probe_return"] == -2.5


def test_unitree_backend_records_curriculum_signal_scalars() -> None:
    from autoresearch_gym.external.unitree_backend import MJLAB_TRAIN_SCRIPT

    assert "CURRICULUM_SIGNAL_KEYS" in MJLAB_TRAIN_SCRIPT
    assert "episode_reward_track_linear_velocity" in MJLAB_TRAIN_SCRIPT
    assert "episode_reward_body_orientation_l2" in MJLAB_TRAIN_SCRIPT
    assert "episode_reward_motion_global_root_pos" in MJLAB_TRAIN_SCRIPT
    assert "episode_reward_motion_body_ang_vel" in MJLAB_TRAIN_SCRIPT
    assert "episode_termination_anchor_pos" in MJLAB_TRAIN_SCRIPT
    assert "metrics_mpkpe" in MJLAB_TRAIN_SCRIPT
    assert "episode_termination_illegal_contact" in MJLAB_TRAIN_SCRIPT
    assert "curriculum_terrain_levels" in MJLAB_TRAIN_SCRIPT
    assert "curriculum_command_stage" in MJLAB_TRAIN_SCRIPT
    assert '"curriculum_command_lin_vel_x"' in MJLAB_TRAIN_SCRIPT
    assert "info_metrics[key] = value" in MJLAB_TRAIN_SCRIPT
    assert "diagnostic_series" in MJLAB_TRAIN_SCRIPT
    assert "_recipe_diagnostic_series(recipe)" in MJLAB_TRAIN_SCRIPT
    assert "_diagnostic_series_metadata(records, recipe)" in MJLAB_TRAIN_SCRIPT


def test_dashboard_diagnostics_are_metadata_driven() -> None:
    source = Path("dashboard/index.html").read_text(encoding="utf-8")

    assert "diagnostic_series" in source
    assert "data-diagnostic-series" in source
    assert "inferDiagnosticSeriesSpecs" in source
    assert "episode_reward_" in source
    assert "normalizeDiagnosticSeriesSpecs" in source
    assert "CURRICULUM_DIAGNOSTIC_SERIES" not in source
    assert "episode_reward_track_linear_velocity" not in source


def test_unitree_mjlab_train_context_probes_run_out_of_process() -> None:
    from autoresearch_gym.external.unitree_backend import MJLAB_TRAIN_SCRIPT

    assert "def _run_train_context_probe_subprocess" in MJLAB_TRAIN_SCRIPT
    assert "--probe-checkpoint" in MJLAB_TRAIN_SCRIPT
    assert "if args.probe_checkpoint" in MJLAB_TRAIN_SCRIPT
    assert "train_script=Path(__file__).resolve()" in MJLAB_TRAIN_SCRIPT
    assert "_run_train_context_probe_subprocess(" in MJLAB_TRAIN_SCRIPT
    assert "def _probe_subprocess_env" in MJLAB_TRAIN_SCRIPT
    assert 'env.setdefault("PYTHONIOENCODING", "utf-8")' in MJLAB_TRAIN_SCRIPT
    assert "_run_checkpoint_probe(" in MJLAB_TRAIN_SCRIPT
    assert "policy_probes\" / \"logs" in MJLAB_TRAIN_SCRIPT
    assert "error=no_rollout" in MJLAB_TRAIN_SCRIPT
    assert "env_cfg.scene.num_envs = 1 if frame_dir is not None else requested_num_envs" in MJLAB_TRAIN_SCRIPT
    assert "command_metrics" in MJLAB_TRAIN_SCRIPT
    assert "get_command(name)" in MJLAB_TRAIN_SCRIPT
    assert "except PermissionError" in MJLAB_TRAIN_SCRIPT
    assert "monitor_errors.log" in MJLAB_TRAIN_SCRIPT


def test_unitree_go2_mjlab_uses_return_primary_without_fabricated_success(tmp_path, monkeypatch) -> None:
    from autoresearch_gym.external import unitree_backend

    for benchmark_name in ("benchmark.json", "benchmark_wall_clock.json", "benchmark_lower_level.json"):
        payload = json.loads(
            Path("autoresearch_gym/tasks/unitree_go2_rough_locomotion_v0", benchmark_name).read_text(encoding="utf-8")
        )
        assert payload["primary_metric"] == "eval_avg_return"
        assert payload["primary_metric_mode"] == "maximize"

    checkpoint = tmp_path / "agent_checkpoint.pt"
    checkpoint.write_text("checkpoint", encoding="utf-8")

    def fake_run_subprocess(argv, **kwargs):
        out_json = Path(argv[argv.index("--out-json") + 1])
        out_json.write_text(
            json.dumps(
                {
                    "task_id": "Unitree-Go2-Rough",
                    "steps": 200,
                    "num_envs": 64,
                    "avg_step_reward": -0.5,
                    "return": -100.0,
                    "done_fraction": 0.1,
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(unitree_backend, "_run_subprocess", fake_run_subprocess)
    monkeypatch.setattr(unitree_backend, "_unitree_root", lambda bundle: tmp_path)
    monkeypatch.setattr(unitree_backend, "_mjlab_python", lambda bundle: "python")

    bundle = {
        "run_id": "pytest-go2",
        "task_family": "go2_rough_locomotion",
        "benchmark": {
            "env_kwargs": {"task_id": "Unitree-Go2-Rough", "eval_num_envs": 64},
            "max_steps": 200,
            "eval_episodes": 2,
            "primary_metric": "eval_avg_return",
        },
        "candidate": {"recipe": {"runner": {"eval_num_envs": 64, "seed": 52}}},
    }
    unitree_backend._run_mjlab_rollout(bundle, tmp_path, checkpoint, mode="eval")
    summary = json.loads((tmp_path / "eval_result.json").read_text(encoding="utf-8"))

    assert summary["avg_return"] == -100.0
    assert summary["metric_source"] == "mjlab_rollout_reward"
    assert "success_rate" not in summary
    assert "success" not in summary["episode_records"][0]


def test_unitree_mjlab_train_bridge_compiles_with_recipe_overrides() -> None:
    from autoresearch_gym.external.unitree_backend import MJLAB_TRAIN_SCRIPT

    compile(MJLAB_TRAIN_SCRIPT, "mjlab_train_bridge.py", "exec")
    assert "--recipe-json" in MJLAB_TRAIN_SCRIPT
    assert "reward_weights" in MJLAB_TRAIN_SCRIPT
    assert "curriculum_overrides" in MJLAB_TRAIN_SCRIPT
    assert "train_result_partial.json" in MJLAB_TRAIN_SCRIPT
    assert "policy_probe_records.jsonl" in MJLAB_TRAIN_SCRIPT
    assert "current_run_metrics.json" in MJLAB_TRAIN_SCRIPT
    assert "--sample-rollout-frame-count" in MJLAB_TRAIN_SCRIPT
    assert "--sample-trajectory-source" in MJLAB_TRAIN_SCRIPT
    assert "_run_train_context_sample" in MJLAB_TRAIN_SCRIPT
    assert "mjlab_live_probe" in MJLAB_TRAIN_SCRIPT


def _literal_eval_with_module_constants(node: ast.AST, constants: dict[str, object]) -> object:
    class ConstantResolver(ast.NodeTransformer):
        def visit_Name(self, name_node: ast.Name) -> ast.AST:
            if name_node.id not in constants:
                return name_node
            replacement = ast.parse(repr(constants[name_node.id]), mode="eval").body
            return ast.copy_location(replacement, name_node)

    resolved = ConstantResolver().visit(node)
    ast.fix_missing_locations(resolved)
    return ast.literal_eval(resolved)


def test_custom_trajectory_sampling_task_recipes_use_generic_live_writer_contract(tmp_path: Path) -> None:
    custom_seed_recipes: list[tuple[Path, dict[str, object]]] = []
    for seed_path in Path("autoresearch_gym/tasks").glob("*/seed_trainable*.py"):
        source = seed_path.read_text(encoding="utf-8")
        if "sample_trajectory_source" not in source:
            continue
        module_ast = ast.parse(source, filename=str(seed_path))
        constants: dict[str, object] = {}
        for node in module_ast.body:
            if isinstance(node, ast.ImportFrom) and node.module:
                imported = __import__(node.module, fromlist=[alias.name for alias in node.names])
                for alias in node.names:
                    constants[alias.asname or alias.name] = getattr(imported, alias.name)
                continue
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Name) or target.id == "RECIPE":
                continue
            try:
                constants[target.id] = ast.literal_eval(node.value)
            except (SyntaxError, ValueError):
                continue
        recipe: dict[str, object] | None = None
        for node in module_ast.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "RECIPE" for target in node.targets
            ):
                value = _literal_eval_with_module_constants(node.value, constants)
                if isinstance(value, dict):
                    recipe = value
                break
        assert recipe is not None, f"{seed_path} declares sample_trajectory_source without a literal RECIPE"
        runner = recipe.get("runner")
        assert isinstance(runner, dict)
        source_name = runner.get("sample_trajectory_source")
        if source_name and source_name != "fallback":
            custom_seed_recipes.append((seed_path, recipe))

    assert custom_seed_recipes, "expected at least one task seed to exercise custom trajectory sampling"

    for seed_path, recipe in custom_seed_recipes:
        runner = recipe["runner"]
        assert isinstance(runner, dict)
        sample_source = str(runner["sample_trajectory_source"])
        benchmark = BenchmarkSpec(
            name=seed_path.parent.name,
            env_id="GenericCustomSampling-v0",
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
        run_id = f"{seed_path.parent.name}-{seed_path.stem}"
        writer = make_live_writer(
            tmp_path / "session",
            run_id,
            "tag-1",
            benchmark,
            {"description": str(seed_path), "recipe": recipe},
        )
        assert writer is not None

        response = writer(
            status="running",
            episode_records=[],
            total_steps=0,
            last_metrics=None,
            current_episode=1,
            episode_return=0.0,
            episode_length=0,
        )
        request = response["sampled_trajectory_request"]
        assert request["requested"] is True
        assert request["source"] == sample_source

        writer(
            status="running",
            episode_records=[],
            total_steps=24,
            last_metrics=None,
            current_episode=int(request["episode"]),
            episode_return=0.0,
            episode_length=24,
            sampled_trajectory={
                "episode": request["episode"],
                "sample_index": request["sample_index"],
                "source": sample_source,
                "frames": [
                    np.zeros((8, 8, 3), dtype=np.uint8),
                    np.full((8, 8, 3), 64, dtype=np.uint8),
                ],
                "metadata": {"seed_path": str(seed_path)},
            },
        )

        metrics = json.loads(
            (tmp_path / "session" / "live" / "current_run_metrics.json").read_text(encoding="utf-8")
        )
        manifest_path = Path(metrics["visual"]["trajectory_manifest_path"])
        if not manifest_path.is_absolute():
            manifest_path = Path.cwd() / manifest_path
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["source"] == sample_source
        assert manifest["frame_count"] == 2
        assert manifest["metadata"]["seed_path"] == str(seed_path)


def test_ssh_target_sync_live_mirrors_remote_dashboard_metrics(tmp_path, monkeypatch) -> None:
    from autoresearch_gym.external.targets import SshTarget, TargetConfig

    remote_external = tmp_path / "remote" / "autoresearch_runs" / "external_remote" / "run-1" / "external"
    remote_live = remote_external / "live"
    remote_live.mkdir(parents=True)
    (remote_live / "current_run_metrics.json").write_text(json.dumps({"episodes": [{"return": 1.0}]}), encoding="utf-8")
    (remote_live / "status.log").write_text("st=running step=1\n", encoding="utf-8")

    def fake_run(argv, **kwargs):
        assert argv[0] == "scp"
        remote_arg = str(argv[-2]).replace("\\", "/")
        destination = Path(argv[-1])
        if remote_arg.endswith("/live/current_run_metrics.json"):
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(remote_live / "current_run_metrics.json", destination)
        elif remote_arg.endswith("/live/status.log"):
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(remote_live / "status.log", destination)
        else:
            return types.SimpleNamespace(returncode=1, stdout="", stderr="missing")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("autoresearch_gym.external.targets.subprocess.run", fake_run)
    target = SshTarget(
        TargetConfig(
            name="pytest-ssh",
            kind="ssh",
            host="example.invalid",
            remote_root=str(tmp_path / "remote"),
            path_style="posix",
        )
    )
    session_dir = tmp_path / "session"
    bundle = types.SimpleNamespace(run_id="run-1", external_dir=tmp_path / "local_external", session_dir=session_dir)

    target.sync_live(bundle)

    mirrored = json.loads((session_dir / "live" / "current_run_metrics.json").read_text(encoding="utf-8"))
    assert mirrored["episodes"][0]["return"] == 1.0
    assert (session_dir / "live" / "status.log").read_text(encoding="utf-8") == "st=running step=1\n"


def test_ssh_target_sync_live_localizes_remote_sampled_rollout_paths(tmp_path, monkeypatch) -> None:
    from autoresearch_gym.external.targets import SshTarget, TargetConfig

    remote_root = "C:/code/autoresearch-gym"
    run_id = "run-1"
    remote_external = tmp_path / "remote" / "autoresearch_runs" / "external_remote" / run_id / "external"
    remote_live = remote_external / "live"
    remote_trajectory = remote_external / "trajectories" / "sample_000001"
    remote_live.mkdir(parents=True)
    remote_trajectory.mkdir(parents=True)
    windows_manifest = (
        f"{remote_root}\\autoresearch_runs\\external_remote\\{run_id}\\external"
        "\\trajectories\\sample_000001\\manifest.json"
    )
    windows_frame = (
        f"{remote_root}\\autoresearch_runs\\external_remote\\{run_id}\\external"
        "\\trajectories\\sample_000001\\frame_0000.jpg"
    )
    (remote_live / "current_run_metrics.json").write_text(
        json.dumps(
            {
                "run": {
                    "run_id": run_id,
                    "trajectory_manifest_path": windows_manifest,
                    "visual": {
                        "mode": "sampled_trajectory",
                        "trajectory_manifest_path": windows_manifest,
                        "trajectory_latest_frame_path": windows_frame,
                    },
                },
                "episodes": [],
            }
        ),
        encoding="utf-8",
    )
    (remote_trajectory / "manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "frames": [windows_frame],
                "latest_frame_path": windows_frame,
                "frame_count": 1,
            }
        ),
        encoding="utf-8",
    )
    (remote_trajectory / "frame_0000.jpg").write_bytes(b"fake image")

    def fake_run(argv, **kwargs):
        assert argv[0] == "scp"
        remote_arg = str(argv[-2]).replace("\\", "/")
        destination = Path(argv[-1])
        if remote_arg.endswith("/live/current_run_metrics.json"):
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(remote_live / "current_run_metrics.json", destination)
        elif remote_arg.endswith("/trajectories/sample_000001/manifest.json"):
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(remote_trajectory / "manifest.json", destination)
        elif remote_arg.endswith("/trajectories/sample_000001/frame_0000.jpg"):
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(remote_trajectory / "frame_0000.jpg", destination)
        else:
            return types.SimpleNamespace(returncode=1, stdout="", stderr="missing")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("autoresearch_gym.external.targets.subprocess.run", fake_run)
    target = SshTarget(
        TargetConfig(
            name="pytest-ssh",
            kind="ssh",
            host="example.invalid",
            remote_root=remote_root,
            path_style="windows",
        )
    )
    session_dir = tmp_path / "session"
    bundle = types.SimpleNamespace(run_id=run_id, external_dir=tmp_path / "local_external", session_dir=session_dir)

    target.sync_live(bundle)

    mirrored = json.loads((session_dir / "live" / "current_run_metrics.json").read_text(encoding="utf-8"))
    manifest_path = mirrored["run"]["visual"]["trajectory_manifest_path"]
    latest_frame_path = mirrored["run"]["visual"]["trajectory_latest_frame_path"]
    assert "C:" not in manifest_path
    assert "external_remote" not in manifest_path
    assert manifest_path.endswith("local_external/trajectories/sample_000001/manifest.json")
    assert latest_frame_path.endswith("local_external/trajectories/sample_000001/frame_0000.jpg")

    manifest = json.loads((tmp_path / "local_external" / "trajectories" / "sample_000001" / "manifest.json").read_text())
    assert manifest["frames"][0].endswith("local_external/trajectories/sample_000001/frame_0000.jpg")
    assert "C:" not in manifest["latest_frame_path"]


def test_ssh_target_fetch_artifacts_uses_single_remote_archive(tmp_path, monkeypatch) -> None:
    from autoresearch_gym.external.targets import SshTarget, TargetConfig

    remote_external = tmp_path / "remote" / "autoresearch_runs" / "external_remote" / "run-1" / "external"
    remote_external.mkdir(parents=True)
    (remote_external / "train_result.json").write_text(json.dumps({"total_steps": 12}), encoding="utf-8")
    (remote_external / "agent_checkpoint.pt").write_bytes(b"checkpoint")

    archive_bytes = io.BytesIO()
    with tarfile.open(fileobj=archive_bytes, mode="w") as archive:
        for path in remote_external.iterdir():
            archive.add(path, arcname=path.name)

    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        if argv[0] == "ssh":
            assert "tar -cf -" in argv[-1]
            return types.SimpleNamespace(returncode=0, stdout=archive_bytes.getvalue(), stderr=b"")
        if argv[0] == "tar":
            destination = Path(argv[argv.index("-C") + 1])
            with tarfile.open(fileobj=io.BytesIO(kwargs["input"]), mode="r:") as archive:
                archive.extractall(destination)
            return types.SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
        raise AssertionError(f"unexpected command: {argv}")

    monkeypatch.setattr("autoresearch_gym.external.targets.subprocess.run", fake_run)
    target = SshTarget(
        TargetConfig(
            name="pytest-ssh",
            kind="ssh",
            host="example.invalid",
            remote_root=str(tmp_path / "remote"),
            path_style="posix",
        )
    )
    bundle = types.SimpleNamespace(run_id="run-1", external_dir=tmp_path / "local_external", session_dir=None)

    artifacts = target.fetch_artifacts(bundle)

    assert artifacts.root == tmp_path / "local_external"
    assert json.loads((artifacts.root / "train_result.json").read_text(encoding="utf-8"))["total_steps"] == 12
    assert (artifacts.root / "agent_checkpoint.pt").read_bytes() == b"checkpoint"
    assert [call[0] for call in calls] == ["ssh", "tar"]


def test_unitree_lower_level_cleanrl_seed_trains_evals_and_renders(tmp_path) -> None:
    from autoresearch_gym.external.cleanrl_backend import CleanRlExternalBackend
    from autoresearch_gym.tasks.unitree_g1_motion_mirror_v0 import seed_trainable_lower_level_cleanrl as g1_seed
    from autoresearch_gym.tasks.unitree_go2_rough_locomotion_v0 import seed_trainable_lower_level_cleanrl as go2_seed

    assert CleanRlExternalBackend().normalize_media(ArtifactSet(root=tmp_path)) == {"media_available": False}
    g1_benchmark = types.SimpleNamespace(
        env_kwargs={"render_mode": "rgb_array", "num_envs": 2, "steps_per_env_per_iteration": 4},
        train_episodes=1,
        train_seed=7,
        eval_seed_start=17,
        eval_episodes=1,
        max_steps=8,
        device="cpu",
    )
    agent, summary = g1_seed.train_agent(
        g1_benchmark,
        lambda control_type=None, reward_recipe=None: g1_seed.make_external_env(
            g1_benchmark,
            control_type=control_type,
            reward_recipe=reward_recipe,
        ),
        g1_seed.get_candidate(),
        "cpu",
    )

    assert summary["total_steps"] == 8
    assert summary["completed_episodes"] == summary["episodes_completed"]
    assert validate_summary(summary, require_gradient_updates=True) == []
    assert validate_records(summary["episode_records"]) == []
    assert summary["episode_records"][0]["record_type"] == "train_collection_window"
    g1_env = g1_seed.make_external_env(g1_benchmark)
    obs_start, _ = g1_env.reset(seed=123, options={"fixed_case": {"start_frame": 0, "end_frame": 4}})
    obs_contact, _ = g1_env.reset(seed=123, options={"fixed_case": {"start_frame": 120, "end_frame": 124}})
    assert not np.allclose(obs_start[:2], obs_contact[:2])
    g1_env.close()
    assert g1_seed.evaluate_agent(agent, g1_benchmark)["episodes"] == 1
    media = g1_seed.render_policy(agent, g1_benchmark, tmp_path / "g1-media")
    assert media["media_available"] is True
    assert Path(media["live_frame_path"]).exists()

    go2_benchmark = types.SimpleNamespace(env_kwargs={"render_mode": "rgb_array"}, max_steps=8)
    go2_env = go2_seed.make_external_env(go2_benchmark)
    obs, _ = go2_env.reset(seed=11)
    obs, reward, terminated, truncated, info = go2_env.step(go2_env.action_space.sample())
    assert obs.shape == go2_env.observation_space.shape
    assert isinstance(float(reward), float)
    assert "command_tracking_error" in info
    assert go2_env.render().ndim == 3
    assert terminated in {True, False}
    assert truncated in {True, False}
    go2_agent = go2_seed.Agent(int(go2_env.observation_space.shape[0]), int(go2_env.action_space.shape[0]))
    go2_env.close()
    go2_train_benchmark = types.SimpleNamespace(
        env_kwargs={"render_mode": "rgb_array", "num_envs": 2, "steps_per_env_per_iteration": 4},
        train_episodes=1,
        train_seed=13,
        eval_seed_start=9200,
        eval_episodes=1,
        max_steps=8,
        device="cpu",
    )
    _, go2_train_summary = go2_seed.train_agent(
        go2_train_benchmark,
        lambda control_type=None, reward_recipe=None: go2_seed.make_external_env(
            go2_train_benchmark,
            control_type=control_type,
            reward_recipe=reward_recipe,
        ),
        go2_seed.get_candidate(),
        "cpu",
    )
    assert go2_train_summary["completed_episodes"] == go2_train_summary["episodes_completed"]
    assert validate_summary(go2_train_summary, require_gradient_updates=True) == []
    assert validate_records(go2_train_summary["episode_records"]) == []
    go2_eval_benchmark = types.SimpleNamespace(
        env_kwargs={"render_mode": "rgb_array"},
        eval_seed_start=9200,
        eval_episodes=2,
        max_steps=4,
        primary_metric="eval_avg_return",
        primary_metric_mode="maximize",
    )
    go2_eval_cases = json.loads(
        Path("autoresearch_gym/tasks/unitree_go2_rough_locomotion_v0/eval_cases.json").read_text(encoding="utf-8")
    )["cases"]
    go2_eval = go2_seed.evaluate_agent(go2_agent, go2_eval_benchmark, eval_cases=go2_eval_cases)
    assert go2_eval["episodes"] == 2
    assert go2_eval["metric_source"] == "lower_level_rollout_reward"
    assert "success_rate" not in go2_eval
    assert [record["case_label"] for record in go2_eval["episode_records"]] == ["forward-rough", "turning-rough"]
    assert "success" not in go2_eval["episode_records"][0]


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


def test_inverted_pendulum_kinematic_render_fallback_uses_mujoco_state() -> None:
    fake_env = types.SimpleNamespace(
        spec=types.SimpleNamespace(id="InvertedPendulum-v5"),
        data=types.SimpleNamespace(qpos=np.asarray([0.25, 0.15], dtype=np.float64)),
    )

    frame = render_mujoco_kinematic_frame(fake_env, height=180, width=240)

    assert frame is not None
    assert frame.shape == (180, 240, 3)
    assert frame.dtype == np.uint8
    assert np.unique(frame.reshape(-1, 3), axis=0).shape[0] > 3


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


def test_all_bundled_seed_live_metrics_include_candidate_description(tmp_path) -> None:
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
    seed_paths = sorted(Path("autoresearch_gym/tasks").glob("*/seed_trainable*.py"))
    assert seed_paths

    for index, seed_path in enumerate(seed_paths):
        module_name = "autoresearch_gym_test_seed_" + "_".join(seed_path.with_suffix("").parts[-3:])
        spec = importlib.util.spec_from_file_location(module_name, seed_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        candidate = module.get_candidate()
        metadata = candidate_metadata(candidate)
        description = metadata.get("description")
        assert isinstance(description, str) and description.strip(), f"{seed_path} must expose a candidate description"

        writer = make_live_writer(tmp_path / f"session-{index}", "run-1", "tag-1", benchmark, candidate)
        assert writer is not None
        writer(status="running", episode_records=[], total_steps=1, last_metrics=None)
        payload = json.loads((tmp_path / f"session-{index}" / "live" / "current_run_metrics.json").read_text(encoding="utf-8"))
        assert payload["run"]["candidate"]["description"] == description


def test_external_live_status_preserves_candidate_description(tmp_path) -> None:
    from autoresearch_gym.external.runner import _write_external_live_status

    _write_external_live_status(
        tmp_path / "session",
        "run-1",
        "tag-1",
        "finished",
        {
            "benchmark": {
                "train_episodes": 1,
                "train_seconds": None,
                "budget_mode": "episodes",
                "eval_episodes": 1,
                "max_steps": 5,
                "render_mode": "rgb_array",
            },
            "candidate": {"description": "external candidate"},
            "train": {"total_steps": 5, "env_steps": 5, "episodes_completed": 1, "episode_batches": 1},
            "media": {},
        },
        {"episode_records": [], "last_metrics": {}},
    )
    payload = json.loads((tmp_path / "session" / "live" / "current_run_metrics.json").read_text(encoding="utf-8"))
    assert payload["run"]["candidate"]["description"] == "external candidate"


def test_unitree_dry_run_live_metrics_preserve_candidate_description(tmp_path) -> None:
    from autoresearch_gym.external.unitree_backend import _run_media

    session_dir = tmp_path / "session"
    bundle = {
        "run_id": "run-1",
        "tag": "tag-1",
        "task_family": "g1_motion_mirror",
        "dry_run": True,
        "required_paths": [],
        "benchmark": {"eval_episodes": 1},
        "candidate": {"description": "unitree candidate"},
        "session_dir": str(session_dir),
    }
    _run_media(bundle, tmp_path / "external")

    payload = json.loads((session_dir / "live" / "current_run_metrics.json").read_text(encoding="utf-8"))
    assert payload["run"]["candidate"]["description"] == "unitree candidate"


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
