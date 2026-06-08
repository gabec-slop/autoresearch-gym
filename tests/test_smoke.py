from __future__ import annotations

import argparse
import ast
import base64
import importlib.util
import io
import json
import shutil
import subprocess
import sys
import tarfile
import types
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
from PIL import Image

import autoresearch_gym  # noqa: F401
from autoresearch_gym import cli
from autoresearch_gym.external.base import ArtifactSet, RunBundle
from autoresearch_gym.runner.experiment import (
    BenchmarkSpec,
    SAMPLE_TRAJECTORY_SOURCE_CANDIDATE_PROVIDED,
    SAMPLE_TRAJECTORY_SOURCE_RUNNER_EVAL,
    SAMPLE_TRAJECTORY_SOURCE_VALUES,
    TrainProbeSpec,
    apply_headless_env_override,
    candidate_metadata,
    compact_status_line,
    env_kwargs_for_candidate,
    load_benchmark,
    make_compact_status_writer,
    make_live_writer,
    make_policy_probe_callback,
    normalize_train_summary_curve,
    normalize_run_tag,
    sampled_trajectory_source,
    should_sample_visual_episode,
    trajectory_sample_interval_seconds,
    utilization_flags,
    validate_sample_trajectory_source_contract,
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


def _mjlab_train_bridge_source() -> str:
    from autoresearch_gym.external import unitree_backend

    return unitree_backend._bridge_source_text(unitree_backend.MJLAB_TRAIN_BRIDGE)


def _mjlab_rollout_bridge_source() -> str:
    from autoresearch_gym.external import unitree_backend

    return unitree_backend._bridge_source_text(unitree_backend.MJLAB_ROLLOUT_BRIDGE)


def _decode_powershell_encoded_command(command: str) -> str:
    marker = " -EncodedCommand "
    assert marker in command
    encoded = command.split(marker, 1)[1].split()[0]
    return base64.b64decode(encoded).decode("utf-16-le")


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


def test_remote_in_process_extracts_summary_from_noisy_stdout() -> None:
    from autoresearch_gym.external.in_process import _extract_summary

    summary = {
        "run_id": "20260530-remote",
        "benchmark": {"name": "task"},
        "train": {"total_steps": 12},
        "eval": {"avg_return": 1.5},
    }

    assert _extract_summary("kernel warmup\n" + json.dumps({"not": "summary"}) + "\n" + json.dumps(summary)) == summary


def test_remote_in_process_windows_command_uses_target_python_and_paths() -> None:
    from autoresearch_gym.external.in_process import _remote_command
    from autoresearch_gym.external.targets import SshTarget, TargetConfig

    target = SshTarget(
        TargetConfig(
            name="windows_gpu",
            kind="ssh",
            host="user@example.invalid",
            remote_root="C:/code/autoresearch author's gym",
            path_style="windows",
            python="C:/Program Files/Python/python.exe",
        )
    )

    command = _remote_command(
        target,
        [
            "-m",
            "autoresearch_gym.cli",
            "run",
            "--benchmark",
            "C:/code/autoresearch-gym/tasks/benchmark.json",
            "--tag",
            "pass01 author's baseline",
        ],
    )

    assert command.startswith("powershell.exe -NoProfile -EncodedCommand ")
    assert "Set-Location" not in command
    script = _decode_powershell_encoded_command(command)
    assert "Set-Location -LiteralPath 'C:/code/autoresearch author''s gym'" in script
    assert "& 'C:\\Program Files\\Python\\python.exe'" in script
    assert "'-m' 'autoresearch_gym.cli' 'run'" in script
    assert "'--benchmark' 'C:/code/autoresearch-gym/tasks/benchmark.json'" in script
    assert "'--tag' 'pass01 author''s baseline'" in script


def test_remote_in_process_remote_cli_disables_dashboard() -> None:
    source = Path("autoresearch_gym/external/in_process.py").read_text(encoding="utf-8")

    assert '"--no-record",\n        "--no-dashboard",' in source


def test_remote_environment_fingerprint_windows_command_uses_encoded_python() -> None:
    from autoresearch_gym.external.remote_session import remote_environment_fingerprint_command
    from autoresearch_gym.external.targets import SshTarget, TargetConfig

    target = SshTarget(
        TargetConfig(
            name="windows_gpu",
            kind="ssh",
            host="user@example.invalid",
            remote_root="C:/code/autoresearch author's gym",
            path_style="windows",
            python="C:/Program Files/Python/python.exe",
        )
    )

    command = remote_environment_fingerprint_command(target)

    assert command.startswith("powershell.exe -NoProfile -EncodedCommand ")
    assert "git\",\"rev-parse" not in command
    assert "git status" not in command
    script = _decode_powershell_encoded_command(command)
    assert "Set-Location -LiteralPath 'C:/code/autoresearch author''s gym'" in script
    assert "$env:AR_CODE=" in script
    assert "& 'C:\\Program Files\\Python\\python.exe' -c" in script


def test_remote_session_doctor_windows_command_uses_encoded_python() -> None:
    from autoresearch_gym.external.remote_session import remote_session_doctor_command
    from autoresearch_gym.external.targets import SshTarget, TargetConfig

    target = SshTarget(
        TargetConfig(
            name="windows_gpu",
            kind="ssh",
            host="user@example.invalid",
            remote_root="C:/code/autoresearch author's gym",
            path_style="windows",
            python="C:/Program Files/Python/python.exe",
        )
    )

    command = remote_session_doctor_command(
        target,
        {
            "benchmark": "benchmark author's.json",
            "required_paths": [".external/unitree_rl_mjlab"],
            "packages": ["torch", "warp"],
        },
    )

    assert command.startswith("powershell.exe -NoProfile -EncodedCommand ")
    assert "unitree_rl_mjlab" not in command
    script = _decode_powershell_encoded_command(command)
    assert "Set-Location -LiteralPath 'C:/code/autoresearch author''s gym'" in script
    assert "$env:AR_SESSION_DOCTOR_CODE=" in script
    assert "$env:AR_SESSION_DOCTOR_PACKAGES=" in script
    assert "& 'C:\\Program Files\\Python\\python.exe' -c" in script


def test_remote_environment_fingerprint_posix_command_quotes_paths() -> None:
    from autoresearch_gym.external.remote_session import remote_environment_fingerprint_command
    from autoresearch_gym.external.targets import SshTarget, TargetConfig

    target = SshTarget(
        TargetConfig(
            name="linux_gpu",
            kind="ssh",
            host="user@example.invalid",
            remote_root="/tmp/autoresearch gym",
            path_style="posix",
            python="/opt/python env/bin/python",
        )
    )

    command = remote_environment_fingerprint_command(target)

    assert "cd '/tmp/autoresearch gym'" in command
    assert "'/opt/python env/bin/python' -c" in command
    assert "git\",\"rev-parse" not in command


def test_ssh_target_windows_run_quotes_python_module_and_mode(tmp_path, monkeypatch) -> None:
    from autoresearch_gym.external.base import CommandSpec
    from autoresearch_gym.external.targets import SshTarget, TargetConfig

    calls: list[list[str]] = []

    class FakeProcess:
        returncode = 0

        def poll(self):
            return 0

        def communicate(self):
            return ("{}", "")

    def fake_popen(argv, **kwargs):
        calls.append(list(argv))
        return FakeProcess()

    target = SshTarget(
        TargetConfig(
            name="windows_gpu",
            kind="ssh",
            host="user@example.invalid",
            remote_root="C:/code/autoresearch author's gym",
            path_style="windows",
            python="C:/Program Files/Python/python.exe",
        )
    )
    monkeypatch.setattr("autoresearch_gym.external.targets.subprocess.Popen", fake_popen)
    monkeypatch.setattr(target, "sync_live", lambda bundle: None)
    bundle = types.SimpleNamespace(
        run_id="run author's spaces",
        external_dir=tmp_path / "external artifacts",
        session_dir=None,
    )

    target.run(
        CommandSpec(
            argv=["python", "-m", "autoresearch_gym.external.unitree_backend", "--mode", "train"],
            label="train",
            timeout_seconds=1.0,
        ),
        bundle,
    )

    remote_command = calls[0][-1]
    assert remote_command.startswith("powershell.exe -NoProfile -EncodedCommand ")
    assert "Set-Location" not in remote_command
    script = _decode_powershell_encoded_command(remote_command)
    assert "Set-Location -LiteralPath 'C:/code/autoresearch author''s gym'" in script
    assert "& 'C:\\Program Files\\Python\\python.exe'" in script
    assert "-m 'autoresearch_gym.external.unitree_backend'" in script
    assert "--mode 'train'" in script
    assert (
        "--bundle 'C:/code/autoresearch author's gym\\autoresearch_runs\\external_remote\\run author's spaces\\external\\bundle.json'"
        not in script
    )
    assert (
        "--bundle 'C:/code/autoresearch author''s gym\\autoresearch_runs\\external_remote\\run author''s spaces\\external\\bundle.json'"
        in script
    )


def test_ssh_target_windows_utility_commands_use_encoded_powershell(monkeypatch) -> None:
    from autoresearch_gym.external.base import CommandResult
    from autoresearch_gym.external.targets import SshTarget, TargetConfig

    commands: list[str] = []

    def fake_ssh(command: str, timeout: float = 30.0):
        commands.append(command)
        return CommandResult(returncode=0, stdout="ok\n", stderr="")

    target = SshTarget(
        TargetConfig(
            name="windows_gpu",
            kind="ssh",
            host="user@example.invalid",
            remote_root="C:/code/autoresearch author's gym",
            path_style="windows",
            python="C:/Program Files/Python/python.exe",
        )
    )
    monkeypatch.setattr(target, "_ssh", fake_ssh)

    target.ensure_remote_dir("C:/tmp/author's live")
    tar_command = target._remote_tar_command("C:/tmp/author's final")
    preflight = target.preflight(types.SimpleNamespace())

    mkdir_script = _decode_powershell_encoded_command(commands[0])
    tar_script = _decode_powershell_encoded_command(tar_command)
    preflight_script = _decode_powershell_encoded_command(commands[1])
    assert preflight.ok is True
    assert "New-Item -ItemType Directory -Force -Path 'C:/tmp/author''s live'" in mkdir_script
    assert "Test-Path -LiteralPath 'C:/tmp/author''s final'" in tar_script
    assert "tar -cf - -C 'C:/tmp/author''s final' ." in tar_script
    assert "$root='C:/code/autoresearch author''s gym'" in preflight_script
    assert "Set-Location -LiteralPath $root" in preflight_script
    assert "& 'C:\\Program Files\\Python\\python.exe' --version" in preflight_script


def test_ssh_target_windows_remote_path_info_uses_encoded_powershell(monkeypatch) -> None:
    from autoresearch_gym.external.base import CommandResult
    from autoresearch_gym.external.targets import SshTarget, TargetConfig

    commands: list[str] = []

    def fake_ssh(command: str, timeout: float = 30.0):
        commands.append(command)
        return CommandResult(
            returncode=0,
            stdout=json.dumps({"exists": True, "length": 12, "age_seconds": 3.5}) + "\n",
            stderr="",
        )

    target = SshTarget(
        TargetConfig(
            name="windows_gpu",
            kind="ssh",
            host="user@example.invalid",
            remote_root="C:/code/autoresearch author's gym",
            path_style="windows",
        )
    )
    monkeypatch.setattr(target, "_ssh", fake_ssh)

    info = target.remote_path_info("C:/tmp/status author's.log")

    script = _decode_powershell_encoded_command(commands[0])
    assert info["exists"] is True
    assert info["age_seconds"] == 3.5
    assert "$payload = if (Test-Path -LiteralPath $p)" in script
    assert "Test-Path -LiteralPath $p" in script
    assert "$p='C:/tmp/status author''s.log'" in script
    assert "$payload | ConvertTo-Json -Compress" in script


def test_ssh_target_windows_terminate_processes_matching_is_guarded_and_encoded(monkeypatch) -> None:
    from autoresearch_gym.external.base import CommandResult
    from autoresearch_gym.external.targets import SshTarget, TargetConfig

    commands: list[str] = []

    def fake_ssh(command: str, timeout: float = 30.0):
        commands.append(command)
        return CommandResult(
            returncode=0,
            stdout=json.dumps([{"pid": 123, "command": "python -m autoresearch_gym.cli run --tag pass02"}]) + "\n",
            stderr="",
        )

    target = SshTarget(
        TargetConfig(
            name="windows_gpu",
            kind="ssh",
            host="user@example.invalid",
            remote_root="C:/code/autoresearch author's gym",
            path_style="windows",
        )
    )
    monkeypatch.setattr(target, "_ssh", fake_ssh)

    with pytest.raises(ValueError, match="at least two match terms"):
        target.terminate_processes_matching(["pass02"])
    killed = target.terminate_processes_matching(["autoresearch_gym.cli", "--tag", "pass02"])

    script = _decode_powershell_encoded_command(commands[0])
    assert killed == [{"pid": 123, "command": "python -m autoresearch_gym.cli run --tag pass02"}]
    assert "Get-CimInstance Win32_Process" in script
    assert "Stop-Process -Id $_.ProcessId -Force" in script
    assert "$terms=@('autoresearch_gym.cli','--tag','pass02')" in script


def test_remote_in_process_stale_status_uses_remote_file_age() -> None:
    from autoresearch_gym.external.in_process import _remote_status_stale_info

    class FakeTarget:
        def remote_path_info(self, remote_status_file):
            assert remote_status_file == "C:/session/live/status.log"
            return {"exists": True, "age_seconds": 901.0, "length": 100}

    stale = _remote_status_stale_info(FakeTarget(), "C:/session/live/status.log", stale_seconds=600.0)

    assert stale is not None
    assert stale["age_seconds"] == 901.0
    assert stale["stale_seconds"] == 600.0


def test_remote_in_process_localizes_fetched_checkpoint_path(tmp_path) -> None:
    from autoresearch_gym.external.in_process import _localize_fetched_summary_artifacts

    local_run_dir = tmp_path / "runs" / "run-1"
    local_run_dir.mkdir(parents=True)
    checkpoint = local_run_dir / "agent_checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    summary = {"artifacts": {"checkpoint_path": r"C:\remote\run-1\agent_checkpoint.pt"}}

    _localize_fetched_summary_artifacts(summary, local_run_dir)

    assert summary["artifacts"]["checkpoint_path"] == str(checkpoint.resolve())


def test_remote_in_process_sync_fetches_live_sampled_rollout_refs(tmp_path) -> None:
    from autoresearch_gym.external.in_process import _sync_remote_session

    session_name = "20260530-panda-remote"
    remote_session = f"C:/code/autoresearch-gym/autoresearch_runs/sessions/{session_name}"
    remote_live = tmp_path / "remote" / "live"
    remote_trajectory = remote_live / "trajectories" / "run-1" / "sample_000001"
    remote_trajectory.mkdir(parents=True)
    windows_manifest = f"{remote_session}\\live\\trajectories\\run-1\\sample_000001\\manifest.json"
    windows_frame = f"{remote_session}\\live\\trajectories\\run-1\\sample_000001\\frame_0000.jpg"
    windows_world_feed = f"{remote_session}\\live\\trajectories\\run-1\\sample_000001\\feeds\\frame_0000_world.jpg"
    windows_wrist_feed = f"{remote_session}\\live\\trajectories\\run-1\\sample_000001\\feeds\\frame_0000_wrist.jpg"
    (remote_live / "current_run_metrics.json").write_text(
        json.dumps(
            {
                "run": {"run_id": "run-1", "visual": {"trajectory_manifest_path": windows_manifest}},
                "visual": {
                    "trajectory_manifest_path": windows_manifest,
                    "trajectory_latest_frame_path": windows_frame,
                    "sampled_status": "completed",
                },
            }
        ),
        encoding="utf-8",
    )
    (remote_live / "status.log").write_text("st=run step=1\n", encoding="utf-8")
    (remote_live / "control.json").write_text("{}", encoding="utf-8")
    (remote_trajectory / "manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "frames": [windows_frame],
                "steps": [
                    {
                        "index": 0,
                        "frame_path": windows_frame,
                        "feeds": {"world": windows_world_feed, "wrist": windows_wrist_feed},
                        "policy_feeds": ["wrist"],
                    }
                ],
                "feed_specs": {"wrist": {"role": "policy_input"}},
                "latest_frame_path": windows_frame,
                "frame_count": 1,
            }
        ),
        encoding="utf-8",
    )
    (remote_trajectory / "frame_0000.jpg").write_bytes(b"fake image")
    (remote_trajectory / "feeds").mkdir()
    (remote_trajectory / "feeds" / "frame_0000_world.jpg").write_bytes(b"fake world")
    (remote_trajectory / "feeds" / "frame_0000_wrist.jpg").write_bytes(b"fake wrist")
    fetched_suffixes = []

    class FakeTarget:
        def remote_join(self, *parts):
            return "/".join(str(part).replace("\\", "/").strip("/") for part in parts)

        def fetch_remote_file(self, remote_file, local_file, *, timeout=15.0):
            del timeout
            suffix = str(remote_file).split("/live/", 1)[1]
            fetched_suffixes.append(suffix)
            source = remote_live / suffix
            if not source.exists():
                return False
            local_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, local_file)
            return True

    session_dir = tmp_path / session_name
    _sync_remote_session(FakeTarget(), remote_session, session_dir)

    metrics = json.loads((session_dir / "live" / "current_run_metrics.json").read_text(encoding="utf-8"))
    manifest_path = metrics["visual"]["trajectory_manifest_path"]
    latest_frame_path = metrics["visual"]["trajectory_latest_frame_path"]
    assert "C:" not in manifest_path
    assert manifest_path.endswith(f"{session_name}/live/trajectories/run-1/sample_000001/manifest.json")
    assert latest_frame_path.endswith(f"{session_name}/live/trajectories/run-1/sample_000001/frame_0000.jpg")
    manifest = json.loads((session_dir / "live" / "trajectories" / "run-1" / "sample_000001" / "manifest.json").read_text())
    assert manifest["frames"][0].endswith(f"{session_name}/live/trajectories/run-1/sample_000001/frame_0000.jpg")
    assert manifest["steps"][0]["feeds"]["world"].endswith(
        f"{session_name}/live/trajectories/run-1/sample_000001/feeds/frame_0000_world.jpg"
    )
    assert manifest["steps"][0]["feeds"]["wrist"].endswith(
        f"{session_name}/live/trajectories/run-1/sample_000001/feeds/frame_0000_wrist.jpg"
    )
    assert (session_dir / "live" / "trajectories" / "run-1" / "sample_000001" / "frame_0000.jpg").read_bytes() == b"fake image"
    assert (
        session_dir / "live" / "trajectories" / "run-1" / "sample_000001" / "feeds" / "frame_0000_world.jpg"
    ).read_bytes() == b"fake world"
    assert (
        session_dir / "live" / "trajectories" / "run-1" / "sample_000001" / "feeds" / "frame_0000_wrist.jpg"
    ).read_bytes() == b"fake wrist"

    fetched_suffixes.clear()
    _sync_remote_session(FakeTarget(), remote_session, session_dir)
    assert "trajectories/run-1/sample_000001/feeds/frame_0000_world.jpg" not in fetched_suffixes
    assert "trajectories/run-1/sample_000001/feeds/frame_0000_wrist.jpg" not in fetched_suffixes


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
    assert int(g1_recipe["runner"]["save_interval"]) > 0
    assert int(g1_recipe["runner"]["probe_interval_iterations"]) >= 0
    assert int(g1_recipe["runner"]["sample_rollout_frame_count"]) >= 0
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
    assert go2_staged_recipe["terrain"]["max_init_terrain_level"] > 0
    assert go2_staged_recipe["runner"]["sample_trajectory_source"] == SAMPLE_TRAJECTORY_SOURCE_CANDIDATE_PROVIDED
    assert go2_staged_recipe["event_overrides"]["push_robot"]["enabled"] is False
    assert go2_staged_recipe["twist_command"]["ranges"]["lin_vel_x"][1] > 0.0
    assert go2_staged_recipe["curriculum_plan"][0]["name"] == "stand_and_creep"
    velocity_stages = go2_staged_recipe["curriculum_overrides"]["command_vel"]["params"]["velocity_stages"]
    stage_steps = [stage["step"] for stage in velocity_stages]
    assert stage_steps[0] == -1
    assert stage_steps == sorted(stage_steps)


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
    assert _mjlab_probe_interval_iterations(recipe, {"benchmark": {"train_probe_interval_seconds": 120.0}}, 692) == 67

    monkeypatch.setenv("UNITREE_MJLAB_TARGET_POLICY_PROBES", "10")
    assert _mjlab_probe_interval_iterations(recipe, bundle, 692) == 70


def test_unitree_mjlab_train_seconds_timeout_writes_time_budget_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from autoresearch_gym.external import unitree_backend

    out_dir = tmp_path / "external"
    out_dir.mkdir()
    run_dir = tmp_path / "mjlab_run"
    run_dir.mkdir()
    (run_dir / "model_4.pt").write_text("checkpoint", encoding="utf-8")
    (out_dir / "train_result_partial.json").write_text(
        json.dumps(
            {
                "episode_records": [
                    {
                        "record_type": "train_collection_window",
                        "episode": 1,
                        "return": 1.0,
                        "length": 24.0,
                        "step": 24576,
                        "episodes_in_window": 1024,
                    },
                    {
                        "record_type": "train_collection_window",
                        "episode": 2,
                        "return": 2.0,
                        "length": 30.0,
                        "step": 49152,
                        "episodes_in_window": 1024,
                    },
                    {
                        "record_type": "policy_probe",
                        "episode": 2,
                        "return": -1.0,
                        "length": 120.0,
                        "step": 49152,
                    },
                ],
                "total_steps": 49152,
                "last_metrics": {"train_mean_reward": 2.0},
            }
        ),
        encoding="utf-8",
    )

    seen_timeout = {}

    def fake_run_subprocess(argv, **kwargs):
        seen_timeout["timeout"] = kwargs.get("timeout")
        raise subprocess.TimeoutExpired(cmd=argv, timeout=kwargs.get("timeout") or 30.0, output="stdout", stderr="stderr")

    monkeypatch.setattr(unitree_backend, "_run_subprocess", fake_run_subprocess)
    monkeypatch.setattr(unitree_backend, "_unitree_root", lambda bundle: tmp_path)
    monkeypatch.setattr(unitree_backend, "_mjlab_python", lambda bundle: "python")
    monkeypatch.setattr(unitree_backend, "_write_train_script", lambda path: path / "mjlab_train_bridge.py")
    monkeypatch.setattr(unitree_backend, "_write_rollout_script", lambda path: path / "mjlab_rollout_bridge.py")
    monkeypatch.setattr(unitree_backend, "_find_latest_run_dir", lambda bundle, started_at: run_dir)
    monkeypatch.setattr(
        unitree_backend,
        "_terminate_local_processes_matching",
        lambda required_terms: [{"pid": 123, "command": "mjlab_train_bridge.py pytest-timeout"}],
    )

    bundle = {
        "run_id": "pytest-timeout",
        "task_family": "go2_rough_locomotion",
        "benchmark": {
            "env_kwargs": {"task_id": "Unitree-Go2-Rough", "num_envs": 1024},
            "train_seconds": 30.0,
            "max_steps": 200,
        },
        "candidate": {"recipe": {"runner": {"num_envs": 1024, "num_steps_per_env": 24, "max_iterations": 100}}},
    }

    unitree_backend._run_mjlab_train(bundle, out_dir)
    summary = json.loads((out_dir / "train_result.json").read_text(encoding="utf-8"))

    assert summary["stop_reason"] == "time_budget_exhausted"
    assert seen_timeout["timeout"] == 30.0
    assert summary["total_steps"] == 49152
    assert summary["episode_batches"] == 2
    assert summary["gradient_updates"] == 2
    assert summary["episodes_completed"] == 2048
    assert summary["mjlab"]["learning_iterations"] == 2
    assert summary["mjlab"]["target_learning_iterations"] == 100
    assert summary["mjlab"]["terminated_processes"][0]["pid"] == 123
    assert summary["last_metrics"]["time_budget_exhausted"] == 1.0
    assert (out_dir / "agent_checkpoint.pt").read_text(encoding="utf-8") == "checkpoint"


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


def test_dashboard_resolves_absolute_session_local_artifacts(tmp_path) -> None:
    root = tmp_path / "dashboard-root"
    root.mkdir()
    session_dir = tmp_path / "scratch" / "sessions" / "trajectory-smoke"
    frame_path = session_dir / "live" / "trajectories" / "run" / "episode_000001" / "frame_0000.jpg"
    frame_path.parent.mkdir(parents=True)
    frame_path.write_bytes(b"jpeg")
    (session_dir / "session.json").write_text("{}", encoding="utf-8")

    assert cli.resolve_dashboard_session_dir(root, str(session_dir)) == session_dir.resolve()
    assert cli.resolve_dashboard_artifact_path(root, str(session_dir), str(frame_path)) == frame_path.resolve()

    outside_path = tmp_path / "outside.jpg"
    outside_path.write_bytes(b"nope")
    with pytest.raises(ValueError, match="escapes"):
        cli.resolve_dashboard_artifact_path(root, str(session_dir), str(outside_path))

def test_dashboard_sampled_trajectory_refreshes_when_frame_paths_change() -> None:
    source = Path("dashboard/index.html").read_text(encoding="utf-8")

    assert "/_autoresearch/artifact" in source
    assert "cacheBustedUrl(artifactUrl(rawManifestPath))" in source
    assert "let trajectoryFrameSignature = \"\";" in source
    assert "const frameSignature = [" in source
    assert "...frames," in source
    assert "...steps.map((step) => JSON.stringify(step.feeds || {}))," in source
    assert "trajectoryFrameSignature !== frameSignature" in source
    assert "trajectoryFrameIndex = 0;" in source


def test_dashboard_buckets_previous_best_return_ghost_line() -> None:
    source = Path("dashboard/index.html").read_text(encoding="utf-8")
    ghost_block_start = source.index("drawSeries(returnCtx, chart, ghostValues")
    ghost_block = source[ghost_block_start : source.index("});", ghost_block_start)]

    assert "bucketDense: true" in ghost_block
    assert "maxRawPoints: 180" in ghost_block


def test_unitree_mjlab_train_context_probes_run_out_of_process() -> None:
    train_bridge = _mjlab_train_bridge_source()

    assert "def _run_train_context_probe_subprocess" in train_bridge
    assert "--probe-checkpoint" in train_bridge
    assert "if args.probe_checkpoint" in train_bridge
    assert "train_script=Path(__file__).resolve()" in train_bridge
    assert "_run_train_context_probe_subprocess(" in train_bridge
    assert "def _probe_subprocess_env" in train_bridge
    assert 'env.setdefault("PYTHONIOENCODING", "utf-8")' in train_bridge
    assert "_run_checkpoint_probe(" in train_bridge
    assert "policy_probes\" / \"logs" in train_bridge
    assert "error=no_rollout" in train_bridge
    assert "env_cfg.scene.num_envs = 1 if frame_dir is not None else requested_num_envs" in train_bridge
    assert "command_metrics" in train_bridge
    assert "get_command(name)" in train_bridge
    assert "except PermissionError" in train_bridge
    assert "monitor_errors.log" in train_bridge


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


def test_unitree_eval_honors_checkpoint_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from autoresearch_gym.external import unitree_backend

    override_checkpoint = tmp_path / "model_737.pt"
    override_checkpoint.write_text("checkpoint", encoding="utf-8")
    out_dir = tmp_path / "external"
    out_dir.mkdir()
    seen = {}

    def fake_run_mjlab_rollout(bundle, out_dir_arg, checkpoint, *, mode, frame_dir=None):
        seen["checkpoint"] = checkpoint
        seen["mode"] = mode
        (Path(out_dir_arg) / "eval_result.json").write_text(
            json.dumps({"avg_return": 1.0, "episode_records": []}),
            encoding="utf-8",
        )

    monkeypatch.setattr(unitree_backend, "_run_mjlab_rollout", fake_run_mjlab_rollout)
    bundle = {
        "run_id": "pytest-eval-checkpoint",
        "dry_run": False,
        "checkpoint": str(override_checkpoint),
        "task_family": "go2_rough_locomotion",
        "required_paths": [],
        "benchmark": {"env_kwargs": {"task_id": "Unitree-Go2-Rough"}, "eval_episodes": 1, "max_steps": 200},
    }

    unitree_backend._run_eval(bundle, out_dir)

    assert seen == {"checkpoint": override_checkpoint, "mode": "eval"}


def test_unitree_mjlab_train_bridge_compiles_with_recipe_overrides() -> None:
    rollout_bridge = _mjlab_rollout_bridge_source()
    train_bridge = _mjlab_train_bridge_source()

    compile(rollout_bridge, "mjlab_rollout_bridge.py", "exec")
    compile(train_bridge, "mjlab_train_bridge.py", "exec")
    assert "--render-width" in rollout_bridge
    assert "--render-height" in rollout_bridge
    assert "--recipe-json" in train_bridge
    assert "reward_weights" in train_bridge
    assert "curriculum_overrides" in train_bridge
    assert "train_result_partial.json" in train_bridge
    assert "policy_probe_records.jsonl" in train_bridge
    assert "current_run_metrics.json" in train_bridge
    assert "--sample-rollout-frame-count" in train_bridge
    assert "--sample-trajectory-source" in train_bridge
    assert "--probe-render-width" in train_bridge
    assert "--probe-render-height" in train_bridge
    assert "DEFAULT_TRAJECTORY_PLAYBACK_FPS" in train_bridge


def test_unitree_backend_copies_packaged_mjlab_bridge_files(tmp_path: Path) -> None:
    from autoresearch_gym.external import unitree_backend

    rollout_path = unitree_backend._write_rollout_script(tmp_path)
    train_path = unitree_backend._write_train_script(tmp_path)

    assert rollout_path.read_text(encoding="utf-8") == _mjlab_rollout_bridge_source()
    assert train_path.read_text(encoding="utf-8") == _mjlab_train_bridge_source()


def test_unitree_backend_normalizes_media_frames_to_panda_mujoco_size(tmp_path: Path) -> None:
    from autoresearch_gym.external import unitree_backend

    frame_path = tmp_path / "frame.jpg"
    Image.fromarray(np.full((12, 16, 3), 96, dtype=np.uint8)).save(frame_path)

    normalized = unitree_backend._normalize_dashboard_frame(frame_path)

    assert normalized == frame_path
    with Image.open(frame_path) as image:
        assert image.size == (unitree_backend.DASHBOARD_FRAME_WIDTH, unitree_backend.DASHBOARD_FRAME_HEIGHT)
    assert unitree_backend.DASHBOARD_FRAME_WIDTH == 720
    assert unitree_backend.DASHBOARD_FRAME_HEIGHT == 480


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
        if source_name and source_name != SAMPLE_TRAJECTORY_SOURCE_RUNNER_EVAL:
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
        assert manifest["width"] == 720
        assert manifest["height"] == 480
        assert manifest["metadata"]["seed_path"] == str(seed_path)
        first_frame = Path(manifest["frames"][0])
        if not first_frame.is_absolute():
            first_frame = Path.cwd() / first_frame
        with Image.open(first_frame) as image:
            assert image.size == (720, 480)


def test_sample_trajectory_source_schema_rejects_unknown_values() -> None:
    candidate = {"recipe": {"runner": {"sample_trajectory_source": "policy_eval_rollout"}}}

    with pytest.raises(ValueError, match="Unknown sample_trajectory_source"):
        sampled_trajectory_source(candidate)

    with pytest.raises(ValueError, match="_answer_sampled_trajectory_request"):
        validate_sample_trajectory_source_contract(
            {"recipe": {"runner": {"sample_trajectory_source": SAMPLE_TRAJECTORY_SOURCE_CANDIDATE_PROVIDED}}},
            types.SimpleNamespace(),
        )


def test_bundled_seed_sample_trajectory_sources_match_schema() -> None:
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
        assert isinstance(runner, dict), f"{seed_path} sample_trajectory_source must live under RECIPE['runner']"
        source_name = runner.get("sample_trajectory_source")
        assert source_name in SAMPLE_TRAJECTORY_SOURCE_VALUES, f"{seed_path} uses unknown source {source_name!r}"
        if source_name == SAMPLE_TRAJECTORY_SOURCE_CANDIDATE_PROVIDED:
            assert (
                "_answer_sampled_trajectory_request" in source or "must run through execution_backend" in source
            ), f"{seed_path} declares candidate_provided without an in-process or external producer"


def test_tqc_her_ee_seed_owns_sampled_trajectory_generation() -> None:
    from autoresearch_gym.tasks.panda_pick_and_place_mjwarp_v0 import seed_trainable_tqc_her_ee

    candidate = seed_trainable_tqc_her_ee.get_candidate()
    assert sampled_trajectory_source(candidate) == SAMPLE_TRAJECTORY_SOURCE_CANDIDATE_PROVIDED
    assert callable(getattr(seed_trainable_tqc_her_ee, "_answer_sampled_trajectory_request", None))


def test_tqc_her_ee_sampled_trajectory_steps_tool_actions(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from autoresearch_gym.tasks.panda_pick_and_place_mjwarp_v0 import seed_trainable_tqc_her_ee

    class FakeToolEnv(gym.Env):
        def __init__(self) -> None:
            self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(43,), dtype=np.float32)
            self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
            self.steps = 0
            self.actions: list[np.ndarray] = []

        def reset(self, *, seed=None, options=None):
            self.steps = 0
            return np.zeros(43, dtype=np.float32), {}

        def step(self, action):
            action = np.asarray(action, dtype=np.float32)
            assert action.shape == (4,)
            self.actions.append(action.copy())
            self.steps += 1
            return np.zeros(43, dtype=np.float32), -1.0, self.steps >= 3, False, {}

        def render(self):
            return np.full((8, 8, 3), self.steps, dtype=np.uint8)

    env = FakeToolEnv()

    def env_factory(control_type=None, reward_recipe=None):
        return env

    benchmark = BenchmarkSpec(
        name="tqc-sampled-trajectory",
        env_id="FakeTool-v0",
        env_kwargs={"render_mode": "rgb_array"},
        train_episodes=1,
        train_seconds=None,
        eval_episodes=1,
        max_steps=4,
        reward_type=None,
        render_mode="rgb_array",
        primary_metric="eval_avg_return",
        primary_metric_mode="maximize",
        train_seed=1,
        eval_seed_start=100,
        device="cpu",
        eval_case_bank=None,
        train_probe=TrainProbeSpec(enabled=False),
    )
    candidate = seed_trainable_tqc_her_ee.get_candidate()
    writer = make_live_writer(tmp_path / "session", "run-tqc", "tag-tqc", benchmark, candidate)
    assert writer is not None
    response = writer(
        status="running",
        episode_records=[],
        total_steps=0,
        last_metrics={},
        current_episode=1,
    )
    agent = seed_trainable_tqc_her_ee.Agent(43, torch.device("cpu"))
    seed_trainable_tqc_her_ee._answer_sampled_trajectory_request(
        response,
        live_callback=writer,
        agent=agent,
        env_factory=env_factory,
        benchmark=benchmark,
        records=[],
        global_step=0,
        last_metrics={},
        elapsed_seconds=0.0,
    )
    assert env.actions, "sampled trajectory did not step the fake env"
    metrics = json.loads((tmp_path / "session" / "live" / "current_run_metrics.json").read_text(encoding="utf-8"))
    manifest_path = Path(metrics["visual"]["trajectory_manifest_path"])
    if not manifest_path.is_absolute():
        manifest_path = Path.cwd() / manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["source"] == SAMPLE_TRAJECTORY_SOURCE_CANDIDATE_PROVIDED
    assert manifest["status"] == "completed"
    assert manifest["frame_count"] >= 2


def test_panda_mjwarp_benchmarks_keep_train_probes_enabled() -> None:
    task_dir = Path("autoresearch_gym/tasks/panda_pick_and_place_mjwarp_v0")
    for benchmark_path in task_dir.glob("benchmark*.json"):
        payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
        train_probe = payload.get("train_probe")
        assert isinstance(train_probe, dict), f"{benchmark_path} must declare train_probe"
        assert train_probe.get("enabled") is True, f"{benchmark_path} must not disable train probes"


def test_seed_artifact_smoke_cases_cover_every_bundled_seed_trainable() -> None:
    from scripts.smoke_seed_artifacts import SEED_CASES

    task_seeds = set(Path("autoresearch_gym/tasks").glob("*/seed_trainable*.py"))
    smoke_seeds = {Path(case.seed) for case in SEED_CASES}

    assert smoke_seeds == task_seeds
    for case in SEED_CASES:
        assert Path(case.benchmark).exists(), f"{case.name} benchmark is missing"


def test_inverted_pendulum_smoke_does_not_use_kinematic_visual_fallback(monkeypatch) -> None:
    from scripts import smoke_seed_artifacts

    runner_source = Path("autoresearch_gym/runner/experiment.py").read_text(encoding="utf-8")
    case = next(case for case in smoke_seed_artifacts.SEED_CASES if case.name == "inverted-pendulum")

    assert "render_mujoco_kinematic_frame" not in runner_source
    assert "MUJOCO_KINEMATIC_RENDER_ENV_IDS" not in runner_source

    monkeypatch.setattr(smoke_seed_artifacts.sys, "platform", "darwin")
    monkeypatch.delenv("AUTORESEARCH_SMOKE_VISUALS", raising=False)
    assert smoke_seed_artifacts.visual_artifact_smoke_enabled(case) is False

    monkeypatch.setenv("AUTORESEARCH_SMOKE_VISUALS", "1")
    assert smoke_seed_artifacts.visual_artifact_smoke_enabled(case) is True


def test_panda_mjwarp_vectorized_seed_answers_sampled_trajectory_requests(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from autoresearch_gym.tasks.panda_pick_and_place_mjwarp_v0 import seed_trainable

    class FakeVectorEnv:
        def __init__(self, num_envs: int) -> None:
            self.num_envs = num_envs
            self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        def reset(self, seed: int | None = None):
            return np.zeros((self.num_envs, 3), dtype=np.float32)

        def step(self, actions):
            obs = np.zeros((self.num_envs, 3), dtype=np.float32)
            rewards = np.zeros(self.num_envs, dtype=np.float32)
            dones = np.zeros(self.num_envs, dtype=bool)
            infos = {
                "ee_to_cube_distance": np.full(self.num_envs, 0.5, dtype=np.float32),
                "cube_to_goal_distance": np.full(self.num_envs, 0.4, dtype=np.float32),
                "cube_lift_height": np.zeros(self.num_envs, dtype=np.float32),
                "near_cube": np.zeros(self.num_envs, dtype=bool),
                "gripper_closed_near_cube": np.zeros(self.num_envs, dtype=bool),
                "lifted": np.zeros(self.num_envs, dtype=bool),
                "placed_success": np.zeros(self.num_envs, dtype=bool),
                "is_success": np.zeros(self.num_envs, dtype=bool),
            }
            return obs, rewards, dones, infos

        def reset_worlds(self, dones):
            return np.zeros((self.num_envs, 3), dtype=np.float32)

        def close(self):
            pass

    class FakePandaEnv(gym.Env):
        metadata = {"render_modes": ["rgb_array"]}

        def __init__(self) -> None:
            self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
            self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
            self.steps = 0

        def make_vectorized(self, num_envs: int, seed: int):
            return FakeVectorEnv(num_envs)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.steps = 0
            return np.zeros(3, dtype=np.float32), {}

        def step(self, action):
            self.steps += 1
            info = {
                "ee_to_cube_distance": 0.5,
                "cube_to_goal_distance": 0.4,
                "cube_lift_height": 0.0,
                "near_cube": False,
                "gripper_closed_near_cube": False,
                "lifted": False,
                "placed_success": False,
                "is_success": False,
            }
            return np.zeros(3, dtype=np.float32), 0.0, self.steps >= 4, False, info

        def render(self, *args, **kwargs):
            return np.full((8, 8, 3), self.steps, dtype=np.uint8)

    benchmark = BenchmarkSpec(
        name="panda-mjwarp-test",
        env_id="FakePanda-v0",
        env_kwargs={"render_mode": "rgb_array", "backend": "mujoco_warp", "num_envs": 2, "steps_per_env_per_iteration": 2},
        train_episodes=1,
        train_seconds=None,
        eval_episodes=1,
        max_steps=4,
        reward_type=None,
        render_mode="rgb_array",
        primary_metric="eval_avg_return",
        primary_metric_mode="maximize",
        train_seed=1,
        eval_seed_start=100,
        device="cpu",
        eval_case_bank=None,
        train_probe=TrainProbeSpec(enabled=False),
    )
    candidate = seed_trainable.get_candidate()
    writer = make_live_writer(tmp_path / "session", "run-1", "tag-1", benchmark, candidate)
    assert writer is not None

    def env_factory(control_type=None, reward_recipe=None):
        return FakePandaEnv()

    _, summary = seed_trainable.train_agent(
        benchmark,
        env_factory,
        candidate,
        torch.device("cpu"),
        live_callback=writer,
    )

    metrics = json.loads((tmp_path / "session" / "live" / "current_run_metrics.json").read_text(encoding="utf-8"))
    manifest_path = Path(metrics["visual"]["trajectory_manifest_path"])
    if not manifest_path.is_absolute():
        manifest_path = Path.cwd() / manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert summary["vectorized_backend"] == "mujoco_warp_vectorized"
    assert summary["num_envs"] == 2
    assert summary["vector_envs"] == 2
    assert summary["last_metrics"]["num_envs"] == 2.0
    assert manifest["source"] == SAMPLE_TRAJECTORY_SOURCE_CANDIDATE_PROVIDED
    assert manifest["status"] == "completed"
    assert manifest["frame_count"] >= 2


def test_panda_mjwarp_render_policy_frame_falls_back_after_width_typeerror() -> None:
    from autoresearch_gym.tasks.panda_pick_and_place_mjwarp_v0.seed_trainable import _render_policy_frame

    class FakeRenderEnv:
        def __init__(self) -> None:
            self.calls: list[dict[str, int]] = []

        def render(self, **kwargs):
            self.calls.append(dict(kwargs))
            if kwargs:
                raise TypeError("render() got an unexpected keyword argument 'width'")
            return np.full((5, 6, 3), 17, dtype=np.uint8)

    class FakeWrappedEnv:
        def __init__(self) -> None:
            self.unwrapped = FakeRenderEnv()

    env = FakeWrappedEnv()
    frame = _render_policy_frame(env)

    assert frame is not None
    assert frame.shape == (5, 6, 3)
    assert frame.dtype == np.uint8
    assert env.unwrapped.calls == [{"width": 720, "height": 480}, {}]


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


def test_ssh_target_windows_fetch_artifacts_skips_archive_first(tmp_path, monkeypatch) -> None:
    from autoresearch_gym.external.targets import SshTarget, TargetConfig

    archive_calls: list[str] = []
    file_calls: list[str] = []
    dir_calls: list[str] = []
    target = SshTarget(
        TargetConfig(
            name="pytest-windows-ssh",
            kind="ssh",
            host="example.invalid",
            remote_root="C:/code/autoresearch gym",
            path_style="windows",
        )
    )

    def fake_archive(remote_dir, local_dir, *, timeout):
        archive_calls.append(remote_dir)
        return True

    def fake_file(remote_file, local_file, **kwargs):
        file_calls.append(remote_file)
        return True

    def fake_dir(remote_dir, local_dir, **kwargs):
        dir_calls.append(remote_dir)
        return True

    monkeypatch.setattr(target, "_fetch_remote_dir_archive", fake_archive)
    monkeypatch.setattr(target, "_scp_remote_file", fake_file)
    monkeypatch.setattr(target, "_scp_remote_dir_contents", fake_dir)
    monkeypatch.setattr(target, "_merge_live_policy_probe_records", lambda bundle: None)
    monkeypatch.setattr(target, "_localize_live_artifacts", lambda bundle: None)
    bundle = types.SimpleNamespace(run_id="run-1", external_dir=tmp_path / "local_external", session_dir=None)

    target.fetch_artifacts(bundle)

    assert archive_calls == []
    assert any(path.endswith("train_result.json") for path in file_calls)
    assert any(path.endswith("live") for path in dir_calls)


def test_fetch_remote_session_final_artifacts_fetches_critical_files_before_optional(tmp_path) -> None:
    from autoresearch_gym.external import remote_session

    calls: list[tuple[str, str]] = []

    class FakeTarget:
        config = types.SimpleNamespace(path_style="windows")

        def remote_join(self, *parts):
            return "\\".join(str(part).strip("\\/") for part in parts if str(part))

        def fetch_remote_file(self, remote_file, local_file, *, timeout=30.0):
            calls.append(("file", remote_file))
            Path(local_file).parent.mkdir(parents=True, exist_ok=True)
            filename = remote_file.replace("\\", "/").rsplit("/", 1)[-1]
            if filename in remote_session.FINAL_CRITICAL_RUN_FILES:
                Path(local_file).write_text("{}", encoding="utf-8")
            return not remote_file.endswith("agent_checkpoint.pt")

        def fetch_remote_dir_contents(self, remote_dir, local_dir, *, timeout=60.0):
            calls.append(("dir", remote_dir))
            Path(local_dir).mkdir(parents=True, exist_ok=True)
            return False

        def fetch_remote_dir_archive(self, remote_dir, local_dir, *, timeout=60.0):
            calls.append(("archive", remote_dir))
            return False

    local_run = remote_session.fetch_remote_session_final_artifacts(
        FakeTarget(),
        remote_session=None,
        session_dir=None,
        remote_out_dir="C:/remote/out",
        out_dir=tmp_path / "runs",
        run_id="run-1",
    )

    first_calls = calls[: len(remote_session.FINAL_CRITICAL_RUN_FILES)]
    assert local_run == tmp_path / "runs" / "run-1"
    assert [kind for kind, _ in first_calls] == ["file"] * len(remote_session.FINAL_CRITICAL_RUN_FILES)
    assert [path.replace("\\", "/").rsplit("/", 1)[-1] for _, path in first_calls] == list(
        remote_session.FINAL_CRITICAL_RUN_FILES
    )
    assert not any(kind == "archive" for kind, _ in calls)
    assert any(path.endswith("agent_checkpoint.pt") for kind, path in calls if kind == "file")


def test_fetch_remote_session_final_artifacts_raises_when_critical_files_missing(tmp_path) -> None:
    from autoresearch_gym.external import remote_session

    class FakeTarget:
        config = types.SimpleNamespace(path_style="windows")

        def remote_join(self, *parts):
            return "\\".join(str(part).strip("\\/") for part in parts if str(part))

        def fetch_remote_file(self, remote_file, local_file, *, timeout=30.0):
            return False

        def fetch_remote_dir_contents(self, remote_dir, local_dir, *, timeout=60.0):
            Path(local_dir).mkdir(parents=True, exist_ok=True)
            return False

        def fetch_remote_dir_archive(self, remote_dir, local_dir, *, timeout=60.0):
            return False

    with pytest.raises(RuntimeError, match="final artifacts are incomplete"):
        remote_session.fetch_remote_session_final_artifacts(
            FakeTarget(),
            remote_session=None,
            session_dir=None,
            remote_out_dir="C:/remote/out",
            out_dir=tmp_path / "runs",
            run_id="run-1",
        )


def test_unitree_lower_level_cleanrl_seed_trains_evals_and_reports_missing_renderer(tmp_path) -> None:
    from autoresearch_gym.external.cleanrl_backend import CleanRlExternalBackend
    from autoresearch_gym.tasks.unitree_g1_motion_mirror_v0 import seed_trainable_lower_level_cleanrl as g1_seed
    from autoresearch_gym.tasks.unitree_go2_rough_locomotion_v0 import seed_trainable_lower_level_cleanrl as go2_seed

    assert CleanRlExternalBackend().normalize_media(ArtifactSet(root=tmp_path)) == {"media_available": False}
    assert (g1_seed.RENDER_WIDTH, g1_seed.RENDER_HEIGHT) == (720, 480)
    assert (go2_seed.RENDER_WIDTH, go2_seed.RENDER_HEIGHT) == (720, 480)
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
    assert media["media_available"] is False
    assert media["visual"]["disabled_reason"] == "real_unitree_renderer_unavailable"

    go2_benchmark = types.SimpleNamespace(env_kwargs={"render_mode": "rgb_array"}, max_steps=8)
    go2_env = go2_seed.make_external_env(go2_benchmark)
    obs, _ = go2_env.reset(seed=11)
    obs, reward, terminated, truncated, info = go2_env.step(go2_env.action_space.sample())
    assert obs.shape == go2_env.observation_space.shape
    assert isinstance(float(reward), float)
    assert "command_tracking_error" in info
    frame = go2_env.render()
    if getattr(go2_env.unwrapped, "model", None) is None:
        assert frame is None
    else:
        assert frame.ndim == 3
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


def test_benchmark_max_steps_is_single_horizon_source() -> None:
    for path in sorted(Path("autoresearch_gym/tasks").glob("*/benchmark*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert "max_steps" in payload, f"{path} missing top-level max_steps"
        env_kwargs = payload.get("env_kwargs") or {}
        assert "max_steps" not in env_kwargs, f"{path} duplicates max_steps inside env_kwargs"
        assert "max_episode_steps" not in env_kwargs, f"{path} duplicates max_episode_steps inside env_kwargs"


def test_env_kwargs_for_candidate_injects_benchmark_horizon_aliases() -> None:
    so101_benchmark = BenchmarkSpec(
        name="test",
        env_id="AutoresearchMujocoSO101Reach-v0",
        env_kwargs={"render_mode": "rgb_array"},
        train_episodes=1,
        train_seconds=None,
        eval_episodes=1,
        max_steps=150,
        reward_type=None,
        render_mode="rgb_array",
        primary_metric="eval_success_rate",
        primary_metric_mode="maximize",
        train_seed=1,
        eval_seed_start=2,
        device="cpu",
        eval_case_bank=None,
    )

    env_kwargs = env_kwargs_for_candidate(so101_benchmark)

    assert env_kwargs["max_steps"] == 150
    assert env_kwargs["max_episode_steps"] == 150

    hopper_benchmark = BenchmarkSpec(
        name="test",
        env_id="Hopper-v5",
        env_kwargs={"render_mode": "rgb_array"},
        train_episodes=1,
        train_seconds=None,
        eval_episodes=1,
        max_steps=1000,
        reward_type=None,
        render_mode="rgb_array",
        primary_metric="eval_avg_return",
        primary_metric_mode="maximize",
        train_seed=1,
        eval_seed_start=2,
        device="cpu",
        eval_case_bank=None,
    )

    hopper_kwargs = env_kwargs_for_candidate(hopper_benchmark)

    assert hopper_kwargs["max_episode_steps"] == 1000
    assert "max_steps" not in hopper_kwargs


def test_load_benchmark_rejects_env_horizon_drift(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "benchmark.json"
    benchmark_path.write_text(
        json.dumps(
            {
                "name": "drift",
                "env_id": "Hopper-v5",
                "env_kwargs": {"render_mode": "rgb_array", "max_episode_steps": 4},
                "train_episodes": 1,
                "eval_episodes": 1,
                "max_steps": 5,
                "render_mode": "rgb_array",
                "primary_metric": "eval_avg_return",
                "primary_metric_mode": "maximize",
                "train_seed": 1,
                "eval_seed_start": 2,
                "device": "cpu",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="conflicting env_kwargs.max_episode_steps"):
        load_benchmark(benchmark_path)


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
    assert payload["visual"]["mode"] == "off"
    assert payload["visual"]["disabled_reason"] == "unitree_dry_run_has_no_real_renderer"
    assert not (session_dir / "live" / "current_run_frame.jpg").exists()
    assert not (session_dir / "live" / "trajectories").exists()
    media = json.loads((tmp_path / "external" / "media_result.json").read_text(encoding="utf-8"))
    assert media["media_available"] is False


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
    assert manifest["width"] == 720
    assert manifest["height"] == 480
    first_frame = Path(manifest["frames"][0])
    if not first_frame.is_absolute():
        first_frame = Path.cwd() / first_frame
    with Image.open(first_frame) as image:
        assert image.size == (720, 480)


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


def test_live_writer_keeps_completed_sample_pointer_after_shutdown_interrupt(tmp_path) -> None:
    class OneStepVisualEnv(gym.Env):
        metadata = {"render_modes": ["rgb_array"]}

        def __init__(self) -> None:
            self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
            self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
            self.step_count = 0

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.step_count = 0
            return np.zeros(1, dtype=np.float32), {}

        def step(self, action):
            self.step_count += 1
            return np.zeros(1, dtype=np.float32), 0.0, False, self.step_count >= 1, {}

        def render(self, *args, **kwargs):
            return np.full((8, 8, 3), self.step_count * 32, dtype=np.uint8)

    benchmark = BenchmarkSpec(
        name="test",
        env_id="OneStepVisual-v0",
        env_kwargs={"render_mode": "rgb_array"},
        train_episodes=10,
        train_seconds=None,
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
    session_dir = tmp_path / "session"
    writer = make_live_writer(session_dir, "run-1", "tag-1", benchmark, {"description": "candidate"})
    assert writer is not None
    control_path = session_dir / "live" / "control.json"
    control = json.loads(control_path.read_text(encoding="utf-8"))
    control["trajectory_sample_rate"] = 1.0
    control["trajectory_sample_interval_seconds"] = 0.0
    control["trajectory_target_samples"] = 0
    control["trajectory_frame_stride"] = 1
    control_path.write_text(json.dumps(control), encoding="utf-8")
    env = writer.wrap_env(OneStepVisualEnv())  # type: ignore[attr-defined]

    env.reset()
    env.step(np.zeros(1, dtype=np.float32))
    env.reset()
    writer(
        status="finished",
        episode_records=[],
        total_steps=1,
        last_metrics=None,
        env=env,
        current_episode=2,
        episode_return=0.0,
        episode_length=0,
    )

    metrics = json.loads((session_dir / "live" / "current_run_metrics.json").read_text(encoding="utf-8"))
    manifest_path = metrics["visual"]["trajectory_manifest_path"]
    completed_manifest = session_dir / "live" / "trajectories" / "run-1" / "episode_000001" / "manifest.json"
    interrupted_manifest = session_dir / "live" / "trajectories" / "run-1" / "episode_000002" / "manifest.json"
    completed = json.loads(completed_manifest.read_text(encoding="utf-8"))
    interrupted = json.loads(interrupted_manifest.read_text(encoding="utf-8"))
    assert manifest_path.replace("\\", "/").endswith("episode_000001/manifest.json")
    assert completed["status"] == "completed"
    assert completed["frame_count"] >= 2
    assert interrupted["status"] == "interrupted"
    assert interrupted["reason"] == "training_finished"


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

    assert "pct=50.0" in line
    assert "eps=2/4" in line
    assert "step=1234" in line
    assert "upd=?" in line
    assert "time=" not in line


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

    line = status_file.read_text(encoding="utf-8").strip()
    assert "time=0/300s" in line
    assert "st=run" in line
    assert "step=12" in line
    assert "upd=?" in line


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
            "--dashboard-port",
            "4180",
            "--dashboard-port-end",
            "4190",
            "--allow-remote-drift",
        ]
    )

    assert args.headless_env is True
    assert args.compact_status is True
    assert str(args.compact_status_file) == "status.log"
    assert args.status_interval_seconds == 2.5
    assert args.probe_interval_seconds == 3.5
    assert args.probe_episodes == 2
    assert args.no_train_probe is True
    assert args.dashboard is True
    assert args.dashboard_port == 4180
    assert args.dashboard_port_end == 4190
    assert args.allow_remote_drift is True


def test_run_parser_accepts_no_dashboard_opt_out() -> None:
    args = parse_args(["--candidate", "candidate.py", "--no-dashboard"])

    assert args.dashboard is False


def test_run_parser_rejects_removed_dashboard_alias() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--candidate", "candidate.py", "--dashboard"])


def test_init_session_dashboard_is_default_and_has_no_dashboard_opt_out() -> None:
    parser = cli.build_parser()

    default_args = parser.parse_args(["init-session", "--label", "dashboard-default"])
    opt_out_args = parser.parse_args(["init-session", "--label", "dashboard-off", "--no-dashboard"])

    assert default_args.dashboard is True
    assert opt_out_args.dashboard is False
    with pytest.raises(SystemExit):
        parser.parse_args(["init-session", "--label", "dashboard-alias", "--dashboard"])


def test_init_session_writes_doctor_report(tmp_path, monkeypatch, capsys) -> None:
    benchmark = tmp_path / "benchmark.json"
    seed_candidate = tmp_path / "seed_trainable.py"
    benchmark.write_text(
        json.dumps(
            {
                "name": "doctor-test",
                "device": "external",
                "execution_backend": {
                    "kind": "external",
                    "required_paths": [".external/unitree_rl_mjlab"],
                },
            }
        ),
        encoding="utf-8",
    )
    seed_candidate.write_text("def get_candidate():\n    return {}\n", encoding="utf-8")
    calls: list[dict[str, object]] = []

    def fake_doctor(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return {"ok": False, "checks": [{"name": "required_path:.external/unitree_rl_mjlab", "status": "fail"}]}

    monkeypatch.setattr(cli, "run_session_doctor", fake_doctor)
    args = argparse.Namespace(
        label="doctor init",
        benchmark=benchmark,
        seed_candidate=seed_candidate,
        base_dir=tmp_path / "runs",
        search_mode="linear",
        execution_target="windows_gpu",
        target_config=tmp_path / "targets.toml",
        skip_doctor=False,
        strict_doctor=False,
        doctor_timeout=12.0,
        dashboard=False,
        dashboard_host="127.0.0.1",
        dashboard_port=4174,
        dashboard_port_end=4199,
        dashboard_ready_timeout=0.01,
    )

    status = cli.cmd_init_session(args)
    payload = json.loads(capsys.readouterr().out)
    session_dir = Path(payload["session_dir"])
    doctor_path = session_dir / "doctor.json"
    session_meta = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))

    assert status == 0
    assert payload["doctor"]["ok"] is False
    assert doctor_path.exists()
    assert json.loads(doctor_path.read_text(encoding="utf-8"))["ok"] is False
    assert session_meta["doctor_ok"] is False
    assert session_meta["doctor_path"] == str(doctor_path)
    assert calls[0]["args"] == (benchmark,)
    assert calls[0]["kwargs"]["execution_target"] == "windows_gpu"
    assert calls[0]["kwargs"]["target_config_path"] == tmp_path / "targets.toml"
    assert calls[0]["kwargs"]["timeout"] == 12.0


def test_init_session_strict_doctor_returns_nonzero(tmp_path, monkeypatch, capsys) -> None:
    benchmark = tmp_path / "benchmark.json"
    seed_candidate = tmp_path / "seed_trainable.py"
    benchmark.write_text(json.dumps({"name": "strict-doctor"}), encoding="utf-8")
    seed_candidate.write_text("def get_candidate():\n    return {}\n", encoding="utf-8")
    monkeypatch.setattr(cli, "run_session_doctor", lambda *args, **kwargs: {"ok": False, "checks": []})

    args = argparse.Namespace(
        label="strict doctor",
        benchmark=benchmark,
        seed_candidate=seed_candidate,
        base_dir=tmp_path / "runs",
        search_mode="linear",
        execution_target=None,
        target_config=None,
        skip_doctor=False,
        strict_doctor=True,
        doctor_timeout=12.0,
        dashboard=False,
        dashboard_host="127.0.0.1",
        dashboard_port=4174,
        dashboard_port_end=4199,
        dashboard_ready_timeout=0.01,
    )

    status = cli.cmd_init_session(args)
    payload = json.loads(capsys.readouterr().out)

    assert status == 1
    assert Path(payload["session_dir"], "doctor.json").exists()


def test_dashboard_url_uses_localhost_for_wildcard_host() -> None:
    from autoresearch_gym.runner.dashboard_server import dashboard_url

    assert (
        dashboard_url("0.0.0.0", 4174, "autoresearch_runs/sessions/example session")
        == "http://127.0.0.1:4174/dashboard/?session=autoresearch_runs/sessions/example%20session"
    )


def test_session_dashboard_ensure_reuses_ready_pointer(tmp_path, monkeypatch) -> None:
    from autoresearch_gym.runner import dashboard_server

    session_dir = tmp_path / "autoresearch_runs" / "sessions" / "example"
    pointer = {
        "url": "http://127.0.0.1:4174/dashboard/?session=autoresearch_runs/sessions/example",
        "ready": True,
        "pid": 123,
        "managed_by": dashboard_server.DASHBOARD_MANAGER,
    }
    dashboard_server.write_dashboard_pointer(session_dir, pointer)
    monkeypatch.setattr(dashboard_server, "wait_for_dashboard", lambda url, *, timeout_seconds=1.0: True)

    payload = dashboard_server.ensure_session_dashboard(
        session_dir,
        host="127.0.0.1",
        port=4174,
        port_end=4199,
        root=tmp_path,
    )

    assert payload["reused"] is True
    assert payload["pid"] == 123
    assert dashboard_server.read_dashboard_pointer(session_dir)["reused"] is True


def test_session_dashboard_reuses_preferred_port_identity_for_same_root(tmp_path, monkeypatch) -> None:
    from autoresearch_gym.runner import dashboard_server

    session_dir = tmp_path / "autoresearch_runs" / "sessions" / "new-session"
    launched: list[list[str]] = []
    monkeypatch.setattr(
        dashboard_server,
        "read_dashboard_identity",
        lambda host, port, *, timeout_seconds=1.0: {
            "ok": True,
            "managed_by": dashboard_server.DASHBOARD_MANAGER,
            "root": str(tmp_path.resolve()),
            "pid": 777,
        },
    )
    monkeypatch.setattr(dashboard_server.subprocess, "Popen", lambda cmd, **kwargs: launched.append(list(cmd)))

    payload = dashboard_server.ensure_session_dashboard(
        session_dir,
        host="127.0.0.1",
        port=4174,
        port_end=4199,
        root=tmp_path,
    )

    assert payload["reused"] is True
    assert payload["shared"] is True
    assert payload["pid"] == 777
    assert payload["port"] == 4174
    assert payload["url"].endswith("?session=autoresearch_runs/sessions/new-session")
    assert launched == []
    assert dashboard_server.read_dashboard_pointer(session_dir)["source"] == "preferred_port_identity"


def test_session_dashboard_restarts_wrong_root_dashboard_on_preferred_port(tmp_path, monkeypatch) -> None:
    from autoresearch_gym.runner import dashboard_server

    session_dir = tmp_path / "autoresearch_runs" / "sessions" / "example"
    terminated: list[dict[str, object]] = []
    launched: list[list[str]] = []

    class FakeProcess:
        pid = 888

    monkeypatch.setattr(
        dashboard_server,
        "read_dashboard_identity",
        lambda host, port, *, timeout_seconds=1.0: {
            "ok": True,
            "managed_by": dashboard_server.DASHBOARD_MANAGER,
            "root": str((tmp_path / "other-checkout").resolve()),
            "pid": 444,
        },
    )
    monkeypatch.setattr(
        dashboard_server,
        "terminate_managed_dashboard_identity",
        lambda identity: terminated.append(dict(identity)) or {"ok": True, "action": "terminated", "pid": identity.get("pid")},
    )
    monkeypatch.setattr(dashboard_server, "port_is_available", lambda host, port: port == 4174)
    monkeypatch.setattr(dashboard_server, "wait_for_dashboard", lambda url, *, timeout_seconds=1.0: True)
    monkeypatch.setattr(dashboard_server.subprocess, "Popen", lambda cmd, **kwargs: launched.append(list(cmd)) or FakeProcess())

    payload = dashboard_server.ensure_session_dashboard(
        session_dir,
        host="127.0.0.1",
        port=4174,
        port_end=4199,
        root=tmp_path,
    )

    assert terminated and terminated[0]["pid"] == 444
    assert payload["reused"] is False
    assert payload["port"] == 4174
    assert launched[0][launched[0].index("--port") + 1] == "4174"
    assert launched[0][launched[0].index("--port-end") + 1] == "4174"


def test_session_dashboard_falls_back_when_preferred_port_is_unmanaged(tmp_path, monkeypatch) -> None:
    from autoresearch_gym.runner import dashboard_server

    session_dir = tmp_path / "autoresearch_runs" / "sessions" / "example"
    launched: list[list[str]] = []

    class FakeProcess:
        pid = 999

    monkeypatch.setattr(dashboard_server, "read_dashboard_identity", lambda host, port, *, timeout_seconds=1.0: None)
    monkeypatch.setattr(dashboard_server, "port_is_available", lambda host, port: port == 4175)
    monkeypatch.setattr(dashboard_server, "wait_for_dashboard", lambda url, *, timeout_seconds=1.0: True)
    monkeypatch.setattr(dashboard_server.subprocess, "Popen", lambda cmd, **kwargs: launched.append(list(cmd)) or FakeProcess())

    payload = dashboard_server.ensure_session_dashboard(
        session_dir,
        host="127.0.0.1",
        port=4174,
        port_end=4199,
        root=tmp_path,
    )

    assert payload["reused"] is False
    assert payload["port"] == 4175
    assert launched[0][launched[0].index("--port") + 1] == "4175"
    assert launched[0][launched[0].index("--port-end") + 1] == "4175"


def test_session_dashboard_child_command_uses_selected_port_end(tmp_path, monkeypatch) -> None:
    from autoresearch_gym.runner import dashboard_server

    session_dir = tmp_path / "autoresearch_runs" / "sessions" / "example"
    launched: list[list[str]] = []

    class FakeProcess:
        pid = 321

    def fake_popen(cmd, **kwargs):
        launched.append(list(cmd))
        return FakeProcess()

    monkeypatch.setattr(dashboard_server, "find_available_port", lambda host, port, port_end: 4200)
    monkeypatch.setattr(dashboard_server, "wait_for_dashboard", lambda url, *, timeout_seconds=1.0: True)
    monkeypatch.setattr(dashboard_server.subprocess, "Popen", fake_popen)

    payload = dashboard_server.ensure_session_dashboard(
        session_dir,
        host="127.0.0.1",
        port=4200,
        port_end=4299,
        root=tmp_path,
    )

    assert payload["ready"] is True
    assert launched
    command = launched[0]
    assert command[command.index("--port") + 1] == "4200"
    assert command[command.index("--port-end") + 1] == "4200"


def test_session_dashboard_teardown_only_kills_managed_pointer(tmp_path, monkeypatch) -> None:
    from autoresearch_gym.runner import dashboard_server

    session_dir = tmp_path / "autoresearch_runs" / "sessions" / "example"
    dashboard_server.write_dashboard_pointer(session_dir, {"pid": 123, "managed_by": "someone_else"})
    kill_calls: list[tuple[int, int]] = []
    monkeypatch.setattr(dashboard_server.os, "kill", lambda pid, sig: kill_calls.append((pid, sig)))

    unmanaged = dashboard_server.terminate_session_dashboard(session_dir)

    assert unmanaged["ok"] is False
    assert unmanaged["reason"] == "unmanaged_pointer"
    assert kill_calls == []

    dashboard_server.write_dashboard_pointer(
        session_dir,
        {"pid": 456, "managed_by": dashboard_server.DASHBOARD_MANAGER, "url": "http://127.0.0.1:4174/dashboard/"},
    )
    alive = {"value": True}

    def fake_is_alive(pid):
        return bool(alive["value"])

    def fake_kill(pid, sig):
        kill_calls.append((pid, sig))
        alive["value"] = False

    monkeypatch.setattr(dashboard_server, "is_process_alive", fake_is_alive)
    monkeypatch.setattr(dashboard_server.os, "kill", fake_kill)

    managed = dashboard_server.terminate_session_dashboard(session_dir)

    assert managed["ok"] is True
    assert managed["action"] == "terminated"
    assert kill_calls == [(456, dashboard_server.signal.SIGTERM)]
    assert dashboard_server.read_dashboard_pointer(session_dir)["stopped"] is True


def test_remote_session_pass_wrapper_adds_dashboard_and_status_defaults(tmp_path) -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_session_remote_pass.py"
    spec = importlib.util.spec_from_file_location("run_session_remote_pass", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    session_dir = tmp_path / "autoresearch_runs" / "sessions" / "example"
    candidate = session_dir / "candidates" / "pass02_test.py"
    candidate.parent.mkdir(parents=True)
    candidate.write_text("def get_candidate():\n    return {}\n", encoding="utf-8")

    module.validate_session_candidate(session_dir, candidate)
    args = argparse.Namespace(
        session_dir=session_dir,
        candidate=candidate,
        execution_target="example_windows_gpu",
        target_config=Path(".autoresearch.local.toml"),
        no_compact_status=False,
        no_dashboard=False,
        dashboard_host="127.0.0.1",
        dashboard_port=4174,
        dashboard_port_end=4199,
        disable_train_probes=True,
        allow_remote_drift=False,
    )

    run_args = module.build_run_args(args, ["--benchmark", "benchmark.json", "--tag", "pass02-test"])

    assert "--compact-status" in run_args
    assert run_args[run_args.index("--compact-status-file") + 1] == str(session_dir / "live" / "status.log")
    assert "--dashboard" not in run_args
    assert "--no-dashboard" not in run_args
    assert "--dashboard-host" in run_args
    assert run_args[run_args.index("--execution-target") + 1] == "example_windows_gpu"
    assert "--no-train-probe" in run_args

    args.no_dashboard = True
    opt_out_args = module.build_run_args(args, ["--benchmark", "benchmark.json", "--tag", "pass02-test"])
    assert "--no-dashboard" in opt_out_args
    assert "--dashboard-host" not in opt_out_args


def test_launch_autoresearch_pass_writes_agent_prerun_log(tmp_path) -> None:
    script_dir = Path(__file__).resolve().parents[1] / "scripts"
    script_path = script_dir / "launch_autoresearch_pass.py"
    sys.path.insert(0, str(script_dir))
    try:
        spec = importlib.util.spec_from_file_location("launch_autoresearch_pass", script_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(script_dir))

    session_dir = tmp_path / "autoresearch_runs" / "sessions" / "example"
    candidate = session_dir / "candidates" / "pass03_contact_replay.py"
    candidate.parent.mkdir(parents=True)
    candidate.write_text("def get_candidate():\n    return {}\n", encoding="utf-8")
    args = argparse.Namespace(
        session_dir=session_dir,
        candidate=candidate,
        execution_target=None,
        hypothesis="Contact-heavy replay should increase object motion.",
        mutation_summary="Increase replay sampling weight for contact windows.",
        expected_diagnostics="near_cube and lifted_ever should improve together.",
        success_criteria=None,
        next_planned_mutation="Keep if fixed eval lift improves.",
    )

    module.append_agent_prerun_log(
        args,
        [
            "--benchmark",
            "benchmark.json",
            "--seed-candidate",
            "seed_trainable.py",
            "--tag",
            "pass03-contact-replay",
            "--mutation-family",
            "replay",
        ],
    )
    module.append_agent_prerun_log(
        args,
        [
            "--benchmark",
            "benchmark.json",
            "--seed-candidate",
            "seed_trainable.py",
            "--tag",
            "pass03-contact-replay",
            "--mutation-family",
            "replay",
        ],
    )

    log = (session_dir / "outer_loop_log.md").read_text(encoding="utf-8")
    assert "## pass03-contact-replay pre-run plan" in log
    assert log.count("## pass03-contact-replay pre-run plan") == 1
    assert "- authored_by: agent" in log
    assert "Contact-heavy replay should increase object motion." in log
    assert "Increase replay sampling weight for contact windows." in log
    assert "near_cube and lifted_ever should improve together." in log


def test_session_run_no_dashboard_preserves_existing_dashboard_pointer(tmp_path) -> None:
    from autoresearch_gym.runner import session_run

    session_dir = tmp_path / "autoresearch_runs" / "sessions" / "example"
    live_dir = session_dir / "live"
    live_dir.mkdir(parents=True)
    (live_dir / "dashboard.json").write_text(
        json.dumps({"url": "http://127.0.0.1:4174/dashboard/?session=stale", "ready": True}),
        encoding="utf-8",
    )
    args = argparse.Namespace(dashboard=False)

    payload = session_run.start_dashboard(args, session_dir)

    assert payload == {"url": "http://127.0.0.1:4174/dashboard/?session=stale", "ready": True}
    written = json.loads((live_dir / "dashboard.json").read_text(encoding="utf-8"))
    assert written == payload


def test_session_run_dashboard_ensures_session_dashboard(tmp_path, monkeypatch) -> None:
    from autoresearch_gym.runner import session_run

    calls: list[dict[str, object]] = []

    def fake_ensure(session_dir, **kwargs):
        calls.append({"session_dir": session_dir, **kwargs})
        return {
            "url": "http://127.0.0.1:4174/dashboard/?session=example",
            "ready": True,
            "reused": True,
            "log_path": str(session_dir / "live" / "dashboard.log"),
        }

    monkeypatch.setattr(session_run, "ensure_session_dashboard", fake_ensure)
    session_dir = tmp_path / "autoresearch_runs" / "sessions" / "example"
    args = argparse.Namespace(
        dashboard=True,
        dashboard_host="127.0.0.1",
        dashboard_port=4174,
        dashboard_port_end=4199,
        dashboard_ready_timeout=2.0,
    )

    payload = session_run.start_dashboard(args, session_dir)

    assert payload["reused"] is True
    assert calls == [
        {
            "session_dir": session_dir,
            "host": "127.0.0.1",
            "port": 4174,
            "port_end": 4199,
            "root": Path.cwd(),
            "ready_timeout": 2.0,
        }
    ]


def test_launch_autoresearch_pass_builds_fragile_remote_defaults(tmp_path) -> None:
    script_dir = Path(__file__).resolve().parents[1] / "scripts"
    script_path = script_dir / "launch_autoresearch_pass.py"
    sys.path.insert(0, str(script_dir))
    try:
        spec = importlib.util.spec_from_file_location("launch_autoresearch_pass_defaults", script_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(script_dir))

    session_dir = tmp_path / "autoresearch_runs" / "sessions" / "example"
    candidate = session_dir / "candidates" / "pass04_remote.py"
    args = argparse.Namespace(
        session_dir=session_dir,
        candidate=candidate,
        execution_target="example_windows_gpu",
        target_config=None,
        no_compact_status=False,
        no_dashboard=False,
        dashboard_host="127.0.0.1",
        dashboard_port=4174,
        dashboard_port_end=4199,
        disable_train_probes=False,
        profile="fragile-remote",
        allow_remote_drift=True,
    )

    run_args = module.build_run_args(args, ["--benchmark", "benchmark.json", "--tag", "pass04-remote"])

    assert run_args[run_args.index("--execution-target") + 1] == "example_windows_gpu"
    assert "--compact-status" in run_args
    assert "--dashboard" not in run_args
    assert "--no-dashboard" not in run_args
    assert "--dashboard-host" in run_args
    assert "--no-train-probe" in run_args
    assert "--allow-remote-drift" in run_args

    args.no_dashboard = True
    opt_out_args = module.build_run_args(args, ["--benchmark", "benchmark.json", "--tag", "pass04-remote"])
    assert "--no-dashboard" in opt_out_args
    assert "--dashboard-host" not in opt_out_args


def test_remote_environment_verification_compares_git_and_lock_hashes(monkeypatch) -> None:
    from autoresearch_gym.external import remote_session

    local = {
        "repo_root": str(Path.cwd()),
        "git_head": "local-head",
        "dirty_paths": ["?? autoresearch_runs/sessions/example/candidates/pass02.py"],
        "lock_hashes": {"pyproject.toml": "pyproject-hash", "uv.lock": "uv-hash"},
    }

    class FakeTarget:
        config = types.SimpleNamespace(kind="ssh")

    monkeypatch.setattr(remote_session, "load_ssh_target", lambda *args, **kwargs: FakeTarget())
    monkeypatch.setattr(remote_session, "local_environment_fingerprint", lambda repo_root: local)
    monkeypatch.setattr(
        remote_session,
        "remote_environment_fingerprint",
        lambda target: {
            "git_head": "local-head",
            "dirty_paths": [],
            "python_version": "3.10.test",
            "lock_hashes": dict(local["lock_hashes"]),
        },
    )

    status = remote_session.verify_remote_environment("pytest-target", repo_root=Path.cwd())

    assert status["ok"] is True
    assert {check["name"] for check in status["checks"]} == {
        "git_head",
        "pyproject_hash",
        "uv_lock_hash",
        "local_environment_clean",
        "remote_environment_clean",
    }

    monkeypatch.setattr(
        remote_session,
        "remote_environment_fingerprint",
        lambda target: {
            "git_head": "different",
            "dirty_paths": [],
            "python_version": "3.10.test",
            "lock_hashes": dict(local["lock_hashes"]),
        },
    )

    with pytest.raises(RuntimeError, match="remote environment differs"):
        remote_session.verify_remote_environment("pytest-target", repo_root=Path.cwd())


def test_remote_environment_verification_blocks_dirty_package_paths(monkeypatch) -> None:
    from autoresearch_gym.external import remote_session

    local = {
        "repo_root": str(Path.cwd()),
        "git_head": "same-head",
        "dirty_paths": [" M autoresearch_gym/external/remote_session.py"],
        "lock_hashes": {"pyproject.toml": "pyproject-hash", "uv.lock": "uv-hash"},
    }

    class FakeTarget:
        config = types.SimpleNamespace(kind="ssh")

    monkeypatch.setattr(remote_session, "load_ssh_target", lambda *args, **kwargs: FakeTarget())
    monkeypatch.setattr(remote_session, "local_environment_fingerprint", lambda repo_root: local)
    monkeypatch.setattr(
        remote_session,
        "remote_environment_fingerprint",
        lambda target: {
            "git_head": "same-head",
            "dirty_paths": [],
            "python_version": "3.10.test",
            "lock_hashes": dict(local["lock_hashes"]),
        },
    )

    with pytest.raises(RuntimeError, match="local_environment_clean"):
        remote_session.verify_remote_environment("pytest-target", repo_root=Path.cwd())

    status = remote_session.verify_remote_environment("pytest-target", repo_root=Path.cwd(), allow_remote_drift=True)
    assert status["ok"] is False
    failed = [check for check in status["checks"] if not check["ok"]]
    assert failed[0]["dirty_paths"] == ["autoresearch_gym/external/remote_session.py"]


def test_remote_in_process_sync_delegates_to_shared_remote_session(monkeypatch, tmp_path) -> None:
    from autoresearch_gym.external import in_process

    calls: list[tuple[object, str, Path | None]] = []

    def fake_sync(target, remote_session, session_dir):
        calls.append((target, remote_session, session_dir))

    monkeypatch.setattr(in_process, "sync_remote_session_live", fake_sync)
    target = object()
    session_dir = tmp_path / "session"

    in_process._sync_remote_session(target, "C:/repo/autoresearch_runs/sessions/example", session_dir)

    assert calls == [(target, "C:/repo/autoresearch_runs/sessions/example", session_dir)]


def test_pre_commit_affected_plan_selects_exact_seed_case() -> None:
    script_dir = Path(__file__).resolve().parents[1] / "scripts"
    script_path = script_dir / "pre_commit_checks.py"
    sys.path.insert(0, str(script_dir))
    try:
        spec = importlib.util.spec_from_file_location("pre_commit_checks", script_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(script_dir))

    path = Path("autoresearch_gym/tasks/panda_pick_and_place_mjwarp_pandagym_dense_v0/seed_trainable_guided_warmup.py")
    commands = module.affected_plan([path], "python", 12.0, Path.cwd())
    flattened = [" ".join(cmd) for _, cmd in commands]

    assert any("py_compile" in command and "seed_trainable_guided_warmup.py" in command for command in flattened)
    assert any("--case panda-mjwarp-pandagym-dense-guided-warmup" in command for command in flattened)
    assert not any("--case panda-mjwarp-pandagym-dense " in f"{command} " for command in flattened)


def test_pre_commit_affected_plan_skips_docs_only() -> None:
    script_dir = Path(__file__).resolve().parents[1] / "scripts"
    script_path = script_dir / "pre_commit_checks.py"
    sys.path.insert(0, str(script_dir))
    try:
        spec = importlib.util.spec_from_file_location("pre_commit_checks_docs", script_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(script_dir))

    assert module.affected_plan([Path("AUTORESEARCH.md"), Path("docs/example.md")], "python", 12.0, Path.cwd()) == []


def test_pre_commit_affected_plan_uses_changed_test_names() -> None:
    script_dir = Path(__file__).resolve().parents[1] / "scripts"
    script_path = script_dir / "pre_commit_checks.py"
    sys.path.insert(0, str(script_dir))
    try:
        spec = importlib.util.spec_from_file_location("pre_commit_checks_tests", script_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(script_dir))

    commands = module.affected_plan(
        [Path("tests/test_smoke.py")],
        "python",
        12.0,
        Path.cwd(),
        {"test_new_contract"},
    )

    assert commands == [
        (
            "affected unit smoke tests",
            ["python", "-m", "pytest", "tests/test_smoke.py", "-k", "test_new_contract"],
        )
    ]


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


def test_all_bundled_eval_case_banks_cover_benchmark_eval_budget() -> None:
    from pathlib import Path

    task_root = Path(__file__).resolve().parents[1] / "autoresearch_gym" / "tasks"
    benchmark_paths = sorted(task_root.glob("*/benchmark*.json"))
    assert benchmark_paths
    for benchmark_path in benchmark_paths:
        benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
        eval_case_bank = benchmark.get("eval_case_bank")
        if not eval_case_bank:
            continue
        case_bank_path = benchmark_path.parent / str(eval_case_bank)
        cases = json.loads(case_bank_path.read_text(encoding="utf-8")).get("cases", [])
        assert len(cases) >= int(benchmark["eval_episodes"]), (
            f"{benchmark_path} requests {benchmark['eval_episodes']} eval episodes "
            f"but {case_bank_path} only has {len(cases)} cases"
        )


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


def test_mujoco_panda_pick_and_place_curriculum_wrapper_shapes_subskills() -> None:
    from autoresearch_gym.tasks.panda_pick_and_place_mjwarp_v0.seed_trainable import RewardRecipeWrapper

    class DummyGoalEnv(gym.Env):
        action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        observation_space = gym.spaces.Dict(
            {
                "observation": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32),
                "achieved_goal": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                "desired_goal": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            }
        )

        def reset(self, *, seed=None, options=None):
            del seed, options
            return {
                "observation": np.zeros(7, dtype=np.float32),
                "achieved_goal": np.zeros(3, dtype=np.float32),
                "desired_goal": np.ones(3, dtype=np.float32),
            }, {}

        def step(self, action):
            del action
            obs = {
                "observation": np.zeros(7, dtype=np.float32),
                "achieved_goal": np.zeros(3, dtype=np.float32),
                "desired_goal": np.ones(3, dtype=np.float32),
            }
            return obs, -1.0, False, False, {
                "ee_to_cube_distance": 0.03,
                "cube_to_goal_distance": 0.40,
                "initial_ee_to_cube_distance": 0.10,
                "initial_cube_to_goal_distance": 0.40,
                "ee_to_cube_progress": 0.07,
                "cube_to_goal_progress": 0.00,
                "cube_lift_height": 0.00,
                "near_cube": True,
                "gripper_closed_near_cube": True,
                "lifted": False,
                "placed_success": False,
            }

    env = RewardRecipeWrapper(DummyGoalEnv(), "subskill_curriculum")
    obs, _ = env.reset()
    assert obs.shape == env.observation_space.shape
    _, reward, _, _, info = env.step(np.zeros(2, dtype=np.float32))
    assert reward > -1.0
    assert info["training_reward"] == reward
    assert info["curriculum_phase_index"] == 0.0


def test_mujoco_panda_pick_and_place_curriculum_rewards_use_reset_relative_progress() -> None:
    from autoresearch_gym.tasks.panda_pick_and_place_mjwarp_v0.seed_trainable import (
        GRASP_LIFT_STEPS,
        _curriculum_reward,
        _curriculum_reward_vector,
    )

    base_info = {
        "ee_to_cube_distance": 0.05,
        "cube_to_goal_distance": 0.16,
        "initial_ee_to_cube_distance": 0.20,
        "initial_cube_to_goal_distance": 0.30,
        "cube_lift_height": 0.00,
        "near_cube": False,
        "gripper_closed_near_cube": False,
        "lifted": False,
        "placed_success": False,
    }
    no_progress = {**base_info, "ee_to_cube_progress": 0.00, "cube_to_goal_progress": 0.00}
    approach_progress = {**base_info, "ee_to_cube_progress": 0.10, "cube_to_goal_progress": 0.00}
    assert _curriculum_reward(-0.16, approach_progress, 0) > _curriculum_reward(-0.16, no_progress, 0)

    place_progress = {**base_info, "ee_to_cube_progress": 0.00, "cube_to_goal_progress": 0.12, "lifted": True}
    assert _curriculum_reward(-0.16, place_progress, GRASP_LIFT_STEPS) > _curriculum_reward(-0.16, no_progress, GRASP_LIFT_STEPS)

    push_progress = {
        **base_info,
        "ee_to_cube_progress": 0.00,
        "cube_to_goal_progress": 0.12,
        "lifted": False,
        "lifted_ever": False,
    }
    lifted_progress = {**push_progress, "lifted_ever": True}
    assert _curriculum_reward(-0.16, push_progress, GRASP_LIFT_STEPS) == pytest.approx(
        _curriculum_reward(-0.16, no_progress, GRASP_LIFT_STEPS)
    )
    assert _curriculum_reward(-0.16, lifted_progress, GRASP_LIFT_STEPS) > _curriculum_reward(-0.16, push_progress, GRASP_LIFT_STEPS)

    vector_rewards = _curriculum_reward_vector(
        np.array([-0.16, -0.16], dtype=np.float32),
        {
            "ee_to_cube_distance": np.array([0.05, 0.05], dtype=np.float32),
            "cube_to_goal_distance": np.array([0.16, 0.16], dtype=np.float32),
            "initial_ee_to_cube_distance": np.array([0.20, 0.20], dtype=np.float32),
            "initial_cube_to_goal_distance": np.array([0.30, 0.30], dtype=np.float32),
            "ee_to_cube_progress": np.array([0.00, 0.10], dtype=np.float32),
            "cube_to_goal_progress": np.array([0.00, 0.00], dtype=np.float32),
            "near_cube": np.array([False, False]),
            "gripper_closed_near_cube": np.array([False, False]),
            "lifted": np.array([False, False]),
            "placed_success": np.array([False, False]),
        },
        0,
    )
    assert vector_rewards[1] > vector_rewards[0]

    place_vector_rewards = _curriculum_reward_vector(
        np.array([-0.16, -0.16], dtype=np.float32),
        {
            "ee_to_cube_distance": np.array([0.05, 0.05], dtype=np.float32),
            "cube_to_goal_distance": np.array([0.16, 0.16], dtype=np.float32),
            "initial_ee_to_cube_distance": np.array([0.20, 0.20], dtype=np.float32),
            "initial_cube_to_goal_distance": np.array([0.30, 0.30], dtype=np.float32),
            "ee_to_cube_progress": np.array([0.00, 0.00], dtype=np.float32),
            "cube_to_goal_progress": np.array([0.12, 0.12], dtype=np.float32),
            "near_cube": np.array([False, False]),
            "gripper_closed_near_cube": np.array([False, False]),
            "lifted": np.array([False, False]),
            "lifted_ever": np.array([False, True]),
            "placed_success": np.array([False, False]),
        },
        GRASP_LIFT_STEPS,
    )
    assert place_vector_rewards[1] > place_vector_rewards[0]


def test_mujoco_panda_pick_and_place_success_requires_lift_before_place() -> None:
    from autoresearch_gym.envs import mujoco_panda_pick_and_place as panda_env

    assert not bool(panda_env._pick_place_success(0.0, False))
    assert bool(panda_env._pick_place_success(0.0, True))
    assert not bool(panda_env._pick_place_success(panda_env.SUCCESS_THRESHOLD + 0.01, True))

    env = panda_env.AutoresearchMujocoPandaPickAndPlaceEnv.__new__(panda_env.AutoresearchMujocoPandaPickAndPlaceEnv)
    env.reward_type = "dense"
    at_goal = np.zeros(3, dtype=np.float32)
    assert float(env.compute_reward(at_goal, at_goal, {"lifted_ever": False})) == pytest.approx(-1.0)
    assert float(env.compute_reward(at_goal, at_goal, {"lifted_ever": True})) == pytest.approx(0.0)


def test_mujoco_panda_scene_matches_panda_gym_work_surface_layout(tmp_path: Path) -> None:
    from autoresearch_gym.envs import mujoco_panda_pick_and_place as panda_env

    panda_xml = tmp_path / "panda.xml"
    panda_xml.write_text(
        '<mujoco model="panda"><compiler/><worldbody><body name="link0" pos="0 0 0"/></worldbody></mujoco>',
        encoding="utf-8",
    )

    scene_path = panda_env._write_scene_xml(panda_xml)
    root = panda_env.ET.parse(scene_path).getroot()
    worldbody = root.find("worldbody")
    assert worldbody is not None

    robot_base = worldbody.find("./body[@name='link0']")
    table = worldbody.find("./geom[@name='table']")
    cube = worldbody.find("./body[@name='object']")
    target = worldbody.find("./site[@name='target']")
    visual = root.find("visual/global")
    assert robot_base is not None
    assert table is not None
    assert cube is not None
    assert target is not None
    assert visual is not None

    assert robot_base.get("pos") == "-0.600 0 0"
    assert int(visual.get("offwidth", "0")) >= 1024
    assert int(visual.get("offheight", "0")) >= 768
    assert [float(value) for value in table.get("pos", "").split()] == pytest.approx([-0.3, 0.0, -0.2])
    assert [float(value) for value in table.get("size", "").split()] == pytest.approx([0.55, 0.35, 0.2])
    assert [float(value) for value in cube.get("pos", "").split()] == pytest.approx([0.0, 0.0, 0.02])
    assert [float(value) for value in target.get("pos", "").split()] == pytest.approx([0.12, 0.0, 0.02])
    assert target.get("type") == "box"
    assert [float(value) for value in target.get("size", "").split()] == pytest.approx([0.02, 0.02, 0.02])


def test_mujoco_panda_goal_sampler_uses_tabletop_goals() -> None:
    from autoresearch_gym.envs import mujoco_panda_pick_and_place as panda_env

    env = panda_env.AutoresearchMujocoPandaPickAndPlaceEnv.__new__(panda_env.AutoresearchMujocoPandaPickAndPlaceEnv)
    env.rng = np.random.default_rng(4500)

    goal, cube, attempts = env._sample_goal_and_cube_pos()

    assert goal == pytest.approx([0.00691743, -0.07961766, 0.02])
    assert cube == pytest.approx([-0.05418519, -0.10684564, 0.02])
    assert attempts == 0

    for seed in range(4500, 4550):
        env.rng = np.random.default_rng(seed)
        goal, _, _ = env._sample_goal_and_cube_pos()
        assert goal[2] == pytest.approx(panda_env.CUBE_Z)


def test_mujoco_panda_goal_sampler_rejects_initial_successes() -> None:
    from autoresearch_gym.envs import mujoco_panda_pick_and_place as panda_env

    env = panda_env.AutoresearchMujocoPandaPickAndPlaceEnv.__new__(panda_env.AutoresearchMujocoPandaPickAndPlaceEnv)
    env.rng = np.random.default_rng(7)

    for _ in range(500):
        goal, cube, _ = env._sample_goal_and_cube_pos()
        assert np.linalg.norm(cube - goal) >= panda_env.SUCCESS_THRESHOLD

    vec = panda_env.MujocoWarpPandaPickAndPlaceVectorEnv.__new__(panda_env.MujocoWarpPandaPickAndPlaceVectorEnv)
    vec.rng = np.random.default_rng(7)
    goals, cubes = vec._sample_goal_and_cube_positions(2048)
    assert np.min(np.linalg.norm(cubes - goals, axis=1)) >= panda_env.SUCCESS_THRESHOLD
    assert goals[:, 2] == pytest.approx(np.full(2048, panda_env.CUBE_Z))


def test_mujoco_panda_fixed_eval_cases_use_work_surface_coordinates() -> None:
    cases = json.loads(Path("autoresearch_gym/tasks/panda_pick_and_place_mjwarp_v0/eval_cases.json").read_text(encoding="utf-8"))["cases"]

    assert cases
    for case in cases:
        cube = np.asarray(case["cube_pos"], dtype=np.float32)
        goal = np.asarray(case["goal_pos"], dtype=np.float32)
        assert -0.16 <= cube[0] <= 0.16
        assert -0.16 <= cube[1] <= 0.16
        assert cube[2] == pytest.approx(0.02)
        assert -0.16 <= goal[0] <= 0.16
        assert -0.16 <= goal[1] <= 0.16
        assert goal[2] == pytest.approx(0.02)
        assert np.linalg.norm(cube - goal) >= 0.05
    assert all(float(case["goal_pos"][2]) == pytest.approx(0.02) for case in cases)


def test_mujoco_panda_tqc_her_buffer_samples_same_env_episode_future_goals() -> None:
    from autoresearch_gym.tasks.panda_pick_and_place_mjwarp_v0.seed_trainable_tqc_her_ee import HerReplayBuffer

    rb = HerReplayBuffer(obs_dim=18, act_dim=4, capacity=8, relabel_fraction=1.0)
    obs = np.zeros((4, 18), dtype=np.float32)
    next_obs = np.zeros((4, 18), dtype=np.float32)
    next_obs[:, 3:6] = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    rb.add_batch(
        obs,
        next_obs,
        np.zeros((4, 4), dtype=np.float32),
        np.full(4, -1.0, dtype=np.float32),
        np.asarray([False, True, True, True], dtype=bool),
        env_ids=np.asarray([0, 0, 1, 0], dtype=np.int32),
        episode_ids=np.asarray([0, 0, 0, 1], dtype=np.int64),
        episode_steps=np.asarray([0, 1, 0, 0], dtype=np.int32),
        next_lifted_ever=np.zeros(4, dtype=bool),
    )

    for _ in range(50):
        future = rb._sample_future_indices(np.asarray([0, 1, 2, 3], dtype=np.int64))
        assert future[0] in {0, 1}
        assert future[1] == 1
        assert future[2] == 2
        assert future[3] == 3
        for source, sampled in zip([0, 1, 2, 3], future):
            assert rb.env_ids[sampled] == rb.env_ids[source]
            assert rb.episode_ids[sampled] == rb.episode_ids[source]
            assert rb.episode_steps[sampled] >= rb.episode_steps[source]


def test_mujoco_panda_tqc_her_buffer_samples_only_completed_episodes_and_purges_overwritten_episodes() -> None:
    from autoresearch_gym.tasks.panda_pick_and_place_mjwarp_v0.seed_trainable_tqc_her_ee import HerReplayBuffer

    rb = HerReplayBuffer(obs_dim=18, act_dim=4, capacity=3, relabel_fraction=1.0)
    obs = np.zeros((1, 18), dtype=np.float32)
    action = np.zeros((1, 4), dtype=np.float32)
    reward = np.full(1, -1.0, dtype=np.float32)
    lifted = np.zeros(1, dtype=bool)

    for step in range(2):
        next_obs = np.zeros((1, 18), dtype=np.float32)
        next_obs[:, 3] = float(step + 1)
        rb.add_batch(
            obs,
            next_obs,
            action,
            reward,
            np.asarray([step == 1], dtype=bool),
            env_ids=np.asarray([0], dtype=np.int32),
            episode_ids=np.asarray([0], dtype=np.int64),
            episode_steps=np.asarray([step], dtype=np.int32),
            next_lifted_ever=lifted,
        )

    assert rb.sample_size == 2
    assert rb._sample_future_indices(np.asarray([0, 1], dtype=np.int64))[0] in {0, 1}

    rb.add_batch(
        obs,
        np.zeros((1, 18), dtype=np.float32),
        action,
        reward,
        np.asarray([False], dtype=bool),
        env_ids=np.asarray([0], dtype=np.int32),
        episode_ids=np.asarray([1], dtype=np.int64),
        episode_steps=np.asarray([0], dtype=np.int32),
        next_lifted_ever=lifted,
    )
    rb.add_batch(
        obs,
        np.zeros((1, 18), dtype=np.float32),
        action,
        reward,
        np.asarray([False], dtype=bool),
        env_ids=np.asarray([0], dtype=np.int32),
        episode_ids=np.asarray([1], dtype=np.int64),
        episode_steps=np.asarray([1], dtype=np.int32),
        next_lifted_ever=lifted,
    )

    assert rb.sample_size == 0
    with pytest.raises(RuntimeError, match="no completed episodes"):
        rb.sample(1, __import__("torch").device("cpu"))


def test_mujoco_panda_tqc_her_sample_recomputes_relabel_reward_with_lift_gate() -> None:
    import torch

    from autoresearch_gym.tasks.panda_pick_and_place_mjwarp_v0.seed_trainable_tqc_her_ee import HerReplayBuffer

    def sampled_reward(lifted: bool) -> float:
        rb = HerReplayBuffer(obs_dim=18, act_dim=4, capacity=4, relabel_fraction=1.0)
        obs = np.zeros((1, 18), dtype=np.float32)
        next_obs = np.zeros((1, 18), dtype=np.float32)
        next_obs[:, 3:6] = np.asarray([0.12, 0.0, 0.0], dtype=np.float32)
        rb.add_batch(
            obs,
            next_obs,
            np.zeros((1, 4), dtype=np.float32),
            np.asarray([-1.0], dtype=np.float32),
            np.asarray([True], dtype=bool),
            env_ids=np.asarray([0], dtype=np.int32),
            episode_ids=np.asarray([0], dtype=np.int64),
            episode_steps=np.asarray([0], dtype=np.int32),
            next_lifted_ever=np.asarray([lifted], dtype=bool),
        )
        _obs, _next_obs, _actions, rewards, _dones = rb.sample(1, torch.device("cpu"))
        sampled_next = _next_obs.cpu().numpy()[0]
        assert sampled_next[6:9] == pytest.approx(sampled_next[3:6])
        return float(rewards.cpu().numpy()[0, 0])

    assert sampled_reward(False) == pytest.approx(-1.0)
    assert sampled_reward(True) == pytest.approx(0.0)


def test_mujoco_panda_tqc_seed_reward_requires_lift_gate() -> None:
    from autoresearch_gym.tasks.panda_pick_and_place_mjwarp_v0.seed_trainable_tqc_her_ee import _goal_reward

    next_obs = np.zeros((4, 18), dtype=np.float32)
    next_obs[:, 6:9] = 0.0
    next_obs[0, 3:6] = 0.0
    next_obs[1, 3:6] = 0.0
    next_obs[2, 3:6] = np.asarray([0.1, 0.0, 0.0], dtype=np.float32)
    next_obs[3, 3:6] = np.asarray([0.1, 0.0, 0.0], dtype=np.float32)

    rewards = _goal_reward(next_obs, np.asarray([False, True, False, True]))

    assert rewards[0] == pytest.approx(-1.0)
    assert rewards[1] == pytest.approx(0.0)
    assert rewards[2] == pytest.approx(-1.0)
    assert rewards[3] == pytest.approx(-0.1)
    assert _goal_reward(next_obs[:1])[0] == pytest.approx(-1.0)


def test_mujoco_panda_actuated_arm_indices_exclude_tendon_gripper_when_assets_are_installed() -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("robot_descriptions")
    from autoresearch_gym.envs import mujoco_panda_pick_and_place as panda_env

    try:
        env = gym.make("AutoresearchMujocoPandaPickAndPlaceDense-v0", backend="mujoco", max_steps=2)
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    try:
        obs, info = env.reset(seed=4500)
        unwrapped = env.unwrapped

        assert unwrapped.robot_qpos_adrs.tolist() == list(range(7))
        assert unwrapped.robot_qvel_adrs.tolist() == list(range(7))
        assert unwrapped.finger_qpos_adrs.tolist() == [7, 8]
        assert panda_env._finger_width(unwrapped.data, unwrapped.finger_qpos_adrs) == pytest.approx(0.08)
        assert unwrapped.last_action[7] == pytest.approx(1.0)
        assert obs["observation"].shape == (37,)
        assert not bool(info["gripper_closed_near_cube"])
    finally:
        env.close()


def test_mujoco_panda_tqc_ee_tool_uses_positive_close_convention_when_assets_are_installed() -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("robot_descriptions")
    from autoresearch_gym.tasks.panda_pick_and_place_mjwarp_v0.seed_trainable_tqc_her_ee import (
        EndEffectorDeltaTool,
        flatten_observation,
    )

    try:
        env = gym.make("AutoresearchMujocoPandaPickAndPlaceDense-v0", backend="mujoco", max_steps=2)
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    try:
        obs, _ = env.reset(seed=4500)
        tool = EndEffectorDeltaTool.from_env(env.unwrapped)
        raw_close = tool.single_action(flatten_observation(obs), np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
        raw_open = tool.single_action(flatten_observation(obs), np.asarray([0.0, 0.0, 0.0, -1.0], dtype=np.float32))

        assert raw_close[7] == pytest.approx(-1.0)
        assert raw_open[7] == pytest.approx(1.0)
    finally:
        env.close()


def test_mujoco_panda_benchmarks_select_lift_gated_success_metric() -> None:
    for name in ["benchmark.json", "benchmark_wall_clock.json", "benchmark_large_vector_wall_clock.json"]:
        payload = json.loads(Path("autoresearch_gym/tasks/panda_pick_and_place_mjwarp_v0", name).read_text(encoding="utf-8"))
        assert payload["primary_metric"] == "eval_success_rate"
        assert payload["primary_metric_mode"] == "maximize"


def test_mujoco_pandagym_dense_port_benchmarks_use_pandagym_contract() -> None:
    task_dir = Path("autoresearch_gym/tasks/panda_pick_and_place_mjwarp_pandagym_dense_v0")
    for name in ["benchmark.json", "benchmark_wall_clock.json"]:
        payload = json.loads((task_dir / name).read_text(encoding="utf-8"))
        kwargs = payload["env_kwargs"]
        assert payload["env_id"] == "AutoresearchMujocoPandaGymPickAndPlaceDense-v0"
        assert payload["primary_metric"] == "eval_success_rate"
        assert payload["max_steps"] == 400
        assert kwargs["reward_type"] == "dense"
        assert kwargs["success_requires_lift"] is False
        assert "max_steps" not in kwargs
        assert "max_episode_steps" not in kwargs
        assert kwargs["goal_xy_range"] == pytest.approx(0.30)
        assert kwargs["goal_z_range"] == pytest.approx(0.20)
        assert kwargs["obj_xy_range"] == pytest.approx(0.30)
        assert kwargs["tabletop_goal_probability"] == pytest.approx(0.30)


def test_mujoco_pandagym_dense_port_reward_is_distance_only_when_assets_are_installed() -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("robot_descriptions")

    try:
        env = gym.make("AutoresearchMujocoPandaGymPickAndPlaceDense-v0", backend="mujoco", max_steps=2)
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    try:
        unwrapped = env.unwrapped
        achieved = np.asarray([[0.0, 0.0, 0.02], [0.0, 0.0, 0.02]], dtype=np.float32)
        desired = np.asarray([[0.03, 0.0, 0.02], [0.12, 0.0, 0.02]], dtype=np.float32)
        reward = unwrapped.compute_reward(achieved, desired, {"lifted_ever": np.asarray([False, False])})
        assert np.asarray(reward).tolist() == pytest.approx([-0.03, -0.12])

        unwrapped.goal = desired[0].astype(np.float32)
        obs = {"achieved_goal": achieved[0], "desired_goal": desired[0], "observation": np.zeros(37, dtype=np.float32)}
        info = unwrapped._info(obs)
        assert info["cube_at_goal"] is True
        assert info["is_success"] is True
        assert info["placed_success"] is True
        assert info["lifted_ever"] is False
    finally:
        env.close()


def test_mujoco_pandagym_dense_port_seed_recomputes_pandagym_her_reward() -> None:
    from autoresearch_gym.tasks.panda_pick_and_place_mjwarp_pandagym_dense_v0 import seed_trainable_tqc_her_ee

    obs = np.zeros((2, 43), dtype=np.float32)
    obs[:, 3:6] = np.asarray([[0.0, 0.0, 0.02], [0.0, 0.0, 0.02]], dtype=np.float32)
    obs[:, 6:9] = np.asarray([[0.03, 0.0, 0.02], [0.12, 0.0, 0.02]], dtype=np.float32)
    reward = seed_trainable_tqc_her_ee._goal_reward(obs, np.asarray([False, False]))
    assert reward.tolist() == pytest.approx([-0.03, -0.12])


def test_mujoco_pandagym_dense_guided_warmup_seed_exposes_scripted_controller() -> None:
    from autoresearch_gym.tasks.panda_pick_and_place_mjwarp_pandagym_dense_v0 import seed_trainable_guided_warmup

    candidate = seed_trainable_guided_warmup.get_candidate()
    recipe = candidate["recipe"]

    assert recipe["runner"]["sample_trajectory_source"] == SAMPLE_TRAJECTORY_SOURCE_CANDIDATE_PROVIDED
    assert 0.0 < seed_trainable_guided_warmup.SCRIPTED_WARMUP_FRACTION < 1.0
    assert seed_trainable_guided_warmup.SCRIPTED_WARMUP_STEPS > 0
    phases = seed_trainable_guided_warmup.SCRIPTED_PHASES
    assert phases.index("hover_cube") < phases.index("close") < phases.index("open")

    obs = np.zeros((2, 43), dtype=np.float32)
    obs[:, 3:6] = np.asarray([[0.05, -0.02, 0.02], [0.05, -0.02, 0.02]], dtype=np.float32)
    obs[:, 6:9] = np.asarray([[0.10, 0.04, 0.02], [0.10, 0.04, 0.10]], dtype=np.float32)
    phases = np.asarray([0, 2], dtype=np.int32)
    phase_steps = np.zeros(2, dtype=np.int32)

    target, gripper = seed_trainable_guided_warmup._scripted_targets(obs, phases, phase_steps)

    assert target[0, :2].tolist() == pytest.approx(obs[0, 3:5].tolist())
    assert target[0, 2] == pytest.approx(seed_trainable_guided_warmup.SCRIPTED_CONFIG["hover_z"])
    assert target[1, :2].tolist() == pytest.approx(obs[1, 3:5].tolist())
    assert target[1, 2] == pytest.approx(seed_trainable_guided_warmup.SCRIPTED_CONFIG["grasp_z"])
    assert gripper.tolist() == pytest.approx([-1.0, 1.0])


def test_mujoco_pandagym_dense_guided_warmup_reward_biases_lift_over_push() -> None:
    from autoresearch_gym.tasks.panda_pick_and_place_mjwarp_pandagym_dense_v0 import seed_trainable_guided_warmup

    obs = np.zeros((2, 43), dtype=np.float32)
    obs[0, 0:3] = np.asarray([0.0, 0.0, 0.02], dtype=np.float32)
    obs[0, 3:6] = np.asarray([0.0, 0.0, 0.02], dtype=np.float32)
    obs[0, 6:9] = np.asarray([0.0, 0.0, 0.02], dtype=np.float32)
    obs[1, 0:3] = np.asarray([0.0, 0.0, 0.08], dtype=np.float32)
    obs[1, 3:6] = np.asarray([0.0, 0.0, 0.08], dtype=np.float32)
    obs[1, 6:9] = np.asarray([0.0, 0.0, 0.08], dtype=np.float32)

    reward = seed_trainable_guided_warmup._goal_reward(obs, lifted_ever=np.asarray([False, True]))

    assert reward[1] > reward[0]
    assert reward[1] - reward[0] == pytest.approx(
        seed_trainable_guided_warmup.LIFT_REWARD_WEIGHT * 0.06
        + seed_trainable_guided_warmup.LIFTED_EVER_BONUS
        + seed_trainable_guided_warmup.UNLIFTED_GOAL_PENALTY
    )


def test_mujoco_pandagym_dense_guided_warmup_sampling_uses_scripted_actions() -> None:
    from autoresearch_gym.tasks.panda_pick_and_place_mjwarp_pandagym_dense_v0 import seed_trainable_guided_warmup

    class FailingActor:
        def get_action(self, obs):
            raise AssertionError("policy actor should not be used for scripted warmup sampling")

    class FakeGuidedEnv(gym.Env):
        def __init__(self) -> None:
            self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(43,), dtype=np.float32)
            self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
            self.actions: list[np.ndarray] = []
            self.steps = 0

        def reset(self, *, seed=None, options=None):
            del seed, options
            self.steps = 0
            return self._obs(), {}

        def step(self, action):
            action = np.asarray(action, dtype=np.float32)
            assert action.shape == (4,)
            self.actions.append(action.copy())
            self.steps += 1
            return self._obs(), -1.0, self.steps >= 2, False, {"lifted_ever": False}

        def render(self):
            return np.full((8, 8, 3), self.steps, dtype=np.uint8)

        def _obs(self) -> np.ndarray:
            obs = np.zeros(43, dtype=np.float32)
            obs[0:3] = np.asarray([0.0, 0.0, 0.10], dtype=np.float32)
            obs[3:6] = np.asarray([0.05, -0.02, 0.02], dtype=np.float32)
            obs[6:9] = np.asarray([0.10, 0.04, 0.02], dtype=np.float32)
            return obs

    env = FakeGuidedEnv()

    sampled = seed_trainable_guided_warmup._sample_policy_trajectory(
        types.SimpleNamespace(device="cpu", actor=FailingActor()),
        lambda control_type=None, reward_recipe=None: env,
        types.SimpleNamespace(max_steps=3, eval_seed_start=4500, train_seed=1),
        {"episode": 1, "sample_index": 1, "frame_stride": 1, "playback_fps": 20.0},
        scripted=True,
    )

    assert sampled["metadata"]["playback_source"] == "guided_warmup"
    assert sampled["frames"]
    assert env.actions, "scripted warmup sampling did not step the fake env"
    assert env.actions[0][3] == pytest.approx(-1.0)
    assert env.actions[0][:3].tolist() == pytest.approx(
        seed_trainable_guided_warmup.ScriptedE2EWarmupState(1).actions(env._obs().reshape(1, -1))[0][:3].tolist()
    )


def test_mujoco_panda_pick_and_place_seed_task_resets_and_steps_when_assets_are_installed() -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("robot_descriptions")
    from autoresearch_gym.tasks.panda_pick_and_place_mjwarp_v0.seed_trainable import RewardRecipeWrapper

    try:
        env = RewardRecipeWrapper(
            gym.make("AutoresearchMujocoPandaPickAndPlaceDense-v0", render_mode="rgb_array", backend="mujoco", max_steps=4),
            "subskill_curriculum",
        )
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    obs, info = env.reset(seed=123)
    assert obs.shape == env.observation_space.shape
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)
    assert next_obs.shape == env.observation_space.shape
    assert isinstance(float(reward), float)
    assert isinstance(bool(terminated), bool)
    assert isinstance(bool(truncated), bool)
    assert "ee_to_cube_distance" in info
    assert "ee_to_cube_progress" in info
    assert "initial_ee_to_cube_distance" in info
    env.close()


def test_mujoco_so101_reach_seed_uses_wall_clock_time_source() -> None:
    source = Path("autoresearch_gym/tasks/so101_reach_mujoco_v0/seed_trainable.py").read_text(encoding="utf-8")

    assert "started_at = time.time()" in source
    assert "started_at = time.perf_counter()" not in source


def test_mujoco_so101_reach_seed_is_cleanrl_style_sac() -> None:
    seed = importlib.import_module("autoresearch_gym.tasks.so101_reach_mujoco_v0.seed_trainable")
    candidate = seed.get_candidate()

    assert seed.ALGORITHM == "sac"
    assert candidate["recipe"]["algorithm"] == "sac"
    assert hasattr(seed, "ReplayBuffer")
    assert hasattr(seed.Agent, "update")


def test_mujoco_so101_default_seeds_are_real_training_recipes() -> None:
    module_names = [
        "autoresearch_gym.tasks.so101_reach_mujoco_v0.seed_trainable",
        "autoresearch_gym.tasks.so101_cube_to_bin_mujoco_v0.seed_trainable",
        "autoresearch_gym.tasks.so101_vial_to_rack_mujoco_v0.seed_trainable",
    ]

    for module_name in module_names:
        seed = importlib.import_module(module_name)
        candidate = seed.get_candidate()
        assert candidate["recipe"]["algorithm"] == "sac"
        assert hasattr(seed, "ReplayBuffer")
        assert hasattr(seed.Agent, "update")


def test_mujoco_so101_seeds_pass_agent_to_live_callback() -> None:
    seed_paths = [
        Path("autoresearch_gym/tasks/so101_reach_mujoco_v0/seed_trainable.py"),
        Path("autoresearch_gym/tasks/so101_reach_mujoco_v0/seed_trainable_vectorized.py"),
        Path("autoresearch_gym/tasks/so101_cube_to_bin_mujoco_v0/seed_trainable_vectorized.py"),
        Path("autoresearch_gym/tasks/so101_vial_to_rack_mujoco_v0/seed_trainable_vectorized.py"),
        Path("autoresearch_gym/tasks/so101_pixel_actor_critic_seed.py"),
    ]

    for seed_path in seed_paths:
        seed_source = seed_path.read_text(encoding="utf-8")
        assert "agent=agent" in seed_source or "seed_trainable_vectorized import" in seed_source


def test_mujoco_so101_no_scripted_default_seed_remains() -> None:
    so101_sources = [
        path.read_text(encoding="utf-8")
        for path in Path("autoresearch_gym/tasks").glob("so101_*_mujoco_v0/seed_trainable.py")
    ]

    assert not Path("autoresearch_gym/tasks/so101_scripted_pick_place_seed.py").exists()
    assert all("so101_scripted_pick_place_seed" not in source for source in so101_sources)
    assert all("so101_scripted_pick_place_baseline" not in source for source in so101_sources)


def test_visual_sampling_derives_wall_clock_interval_from_target_samples() -> None:
    control = {
        "trajectory_sample_rate": 0.05,
        "trajectory_target_samples": 24,
        "trajectory_sample_interval_seconds": 0.0,
    }

    assert trajectory_sample_interval_seconds(control, 1200.0) == pytest.approx(50.0)
    assert should_sample_visual_episode(
        1,
        control,
        elapsed_seconds=0.0,
        last_sample_elapsed=None,
        train_seconds=1200.0,
    )
    assert not should_sample_visual_episode(
        2,
        control,
        elapsed_seconds=10.0,
        last_sample_elapsed=0.0,
        train_seconds=1200.0,
    )
    assert should_sample_visual_episode(
        3,
        control,
        elapsed_seconds=51.0,
        last_sample_elapsed=0.0,
        train_seconds=1200.0,
    )


def test_mujoco_so101_scene_exposes_upstream_joint_names_and_limits() -> None:
    pytest.importorskip("mujoco")
    from autoresearch_gym.envs import mujoco_so101_reach as so101_env

    env = gym.make("AutoresearchMujocoSO101Reach-v0", render_mode="rgb_array", max_steps=4)
    try:
        model = env.unwrapped.model
        mujoco = env.unwrapped.mujoco
        for name in so101_env.JOINT_NAMES:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            assert joint_id >= 0
            assert actuator_id >= 0

        ctrl_ranges = np.asarray(model.actuator_ctrlrange)
        assert ctrl_ranges[0] == pytest.approx([-1.91986, 1.91986])
        assert ctrl_ranges[1] == pytest.approx([-1.74533, 1.74533])
        assert ctrl_ranges[2] == pytest.approx([-1.69, 1.69])
        assert ctrl_ranges[3] == pytest.approx([-1.65806, 1.65806])
        assert ctrl_ranges[4] == pytest.approx([-2.74385, 2.84121])
        assert ctrl_ranges[5] == pytest.approx([-0.17453, 1.74533])
    finally:
        env.close()


def test_mujoco_so101_reach_fails_when_real_assets_are_missing(monkeypatch) -> None:
    pytest.importorskip("mujoco")
    from autoresearch_gym.envs import mujoco_so101_reach as so101_env

    monkeypatch.setattr(so101_env, "resolve_so101_xml_path", lambda model_path=None: None)

    with pytest.raises(FileNotFoundError, match="real RobotStudio/Menagerie model"):
        so101_env.AutoresearchMujocoSO101ReachEnv(render_mode="rgb_array")


def test_mujoco_so101_reach_contract_matches_nexus_reach_defaults() -> None:
    from autoresearch_gym.envs import mujoco_so101_reach as so101_env

    assert so101_env.SUCCESS_THRESHOLD == pytest.approx(0.02)
    assert so101_env.TARGET_WORKSPACE_HALF_EXTENT == pytest.approx(0.15)
    np.testing.assert_allclose(so101_env.TARGET_LOW, np.asarray([0.17, -0.15, 0.03], dtype=np.float32))
    np.testing.assert_allclose(so101_env.TARGET_HIGH, np.asarray([0.47, 0.15, 0.33], dtype=np.float32))


def test_mujoco_so101_mjwarp_profile_marks_approximations() -> None:
    from autoresearch_gym.envs import mujoco_so101_reach as so101_env

    profile = so101_env.SO101_MJWARP_GUESSED_PHYSICS_PROFILE
    assert profile["calibration_source"] == "engineering_guess"
    assert profile["robot_asset_source"] == "mujoco_menagerie_robotstudio_so101"
    assert profile["actuator_profile"]["motor_family"].startswith("Feetech")
    assert profile["printed_gripper_profile"]["material_assumption"].startswith("PLA+")
    assert profile["printed_gripper_profile"]["contact_defaults"]["condim"] == 4


def test_mujoco_so101_mjwarp_benchmark_is_opt_in_and_uses_real_assets() -> None:
    benchmarks = {
        "AutoresearchMujocoSO101Reach-v0": Path("autoresearch_gym/tasks/so101_reach_mujoco_v0/benchmark_mjwarp_wall_clock.json"),
        "AutoresearchMujocoSO101CubeToBin-v0": Path("autoresearch_gym/tasks/so101_cube_to_bin_mujoco_v0/benchmark_mjwarp_wall_clock.json"),
        "AutoresearchMujocoSO101VialToRack-v0": Path("autoresearch_gym/tasks/so101_vial_to_rack_mujoco_v0/benchmark_mjwarp_wall_clock.json"),
    }

    for env_id, path in benchmarks.items():
        benchmark = json.loads(path.read_text(encoding="utf-8"))
        assert benchmark["env_id"] == env_id
        assert benchmark["env_kwargs"]["backend"] == "mujoco_warp"
        assert benchmark["env_kwargs"]["num_envs"] == 64
        assert benchmark["env_kwargs"]["physics_profile"] == "so101_follower_feetech_plastic_guess_v0"
        notes = " ".join(benchmark["notes"]).lower()
        assert "real mujoco menagerie" in notes
        assert "engineering-guess" in notes
        assert "cpu mujoco" in notes
        assert "fake" not in notes
        if env_id == "AutoresearchMujocoSO101Reach-v0":
            assert benchmark["max_steps"] == 150
        else:
            assert benchmark["max_steps"] == 512
            assert benchmark["env_kwargs"]["printed_gripper_profile"] == "so101_printed_plastic_gripper_guess_v0"
            assert "so101-nexus" in notes


def test_mujoco_so101_vectorized_seed_wires_mjwarp_collector() -> None:
    seed = importlib.import_module("autoresearch_gym.tasks.so101_reach_mujoco_v0.seed_trainable_vectorized")
    source = Path("autoresearch_gym/tasks/so101_reach_mujoco_v0/seed_trainable_vectorized.py").read_text(
        encoding="utf-8"
    )
    candidate = seed.get_candidate()

    assert "AutoresearchMujocoSO101ReachWarpVectorEnv" in source
    assert "reset_worlds(dones)" in source
    assert "_FeetechDeltaRenderController" in source
    assert "cpu_mujoco_feetech_delta_replay" in source
    assert seed.MJWARP_NUM_ENVS == 64
    assert candidate["recipe"]["mjwarp_physics_profile"] == "so101_follower_feetech_plastic_guess_v0"
    assert candidate["recipe"]["runner"]["sample_trajectory_source"] == SAMPLE_TRAJECTORY_SOURCE_CANDIDATE_PROVIDED
    assert callable(getattr(seed, "_answer_sampled_trajectory_request", None))


def test_mujoco_so101_manip_reward_matches_nexus_shape() -> None:
    from autoresearch_gym.envs.mujoco_so101_pick_place import (
        NEXUS_REWARD_COMPLETION_BONUS,
        NEXUS_REWARD_GRASPING,
        NEXUS_REWARD_REACHING,
        NEXUS_REWARD_SOURCE,
        NEXUS_REWARD_TASK_OBJECTIVE,
        _nexus_pick_place_reward,
        _nexus_reach_progress,
    )

    far = float(_nexus_pick_place_reward(tcp_to_obj_dist=10.0, obj_to_target_dist=10.0, is_grasped=False, is_complete=False))
    reach_only = float(_nexus_pick_place_reward(tcp_to_obj_dist=0.0, obj_to_target_dist=10.0, is_grasped=False, is_complete=False))
    grasped = float(_nexus_pick_place_reward(tcp_to_obj_dist=0.0, obj_to_target_dist=10.0, is_grasped=True, is_complete=False))
    placed = float(_nexus_pick_place_reward(tcp_to_obj_dist=0.0, obj_to_target_dist=0.0, is_grasped=True, is_complete=True))

    assert "SO101-Nexus" in NEXUS_REWARD_SOURCE
    assert far == pytest.approx(0.0, abs=1e-5)
    assert reach_only == pytest.approx(NEXUS_REWARD_REACHING)
    assert grasped == pytest.approx(NEXUS_REWARD_REACHING + NEXUS_REWARD_GRASPING)
    assert placed == pytest.approx(
        NEXUS_REWARD_REACHING + NEXUS_REWARD_GRASPING + NEXUS_REWARD_TASK_OBJECTIVE + NEXUS_REWARD_COMPLETION_BONUS
    )
    assert float(_nexus_reach_progress(0.0)) == pytest.approx(1.0)


def test_mujoco_so101_manipulation_vectorized_seeds_use_mjwarp_physics() -> None:
    task_modules = {
        "cube_to_bin": "autoresearch_gym.tasks.so101_cube_to_bin_mujoco_v0.seed_trainable_vectorized",
        "vial_to_rack": "autoresearch_gym.tasks.so101_vial_to_rack_mujoco_v0.seed_trainable_vectorized",
    }
    shared_source = Path("autoresearch_gym/tasks/so101_reach_mujoco_v0/seed_trainable_vectorized.py").read_text(
        encoding="utf-8"
    )

    assert "AutoresearchMujocoSO101PickPlaceWarpVectorEnv" in shared_source
    assert "_mjwarp_task_kind" in shared_source
    for task, module_name in task_modules.items():
        seed = importlib.import_module(module_name)
        candidate = seed.get_candidate()
        assert candidate["recipe"]["algorithm"] == "sac"
        assert candidate["recipe"]["task"] == task
        assert candidate["recipe"]["control"] == "mjwarp_feetech_delta_targets"
        assert candidate["recipe"]["mjwarp_physics_profile"] == "so101_follower_feetech_plastic_guess_v0"
        assert hasattr(seed.Agent, "update")


def test_mujoco_so101_manipulation_vectorized_seeds_are_mutable_task_local_recipes() -> None:
    for path in (
        Path("autoresearch_gym/tasks/so101_cube_to_bin_mujoco_v0/seed_trainable_vectorized.py"),
        Path("autoresearch_gym/tasks/so101_vial_to_rack_mujoco_v0/seed_trainable_vectorized.py"),
    ):
        source = path.read_text(encoding="utf-8")
        assert "def train_agent(" in source
        assert "def _make_mjwarp_vector_env(" in source
        assert "from autoresearch_gym.tasks.so101_reach_mujoco_v0.seed_trainable_vectorized import" not in source


def test_mujoco_so101_mjwarp_sampled_trajectory_replays_world_and_wrist_feeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed = importlib.import_module("autoresearch_gym.tasks.so101_reach_mujoco_v0.seed_trainable_vectorized")

    class FakeAgent:
        def act(self, obs, deterministic: bool = False):
            del obs, deterministic
            return np.asarray([0.5, -0.25], dtype=np.float32)

    class FakeReplayEnv(gym.Env):
        metadata = {"render_modes": ["rgb_array"]}

        def __init__(self, **kwargs) -> None:
            self.kwargs = dict(kwargs)
            self.observation_space = gym.spaces.Dict(
                {
                    "observation": gym.spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32),
                    "achieved_goal": gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                    "desired_goal": gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                }
            )
            self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
            self.model = types.SimpleNamespace(nu=2)
            self.data = types.SimpleNamespace(ctrl=np.zeros(2, dtype=np.float32))
            self.ctrl_low = np.full(2, -1.0, dtype=np.float32)
            self.ctrl_high = np.full(2, 1.0, dtype=np.float32)
            self.frame_skip = 1
            self.max_steps = 3
            self.step_count = 0
            self.last_action = np.zeros(2, dtype=np.float32)
            self.mujoco = types.SimpleNamespace(mj_step=lambda model, data: None)
            self.closed = False

        def reset(self, *, seed=None, options=None):
            del seed, options
            self.step_count = 0
            self.data.ctrl[:] = 0.0
            return self._get_obs(), {}

        def _get_obs(self):
            return {
                "observation": np.asarray([self.step_count, float(self.data.ctrl[0])], dtype=np.float32),
                "achieved_goal": np.asarray([0.1 + 0.01 * self.step_count, 0.0, 0.1], dtype=np.float32),
                "desired_goal": np.asarray([0.2, 0.0, 0.1], dtype=np.float32),
            }

        def _info(self, obs):
            distance = float(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"]))
            return {
                "ee_to_target_distance": distance,
                "is_success": distance < 0.02,
            }

        def compute_reward(self, achieved_goal, desired_goal, info):
            del info
            return -float(np.linalg.norm(achieved_goal - desired_goal))

        def sample_feeds(self):
            return {
                "world": np.full((12, 16, 3), 40 + self.step_count, dtype=np.uint8),
                "wrist": np.full((8, 8, 3), 140 + self.step_count, dtype=np.uint8),
            }

        def sample_feed_specs(self):
            return {
                "world": {"role": "debug", "shape": [12, 16, 3]},
                "wrist": {"role": "policy_input", "shape": [8, 8, 3]},
            }

        def render(self, *args, **kwargs):
            del args, kwargs
            return np.full((12, 16, 3), 90 + self.step_count, dtype=np.uint8)

        def close(self):
            self.closed = True

    made_envs: list[FakeReplayEnv] = []

    def fake_make(env_id: str, **kwargs):
        assert env_id == "AutoresearchMujocoSO101Reach-v0"
        env = FakeReplayEnv(**kwargs)
        made_envs.append(env)
        return env

    def env_factory(*args, **kwargs):
        raise AssertionError("MJWarp sampled trajectory rendering must use CPU MuJoCo replay, not the training env factory")

    monkeypatch.setattr(seed.gym, "make", fake_make)
    benchmark = types.SimpleNamespace(
        env_id="AutoresearchMujocoSO101Reach-v0",
        env_kwargs={"render_mode": "rgb_array", "backend": "mujoco_warp", "num_envs": 64},
        max_steps=3,
        train_seed=101,
    )
    request = {
        "requested": True,
        "episode": 2,
        "sample_index": 1,
        "source": SAMPLE_TRAJECTORY_SOURCE_CANDIDATE_PROVIDED,
        "frame_stride": 1,
        "playback_fps": 20.0,
    }

    payload = seed._make_custom_sampled_trajectory(
        request=request,
        agent=FakeAgent(),
        env_factory=env_factory,
        benchmark=benchmark,
    )

    assert payload["source"] == SAMPLE_TRAJECTORY_SOURCE_CANDIDATE_PROVIDED
    assert payload["metadata"]["sampled_trajectory_render_backend"] == "cpu_mujoco_feetech_delta_replay"
    assert payload["metadata"]["control"] == "mjwarp_feetech_delta_targets"
    assert payload["feed_specs"]["world"]["role"] == "debug"
    assert payload["feed_specs"]["wrist"]["role"] == "policy_input"
    assert payload["frames"] and payload["frames"][0].shape == (12, 16, 3)
    assert payload["steps"] and set(payload["steps"][0]["feeds"]) == {"world", "wrist"}
    assert payload["steps"][0]["feeds"]["world"].shape == (12, 16, 3)
    assert payload["steps"][0]["feeds"]["wrist"].shape == (8, 8, 3)
    assert made_envs and made_envs[0].closed


def test_mujoco_so101_mjwarp_vector_env_contract_with_stub_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("mujoco")
    from autoresearch_gym.envs.mujoco_so101_reach import AutoresearchMujocoSO101ReachWarpVectorEnv

    class _ArrayField:
        def __init__(self, value: np.ndarray) -> None:
            self.value = np.asarray(value, dtype=np.float32).copy()

        def assign(self, value: np.ndarray) -> None:
            self.value = np.asarray(value, dtype=np.float32).copy()

        def numpy(self) -> np.ndarray:
            return self.value.copy()

    class _WarpData:
        def __init__(self, model, data, nworld: int) -> None:
            self.qpos = _ArrayField(np.repeat(np.asarray(data.qpos, dtype=np.float32)[None, :], nworld, axis=0))
            self.qvel = _ArrayField(np.zeros((nworld, model.nv), dtype=np.float32))
            self.ctrl = _ArrayField(np.zeros((nworld, model.nu), dtype=np.float32))
            site_xpos = np.zeros((nworld, model.nsite, 3), dtype=np.float32)
            site_xpos[:] = np.asarray([0.22, 0.0, 0.17], dtype=np.float32)
            self.site_xpos = _ArrayField(site_xpos)
            self.xpos = _ArrayField(np.zeros((nworld, model.nbody, 3), dtype=np.float32))

    class _StubWarp:
        @staticmethod
        def put_model(model):
            return model

        @staticmethod
        def put_data(model, data, nworld: int, nconmax: int | None = None, njmax: int | None = None):
            del nconmax, njmax
            return _WarpData(model, data, nworld)

        @staticmethod
        def forward(model, data) -> None:
            del model, data

        @staticmethod
        def step(model, data) -> None:
            del model
            site_xpos = data.site_xpos.numpy()
            site_xpos[:, :, 0] += 0.001 * np.tanh(np.sum(data.ctrl.numpy(), axis=1))[:, None]
            data.site_xpos.assign(site_xpos)

    monkeypatch.setitem(sys.modules, "mujoco_warp", _StubWarp)
    env = AutoresearchMujocoSO101ReachWarpVectorEnv(num_envs=2, seed=123, max_steps=2, frame_skip=1)
    try:
        obs = env.reset(seed=123)
        assert obs.shape == (2, *env.observation_space.shape)
        before = env.applied_ctrl.copy()
        action = np.ones((2, env.action_space.shape[0]), dtype=np.float32)
        next_obs, reward, terminated, truncated, info = env.step(action)
        assert next_obs.shape == (2, *env.observation_space.shape)
        assert reward.shape == (2,)
        assert terminated.shape == (2,)
        assert truncated.shape == (2,)
        assert "ee_to_target_distance" in info
        assert env.physics_profile_metadata()["calibration_source"] == "engineering_guess"
        assert np.any(env.applied_ctrl != before)
        assert np.any(np.abs(env.ctrl_targets - env.applied_ctrl) > 0.0)
        reset_obs = env.reset_worlds(np.asarray([True, False]))
        assert reset_obs.shape == (2, *env.observation_space.shape)
    finally:
        env.close()


def test_mujoco_so101_manipulation_mjwarp_vector_env_contract_with_stub_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("mujoco")
    from autoresearch_gym.envs.mujoco_so101_pick_place import AutoresearchMujocoSO101PickPlaceWarpVectorEnv

    class _ArrayField:
        def __init__(self, value: np.ndarray) -> None:
            self.value = np.asarray(value, dtype=np.float32).copy()

        def assign(self, value: np.ndarray) -> None:
            self.value = np.asarray(value, dtype=np.float32).copy()

        def numpy(self) -> np.ndarray:
            return self.value.copy()

    class _WarpData:
        def __init__(self, model, data, nworld: int) -> None:
            self.qpos = _ArrayField(np.repeat(np.asarray(data.qpos, dtype=np.float32)[None, :], nworld, axis=0))
            self.qvel = _ArrayField(np.zeros((nworld, model.nv), dtype=np.float32))
            self.ctrl = _ArrayField(np.zeros((nworld, model.nu), dtype=np.float32))
            site_xpos = np.zeros((nworld, model.nsite, 3), dtype=np.float32)
            site_xpos[:] = np.asarray([0.22, 0.0, 0.17], dtype=np.float32)
            self.site_xpos = _ArrayField(site_xpos)
            self.xpos = _ArrayField(np.zeros((nworld, model.nbody, 3), dtype=np.float32))

    class _StubWarp:
        @staticmethod
        def put_model(model):
            return model

        @staticmethod
        def put_data(model, data, nworld: int, nconmax: int | None = None, njmax: int | None = None):
            del nconmax, njmax
            return _WarpData(model, data, nworld)

        @staticmethod
        def forward(model, data) -> None:
            del model, data

        @staticmethod
        def step(model, data) -> None:
            del model
            qpos = data.qpos.numpy()
            qpos[:, : min(2, qpos.shape[1])] += 0.0005
            data.qpos.assign(qpos)

    monkeypatch.setitem(sys.modules, "mujoco_warp", _StubWarp)
    for task_kind, metric_name in (("cube_to_bin", "cube_to_bin_distance"), ("vial_to_rack", "vial_to_slot_distance")):
        try:
            env = AutoresearchMujocoSO101PickPlaceWarpVectorEnv(
                task_kind=task_kind,
                num_envs=2,
                seed=123,
                max_steps=2,
                frame_skip=1,
            )
        except FileNotFoundError as exc:
            pytest.skip(str(exc))
        try:
            obs = env.reset(seed=123)
            assert obs.shape == (2, *env.observation_space.shape)
            action = np.zeros((2, env.action_space.shape[0]), dtype=np.float32)
            next_obs, reward, terminated, truncated, info = env.step(action)
            assert next_obs.shape == (2, *env.observation_space.shape)
            assert reward.shape == (2,)
            assert terminated.shape == (2,)
            assert truncated.shape == (2,)
            assert metric_name in info
            assert info["printed_gripper_profile"][0] == "so101_printed_plastic_gripper_guess_v0"
            metadata = env.physics_profile_metadata()
            assert metadata["task_kind"] == task_kind
            assert metadata["printed_gripper_profile"]["calibration_source"] == "engineering_guess"
            reset_obs = env.reset_worlds(np.asarray([True, False]))
            assert reset_obs.shape == (2, *env.observation_space.shape)
        finally:
            env.close()


def test_mujoco_so101_mjwarp_vector_env_reset_step_when_available() -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("mujoco_warp")
    from autoresearch_gym.envs.mujoco_so101_reach import AutoresearchMujocoSO101ReachWarpVectorEnv

    try:
        env = AutoresearchMujocoSO101ReachWarpVectorEnv(num_envs=2, seed=123, max_steps=2, frame_skip=1)
    except (FileNotFoundError, RuntimeError) as exc:
        pytest.skip(str(exc))
    try:
        obs = env.reset(seed=123)
        assert obs.shape == (2, *env.observation_space.shape)
        action = np.zeros((2, env.action_space.shape[0]), dtype=np.float32)
        next_obs, reward, terminated, truncated, info = env.step(action)
        assert next_obs.shape == (2, *env.observation_space.shape)
        assert reward.shape == (2,)
        assert terminated.shape == (2,)
        assert truncated.shape == (2,)
        assert "ee_to_target_distance" in info
        assert env.physics_profile_metadata()["calibration_source"] == "engineering_guess"
    finally:
        env.close()


def test_mujoco_so101_reach_fixed_case_reset_step_and_render() -> None:
    pytest.importorskip("mujoco")

    env = gym.make("AutoresearchMujocoSO101Reach-v0", render_mode="rgb_array", max_steps=4)
    try:
        target = np.asarray([0.32, 0.0, 0.18], dtype=np.float32)
        obs, info = env.reset(seed=123, options={"fixed_case": {"target_pos": target.tolist()}})
        assert obs["desired_goal"] == pytest.approx(target)
        assert obs["achieved_goal"].shape == (3,)
        assert "ee_to_target_distance" in info

        next_obs, reward, terminated, truncated, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        assert next_obs["observation"].shape == env.observation_space["observation"].shape
        assert isinstance(float(reward), float)
        assert isinstance(bool(terminated), bool)
        assert isinstance(bool(truncated), bool)
        assert "ee_to_target_progress" in info

        frame = env.render()
        assert isinstance(frame, np.ndarray)
        assert frame.dtype == np.uint8
        assert frame.shape == (480, 720, 3)
        assert float(np.std(frame)) > 0.0
    finally:
        env.close()


def test_mujoco_so101_reach_seed_task_resets_and_steps_when_mujoco_is_installed() -> None:
    pytest.importorskip("mujoco")
    from autoresearch_gym.tasks.so101_reach_mujoco_v0.seed_trainable import RewardRecipeWrapper

    env = RewardRecipeWrapper(gym.make("AutoresearchMujocoSO101Reach-v0", render_mode="rgb_array", max_steps=4), "task_dense")
    obs, info = env.reset(seed=123)
    assert obs.shape == env.observation_space.shape
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)
    assert next_obs.shape == env.observation_space.shape
    assert isinstance(float(reward), float)
    assert isinstance(bool(terminated), bool)
    assert isinstance(bool(truncated), bool)
    assert "ee_to_target_distance" in info
    assert "initial_ee_to_target_distance" in info
    env.close()


def test_mujoco_so101_reach_sample_feeds_include_world_and_wrist() -> None:
    pytest.importorskip("mujoco")

    env = gym.make("AutoresearchMujocoSO101Reach-v0", render_mode="rgb_array", max_steps=4)
    try:
        env.reset(seed=123)
        specs = env.unwrapped.sample_feed_specs()
        assert specs["world"]["role"] == "debug"
        assert specs["wrist"]["role"] == "policy_input"
        feeds = env.unwrapped.sample_feeds()
        assert set(feeds) == {"world", "wrist"}
        assert feeds["world"].shape == (480, 720, 3)
        assert feeds["wrist"].shape == (128, 128, 3)
        assert float(np.std(feeds["world"])) > 0.0
        assert float(np.std(feeds["wrist"])) > 0.0
    finally:
        env.close()


@pytest.mark.parametrize(
    ("env_id", "fixed_case", "metric_prefix"),
    [
        (
            "AutoresearchMujocoSO101CubeToBin-v0",
            {"object_pos": [0.27, -0.06, 0.0125], "target_pos": [0.36, 0.065, 0.03]},
            "cube_to_bin",
        ),
        (
            "AutoresearchMujocoSO101VialToRack-v0",
            {"object_pos": [0.28, -0.055, 0.045], "target_pos": [0.35, 0.055, 0.104]},
            "vial_to_slot",
        ),
    ],
)
def test_mujoco_so101_pick_place_fixed_case_reset_step_and_render(
    env_id: str,
    fixed_case: dict[str, list[float]],
    metric_prefix: str,
) -> None:
    pytest.importorskip("mujoco")

    env = gym.make(env_id, render_mode="rgb_array", max_steps=4)
    try:
        obs, info = env.reset(seed=123, options={"fixed_case": fixed_case})
        assert obs["desired_goal"] == pytest.approx(np.asarray(fixed_case["target_pos"], dtype=np.float32))
        assert obs["achieved_goal"].shape == (3,)
        assert f"{metric_prefix}_distance" in info
        if env_id == "AutoresearchMujocoSO101CubeToBin-v0":
            from autoresearch_gym.envs.mujoco_so101_pick_place import CUBE_HALF_EXTENTS, CUBE_MASS

            cube_geom_id = env.unwrapped.mujoco.mj_name2id(
                env.unwrapped.model,
                env.unwrapped.mujoco.mjtObj.mjOBJ_GEOM,
                "cube",
            )
            assert cube_geom_id >= 0
            assert env.unwrapped.model.geom_size[cube_geom_id] == pytest.approx(CUBE_HALF_EXTENTS)
            assert env.unwrapped.model.body_mass[env.unwrapped.object_body_id] == pytest.approx(CUBE_MASS)

        next_obs, reward, terminated, truncated, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        assert next_obs["observation"].shape == env.observation_space["observation"].shape
        assert isinstance(float(reward), float)
        assert 0.0 <= float(reward) <= 1.0
        assert isinstance(bool(terminated), bool)
        assert isinstance(bool(truncated), bool)
        assert f"{metric_prefix}_progress" in info
        assert "object_to_target_distance" in info
        assert "ee_to_object_distance" in info
        assert "is_grasped" in info
        assert "is_robot_static" in info
        assert "lift_height" in info
        assert "SO101-Nexus" in info["reward_source"]
        if env_id == "AutoresearchMujocoSO101VialToRack-v0":
            assert "vial_uprightness" in info
            assert "vial_height_error" in info

        frame = env.render()
        assert isinstance(frame, np.ndarray)
        assert frame.dtype == np.uint8
        assert frame.shape == (480, 720, 3)
        assert float(np.std(frame)) > 0.0
    finally:
        env.close()


def test_mujoco_so101_pick_place_fails_when_real_assets_are_missing(monkeypatch) -> None:
    pytest.importorskip("mujoco")
    from autoresearch_gym.envs import mujoco_so101_pick_place as so101_pick_place

    monkeypatch.setattr(so101_pick_place, "resolve_so101_xml_path", lambda model_path=None: None)

    with pytest.raises(FileNotFoundError, match="real RobotStudio/Menagerie model"):
        so101_pick_place.AutoresearchMujocoSO101CubeToBinEnv(render_mode="rgb_array")


@pytest.mark.parametrize(
    ("env_id", "seed_module", "expected_info_key"),
    [
        (
            "AutoresearchMujocoSO101CubeToBin-v0",
            "autoresearch_gym.tasks.so101_cube_to_bin_mujoco_v0.seed_trainable",
            "cube_to_bin_distance",
        ),
        (
            "AutoresearchMujocoSO101VialToRack-v0",
            "autoresearch_gym.tasks.so101_vial_to_rack_mujoco_v0.seed_trainable",
            "vial_to_slot_distance",
        ),
    ],
)
def test_mujoco_so101_pick_place_seed_task_resets_and_steps_when_mujoco_is_installed(
    env_id: str,
    seed_module: str,
    expected_info_key: str,
) -> None:
    pytest.importorskip("mujoco")
    seed = importlib.import_module(seed_module)

    env = seed.RewardRecipeWrapper(gym.make(env_id, render_mode="rgb_array", max_steps=4), "task_dense")
    obs, info = env.reset(seed=123)
    assert obs.shape == env.observation_space.shape
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)
    assert next_obs.shape == env.observation_space.shape
    assert isinstance(float(reward), float)
    assert 0.0 <= float(reward) <= 1.0
    assert isinstance(bool(terminated), bool)
    assert isinstance(bool(truncated), bool)
    assert expected_info_key in info
    assert f"initial_{expected_info_key}" in info
    env.close()


def test_mujoco_so101_pixel_actor_critic_seed_uses_low_res_wrist_pixels() -> None:
    pytest.importorskip("mujoco")
    seed = importlib.import_module("autoresearch_gym.tasks.so101_cube_to_bin_mujoco_v0.seed_trainable_pixel_actor_critic")

    env = seed.RewardRecipeWrapper(
        gym.make(
            "AutoresearchMujocoSO101CubeToBin-v0",
            render_mode="rgb_array",
            max_steps=4,
            vision_observation=True,
            vision_image_size=84,
            vision_frame_stack=3,
        ),
        "task_dense",
    )
    try:
        obs, info = env.reset(seed=123)
        assert obs["pixels"].shape == (9, 84, 84)
        assert obs["pixels"].dtype == np.uint8
        assert obs["proprio"].shape == (18,)
        agent = seed.Agent(env, device=torch.device("cpu"))
        action = agent.act(obs, deterministic=True)
        assert action.shape == env.action_space.shape
        next_obs, reward, terminated, truncated, info = env.step(action)
        assert next_obs["pixels"].shape == obs["pixels"].shape
        assert isinstance(float(reward), float)
        assert isinstance(bool(terminated), bool)
        assert isinstance(bool(truncated), bool)
    finally:
        env.close()


def test_mujoco_so101_vial_rack_slot_rails_are_colliding() -> None:
    pytest.importorskip("mujoco")

    env = gym.make("AutoresearchMujocoSO101VialToRack-v0", render_mode="rgb_array", max_steps=4)
    try:
        model = env.unwrapped.model
        mujoco = env.unwrapped.mujoco
        for name in (
            "rack_center_slot_back",
            "rack_center_slot_front",
            "rack_center_slot_left",
            "rack_center_slot_right",
            "rack_backstop",
        ):
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            assert geom_id >= 0
            assert int(model.geom_contype[geom_id]) != 0
            assert int(model.geom_conaffinity[geom_id]) != 0
    finally:
        env.close()


def test_mujoco_panda_real_env_default_render_returns_nonblank_frame_when_assets_are_installed() -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("robot_descriptions")

    try:
        env = gym.make("AutoresearchMujocoPandaPickAndPlaceDense-v0", render_mode="rgb_array", backend="mujoco", max_steps=2)
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    try:
        env.reset(seed=4500)
        frame = env.render()
    finally:
        env.close()

    assert isinstance(frame, np.ndarray)
    assert frame.dtype == np.uint8
    assert frame.ndim == 3
    assert frame.shape == (480, 720, 3)
    assert float(np.std(frame)) > 0.0


def test_panda_mjwarp_real_sampled_policy_trajectory_captures_frames_when_assets_are_installed() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("mujoco")
    pytest.importorskip("robot_descriptions")
    from autoresearch_gym.tasks.panda_pick_and_place_mjwarp_v0.seed_trainable import (
        Agent,
        RewardRecipeWrapper,
        _sample_policy_trajectory,
    )

    def env_factory(control_type=None, reward_recipe=None):
        del control_type
        return RewardRecipeWrapper(
            gym.make("AutoresearchMujocoPandaPickAndPlaceDense-v0", render_mode="rgb_array", backend="mujoco", max_steps=3),
            reward_recipe,
        )

    try:
        probe_env = env_factory()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    obs_dim = int(np.prod(probe_env.observation_space.shape))
    action_dim = int(np.prod(probe_env.action_space.shape))
    probe_env.close()

    agent = Agent(obs_dim, action_dim).to(torch.device("cpu"))
    sampled = _sample_policy_trajectory(
        agent,
        env_factory,
        types.SimpleNamespace(max_steps=2, eval_seed_start=4500, train_seed=1),
        {"episode": 1, "sample_index": 1, "frame_stride": 1, "playback_fps": 20.0},
    )

    assert sampled["source"] == SAMPLE_TRAJECTORY_SOURCE_CANDIDATE_PROVIDED
    assert sampled["frames"]
    assert all(isinstance(frame, np.ndarray) for frame in sampled["frames"])
    assert all(frame.shape == (480, 720, 3) for frame in sampled["frames"])
    assert any(float(np.std(frame)) > 0.0 for frame in sampled["frames"])
