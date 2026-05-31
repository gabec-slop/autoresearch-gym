from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

from autoresearch_gym.external.targets import SshTarget, load_target_config, make_target
from autoresearch_gym.runner.experiment import (
    BenchmarkSpec,
    append_result,
    json_default,
    normalize_run_tag,
)


def _repo_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path).resolve()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _json_objects(text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            objects.append(payload)
    return objects


def _extract_summary(stdout: str) -> dict[str, Any]:
    for payload in reversed(_json_objects(stdout)):
        if payload.get("run_id") and payload.get("benchmark") and payload.get("train") and payload.get("eval"):
            return payload
    raise RuntimeError(f"remote run did not print a final summary JSON object; stdout tail:\n{stdout[-2000:]}")


def _quote_arg(target: SshTarget, arg: str) -> str:
    if target.config.path_style == "windows":
        return target.quote_remote(arg)
    return shlex.quote(arg)


def _remote_python(target: SshTarget) -> str:
    if target.config.path_style == "windows":
        return target.config.python.replace("/", "\\")
    return shlex.quote(target.config.python)


def _remote_command(target: SshTarget, args: list[str]) -> str:
    remote_root = target.config.remote_root or "."
    if target.config.path_style == "windows":
        rendered = " ".join(_quote_arg(target, arg) for arg in args)
        return (
            "powershell.exe -NoProfile -Command "
            f"\"Set-Location {target.quote_remote(remote_root)}; {_remote_python(target)} {rendered}\""
        )
    rendered = " ".join(_quote_arg(target, arg) for arg in args)
    return f"cd {target.quote_remote(remote_root)} && {_remote_python(target)} {rendered}"


def _remote_parent(target: SshTarget, remote_file: str) -> str:
    separator = "\\" if target.config.path_style == "windows" else "/"
    return remote_file.rsplit(separator, 1)[0] if separator in remote_file else target.config.remote_root or "."


def _stage_repo_file(target: SshTarget, local_path: Path, repo_root: Path) -> str:
    if not _is_relative_to(local_path, repo_root):
        raise ValueError(f"remote in-process runs can only stage repo-local files, got {local_path}")
    remote_path = target.remote_repo_path(local_path, repo_root=repo_root)
    target.ensure_remote_dir(_remote_parent(target, remote_path))
    if not target.scp_local_file(local_path, remote_path):
        raise RuntimeError(f"failed to stage {local_path} to remote target")
    return remote_path


def _remote_cli_path(target: SshTarget, local_path: Path, repo_root: Path) -> str:
    return target.remote_repo_path(local_path, repo_root=repo_root)


def _copy_live_file(target: SshTarget, remote_session: str, session_dir: Path, suffix: str, timeout: float = 15.0) -> None:
    target.fetch_remote_file(target.remote_join(remote_session, "live", suffix), session_dir / "live" / suffix, timeout=timeout)


def _dashboard_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _remote_path_values(payload: Any) -> list[str]:
    values: list[str] = []
    if isinstance(payload, dict):
        for item in payload.values():
            values.extend(_remote_path_values(item))
    elif isinstance(payload, list):
        for item in payload:
            values.extend(_remote_path_values(item))
    elif isinstance(payload, str):
        values.append(payload)
    return values


def _remote_live_suffix(remote_session: str, session_dir: Path, value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.replace("\\", "/")
    remote_live = (remote_session.replace("\\", "/").rstrip("/") + "/live/")
    if normalized.startswith(remote_live):
        return normalized[len(remote_live) :]
    marker = f"autoresearch_runs/sessions/{session_dir.name}/live/"
    marker_index = normalized.find(marker)
    if marker_index >= 0:
        return normalized[marker_index + len(marker) :]
    return None


def _localize_live_path_value(remote_session: str, session_dir: Path, value: Any) -> Any:
    suffix = _remote_live_suffix(remote_session, session_dir, value)
    if suffix is None:
        return value
    return _dashboard_path(session_dir / "live" / suffix)


def _localize_live_paths(remote_session: str, session_dir: Path, payload: Any) -> Any:
    if isinstance(payload, dict):
        return {key: _localize_live_paths(remote_session, session_dir, item) for key, item in payload.items()}
    if isinstance(payload, list):
        return [_localize_live_paths(remote_session, session_dir, item) for item in payload]
    return _localize_live_path_value(remote_session, session_dir, payload)


def _localize_live_json_files(remote_session: str, session_dir: Path) -> None:
    live_dir = session_dir / "live"
    json_paths = [live_dir / "current_run_metrics.json"]
    json_paths.extend((live_dir / "trajectories").glob("**/manifest.json"))
    for path in json_paths:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        path.write_text(json.dumps(_localize_live_paths(remote_session, session_dir, payload), indent=2), encoding="utf-8")


def _sync_live_artifact_refs(target: SshTarget, remote_session: str, session_dir: Path) -> None:
    metrics_path = session_dir / "live" / "current_run_metrics.json"
    if not metrics_path.exists():
        return
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return
    suffixes = {
        suffix
        for value in _remote_path_values(metrics)
        if (suffix := _remote_live_suffix(remote_session, session_dir, value)) is not None
        and (suffix.startswith("trajectories/") or suffix == "current_run_frame.jpg")
    }
    for suffix in sorted(suffixes):
        _copy_live_file(target, remote_session, session_dir, suffix)
    manifests = [session_dir / "live" / suffix for suffix in suffixes if suffix.endswith("manifest.json")]
    for manifest_path in manifests:
        if not manifest_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for value in _remote_path_values(manifest):
            suffix = _remote_live_suffix(remote_session, session_dir, value)
            if suffix is None or not suffix.startswith("trajectories/"):
                continue
            _copy_live_file(target, remote_session, session_dir, suffix)
    _localize_live_json_files(remote_session, session_dir)


def _sync_remote_session(target: SshTarget, remote_session: str, session_dir: Path | None) -> None:
    if session_dir is None:
        return
    (session_dir / "live").mkdir(parents=True, exist_ok=True)
    for suffix in ("current_run_metrics.json", "status.log", "control.json"):
        _copy_live_file(target, remote_session, session_dir, suffix)
    _sync_live_artifact_refs(target, remote_session, session_dir)


def _fetch_final_artifacts(
    target: SshTarget,
    *,
    remote_session: str | None,
    session_dir: Path | None,
    remote_out_dir: str,
    out_dir: Path,
    run_id: str,
) -> None:
    if remote_session is not None and session_dir is not None:
        target.fetch_remote_dir_contents(target.remote_join(remote_session, "live"), session_dir / "live", timeout=60.0)
        remote_run_dir = target.remote_join(remote_session, "runs", run_id)
        local_run_dir = session_dir / "runs" / run_id
    else:
        remote_run_dir = target.remote_join(remote_out_dir, run_id)
        local_run_dir = out_dir / run_id
    if not target.fetch_remote_dir_archive(remote_run_dir, local_run_dir, timeout=float(os.environ.get("AUTORESEARCH_SSH_FETCH_TIMEOUT_SECONDS", "60"))):
        target.fetch_remote_dir_contents(remote_run_dir, local_run_dir, timeout=60.0)


def run_remote_in_process_experiment(
    *,
    benchmark: BenchmarkSpec,
    benchmark_path: Path,
    candidate_path: Path,
    tag: str,
    out_dir: Path,
    results_path: Path | None,
    train_episodes_override: int | None,
    train_seconds_override: float | None,
    eval_episodes_override: int | None,
    init_checkpoint: Path | None,
    session_dir: Path | None,
    evolution_metadata: dict[str, Any] | None,
    headless_env: bool,
    status_interval_seconds: float,
    compact_status: bool,
    compact_status_file: Path | None,
    train_probe_enabled: bool | None,
    train_probe_interval_seconds: float | None,
    train_probe_episodes: int | None,
    execution_target_override: str,
    target_config_path: Path | None,
) -> dict[str, Any]:
    tag = normalize_run_tag(tag)
    repo_root = Path.cwd().resolve()
    target_config = load_target_config(execution_target_override, config_path=target_config_path, repo_root=repo_root)
    target = make_target(target_config)
    if not isinstance(target, SshTarget):
        raise ValueError("remote in-process execution currently requires an ssh execution target")

    local_benchmark = _repo_path(benchmark_path, repo_root)
    local_candidate = _repo_path(candidate_path, repo_root)
    remote_benchmark = _stage_repo_file(target, local_benchmark, repo_root)
    remote_candidate = _stage_repo_file(target, local_candidate, repo_root)
    remote_init_checkpoint = None
    if init_checkpoint is not None:
        remote_init_checkpoint = _stage_repo_file(target, _repo_path(init_checkpoint, repo_root), repo_root)
    if benchmark.eval_case_bank is not None and benchmark.eval_case_bank.exists() and _is_relative_to(benchmark.eval_case_bank, repo_root):
        _stage_repo_file(target, benchmark.eval_case_bank, repo_root)

    remote_session = None
    if session_dir is not None:
        session_dir.mkdir(parents=True, exist_ok=True)
        remote_session = target.remote_session_path(session_dir, repo_root=repo_root)
        target.ensure_remote_dir(target.remote_join(remote_session, "candidates"))
        target.ensure_remote_dir(target.remote_join(remote_session, "live"))
        target.ensure_remote_dir(target.remote_join(remote_session, "runs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    remote_out_dir = target.remote_repo_path(out_dir, repo_root=repo_root)
    target.ensure_remote_dir(remote_out_dir)

    remote_status_file = None
    if compact_status_file is not None:
        status_path = _repo_path(compact_status_file, repo_root)
        if _is_relative_to(status_path, repo_root):
            remote_status_file = _remote_cli_path(target, status_path, repo_root)
    elif compact_status and remote_session is not None:
        remote_status_file = target.remote_join(remote_session, "live", "status.log")

    argv = [
        "-m",
        "autoresearch_gym.cli",
        "run",
        "--benchmark",
        target._remote_display_path(remote_benchmark),
        "--candidate",
        target._remote_display_path(remote_candidate),
        "--tag",
        tag,
        "--out-dir",
        target._remote_display_path(remote_out_dir),
        "--no-record",
    ]
    if remote_session is not None:
        argv.extend(["--session-dir", target._remote_display_path(remote_session)])
    if remote_init_checkpoint is not None:
        argv.extend(["--init-checkpoint", target._remote_display_path(remote_init_checkpoint)])
    if train_episodes_override is not None:
        argv.extend(["--train-episodes", str(train_episodes_override)])
    if train_seconds_override is not None:
        argv.extend(["--train-seconds", str(train_seconds_override)])
    if eval_episodes_override is not None:
        argv.extend(["--eval-episodes", str(eval_episodes_override)])
    if headless_env:
        argv.append("--headless-env")
    if compact_status:
        argv.append("--compact-status")
    if remote_status_file is not None:
        argv.extend(["--compact-status-file", target._remote_display_path(remote_status_file)])
    argv.extend(["--status-interval-seconds", str(status_interval_seconds)])
    if train_probe_enabled is False:
        argv.append("--no-train-probe")
    if train_probe_interval_seconds is not None:
        argv.extend(["--probe-interval-seconds", str(train_probe_interval_seconds)])
    if train_probe_episodes is not None:
        argv.extend(["--probe-episodes", str(train_probe_episodes)])
    for key, option in (
        ("search_mode", "--search-mode"),
        ("mutation_family", "--mutation-family"),
        ("proposal_id", "--proposal-id"),
        ("prompt_strategy", "--prompt-strategy"),
        ("declared_parent_run_id", "--parent-run-id"),
        ("archive_role", "--archive-role"),
        ("decision", "--decision"),
        ("notes", "--notes"),
        ("epoch", "--epoch"),
        ("batch_candidate_index", "--batch-candidate-index"),
        ("screening_stage", "--screening-stage"),
        ("parent_kind", "--parent-kind"),
        ("confirmation_source_run_id", "--confirmation-source-run-id"),
    ):
        value = (evolution_metadata or {}).get(key)
        if value is not None:
            argv.extend([option, str(value)])

    preflight = target.preflight(
        type(
            "RemoteInProcessBundle",
            (),
            {
                "local_run_dir": out_dir,
            },
        )()
    )
    if not preflight.ok:
        raise RuntimeError(f"remote in-process target preflight failed: {json.dumps(preflight.checks, indent=2)}")

    started_at = time.time()
    command = _remote_command(target, argv)
    process = subprocess.Popen(
        [*target.ssh_base_args(), target.config.host or "", command],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    timeout = float(benchmark.train_seconds or 0.0) + float(target.config.extra.get("in_process_timeout_padding_seconds", 1800.0))
    if timeout <= 0:
        timeout = float(target.config.extra.get("in_process_timeout_seconds", 3600.0))
    sync_interval = float(target.config.extra.get("live_sync_interval_seconds", 20.0))
    deadline = started_at + timeout
    next_sync = time.time() + max(1.0, sync_interval)
    timed_out = False
    while process.poll() is None:
        now = time.time()
        if now >= deadline:
            timed_out = True
            process.kill()
            break
        if sync_interval > 0 and now >= next_sync:
            _sync_remote_session(target, remote_session, session_dir) if remote_session is not None else None
            next_sync = now + max(1.0, sync_interval)
        time.sleep(1.0)
    stdout, stderr = process.communicate()
    _sync_remote_session(target, remote_session, session_dir) if remote_session is not None else None
    if timed_out:
        stderr = (stderr or "") + f"\nremote in-process command timed out after {timeout:.1f} seconds"
        returncode = 124
    else:
        returncode = int(process.returncode or 0)
    if returncode != 0:
        raise RuntimeError(
            f"remote in-process command failed with exit code {returncode}\n"
            f"stdout:\n{stdout[-2000:]}\n"
            f"stderr:\n{stderr[-2000:]}"
        )

    summary = _extract_summary(stdout)
    summary["execution"] = {
        "mode": "remote_in_process",
        **target_config.redacted_summary(),
        "preflight": {
            "ok": preflight.ok,
            "checks": preflight.checks,
        },
    }
    run_id = str(summary["run_id"])
    _fetch_final_artifacts(
        target,
        remote_session=remote_session,
        session_dir=session_dir,
        remote_out_dir=remote_out_dir,
        out_dir=out_dir,
        run_id=run_id,
    )
    local_run_dir = (session_dir / "runs" / run_id) if session_dir is not None else (out_dir / run_id)
    local_run_dir.mkdir(parents=True, exist_ok=True)
    (local_run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=json_default), encoding="utf-8")
    (local_run_dir / "remote_command.stdout.log").write_text(stdout or "", encoding="utf-8")
    (local_run_dir / "remote_command.stderr.log").write_text(stderr or "", encoding="utf-8")
    if results_path is not None:
        append_result(results_path, summary)
    return summary
