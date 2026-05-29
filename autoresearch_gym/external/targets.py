from __future__ import annotations

import json
import os
import shutil
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autoresearch_gym.external.base import ArtifactSet, CommandResult, CommandSpec, RunBundle, TargetPreflight


@dataclass
class TargetConfig:
    name: str
    kind: str = "local"
    host: str | None = None
    remote_root: str | None = None
    path_style: str = "posix"
    python: str = "python"
    artifact_sync: str = "none"
    extra: dict[str, Any] = field(default_factory=dict)

    def redacted_summary(self) -> dict[str, Any]:
        return {
            "target": self.name,
            "target_kind": self.kind,
            "host_redacted": bool(self.host),
            "remote_root_redacted": bool(self.remote_root),
            "path_style": self.path_style,
        }


def _coerce_scalar(raw: str) -> Any:
    value = raw.strip()
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _load_simple_toml(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    section: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = [part.strip() for part in line[1:-1].split(".") if part.strip()]
            cursor = data
            for part in section:
                cursor = cursor.setdefault(part, {})
            continue
        if "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        cursor = data
        for part in section:
            cursor = cursor.setdefault(part, {})
        cursor[key.strip()] = _coerce_scalar(raw_value)
    return data


def _target_from_payload(name: str, payload: dict[str, Any] | None) -> TargetConfig:
    payload = dict(payload or {})
    kind = str(payload.pop("kind", "local"))
    host = payload.pop("host", None)
    remote_root = payload.pop("remote_root", None)
    path_style = str(payload.pop("path_style", "windows" if kind == "ssh" and remote_root and ":" in str(remote_root) else "posix"))
    python = str(payload.pop("python", "python"))
    artifact_sync = str(payload.pop("artifact_sync", "scp" if kind == "ssh" else "none"))
    if kind == "ssh" and artifact_sync == "sftp":
        artifact_sync = "scp"
    if kind == "ssh" and artifact_sync != "scp":
        raise ValueError("ssh target artifact_sync must be 'scp'")
    return TargetConfig(
        name=name,
        kind=kind,
        host=str(host) if host is not None else None,
        remote_root=str(remote_root) if remote_root is not None else None,
        path_style=path_style,
        python=python,
        artifact_sync=artifact_sync,
        extra=payload,
    )


def load_target_config(
    target_name: str | None,
    *,
    config_path: Path | None = None,
    repo_root: Path | None = None,
) -> TargetConfig:
    name = target_name or "local"
    if name in {"local", "default"}:
        return TargetConfig(name=name, kind="local")
    if name == "fake":
        return TargetConfig(name=name, kind="fake")

    paths: list[Path] = []
    if config_path is not None:
        paths.append(config_path.expanduser())
    paths.append(Path("~/.config/autoresearch-gym/targets.toml").expanduser())
    if repo_root is not None:
        paths.append(repo_root / ".autoresearch.local.toml")

    merged: dict[str, Any] = {}
    for path in paths:
        if not path.exists():
            continue
        payload = _load_simple_toml(path)
        targets = payload.get("targets")
        if isinstance(targets, dict) and isinstance(targets.get(name), dict):
            merged.update(targets[name])

    if not merged:
        raise KeyError(f"execution target '{name}' was not found in private target config")
    return _target_from_payload(name, merged)


class LocalSubprocessTarget:
    def __init__(self, config: TargetConfig) -> None:
        self.config = config

    def preflight(self, bundle: RunBundle) -> TargetPreflight:
        checks = [
            {
                "name": "local_run_dir",
                "ok": bundle.local_run_dir.exists(),
                "message": str(bundle.local_run_dir),
            },
        ]
        return TargetPreflight(
            ok=all(bool(check["ok"]) for check in checks),
            kind=self.config.kind,
            target=self.config.name,
            checks=checks,
            redacted=self.config.redacted_summary(),
        )

    def stage(self, bundle: RunBundle) -> None:
        bundle.external_dir.mkdir(parents=True, exist_ok=True)

    def run(self, command: CommandSpec, bundle: RunBundle) -> CommandResult:
        env = os.environ.copy()
        env.update(command.env)
        started_at = time.time()
        completed = subprocess.run(
            command.argv,
            cwd=str(command.cwd) if command.cwd is not None else None,
            env=env,
            capture_output=True,
            text=True,
            timeout=command.timeout_seconds,
            check=False,
        )
        result = CommandResult(
            returncode=int(completed.returncode),
            stdout=completed.stdout,
            stderr=completed.stderr,
            started_at=started_at,
            finished_at=time.time(),
        )
        self._write_command_log(command, bundle, result)
        return result

    def sync_live(self, bundle: RunBundle) -> None:
        return None

    def fetch_artifacts(self, bundle: RunBundle) -> ArtifactSet:
        return ArtifactSet(root=bundle.external_dir)

    def _write_command_log(self, command: CommandSpec, bundle: RunBundle, result: CommandResult) -> None:
        log_dir = bundle.external_dir / "command_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        safe_label = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in command.label)
        (log_dir / f"{safe_label}.stdout.log").write_text(result.stdout, encoding="utf-8")
        (log_dir / f"{safe_label}.stderr.log").write_text(result.stderr, encoding="utf-8")


class FakeTarget(LocalSubprocessTarget):
    pass


class SshTarget:
    def __init__(self, config: TargetConfig) -> None:
        self.config = config
        if not config.host:
            raise ValueError("ssh target requires host")
        if not config.remote_root:
            raise ValueError("ssh target requires remote_root")

    def _ssh(self, remote_command: str, timeout: float = 30.0) -> CommandResult:
        started_at = time.time()
        completed = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", self.config.host or "", remote_command],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return CommandResult(
            returncode=int(completed.returncode),
            stdout=completed.stdout,
            stderr=completed.stderr,
            started_at=started_at,
            finished_at=time.time(),
        )

    def _quote_remote(self, value: str) -> str:
        if self.config.path_style == "windows":
            return "'" + value.replace("'", "''") + "'"
        return shlex.quote(value)

    def _remote_join(self, *parts: str) -> str:
        sep = "\\" if self.config.path_style == "windows" else "/"
        cleaned = [str(part).strip("\\/") for part in parts if str(part)]
        if not cleaned:
            return self.config.remote_root or "."
        first = cleaned[0]
        if len(first) == 2 and first[1] == ":":
            return first + sep + sep.join(cleaned[1:])
        if ":" in first[:3]:
            return first.rstrip("\\/") + sep + sep.join(cleaned[1:])
        return sep.join(cleaned)

    def _remote_external_dir(self, bundle: RunBundle) -> str:
        return self._remote_join(
            self.config.remote_root or ".",
            "autoresearch_runs",
            "external_remote",
            bundle.run_id,
            "external",
        )

    def _remote_display_path(self, path: str) -> str:
        return path.replace("\\", "/") if self.config.path_style == "windows" else path

    def preflight(self, bundle: RunBundle) -> TargetPreflight:
        checks: list[dict[str, Any]] = []
        remote_root = self.config.remote_root or "."
        if self.config.path_style == "windows":
            command = (
                "powershell.exe -NoProfile -Command "
                f"\"$root={self._quote_remote(remote_root)}; "
                "if (!(Test-Path $root)) { exit 2 }; "
                "Set-Location $root; "
                f"{self.config.python} --version; "
                "if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }; "
                "New-Item -ItemType Directory -Force autoresearch_runs\\external_preflight | Out-Null; "
                "Write-Output ok\""
            )
        else:
            command = (
                f"cd {self._quote_remote(remote_root)} && "
                f"{shlex.quote(self.config.python)} --version && "
                "mkdir -p autoresearch_runs/external_preflight && echo ok"
            )
        result = self._ssh(command)
        checks.append(
            {
                "name": "ssh_remote_root_python",
                "ok": result.ok,
                "returncode": result.returncode,
                "stdout_tail": result.stdout[-500:],
                "stderr_tail": result.stderr[-500:],
            }
        )
        return TargetPreflight(
            ok=all(bool(check["ok"]) for check in checks),
            kind=self.config.kind,
            target=self.config.name,
            checks=checks,
            redacted=self.config.redacted_summary(),
        )

    def stage(self, bundle: RunBundle) -> None:
        bundle.external_dir.mkdir(parents=True, exist_ok=True)
        remote_external = self._remote_external_dir(bundle)
        if self.config.path_style == "windows":
            mkdir_command = (
                "powershell.exe -NoProfile -Command "
                f"\"New-Item -ItemType Directory -Force {self._quote_remote(remote_external)} | Out-Null\""
            )
        else:
            mkdir_command = f"mkdir -p {self._quote_remote(remote_external)}"
        mkdir_result = self._ssh(mkdir_command)
        if not mkdir_result.ok:
            raise RuntimeError(f"failed to create remote external dir: {mkdir_result.stderr[-1000:]}")
        bundle_path = bundle.external_dir / "bundle.json"
        if not bundle_path.exists():
            raise FileNotFoundError(f"bundle was not written before staging: {bundle_path}")
        remote_stage = self._remote_display_path(remote_external)
        completed = subprocess.run(
            ["scp", "-r", f"{bundle.external_dir}/.", f"{self.config.host}:{remote_stage}/"],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"failed to stage external bundle over scp: {completed.stderr[-1000:]}")

    def run(self, command: CommandSpec, bundle: RunBundle) -> CommandResult:
        remote_external = self._remote_external_dir(bundle)
        remote_bundle = self._remote_join(remote_external, "bundle.json")
        remote_checkpoint = self._remote_join(remote_external, "agent_checkpoint.pt")
        mode = None
        module = None
        argv = list(command.argv)
        if "-m" in argv:
            module_index = argv.index("-m") + 1
            if module_index < len(argv):
                module = argv[module_index]
        if "--mode" in argv:
            mode_index = argv.index("--mode") + 1
            if mode_index < len(argv):
                mode = argv[mode_index]
        if module is None or mode is None:
            raise ValueError(f"SshTarget can only run Python module external commands, got {command.argv!r}")
        remote_root = self.config.remote_root or "."
        if self.config.path_style == "windows":
            python_cmd = self.config.python.replace("/", "\\")
            remote_command = (
                "powershell.exe -NoProfile -Command "
                f"\"Set-Location {self._quote_remote(remote_root)}; "
                f"{python_cmd} -m {module} --mode {mode} "
                f"--bundle {self._quote_remote(remote_bundle)} "
                f"--out-dir {self._quote_remote(remote_external)}"
            )
            if "--checkpoint" in argv:
                remote_command += f" --checkpoint {self._quote_remote(remote_checkpoint)}"
            remote_command += "\""
        else:
            remote_command = (
                f"cd {self._quote_remote(remote_root)} && "
                f"{shlex.quote(self.config.python)} -m {shlex.quote(module)} --mode {shlex.quote(mode)} "
                f"--bundle {self._quote_remote(remote_bundle)} "
                f"--out-dir {self._quote_remote(remote_external)}"
            )
            if "--checkpoint" in argv:
                remote_command += f" --checkpoint {self._quote_remote(remote_checkpoint)}"
        started_at = time.time()
        process = subprocess.Popen(
            ["ssh", "-o", "BatchMode=yes", self.config.host or "", remote_command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        deadline = started_at + float(command.timeout_seconds or 600.0)
        sync_interval = float(self.config.extra.get("live_sync_interval_seconds", 20.0))
        next_sync = time.time() + max(1.0, sync_interval)
        timed_out = False
        while process.poll() is None:
            now = time.time()
            if now >= deadline:
                timed_out = True
                process.kill()
                break
            if sync_interval > 0 and now >= next_sync:
                self.sync_live(bundle)
                next_sync = now + max(1.0, sync_interval)
            time.sleep(1.0)
        stdout, stderr = process.communicate()
        if timed_out:
            returncode = 124
            stderr = (stderr or "") + f"\ncommand timed out after {command.timeout_seconds or 600.0} seconds"
        else:
            returncode = int(process.returncode or 0)
        result = CommandResult(
            returncode=returncode,
            stdout=stdout or "",
            stderr=stderr or "",
            started_at=started_at,
            finished_at=time.time(),
        )
        self.sync_live(bundle)
        log_dir = bundle.external_dir / "command_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        safe_label = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in command.label)
        (log_dir / f"{safe_label}.stdout.log").write_text(result.stdout, encoding="utf-8")
        (log_dir / f"{safe_label}.stderr.log").write_text(result.stderr, encoding="utf-8")
        return result

    def sync_live(self, bundle: RunBundle) -> None:
        remote_external = self._remote_display_path(self._remote_external_dir(bundle))
        bundle.external_dir.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(
            ["scp", "-r", f"{self.config.host}:{remote_external}/.", str(bundle.external_dir)],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            return
        self._localize_live_artifacts(bundle)
        if bundle.session_dir is not None:
            remote_live_metrics = bundle.external_dir / "live" / "current_run_metrics.json"
            if remote_live_metrics.exists():
                local_live = bundle.session_dir / "live"
                local_live.mkdir(parents=True, exist_ok=True)
                shutil.copy2(remote_live_metrics, local_live / "current_run_metrics.json")
            remote_status = bundle.external_dir / "live" / "status.log"
            if remote_status.exists():
                local_live = bundle.session_dir / "live"
                local_live.mkdir(parents=True, exist_ok=True)
                shutil.copy2(remote_status, local_live / "status.log")

    def _dashboard_path(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
        except ValueError:
            return str(path.resolve())

    def _localize_remote_path_value(self, bundle: RunBundle, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        normalized = value.replace("\\", "/")
        remote_external = self._remote_display_path(self._remote_external_dir(bundle)).replace("\\", "/").rstrip("/")
        suffix: str | None = None
        if normalized.startswith(remote_external + "/"):
            suffix = normalized[len(remote_external) + 1 :]
        else:
            marker = f"autoresearch_runs/external_remote/{bundle.run_id}/external/"
            marker_index = normalized.find(marker)
            if marker_index >= 0:
                suffix = normalized[marker_index + len(marker) :]
        if suffix is None:
            return value
        return self._dashboard_path(bundle.external_dir / suffix)

    def _localize_remote_paths(self, bundle: RunBundle, payload: Any) -> Any:
        if isinstance(payload, dict):
            return {key: self._localize_remote_paths(bundle, item) for key, item in payload.items()}
        if isinstance(payload, list):
            return [self._localize_remote_paths(bundle, item) for item in payload]
        return self._localize_remote_path_value(bundle, payload)

    def _localize_live_artifacts(self, bundle: RunBundle) -> None:
        json_paths = [bundle.external_dir / "live" / "current_run_metrics.json"]
        json_paths.extend((bundle.external_dir / "trajectories").glob("sample_*/manifest.json"))
        for path in json_paths:
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            localized = self._localize_remote_paths(bundle, payload)
            path.write_text(json.dumps(localized, indent=2), encoding="utf-8")

    def fetch_artifacts(self, bundle: RunBundle) -> ArtifactSet:
        remote_external = self._remote_display_path(self._remote_external_dir(bundle))
        bundle.external_dir.mkdir(parents=True, exist_ok=True)
        timeout = float(os.environ.get("AUTORESEARCH_SSH_FETCH_TIMEOUT_SECONDS", "300"))
        completed = subprocess.run(
            ["scp", "-r", f"{self.config.host}:{remote_external}/.", str(bundle.external_dir)],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"failed to fetch remote artifacts over scp: {completed.stderr[-1000:]}")
        return ArtifactSet(root=bundle.external_dir)


def make_target(config: TargetConfig) -> LocalSubprocessTarget | FakeTarget | SshTarget:
    if config.kind == "fake":
        return FakeTarget(config)
    if config.kind == "local":
        return LocalSubprocessTarget(config)
    if config.kind == "ssh":
        return SshTarget(config)
    raise ValueError(f"unknown execution target kind: {config.kind}")
