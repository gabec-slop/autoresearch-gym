from __future__ import annotations

import base64
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
            [*self._ssh_base_args(), self.config.host or "", remote_command],
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

    def _powershell_command(self, script: str) -> str:
        script = "$ProgressPreference='SilentlyContinue'; " + script
        encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
        return f"powershell.exe -NoProfile -EncodedCommand {encoded}"

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

    def remote_repo_path(self, local_path: Path, repo_root: Path | None = None) -> str:
        repo = (repo_root or Path.cwd()).resolve()
        resolved = local_path.resolve()
        try:
            relative = resolved.relative_to(repo)
        except ValueError as exc:
            raise ValueError(f"remote SSH target can only stage repo-local paths, got {resolved}") from exc
        return self._remote_join(self.config.remote_root or ".", *relative.parts)

    def remote_session_path(self, session_dir: Path, repo_root: Path | None = None) -> str:
        return self.remote_repo_path(session_dir, repo_root=repo_root)

    def ensure_remote_dir(self, remote_dir: str) -> None:
        if self.config.path_style == "windows":
            command = self._powershell_command(
                f"New-Item -ItemType Directory -Force -Path {self._quote_remote(remote_dir)} | Out-Null"
            )
        else:
            command = f"mkdir -p {self._quote_remote(remote_dir)}"
        result = self._ssh(command)
        if not result.ok:
            raise RuntimeError(f"failed to create remote dir {remote_dir}: {result.stderr[-1000:]}")

    def scp_local_file(self, local_file: Path, remote_file: str, *, timeout: float = 30.0) -> bool:
        self.ensure_remote_dir(str(Path(remote_file).parent) if self.config.path_style != "windows" else remote_file.rsplit("\\", 1)[0])
        completed = subprocess.run(
            [*self._scp_base_args(), str(local_file), f"{self.config.host}:{self._remote_display_path(remote_file)}"],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
        return completed.returncode == 0

    def fetch_remote_file(self, remote_file: str, local_file: Path, *, timeout: float = 30.0) -> bool:
        return self._scp_remote_file(remote_file, local_file, timeout=timeout)

    def fetch_remote_dir_contents(self, remote_dir: str, local_dir: Path, *, timeout: float = 60.0) -> bool:
        return self._scp_remote_dir_contents(remote_dir, local_dir, timeout=timeout)

    def fetch_remote_dir_archive(self, remote_dir: str, local_dir: Path, *, timeout: float = 60.0) -> bool:
        return self._fetch_remote_dir_archive(remote_dir, local_dir, timeout=timeout)

    def remote_path_info(self, remote_path: str, *, timeout: float = 15.0) -> dict[str, Any]:
        if self.config.path_style == "windows":
            command = self._powershell_command(
                "$p="
                + self._quote_remote(remote_path)
                + "; $payload = if (Test-Path -LiteralPath $p) { "
                + "$i=Get-Item -LiteralPath $p; "
                + "$age=([DateTime]::UtcNow - $i.LastWriteTimeUtc).TotalSeconds; "
                + "[ordered]@{exists=$true; path=[string]$i.FullName; length=[int64]$i.Length; "
                + "last_write_utc=[string]$i.LastWriteTimeUtc; age_seconds=[double]$age} "
                + "} else { [ordered]@{exists=$false; path=$p; length=$null; last_write_utc=$null; age_seconds=$null} } "
                + "; $payload | ConvertTo-Json -Compress"
            )
        else:
            code = (
                "import json,os,pathlib,time;"
                f"p=pathlib.Path({remote_path!r});"
                "exists=p.exists();"
                "st=p.stat() if exists else None;"
                "print(json.dumps({'exists':exists,'path':str(p),'length':(st.st_size if st else None),"
                "'last_write_utc':(st.st_mtime if st else None),"
                "'age_seconds':((time.time()-st.st_mtime) if st else None)}))"
            )
            command = f"{shlex.quote(self.config.python)} -c {shlex.quote(code)}"
        result = self._ssh(command, timeout=timeout)
        if not result.ok:
            raise RuntimeError(f"failed to inspect remote path {remote_path}: {result.stderr[-1000:]}")
        for line in reversed(result.stdout.splitlines()):
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
        raise RuntimeError(f"remote path inspection did not produce JSON; stdout tail: {result.stdout[-1000:]}")

    def terminate_processes_matching(self, required_terms: list[str], *, timeout: float = 15.0) -> list[dict[str, Any]]:
        terms = [term for term in required_terms if term]
        if len(terms) < 2:
            raise ValueError("refusing to terminate remote processes without at least two match terms")
        if self.config.path_style == "windows":
            array_items = ",".join(self._quote_remote(term) for term in terms)
            command = self._powershell_command(
                f"$terms=@({array_items}); "
                + "$matches=@(); "
                + "Get-CimInstance Win32_Process -Filter \"name = 'python.exe' or name = 'python3.exe'\" | "
                + "ForEach-Object { $cmd=[string]$_.CommandLine; "
                + "$ok=$true; foreach ($term in $terms) { if ($cmd -notlike ('*' + $term + '*')) { $ok=$false } }; "
                + "if ($ok) { $matches += [ordered]@{pid=[int]$_.ProcessId; command=$cmd}; "
                + "Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue } }; "
                + "$matches | ConvertTo-Json -Depth 4 -Compress"
            )
        else:
            terms_json = json.dumps(terms)
            code = (
                "import json,os,signal,subprocess;"
                f"terms=json.loads({terms_json!r});"
                "rows=subprocess.check_output(['ps','-eo','pid=,args='],text=True).splitlines();"
                "matches=[];"
                "self_pid=os.getpid();"
                "\nfor row in rows:\n"
                "    stripped=row.strip()\n"
                "    if not stripped: continue\n"
                "    pid_s, _, cmd = stripped.partition(' ')\n"
                "    try: pid=int(pid_s)\n"
                "    except ValueError: continue\n"
                "    if pid == self_pid: continue\n"
                "    if all(term in cmd for term in terms):\n"
                "        matches.append({'pid': pid, 'command': cmd})\n"
                "        os.kill(pid, signal.SIGTERM)\n"
                "print(json.dumps(matches))"
            )
            command = f"{shlex.quote(self.config.python)} -c {shlex.quote(code)}"
        result = self._ssh(command, timeout=timeout)
        if not result.ok:
            raise RuntimeError(f"failed to terminate remote processes: {result.stderr[-1000:]}")
        text = result.stdout.strip()
        if not text:
            return []
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            for line in reversed(result.stdout.splitlines()):
                line = line.strip()
                if line.startswith("[") or line.startswith("{"):
                    payload = json.loads(line)
                    break
            else:
                raise RuntimeError(f"remote termination did not produce JSON; stdout tail: {result.stdout[-1000:]}")
        if isinstance(payload, dict):
            return [payload]
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        return []

    def ssh_base_args(self) -> list[str]:
        return self._ssh_base_args()

    def quote_remote(self, value: str) -> str:
        return self._quote_remote(value)

    def powershell_command(self, script: str) -> str:
        return self._powershell_command(script)

    def remote_join(self, *parts: str) -> str:
        return self._remote_join(*parts)

    def _scp_base_args(self) -> list[str]:
        return [
            "scp",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "ConnectionAttempts=1",
        ]

    def _ssh_base_args(self) -> list[str]:
        return [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "ConnectionAttempts=1",
        ]

    def _scp_remote_file(
        self,
        remote_file: str,
        local_file: Path,
        *,
        timeout: float = 30.0,
        skip_existing: bool = False,
    ) -> bool:
        if skip_existing and local_file.exists():
            return True
        local_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            completed = subprocess.run(
                [*self._scp_base_args(), f"{self.config.host}:{self._remote_display_path(remote_file)}", str(local_file)],
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return False
        return completed.returncode == 0

    def _scp_remote_dir_contents(self, remote_dir: str, local_dir: Path, *, timeout: float = 60.0) -> bool:
        local_dir.mkdir(parents=True, exist_ok=True)
        remote_display = self._remote_display_path(remote_dir).rstrip("/")
        try:
            completed = subprocess.run(
                [*self._scp_base_args(), "-r", f"{self.config.host}:{remote_display}/.", str(local_dir)],
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return False
        return completed.returncode == 0

    def _remote_tar_command(self, remote_dir: str) -> str:
        if self.config.path_style == "windows":
            quoted = self._quote_remote(remote_dir)
            return self._powershell_command(
                f"if (!(Test-Path -LiteralPath {quoted})) {{ exit 2 }}; tar -cf - -C {quoted} ."
            )
        quoted = self._quote_remote(remote_dir)
        return f"test -d {quoted} && tar -cf - -C {quoted} ."

    def _fetch_remote_dir_archive(self, remote_dir: str, local_dir: Path, *, timeout: float) -> bool:
        local_dir.mkdir(parents=True, exist_ok=True)
        try:
            archive = subprocess.run(
                [*self._ssh_base_args(), self.config.host or "", self._remote_tar_command(remote_dir)],
                capture_output=True,
                check=False,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return False
        if archive.returncode != 0:
            return False
        try:
            extracted = subprocess.run(
                ["tar", "-xf", "-", "-C", str(local_dir)],
                input=archive.stdout,
                capture_output=True,
                check=False,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return False
        return extracted.returncode == 0

    def preflight(self, bundle: RunBundle) -> TargetPreflight:
        checks: list[dict[str, Any]] = []
        remote_root = self.config.remote_root or "."
        if self.config.path_style == "windows":
            python_cmd = self.config.python.replace("/", "\\")
            command = self._powershell_command(
                f"$root={self._quote_remote(remote_root)}; "
                "if (!(Test-Path -LiteralPath $root)) { exit 2 }; "
                "Set-Location -LiteralPath $root; "
                f"& {self._quote_remote(python_cmd)} --version; "
                "if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }; "
                "New-Item -ItemType Directory -Force -Path 'autoresearch_runs\\external_preflight' | Out-Null; "
                "Write-Output ok"
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
            mkdir_command = self._powershell_command(
                f"New-Item -ItemType Directory -Force -Path {self._quote_remote(remote_external)} | Out-Null"
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
            [*self._scp_base_args(), "-r", f"{bundle.external_dir}/.", f"{self.config.host}:{remote_stage}/"],
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
            script = (
                f"Set-Location -LiteralPath {self._quote_remote(remote_root)}; "
                f"& {self._quote_remote(python_cmd)} -m {self._quote_remote(module)} --mode {self._quote_remote(mode)} "
                f"--bundle {self._quote_remote(remote_bundle)} "
                f"--out-dir {self._quote_remote(remote_external)}"
            )
            if "--checkpoint" in argv:
                script += f" --checkpoint {self._quote_remote(remote_checkpoint)}"
            remote_command = self._powershell_command(script)
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
            [*self._ssh_base_args(), self.config.host or "", remote_command],
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
        bundle.external_dir.mkdir(parents=True, exist_ok=True)
        remote_external = self._remote_external_dir(bundle)
        self._scp_remote_file(self._remote_join(remote_external, "live", "current_run_metrics.json"), bundle.external_dir / "live" / "current_run_metrics.json", timeout=15.0)
        self._scp_remote_file(self._remote_join(remote_external, "live", "status.log"), bundle.external_dir / "live" / "status.log", timeout=15.0)
        self._scp_remote_file(self._remote_join(remote_external, "current_run_frame.jpg"), bundle.external_dir / "current_run_frame.jpg", timeout=15.0)
        self._scp_remote_file(self._remote_join(remote_external, "policy_probe_records.jsonl"), bundle.external_dir / "policy_probe_records.jsonl", timeout=15.0)
        self._merge_live_policy_probe_records(bundle)
        self._sync_live_trajectory_refs(bundle)
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

    def _remote_suffix_from_path_value(self, bundle: RunBundle, value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        normalized = value.replace("\\", "/")
        remote_external = self._remote_display_path(self._remote_external_dir(bundle)).replace("\\", "/").rstrip("/")
        if normalized.startswith(remote_external + "/"):
            return normalized[len(remote_external) + 1 :]
        marker = f"autoresearch_runs/external_remote/{bundle.run_id}/external/"
        marker_index = normalized.find(marker)
        if marker_index >= 0:
            return normalized[marker_index + len(marker) :]
        return None

    def _localize_remote_path_value(self, bundle: RunBundle, value: Any) -> Any:
        suffix = self._remote_suffix_from_path_value(bundle, value)
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

    def _read_jsonl_records(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        records: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if isinstance(payload, dict):
                records.append(payload)
        return records

    def _merge_live_policy_probe_records(self, bundle: RunBundle) -> None:
        metrics_path = bundle.external_dir / "live" / "current_run_metrics.json"
        probe_records = self._read_jsonl_records(bundle.external_dir / "policy_probe_records.jsonl")
        if not metrics_path.exists() or not probe_records:
            return
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            return
        episodes = metrics.get("episodes")
        if not isinstance(episodes, list):
            episodes = []
        seen = {
            (
                str(record.get("record_type") or ""),
                str(record.get("step") or ""),
                str(record.get("probe_seed_start") or ""),
            )
            for record in episodes
            if isinstance(record, dict)
        }
        merged = [record for record in episodes if isinstance(record, dict)]
        for record in probe_records:
            key = (
                str(record.get("record_type") or ""),
                str(record.get("step") or ""),
                str(record.get("probe_seed_start") or ""),
            )
            if key not in seen:
                merged.append(record)
                seen.add(key)
        merged.sort(
            key=lambda record: (
                float(record.get("step") or 0.0),
                1 if str(record.get("record_type") or "") == "policy_probe" else 0,
            )
        )
        metrics["episodes"] = merged
        info_metrics = metrics.setdefault("current", {}).setdefault("info_metrics", {})
        if isinstance(info_metrics, dict):
            info_metrics["policy_probe_count"] = float(len(probe_records))
            latest_probe = probe_records[-1]
            info_metrics["policy_probe_return"] = float(latest_probe.get("return") or 0.0)
            info_metrics["policy_probe_length"] = float(latest_probe.get("length") or 0.0)
            if "elapsed_seconds" in latest_probe:
                info_metrics["policy_probe_elapsed_seconds"] = float(latest_probe.get("elapsed_seconds") or 0.0)
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    def _remote_path_values(self, payload: Any) -> list[str]:
        values: list[str] = []
        if isinstance(payload, dict):
            for item in payload.values():
                values.extend(self._remote_path_values(item))
        elif isinstance(payload, list):
            for item in payload:
                values.extend(self._remote_path_values(item))
        elif isinstance(payload, str):
            values.append(payload)
        return values

    def _sync_live_trajectory_refs(self, bundle: RunBundle) -> None:
        metrics_path = bundle.external_dir / "live" / "current_run_metrics.json"
        if not metrics_path.exists():
            return
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            return
        remote_external = self._remote_external_dir(bundle)
        suffixes = {
            suffix
            for value in self._remote_path_values(metrics)
            if (suffix := self._remote_suffix_from_path_value(bundle, value)) is not None
            and suffix.startswith("trajectories/")
        }
        for suffix in sorted(suffixes):
            self._scp_remote_file(
                self._remote_join(remote_external, suffix),
                bundle.external_dir / suffix,
                timeout=15.0,
                skip_existing=not suffix.endswith("manifest.json"),
            )
        manifests = [bundle.external_dir / suffix for suffix in suffixes if suffix.endswith("manifest.json")]
        for manifest_path in manifests:
            if not manifest_path.exists():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            for value in self._remote_path_values(manifest):
                suffix = self._remote_suffix_from_path_value(bundle, value)
                if suffix is None or not suffix.startswith("trajectories/"):
                    continue
                self._scp_remote_file(
                    self._remote_join(remote_external, suffix),
                    bundle.external_dir / suffix,
                    timeout=15.0,
                    skip_existing=not suffix.endswith("manifest.json"),
                )

    def fetch_artifacts(self, bundle: RunBundle) -> ArtifactSet:
        remote_external = self._remote_external_dir(bundle)
        bundle.external_dir.mkdir(parents=True, exist_ok=True)
        timeout = float(os.environ.get("AUTORESEARCH_SSH_FETCH_TIMEOUT_SECONDS", "60"))
        if self.config.path_style != "windows" and self._fetch_remote_dir_archive(remote_external, bundle.external_dir, timeout=timeout):
            self._merge_live_policy_probe_records(bundle)
            self._localize_live_artifacts(bundle)
            return ArtifactSet(root=bundle.external_dir)
        for suffix in (
            "bundle.json",
            "candidate_trainable.py",
            "benchmark_snapshot.json",
            "eval_cases.json",
            "candidate_metadata.json",
            "train_result.json",
            "eval_result.json",
            "media_result.json",
            "agent_checkpoint.pt",
            "current_run_frame.jpg",
            "train_result_partial.json",
            "policy_probe_records.jsonl",
        ):
            self._scp_remote_file(self._remote_join(remote_external, suffix), bundle.external_dir / suffix, timeout=min(timeout, 30.0))
        for suffix in ("live", "trajectories", "policy_probes", "command_logs"):
            self._scp_remote_dir_contents(self._remote_join(remote_external, suffix), bundle.external_dir / suffix, timeout=timeout)
        self._merge_live_policy_probe_records(bundle)
        self._localize_live_artifacts(bundle)
        return ArtifactSet(root=bundle.external_dir)


def make_target(config: TargetConfig) -> LocalSubprocessTarget | FakeTarget | SshTarget:
    if config.kind == "fake":
        return FakeTarget(config)
    if config.kind == "local":
        return LocalSubprocessTarget(config)
    if config.kind == "ssh":
        return SshTarget(config)
    raise ValueError(f"unknown execution target kind: {config.kind}")
