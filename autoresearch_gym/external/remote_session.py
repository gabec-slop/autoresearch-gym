from __future__ import annotations

import base64
import hashlib
import importlib
import importlib.metadata
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from autoresearch_gym.external.targets import SshTarget, load_target_config, make_target


FINAL_CRITICAL_RUN_FILES = (
    "summary.json",
    "eval_episodes.json",
    "train_episodes.json",
    "benchmark_snapshot.json",
    "candidate_snapshot.json",
    "trainable_snapshot.py",
)
FINAL_OPTIONAL_RUN_FILES = (
    "agent_checkpoint.pt",
    "current_run_frame.jpg",
    "policy_probe_records.jsonl",
    "train_result.json",
    "train_result_partial.json",
    "eval_result.json",
    "media_result.json",
)
FINAL_OPTIONAL_RUN_DIRS = ("trajectories", "policy_probes", "command_logs")

PACKAGE_DISTRIBUTIONS = {
    "mujoco_warp": "mujoco-warp",
    "warp": "warp-lang",
}


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def validate_session_candidate(session_dir: Path, candidate: Path) -> None:
    candidates_dir = session_dir.resolve() / "candidates"
    resolved_candidate = candidate.resolve()
    try:
        resolved_candidate.relative_to(candidates_dir)
    except ValueError as exc:
        raise ValueError(f"candidate must live under {candidates_dir}: {resolved_candidate}") from exc
    if not resolved_candidate.exists():
        raise FileNotFoundError(f"candidate does not exist: {resolved_candidate}")


def has_cli_option(argv: list[str], option: str) -> bool:
    return option in argv or any(arg.startswith(f"{option}=") for arg in argv)


def append_cli_option(argv: list[str], option: str, value: object | None = None) -> None:
    if has_cli_option(argv, option):
        return
    argv.append(option)
    if value is not None:
        argv.append(str(value))


def dirty_path_from_status(line: str) -> str:
    value = line[3:] if len(line) > 3 else line
    if " -> " in value:
        value = value.rsplit(" -> ", 1)[-1]
    return value.strip()


def is_environment_path(path: str) -> bool:
    return (
        path == "pyproject.toml"
        or path == "uv.lock"
        or path.startswith("autoresearch_gym/")
    )


def environment_dirty_paths(dirty_paths: list[str]) -> list[str]:
    return [
        path
        for line in dirty_paths
        if (path := dirty_path_from_status(line)) and is_environment_path(path)
    ]


def _git(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or f"git {' '.join(args)} failed")
    return completed.stdout.strip()


def _file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        stripped = str(value).strip()
        if not stripped or stripped in seen:
            continue
        seen.add(stripped)
        unique.append(stripped)
    return unique


def session_doctor_request(benchmark_path: Path) -> dict[str, Any]:
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    execution_backend = benchmark.get("execution_backend") if isinstance(benchmark.get("execution_backend"), dict) else {}
    required_paths = _unique_strings([str(path) for path in execution_backend.get("required_paths", [])])

    signature_parts = [
        str(benchmark.get("name", "")),
        str(benchmark.get("env_id", "")),
        str(benchmark.get("device", "")),
        str(execution_backend.get("name", "")),
        str(execution_backend.get("adapter", "")),
        str(execution_backend.get("task_family", "")),
        " ".join(required_paths),
    ]
    signature = " ".join(signature_parts).lower()
    packages = ["gymnasium", "numpy"]
    if benchmark.get("device") == "external" or execution_backend.get("kind") == "external":
        packages.append("torch")
    if "unitree" in signature or "mjlab" in signature:
        packages.extend(["mujoco", "mujoco_warp", "warp", "mjlab", "scipy"])
    elif "mujoco_warp" in signature or "mjwarp" in signature:
        packages.extend(["mujoco", "mujoco_warp", "warp", "torch"])

    return {
        "benchmark": str(benchmark_path),
        "benchmark_name": benchmark.get("name"),
        "device": benchmark.get("device"),
        "execution_backend": execution_backend,
        "required_paths": required_paths,
        "packages": _unique_strings(packages),
    }


def _package_version(module_name: str) -> str | None:
    distribution = PACKAGE_DISTRIBUTIONS.get(module_name, module_name)
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _nvidia_smi_gpus() -> list[dict[str, Any]]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if completed.returncode != 0:
        return []
    gpus: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3 or not parts[0]:
            continue
        try:
            memory_mb: int | None = int(float(parts[1]))
        except ValueError:
            memory_mb = None
        gpus.append({"name": parts[0], "memory_total_mb": memory_mb, "driver_version": parts[2]})
    return gpus


def _package_info(module_name: str) -> dict[str, Any]:
    info: dict[str, Any] = {
        "module": module_name,
        "distribution": PACKAGE_DISTRIBUTIONS.get(module_name, module_name),
        "installed": False,
    }
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 - doctor reports import failures without hiding their type.
        info.update({"error": f"{type(exc).__name__}: {exc}", "version": _package_version(module_name)})
        return info

    version = getattr(module, "__version__", None) or _package_version(module_name)
    info.update({"installed": True, "version": version})
    if module_name == "torch":
        cuda = getattr(module, "cuda", None)
        version_obj = getattr(module, "version", None)
        info.update(
            {
                "cuda_version": getattr(version_obj, "cuda", None),
                "cuda_available": bool(cuda and cuda.is_available()),
                "cuda_device_count": int(cuda.device_count()) if cuda else 0,
            }
        )
    if module_name == "warp":
        info["has_context"] = hasattr(module, "context")
    return info


def _session_doctor_payload(repo_root: Path, request: dict[str, Any]) -> dict[str, Any]:
    root = repo_root.resolve()
    checks: list[dict[str, Any]] = []

    required_paths = _unique_strings([str(path) for path in request.get("required_paths", [])])
    path_payloads: list[dict[str, Any]] = []
    for relative in required_paths:
        exists = (root / relative).exists()
        path_payloads.append({"path": relative, "exists": exists})
        checks.append(
            {
                "name": f"required_path:{relative}",
                "status": "ok" if exists else "fail",
                "message": "Required benchmark path exists." if exists else "Required benchmark path is missing.",
            }
        )

    package_payloads = [_package_info(module_name) for module_name in _unique_strings([str(name) for name in request.get("packages", [])])]
    for package in package_payloads:
        checks.append(
            {
                "name": f"package:{package['module']}",
                "status": "ok" if package.get("installed") else "fail",
                "message": "Package imports successfully." if package.get("installed") else str(package.get("error", "Package is not installed.")),
            }
        )

    packages_by_name = {str(package["module"]): package for package in package_payloads}
    nvidia_gpus = _nvidia_smi_gpus()
    torch_info = packages_by_name.get("torch", {})
    if nvidia_gpus and torch_info.get("installed") and not torch_info.get("cuda_available"):
        checks.append(
            {
                "name": "torch_cuda_matches_nvidia",
                "status": "warn",
                "message": "NVIDIA GPU detected by nvidia-smi, but PyTorch CUDA is unavailable.",
            }
        )
    if "mjlab" in packages_by_name and "warp" in packages_by_name:
        warp_info = packages_by_name["warp"]
        checks.append(
            {
                "name": "mjlab_warp_context_api",
                "status": "ok" if warp_info.get("has_context") else "fail",
                "message": (
                    "Warp exposes the context API used by MJLab."
                    if warp_info.get("has_context")
                    else "Warp does not expose wp.context; this MJLab stack is likely incompatible."
                ),
            }
        )

    payload = {
        "ok": all(check["status"] == "ok" for check in checks),
        "repo_root": str(root),
        "python": {"executable": sys.executable, "version": sys.version.split()[0]},
        "benchmark": request.get("benchmark"),
        "benchmark_name": request.get("benchmark_name"),
        "device": request.get("device"),
        "execution_backend": request.get("execution_backend"),
        "required_paths": path_payloads,
        "packages": package_payloads,
        "nvidia_smi_gpus": nvidia_gpus,
        "checks": checks,
    }
    return payload


_REMOTE_SESSION_PACKAGE_DOCTOR_CODE = """
import base64, importlib, importlib.metadata as md, json, os, subprocess, sys
mods = json.loads(base64.b64decode(os.environ["AR_SESSION_DOCTOR_PACKAGES"]).decode())
dist = {"mujoco_warp": "mujoco-warp", "warp": "warp-lang"}
def ver(name):
    try:
        return md.version(dist.get(name, name))
    except md.PackageNotFoundError:
        return None
def info(name):
    result = {"module": name, "distribution": dist.get(name, name), "installed": False}
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        result.update({"error": type(exc).__name__ + ": " + str(exc), "version": ver(name)})
        return result
    result.update({"installed": True, "version": getattr(module, "__version__", None) or ver(name)})
    if name == "torch":
        cuda = getattr(module, "cuda", None)
        version_obj = getattr(module, "version", None)
        result.update({
            "cuda_version": getattr(version_obj, "cuda", None),
            "cuda_available": bool(cuda and cuda.is_available()),
            "cuda_device_count": int(cuda.device_count()) if cuda else 0,
        })
    if name == "warp":
        result["has_context"] = hasattr(module, "context")
    return result
def gpus():
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
    except Exception:
        return []
    rows = []
    if completed.returncode != 0:
        return rows
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 3 and parts[0]:
            try:
                memory_mb = int(float(parts[1]))
            except ValueError:
                memory_mb = None
            rows.append({"name": parts[0], "memory_total_mb": memory_mb, "driver_version": parts[2]})
    return rows
print(json.dumps({
    "python": {"executable": sys.executable, "version": sys.version.split()[0]},
    "packages": [info(name) for name in mods],
    "nvidia_smi_gpus": gpus(),
}))
""".strip()


def local_environment_fingerprint(repo_root: Path | None = None) -> dict[str, Any]:
    root = (repo_root or Path.cwd()).resolve()
    dirty = [line for line in _git(root, "status", "--porcelain").splitlines() if line.strip()]
    return {
        "repo_root": str(root),
        "git_head": _git(root, "rev-parse", "HEAD"),
        "dirty_paths": dirty,
        "lock_hashes": {
            "pyproject.toml": _file_sha256(root / "pyproject.toml"),
            "uv.lock": _file_sha256(root / "uv.lock"),
        },
    }


_REMOTE_FINGERPRINT_CODE = """
import hashlib,json,pathlib,subprocess,sys
root=pathlib.Path.cwd()
def run(args):
    return subprocess.check_output(args,text=True).strip()
def digest(name):
    path=root/name
    if not path.exists():
        return None
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1048576),b""):
            h.update(chunk)
    return h.hexdigest()
payload={
    "repo_root": str(root),
    "git_head": run(["git","rev-parse","HEAD"]),
    "dirty_paths": [line for line in run(["git","status","--porcelain"]).splitlines() if line.strip()],
    "python_version": sys.version.split()[0],
    "lock_hashes": {
        "pyproject.toml": digest("pyproject.toml"),
        "uv.lock": digest("uv.lock"),
    },
}
print(json.dumps(payload))
""".strip()


def remote_environment_fingerprint_command(target: SshTarget) -> str:
    remote_root = target.config.remote_root or "."
    encoded = base64.b64encode(_REMOTE_FINGERPRINT_CODE.encode("utf-8")).decode("ascii")
    python_snippet = (
        "import os,base64;"
        "exec(base64.b64decode(os.environ[chr(65)+chr(82)+chr(95)+chr(67)+chr(79)+chr(68)+chr(69)]).decode())"
    )
    if target.config.path_style == "windows":
        python_cmd = target.config.python.replace("/", "\\")
        return target.powershell_command(
            f"Set-Location -LiteralPath {target.quote_remote(remote_root)}; "
            f"$env:AR_CODE={target.quote_remote(encoded)}; "
            f"& {target.quote_remote(python_cmd)} -c "
            f"{target.quote_remote(python_snippet)}"
        )
    return (
        f"cd {target.quote_remote(remote_root)} && "
        f"{shlex.quote(target.config.python)} -c "
        f"{shlex.quote(python_snippet)}"
    )


def remote_environment_fingerprint(target: SshTarget, *, timeout: float = 30.0) -> dict[str, Any]:
    command = remote_environment_fingerprint_command(target)
    result = target._ssh(command, timeout=timeout)
    if not result.ok:
        raise RuntimeError(f"remote environment fingerprint failed: {result.stderr[-1000:]}")
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
    raise RuntimeError(f"remote environment fingerprint did not produce JSON; stdout tail: {result.stdout[-1000:]}")


def remote_session_doctor_command(target: SshTarget, request: dict[str, Any]) -> str:
    remote_root = target.config.remote_root or "."
    encoded_code = base64.b64encode(_REMOTE_SESSION_PACKAGE_DOCTOR_CODE.encode("utf-8")).decode("ascii")
    encoded_packages = base64.b64encode(json.dumps(request.get("packages", [])).encode("utf-8")).decode("ascii")
    python_snippet = (
        "import os,base64;"
        "exec(base64.b64decode(os.environ[chr(65)+chr(82)+chr(95)+chr(83)+chr(69)+chr(83)+chr(83)+chr(73)+chr(79)+chr(78)+chr(95)+chr(68)+chr(79)+chr(67)+chr(84)+chr(79)+chr(82)+chr(95)+chr(67)+chr(79)+chr(68)+chr(69)]).decode())"
    )
    if target.config.path_style == "windows":
        python_cmd = target.config.python.replace("/", "\\")
        return target.powershell_command(
            f"Set-Location -LiteralPath {target.quote_remote(remote_root)}; "
            f"$env:AR_SESSION_DOCTOR_CODE={target.quote_remote(encoded_code)}; "
            f"$env:AR_SESSION_DOCTOR_PACKAGES={target.quote_remote(encoded_packages)}; "
            f"& {target.quote_remote(python_cmd)} -c {target.quote_remote(python_snippet)}"
        )
    return (
        f"cd {target.quote_remote(remote_root)} && "
        f"AR_SESSION_DOCTOR_CODE={shlex.quote(encoded_code)} "
        f"AR_SESSION_DOCTOR_PACKAGES={shlex.quote(encoded_packages)} "
        f"{shlex.quote(target.config.python)} -c {shlex.quote(python_snippet)}"
    )


def _remote_package_info_code(module_name: str) -> str:
    dist = PACKAGE_DISTRIBUTIONS.get(module_name, module_name)
    return (
        "import importlib,importlib.metadata as md,json;"
        f"n={module_name!r};d={dist!r};r={{'module':n,'distribution':d,'installed':False}};"
        "\ntry:\n"
        "    m=importlib.import_module(n)\n"
        "except Exception as e:\n"
        "    try:v=md.version(d)\n"
        "    except md.PackageNotFoundError:v=None\n"
        "    r.update({'error':type(e).__name__+': '+str(e),'version':v})\n"
        "else:\n"
        "    try:v=md.version(d)\n"
        "    except md.PackageNotFoundError:v=getattr(m,'__version__',None)\n"
        "    r.update({'installed':True,'version':getattr(m,'__version__',None) or v})\n"
        "    if n=='torch':\n"
        "        c=getattr(m,'cuda',None);vo=getattr(m,'version',None);r.update({'cuda_version':getattr(vo,'cuda',None),'cuda_available':bool(c and c.is_available()),'cuda_device_count':int(c.device_count()) if c else 0})\n"
        "    if n=='warp':r['has_context']=hasattr(m,'context')\n"
        "print(json.dumps(r))"
    )


def remote_package_info_command(target: SshTarget, module_name: str) -> str:
    remote_root = target.config.remote_root or "."
    encoded_code = base64.b64encode(_remote_package_info_code(module_name).encode("utf-8")).decode("ascii")
    python_snippet = (
        "import base64;"
        f"exec(base64.b64decode({encoded_code!r}).decode())"
    )
    if target.config.path_style == "windows":
        python_cmd = target.config.python.replace("/", "\\")
        return target.powershell_command(
            f"Set-Location -LiteralPath {target.quote_remote(remote_root)}; "
            f"& {target.quote_remote(python_cmd)} -c {target.quote_remote(python_snippet)}"
        )
    return (
        f"cd {target.quote_remote(remote_root)} && "
        f"{shlex.quote(target.config.python)} -c {shlex.quote(python_snippet)}"
    )


def remote_package_info(target: SshTarget, module_name: str, *, timeout: float = 30.0) -> dict[str, Any]:
    result = target._ssh(remote_package_info_command(target, module_name), timeout=timeout)
    if not result.ok:
        return {
            "module": module_name,
            "distribution": PACKAGE_DISTRIBUTIONS.get(module_name, module_name),
            "installed": False,
            "error": result.stderr[-1000:] or result.stdout[-1000:] or "remote package probe failed",
        }
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
    return {
        "module": module_name,
        "distribution": PACKAGE_DISTRIBUTIONS.get(module_name, module_name),
        "installed": False,
        "error": f"remote package probe did not produce JSON; stdout tail: {result.stdout[-1000:]}",
    }


def remote_nvidia_smi_gpus(target: SshTarget, *, timeout: float = 15.0) -> list[dict[str, Any]]:
    command = "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits"
    result = target._ssh(command, timeout=timeout)
    if not result.ok:
        return []
    gpus: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3 or not parts[0]:
            continue
        try:
            memory_mb: int | None = int(float(parts[1]))
        except ValueError:
            memory_mb = None
        gpus.append({"name": parts[0], "memory_total_mb": memory_mb, "driver_version": parts[2]})
    return gpus


def remote_session_doctor(target: SshTarget, request: dict[str, Any], *, timeout: float = 60.0) -> dict[str, Any]:
    required_paths = _unique_strings([str(path) for path in request.get("required_paths", [])])
    path_payloads: list[dict[str, Any]] = []
    path_checks: list[dict[str, Any]] = []
    for relative in required_paths:
        remote_path = target.remote_join(target.config.remote_root or ".", relative)
        try:
            info = target.remote_path_info(remote_path, timeout=min(timeout, 15.0))
            exists = bool(info.get("exists"))
        except Exception as exc:  # noqa: BLE001 - doctor records path inspection failure.
            info = {"error": f"{type(exc).__name__}: {exc}"}
            exists = False
        path_payloads.append({"path": relative, "exists": exists, "info": info})
        path_checks.append(
            {
                "name": f"required_path:{relative}",
                "status": "ok" if exists else "fail",
                "message": "Required benchmark path exists." if exists else "Required benchmark path is missing.",
            }
        )

    packages = [
        remote_package_info(target, module_name, timeout=min(timeout, 30.0))
        for module_name in _unique_strings([str(name) for name in request.get("packages", [])])
    ]
    package_checks = [
        {
            "name": f"package:{package.get('module')}",
            "status": "ok" if package.get("installed") else "fail",
            "message": (
                "Package imports successfully."
                if package.get("installed")
                else str(package.get("error", "Package is not installed."))
            ),
        }
        for package in packages
    ]
    packages_by_name = {
        str(package["module"]): package for package in packages if isinstance(package, dict) and "module" in package
    }
    nvidia_gpus = remote_nvidia_smi_gpus(target, timeout=min(timeout, 15.0))
    extra_checks = []
    torch_info = packages_by_name.get("torch", {})
    if nvidia_gpus and torch_info.get("installed") and not torch_info.get("cuda_available"):
        extra_checks.append(
            {
                "name": "torch_cuda_matches_nvidia",
                "status": "warn",
                "message": "NVIDIA GPU detected by nvidia-smi, but PyTorch CUDA is unavailable.",
            }
        )
    if "mjlab" in packages_by_name and "warp" in packages_by_name:
        warp_info = packages_by_name["warp"]
        extra_checks.append(
            {
                "name": "mjlab_warp_context_api",
                "status": "ok" if warp_info.get("has_context") else "fail",
                "message": (
                    "Warp exposes the context API used by MJLab."
                    if warp_info.get("has_context")
                    else "Warp does not expose wp.context; this MJLab stack is likely incompatible."
                ),
            }
        )
    checks = path_checks + package_checks + extra_checks
    return {
        "ok": all(check["status"] == "ok" for check in checks),
        "target": target.config.redacted_summary(),
        "benchmark": request.get("benchmark"),
        "benchmark_name": request.get("benchmark_name"),
        "device": request.get("device"),
        "execution_backend": request.get("execution_backend"),
        "required_paths": path_payloads,
        "python": {},
        "packages": packages,
        "nvidia_smi_gpus": nvidia_gpus,
        "checks": checks,
    }


def run_session_doctor(
    benchmark_path: Path,
    *,
    execution_target: str | None = None,
    target_config_path: Path | None = None,
    repo_root: Path | None = None,
    timeout: float = 60.0,
) -> dict[str, Any]:
    root = (repo_root or Path.cwd()).resolve()
    request = session_doctor_request(benchmark_path)
    if execution_target is None:
        payload = _session_doctor_payload(root, request)
        payload["target"] = {"target_kind": "local", "target": "local"}
        return payload
    target = load_ssh_target(execution_target, target_config_path, root)
    return remote_session_doctor(target, request, timeout=timeout)


def load_ssh_target(target_name: str, target_config_path: Path | None = None, repo_root: Path | None = None) -> SshTarget:
    config = load_target_config(target_name, config_path=target_config_path, repo_root=repo_root or Path.cwd())
    target = make_target(config)
    if not isinstance(target, SshTarget):
        raise ValueError(f"execution target {target_name!r} must be an ssh target, got {config.kind!r}")
    return target


def verify_remote_environment(
    target_name: str,
    target_config_path: Path | None = None,
    *,
    repo_root: Path | None = None,
    allow_remote_drift: bool = False,
    require_clean_local: bool = False,
) -> dict[str, Any]:
    root = (repo_root or Path.cwd()).resolve()
    target = load_ssh_target(target_name, target_config_path, root)
    local = local_environment_fingerprint(root)
    remote = remote_environment_fingerprint(target)
    local_environment_dirty = environment_dirty_paths(list(local["dirty_paths"]))
    remote_environment_dirty = environment_dirty_paths(list(remote.get("dirty_paths") or []))
    checks = [
        {
            "name": "git_head",
            "ok": local["git_head"] == remote.get("git_head"),
            "local": local["git_head"],
            "remote": remote.get("git_head"),
        },
        {
            "name": "pyproject_hash",
            "ok": local["lock_hashes"].get("pyproject.toml") == remote.get("lock_hashes", {}).get("pyproject.toml"),
            "local": local["lock_hashes"].get("pyproject.toml"),
            "remote": remote.get("lock_hashes", {}).get("pyproject.toml"),
        },
        {
            "name": "uv_lock_hash",
            "ok": local["lock_hashes"].get("uv.lock") == remote.get("lock_hashes", {}).get("uv.lock"),
            "local": local["lock_hashes"].get("uv.lock"),
            "remote": remote.get("lock_hashes", {}).get("uv.lock"),
        },
        {
            "name": "local_environment_clean",
            "ok": not local_environment_dirty,
            "dirty_paths": local_environment_dirty,
        },
        {
            "name": "remote_environment_clean",
            "ok": not remote_environment_dirty,
            "dirty_paths": remote_environment_dirty,
        },
    ]
    if require_clean_local:
        checks.append({"name": "local_clean", "ok": not local["dirty_paths"], "dirty_paths": local["dirty_paths"]})
    status = {
        "target": target_name,
        "target_kind": target.config.kind,
        "local": local,
        "remote": remote,
        "checks": checks,
        "ok": all(bool(check["ok"]) for check in checks),
    }
    if not status["ok"] and not allow_remote_drift:
        failed = [check for check in checks if not check["ok"]]
        raise RuntimeError(f"remote environment differs from local intended checkout: {json.dumps(failed, indent=2)}")
    return status


def _copy_live_file(
    target: SshTarget,
    remote_session: str,
    session_dir: Path,
    suffix: str,
    timeout: float = 15.0,
    *,
    skip_existing: bool = False,
) -> None:
    local_path = session_dir / "live" / suffix
    if skip_existing and local_path.exists():
        return
    tmp_path = local_path.with_suffix(local_path.suffix + f".fetch.{os.getpid()}.{time.time_ns()}.tmp")
    ok = target.fetch_remote_file(target.remote_join(remote_session, "live", suffix), tmp_path, timeout=timeout)
    if ok:
        tmp_path.replace(local_path)
    else:
        tmp_path.unlink(missing_ok=True)


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
    remote_live = remote_session.replace("\\", "/").rstrip("/") + "/live/"
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


def localize_live_json_files(remote_session: str, session_dir: Path) -> None:
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


def _live_trajectory_frame_cap() -> int:
    raw_value = os.environ.get("AUTORESEARCH_LIVE_TRAJECTORY_MAX_FRAMES", "32")
    try:
        return max(0, int(raw_value))
    except ValueError:
        return 32


def _sample_live_trajectory_frames(frames: list[str]) -> list[str]:
    frame_cap = _live_trajectory_frame_cap()
    if frame_cap <= 0 or len(frames) <= frame_cap:
        return frames
    if frame_cap == 1:
        return [frames[-1]]
    last_index = len(frames) - 1
    selected_indexes = {
        round(index * last_index / (frame_cap - 1))
        for index in range(frame_cap)
    }
    return [frames[index] for index in sorted(selected_indexes)]


def _sample_live_trajectory_steps(manifest: dict[str, Any], selected_frames: list[str]) -> list[dict[str, Any]]:
    steps = [step for step in manifest.get("steps", []) if isinstance(step, dict)]
    if not steps or not selected_frames:
        return steps
    selected_frame_set = set(selected_frames)
    selected_steps = [
        step
        for step in steps
        if isinstance(step.get("frame_path"), str) and step["frame_path"] in selected_frame_set
    ]
    if selected_steps:
        return selected_steps
    frames = [value for value in manifest.get("frames", []) if isinstance(value, str)]
    if len(frames) == len(steps):
        selected_indexes = {index for index, frame in enumerate(frames) if frame in selected_frame_set}
        return [step for index, step in enumerate(steps) if index in selected_indexes]
    return steps


def sync_live_artifact_refs(target: SshTarget, remote_session: str, session_dir: Path) -> None:
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
        _copy_live_file(
            target,
            remote_session,
            session_dir,
            suffix,
            skip_existing=suffix.startswith("trajectories/") and not suffix.endswith("manifest.json"),
        )
    manifests = [session_dir / "live" / suffix for suffix in suffixes if suffix.endswith("manifest.json")]
    for manifest_path in manifests:
        if not manifest_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        frames = [value for value in manifest.get("frames", []) if isinstance(value, str)]
        selected_frames = _sample_live_trajectory_frames(frames)
        selected_steps = _sample_live_trajectory_steps(manifest, selected_frames)
        manifest_changed = False
        if frames and selected_frames != frames:
            manifest["source_frame_count"] = len(frames)
            manifest["live_frame_count"] = len(selected_frames)
            manifest["frames"] = selected_frames
            manifest_changed = True
        if selected_steps != manifest.get("steps", []):
            manifest["source_step_count"] = len([step for step in manifest.get("steps", []) if isinstance(step, dict)])
            manifest["live_step_count"] = len(selected_steps)
            manifest["steps"] = selected_steps
            manifest_changed = True
        if manifest_changed:
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        for value in selected_frames:
            suffix = _remote_live_suffix(remote_session, session_dir, value)
            if suffix is None or not suffix.startswith("trajectories/"):
                continue
            _copy_live_file(target, remote_session, session_dir, suffix, skip_existing=not suffix.endswith("manifest.json"))
        for value in _remote_path_values(selected_steps):
            suffix = _remote_live_suffix(remote_session, session_dir, value)
            if suffix is None or not suffix.startswith("trajectories/"):
                continue
            _copy_live_file(target, remote_session, session_dir, suffix, skip_existing=not suffix.endswith("manifest.json"))
    localize_live_json_files(remote_session, session_dir)


def sync_remote_session_live(target: SshTarget, remote_session: str, session_dir: Path | None) -> None:
    if session_dir is None:
        return
    (session_dir / "live").mkdir(parents=True, exist_ok=True)
    for suffix in ("current_run_metrics.json", "status.log", "control.json"):
        _copy_live_file(target, remote_session, session_dir, suffix)
    sync_live_artifact_refs(target, remote_session, session_dir)


def fetch_remote_run_files(
    target: SshTarget,
    remote_run_dir: str,
    local_run_dir: Path,
    filenames: tuple[str, ...],
    *,
    timeout: float,
) -> dict[str, bool]:
    local_run_dir.mkdir(parents=True, exist_ok=True)
    return {
        filename: target.fetch_remote_file(
            target.remote_join(remote_run_dir, filename),
            local_run_dir / filename,
            timeout=timeout,
        )
        for filename in filenames
    }


def fetch_remote_session_final_artifacts(
    target: SshTarget,
    *,
    remote_session: str | None,
    session_dir: Path | None,
    remote_out_dir: str,
    out_dir: Path,
    run_id: str,
) -> Path:
    if remote_session is not None and session_dir is not None:
        sync_remote_session_live(target, remote_session, session_dir)
        remote_run_dir = target.remote_join(remote_session, "runs", run_id)
        local_run_dir = session_dir / "runs" / run_id
    else:
        remote_run_dir = target.remote_join(remote_out_dir, run_id)
        local_run_dir = out_dir / run_id
    local_run_dir.mkdir(parents=True, exist_ok=True)

    file_timeout = float(os.environ.get("AUTORESEARCH_SSH_FETCH_FILE_TIMEOUT_SECONDS", "30"))
    critical = fetch_remote_run_files(
        target,
        remote_run_dir,
        local_run_dir,
        FINAL_CRITICAL_RUN_FILES,
        timeout=file_timeout,
    )
    missing_critical = [name for name, ok in critical.items() if not ok]
    if missing_critical:
        dir_timeout = float(os.environ.get("AUTORESEARCH_SSH_FETCH_DIR_TIMEOUT_SECONDS", "60"))
        target.fetch_remote_dir_contents(remote_run_dir, local_run_dir, timeout=dir_timeout)

    optional_timeout = float(os.environ.get("AUTORESEARCH_SSH_FETCH_FILE_TIMEOUT_SECONDS", "30"))
    checkpoint_timeout = float(os.environ.get("AUTORESEARCH_SSH_CHECKPOINT_FETCH_TIMEOUT_SECONDS", "120"))
    for filename in FINAL_OPTIONAL_RUN_FILES:
        timeout = checkpoint_timeout if filename == "agent_checkpoint.pt" else optional_timeout
        target.fetch_remote_file(target.remote_join(remote_run_dir, filename), local_run_dir / filename, timeout=timeout)

    dir_timeout = float(os.environ.get("AUTORESEARCH_SSH_FETCH_DIR_TIMEOUT_SECONDS", "60"))
    for dirname in FINAL_OPTIONAL_RUN_DIRS:
        target.fetch_remote_dir_contents(target.remote_join(remote_run_dir, dirname), local_run_dir / dirname, timeout=dir_timeout)

    if target.config.path_style != "windows" and not all((local_run_dir / name).exists() for name in FINAL_CRITICAL_RUN_FILES):
        archive_timeout = float(os.environ.get("AUTORESEARCH_SSH_FETCH_TIMEOUT_SECONDS", "60"))
        target.fetch_remote_dir_archive(remote_run_dir, local_run_dir, timeout=archive_timeout)
    missing_after_fallbacks = [name for name in FINAL_CRITICAL_RUN_FILES if not (local_run_dir / name).exists()]
    if missing_after_fallbacks:
        raise RuntimeError(
            "remote run final artifacts are incomplete; missing "
            f"{', '.join(missing_after_fallbacks)} from {remote_run_dir}. "
            "Check the remote live/status.log and remote process state before treating this pass as evidence."
        )
    return local_run_dir
