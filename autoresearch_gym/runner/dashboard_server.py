from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import quote


DASHBOARD_MANAGER = "autoresearch_gym.session_dashboard"


def dashboard_url(host: str, port: int, session: str | None = None) -> str:
    display_host = "127.0.0.1" if host in {"", "0.0.0.0", "::"} else host
    url = f"http://{display_host}:{port}/dashboard/"
    if session:
        url += f"?session={quote(session, safe='/._-')}"
    return url


def find_available_port(host: str, start: int, end: int | None = None) -> int:
    last = start if end is None else end
    if start <= 0 or last < start:
        raise ValueError("dashboard port range must be positive and ordered")
    for port in range(start, last + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"no available dashboard port in range {start}-{last} on {host}")


def wait_for_dashboard(url: str, *, timeout_seconds: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=0.5) as response:
                if int(response.status) < 500:
                    return True
        except (OSError, urllib.error.URLError):
            time.sleep(0.1)
    return False


def dashboard_pointer_path(session_dir: Path) -> Path:
    return session_dir / "live" / "dashboard.json"


def read_dashboard_pointer(session_dir: Path) -> dict | None:
    path = dashboard_pointer_path(session_dir)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def write_dashboard_pointer(session_dir: Path, payload: dict) -> None:
    path = dashboard_pointer_path(session_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def is_process_alive(pid: object) -> bool:
    try:
        value = int(pid)
    except (TypeError, ValueError):
        return False
    if value <= 0:
        return False
    try:
        os.kill(value, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def dashboard_pointer_ready(pointer: dict | None, *, timeout_seconds: float = 1.0) -> bool:
    if not pointer or not pointer.get("url"):
        return False
    return wait_for_dashboard(str(pointer["url"]), timeout_seconds=timeout_seconds)


def terminate_session_dashboard(session_dir: Path, *, wait_seconds: float = 3.0) -> dict:
    pointer = read_dashboard_pointer(session_dir)
    if not pointer:
        return {"ok": True, "action": "none", "reason": "missing_pointer"}
    if pointer.get("managed_by") != DASHBOARD_MANAGER:
        return {"ok": False, "action": "none", "reason": "unmanaged_pointer", "pointer": pointer}
    pid = pointer.get("pid")
    if not is_process_alive(pid):
        payload = {
            **pointer,
            "ready": False,
            "stopped": True,
            "stopped_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        write_dashboard_pointer(session_dir, payload)
        return {"ok": True, "action": "marked_stopped", "pid": pid, "pointer": payload}
    os.kill(int(pid), signal.SIGTERM)
    deadline = time.monotonic() + max(0.0, wait_seconds)
    while time.monotonic() < deadline and is_process_alive(pid):
        time.sleep(0.1)
    stopped = not is_process_alive(pid)
    payload = {
        **pointer,
        "ready": False,
        "stopped": stopped,
        "stopped_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_dashboard_pointer(session_dir, payload)
    return {"ok": stopped, "action": "terminated", "pid": pid, "pointer": payload}


def ensure_session_dashboard(
    session_dir: Path,
    *,
    host: str,
    port: int,
    port_end: int | None,
    root: Path,
    ready_timeout: float = 5.0,
    force_restart: bool = False,
) -> dict:
    session_dir = session_dir.resolve()
    root = root.resolve()
    session_path = session_dir.relative_to(root).as_posix() if session_dir.is_relative_to(root) else str(session_dir)
    existing = read_dashboard_pointer(session_dir)
    if not force_restart and dashboard_pointer_ready(existing, timeout_seconds=1.0):
        assert existing is not None
        payload = {
            **existing,
            "ready": True,
            "reused": True,
            "session_path": session_path,
        }
        write_dashboard_pointer(session_dir, payload)
        return payload

    if existing and existing.get("managed_by") == DASHBOARD_MANAGER and is_process_alive(existing.get("pid")):
        terminate_session_dashboard(session_dir)

    selected_port = find_available_port(host, port, port_end)
    url = dashboard_url(host, selected_port, session_path)
    log_path = session_dir / "live" / "dashboard.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "autoresearch_gym.cli",
        "dashboard",
        "--host",
        host,
        "--port",
        str(selected_port),
        "--port-end",
        str(selected_port),
        "--root",
        str(root),
        "--session",
        session_path,
    ]
    with log_path.open("ab") as log_handle:
        process = subprocess.Popen(
            cmd,
            cwd=root,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    ready = wait_for_dashboard(url, timeout_seconds=ready_timeout)
    payload = {
        "url": url,
        "host": host,
        "port": selected_port,
        "pid": process.pid,
        "ready": ready,
        "reused": False,
        "managed_by": DASHBOARD_MANAGER,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "log_path": str(log_path),
        "session_path": session_path,
        "root": str(root),
        "command": cmd,
    }
    write_dashboard_pointer(session_dir, payload)
    return payload


def session_dashboard_status(session_dir: Path) -> dict:
    pointer = read_dashboard_pointer(session_dir)
    ready = dashboard_pointer_ready(pointer, timeout_seconds=1.0)
    pid_alive = is_process_alive(pointer.get("pid")) if pointer else False
    return {
        "ok": bool(pointer and ready),
        "ready": ready,
        "pid_alive": pid_alive,
        "pointer": pointer,
    }
