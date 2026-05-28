from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass
class CommandSpec:
    argv: list[str]
    cwd: Path | str | None = None
    env: dict[str, str] = field(default_factory=dict)
    label: str = "external"
    timeout_seconds: float | None = None


@dataclass
class CommandResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""
    started_at: float | None = None
    finished_at: float | None = None

    @property
    def ok(self) -> bool:
        return self.returncode == 0


@dataclass
class TargetPreflight:
    ok: bool
    kind: str
    target: str
    checks: list[dict[str, Any]]
    redacted: dict[str, Any] = field(default_factory=dict)


@dataclass
class ArtifactSet:
    root: Path
    files: dict[str, Path] = field(default_factory=dict)


@dataclass
class RunBundle:
    run_id: str
    tag: str
    benchmark_path: Path
    candidate_path: Path
    local_run_dir: Path
    external_dir: Path
    benchmark: Any
    candidate: Any
    candidate_metadata: dict[str, Any]
    execution_backend: dict[str, Any]
    eval_cases: list[dict[str, Any]] | None
    train_episodes: int
    train_seconds: float | None
    eval_episodes: int
    max_steps: int
    compact_status_file: Path | None
    session_dir: Path | None
    target_name: str


class ExternalBackend(Protocol):
    def build_bundle(self, bundle: RunBundle) -> RunBundle:
        ...

    def training_command(self, bundle: RunBundle) -> CommandSpec:
        ...

    def eval_command(self, bundle: RunBundle, checkpoint_path: Path) -> CommandSpec:
        ...

    def media_command(self, bundle: RunBundle, checkpoint_path: Path) -> CommandSpec | None:
        ...

    def normalize_train(self, artifacts: ArtifactSet) -> dict[str, Any]:
        ...

    def normalize_eval(self, artifacts: ArtifactSet) -> dict[str, Any]:
        ...

    def normalize_media(self, artifacts: ArtifactSet) -> dict[str, Any]:
        ...
