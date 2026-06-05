from __future__ import annotations

from autoresearch_gym.external.base import (
    ArtifactSet,
    CommandResult,
    CommandSpec,
    ExternalBackend,
    RunBundle,
    TargetPreflight,
)
from autoresearch_gym.external.targets import (
    FakeTarget,
    LocalSubprocessTarget,
    SshTarget,
    TargetConfig,
    load_target_config,
    make_target,
)
from autoresearch_gym.external.remote_session import (
    append_cli_option,
    fetch_remote_session_final_artifacts,
    has_cli_option,
    local_environment_fingerprint,
    remote_environment_fingerprint,
    remote_session_doctor_command,
    run_session_doctor,
    sync_remote_session_live,
    validate_session_candidate,
    verify_remote_environment,
)

__all__ = [
    "ArtifactSet",
    "CommandResult",
    "CommandSpec",
    "ExternalBackend",
    "FakeTarget",
    "LocalSubprocessTarget",
    "RunBundle",
    "SshTarget",
    "TargetConfig",
    "TargetPreflight",
    "append_cli_option",
    "fetch_remote_session_final_artifacts",
    "has_cli_option",
    "local_environment_fingerprint",
    "load_target_config",
    "make_target",
    "remote_environment_fingerprint",
    "remote_session_doctor_command",
    "run_session_doctor",
    "sync_remote_session_live",
    "validate_session_candidate",
    "verify_remote_environment",
]
