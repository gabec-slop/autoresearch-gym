from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

from autoresearch_gym.external.remote_session import append_cli_option, validate_session_candidate, verify_remote_environment
from autoresearch_gym.runner import session_run


def build_run_args(args: argparse.Namespace, passthrough: list[str]) -> list[str]:
    run_args = list(passthrough)
    append_cli_option(run_args, "--session-dir", args.session_dir)
    append_cli_option(run_args, "--candidate", args.candidate)
    append_cli_option(run_args, "--execution-target", args.execution_target)
    if args.target_config is not None:
        append_cli_option(run_args, "--target-config", args.target_config)
    if args.allow_remote_drift:
        append_cli_option(run_args, "--allow-remote-drift")

    if not args.no_compact_status:
        append_cli_option(run_args, "--compact-status")
        append_cli_option(run_args, "--compact-status-file", args.session_dir / "live" / "status.log")

    if args.no_dashboard:
        append_cli_option(run_args, "--no-dashboard")
    else:
        append_cli_option(run_args, "--dashboard-host", args.dashboard_host)
        append_cli_option(run_args, "--dashboard-port", args.dashboard_port)
        append_cli_option(run_args, "--dashboard-port-end", args.dashboard_port_end)

    if args.disable_train_probes:
        append_cli_option(run_args, "--no-train-probe")

    return run_args


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Run one already-authored session candidate on a remote execution target "
            "with compact status and dashboard defaults."
        )
    )
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--execution-target", required=True)
    parser.add_argument("--target-config", type=Path, default=None)
    parser.add_argument("--skip-remote-checkout-check", action="store_true")
    parser.add_argument("--allow-remote-drift", action="store_true")
    parser.add_argument("--require-clean-local", action="store_true")
    parser.add_argument("--no-compact-status", action="store_true")
    parser.add_argument("--no-dashboard", action="store_true")
    parser.add_argument("--dashboard-host", default="127.0.0.1")
    parser.add_argument("--dashboard-port", type=int, default=4174)
    parser.add_argument("--dashboard-port-end", type=int, default=4199)
    parser.add_argument("--disable-train-probes", action="store_true")
    return parser.parse_known_args(argv)


def main(argv: list[str] | None = None) -> int:
    args, passthrough = parse_args(argv)
    try:
        validate_session_candidate(args.session_dir, args.candidate)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    if not args.skip_remote_checkout_check:
        status = verify_remote_environment(
            args.execution_target,
            args.target_config,
            allow_remote_drift=args.allow_remote_drift,
            require_clean_local=args.require_clean_local,
        )
        print(f"remote environment: {status}", file=sys.stderr, flush=True)

    run_args = build_run_args(args, passthrough)
    print("autoresearch-gym run " + shlex.join([str(arg) for arg in run_args]), file=sys.stderr, flush=True)
    session_run.main(run_args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
