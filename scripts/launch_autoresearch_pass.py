from __future__ import annotations

import argparse
import shlex
import sys
import time
from pathlib import Path

from autoresearch_gym.external.remote_session import (
    append_cli_option,
    has_cli_option,
    validate_session_candidate,
    verify_remote_environment,
)
from autoresearch_gym.runner import session_run


def _option_value(argv: list[str], option: str) -> str | None:
    prefix = f"{option}="
    for index, arg in enumerate(argv):
        if arg.startswith(prefix):
            return arg[len(prefix) :]
        if arg == option and index + 1 < len(argv):
            return argv[index + 1]
    return None


def append_agent_prerun_log(args: argparse.Namespace, passthrough: list[str]) -> None:
    log_path = args.session_dir / "outer_loop_log.md"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        log_path.write_text(session_run.SESSION_LOG_TEMPLATE, encoding="utf-8")

    tag = _option_value(passthrough, "--tag") or args.candidate.stem
    existing_log = log_path.read_text(encoding="utf-8")
    heading = f"## {tag} pre-run plan"
    if heading in existing_log:
        return
    benchmark = _option_value(passthrough, "--benchmark") or "-"
    seed_candidate = _option_value(passthrough, "--seed-candidate") or "-"
    stage = _option_value(passthrough, "--screening-stage") or "-"
    mutation_family = _option_value(passthrough, "--mutation-family") or "-"
    proposal_id = _option_value(passthrough, "--proposal-id") or "-"
    parent_run_id = _option_value(passthrough, "--parent-run-id") or "-"

    lines = [
        "",
        heading,
        "",
        "- authored_by: agent",
        f"- created_at: {time.strftime('%Y-%m-%dT%H:%M:%S')}",
        f"- candidate: `{args.candidate}`",
        f"- fixed_benchmark: `{benchmark}`",
        f"- seed_candidate: `{seed_candidate}`",
        f"- stage: {stage}",
        f"- execution_target: {args.execution_target or 'local'}",
        f"- mutation_family: {mutation_family}",
        f"- proposal_id: {proposal_id}",
        f"- parent_run_id: {parent_run_id}",
        "",
        "### Hypothesis",
        "",
        args.hypothesis.strip(),
        "",
        "### Mutation Summary",
        "",
        args.mutation_summary.strip(),
    ]
    if args.expected_diagnostics:
        lines.extend(["", "### Expected Diagnostics", "", args.expected_diagnostics.strip()])
    if args.success_criteria:
        lines.extend(["", "### Success Criteria", "", args.success_criteria.strip()])
    if args.next_planned_mutation:
        lines.extend(["", "### Next Planned Mutation", "", args.next_planned_mutation.strip()])
    lines.append("")

    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def build_run_args(args: argparse.Namespace, passthrough: list[str]) -> list[str]:
    run_args = list(passthrough)
    append_cli_option(run_args, "--session-dir", args.session_dir)
    append_cli_option(run_args, "--candidate", args.candidate)
    if args.execution_target is not None:
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

    should_disable_probes = args.disable_train_probes or args.profile == "fragile-remote"
    if should_disable_probes and not has_cli_option(run_args, "--no-train-probe"):
        run_args.append("--no-train-probe")

    return run_args


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Launch one already-authored autoresearch session candidate while writing "
            "agent-authored outer-loop context."
        )
    )
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument("--mutation-summary", required=True)
    parser.add_argument("--expected-diagnostics", default=None)
    parser.add_argument("--success-criteria", default=None)
    parser.add_argument("--next-planned-mutation", default=None)
    parser.add_argument("--execution-target", default=None)
    parser.add_argument("--target-config", type=Path, default=None)
    parser.add_argument("--skip-remote-checkout-check", action="store_true")
    parser.add_argument("--allow-remote-drift", action="store_true")
    parser.add_argument("--require-clean-local", action="store_true")
    parser.add_argument("--profile", choices=["standard", "fragile-remote"], default="standard")
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

    if args.execution_target is not None and not args.skip_remote_checkout_check:
        status = verify_remote_environment(
            args.execution_target,
            args.target_config,
            allow_remote_drift=args.allow_remote_drift,
            require_clean_local=args.require_clean_local,
        )
        print(f"remote environment: {status}", file=sys.stderr, flush=True)

    append_agent_prerun_log(args, passthrough)
    run_args = build_run_args(args, passthrough)
    print("autoresearch-gym run " + shlex.join([str(arg) for arg in run_args]), file=sys.stderr, flush=True)
    session_run.main(run_args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
