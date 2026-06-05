from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
from pathlib import Path

from smoke_seed_artifacts import SEED_CASES, SeedCase

SEED_TRAINABLES = sorted({Path(case.seed) for case in SEED_CASES})

PYTEST_KEYWORDS_BY_PATH = {
    Path("autoresearch_gym/cli.py"): "dashboard or doctor",
    Path("autoresearch_gym/runner/dashboard_server.py"): "dashboard_url",
    Path("autoresearch_gym/runner/session_run.py"): "run_parser or live_session_pointer",
    Path("scripts/launch_autoresearch_pass.py"): "launch_autoresearch_pass",
    Path("scripts/pre_commit_checks.py"): "pre_commit_affected_plan",
    Path("scripts/run_session_remote_pass.py"): "remote_session_pass_wrapper",
}

PYTEST_KEYWORDS_BY_PREFIX = {
    "autoresearch_gym/external/": "ssh or external or target or unitree",
    "autoresearch_gym/runner/": "compact_status or fixed_eval or live_writer or policy_probe or render or session",
    "dashboard/": "dashboard",
}

DOC_PATHS = {"AGENTS.md", "AUTORESEARCH.md", "README.md"}
SHELL_SYNTAX_PATHS = {Path(".githooks/pre-commit")}


def python_executable(repo_root: Path) -> str:
    local_python = repo_root / ".venv" / "bin" / "python"
    return str(local_python) if local_python.exists() else sys.executable


def run_step(name: str, cmd: list[str], repo_root: Path) -> None:
    print(f"\n== {name} ==", flush=True)
    print(shlex.join(cmd), flush=True)
    subprocess.run(cmd, cwd=repo_root, check=True)


def changed_paths(repo_root: Path, base: str, *, staged_only: bool = False) -> list[Path]:
    diff_cmd = ["git", "diff", "--name-only"]
    if staged_only:
        diff_cmd.append("--cached")
    diff_cmd.append(base)
    completed = subprocess.run(
        diff_cmd,
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or f"git diff failed against {base}")
    if staged_only:
        paths = {Path(line.strip()) for line in completed.stdout.splitlines() if line.strip()}
        return sorted(paths, key=lambda path: path.as_posix())
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if untracked.returncode != 0:
        raise RuntimeError(untracked.stderr.strip() or "git ls-files failed")
    paths = {
        Path(line.strip())
        for line in [*completed.stdout.splitlines(), *untracked.stdout.splitlines()]
        if line.strip()
    }
    return sorted(paths, key=lambda path: path.as_posix())


def _is_doc_only(path: Path) -> bool:
    value = path.as_posix()
    return value in DOC_PATHS or value.startswith("docs/") or value.endswith(".md")


def seed_cases_for_path(path: Path) -> list[SeedCase]:
    cases: list[SeedCase] = []
    if path.name.startswith("seed_trainable") and path.suffix == ".py":
        return [case for case in SEED_CASES if path == Path(case.seed)]
    for case in SEED_CASES:
        if path == Path(case.seed) or path == Path(case.benchmark):
            cases.append(case)
            continue
        task_dir = Path(case.seed).parent
        try:
            path.relative_to(task_dir)
        except ValueError:
            continue
        if path.suffix in {".py", ".json"}:
            cases.append(case)
    unique: dict[str, SeedCase] = {case.name: case for case in cases}
    return list(unique.values())


def changed_test_names(repo_root: Path, base: str, test_path: Path, *, staged_only: bool = False) -> set[str]:
    diff_cmd = ["git", "diff", "--unified=0"]
    if staged_only:
        diff_cmd.append("--cached")
    diff_cmd.extend([base, "--", test_path.as_posix()])
    completed = subprocess.run(
        diff_cmd,
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return set()
    names: set[str] = set()
    for line in completed.stdout.splitlines():
        match = re.match(r"^\+def (test_[a-zA-Z0-9_]+)\(", line)
        if match:
            names.add(match.group(1))
    return names


def affected_plan(
    paths: list[Path],
    py: str,
    artifact_timeout: float,
    repo_root: Path | None = None,
    changed_tests: set[str] | None = None,
) -> list[tuple[str, list[str]]]:
    if not paths:
        return []

    compile_paths: set[Path] = set()
    shell_paths: set[Path] = set()
    pytest_keywords: set[str] = set()
    smoke_cases: dict[str, SeedCase] = {}
    force_unit_smoke = False

    for path in paths:
        if _is_doc_only(path):
            continue
        if path in SHELL_SYNTAX_PATHS:
            shell_paths.add(path)
        if path.suffix == ".py" and not path.as_posix().startswith("tests/"):
            compile_paths.add(path)

        for case in seed_cases_for_path(path):
            smoke_cases[case.name] = case
            compile_paths.add(Path(case.seed))

        keyword = PYTEST_KEYWORDS_BY_PATH.get(path)
        if keyword:
            pytest_keywords.add(keyword)
        for prefix, prefix_keyword in PYTEST_KEYWORDS_BY_PREFIX.items():
            if path.as_posix().startswith(prefix):
                pytest_keywords.add(prefix_keyword)

        if path.as_posix().startswith("tests/"):
            if path == Path("tests/test_smoke.py") and changed_tests:
                pytest_keywords.update(changed_tests)
            else:
                force_unit_smoke = True
        if path in {Path("pyproject.toml"), Path("uv.lock")}:
            force_unit_smoke = True
        if path.suffix == ".py" and not path.as_posix().startswith("tests/") and not (
            path.as_posix().startswith("autoresearch_gym/tasks/")
            or path in PYTEST_KEYWORDS_BY_PATH
            or any(path.as_posix().startswith(prefix) for prefix in PYTEST_KEYWORDS_BY_PREFIX)
        ):
            force_unit_smoke = True

    commands: list[tuple[str, list[str]]] = []
    root = repo_root or Path.cwd()
    existing_compile_paths = [
        str(path)
        for path in sorted(compile_paths, key=lambda item: item.as_posix())
        if (root / path).exists()
    ]
    if existing_compile_paths:
        commands.append(("affected python syntax", [py, "-m", "py_compile", *existing_compile_paths]))
    existing_shell_paths = [
        str(path)
        for path in sorted(shell_paths, key=lambda item: item.as_posix())
        if (root / path).exists()
    ]
    if existing_shell_paths:
        commands.append(("affected shell syntax", ["sh", "-n", *existing_shell_paths]))

    if force_unit_smoke:
        commands.append(("affected unit smoke tests", [py, "-m", "pytest", "tests/test_smoke.py"]))
    elif pytest_keywords:
        expression = " or ".join(sorted(pytest_keywords))
        commands.append(("affected unit smoke tests", [py, "-m", "pytest", "tests/test_smoke.py", "-k", expression]))

    for case_name in sorted(smoke_cases):
        commands.append(
            (
                f"affected artifact smoke: {case_name}",
                [py, "scripts/smoke_seed_artifacts.py", "--timeout", str(artifact_timeout), "--case", case_name],
            )
        )

    return commands


def run_affected(args: argparse.Namespace, repo_root: Path, py: str) -> int:
    paths = changed_paths(repo_root, args.changed_since, staged_only=args.staged)
    print("affected paths:", flush=True)
    if paths:
        for path in paths:
            print(f"  {path.as_posix()}", flush=True)
    else:
        print("  none", flush=True)

    test_names = changed_test_names(repo_root, args.changed_since, Path("tests/test_smoke.py"), staged_only=args.staged)
    commands = affected_plan(paths, py, args.artifact_timeout, repo_root, test_names)
    if not commands:
        print("\nno affected validation needed", flush=True)
        return 0
    for name, cmd in commands:
        if args.dry_run:
            print(f"\n== {name} ==")
            print(shlex.join(cmd), flush=True)
        else:
            run_step(name, cmd, repo_root)
    if args.dry_run:
        print("\naffected validation dry run complete", flush=True)
    else:
        print("\naffected validation passed", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-commit validation for autoresearch-gym.")
    parser.add_argument(
        "--affected",
        action="store_true",
        help="Run a path-sensitive validation subset for local iteration. Use the full gate before promotion.",
    )
    parser.add_argument(
        "--changed-since",
        default="HEAD",
        help="Git revision used by --affected to find changed tracked files.",
    )
    parser.add_argument(
        "--staged",
        action="store_true",
        help="With --affected, only consider staged changes. Intended for the git pre-commit hook.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print selected validation commands without running them.")
    parser.add_argument(
        "--skip-artifact-smoke",
        action="store_true",
        help="Developer escape hatch for non-code-only commits. Do not use for trainable, runner, dashboard, or task changes.",
    )
    parser.add_argument(
        "--artifact-timeout",
        type=float,
        default=180.0,
        help="Per-run timeout for seed artifact smoke runs.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    py = python_executable(repo_root)
    if args.affected:
        return run_affected(args, repo_root, py)

    seed_paths = [str(path) for path in SEED_TRAINABLES]

    run_step("seed syntax", [py, "-m", "py_compile", *seed_paths], repo_root)
    run_step("unit smoke tests", [py, "-m", "pytest", "tests/test_smoke.py"], repo_root)
    if not args.skip_artifact_smoke:
        run_step(
            "seed logging and visual artifact smoke",
            [py, "scripts/smoke_seed_artifacts.py", "--timeout", str(args.artifact_timeout)],
            repo_root,
        )

    print("\npre-commit checks passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
