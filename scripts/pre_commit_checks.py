from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from smoke_seed_artifacts import SEED_CASES

SEED_TRAINABLES = sorted({Path(case.seed) for case in SEED_CASES})


def python_executable(repo_root: Path) -> str:
    local_python = repo_root / ".venv" / "bin" / "python"
    return str(local_python) if local_python.exists() else sys.executable


def run_step(name: str, cmd: list[str], repo_root: Path) -> None:
    print(f"\n== {name} ==", flush=True)
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=repo_root, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-commit validation for autoresearch-gym.")
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
