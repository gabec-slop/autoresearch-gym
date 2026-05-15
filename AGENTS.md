# Project Maintainer Guide

This file is for maintaining and extending the `autoresearch-gym` package:
adding tasks, environments, seeds, dependencies, tests, docs, dashboard features,
and release polish.

For running a research session and mutating session-local candidates, read
`AUTORESEARCH.md` instead.

## Project Purpose

`autoresearch-gym` is a Gymnasium workbench for agentic, model-driven
exploration of reinforcement-learning approaches in robotics and
continuous-control simulation.

The package should make it easy for a coding agent to:

- choose a fixed benchmark
- mutate training code under that fixed benchmark
- run deterministic evaluation
- inspect metrics, curves, artifacts, frames, and logs
- use prior experiments to choose the next hypothesis

## Design Principles

- Fixed benchmarks, mutable training recipes. Benchmark files own environment
  identity, budgets, seeds, eval cases, success definitions, and primary metrics.
  Trainables own how learning happens under that contract.
- Session discipline lives in `AUTORESEARCH.md`. Do not duplicate the full run
  loop here.
- Prefer recognizable tasks: canonical MuJoCo control, Gymnasium-Robotics
  manipulation, PandaGym tasks, or custom robot tasks with clear visual behavior
  and fixed evaluation.
- Prefer readable, mutable seeds. A seed should be self-contained enough for an
  agent to edit model architecture, replay, warmup, loss terms, exploration,
  update cadence, reward shaping, HER, curriculum, vectorization, and logging.
- Keep optional simulator dependencies isolated behind extras. The base install
  should stay useful without pulling every simulator stack.
- Keep benchmark filenames generic. The exact wall-clock duration belongs inside
  JSON as `train_seconds`, not in the filename.
- Make live inspection first-class. Tasks should expose RGB frames when feasible,
  and runner/dashboard artifacts should be stable enough for agents and humans to
  inspect mid-run.
- Preserve the CLI stdout contract for `autoresearch-gym run`: stdout is the
  final JSON summary. Use compact status stderr/file output for live human or
  agent monitoring, especially `--compact-status-file` for tools that do not
  surface live stderr reliably.
- Treat rendering as environment-sensitive. A missing live frame can mean the
  benchmark did not request `render_mode="rgb_array"`, but it can also mean the
  current process sandbox blocks the simulator's graphics backend. On macOS,
  MuJoCo's CGL/GLFW path may need a real CoreGraphics/WindowServer connection
  and can fail in restrictive sandboxes with errors such as `CGLError: invalid
  CoreGraphics connection`; verify those failures from a GUI-permitted shell
  before changing benchmark or runner code. On Linux/headless hosts, prefer a
  configured offscreen backend such as EGL or OSMesa when available, and record
  the required environment setup with the run.

## Repository Layout

- `autoresearch_gym/runner/`: benchmark loading, candidate execution, fixed
  evaluation, live metrics, summaries, utilization notes, and rollout rendering.
- `autoresearch_gym/envs/`: custom environments owned by this package.
- `autoresearch_gym/tasks/`: bundled task specs, eval cases, and seed trainables.
- `dashboard/`: static dashboard served by `autoresearch-gym dashboard`.
- `docs/`: installation, staging, and project notes.
- `tests/`: smoke and contract tests.

## Task Layout

Each task lives under `autoresearch_gym/tasks/<task_name>/` and should contain:

- `benchmark.json`: episode-budget benchmark.
- `benchmark_wall_clock.json`: wall-clock-budget benchmark when useful.
- `eval_cases.json`: fixed case bank if the environment supports fixed cases;
  otherwise an explanatory empty file.
- `seed_trainable.py`: conservative baseline trainable.
- optional alternate seeds such as `seed_trainable_her.py` or
  `seed_trainable_vectorized.py` when they represent a meaningful common
  approach.

The README should mention any bundled task that downstream users are expected to
discover.

## Benchmark Rules

Benchmark JSON owns:

- `env_id`
- `env_kwargs`
- train budget: `train_episodes` and optional `train_seconds`
- eval budget: `eval_episodes`, eval seeds, and optional fixed case bank
- `max_steps`
- `primary_metric`
- `primary_metric_mode`
- device preference

Episode-budget `benchmark.json` should not include `train_seconds`.
Wall-clock `benchmark_wall_clock.json` should include `train_seconds` and a high
`train_episodes` safety cap.

Do not encode candidate-specific recipe choices in benchmark files. Warmup,
batch size, replay behavior, HER fraction, update cadence, vector-env count, and
model size belong in trainables.

## Environment Rules

Use an upstream Gymnasium-compatible environment when one exists. Add a custom
environment under `autoresearch_gym/envs/` only when this package truly owns the
task dynamics or wrappers.

Custom environments should:

- register their env ID in `autoresearch_gym/envs/__init__.py`
- keep optional simulator imports lazy when possible
- support `gym.make(env_id, **env_kwargs)`
- support `reset(seed=...)`
- support `reset(seed=..., options={"fixed_case": ...})` when fixed eval cases
  are part of the task
- return standard Gymnasium `step` values
- include `info["is_success"]` when success rate is a benchmark metric
- expose useful scalar `info` metrics for summaries and dashboard analysis
- support `render_mode="rgb_array"` when visual monitoring is expected

## Seed Trainable Rules

Every seed trainable must expose:

- `get_candidate()`
- `RewardRecipeWrapper`
- `train_agent(...)`
- `save_agent_checkpoint(...)`

If checkpoint rendering should work, also expose `Agent` and
`load_agent_checkpoint(...)` using the established local pattern.

Use CleanRL-style explicit code over hidden abstractions. Shared utilities are
acceptable only when they preserve readability and do not hide the training
recipe from the autoresearch loop.

`get_candidate()` should describe the seed or candidate in human language. The
runner surfaces it in summaries and dashboards, but the actual training recipe
belongs in executable code.

Add alternate seeds sparingly. Good alternate seeds include:

- SAC+HER for goal-conditioned manipulation
- vectorized SAC for wall-clock utilization research
- TD3 or PPO when they are credible baselines for the task

Avoid shipping many weak variants. A small number of reliable seeds is more
useful than a large menu of unvalidated ones.

## Dependency Rules

Keep the install surface split by use:

- base: runner, dashboard serving, Gymnasium, NumPy, Pillow, and lightweight
  shared utilities
- `.[mujoco]`: MuJoCo and Gymnasium-Robotics tasks
- `.[panda]`: Panda/PyBullet and PandaGym tasks
- `.[dev]`: tests, packaging, and local development tools

Use precise platform markers. For example, Apple Silicon PyBullet workarounds
should be scoped to Darwin plus arm64, not every arm64 platform.

## Validation Checklist

After adding or changing a task, environment, benchmark, seed, runner behavior,
or dashboard behavior:

- run focused unit or smoke tests for the changed surface
- run the relevant optional-extra smoke when simulator dependencies are involved
- run a one-episode `autoresearch-gym run` smoke for any new seed
- for wall-clock benchmarks, verify a tiny `--train-seconds` override reports
  `stop_reason = "time_budget_exhausted"`
- verify dashboard live metrics and frames when the change affects run artifacts
  or visualization
- verify compact status stderr/file output when the change affects runner status
  reporting
- remove generated `__pycache__`, `*.egg-info`, `autoresearch_runs/`, temporary
  checkpoints, logs, and rendered rollout artifacts before publishing

## Public Repo Hygiene

Do not commit personal run artifacts, local machine paths, one-off utilities,
large checkpoints, generated media, or private session logs.

Keep public examples generic and reproducible. If a note is only useful for one
local session, put it in that session's `outer_loop_log.md`, not in package docs.

When changing docs, keep the boundary clear:

- `AUTORESEARCH.md`: how to run a research session
- `AGENTS.md`: how to maintain and extend the package
- `README.md`: user-facing install, quickstart, task overview, and CLI examples
- `docs/`: focused supporting notes
