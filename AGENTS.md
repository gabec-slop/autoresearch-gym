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

`train_agent(...)` must return dashboard/autoresearch curve data in
`episode_records`. New records should use the shared constructors in
`autoresearch_gym.runner.curves` so every task speaks the same logging contract.
The trainable logging terms are:

- `env step`: one environment transition from one simulator instance. For
  vectorized training, one `envs.step(...)` call contributes `num_envs` env
  steps. `train_summary["total_steps"]` and optional
  `train_summary["env_steps"]` both mean cumulative env steps and must match.
- `completed episode`: one finished rollout in one simulator instance, from
  reset until terminated or truncated.
- `episode batch`: one dashboard/chart record. A batch may summarize one
  completed rollout or many completed rollouts from a vectorized/windowed
  collector. The record field named `episode` is the episode-batch/chart index,
  not necessarily total completed episodes.
- `epoch`: do not use this word for trainable internals. In autoresearch logs it
  may refer to an outer-loop candidate/pass, not an RL rollout or minibatch.

The standard record types are:

- `train_episode`: one completed training episode.
- `train_collection_window`: sampled/windowed collection stats for highly
  batched simulators. Use this instead of logging every world episode when doing
  so would add too much storage or synchronization overhead. It must include
  `episodes_in_window`; `return` and `length` should be window averages, `step`
  should be cumulative env steps, and `env_steps_in_window` is recommended.
- `policy_probe`: deterministic train-time policy performance. The runner owns
  generic probes when the agent exposes `act(obs, deterministic=True)`; special
  external trainers may implement `probe_policy(...)`.

If training completes any episodes or steps, `episode_records` must contain
records with at least `record_type`, `return`, `length`, and a charting axis such
as `step`, `elapsed_seconds`, or `episode`. Windowed records must include
`episodes_in_window`. Older artifacts without `record_type` remain dashboard
compatible and are treated as `train_episode`, but new seeds should emit typed
records.

`train_agent(...)` summaries should expose:

- `total_steps`: cumulative env steps.
- `env_steps`: optional alias for `total_steps`; if present it must match.
- `episodes_completed`: total completed training rollouts, summing
  `episodes_in_window` for collection-window records.
- `episode_batches`: number of dashboard/chart collection records.
- `episode_records`: the typed episode-batch records described above.

Runner-owned probes must not affect the training recipe or episode cap. Seeds
should pass the current `agent` into `live_callback(...)`; the runner records
policy probes separately for live/final artifacts and aggregates collection
records separately from probe records.

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

- before committing code, run `.venv/bin/python scripts/pre_commit_checks.py`; this is the
  repo-level gate for unit smoke tests plus seed logging/visual artifact smoke
- run focused unit or smoke tests for the changed surface
- run the relevant optional-extra smoke when simulator dependencies are involved
- run a one-episode `autoresearch-gym run` smoke for any new seed
- for wall-clock benchmarks, verify a tiny `--train-seconds` override reports
  `stop_reason = "time_budget_exhausted"`
- verify dashboard live metrics and frames when the change affects run artifacts
  or visualization
- verify compact status stderr/file output when the change affects runner status
  reporting
- for trainable, runner, dashboard, benchmark, or task changes, do not use
  `scripts/pre_commit_checks.py --skip-artifact-smoke`; the full artifact smoke
  must pass before commit
- on macOS, if MuJoCo/PyBullet visual smoke fails inside a restricted sandbox,
  rerun the same pre-commit command from a normal GUI-permitted shell before
  changing render code
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
