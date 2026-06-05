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
- Treat hanging simulator tests as sandbox-suspect first. If a MuJoCo,
  Gymnasium-Robotics, PyBullet, or visual artifact smoke/pre-commit command
  hangs or times out in Codex's normal sandbox, rerun the exact same command
  outside the sandbox / from a GUI-permitted shell before editing code. In
  Codex, request escalation for that rerun instead of replacing render paths,
  skipping visuals, or weakening tests based only on the sandbox result.

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

If a task wants dashboard charts for task-specific diagnostics, the training run
should emit metadata with the artifacts instead of requiring dashboard
hard-coding. Use a `diagnostic_series` object with a `series` list; each series
should name the metric key, label, color, source such as `info_metrics`, chart
type such as `normalized_line`, and optional group. The dashboard consumes this
metadata from live/final run payloads and falls back to generic update metrics
when no diagnostic series metadata is present.

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

The harness owns sampled trajectory cadence, output paths, and dashboard
consumption. Most seeds should use the default sampled-trajectory path, where
the runner or backend renders the latest checkpoint in the benchmark/eval
context and writes the standard sampled trajectory artifact.

A seed may opt into a task-aware sampled trajectory override only when the
default context cannot represent what the policy was actually trained on. Common
cases include curricula, training-only domain randomization, environment
wrappers, staged terrain/task parameters, or external trainers whose eval/play
configuration is materially different from the train configuration. Keep the
override generic and seed-owned: the seed or backend should generate the same
standard sampled trajectory artifact from the training process or resolved
training configuration, while the harness still decides when to request samples
and where artifacts are written. For recipe-driven backends that support it,
prefer a narrow runner recipe flag such as `sample_trajectory_source` over new
manifest formats or task-specific runner branches.

For in-process trainables, the generic contract is the `live_callback`: when a
seed opts into a non-fallback `sample_trajectory_source`, the callback returns a
`sampled_trajectory_request` at the harness-selected cadence. The trainable may
answer with `sampled_trajectory={"episode": ..., "sample_index": ..., "source":
..., "frames": [...]}` on a later callback call. Frames may be RGB arrays or
paths to rendered images; the live writer owns converting them into the standard
sampled trajectory artifact and dashboard pointers.

Do not use a sampled trajectory override to change evaluation semantics,
primary metrics, fixed eval seeds, or benchmark conditions. It is only for
visual/debug trajectory sampling. If a seed does not opt into custom sampling,
the framework must fall back to the existing generic sampled trajectory behavior.

Use CleanRL-style explicit code over hidden abstractions. Shared utilities are
acceptable only when they preserve readability and do not hide the training
recipe from the autoresearch loop.

External-environment seeds must stay target-agnostic. Do not branch inside a
seed on local vs remote execution, SSH hostnames, Windows drive paths, private
remote roots, or per-machine account names. Benchmark and runner/external
target code own backend selection, path style, staging, launch, artifact fetch,
and redaction. The seed should expose the editable recipe and produce the same
normalized train/eval artifacts regardless of whether the harness executes on
the local machine, a networked Windows GPU box, or another SSH target.

Before running any remote external benchmark or remote autoresearch pass, sync
or explicitly verify the remote checkout is current with the local intended
code. For Git-backed targets, prefer a fast-forward pull on the remote checkout
such as `git -C <remote_root> pull --ff-only` before staging and launching the
run. Record the sync or verification in the session log when running an
autoresearch session. If a remote run starts before this sync/verification, mark
that attempt invalidated and restart it after the remote checkout is current;
do not use its metrics as research evidence.

Remote execution code should stay centralized and script-driven:

- Session init doctor behavior is setup/scaffold behavior, not only a launch
  gate. When `init-session` or `autoresearch-gym doctor --benchmark ...` reports
  missing remote prerequisites for the selected benchmark, the agent should
  repair or scaffold known-safe prerequisites before launching: create required
  directories, clone documented external repos, install missing packages, restore
  compatible CUDA/simulator package versions, or copy required source artifacts.
  Then rerun the doctor for the same benchmark and target. If the needed source,
  revision, credential, or artifact is unknown, stop and record an explicit
  blocker. Do not silently switch to a different benchmark just because the
  doctor failed.
- Autoresearch agents should launch ordinary remote passes with
  `scripts/launch_autoresearch_pass.py`. Use
  `scripts/run_session_remote_pass.py` only for the lower-level case where the
  outer-loop log entry has already been written.
- Dashboard lifecycle is session-owned and default-on for sessions. `init-session`
  and ordinary session passes should start or reuse the session dashboard unless
  `--no-dashboard` is explicitly passed. Use `autoresearch-gym session-dashboard
  ensure/status/teardown --session-dir ...` only to recover, inspect, or stop the
  dashboard for a whole session. `--no-dashboard` must not overwrite
  `live/dashboard.json` with a disabled pointer.
- Do not add new ad hoc remote launch scripts that duplicate checkout
  verification, candidate validation, compact status, dashboard startup, live
  sync, or final artifact fetch. Put shared behavior in
  `autoresearch_gym.external.remote_session`, `autoresearch_gym.external.targets`,
  or the runner.
- Do not put machine-specific remote paths, SSH hostnames, Windows drive roots,
  or account names in seeds, benchmarks, or public docs. Those belong in local
  target config and session logs.
- Do not build Windows PowerShell command strings by hand. Use the shared
  `SshTarget.powershell_command(...)` encoded-command helper and
  `SshTarget.quote_remote(...)` for remote arguments.
- Do not use inline `.venv/bin/python -c ...` snippets, raw `ssh`/`scp`
  sequences, or shell process-list probes as the normal way to inspect or
  recover remote runs. Add or reuse a Python helper in the package/scripts so
  quoting and timeouts are tested.
- Remote run health should come from session files and SSH-side metadata, not
  from local wrapper assumptions. If `live/status.log` becomes older than the
  configured stale threshold while the remote command is still running, fail the
  pass and clean up only processes matching run-specific command terms.
- Final remote fetch should retrieve small, critical correctness artifacts
  first (`summary.json`, eval/train JSON, benchmark/candidate snapshots, and
  trainable snapshot), then raise if any critical file is still missing. Large
  checkpoints, media, trajectory dirs, and archive fetches are optional follow-up
  artifacts and must not block access to the run result, especially on Windows
  SSH targets.

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

- while iterating, prefer `.venv/bin/python scripts/pre_commit_checks.py
  --affected` to run a path-sensitive validation subset for the files changed
  since `HEAD`; use `--dry-run` first when you want to inspect the selected
  checks. The git pre-commit hook runs the same affected plan against staged
  paths only.
- before promotion, release, or broad runner/task changes, run
  `.venv/bin/python scripts/pre_commit_checks.py`; this is the repo-level gate
  for unit smoke tests plus seed logging/visual artifact smoke
- run focused unit or smoke tests for the changed surface
- run the relevant optional-extra smoke when simulator dependencies are involved
- run a one-episode `autoresearch-gym run` smoke for any new seed
- for wall-clock benchmarks, verify a tiny `--train-seconds` override reports
  `stop_reason = "time_budget_exhausted"`
- verify dashboard live metrics and frames when the change affects run artifacts
  or visualization
- verify compact status stderr/file output when the change affects runner status
  reporting
- for remote external runs, sync or verify the remote checkout before launch and
  record that fact in the session log
- for remote management changes, add or update focused tests for Windows quoting,
  checkout verification, live sync, stale-status detection/process cleanup, and
  final artifact fetch ordering. Prefer mock-based tests of command construction
  and fetch ordering over live SSH tests unless the user explicitly asks for
  target-machine validation
- for dashboard lifecycle changes, add or update focused tests for session-level
  ensure/reuse, guarded teardown, stale pointer handling, and `--no-dashboard`
  preserving the existing session pointer
- for trainable, runner, dashboard, benchmark, or task changes, do not use
  `scripts/pre_commit_checks.py --skip-artifact-smoke` for the full gate; the
  full artifact smoke must pass before promotion
- on macOS, if MuJoCo/PyBullet visual smoke fails inside a restricted sandbox,
  rerun the same pre-commit command from a normal GUI-permitted shell before
  changing render code
- Codex command rules for the simulator visual smoke entrypoints live in
  `.codex/rules/autoresearch.rules`. Keep those rules narrow and do not add
  broad Python prefixes; the intent is to run the real renderer outside the
  sandbox, not to allow arbitrary unsandboxed Python.
- if any simulator smoke or pre-commit test hangs, timeouts, or emits
  CoreGraphics/WindowServer/service-connection errors under Codex, rerun the
  exact same command with sandbox escalation / a GUI-permitted shell before
  marking it failed or changing code
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
