# Autoresearch Session Runbook

This file is for running research sessions. It is meant to be read by Codex,
Claude Code, or another coding agent before modifying session candidates.

For project maintenance, new task design, environment implementation, dependency
changes, or release work, read `AGENTS.md` instead.

## Goal

Improve a trainable controller for the fixed Gymnasium benchmark selected by
`--benchmark`.

Optimize the benchmark's `primary_metric` according to `primary_metric_mode`.
Use benchmark-declared secondary metrics, average eval return, simpler code, and
lower training cost as tie breakers, in that order.

Never compare candidates across different training or evaluation budgets unless
the log explicitly labels the comparison as budget-mismatched.

## Default Mode

The default mode is cold-start learning-process search.

Each pass trains a fresh policy from scratch under the fixed benchmark budget.
The point is to improve the training recipe: code structure, losses, reward
shaping, architecture, replay behavior, update cadence, exploration, curriculum,
vectorization, utilization, or any other mechanism that can be expressed in the
training code. The point is not to continue a partially trained policy until it
reaches the best score.

Do not use checkpoint warm starts unless the user explicitly asks for a
warm-start diagnostic. In normal runs, do not pass `--init-checkpoint`,
`--init-from-run`, or `--init-from-best`.

When this file says "parent", it means candidate-code parent, not policy
checkpoint parent.

## Session Invariants

- The benchmark, eval cases, success definition, environment, and runner are
  fixed during a session.
- Only edit files under
  `autoresearch_runs/sessions/<session-id>/candidates/`.
- Candidate files are authored by the model, one at a time.
- Scripts may run one already-authored candidate and collect artifacts. Scripts
  must not generate, mutate, queue, or prewrite `passNN_*.py` files.
- Pass 1 is always a verbatim copy of the selected seed trainable, usually
  `candidates/pass01_baseline.py`. Do not edit pass 1 code or `get_candidate()`.
- Later passes are bounded mutations from the chosen candidate-code parent.
- Each ordinary pass should produce `lineage.mode = "from_scratch"` and
  `evolution.uses_warm_start = false`.

If a task or runner bug is suspected, stop the candidate loop and write the
suspected issue to `outer_loop_log.md`. Do not silently change the benchmark to
make a candidate look better.

## Candidate Contract

Every candidate must expose the same public contract as the seed:

- `get_candidate`
- `RewardRecipeWrapper`
- `train_agent`
- `save_agent_checkpoint`

`get_candidate()` may return a plain free-text string. It is metadata for the
dashboard, summaries, and research log; it is not the training recipe. Put the
recipe in code.

After pass 1, update `get_candidate()` whenever you mutate a candidate. The text
should name the hypothesis, the changed mechanism, and the expected effect. Do
not leave the seed description on changed code.

Good examples:

```python
def get_candidate() -> str:
    return (
        "FetchPushDense SAC candidate testing goal-delta reward shaping during "
        "training. Keeps the official dense task reward for evaluation, but "
        "adds a training-only progress bonus to encourage earlier contact and "
        "object motion toward the desired goal."
    )
```

```python
def get_candidate() -> str:
    return (
        "Hopper SAC candidate testing a slower actor update schedule and wider "
        "critics. Hypothesis: delaying policy updates while increasing critic "
        "capacity improves early stability under the fixed budget."
    )
```

Prefer CleanRL-style self-contained trainables. The candidate file should expose
the actual recipe in editable code: networks, replay, optimizer setup, losses,
update cadence, reward transforms, logging, exploration, vectorization, and any
other task-relevant mechanisms.

Before committing any change to a seed trainable, runner logging, dashboard
visualization, benchmark, or task environment, run:

```bash
.venv/bin/python scripts/pre_commit_checks.py
```

This gate runs unit smoke tests plus the full seed artifact smoke matrix for
live metrics, frames, and sampled rollouts. Session-local candidate experiments
do not need this gate for every exploratory pass, but any bug fix promoted back
to package code does.

## Before Starting

Read:

1. `README.md`
2. this file
3. the selected task's benchmark JSON
4. the selected seed trainable

Choose:

- budget style: `benchmark.json`, `benchmark_wall_clock.json`, or a task-specific
  alternative such as `benchmark_vectorized_wall_clock.json`
- seed trainable: usually `seed_trainable.py`, with task-specific alternatives
  such as `seed_trainable_her.py` or `seed_trainable_vectorized.py`
- session label

Budget choice matters. Use episode budgets for sample-count comparisons and
simple debugging. Use wall-clock budgets for utilization, vectorization, and
fixed-time learning-efficiency questions.

## Start A Session

Create the session:

```bash
autoresearch-gym init-session \
  --label <label> \
  --benchmark autoresearch_gym/tasks/<task_name>/<benchmark>.json \
  --seed-candidate autoresearch_gym/tasks/<task_name>/<seed_trainable>.py
```

`init-session` records the selected seed path and creates an empty `candidates/`
directory. It does not create pass files.

For pass 1, create `candidates/pass01_baseline.py` as a verbatim copy of the
selected seed trainable. Do not edit the copied file.

Run pass 1:

```bash
autoresearch-gym run \
  --benchmark autoresearch_gym/tasks/<task_name>/<benchmark>.json \
  --seed-candidate autoresearch_gym/tasks/<task_name>/<seed_trainable>.py \
  --session-dir autoresearch_runs/sessions/<session-id> \
  --candidate autoresearch_runs/sessions/<session-id>/candidates/pass01_baseline.py \
  --tag pass01-baseline
```

Unless the user explicitly asks for a headless run, share the dashboard URL for
every session:

```text
http://127.0.0.1:4174/dashboard/?session=autoresearch_runs/sessions/<session-id>
```

If the dashboard server is not running, start it with `autoresearch-gym
dashboard` and verify the URL is reachable.

For long runs, use compact status output so humans and coding agents can monitor
progress without parsing the final JSON summary or relying on live stderr
streaming:

```bash
autoresearch-gym run \
  --benchmark autoresearch_gym/tasks/<task_name>/<benchmark>.json \
  --session-dir autoresearch_runs/sessions/<session-id> \
  --candidate autoresearch_runs/sessions/<session-id>/candidates/passNN_<slug>.py \
  --tag passNN-<slug> \
  --compact-status \
  --compact-status-file autoresearch_runs/sessions/<session-id>/live/status.log
```

Tail `autoresearch_runs/sessions/<session-id>/live/status.log` for compact
progress lines. On wall-clock benchmarks, `pct` tracks elapsed time over
`train_seconds`; on episode-budget benchmarks, it tracks completed episodes over
`train_episodes`. Keep stdout reserved for the final JSON summary.

## Per-Pass Loop

For every later pass:

1. Inspect `results.jsonl`, `outer_loop_log.md`, and the latest relevant
   `runs/<run-id>/summary.json`.
2. When useful, inspect `runs/<run-id>/train_episodes.json` and
   `runs/<run-id>/eval_episodes.json` for curve shape: plateaus, collapses,
   late improvement, reward spikes, high variance, episode length changes, and
   secondary metrics.
3. For wall-clock or GPU work, inspect `system_utilization_notes` and
   `system_utilization` in `summary.json`.
4. Write one bounded hypothesis in `outer_loop_log.md` before editing.
5. As the model, create exactly one new `candidates/passNN_<slug>.py` file from
   the chosen candidate-code parent.
6. Make one coherent recipe change in that file.
7. Update `get_candidate()` to describe that exact changed recipe.
8. Run the already-authored candidate:

```bash
autoresearch-gym run \
  --benchmark autoresearch_gym/tasks/<task_name>/<benchmark>.json \
  --session-dir autoresearch_runs/sessions/<session-id> \
  --candidate autoresearch_runs/sessions/<session-id>/candidates/passNN_<slug>.py \
  --tag passNN-<slug>
```

9. Inspect the generated `summary.json`.
10. Append the hypothesis, changed mechanism, primary metric, secondary metrics,
    decision, interpretation, and next idea to `outer_loop_log.md`.
11. Choose the next candidate-code parent from the best non-dominated candidate
    frontier.

Do not create pass `N + 1` while pass `N` is still running. Do not create a
queue of candidates at session start. Do not run a "full loop" script that both
writes candidates and executes them.

You may keep a tentative idea backlog in `outer_loop_log.md` for plausible
future hypotheses. Treat it as a scratchpad, not an execution queue: after each
run, inspect the generated `summary.json` and use the evidence at hand to choose
the best next thing to try. The next pass may come from the backlog, or it may be
a new idea sparked by the previous run.

## Keep Or Reject

Keep a candidate as the new code parent if it improves the benchmark primary
metric from a cold start under the fixed budget.

Keep it on the frontier, but not as the sole parent, if the primary metric is
tied and secondary metrics improve without a meaningful return or complexity
regression.

Reject it if it crashes, regresses the primary metric, only improves training
reward, depends on changed eval conditions, or adds complexity without a
fixed-eval gain.

If a candidate crashes from a trivial syntax, import, or shape mistake, repair it
at most twice. After two repair attempts, mark it failed and move on.

## Budget Discipline

Use tiny smoke runs only to check that a candidate executes. Do not treat smoke
runs as evidence of learning.

Use the benchmark default budget for normal comparisons unless the user asks for
a different budget. Use larger confirmation runs only after a candidate wins at
the normal budget.

Always record `train_episodes`, `eval_episodes`, seed settings, benchmark path,
candidate path, and primary metric in the log entry.

## Stopping

When stopped, leave the session resumable:

- latest metrics written
- current candidate-code parent identified
- failed ideas recorded
- next plausible hypothesis noted
- dashboard URL and session path available
