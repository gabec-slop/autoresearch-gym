# Trainable Contract Checklist

Use this before treating any edited session-local `train.py` or
`candidates/passNN_*.py` as a real result.

## Pre-Run

- Author exactly one candidate file.
- Keep the benchmark budget mode mutually exclusive:
  - rollout budget: pass `--train-episodes N` and do not pass `--train-seconds`
  - time budget: pass `--train-seconds N` and use only a high episode safety cap
- Confirm `get_candidate()` describes the exact mutation.
- Confirm candidate code returns the standard functions:
  `get_candidate`, `RewardRecipeWrapper`, `train_agent`, and
  `save_agent_checkpoint`.

## Required Training Summary Fields

Every trainable must return:

- `episode_records`
- `total_steps`
- `env_steps`
- `episodes_completed`
- `completed_episodes`
- `episode_batches`
- `last_metrics`

If the trainable performs optimizer updates, it must also return:

- top-level `gradient_updates`
- `last_metrics["gradient_updates"]` while live updates are available

`env_steps` must equal `total_steps`. `completed_episodes` must equal
`episodes_completed`. Policy probe records must carry an `episode` coordinate
that represents completed training rollouts at probe time, not mixed array
index.

## Smoke Command

Run a short budget that is long enough to cross `update_after` for the candidate:

```bash
.venv/bin/python -m autoresearch_gym.cli run \
  --benchmark autoresearch_gym/tasks/bat_to_goal_v0/benchmark.json \
  --session-dir autoresearch_runs/sessions/<session-id> \
  --candidate autoresearch_runs/sessions/<session-id>/candidates/passNN_<slug>.py \
  --tag contract-smoke-passNN \
  --train-episodes 5 \
  --eval-episodes 1 \
  --compact-status \
  --compact-status-file autoresearch_runs/sessions/<session-id>/live/status.log
```

Then validate the generated run bundle:

```bash
.venv/bin/python scripts/check_trainable_contract.py \
  autoresearch_runs/sessions/<session-id>/runs/<run-id>/summary.json \
  --require-gradient-updates
```

Only promote the candidate to a full comparison run after this check passes.

## Before Commit

Before committing package code that touches trainables, runner logging,
dashboard visualization, benchmarks, or task environments, run the full gate:

```bash
.venv/bin/python scripts/pre_commit_checks.py
```

The gate runs:

- seed trainable syntax checks
- `tests/test_smoke.py`
- `scripts/smoke_seed_artifacts.py --timeout 180`

The artifact smoke creates temporary session-backed runs under `/private/tmp`
and verifies live metrics, required logging counters, live frame files, and
sampled rollout manifests/frames for every bundled seed trainable.
