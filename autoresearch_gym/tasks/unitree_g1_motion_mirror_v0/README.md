# Unitree G1 Motion Mirror

Unitree G1 motion-capture mirroring benchmark for external simulator
execution. The benchmark fixes the motion family, eval cases, primary metric,
budget, and artifact contract; the trainable owns the learning recipe.

The bundled task has two supported paths:

- `benchmark.json` with `seed_trainable.py`: MJLab-backed PPO training and
  evaluation through `UnitreeExternalBackend`.
- `benchmark_lower_level.json` with `seed_trainable_lower_level_cleanrl.py`:
  lower-level CleanRL-style PPO where the seed owns the adapter and training
  loop directly.

Both benchmark files default to `execution_target = "local"`. That means a
Windows or Linux machine with the required simulator stack can run the task
directly without any private target config. Remote execution is selected at run
time with `--execution-target <name>` and an ignored target config file; the
seed file should not branch on whether the target is local or remote.

## MJLab Setup

The MJLab path expects the upstream simulator repos and source motion artifact
under the checkout root:

```text
.external/mjlab
.external/unitree_rl_mjlab
autoresearch_runs/source_motions/pbhc_side_kick_mjlab_motion.npz
```

The backend launches bridge scripts from the local or remote checkout, runs the
MJLab train/eval flow, normalizes metrics into the autoresearch artifact schema,
and exports checkpoint/media artifacts for the dashboard.

## Local Run

Run from the machine that has MJLab and CUDA Torch installed:

```bash
uv run autoresearch-gym run \
  --benchmark autoresearch_gym/tasks/unitree_g1_motion_mirror_v0/benchmark.json \
  --seed-candidate autoresearch_gym/tasks/unitree_g1_motion_mirror_v0/seed_trainable.py \
  --candidate autoresearch_gym/tasks/unitree_g1_motion_mirror_v0/seed_trainable.py \
  --tag g1-mjlab-smoke \
  --train-episodes 1 \
  --eval-episodes 1
```

For a quick lower-level smoke:

```bash
uv run autoresearch-gym run \
  --benchmark autoresearch_gym/tasks/unitree_g1_motion_mirror_v0/benchmark_lower_level.json \
  --seed-candidate autoresearch_gym/tasks/unitree_g1_motion_mirror_v0/seed_trainable_lower_level_cleanrl.py \
  --candidate autoresearch_gym/tasks/unitree_g1_motion_mirror_v0/seed_trainable_lower_level_cleanrl.py \
  --tag g1-lower-level-smoke \
  --train-seconds 2 \
  --eval-episodes 1
```

## Remote Run

Copy private SSH settings from `examples/remote_targets.example.toml` into an
ignored config such as `.autoresearch.local.toml` or
`~/.config/autoresearch-gym/targets.toml`, then override only the execution
target:

```bash
uv run autoresearch-gym run \
  --benchmark autoresearch_gym/tasks/unitree_g1_motion_mirror_v0/benchmark.json \
  --seed-candidate autoresearch_gym/tasks/unitree_g1_motion_mirror_v0/seed_trainable.py \
  --candidate autoresearch_gym/tasks/unitree_g1_motion_mirror_v0/seed_trainable.py \
  --tag g1-mjlab-remote-smoke \
  --train-episodes 1 \
  --eval-episodes 1 \
  --execution-target windows_gpu
```

The candidate and benchmark are the same in local and remote mode. Target
selection, path style, staging, and artifact fetch are handled by the runner and
external backend.

## Metrics And Artifacts

The primary metric is `eval_avg_mpkpe`, minimized over fixed motion-mirroring
eval cases. Runs should produce normalized train/eval summaries, checkpoint
metadata, compact status, and sampled rollout media when the simulator render
path is available.
