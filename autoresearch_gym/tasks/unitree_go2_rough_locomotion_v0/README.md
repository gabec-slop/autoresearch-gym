# Unitree Go2 Rough Locomotion

Unitree Go2 rough-terrain locomotion benchmark for external simulator
execution. The benchmark fixes rough-terrain eval cases, command targets,
primary reward metric, budget, and artifact contract; the trainable owns the
learning recipe.

The bundled task has two supported paths:

- `benchmark.json` with `seed_trainable.py`: MJLab-backed PPO training and
  evaluation through `UnitreeExternalBackend`.
- `benchmark_lower_level.json` with `seed_trainable_lower_level_cleanrl.py`:
  lower-level CleanRL-style PPO where the seed owns the adapter and training
  loop directly.

Both benchmark files default to `execution_target = "local"`. A Windows or
Linux machine with the simulator stack can run the task directly. Remote
execution is selected at run time with `--execution-target <name>` and an
ignored target config file; the seed file should not branch on whether the
target is local or remote.

The public task name uses Go2 because that is the Unitree quadruped target
validated for this rough-locomotion setup.

## MJLab Setup

The MJLab path expects the upstream Unitree MJLab repository under the checkout
root:

```text
.external/unitree_rl_mjlab
```

The backend launches bridge scripts from the local or remote checkout, runs the
MJLab train/eval flow, normalizes metrics into the autoresearch artifact schema,
and exports checkpoint/media artifacts for the dashboard. The bundled seed keeps
the main MJLab levers in Python constants, including parallel env count, PPO
settings, curriculum, rewards, termination, randomization, and command ranges.

The default seed uses a conservative Windows-tested parallelism setting rather
than the largest MJLab default, so it can smoke on smaller NVIDIA GPUs.

## Local Run

Run from the machine that has MJLab and CUDA Torch installed:

```bash
uv run autoresearch-gym run \
  --benchmark autoresearch_gym/tasks/unitree_go2_rough_locomotion_v0/benchmark.json \
  --seed-candidate autoresearch_gym/tasks/unitree_go2_rough_locomotion_v0/seed_trainable.py \
  --candidate autoresearch_gym/tasks/unitree_go2_rough_locomotion_v0/seed_trainable.py \
  --tag go2-mjlab-smoke \
  --train-episodes 1 \
  --eval-episodes 1
```

For a quick lower-level smoke:

```bash
uv run autoresearch-gym run \
  --benchmark autoresearch_gym/tasks/unitree_go2_rough_locomotion_v0/benchmark_lower_level.json \
  --seed-candidate autoresearch_gym/tasks/unitree_go2_rough_locomotion_v0/seed_trainable_lower_level_cleanrl.py \
  --candidate autoresearch_gym/tasks/unitree_go2_rough_locomotion_v0/seed_trainable_lower_level_cleanrl.py \
  --tag go2-lower-level-smoke \
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
  --benchmark autoresearch_gym/tasks/unitree_go2_rough_locomotion_v0/benchmark.json \
  --seed-candidate autoresearch_gym/tasks/unitree_go2_rough_locomotion_v0/seed_trainable.py \
  --candidate autoresearch_gym/tasks/unitree_go2_rough_locomotion_v0/seed_trainable.py \
  --tag go2-mjlab-remote-smoke \
  --train-episodes 1 \
  --eval-episodes 1 \
  --execution-target windows_gpu
```

The candidate and benchmark are the same in local and remote mode. Target
selection, path style, staging, and artifact fetch are handled by the runner and
external backend.

## Metrics And Artifacts

The primary metric is `eval_avg_return`, maximized from the MJLab rollout
reward signal. The MJLab-backed path does not currently report a binary success
rate because MJLab provides reward, termination, and rollout diagnostics rather
than a canonical Gym-style `is_success` flag for this task. Runs should produce
normalized train/eval summaries, checkpoint metadata, compact status, and
sampled rollout media when the simulator render path is available.
