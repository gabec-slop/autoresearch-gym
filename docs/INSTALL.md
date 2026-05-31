# Install

Autoresearch Gym uses optional dependency groups so the core package can stay
small. Install the simulator stack for the tasks you want to run.

## macOS or Linux

Core package only:

```bash
uv venv --seed --python 3.10 .venv
uv sync
```

MuJoCo locomotion, control, and manipulation tasks:

```bash
uv sync --extra mujoco
```

MuJoCo Warp tasks using MuJoCo Menagerie assets:

```bash
uv sync --extra mujoco-warp
```

Panda/PyBullet and PandaGym tasks:

```bash
uv sync --extra panda
```

On Apple Silicon, `panda-gym==3.0.7` declares an upstream `pybullet`
dependency. The `panda` extra installs `pybullet-arm64` instead, so install
`panda-gym` without dependencies after syncing:

```bash
uv pip install --no-deps "panda-gym==3.0.7"
```

Development tools:

```bash
uv sync --extra dev
```

You can combine extras:

```bash
uv sync --extra mujoco --extra panda --extra dev
```

Other platforms use upstream `pybullet` through the `panda` extra.

## Windows PowerShell

Core package only:

```powershell
uv venv --seed --python 3.10 .venv
uv sync
```

MuJoCo tasks:

```powershell
uv sync --extra mujoco
```

MuJoCo Warp tasks using MuJoCo Menagerie assets:

```powershell
uv sync --extra mujoco-warp
```

Panda/PyBullet and PandaGym tasks:

```powershell
uv sync --extra panda
```

If `pybullet` does not have a compatible wheel for your Python/platform
combination, `uv sync --extra panda` may try to build it locally. On Windows
that requires Microsoft Visual C++ Build Tools. MuJoCo tasks can still be run
without the `panda` extra.

Development tools:

```powershell
uv sync --extra dev
```

## Optional Windows CUDA Torch

The default PyTorch package may be enough for CPU or some local setups. For an
NVIDIA GPU, use the current command from the official PyTorch install selector:

<https://docs.pytorch.org/get-started/locally/>

After syncing the desired extra, replace or upgrade Torch inside the venv with
the selector's Windows/Pip/CUDA command. The shape is:

```powershell
uv pip install torch --index-url https://download.pytorch.org/whl/<cuda-wheel-tag>
```

For example, if the selector recommends CUDA 12.6 wheels:

```powershell
uv pip install torch --index-url https://download.pytorch.org/whl/cu126
```

Verify CUDA before launching longer runs:

```powershell
uv run python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

Or run the package health check:

```powershell
uv run autoresearch-gym doctor --strict
```

The health check prints JSON with the installed Torch version, Torch CUDA
runtime, selected training device, and any GPUs visible to `nvidia-smi`. In
`--strict` mode it exits nonzero if `nvidia-smi` sees an NVIDIA GPU but
`torch.cuda.is_available()` is false. That usually means the venv installed a
CPU-only Torch wheel and should be repaired with the PyTorch selector command.

## Unitree MJLab And Lower-Level Tasks

The Unitree G1 and Go2 tasks are external-environment tasks. The base package
contains the runner integration, benchmark files, seed trainables, and artifact
normalization code. It does not vendor MJLab, Unitree simulator repositories,
large motion files, CUDA drivers, or private machine configuration.

For the MJLab-backed Unitree tasks, set up the simulator stack in the checkout
where the harness will run:

```text
.external/mjlab
.external/unitree_rl_mjlab
```

The G1 motion-mirroring task also expects the source motion artifact:

```text
autoresearch_runs/source_motions/pbhc_side_kick_mjlab_motion.npz
```

Install this package and the Python dependencies needed by the selected
simulator stack, then verify Torch sees the GPU:

```powershell
uv venv --seed --python 3.10 .venv
uv sync --extra mujoco --extra dev
uv run autoresearch-gym doctor --strict
```

On Windows with NVIDIA GPUs, install a CUDA-enabled Torch wheel after syncing if
`doctor --strict` reports CPU-only Torch. Use the command from the official
PyTorch selector for the installed driver/CUDA combination.

The Unitree benchmark files default to local execution:

```powershell
uv run autoresearch-gym run `
  --benchmark autoresearch_gym/tasks/unitree_go2_rough_locomotion_v0/benchmark.json `
  --seed-candidate autoresearch_gym/tasks/unitree_go2_rough_locomotion_v0/seed_trainable.py `
  --candidate autoresearch_gym/tasks/unitree_go2_rough_locomotion_v0/seed_trainable.py `
  --tag go2-local-smoke `
  --train-episodes 1 `
  --eval-episodes 1
```

To drive the same benchmark from another machine, keep SSH details in ignored
target config such as `.autoresearch.local.toml` or
`~/.config/autoresearch-gym/targets.toml`, based on
`examples/remote_targets.example.toml`, then pass only the target name:

```bash
uv run autoresearch-gym run \
  --benchmark autoresearch_gym/tasks/unitree_g1_motion_mirror_v0/benchmark.json \
  --seed-candidate autoresearch_gym/tasks/unitree_g1_motion_mirror_v0/seed_trainable.py \
  --candidate autoresearch_gym/tasks/unitree_g1_motion_mirror_v0/seed_trainable.py \
  --tag g1-remote-smoke \
  --train-episodes 1 \
  --eval-episodes 1 \
  --execution-target windows_gpu
```

The same seed and benchmark are used in local and remote mode. The runner and
external backend handle target resolution, path style, command launch, and
artifact fetch. For SSH targets, the implemented artifact sync mode is `scp`.

The same SSH target config can also run ordinary in-process Gymnasium
benchmarks on the remote machine:

```bash
uv run autoresearch-gym run \
  --benchmark autoresearch_gym/tasks/panda_pick_and_place_mjwarp_v0/benchmark_wall_clock.json \
  --session-dir autoresearch_runs/sessions/<session-id> \
  --candidate autoresearch_runs/sessions/<session-id>/candidates/pass01_baseline.py \
  --tag panda-mjwarp-remote \
  --execution-target windows_gpu \
  --compact-status \
  --compact-status-file autoresearch_runs/sessions/<session-id>/live/status.log
```

For in-process remote benchmarks, the local runner stages the benchmark,
eval-case bank, and selected candidate file, mirrors live dashboard artifacts
back into the local session, and appends the final result locally. Keep the
remote checkout and simulator dependencies current before launching; package
code changes outside those staged files must already exist on the remote.

The lower-level CleanRL Unitree benchmarks do not require the MJLab bridge
scripts, but they still exercise the external artifact contract and should be
run on a machine with the simulator/training dependencies needed by the seed.
