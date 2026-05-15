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

Panda/PyBullet and PandaGym tasks:

```powershell
uv sync --extra panda
```

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
