"""MuJoCo/MJWarp Menagerie Panda pick-and-place task.

Two benchmark contracts live in this task dir:

- `benchmark*.json`: lift-gated pick-and-place (success requires lifting the
  cube before placing it), with `eval_cases.json`.
- `benchmark_pandagym_dense*.json`: panda-gym PickAndPlace dense semantics
  (reward = -distance, success = distance < 0.05, no lift gate), with
  `eval_cases_pandagym_dense.json`.

Seeds resolve the reward contract from the benchmark's env_kwargs at train
time. `oracle_policy.py` is task-owned validation infrastructure (a scripted
FK pinch-point pick-place controller), not a seed.
"""
