# Project TODO

- [ ] Implement a Hugging Face Jobs execution target for remote compute.
  - Add an `HfJobsTarget` or equivalent target adapter beside local, fake, and
    SSH targets.
  - Support Docker-based jobs for simulator-heavy workloads rather than assuming
    lightweight Python scripts are enough.
  - Define how run bundles, checkpoints, normalized summaries, logs, and media
    are persisted after the job exits.
  - Add budget controls and clear timeout behavior before running expensive
    managed compute tests.
  - Add fake/unit tests for target config parsing, redaction, command creation,
    artifact fetch, and failure handling.
  - Validate with a real paid smoke only after a compute budget is explicitly
    allocated.
