from __future__ import annotations

from typing import Any

import numpy as np

"""Shared trainable logging contract.

Terminology used by train_agent summaries, live dashboard metrics, and curve
records:

- env step: one transition from one simulator instance. For vectorized training,
  one vector step contributes NUM_ENVS env steps. train_summary["total_steps"]
  and train_summary["env_steps"] both mean cumulative env steps.
- completed episode: one finished rollout from reset to terminated/truncated in
  one simulator instance.
- episode batch: one chart/logging record. A train_episode record represents one
  completed episode; a train_collection_window record represents a batch/window
  of completed episodes and must report episodes_in_window.
- policy probe: deterministic train-time evaluation record. Probe records do not
  count toward completed training episodes.
"""

TRAIN_EPISODE = "train_episode"
TRAIN_COLLECTION_WINDOW = "train_collection_window"
POLICY_PROBE = "policy_probe"

COLLECTION_RECORD_TYPES = {"", TRAIN_EPISODE, TRAIN_COLLECTION_WINDOW}
RECORD_TYPES = {TRAIN_EPISODE, TRAIN_COLLECTION_WINDOW, POLICY_PROBE}


def elapsed_seconds_since(start_time: float) -> float:
    import time

    return float(time.time() - start_time)


def scalar_info_metrics(info: dict[str, Any]) -> dict[str, float | bool]:
    metrics: dict[str, float | bool] = {}
    for key, value in info.items():
        if key == "is_success":
            continue
        if isinstance(value, (bool, np.bool_)):
            metrics[key] = bool(value)
        elif isinstance(value, (int, float, np.integer, np.floating)):
            metrics[key] = float(value)
    return metrics


def record_type(record: dict[str, Any]) -> str:
    value = str(record.get("record_type") or "")
    return value or TRAIN_EPISODE


def is_policy_probe_record(record: dict[str, Any]) -> bool:
    return record_type(record) == POLICY_PROBE


def is_collection_record(record: dict[str, Any]) -> bool:
    return record_type(record) in {TRAIN_EPISODE, TRAIN_COLLECTION_WINDOW}


def collection_episode_records(episode_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in episode_records if is_collection_record(record)]


def episodes_in_record(record: dict[str, Any]) -> int:
    if record_type(record) == TRAIN_COLLECTION_WINDOW:
        return max(0, int(record.get("episodes_in_window") or 0))
    if is_collection_record(record):
        return 1
    return 0


def completed_episode_count(episode_records: list[dict[str, Any]]) -> int:
    return int(sum(episodes_in_record(record) for record in episode_records))


def make_train_episode_record(
    *,
    episode: int,
    return_value: float,
    length: int,
    success: bool = False,
    step: int | None = None,
    elapsed_seconds: float | None = None,
    info_metrics: dict[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Create one completed-rollout record.

    `episode` is the episode-batch/chart index. `step` is cumulative env steps,
    not vector-step calls. `length` is this rollout's length in env steps.
    """
    record: dict[str, Any] = {
        "record_type": TRAIN_EPISODE,
        "episode": int(episode),
        "return": float(return_value),
        "length": int(length),
        "success": bool(success),
        "info_metrics": dict(info_metrics or {}),
    }
    if step is not None:
        record["step"] = int(step)
    if elapsed_seconds is not None:
        record["elapsed_seconds"] = float(elapsed_seconds)
    record.update(extra)
    return record


def make_train_collection_window_record(
    *,
    episode: int,
    return_value: float,
    length: float,
    episodes_in_window: int,
    success: bool = False,
    step: int | None = None,
    env_steps_in_window: int | None = None,
    elapsed_seconds: float | None = None,
    info_metrics: dict[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Create one episode-batch/window record for high-throughput collectors.

    `episode` is the episode-batch/chart index, not total completed episodes.
    `episodes_in_window` is the number of completed rollouts summarized by this
    record. `return_value` and `length` should be window averages unless the
    record explicitly documents another aggregation in `info_metrics`. `step` is
    cumulative env steps. `env_steps_in_window` is optional but recommended for
    vectorized or batched collectors.
    """
    record: dict[str, Any] = {
        "record_type": TRAIN_COLLECTION_WINDOW,
        "episode": int(episode),
        "return": float(return_value),
        "length": float(length),
        "success": bool(success),
        "episodes_in_window": int(episodes_in_window),
        "info_metrics": dict(info_metrics or {}),
    }
    if step is not None:
        record["step"] = int(step)
    if env_steps_in_window is not None:
        record["env_steps_in_window"] = int(env_steps_in_window)
    if elapsed_seconds is not None:
        record["elapsed_seconds"] = float(elapsed_seconds)
    record.update(extra)
    return record


def make_policy_probe_record(
    *,
    episode: int,
    return_value: float,
    length: float,
    step: int,
    elapsed_seconds: float,
    probe_episodes: int,
    probe_seed_start: int,
    success_rate: float = 0.0,
    info_metrics: dict[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "record_type": POLICY_PROBE,
        "episode": int(episode),
        "return": float(return_value),
        "length": float(length),
        "success": bool(success_rate >= 1.0),
        "step": int(step),
        "elapsed_seconds": float(elapsed_seconds),
        "probe_episodes": int(probe_episodes),
        "probe_seed_start": int(probe_seed_start),
        "deterministic": True,
        "info_metrics": {
            "policy_probe_return": float(return_value),
            "policy_probe_length": float(length),
            "policy_probe_success_rate": float(success_rate),
            "policy_probe_episodes": float(probe_episodes),
            **dict(info_metrics or {}),
        },
    }
    record.update(extra)
    return record


def aggregate_info_metrics(episode_records: list[dict[str, Any]]) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    bool_keys: set[str] = set()
    for record in episode_records:
        for key, value in record.get("info_metrics", {}).items():
            if isinstance(value, bool):
                bool_keys.add(key)
                values.setdefault(key, []).append(1.0 if value else 0.0)
            else:
                values.setdefault(key, []).append(float(value))

    aggregates: dict[str, float] = {}
    for key, items in values.items():
        if not items:
            continue
        aggregate_key = f"{key}_rate" if key in bool_keys else f"avg_{key}"
        aggregates[aggregate_key] = float(np.mean(items))
    return aggregates


def validate_train_curve_contract(train_summary: dict[str, Any]) -> None:
    if "episode_records" not in train_summary:
        raise ValueError("train_agent summary must include episode_records for dashboard and autoresearch curves.")
    episode_records = train_summary["episode_records"]
    if not isinstance(episode_records, list):
        raise TypeError("train_agent summary episode_records must be a list.")

    completed = int(train_summary.get("episodes_completed") or 0)
    if completed > 0 and not episode_records:
        raise ValueError(
            "train_agent reported completed episodes but returned no episode_records. "
            "Dashboard visualization and autoresearch curve analysis require at least sampled or windowed "
            "records with return and length."
        )
    total_steps = int(train_summary.get("total_steps") or 0)
    env_steps = train_summary.get("env_steps")
    if env_steps is not None and int(env_steps) != total_steps:
        raise ValueError("train_agent summary env_steps must match total_steps; both mean cumulative env steps.")
    if total_steps > 0 and not episode_records and train_summary.get("curve_status") != "unsupported":
        raise ValueError(
            "train_agent reported training steps but returned no episode_records. "
            "Return train_episode/train_collection_window records, or set curve_status='unsupported' with "
            "curve_status_reason for external trainers that cannot expose train-time curves yet."
        )

    for index, record in enumerate(episode_records):
        if not isinstance(record, dict):
            raise TypeError(f"episode_records[{index}] must be a dict.")
        missing = {"return", "length"} - set(record)
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"episode_records[{index}] is missing required key(s): {missing_text}.")

        kind = record_type(record)
        if kind not in RECORD_TYPES:
            raise ValueError(f"episode_records[{index}] has unknown record_type {kind!r}.")
        if kind == TRAIN_COLLECTION_WINDOW:
            if "episodes_in_window" not in record:
                raise ValueError(f"episode_records[{index}] is a collection window but is missing episodes_in_window.")
            if int(record.get("episodes_in_window") or 0) <= 0:
                raise ValueError(f"episode_records[{index}] collection window episodes_in_window must be positive.")
        if kind != TRAIN_EPISODE and not any(key in record for key in ("step", "elapsed_seconds", "episode")):
            raise ValueError(f"episode_records[{index}] must include step, elapsed_seconds, or episode for charting.")
