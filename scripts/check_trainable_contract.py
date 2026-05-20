from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric, got {value!r}")
    return float(value)


def _train_payload(payload: dict[str, Any]) -> dict[str, Any]:
    train = payload.get("train", payload)
    if not isinstance(train, dict):
        raise ValueError("summary must contain a train object or be a train summary")
    return train


def _record_type(record: dict[str, Any]) -> str:
    value = str(record.get("record_type") or "")
    return value or "train_episode"


def _is_policy_probe(record: dict[str, Any]) -> bool:
    return _record_type(record) == "policy_probe"


def _episodes_in_record(record: dict[str, Any]) -> int:
    if _is_policy_probe(record):
        return 0
    count = record.get("episodes_in_window")
    if count is None:
        return 1
    return max(0, int(count))


def validate_summary(
    payload: dict[str, Any],
    *,
    require_gradient_updates: bool,
) -> list[str]:
    train = _train_payload(payload)
    errors: list[str] = []

    for key in ("total_steps", "episodes_completed"):
        try:
            _number(train.get(key), f"train.{key}")
        except ValueError as exc:
            errors.append(str(exc))

    if "env_steps" not in train:
        errors.append("train.env_steps is required for dashboard counters")
    else:
        try:
            env_steps = int(_number(train.get("env_steps"), "train.env_steps"))
            total_steps = int(_number(train.get("total_steps"), "train.total_steps"))
            if env_steps != total_steps:
                errors.append("train.env_steps must match train.total_steps")
        except ValueError as exc:
            errors.append(str(exc))

    completed = train.get("completed_episodes")
    episodes_completed = train.get("episodes_completed")
    if completed is None:
        errors.append("train.completed_episodes is required for dashboard counters")
    else:
        try:
            if int(_number(completed, "train.completed_episodes")) != int(
                _number(episodes_completed, "train.episodes_completed")
            ):
                errors.append("train.completed_episodes must match train.episodes_completed")
        except ValueError as exc:
            errors.append(str(exc))

    if "episode_batches" not in train:
        errors.append("train.episode_batches is required for dashboard counters")
    else:
        try:
            _number(train.get("episode_batches"), "train.episode_batches")
        except ValueError as exc:
            errors.append(str(exc))

    gradient_updates = train.get("gradient_updates")
    if require_gradient_updates and gradient_updates is None:
        errors.append("train.gradient_updates is required when checking update-cadence logging")
    if gradient_updates is not None:
        try:
            if _number(gradient_updates, "train.gradient_updates") < 0:
                errors.append("train.gradient_updates must be non-negative")
        except ValueError as exc:
            errors.append(str(exc))

    last_metrics = train.get("last_metrics")
    if isinstance(last_metrics, dict) and last_metrics:
        metric_updates = last_metrics.get("gradient_updates")
        if require_gradient_updates and metric_updates is None:
            errors.append("train.last_metrics.gradient_updates is required once update metrics are emitted")
        if metric_updates is not None:
            try:
                if _number(metric_updates, "train.last_metrics.gradient_updates") < 0:
                    errors.append("train.last_metrics.gradient_updates must be non-negative")
            except ValueError as exc:
                errors.append(str(exc))

    return errors


def validate_records(records: list[Any]) -> list[str]:
    errors: list[str] = []
    completed_so_far = 0
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            errors.append(f"train_episodes[{index}] must be an object")
            continue
        for key in ("return", "length"):
            if key not in record:
                errors.append(f"train_episodes[{index}] missing {key}")
        kind = _record_type(record)
        if kind == "train_collection_window" and int(record.get("episodes_in_window") or 0) <= 0:
            errors.append(f"train_episodes[{index}] collection window needs positive episodes_in_window")
        if kind == "policy_probe":
            probe_episode = record.get("episode")
            if probe_episode is None:
                errors.append(f"train_episodes[{index}] policy_probe missing episode axis coordinate")
            else:
                try:
                    axis_episode = int(_number(probe_episode, f"train_episodes[{index}].episode"))
                    if axis_episode > completed_so_far:
                        errors.append(
                            f"train_episodes[{index}] policy_probe episode axis {axis_episode} "
                            f"exceeds completed collection rollouts {completed_so_far}"
                        )
                except ValueError as exc:
                    errors.append(str(exc))
        else:
            completed_so_far += _episodes_in_record(record)
    return errors


def infer_train_episodes_path(summary_path: Path) -> Path | None:
    candidate = summary_path.parent / "train_episodes.json"
    return candidate if candidate.exists() else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate trainable summary/logging fields used by live dashboards.")
    parser.add_argument("summary_json", type=Path, help="Run summary.json, or a raw train summary JSON.")
    parser.add_argument("--train-episodes-json", type=Path, default=None)
    parser.add_argument("--require-gradient-updates", action="store_true")
    args = parser.parse_args()

    summary = _read_json(args.summary_json)
    if not isinstance(summary, dict):
        raise SystemExit("summary_json must contain a JSON object")

    errors = validate_summary(summary, require_gradient_updates=args.require_gradient_updates)
    records_path = args.train_episodes_json or infer_train_episodes_path(args.summary_json)
    if records_path is not None:
        records = _read_json(records_path)
        if not isinstance(records, list):
            errors.append("train_episodes_json must contain a JSON array")
        else:
            errors.extend(validate_records(records))

    if errors:
        print(json.dumps({"ok": False, "errors": errors}, indent=2), file=sys.stderr)
        return 1
    print(json.dumps({"ok": True}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
