from __future__ import annotations

import importlib
import json
import time
from pathlib import Path
from typing import Any

from autoresearch_gym.external.base import ArtifactSet, ExternalBackend, RunBundle
from autoresearch_gym.external.targets import load_target_config, make_target
from autoresearch_gym.runner.experiment import (
    BenchmarkSpec,
    candidate_metadata,
    comparable_score,
    json_default,
    load_eval_cases,
    normalize_run_tag,
    normalize_train_summary_curve,
    public_summary,
    resolve_metric,
    validate_train_curve_contract,
)


def _load_backend(spec: dict[str, Any]) -> ExternalBackend:
    adapter = spec.get("adapter")
    if not adapter:
        raise ValueError("external benchmark requires execution_backend.adapter")
    module_name, _, attr = str(adapter).partition(":")
    if not module_name or not attr:
        raise ValueError(f"invalid backend adapter: {adapter}")
    module = importlib.import_module(module_name)
    backend_cls = getattr(module, attr)
    return backend_cls()


def _require_success(label: str, result: Any) -> None:
    if not result.ok:
        raise RuntimeError(
            f"external {label} command failed with exit code {result.returncode}\n"
            f"stdout:\n{result.stdout[-2000:]}\n"
            f"stderr:\n{result.stderr[-2000:]}"
        )


def _repo_relative_or_absolute(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _normalize_visual_paths_for_dashboard(media_summary: dict[str, Any], local_run_dir: Path) -> dict[str, Any]:
    normalized = dict(media_summary)
    visual = dict(normalized.get("visual") or {})
    for key in ("live_frame_path", "trajectory_manifest_path", "trajectory_latest_frame_path"):
        value = normalized.get(key) or visual.get(key)
        if not value:
            continue
        path = Path(str(value))
        if not path.exists() and not path.is_absolute():
            path = local_run_dir / path
        if path.exists():
            dashboard_path = _repo_relative_or_absolute(path)
            normalized[key] = dashboard_path
            visual[key] = dashboard_path
    normalized["visual"] = visual
    return normalized


def _write_external_live_status(
    session_dir: Path | None,
    run_id: str,
    tag: str,
    status: str,
    summary: dict[str, Any],
    train_summary: dict[str, Any],
) -> None:
    if session_dir is None:
        return
    live_dir = session_dir / "live"
    live_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "run": {
            "run_id": run_id,
            "tag": tag,
            "status": status,
            "started_at": time.time(),
            "updated_at": time.time(),
            "train_episodes": summary.get("benchmark", {}).get("train_episodes"),
            "train_seconds": summary.get("benchmark", {}).get("train_seconds"),
            "budget_mode": summary.get("benchmark", {}).get("budget_mode"),
            "eval_episodes": summary.get("benchmark", {}).get("eval_episodes"),
            "max_steps": summary.get("benchmark", {}).get("max_steps"),
            "render_mode": summary.get("benchmark", {}).get("render_mode", "rgb_array"),
            "candidate": summary.get("candidate", {}),
            "frame_path": summary.get("media", {}).get("live_frame_path"),
            "trajectory_manifest_path": summary.get("media", {}).get("trajectory_manifest_path"),
            "trajectory_latest_frame_path": summary.get("media", {}).get("trajectory_latest_frame_path"),
            "visual": summary.get("media", {}).get("visual", {}),
        },
        "current": {
            "status": status,
            "step": summary.get("train", {}).get("total_steps", 0),
            "env_steps": summary.get("train", {}).get("env_steps", 0),
            "episodes_complete": summary.get("train", {}).get("episodes_completed", 0),
            "completed_episodes": summary.get("train", {}).get("episodes_completed", 0),
            "episode_batches": summary.get("train", {}).get("episode_batches", 0),
            "avg_return": summary.get("train", {}).get("avg_return", 0.0),
            "success_rate": summary.get("train", {}).get("success_rate", 0.0),
            "info_metrics": train_summary.get("last_metrics", {}),
        },
        "episodes": train_summary.get("episode_records", []),
        "latest_losses": train_summary.get("last_metrics", {}),
        "visual": summary.get("media", {}).get("visual", {}),
    }
    (live_dir / "current_run_metrics.json").write_text(json.dumps(metrics, indent=2, default=json_default), encoding="utf-8")


def run_external_experiment(
    *,
    benchmark: BenchmarkSpec,
    benchmark_path: Path,
    candidate_path: Path,
    candidate: Any,
    tag: str,
    out_dir: Path,
    results_path: Path | None,
    session_dir: Path | None,
    evolution_metadata: dict[str, Any] | None,
    compact_status_file: Path | None,
    execution_target_override: str | None,
    target_config_path: Path | None,
) -> dict[str, Any]:
    if benchmark.execution_backend is None:
        raise ValueError("run_external_experiment requires benchmark.execution_backend")

    tag = normalize_run_tag(tag)
    run_id = time.strftime("%Y%m%d-%H%M%S") + f"-{tag}"
    run_dir = out_dir / run_id
    external_dir = run_dir / "external"
    run_dir.mkdir(parents=True, exist_ok=True)
    external_dir.mkdir(parents=True, exist_ok=True)

    backend_spec = dict(benchmark.execution_backend)
    default_target = str(backend_spec.get("execution_target", "local"))
    target_name = execution_target_override or default_target
    target_config = load_target_config(target_name, config_path=target_config_path, repo_root=Path.cwd())
    target = make_target(target_config)
    backend = _load_backend(backend_spec)
    eval_cases = load_eval_cases(benchmark)

    bundle = RunBundle(
        run_id=run_id,
        tag=tag,
        benchmark_path=benchmark_path,
        candidate_path=candidate_path,
        local_run_dir=run_dir,
        external_dir=external_dir,
        benchmark=benchmark,
        candidate=candidate,
        candidate_metadata=candidate_metadata(candidate),
        execution_backend=backend_spec,
        eval_cases=eval_cases,
        train_episodes=benchmark.train_episodes,
        train_seconds=benchmark.train_seconds,
        eval_episodes=benchmark.eval_episodes,
        max_steps=benchmark.max_steps,
        compact_status_file=compact_status_file,
        session_dir=session_dir,
        target_name=target_config.name,
    )
    bundle = backend.build_bundle(bundle)
    target.stage(bundle)
    preflight = target.preflight(bundle)
    if not preflight.ok:
        raise RuntimeError(f"execution target preflight failed: {json.dumps(preflight.checks, indent=2)}")

    train_result = target.run(backend.training_command(bundle), bundle)
    _require_success("train", train_result)
    target.sync_live(bundle)
    artifacts = target.fetch_artifacts(bundle)
    train_summary = backend.normalize_train(artifacts)
    validate_train_curve_contract(train_summary)
    normalize_train_summary_curve(train_summary)

    checkpoint_path = artifacts.root / "agent_checkpoint.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"external backend did not produce checkpoint: {checkpoint_path}")

    eval_result = target.run(backend.eval_command(bundle, checkpoint_path), bundle)
    _require_success("eval", eval_result)
    target.sync_live(bundle)
    artifacts = target.fetch_artifacts(bundle)
    eval_summary = backend.normalize_eval(artifacts)

    media_summary: dict[str, Any] = {"media_available": False}
    media_command = backend.media_command(bundle, checkpoint_path)
    if media_command is not None:
        media_result = target.run(media_command, bundle)
        _require_success("media", media_result)
        target.sync_live(bundle)
        artifacts = target.fetch_artifacts(bundle)
        media_summary = _normalize_visual_paths_for_dashboard(backend.normalize_media(artifacts), run_dir)

    summary = {
        "run_id": run_id,
        "tag": tag,
        "session": {
            "session_dir": str(session_dir) if session_dir is not None else None,
            "results_path": str(results_path) if results_path is not None else None,
            "log_path": str(session_dir / "outer_loop_log.md") if session_dir is not None else None,
        },
        "lineage": {
            "mode": "from_scratch",
            "init_checkpoint": None,
            "parent_run_id": None,
            "parent_tag": None,
        },
        "run_options": {
            "headless_env": False,
            "headless_env_requested": False,
            "headless_env_effective": False,
            "headless_env_reason": None,
            "headless_env_message": None,
            "compact_status": compact_status_file is not None,
            "compact_status_stderr": False,
            "compact_status_file": str(compact_status_file) if compact_status_file is not None else None,
        },
        "execution": {
            "backend": backend_spec.get("name") or backend_spec.get("adapter"),
            "backend_kind": backend_spec.get("kind", "external"),
            **target_config.redacted_summary(),
            "preflight": {
                "ok": preflight.ok,
                "checks": preflight.checks,
            },
        },
        "benchmark": {
            "name": benchmark.name,
            "env_id": benchmark.env_id,
            "env_kwargs": benchmark.env_kwargs,
            "train_episodes": benchmark.train_episodes,
            "train_seconds": benchmark.train_seconds,
            "budget_mode": "time" if benchmark.train_seconds is not None else "episodes",
            "eval_episodes": benchmark.eval_episodes,
            "max_steps": benchmark.max_steps,
            "render_mode": benchmark.render_mode,
            "primary_metric": benchmark.primary_metric,
            "primary_metric_mode": benchmark.primary_metric_mode,
            "device": benchmark.device,
            "eval_case_bank": str(benchmark.eval_case_bank) if benchmark.eval_case_bank is not None else None,
            "train_probe": {},
            "execution_backend": backend_spec,
        },
        "candidate": candidate_metadata(candidate),
        "train": public_summary(train_summary),
        "eval": public_summary(eval_summary),
        "media": media_summary,
        "system_utilization": {
            "device": "external",
            "target_kind": target_config.kind,
            "flags": {
                "external_backend": True,
                "gradient_updates_reported": train_summary.get("gradient_updates") is not None,
            },
        },
        "artifacts": {
            "checkpoint_path": str(checkpoint_path),
        },
        "evolution": evolution_metadata or {},
    }
    metric_value = resolve_metric(summary, benchmark.primary_metric)
    summary["objective"] = {
        "metric": benchmark.primary_metric,
        "mode": benchmark.primary_metric_mode,
        "value": metric_value,
    }
    summary["score"] = comparable_score(metric_value, benchmark.primary_metric_mode)

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=json_default), encoding="utf-8")
    (run_dir / "train_episodes.json").write_text(
        json.dumps(train_summary["episode_records"], indent=2, default=json_default),
        encoding="utf-8",
    )
    (run_dir / "eval_episodes.json").write_text(
        json.dumps(eval_summary["episode_records"], indent=2, default=json_default),
        encoding="utf-8",
    )
    (run_dir / "candidate_snapshot.json").write_text(
        json.dumps({"candidate": candidate_metadata(candidate)}, indent=2, default=json_default),
        encoding="utf-8",
    )
    (run_dir / "benchmark_snapshot.json").write_text(benchmark_path.read_text(encoding="utf-8"), encoding="utf-8")
    (run_dir / "trainable_snapshot.py").write_text(candidate_path.read_text(encoding="utf-8"), encoding="utf-8")
    _write_external_live_status(session_dir, run_id, tag, "finished", summary, train_summary)

    if results_path is not None:
        from autoresearch_gym.runner.experiment import append_result

        append_result(results_path, summary)
    return summary
