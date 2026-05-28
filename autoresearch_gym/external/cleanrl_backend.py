from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path, PureWindowsPath
from types import ModuleType
from typing import Any, Callable

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np

from autoresearch_gym.external.base import ArtifactSet, CommandSpec, RunBundle
from autoresearch_gym.runner.experiment import BenchmarkSpec, TrainProbeSpec, candidate_metadata


class CleanRlExternalBackend:
    """Run a staged CleanRL-style seed directly on the execution target.

    This backend is the lower-level alternative to adapter-owned trainers such
    as MJLab. The mutable training recipe is the candidate Python file itself;
    the backend only stages it, calls the standard train/eval/media hooks, and
    normalizes artifacts back into the autoresearch runner contract.
    """

    def build_bundle(self, bundle: RunBundle) -> RunBundle:
        bundle.external_dir.mkdir(parents=True, exist_ok=True)
        staged_trainable = bundle.external_dir / "candidate_trainable.py"
        shutil.copy2(bundle.candidate_path, staged_trainable)
        benchmark_snapshot = bundle.external_dir / "benchmark_snapshot.json"
        shutil.copy2(bundle.benchmark_path, benchmark_snapshot)
        eval_cases_path = bundle.external_dir / "eval_cases.json"
        eval_cases_path.write_text(json.dumps({"cases": bundle.eval_cases or []}, indent=2), encoding="utf-8")
        payload = {
            "run_id": bundle.run_id,
            "tag": bundle.tag,
            "benchmark": {
                "name": bundle.benchmark.name,
                "env_id": bundle.benchmark.env_id,
                "env_kwargs": bundle.benchmark.env_kwargs,
                "train_episodes": bundle.train_episodes,
                "train_seconds": bundle.train_seconds,
                "eval_episodes": bundle.eval_episodes,
                "max_steps": bundle.max_steps,
                "reward_type": bundle.benchmark.reward_type,
                "render_mode": bundle.benchmark.render_mode,
                "primary_metric": bundle.benchmark.primary_metric,
                "primary_metric_mode": bundle.benchmark.primary_metric_mode,
                "train_seed": bundle.benchmark.train_seed,
                "eval_seed_start": bundle.benchmark.eval_seed_start,
                "device": bundle.benchmark.device,
            },
            "execution_backend": bundle.execution_backend,
            "candidate": bundle.candidate_metadata,
            "trainable_path": "candidate_trainable.py",
            "benchmark_snapshot_path": "benchmark_snapshot.json",
            "eval_cases_path": "eval_cases.json",
            "compact_status_file": str(bundle.compact_status_file) if bundle.compact_status_file is not None else None,
            "session_dir": str(bundle.session_dir) if bundle.session_dir is not None else None,
            "repo_root": str(Path.cwd()),
        }
        (bundle.external_dir / "bundle.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return bundle

    def training_command(self, bundle: RunBundle) -> CommandSpec:
        return self._command("train", bundle)

    def eval_command(self, bundle: RunBundle, checkpoint_path: Path) -> CommandSpec:
        return self._command("eval", bundle, checkpoint_path)

    def media_command(self, bundle: RunBundle, checkpoint_path: Path) -> CommandSpec | None:
        return self._command("media", bundle, checkpoint_path)

    def normalize_train(self, artifacts: ArtifactSet) -> dict[str, Any]:
        return json.loads((artifacts.root / "train_result.json").read_text(encoding="utf-8"))

    def normalize_eval(self, artifacts: ArtifactSet) -> dict[str, Any]:
        return json.loads((artifacts.root / "eval_result.json").read_text(encoding="utf-8"))

    def normalize_media(self, artifacts: ArtifactSet) -> dict[str, Any]:
        path = artifacts.root / "media_result.json"
        if not path.exists():
            return {"media_available": False}
        payload = json.loads(path.read_text(encoding="utf-8"))
        return _localize_media_payload(payload, artifacts.root)

    def _command(self, mode: str, bundle: RunBundle, checkpoint_path: Path | None = None) -> CommandSpec:
        argv = [
            sys.executable,
            "-m",
            "autoresearch_gym.external.cleanrl_backend",
            "--mode",
            mode,
            "--bundle",
            str(bundle.external_dir / "bundle.json"),
            "--out-dir",
            str(bundle.external_dir),
        ]
        if checkpoint_path is not None:
            argv.extend(["--checkpoint", str(checkpoint_path)])
        return CommandSpec(argv=argv, cwd=Path.cwd(), label=f"cleanrl-{mode}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _localize_artifact_path(value: Any, root: Path) -> str | None:
    if not value:
        return None
    raw = str(value)
    path = Path(raw)
    if path.exists():
        return str(path)
    normalized = raw.replace("\\", "/")
    marker = "/external/"
    if marker in normalized:
        candidate = root / normalized.split(marker, 1)[1]
        if candidate.exists():
            return str(candidate)
    candidate = root / PureWindowsPath(raw).name
    if candidate.exists():
        return str(candidate)
    return raw


def _localize_media_payload(payload: dict[str, Any], root: Path) -> dict[str, Any]:
    normalized = dict(payload)
    visual = dict(normalized.get("visual") or {})
    for key in ("live_frame_path", "trajectory_manifest_path", "trajectory_latest_frame_path"):
        localized = _localize_artifact_path(normalized.get(key) or visual.get(key), root)
        if localized is not None:
            normalized[key] = localized
            visual[key] = localized
    manifest_path = Path(str(normalized.get("trajectory_manifest_path", "")))
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        frames = [_localize_artifact_path(frame, root) or str(frame) for frame in manifest.get("frames", [])]
        manifest["frames"] = frames
        if manifest.get("gif_path"):
            manifest["gif_path"] = _localize_artifact_path(manifest.get("gif_path"), root) or str(manifest.get("gif_path"))
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if frames:
            normalized["trajectory_latest_frame_path"] = frames[-1]
            visual["trajectory_latest_frame_path"] = frames[-1]
    normalized["visual"] = visual
    return normalized


def _load_trainable(path: Path) -> ModuleType:
    module_name = f"autoresearch_external_candidate_{time.time_ns()}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load external candidate from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    for attr in ("get_candidate", "RewardRecipeWrapper", "train_agent", "save_agent_checkpoint"):
        if not hasattr(module, attr):
            raise AttributeError(f"external candidate {path} is missing {attr}")
    return module


def _benchmark_from_bundle(bundle: dict[str, Any], out_dir: Path) -> BenchmarkSpec:
    payload = bundle["benchmark"]
    return BenchmarkSpec(
        name=str(payload["name"]),
        env_id=str(payload["env_id"]),
        env_kwargs=dict(payload.get("env_kwargs") or {}),
        train_episodes=int(payload["train_episodes"]),
        train_seconds=float(payload["train_seconds"]) if payload.get("train_seconds") is not None else None,
        eval_episodes=int(payload["eval_episodes"]),
        max_steps=int(payload["max_steps"]),
        reward_type=payload.get("reward_type"),
        render_mode=payload.get("render_mode"),
        primary_metric=str(payload["primary_metric"]),
        primary_metric_mode=str(payload.get("primary_metric_mode", "maximize")),
        train_seed=int(payload.get("train_seed", 1)),
        eval_seed_start=int(payload.get("eval_seed_start", 9000)),
        device=str(payload.get("device", "cpu")),
        eval_case_bank=out_dir / str(bundle.get("eval_cases_path", "eval_cases.json")),
        execution_backend=dict(bundle.get("execution_backend") or {}),
        train_probe=TrainProbeSpec(enabled=False),
    )


def _make_env_factory(module: ModuleType, benchmark: BenchmarkSpec) -> Callable[..., gym.Env[Any, Any]]:
    def make_env(control_type: str | None = None, reward_recipe: str | None = None) -> gym.Env[Any, Any]:
        if hasattr(module, "make_external_env"):
            return module.make_external_env(benchmark, control_type=control_type, reward_recipe=reward_recipe)
        env_kwargs = dict(benchmark.env_kwargs)
        if control_type is not None:
            env_kwargs.setdefault("control_type", control_type)
        env = gym.make(benchmark.env_id, **env_kwargs)
        return module.RewardRecipeWrapper(env, reward_recipe)

    return make_env


def _agent_action(agent: Any, obs: Any, deterministic: bool) -> np.ndarray:
    if hasattr(agent, "act"):
        return np.asarray(agent.act(obs, deterministic=deterministic), dtype=np.float32)
    action = agent(obs)
    return np.asarray(action, dtype=np.float32)


def _load_agent(module: ModuleType, checkpoint: Path, benchmark: BenchmarkSpec) -> Any:
    if not hasattr(module, "load_agent_checkpoint"):
        raise AttributeError("external candidate must expose load_agent_checkpoint for eval/media")
    try:
        return module.load_agent_checkpoint(checkpoint, benchmark=benchmark)
    except TypeError:
        return module.load_agent_checkpoint(checkpoint)


def _evaluate_agent(module: ModuleType, agent: Any, benchmark: BenchmarkSpec, eval_cases: list[dict[str, Any]]) -> dict[str, Any]:
    if hasattr(module, "evaluate_agent"):
        return module.evaluate_agent(agent, benchmark, eval_cases=eval_cases)
    records = []
    make_env = _make_env_factory(module, benchmark)
    for idx in range(benchmark.eval_episodes):
        env = make_env()
        case = eval_cases[idx] if idx < len(eval_cases) else {}
        reset_options = {"fixed_case": case} if case else None
        obs, _ = env.reset(seed=benchmark.eval_seed_start + idx, options=reset_options)
        total_return = 0.0
        success = False
        info: dict[str, Any] = {}
        length = 0
        for step in range(benchmark.max_steps):
            obs, reward, terminated, truncated, info = env.step(_agent_action(agent, obs, deterministic=True))
            total_return += float(reward)
            length = step + 1
            success = bool(info.get("is_success", success))
            if terminated or truncated:
                break
        env.close()
        records.append(
            {
                "episode": idx + 1,
                "seed": benchmark.eval_seed_start + idx,
                "return": float(total_return),
                "length": int(length),
                "success": success,
                "case_label": str(case.get("name", f"case-{idx + 1:02d}")),
                "info_metrics": {key: float(value) for key, value in info.items() if isinstance(value, (int, float, np.floating))},
            }
        )
    summary = {
        "episodes": len(records),
        "avg_return": float(np.mean([record["return"] for record in records])) if records else 0.0,
        "avg_length": float(np.mean([record["length"] for record in records])) if records else 0.0,
        "success_rate": float(np.mean([1.0 if record["success"] else 0.0 for record in records])) if records else 0.0,
        "episode_records": records,
    }
    if records:
        metric_keys = sorted({key for record in records for key in record.get("info_metrics", {})})
        for key in metric_keys:
            values = [record["info_metrics"][key] for record in records if key in record.get("info_metrics", {})]
            if values:
                summary[f"avg_{key}"] = float(np.mean(values))
    return summary


def _render_media(module: ModuleType, agent: Any, benchmark: BenchmarkSpec, out_dir: Path) -> dict[str, Any]:
    if hasattr(module, "render_policy"):
        return module.render_policy(agent, benchmark, out_dir=out_dir)
    frame_dir = out_dir / "trajectories" / "sample_000001"
    frame_dir.mkdir(parents=True, exist_ok=True)
    env = _make_env_factory(module, benchmark)()
    obs, _ = env.reset(seed=benchmark.eval_seed_start)
    frames = []
    for idx in range(min(24, benchmark.max_steps)):
        frame = env.render()
        if frame is not None:
            frame_path = frame_dir / f"frame_{idx:04d}.jpg"
            imageio.imwrite(frame_path, np.asarray(frame, dtype=np.uint8))
            frames.append(str(frame_path))
        obs, _, terminated, truncated, _ = env.step(_agent_action(agent, obs, deterministic=True))
        if terminated or truncated:
            break
    env.close()
    manifest = {"status": "completed", "sample_index": 1, "frames": frames, "frame_count": len(frames)}
    _write_json(frame_dir / "manifest.json", manifest)
    latest = frames[-1] if frames else None
    if latest:
        shutil.copy2(latest, out_dir / "current_run_frame.jpg")
    return {
        "media_available": bool(frames),
        "live_frame_path": str(out_dir / "current_run_frame.jpg") if latest else None,
        "trajectory_manifest_path": str(frame_dir / "manifest.json"),
        "trajectory_latest_frame_path": latest,
        "visual": {
            "live_frame_path": str(out_dir / "current_run_frame.jpg") if latest else None,
            "trajectory_manifest_path": str(frame_dir / "manifest.json"),
            "trajectory_latest_frame_path": latest,
            "sampled_status": "completed",
            "latest_sample_index": 1,
        },
    }


def _load_eval_cases(out_dir: Path, bundle: dict[str, Any]) -> list[dict[str, Any]]:
    path = out_dir / str(bundle.get("eval_cases_path", "eval_cases.json"))
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("cases", []))


def _train(bundle: dict[str, Any], out_dir: Path) -> None:
    module = _load_trainable(out_dir / str(bundle["trainable_path"]))
    benchmark = _benchmark_from_bundle(bundle, out_dir)
    candidate = module.get_candidate()
    agent, train_summary = module.train_agent(
        benchmark,
        _make_env_factory(module, benchmark),
        candidate,
        benchmark.device,
        init_checkpoint=None,
        live_callback=None,
    )
    checkpoint = out_dir / "agent_checkpoint.pt"
    try:
        module.save_agent_checkpoint(agent, checkpoint, metadata={"backend": "cleanrl_external"})
    except TypeError:
        module.save_agent_checkpoint(agent, checkpoint)
    train_summary.setdefault("external_backend_scaffold", False)
    train_summary.setdefault("backend", "cleanrl_external")
    _write_json(out_dir / "candidate_metadata.json", candidate_metadata(candidate))
    _write_json(out_dir / "train_result.json", train_summary)


def _eval(bundle: dict[str, Any], out_dir: Path, checkpoint: Path) -> None:
    module = _load_trainable(out_dir / str(bundle["trainable_path"]))
    benchmark = _benchmark_from_bundle(bundle, out_dir)
    agent = _load_agent(module, checkpoint, benchmark)
    _write_json(out_dir / "eval_result.json", _evaluate_agent(module, agent, benchmark, _load_eval_cases(out_dir, bundle)))


def _media(bundle: dict[str, Any], out_dir: Path, checkpoint: Path) -> None:
    module = _load_trainable(out_dir / str(bundle["trainable_path"]))
    benchmark = _benchmark_from_bundle(bundle, out_dir)
    agent = _load_agent(module, checkpoint, benchmark)
    _write_json(out_dir / "media_result.json", _render_media(module, agent, benchmark, out_dir))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval", "media"], required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()

    bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "train":
        _train(bundle, args.out_dir)
    elif args.mode == "eval":
        _eval(bundle, args.out_dir, args.checkpoint or args.out_dir / "agent_checkpoint.pt")
    else:
        _media(bundle, args.out_dir, args.checkpoint or args.out_dir / "agent_checkpoint.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
