from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from autoresearch_gym.external.base import ArtifactSet, CommandSpec, RunBundle
from autoresearch_gym.runner.curves import make_train_episode_record


class FakeExternalBackend:
    """Small deterministic backend for external-runner contract tests."""

    def build_bundle(self, bundle: RunBundle) -> RunBundle:
        bundle.external_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": bundle.run_id,
            "tag": bundle.tag,
            "benchmark": {
                "name": bundle.benchmark.name,
                "train_episodes": bundle.train_episodes,
                "eval_episodes": bundle.eval_episodes,
                "primary_metric": bundle.benchmark.primary_metric,
            },
            "candidate": bundle.candidate_metadata,
            "eval_cases": bundle.eval_cases or [],
        }
        (bundle.external_dir / "bundle.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return bundle

    def _command(self, mode: str, bundle: RunBundle, checkpoint_path: Path | None = None) -> CommandSpec:
        argv = [
            sys.executable,
            "-m",
            "autoresearch_gym.external.fake_backend",
            "--mode",
            mode,
            "--bundle",
            str(bundle.external_dir / "bundle.json"),
            "--out-dir",
            str(bundle.external_dir),
        ]
        if checkpoint_path is not None:
            argv.extend(["--checkpoint", str(checkpoint_path)])
        return CommandSpec(argv=argv, cwd=Path.cwd(), label=mode)

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
        result_path = artifacts.root / "media_result.json"
        if not result_path.exists():
            return {"media_available": False}
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        frame_path = artifacts.root / "current_run_frame.jpg"
        if frame_path.exists():
            payload["live_frame_path"] = str(frame_path)
            payload.setdefault("visual", {})["live_frame_path"] = str(frame_path)
        return payload


def _write_train(bundle: dict[str, Any], out_dir: Path) -> None:
    episodes = max(1, int(bundle["benchmark"]["train_episodes"]))
    records = [
        make_train_episode_record(
            episode=idx + 1,
            return_value=float(idx + 1),
            length=10 + idx,
            success=idx == episodes - 1,
            step=(idx + 1) * 10,
            elapsed_seconds=0.01 * (idx + 1),
        )
        for idx in range(episodes)
    ]
    total_steps = int(sum(record["length"] for record in records))
    payload = {
        "episode_records": records,
        "total_steps": total_steps,
        "env_steps": total_steps,
        "episodes_completed": episodes,
        "episode_batches": episodes,
        "gradient_updates": 0,
        "last_metrics": {"fake_loss": 0.0, "gradient_updates": 0},
        "stop_reason": "episode_budget_exhausted",
    }
    (out_dir / "train_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "agent_checkpoint.pt").write_text("fake checkpoint\n", encoding="utf-8")


def _write_eval(bundle: dict[str, Any], out_dir: Path) -> None:
    episodes = int(bundle["benchmark"]["eval_episodes"])
    eval_cases = bundle.get("eval_cases") or []
    records = []
    for idx in range(episodes):
        case = eval_cases[idx] if idx < len(eval_cases) else {}
        records.append(
            {
                "episode": idx + 1,
                "seed": 7000 + idx,
                "return": float(10.0 - idx),
                "length": 12,
                "success": True,
                "case_label": str(case.get("name", f"case-{idx + 1:02d}")),
                "info_metrics": {"external_eval_case_index": idx},
            }
        )
    returns = [float(record["return"]) for record in records]
    payload = {
        "episodes": episodes,
        "avg_return": sum(returns) / len(returns) if returns else 0.0,
        "avg_length": 12.0 if records else 0.0,
        "success_rate": 1.0 if records else 0.0,
        "episode_records": records,
    }
    (out_dir / "eval_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_media(bundle: dict[str, Any], out_dir: Path) -> None:
    frame_path = out_dir / "current_run_frame.jpg"
    image = Image.new("RGB", (720, 480), (36, 44, 52))
    draw = ImageDraw.Draw(image)
    draw.rectangle((28, 40, 292, 180), outline=(118, 205, 180), width=4)
    draw.text((42, 88), f"{bundle['benchmark']['name']} fake external", fill=(245, 245, 245))
    image.save(frame_path, format="JPEG", quality=80)
    payload = {
        "media_available": True,
        "live_frame_path": str(frame_path),
        "visual": {"mode": "live_frame", "live_frame_path": str(frame_path), "sampled_status": "completed"},
    }
    (out_dir / "media_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval", "media"], required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args(argv)

    bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "train":
        _write_train(bundle, args.out_dir)
    elif args.mode == "eval":
        _write_eval(bundle, args.out_dir)
    else:
        _write_media(bundle, args.out_dir)


if __name__ == "__main__":
    main()
