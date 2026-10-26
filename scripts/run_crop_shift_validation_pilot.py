"""Resumable sequential runner for crop-shift validation-only pilots."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARMS = (
    "gold_step",
    "frcnn_pred_step",
    "frcnn_mix025_step",
    "frcnn_mix050_step",
    "frcnn_mix075_step",
    "frcnn_curriculum_step",
)
DETECTORS = {
    "anchor": {
        "script": "scripts/evaluate_plate_to_order_smoke.py",
        "checkpoint": "artifacts/experiments/plate_detector_center_20260907/seed_20260931/detector_best.pt",
    },
    "frcnn": {
        "script": "scripts/evaluate_plate_to_order_smoke_strong.py",
        "checkpoint": "artifacts/experiments/plate_detector_strong_end_to_end_20260918/runs/seed_20260981/detector_best.pt",
    },
}


def append_progress(path: Path, **payload: object) -> None:
    record = {"created_at_utc": datetime.now(timezone.utc).isoformat(), **payload}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(record, ensure_ascii=False), flush=True)


def invoke(command: list[str], log_path: Path, progress: Path, name: str) -> None:
    append_progress(progress, step=name, status="started", command=command)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="\n") as handle:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    append_progress(progress, step=name, status="completed", exit_code=completed.returncode, log=str(log_path))
    if completed.returncode:
        raise RuntimeError(f"{name} failed with exit code {completed.returncode}; see {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run validation-only crop-shift pilot arms sequentially")
    parser.add_argument("--root", default="artifacts/experiments/crop_shift_alignment_20261009")
    parser.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS))
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=6)
    args = parser.parse_args()
    experiment = (ROOT / args.root).resolve()
    validation = json.loads((experiment / "validation_report.json").read_text(encoding="utf-8"))
    if validation.get("status") != "passed" or int(validation.get("test_rows_across_all_manifests", -1)) != 0:
        raise ValueError("crop-shift data validation has not passed")
    progress = experiment / "pilot_progress.jsonl"
    python = sys.executable
    completed_arms: list[str] = []
    for arm in args.arms:
        manifest = experiment / "manifests" / arm
        if not manifest.is_dir():
            raise FileNotFoundError(manifest)
        run = experiment / "runs" / f"{arm}_seed{args.seed}"
        metrics = run / "cnn_metrics.json"
        if not metrics.is_file():
            invoke(
                [
                    python,
                    "scripts/train_crop_shift_cnn.py",
                    "--processed-dir",
                    str(manifest.relative_to(ROOT)),
                    "--class-counts",
                    str((experiment / "frozen_class_counts.json").relative_to(ROOT)),
                    "--output-dir",
                    str(run.relative_to(ROOT)),
                    "--seed",
                    str(args.seed),
                    "--epochs",
                    str(args.epochs),
                    "--num-workers",
                    str(args.num_workers),
                ],
                experiment / "logs" / f"train_{arm}_seed{args.seed}.log",
                progress,
                f"train:{arm}",
            )
        else:
            append_progress(progress, step=f"train:{arm}", status="skipped_existing", output=str(metrics))
        payload = json.loads(metrics.read_text(encoding="utf-8"))
        if payload.get("test") is not None or payload.get("training", {}).get("test_access", "").startswith("prohibited") is False:
            raise ValueError(f"training run is not test-locked: {run}")

        for detector, detector_spec in DETECTORS.items():
            evaluation = experiment / "validation" / arm / detector
            summary = evaluation / "summary.json"
            if summary.is_file():
                append_progress(progress, step=f"validate:{arm}:{detector}", status="skipped_existing", output=str(summary))
                continue
            invoke(
                [
                    python,
                    str(detector_spec["script"]),
                    "--detector-checkpoint",
                    str(detector_spec["checkpoint"]),
                    "--vision-checkpoint",
                    str((run / "cnn_best.pt").relative_to(ROOT)),
                    "--source",
                    "both",
                    "--max-plates",
                    "0",
                    "--matching-iou",
                    "0.50",
                    "--score-threshold",
                    "0.30",
                    "--detector-top-k",
                    "100",
                    "--detector-nms-threshold",
                    "0.45",
                    "--decoder-provenance",
                    "crop_shift_validation_frozen_0p30_top100_nms0p45",
                    "--output",
                    str(summary.relative_to(ROOT)),
                    "--per-plate-output",
                    str((evaluation / "per_plate.json").relative_to(ROOT)),
                ],
                experiment / "logs" / f"validate_{arm}_{detector}.log",
                progress,
                f"validate:{arm}:{detector}",
            )
        completed_arms.append(arm)

    invoke(
        [
            python,
            "scripts/aggregate_crop_shift_validation.py",
            "--root",
            str(experiment.relative_to(ROOT)),
            "--arms",
            *completed_arms,
        ],
        experiment / "logs" / "aggregate_validation.log",
        progress,
        "aggregate_validation",
    )


if __name__ == "__main__":
    main()

