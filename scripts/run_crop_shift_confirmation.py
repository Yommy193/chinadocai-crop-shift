"""Resumable multi-seed confirmation for the frozen crop-shift protocol.

The confirmation is additive and validation-only.  It reuses the already
completed pilot checkpoints when their hashes are available, trains only the
missing arm/seed combinations below a separate confirmation directory, and
exports an explicit 5-arm x 3-seed x 2-detector target matrix.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "artifacts" / "experiments" / "crop_shift_alignment_20261009"
CONFIRMATION = EXPERIMENT / "confirmation_20261010"
SEEDS = (20261009, 20261010, 20261011)
ARMS = (
    "gold_step",
    "anchor_pred_step",
    "frcnn_pred_step",
    "anchor_mix075_step",
    "frcnn_mix075_step",
)
DETECTORS = {
    "anchor": "artifacts/experiments/plate_detector_center_20260907/seed_20260931/detector_best.pt",
    "frcnn": "artifacts/experiments/plate_detector_strong_end_to_end_20260918/runs/seed_20260981/detector_best.pt",
}


def _record(path: Path, **payload: Any) -> None:
    record = {"created_at_utc": datetime.now(timezone.utc).isoformat(), **payload}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(record, ensure_ascii=False), flush=True)


def _rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def _pilot_checkpoint(arm: str, seed: int) -> Path | None:
    if seed != 20261009:
        return None
    path = EXPERIMENT / "runs" / f"{arm}_seed{seed}" / "cnn_best.pt"
    return path if path.is_file() else None


def _checkpoint(arm: str, seed: int) -> Path:
    pilot = _pilot_checkpoint(arm, seed)
    if pilot is not None:
        return pilot
    return CONFIRMATION / "runs" / f"{arm}_seed{seed}" / "cnn_best.pt"


def _fresh_log_path(prefix: str) -> Path:
    """Return a new additive log path, preserving logs from aborted retries."""
    log_dir = CONFIRMATION / "logs"
    candidate = log_dir / f"{prefix}.log"
    if not candidate.exists():
        return candidate
    index = 1
    while True:
        candidate = log_dir / f"{prefix}.retry{index:02d}.log"
        if not candidate.exists():
            return candidate
        index += 1


def _train_one(arm: str, seed: int, progress: Path, workers: int) -> Path:
    existing = _checkpoint(arm, seed)
    metrics = existing.parent / "cnn_metrics.json"
    if existing.is_file() and metrics.is_file():
        _record(
            progress,
            step=f"train:{arm}:{seed}",
            status="reused_existing",
            checkpoint=str(existing),
        )
        return existing
    output = CONFIRMATION / "runs" / f"{arm}_seed{seed}"
    command = [
        sys.executable,
        "scripts/train_crop_shift_cnn.py",
        "--processed-dir",
        _rel(EXPERIMENT / "manifests" / arm),
        "--class-counts",
        _rel(EXPERIMENT / "frozen_class_counts.json"),
        "--output-dir",
        _rel(output),
        "--seed",
        str(seed),
        "--epochs",
        "8",
        "--num-workers",
        str(workers),
    ]
    log = _fresh_log_path(f"train_{arm}_seed{seed}")
    _record(progress, step=f"train:{arm}:{seed}", status="started", command=command)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8", newline="\n") as handle:
        completed = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False)
    _record(
        progress,
        step=f"train:{arm}:{seed}",
        status="completed",
        exit_code=completed.returncode,
        log=str(log),
    )
    if completed.returncode or not existing.is_file() or not metrics.is_file():
        raise RuntimeError(f"confirmation training failed: {arm}/{seed}; see {log}")
    return existing


def _export_one(arm: str, seed: int, detector: str, progress: Path, batch_size: int) -> dict[str, Any]:
    if detector not in DETECTORS:
        raise ValueError(detector)
    checkpoint = _checkpoint(arm, seed)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    cell = CONFIRMATION / "target_rows" / f"{arm}_seed{seed}" / detector
    rows = cell / "targets.jsonl"
    summary = cell / "summary.json"
    if rows.exists() != summary.exists():
        raise FileExistsError(f"partial confirmation target cell: {cell}")
    if rows.is_file() and summary.is_file():
        _record(progress, step=f"cell:{arm}:{seed}:{detector}", status="skipped_existing", rows=str(rows))
        return {"arm": arm, "recognition_seed": seed, "eval_detector": detector, "rows": str(rows), "summary": str(summary)}
    command = [
        sys.executable,
        "scripts/evaluate_crop_shift_targets.py",
        "--detector-checkpoint",
        DETECTORS[detector],
        "--vision-checkpoint",
        _rel(checkpoint),
        "--detector-label",
        detector,
        "--class-counts",
        _rel(EXPERIMENT / "frozen_class_counts.json"),
        "--split",
        "val",
        "--source",
        "both",
        "--max-plates",
        "0",
        "--subset-seed",
        "20260907",
        "--matching-iou",
        "0.50",
        "--score-threshold",
        "0.30",
        "--detector-top-k",
        "100",
        "--detector-nms-threshold",
        "0.45",
        "--recognition-batch-size",
        str(batch_size),
        "--rows-output",
        _rel(rows),
        "--summary-output",
        _rel(summary),
    ]
    log = _fresh_log_path(f"targets_{arm}_seed{seed}_{detector}")
    _record(progress, step=f"cell:{arm}:{seed}:{detector}", status="started", command=command)
    cell.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8", newline="\n") as handle:
        completed = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False)
    _record(
        progress,
        step=f"cell:{arm}:{seed}:{detector}",
        status="completed",
        exit_code=completed.returncode,
        log=str(log),
    )
    if completed.returncode or not rows.is_file() or not summary.is_file():
        raise RuntimeError(f"confirmation target export failed: {arm}/{seed}/{detector}; see {log}")
    return {"arm": arm, "recognition_seed": seed, "eval_detector": detector, "rows": str(rows), "summary": str(summary)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen multi-seed crop-shift confirmation")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--train-jobs", type=int, default=1)
    parser.add_argument("--train-workers", type=int, default=6)
    parser.add_argument("--recognition-batch-size", type=int, default=128)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.seeds)
    arms = tuple(str(value) for value in args.arms)
    if not seeds or not arms or args.train_jobs <= 0 or args.train_workers < 0:
        raise ValueError("seeds/arms must be non-empty and job counts must be valid")
    if any(seed not in SEEDS for seed in seeds):
        raise ValueError(f"confirmation seeds are fixed to {SEEDS}")
    if any(arm not in ARMS for arm in arms):
        raise ValueError(f"confirmation arms are fixed to {ARMS}")
    progress = CONFIRMATION / "confirmation_progress.jsonl"
    tasks = [(arm, seed) for seed in seeds for arm in arms]
    if not args.skip_training:
        if args.train_jobs == 1:
            for arm, seed in tasks:
                _train_one(arm, seed, progress, int(args.train_workers))
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=int(args.train_jobs)) as pool:
                futures = [pool.submit(_train_one, arm, seed, progress, int(args.train_workers)) for arm, seed in tasks]
                for future in futures:
                    future.result()
    if args.skip_export:
        return
    cells: list[dict[str, Any]] = []
    for arm, seed in tasks:
        for detector in DETECTORS:
            cells.append(_export_one(arm, seed, detector, progress, int(args.recognition_batch_size)))
    manifest = CONFIRMATION / "confirmation_matrix_cells.json"
    if manifest.exists():
        raise FileExistsError(f"refusing to overwrite confirmation manifest: {manifest}")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(cells, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _record(progress, step="confirmation_manifest", status="completed", output=str(manifest), cells=len(cells))


if __name__ == "__main__":
    main()
