"""Resumable validation-only per-target export for the crop-shift matrix.

This helper is additive: it reads frozen checkpoints and validation inputs and
writes target rows, summaries, and logs below the crop-shift experiment root.
It never accepts a test split and refuses to overwrite a partial cell.
"""

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
    "anchor": "artifacts/experiments/plate_detector_center_20260907/seed_20260931/detector_best.pt",
    "frcnn": "artifacts/experiments/plate_detector_strong_end_to_end_20260918/runs/seed_20260981/detector_best.pt",
}


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def _record(path: Path, **payload: object) -> None:
    record = {"created_at_utc": datetime.now(timezone.utc).isoformat(), **payload}
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(record, ensure_ascii=False), flush=True)


def _run_cell(
    *,
    experiment: Path,
    arm: str,
    seed: int,
    detector: str,
    batch_size: int,
    cpu: bool,
    progress: Path,
) -> dict[str, str | int]:
    if detector not in DETECTORS:
        raise ValueError(f"unsupported detector: {detector}")
    checkpoint = experiment / "runs" / f"{arm}_seed{seed}" / "cnn_best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing recognition checkpoint: {checkpoint}")
    cell_dir = experiment / "target_rows" / f"{arm}_seed{seed}" / detector
    rows = cell_dir / "targets.jsonl"
    summary = cell_dir / "summary.json"
    if rows.exists() != summary.exists():
        raise FileExistsError(f"partial target cell exists; inspect and repair manually: {cell_dir}")
    if rows.is_file() and summary.is_file():
        _record(progress, step=f"cell:{arm}:{seed}:{detector}", status="skipped_existing", rows=str(rows))
        return {
            "arm": arm,
            "recognition_seed": int(seed),
            "eval_detector": detector,
            "rows": str(rows),
            "summary": str(summary),
        }

    log = experiment / "logs" / f"targets_{arm}_seed{seed}_{detector}.log"
    command = [
        sys.executable,
        "scripts/evaluate_crop_shift_targets.py",
        "--detector-checkpoint",
        DETECTORS[detector],
        "--vision-checkpoint",
        _relative(checkpoint),
        "--detector-label",
        detector,
        "--class-counts",
        _relative(experiment / "frozen_class_counts.json"),
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
        _relative(rows),
        "--summary-output",
        _relative(summary),
    ]
    if cpu:
        command.append("--cpu")
    _record(progress, step=f"cell:{arm}:{seed}:{detector}", status="started", command=command)
    log.parent.mkdir(parents=True, exist_ok=True)
    cell_dir.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8", newline="\n") as handle:
        completed = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False)
    _record(
        progress,
        step=f"cell:{arm}:{seed}:{detector}",
        status="completed",
        exit_code=completed.returncode,
        log=str(log),
    )
    if completed.returncode:
        raise RuntimeError(f"target cell failed; see {log}")
    if not rows.is_file() or not summary.is_file():
        raise RuntimeError(f"target cell exited successfully but outputs are missing: {cell_dir}")
    return {
        "arm": arm,
        "recognition_seed": int(seed),
        "eval_detector": detector,
        "rows": str(rows),
        "summary": str(summary),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a validation-only crop-shift target matrix")
    parser.add_argument("--root", default="artifacts/experiments/crop_shift_alignment_20261009")
    parser.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS))
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--detectors", nargs="+", choices=sorted(DETECTORS), default=list(DETECTORS))
    parser.add_argument("--recognition-batch-size", type=int, default=128)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.recognition_batch_size <= 0:
        raise ValueError("--recognition-batch-size must be positive")
    experiment = (ROOT / args.root).resolve()
    if not experiment.is_dir():
        raise FileNotFoundError(experiment)
    progress = experiment / "target_matrix_progress.jsonl"
    cells: list[dict[str, str | int]] = []
    for arm in args.arms:
        for detector in args.detectors:
            cells.append(
                _run_cell(
                    experiment=experiment,
                    arm=str(arm),
                    seed=int(args.seed),
                    detector=str(detector),
                    batch_size=int(args.recognition_batch_size),
                    cpu=bool(args.cpu),
                    progress=progress,
                )
            )
    manifest = experiment / "target_matrix_cells.json"
    if manifest.exists():
        raise FileExistsError(f"refusing to overwrite target matrix manifest: {manifest}")
    manifest.write_text(json.dumps(cells, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _record(progress, step="matrix_manifest", status="completed", output=str(manifest), cells=len(cells))


if __name__ == "__main__":
    main()
