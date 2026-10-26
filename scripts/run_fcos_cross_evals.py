"""Run additive, validation-only cross-detector evaluations for FCOS crop arms.

The script discovers completed recognizer checkpoints under the extension
directory, reads the seed from each checkpoint's metrics metadata, and runs
the frozen Anchor-CenterNet, Faster R-CNN, and FCOS detectors one at a time.
All outputs stay inside the additive extension directory; no test split is
constructed or accessed.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _seed_for(run_dir: Path) -> int:
    payload = json.loads((run_dir / "cnn_metrics.json").read_text(encoding="utf-8"))
    return int(payload["training"]["seed"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="artifacts/experiments/crop_shift_alignment_extension_20261021")
    parser.add_argument("--raw-dir", default="甲骨文多模态数据集")
    parser.add_argument("--recognition-batch-size", type=int, default=512)
    parser.add_argument("--only-seed", type=int, action="append", default=[])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root = _resolve(args.root).resolve()
    results = root / "results"
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    class_counts = root / "frozen_class_counts.json"
    detectors = {
        "anchor": ("anchor_centernet", ROOT / "artifacts/experiments/plate_detector_center_20260907/seed_20260931/detector_best.pt"),
        "frcnn": ("frcnn_strong", ROOT / "artifacts/experiments/plate_detector_strong_end_to_end_20260918/runs/seed_20260981/detector_best.pt"),
        "fcos": ("fcos_r18_fpn_oof", ROOT / "artifacts/experiments/plate_detector_baseline_comparison_provenance_disjoint_20260915/runs/b1_fcos_r18_fpn/fold_0/seed_20260971/detector_best.pt"),
    }
    arms = ("fcos_pred_step", "fcos_mix075_step")
    jobs: list[tuple[str, int, str, Path, Path]] = []
    for arm in arms:
        for run_dir in sorted((root / "runs").glob(f"{arm}_seed*")):
            checkpoint = run_dir / "cnn_best.pt"
            metrics = run_dir / "cnn_metrics.json"
            if not checkpoint.exists() or not metrics.exists():
                continue
            seed = _seed_for(run_dir)
            if args.only_seed and seed not in args.only_seed:
                continue
            for detector_name, (detector_label, detector_checkpoint) in detectors.items():
                if not detector_checkpoint.exists():
                    raise FileNotFoundError(detector_checkpoint)
                stem = f"{detector_name}_val_{arm}_seed{seed}"
                jobs.append((arm, seed, detector_name, checkpoint, detector_checkpoint))
    # Avoid duplicate jobs if a folder alias happens to contain the same seed.
    unique: dict[tuple[str, int, str], tuple[str, int, str, Path, Path]] = {}
    for job in jobs:
        unique[(job[0], job[1], job[2])] = job
    jobs = [unique[key] for key in sorted(unique)]
    completed = 0
    failures: list[dict[str, str]] = []
    for arm, seed, detector_name, vision_checkpoint, detector_checkpoint in jobs:
        detector_label = detectors[detector_name][0]
        stem = f"{detector_name}_val_{arm}_seed{seed}"
        rows_output = results / f"{stem}_targets.jsonl"
        summary_output = results / f"{stem}_summary.json"
        if summary_output.exists() and rows_output.exists() and not args.force:
            print(json.dumps({"skip": stem, "reason": "outputs_exist"}), flush=True)
            completed += 1
            continue
        out_log = logs / f"cross_eval_{stem}.out.log"
        err_log = logs / f"cross_eval_{stem}.err.log"
        command = [
            sys.executable,
            str(ROOT / "scripts/evaluate_crop_shift_targets.py"),
            "--raw-dir", str(_resolve(args.raw_dir)),
            "--vision-checkpoint", str(vision_checkpoint),
            "--detector-checkpoint", str(detector_checkpoint),
            "--detector-label", detector_label,
            "--class-counts", str(class_counts),
            "--split", "val",
            "--max-plates", "0",
            "--source", "both",
            "--recognition-batch-size", str(args.recognition_batch_size),
            "--rows-output", str(rows_output),
            "--summary-output", str(summary_output),
            "--overwrite",
        ]
        print(json.dumps({"start": stem, "command": command}, ensure_ascii=False), flush=True)
        with out_log.open("w", encoding="utf-8") as stdout, err_log.open("w", encoding="utf-8") as stderr:
            result = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, check=False)
        if result.returncode != 0:
            failures.append({"stem": stem, "error_log": str(err_log), "returncode": str(result.returncode)})
            print(json.dumps({"failed": stem, "returncode": result.returncode, "error_log": str(err_log)}), flush=True)
            continue
        completed += 1
        print(json.dumps({"done": stem}), flush=True)
    print(json.dumps({"jobs": len(jobs), "completed_or_skipped": completed, "failures": failures, "test_access": "prohibited"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
