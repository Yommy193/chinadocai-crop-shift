"""Aggregate the additive FCOS three-detector validation summaries."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from make_csig_extension_figures import discover_records, _bootstrap_difference

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="artifacts/experiments/crop_shift_alignment_extension_20261021")
    args = parser.parse_args()
    root = (ROOT / args.root).resolve()
    records = discover_records(root)
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["arm"], record["detector"])].append(record)
    metrics = (
        "matched_target_recall",
        "predicted_crop_top1_conditioned_on_evaluated_match",
        "predicted_crop_top5_conditioned_on_evaluated_match",
        "detector_to_recognition_top1_recovery",
        "detector_to_recognition_top5_recovery",
    )
    cells = []
    for (arm, detector), rows in sorted(grouped.items()):
        cell = {
            "training_arm": arm,
            "evaluation_detector": detector,
            "seeds": sorted(int(r["seed"]) for r in rows),
            "n_seeds": len(rows),
            "mean": {key: sum(float(r["metrics"][key]) for r in rows) / len(rows) for key in metrics},
            "per_seed": {str(r["seed"]): {key: r["metrics"].get(key) for key in metrics} for r in rows},
        }
        cells.append(cell)
    effects = {}
    for detector in ("anchor", "frcnn", "fcos"):
        effects[detector] = {}
        for metric in ("top1", "top5", "recovery1", "recovery5"):
            point, lo, hi, n = _bootstrap_difference(records, detector, metric)
            effects[detector][metric] = {"mean": point, "ci95": [lo, hi], "paired_seed_count": n}
    payload = {
        "schema_version": 1,
        "scope": "additive CSIG supplement only",
        "protocol": {"split": "val", "test_access": "not_accessed", "detectors": ["anchor", "frcnn", "fcos"], "matching_iou": 0.5},
        "training_configuration": {
            "trainer": "crop_shift_alignment_cached",
            "architecture": "resnet18",
            "epochs": 8,
            "batch_size": 1024,
            "num_workers": 0,
            "steps_per_epoch": 133,
            "pred_step_seeds": [20261023, 20261024, 20261025],
            "mix075_step_seeds": [20261023, 20261024, 20261025],
        },
        "cells": cells,
        "mix075_minus_pred_step_plate_bootstrap": effects,
        "original_inputs_read_only": True,
    }
    output = root / "results" / "fcos_cross_eval_summary.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_output = root / "results" / "fcos_cross_eval_cells.csv"
    with csv_output.open("w", newline="", encoding="utf-8-sig") as handle:
        fieldnames = ["training_arm", "evaluation_detector", "n_seeds", *metrics]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for cell in cells:
            writer.writerow({
                "training_arm": cell["training_arm"],
                "evaluation_detector": cell["evaluation_detector"],
                "n_seeds": cell["n_seeds"],
                **cell["mean"],
            })
    print(json.dumps({"records": len(records), "cells": len(cells), "output": str(output), "csv": str(csv_output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
