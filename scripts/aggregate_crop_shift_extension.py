"""Aggregate additive CSIG extension artifacts into a compact report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="artifacts/experiments/crop_shift_alignment_extension_20261021")
    args = parser.parse_args()
    root = (ROOT / args.root).resolve()
    results = root / "results"
    rows = []
    for name in ("gold", "anchorpred", "frcnnpred", "mix075"):
        path = results / f"fcos_val_{name}_summary.json"
        if not path.exists():
            continue
        summary = read(path)
        metrics = summary["metrics"]["overall"]
        rows.append(
            {
                "training_arm": {"gold": "gold_step", "anchorpred": "anchor_pred_step", "frcnnpred": "frcnn_pred_step", "mix075": "fcos_mix075_step"}[name],
                "evaluation_detector": "FCOS-R18-FPN",
                "matched_target_recall": metrics["matched_target_recall"],
                "top1_conditioned": metrics["predicted_crop_top1_conditioned_on_evaluated_match"],
                "top5_conditioned": metrics["predicted_crop_top5_conditioned_on_evaluated_match"],
                "end_to_end_top1_recovery": metrics["detector_to_recognition_top1_recovery"],
                "end_to_end_top5_recovery": metrics["detector_to_recognition_top5_recovery"],
                "gold_crop_top1_diagnostic": metrics["gold_crop_top1"],
                "gold_crop_top5_diagnostic": metrics["gold_crop_top5"],
                "visible_targets": metrics["visible_targets"],
                "matched_targets": metrics["matched_in_vocabulary_targets"],
                "runtime_seconds": summary["runtime_seconds"],
            }
        )
    # The FCOS-trained recognizer is also evaluated on the two historical
    # detector implementations.  These are additive validation-only rows;
    # the historical detector artifacts remain read-only inputs.
    for detector_name, detector_label in (("anchor", "Anchor-CenterNet"), ("frcnn", "Faster-RCNN")):
        path = results / f"{detector_name}_val_fcosmix_summary.json"
        if not path.exists():
            continue
        summary = read(path)
        metrics = summary["metrics"]["overall"]
        rows.append(
            {
                "training_arm": "fcos_mix075_step",
                "evaluation_detector": detector_label,
                "matched_target_recall": metrics["matched_target_recall"],
                "top1_conditioned": metrics["predicted_crop_top1_conditioned_on_evaluated_match"],
                "top5_conditioned": metrics["predicted_crop_top5_conditioned_on_evaluated_match"],
                "end_to_end_top1_recovery": metrics["detector_to_recognition_top1_recovery"],
                "end_to_end_top5_recovery": metrics["detector_to_recognition_top5_recovery"],
                "gold_crop_top1_diagnostic": metrics["gold_crop_top1"],
                "gold_crop_top5_diagnostic": metrics["gold_crop_top5"],
                "visible_targets": metrics["visible_targets"],
                "matched_targets": metrics["matched_in_vocabulary_targets"],
                "runtime_seconds": summary["runtime_seconds"],
            }
        )
    payload = {
        "schema_version": 1,
        "scope": "additive CSIG supplement only",
        "fcos_extension": read(root / "fcos_extension_summary.json") if (root / "fcos_extension_summary.json").exists() else None,
        "hwo_bc": read(root / "classification" / "hwobc_resnet18_cached" / "cnn_metrics.json") if (root / "classification" / "hwobc_resnet18_cached" / "cnn_metrics.json").exists() else None,
        "common_target_intersection": read(results / "common_target_intersection.json") if (results / "common_target_intersection.json").exists() else None,
        "common_target_intersection_all_seeds": read(results / "common_target_intersection_all_seeds.json") if (results / "common_target_intersection_all_seeds.json").exists() else None,
        "fcos_cross_eval_summary": read(results / "fcos_cross_eval_summary.json") if (results / "fcos_cross_eval_summary.json").exists() else None,
        "fcos_validation_rows": rows,
        "original_inputs_read_only": True,
    }
    (root / "extension_results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with (root / "results" / "fcos_validation_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["training_arm"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"fcos_rows": len(rows), "output": str(root / 'extension_results.json')}, ensure_ascii=False))


if __name__ == "__main__":
    main()
