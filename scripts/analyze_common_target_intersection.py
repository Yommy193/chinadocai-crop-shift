"""Recompute the 2x2 detector/crop interaction on a common target universe.

The original confirmation cells intentionally retain detector-specific misses and
therefore have different denominators.  This additive analysis joins stable
target IDs and reports a sensitivity analysis restricted to targets matched by
all four cells.  It never rewrites the original target rows.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[str(row["target_id"])] = row
    return rows


def bootstrap(values_by_plate: dict[str, np.ndarray], replicates: int, seed: int) -> dict[str, float]:
    plates = sorted(values_by_plate)
    if not plates:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "plates": 0}
    # First average within each plate; plates, rather than individual targets,
    # are the resampling units and may contain different numbers of targets.
    matrix = np.asarray([np.asarray(values_by_plate[p], dtype=float).mean() for p in plates], dtype=float)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(plates), size=(replicates, len(plates)))
    estimates = matrix[sampled].mean(axis=1)
    return {
        "mean": float(estimates.mean()),
        "lo": float(np.quantile(estimates, 0.025)),
        "hi": float(np.quantile(estimates, 0.975)),
        "plates": len(plates),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Common-target denominator sensitivity analysis")
    parser.add_argument("--target-root", default="artifacts/experiments/crop_shift_alignment_20261009/confirmation_20261010/target_rows")
    parser.add_argument("--output", default="artifacts/experiments/crop_shift_alignment_extension_20261021/results/common_target_intersection.json")
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--replicates", type=int, default=5000)
    args = parser.parse_args()
    root = (ROOT / args.target_root).resolve()
    out = (ROOT / args.output).resolve()
    cells = {
        "anchor_train_anchor_eval": root / "anchor_pred_step_seed20261009" / "anchor" / "targets.jsonl",
        "anchor_train_frcnn_eval": root / "anchor_pred_step_seed20261009" / "frcnn" / "targets.jsonl",
        "frcnn_train_anchor_eval": root / "frcnn_pred_step_seed20261009" / "anchor" / "targets.jsonl",
        "frcnn_train_frcnn_eval": root / "frcnn_pred_step_seed20261009" / "frcnn" / "targets.jsonl",
    }
    loaded = {name: load(path) for name, path in cells.items()}
    common_ids = set.intersection(*(set(rows) for rows in loaded.values()))
    eligible_ids = {
        target_id
        for target_id in common_ids
        if all(bool(loaded[name][target_id].get("evaluation_eligible")) for name in loaded)
    }
    all_matched_ids = {
        target_id
        for target_id in eligible_ids
        if all(bool(loaded[name][target_id].get("matched_at_iou")) for name in loaded)
    }
    metrics = {}
    for name, rows in loaded.items():
        eligible = [row for row in rows.values() if row.get("evaluation_eligible")]
        matched = [row for row in eligible if row.get("matched_at_iou")]
        metrics[name] = {
            "rows": len(rows),
            "eligible": len(eligible),
            "matched": len(matched),
            "matched_conditional_top1": float(np.mean([row["predicted_crop_top1"] for row in matched])) if matched else None,
            "matched_conditional_top5": float(np.mean([row["predicted_crop_top5"] for row in matched])) if matched else None,
        }

    common_metrics = {}
    for name, rows in loaded.items():
        one = np.asarray([int(rows[target_id]["predicted_crop_top1"]) for target_id in all_matched_ids], dtype=float)
        five = np.asarray([int(rows[target_id]["predicted_crop_top5"]) for target_id in all_matched_ids], dtype=float)
        by_plate_one: dict[str, list[float]] = defaultdict(list)
        by_plate_five: dict[str, list[float]] = defaultdict(list)
        for target_id in all_matched_ids:
            row = rows[target_id]
            by_plate_one[str(row["plate"])].append(float(row["predicted_crop_top1"]))
            by_plate_five[str(row["plate"])].append(float(row["predicted_crop_top5"]))
        common_metrics[name] = {
            "top1": float(one.mean()) if len(one) else None,
            "top5": float(five.mean()) if len(five) else None,
            "top1_plate_bootstrap": bootstrap({p: np.asarray(v) for p, v in by_plate_one.items()}, args.replicates, args.seed),
            "top5_plate_bootstrap": bootstrap({p: np.asarray(v) for p, v in by_plate_five.items()}, args.replicates, args.seed + 1),
        }

    interaction = {}
    for key in ("top1", "top5"):
        aa = np.asarray([int(loaded["anchor_train_anchor_eval"][t][f"predicted_crop_{key}"]) for t in all_matched_ids], dtype=float)
        af = np.asarray([int(loaded["anchor_train_frcnn_eval"][t][f"predicted_crop_{key}"]) for t in all_matched_ids], dtype=float)
        fa = np.asarray([int(loaded["frcnn_train_anchor_eval"][t][f"predicted_crop_{key}"]) for t in all_matched_ids], dtype=float)
        ff = np.asarray([int(loaded["frcnn_train_frcnn_eval"][t][f"predicted_crop_{key}"]) for t in all_matched_ids], dtype=float)
        interaction[key] = {"pp": float((aa.mean() + ff.mean() - af.mean() - fa.mean()) * 100.0), "n": int(len(aa))}

    payload = {
        "schema_version": 1,
        "scope": "additive common-target denominator sensitivity analysis",
        "cells": {name: str(path.relative_to(ROOT)).replace("\\", "/") for name, path in cells.items()},
        "unrestricted_cell_metrics": metrics,
        "common_target_counts": {
            "all_cell_rows": len(common_ids),
            "all_cells_in_vocabulary": len(eligible_ids),
            "all_cells_matched_at_iou": len(all_matched_ids),
            "plates": len({loaded[next(iter(loaded))][t]["plate"] for t in all_matched_ids}),
        },
        "common_target_metrics": common_metrics,
        "common_target_interaction": interaction,
        "bootstrap": {"replicates": int(args.replicates), "seed": int(args.seed), "unit": "validation plate"},
        "interpretation": "The common-target interaction is a sensitivity analysis; the primary detector-to-recognition metric remains the pre-registered matched conditional result.",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
