"""Aggregate the two-detector x two-modality validation selection matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from oracle_vit.utils import atomic_json_dump, file_sha256


ROOT = Path(__file__).resolve().parents[1]
DETECTORS = ("anchor", "frcnn")
SOURCES = ("rubbing", "facsimile")


def complexity_rank(arm: str) -> int:
    if "gold" in arm:
        return 0
    if "pred" in arm:
        return 1
    if "mix" in arm or "jitter" in arm:
        return 2
    if "curriculum" in arm:
        return 3
    return 4


def is_deployable_candidate(arm: str) -> bool:
    """Return whether an arm can consume detector-produced crops at inference."""

    return "gold" not in arm


def load_cell(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    protocol = payload.get("protocol", {})
    if protocol.get("split") != "val":
        raise ValueError(f"non-validation evaluation: {path}")
    if int(payload.get("subset", {}).get("max_plates", -1)) != 0:
        raise ValueError(f"selection must use the complete validation partition: {path}")
    if set(payload.get("sources", {})) != set(SOURCES):
        raise ValueError(f"missing source cell: {path}")
    return payload


def arm_summary(directory: Path) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    selection_values: list[float] = []
    gold_values: list[float] = []
    dr_values: list[float] = []
    plate_sets: list[tuple[str, ...]] = []
    for detector in DETECTORS:
        path = directory / detector / "summary.json"
        payload = load_cell(path)
        plate_sets.append(tuple(str(value) for value in payload["subset"]["selected_plates"]))
        cells[detector] = {"summary": str(path.relative_to(ROOT)).replace("\\", "/"), "sources": {}}
        for source in SOURCES:
            metrics = payload["sources"][source]
            predicted = float(metrics["recognition"]["matched_detected_box_top1"])
            gold = float(metrics["recognition"]["matched_ground_truth_box_top1"])
            dr = float(metrics["joint_detection_recognition"]["dr_f1_at_iou_0_50"])
            cells[detector]["sources"][source] = {
                "matched_predicted_top1": predicted,
                "matched_gold_top1": gold,
                "dr_f1_iou50": dr,
            }
            selection_values.append(predicted)
            gold_values.append(gold)
            dr_values.append(dr)
    if any(value != plate_sets[0] for value in plate_sets[1:]):
        raise ValueError(f"validation plate set differs across detectors: {directory}")
    return {
        "cells": cells,
        "selection_score": sum(selection_values) / len(selection_values),
        "worst_cell_matched_predicted_top1": min(selection_values),
        "mean_matched_gold_top1": sum(gold_values) / len(gold_values),
        "mean_dr_f1_iou50": sum(dr_values) / len(dr_values),
        "validation_plates": len(plate_sets[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank crop-shift arms using validation only")
    parser.add_argument("--root", default="artifacts/experiments/crop_shift_alignment_20261009")
    parser.add_argument("--arms", nargs="+")
    parser.add_argument("--output")
    args = parser.parse_args()
    experiment = (ROOT / args.root).resolve()
    evaluation_root = experiment / "validation"
    arms = args.arms or sorted(path.name for path in evaluation_root.iterdir() if path.is_dir())
    summaries = {arm: arm_summary(evaluation_root / arm) for arm in arms}
    ranking = sorted(
        arms,
        key=lambda arm: (
            -float(summaries[arm]["selection_score"]),
            -float(summaries[arm]["worst_cell_matched_predicted_top1"]),
            complexity_rank(arm),
            arm,
        ),
    )
    deployable_ranking = [arm for arm in ranking if is_deployable_candidate(arm)]
    if not deployable_ranking:
        raise ValueError("method selection requires at least one deployable predicted/mixed arm")
    report = {
        "schema_version": 1,
        "scope": "validation-only method selection; historical test access prohibited",
        "selection_metric": "equal-weight mean matched-predicted Top-1 across anchor/FRCNN x rubbing/facsimile",
        "tie_breakers": [
            "worst of four cells",
            "simpler policy (gold < static predicted < mix/jitter < curriculum)",
            "lexical arm name as deterministic final fallback",
        ],
        "arms": summaries,
        "ranking": ranking,
        "deployable_ranking": deployable_ranking,
        "selected": deployable_ranking[0],
        "gold_upper_bound": "gold_step" if "gold_step" in summaries else None,
        "checkpoint_hashes": {
            arm: file_sha256(experiment / "runs" / f"{arm}_seed20261009" / "cnn_best.pt")
            for arm in arms
        },
        "test_access": "prohibited",
    }
    output = Path(args.output) if args.output else experiment / "validation_selection.json"
    if not output.is_absolute():
        output = ROOT / output
    atomic_json_dump(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

