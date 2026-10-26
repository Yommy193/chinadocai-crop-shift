"""Small contract tests for the additive target-strata analyzer."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_crop_shift_targets import build_analysis  # noqa: E402


def _reference(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "class_train_counts": [5],
                "frequency_bins": [
                    {"name": "rare", "minimum": 1, "maximum": 20},
                    {"name": "medium", "minimum": 21, "maximum": 100},
                    {"name": "frequent", "minimum": 101, "maximum": None},
                ],
                "normalized_gold_box_area_tertiles": {"q33": 0.2, "q67": 0.8},
                "iou_bins": [
                    {"name": "0.50-0.60", "minimum": 0.5, "maximum_exclusive": 0.6},
                    {"name": "0.60-0.75", "minimum": 0.6, "maximum_exclusive": 0.75},
                    {"name": "0.75-0.90", "minimum": 0.75, "maximum_exclusive": 0.9},
                    {"name": "0.90-1.00", "minimum": 0.9, "maximum_inclusive": 1.0},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _row(target_id: str, detector: str, *, top1: int, source: str = "rubbing") -> dict:
    return {
        "target_id": target_id,
        "plate": "P1",
        "source": source,
        "detector_label": detector,
        "group_index": 0,
        "char_index": int(target_id[-1]),
        "target_label": "甲",
        "target_class_id": 0,
        "target_train_count": 5,
        "evaluation_eligible": True,
        "gold_box_xyxy": [0.0, 0.0, 10.0, 10.0],
        "gold_box_area_fraction": 0.5,
        "matched_at_iou": True,
        "matched_status": "matched",
        "matching_iou": 0.8,
        "predicted_crop_top1": top1,
        "predicted_crop_top5": 1,
        "gold_crop_top1": 1,
        "gold_crop_top5": 1,
    }


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def test_complete_matrix_and_synchronized_bootstrap(tmp_path: Path) -> None:
    reference = tmp_path / "strata.json"
    _reference(reference)
    cells = []
    for arm, value in (("gold", 1), ("pred", 0)):
        for detector in ("anchor", "frcnn"):
            rows = tmp_path / f"{arm}_{detector}.jsonl"
            _write(rows, [_row("t0", detector, top1=value), _row("t1", detector, top1=value)])
            cells.extend(["--cell", arm, "1", detector, str(rows)])
    report = build_analysis(
        strata_reference=reference,
        specs=[
            # Keep the test independent of CLI parsing while exercising the same contract.
            __import__("analyze_crop_shift_targets").CellSpec("gold", 1, "anchor", tmp_path / "gold_anchor.jsonl"),
            __import__("analyze_crop_shift_targets").CellSpec("gold", 1, "frcnn", tmp_path / "gold_frcnn.jsonl"),
            __import__("analyze_crop_shift_targets").CellSpec("pred", 1, "anchor", tmp_path / "pred_anchor.jsonl"),
            __import__("analyze_crop_shift_targets").CellSpec("pred", 1, "frcnn", tmp_path / "pred_frcnn.jsonl"),
        ],
        contrasts=[("pred", "gold")],
        bootstrap_replicates=50,
        bootstrap_seed=7,
    )
    assert report["matrix"]["complete_cartesian_product"] is True
    assert report["matrix"]["target_count_per_cell"] == 2
    assert report["bootstrap"]["resampling_unit"] == "plate"
    comparison = report["pairwise_arm_differences"][0]
    assert comparison["matched_conditional"]["overall"]["predicted_crop_top1"]["point_difference"] == -1.0


def test_inconsistent_detector_geometry_is_rejected(tmp_path: Path) -> None:
    reference = tmp_path / "strata.json"
    _reference(reference)
    good = _row("t0", "anchor", top1=1)
    bad = dict(good, matching_iou=0.7)
    specs = []
    for arm, row in (("gold", good), ("pred", bad)):
        path = tmp_path / f"{arm}.jsonl"
        _write(path, [row])
        specs.append(__import__("analyze_crop_shift_targets").CellSpec(arm, 1, "anchor", path))
    # Add the second detector to make the matrix complete; its geometry is valid.
    for arm in ("gold", "pred"):
        path = tmp_path / f"{arm}_frcnn.jsonl"
        _write(path, [_row("t0", "frcnn", top1=1)])
        specs.append(__import__("analyze_crop_shift_targets").CellSpec(arm, 1, "frcnn", path))
    with pytest.raises(ValueError, match="detector match geometry differs"):
        build_analysis(strata_reference=reference, specs=specs, bootstrap_replicates=5)
