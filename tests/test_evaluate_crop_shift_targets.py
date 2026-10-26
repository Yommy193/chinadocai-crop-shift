from __future__ import annotations

import pytest
import torch

from scripts.evaluate_crop_shift_targets import (
    _validate_access,
    assemble_target_rows,
    stable_target_id,
    summarise_rows,
)


def recognition(*labels: str) -> dict:
    return {
        "candidates": [
            {"label": label, "class_id": index, "probability": 0.9 - index * 0.1}
            for index, label in enumerate(labels)
        ]
    }


def test_stable_target_id_is_model_independent_and_source_specific() -> None:
    first = stable_target_id("plate-1", "rubbing", 2, 3)
    assert first == stable_target_id("plate-1", "rubbing", 2, 3)
    assert first != stable_target_id("plate-1", "facsimile", 2, 3)
    assert first != stable_target_id("plate-1", "rubbing", 2, 4)
    assert first.startswith("target_sha256:")


def test_target_rows_retain_misses_oov_and_both_crop_paths() -> None:
    annotations = [
        {"annotation_index": 0, "group_index": 1, "char_index": 0, "label": "A", "box_xyxy": [0, 0, 10, 20]},
        {"annotation_index": 1, "group_index": 1, "char_index": 1, "label": "B", "box_xyxy": [20, 0, 30, 10]},
        {"annotation_index": 2, "group_index": 2, "char_index": 0, "label": "OOV", "box_xyxy": [0, 20, 10, 30]},
    ]
    rows = assemble_target_rows(
        plate="p",
        source="rubbing",
        detector_label="frcnn",
        image_width=100,
        image_height=50,
        annotations=annotations,
        prediction_boxes=torch.tensor([[1, 1, 10, 19], [0, 20, 9, 30]], dtype=torch.float32),
        prediction_scores=torch.tensor([0.9, 0.7]),
        matches=[(0, 0, 0.81), (1, 2, 0.72)],
        predicted_recognition={0: recognition("A", "B"), 1: recognition("B", "A")},
        gold_recognition={0: recognition("A", "B"), 1: recognition("A", "B"), 2: recognition("A", "B")},
        vocabulary={"labels": ["A", "B"], "label_to_class": {"A": 0, "B": 1}},
        class_counts=[11, 4],
    )

    assert len(rows) == 3
    assert len({row["target_id"] for row in rows}) == 3
    assert rows[0]["detector_label"] == "frcnn"
    assert rows[0]["matched_status"] == "matched"
    assert rows[0]["predicted_box_xyxy"] == [1.0, 1.0, 10.0, 19.0]
    assert rows[0]["matching_iou"] == pytest.approx(0.81)
    assert rows[0]["detector_score"] == pytest.approx(0.9)
    assert rows[0]["target_class_id"] == 0
    assert rows[0]["target_train_count"] == 11
    assert rows[0]["predicted_crop_top1"] == 1
    assert rows[0]["predicted_crop_top5"] == 1
    assert rows[0]["gold_crop_top1"] == 1
    assert rows[0]["gold_crop_top5"] == 1

    assert rows[1]["matched_status"] == "unmatched"
    assert rows[1]["predicted_box_xyxy"] is None
    assert rows[1]["predicted_crop_top1"] is None
    assert rows[1]["gold_crop_top1"] == 0
    assert rows[1]["gold_crop_top5"] == 1

    assert rows[2]["matched_status"] == "matched"
    assert rows[2]["evaluation_eligible"] is False
    assert rows[2]["predicted_crop_top1_label"] == "B"
    assert rows[2]["predicted_crop_top1"] is None
    assert rows[2]["gold_crop_top5"] is None

    summary = summarise_rows(rows)
    assert summary["visible_targets"] == 3
    assert summary["in_vocabulary_targets"] == 2
    assert summary["matched_in_vocabulary_targets"] == 1
    assert summary["matched_target_recall"] == pytest.approx(0.5)
    assert summary["detector_to_recognition_top1_recovery"] == pytest.approx(0.5)


def test_test_partition_requires_explicit_acknowledgement() -> None:
    with pytest.raises(PermissionError, match="--allow-test"):
        _validate_access("test", False)
    _validate_access("test", True)
    _validate_access("val", False)


def test_class_count_alignment_is_strict() -> None:
    with pytest.raises(ValueError, match="class_counts"):
        assemble_target_rows(
            plate="p",
            source="rubbing",
            detector_label="anchor",
            image_width=10,
            image_height=10,
            annotations=[],
            prediction_boxes=torch.empty((0, 4)),
            prediction_scores=torch.empty((0,)),
            matches=[],
            predicted_recognition={},
            gold_recognition={},
            vocabulary={"labels": ["A"], "label_to_class": {"A": 0}},
            class_counts=[],
        )
