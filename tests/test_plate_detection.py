from __future__ import annotations

import pytest
import torch

from oracle_vit.plate_detection import box_iou, detection_metrics, is_visible_character_annotation, require_test_access_allowed, sanitize_xywh_boxes


def test_test_evaluator_gate_rejects_development_only_detector_configs() -> None:
    with pytest.raises(PermissionError, match="test dataset construction is prohibited"):
        require_test_access_allowed({"protocol": "Development-only detector loss pilot"})
    with pytest.raises(PermissionError, match="test dataset construction is prohibited"):
        require_test_access_allowed({"test_access": "prohibited"})
    require_test_access_allowed({"protocol": "Frozen final detector evaluation"})


def test_missing_position_placeholder_is_not_a_detection_target() -> None:
    assert is_visible_character_annotation({"Position": "1,2,3,4", "SeatFont": 0})
    assert not is_visible_character_annotation({"Position": "1,2,3,4", "SeatFont": 1})
    assert not is_visible_character_annotation({"SeatFont": 0})


def test_sanitize_boxes_clips_and_deduplicates() -> None:
    boxes, audit = sanitize_xywh_boxes([[-2, 1, 8, 6], [0, 1, 6, 6], [4, 4, -1, 3], [40, 1, 2, 2]], 10, 10)
    assert boxes == [[0.0, 1.0, 6.0, 7.0]]
    assert audit == {"invalid": 2, "clipped": 2, "duplicate": 1, "kept": 1}


def test_box_iou_handles_empty_and_known_overlap() -> None:
    first = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
    second = torch.tensor([[1.0, 1.0, 3.0, 3.0]])
    assert box_iou(first, second).item() == pytest.approx(1.0 / 7.0)
    assert box_iou(torch.empty((0, 4)), second).shape == (0, 1)


def test_detection_metrics_perfect_and_duplicate_prediction() -> None:
    target = {"boxes": torch.tensor([[0.0, 0.0, 2.0, 2.0]])}
    perfect = {"boxes": torch.tensor([[0.0, 0.0, 2.0, 2.0]]), "scores": torch.tensor([0.9])}
    duplicate = {"boxes": torch.tensor([[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 2.0, 2.0]]), "scores": torch.tensor([0.9, 0.8])}
    assert detection_metrics([perfect], [target], (0.5,))["by_iou"]["0.50"] == {"ap": 1.0, "recall": 1.0}
    assert detection_metrics([duplicate], [target], (0.5,))["by_iou"]["0.50"]["ap"] == pytest.approx(1.0)


def test_detection_metrics_missed_box_has_zero_recall() -> None:
    target = {"boxes": torch.tensor([[0.0, 0.0, 2.0, 2.0]])}
    miss = {"boxes": torch.tensor([[5.0, 5.0, 6.0, 6.0]]), "scores": torch.tensor([0.9])}
    result = detection_metrics([miss], [target], (0.5,))
    assert result["by_iou"]["0.50"] == {"ap": 0.0, "recall": 0.0}
