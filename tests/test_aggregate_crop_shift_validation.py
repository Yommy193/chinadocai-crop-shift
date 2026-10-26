import json

import pytest

import scripts.aggregate_crop_shift_validation as aggregate


def _summary(top1: dict[str, float], plates: list[str] | None = None) -> dict:
    sources = {}
    for source in aggregate.SOURCES:
        sources[source] = {
            "recognition": {
                "matched_detected_box_top1": top1[source],
                "matched_ground_truth_box_top1": top1[source] + 0.01,
            },
            "joint_detection_recognition": {"dr_f1_at_iou_0_50": top1[source] - 0.1},
        }
    return {
        "protocol": {"split": "val"},
        "subset": {"max_plates": 0, "selected_plates": plates or ["P1", "P2"]},
        "sources": sources,
    }


def test_arm_summary_is_four_cell_macro_and_checks_plate_identity(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(aggregate, "ROOT", tmp_path)
    arm = tmp_path / "validation" / "arm"
    values = {
        "anchor": {"rubbing": 0.7, "facsimile": 0.8},
        "frcnn": {"rubbing": 0.75, "facsimile": 0.85},
    }
    for detector in aggregate.DETECTORS:
        directory = arm / detector
        directory.mkdir(parents=True)
        (directory / "summary.json").write_text(json.dumps(_summary(values[detector])), encoding="utf-8")
    result = aggregate.arm_summary(arm)
    assert result["selection_score"] == pytest.approx(0.775)
    assert result["worst_cell_matched_predicted_top1"] == pytest.approx(0.7)
    assert result["validation_plates"] == 2

    (arm / "frcnn" / "summary.json").write_text(
        json.dumps(_summary(values["frcnn"], plates=["P2", "P1"])), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="plate set differs"):
        aggregate.arm_summary(arm)


def test_complexity_tie_breaker_order() -> None:
    assert aggregate.complexity_rank("gold_step") < aggregate.complexity_rank("frcnn_pred_step")
    assert aggregate.complexity_rank("frcnn_pred_step") < aggregate.complexity_rank("frcnn_mix050_step")
    assert aggregate.complexity_rank("frcnn_mix050_step") < aggregate.complexity_rank("frcnn_curriculum_step")


def test_gold_is_an_upper_bound_not_a_deployable_candidate() -> None:
    assert not aggregate.is_deployable_candidate("gold_step")
    assert aggregate.is_deployable_candidate("frcnn_pred_step")

