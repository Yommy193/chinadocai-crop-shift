"""Pure helpers for the audited full-plate inference smoke test.

These utilities deliberately contain no model loading, file writing, or data
split selection.  Keeping matching and reading-order scoring here makes the
validation-only ``plate -> crop -> recognition -> order`` runner testable
without changing the deployed inference path.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import torch

from .plate_detection import box_iou


def xyxy_to_xywh(box: Sequence[float]) -> list[int]:
    """Convert one clipped xyxy box to a non-empty integer xywh crop box."""
    if len(box) != 4:
        raise ValueError("box must contain four xyxy coordinates")
    x1, y1, x2, y2 = (float(value) for value in box)
    left, top = int(round(x1)), int(round(y1))
    right, bottom = int(round(x2)), int(round(y2))
    if right <= left or bottom <= top:
        raise ValueError("xyxy box is empty after integer conversion")
    return [left, top, right - left, bottom - top]


def greedy_iou_matches(
    prediction_boxes: torch.Tensor,
    prediction_scores: torch.Tensor,
    target_boxes: torch.Tensor,
    iou_threshold: float,
) -> list[tuple[int, int, float]]:
    """Match score-ranked predictions to unique targets at an IoU threshold.

    The deterministic rule is the same one used by ordinary one-class
    detection evaluation: visit predictions by descending score, give each
    prediction its highest-IoU as-yet-unmatched target, and use the lower
    target index to break exact IoU ties.  It is intentionally not a Hungarian
    assignment, because it mirrors detector confidence semantics.
    """
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in [0, 1]")
    if prediction_boxes.ndim != 2 or prediction_boxes.shape[-1] != 4:
        raise ValueError("prediction_boxes must have shape [N, 4]")
    if target_boxes.ndim != 2 or target_boxes.shape[-1] != 4:
        raise ValueError("target_boxes must have shape [M, 4]")
    if prediction_scores.ndim != 1 or prediction_scores.shape[0] != prediction_boxes.shape[0]:
        raise ValueError("prediction_scores must align with prediction_boxes")
    if not prediction_boxes.numel() or not target_boxes.numel():
        return []

    overlaps = box_iou(prediction_boxes.float().cpu(), target_boxes.float().cpu())
    prediction_order = sorted(
        range(int(prediction_boxes.shape[0])),
        key=lambda index: (-float(prediction_scores[index]), index),
    )
    unmatched = set(range(int(target_boxes.shape[0])))
    matches: list[tuple[int, int, float]] = []
    for prediction_index in prediction_order:
        if not unmatched:
            break
        target_index = min(
            unmatched,
            key=lambda index: (-float(overlaps[prediction_index, index]), index),
        )
        overlap = float(overlaps[prediction_index, target_index])
        if overlap >= iou_threshold:
            matches.append((prediction_index, target_index, overlap))
            unmatched.remove(target_index)
    return matches


def reading_order_metrics(order: Sequence[int], gold_labels: Sequence[str]) -> tuple[int, int, int]:
    """Return content-aware pair accuracy counts and exact-content success.

    Repeated canonical labels are not comparable as a pair, matching the
    sequence evaluator's established metric definition.  Exact content is
    label-based as well, so swapping two identical labels is not treated as a
    measurable reading-order error.
    """
    if sorted(int(index) for index in order) != list(range(len(gold_labels))):
        raise ValueError("order must be a permutation of every gold-label index")
    ranks = [0] * len(order)
    for rank, index in enumerate(order):
        ranks[int(index)] = rank
    correct = total = 0
    for left in range(len(gold_labels)):
        for right in range(left + 1, len(gold_labels)):
            if gold_labels[left] == gold_labels[right]:
                continue
            total += 1
            correct += int(ranks[left] < ranks[right])
    exact = int([gold_labels[int(index)] for index in order] == list(gold_labels))
    return correct, total, exact


def deterministic_plate_subset(
    plate_names: Sequence[str], max_plates: int, seed: int
) -> list[str]:
    """Return a stable, hash-ranked validation subset without data-dependent tuning."""
    if max_plates < 0:
        raise ValueError("max_plates must be non-negative")
    unique = list(dict.fromkeys(str(name) for name in plate_names))
    ranked = sorted(
        unique,
        key=lambda name: (hashlib.sha256(f"{seed}:{name}".encode("utf-8")).hexdigest(), name),
    )
    return ranked if max_plates == 0 else ranked[:max_plates]


def detection_recognition_counts(
    prediction_boxes: torch.Tensor,
    prediction_scores: torch.Tensor,
    prediction_labels: Sequence[str | None],
    target_boxes: torch.Tensor,
    target_labels: Sequence[str],
    ignored_target_boxes: torch.Tensor,
    iou_threshold: float,
) -> dict[str, int]:
    """Score fixed-threshold joint detection-and-recognition decisions.

    A true positive must have a detector box at or above ``iou_threshold`` and
    the recognizer's Top-1 canonical label must equal a unique labelled target.
    Every other evaluated detection is a false positive; each unmatched labelled
    target is a false negative.  Visible targets without a vocabulary label are
    passed as ``ignored_target_boxes``: a prediction that only overlaps one of
    those boxes cannot be fairly classified, so it is excluded from precision.

    This is a single operating-point count, not AP and not a calibrated score
    sweep.  Predictions are processed by descending detector score, with
    original prediction index as a deterministic tie break.
    """
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in [0, 1]")
    if prediction_boxes.ndim != 2 or prediction_boxes.shape[-1] != 4:
        raise ValueError("prediction_boxes must have shape [N, 4]")
    if prediction_scores.ndim != 1 or prediction_scores.shape[0] != prediction_boxes.shape[0]:
        raise ValueError("prediction_scores must align with prediction_boxes")
    if len(prediction_labels) != prediction_boxes.shape[0]:
        raise ValueError("prediction_labels must align with prediction_boxes")
    if target_boxes.ndim != 2 or target_boxes.shape[-1] != 4:
        raise ValueError("target_boxes must have shape [M, 4]")
    if len(target_labels) != target_boxes.shape[0]:
        raise ValueError("target_labels must align with target_boxes")
    if ignored_target_boxes.ndim != 2 or ignored_target_boxes.shape[-1] != 4:
        raise ValueError("ignored_target_boxes must have shape [K, 4]")

    overlaps = box_iou(prediction_boxes.float().cpu(), target_boxes.float().cpu())
    ignored_overlaps = box_iou(prediction_boxes.float().cpu(), ignored_target_boxes.float().cpu())
    unmatched = set(range(int(target_boxes.shape[0])))
    counts = {"true_positive": 0, "false_positive": 0, "false_negative": 0, "ignored_prediction": 0}
    order = sorted(
        range(int(prediction_boxes.shape[0])),
        key=lambda index: (-float(prediction_scores[index]), index),
    )
    for prediction_index in order:
        label = prediction_labels[prediction_index]
        same_label = [
            target_index
            for target_index in unmatched
            if isinstance(label, str) and label == target_labels[target_index]
        ]
        if same_label:
            best_target = min(
                same_label,
                key=lambda target_index: (-float(overlaps[prediction_index, target_index]), target_index),
            )
            if float(overlaps[prediction_index, best_target]) >= iou_threshold:
                counts["true_positive"] += 1
                unmatched.remove(best_target)
                continue
        # A wrong label on a labelled target is an evaluated false positive,
        # even when a nearby unlabelled target also exists.
        overlaps_labelled_target = bool(
            target_boxes.numel() and float(overlaps[prediction_index].max()) >= iou_threshold
        )
        overlaps_ignore_target = bool(
            ignored_target_boxes.numel() and float(ignored_overlaps[prediction_index].max()) >= iou_threshold
        )
        if not overlaps_labelled_target and overlaps_ignore_target:
            counts["ignored_prediction"] += 1
        else:
            counts["false_positive"] += 1
    counts["false_negative"] = len(unmatched)
    return counts
