"""Validation-only full-plate detection-to-reading-order smoke evaluation.

This runner connects the existing frozen components without changing them:

    full plate -> visible-character detector -> predicted-box crop ->
    single-glyph recognizer -> sequence sorter.

It is deliberately a *development/validation* tool.  Its detector checkpoint
was selected on the same validation partition, and ground-truth group
membership is used only after detection to make reading-order scoring
measurable.  Consequently, it must never be reported as a held-out end-to-end
test result or as a deployable automatic group-segmentation result.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from PIL import Image, ImageOps
from torchvision.transforms import functional as TF

from oracle_vit.anchor_free_detector import TinyPlateCenterNet
from oracle_vit.data import _square_crop
from oracle_vit.end_to_end_plate import (
    detection_recognition_counts,
    deterministic_plate_subset,
    greedy_iou_matches,
    reading_order_metrics,
    xyxy_to_xywh,
)
from oracle_vit.infer import _load_sequence, _load_vision, order_tokens, recognize
from oracle_vit.plate_detection import SOURCE_FIELDS, is_visible_character_annotation
from oracle_vit.plate_inference import predict_plate_detections
from oracle_vit.prepare import _assign_splits
from oracle_vit.utils import atomic_json_dump, file_sha256, resolve_project_path


ROOT = Path(__file__).resolve().parents[1]


def _safe_output_path(value: str) -> Path:
    """Resolve an output and reject writes outside the user's project directory."""
    path = resolve_project_path(value).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"output must remain inside {ROOT}: {path}") from error
    return path


def _finite_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _parse_and_clip_xywh(value: Any, width: int, height: int) -> list[float] | None:
    """Parse a raw annotation and return a clipped xyxy box, if valid."""
    if not isinstance(value, str):
        return None
    try:
        parts = [float(part.strip()) for part in value.split(",")]
    except ValueError:
        return None
    if len(parts) != 4:
        return None
    x, y, box_width, box_height = parts
    if box_width <= 0 or box_height <= 0:
        return None
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(float(width), x + box_width), min(float(height), y + box_height)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def collect_visible_annotations(
    record: dict[str, Any], image_width: int, image_height: int
) -> tuple[list[dict[str, Any]], dict[tuple[int, int], str], dict[tuple[int, int], int], Counter[str]]:
    """Create the same visible-box target set used by the detector protocol.

    The returned member maps preserve the original group/character identity so
    that *after* detector matching, an annotation-only reading-order score can
    be computed without pretending the detector has solved group segmentation.
    """
    annotations: list[dict[str, Any]] = []
    member_status: dict[tuple[int, int], str] = {}
    member_to_annotation: dict[tuple[int, int], int] = {}
    seen: set[tuple[float, float, float, float]] = set()
    audit: Counter[str] = Counter()
    for group_index, group in enumerate(record["RecordUtilSentenceGroupVoList"]):
        for char_index, char in enumerate(group["RecordUtilOracleCharVoList"]):
            key = (group_index, char_index)
            if not char.get("Position"):
                member_status[key] = "no_position"
                audit["no_position"] += 1
                continue
            if not is_visible_character_annotation(char):
                member_status[key] = "placeholder_excluded"
                audit["placeholder_excluded"] += 1
                continue
            box = _parse_and_clip_xywh(char["Position"], image_width, image_height)
            if box is None:
                member_status[key] = "invalid"
                audit["invalid"] += 1
                continue
            box_key = tuple(round(part, 5) for part in box)
            if box_key in seen:
                member_status[key] = "duplicate_excluded"
                audit["duplicate_excluded"] += 1
                continue
            seen.add(box_key)
            annotation_index = len(annotations)
            annotations.append(
                {
                    "annotation_index": annotation_index,
                    "group_index": group_index,
                    "char_index": char_index,
                    "category": group.get("GroupCategory"),
                    "box_xyxy": box,
                    "label": char.get("Label"),
                    "order_number": char.get("OrderNumber"),
                }
            )
            member_status[key] = "kept"
            member_to_annotation[key] = annotation_index
            audit["visible_kept"] += 1
    return annotations, member_status, member_to_annotation, audit


def eligible_reading_groups(
    record: dict[str, Any],
    member_status: dict[tuple[int, int], str],
    member_to_annotation: dict[tuple[int, int], int],
    annotations: list[dict[str, Any]],
    vocabulary_labels: set[str],
    max_length: int,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Keep groups for which a complete visible, labelled order is measurable."""
    groups: list[dict[str, Any]] = []
    audit: Counter[str] = Counter()
    for group_index, group in enumerate(record["RecordUtilSentenceGroupVoList"]):
        chars = group["RecordUtilOracleCharVoList"]
        audit["groups_total"] += 1
        if not 2 <= len(chars) <= max_length:
            audit["excluded_length"] += 1
            continue
        members = [(group_index, char_index) for char_index in range(len(chars))]
        if any(member_status.get(member) != "kept" for member in members):
            audit["excluded_nonvisible_or_geometry"] += 1
            continue
        entries = [annotations[member_to_annotation[member]] for member in members]
        labels = [entry["label"] for entry in entries]
        if not all(isinstance(label, str) and label in vocabulary_labels for label in labels):
            audit["excluded_unlabelled_or_oov"] += 1
            continue
        try:
            order_numbers = [int(entry["order_number"]) for entry in entries]
        except (TypeError, ValueError):
            audit["excluded_ambiguous_order"] += 1
            continue
        if len(set(order_numbers)) != len(order_numbers):
            audit["excluded_ambiguous_order"] += 1
            continue
        ordered = sorted(entries, key=lambda entry: (int(entry["order_number"]), int(entry["char_index"])))
        groups.append(
            {
                "group_index": group_index,
                "category": group.get("GroupCategory"),
                "annotation_indexes": [int(entry["annotation_index"]) for entry in ordered],
                "gold_labels": [str(entry["label"]) for entry in ordered],
            }
        )
        audit["eligible"] += 1
    return groups, audit


def load_detector(path: Path, device: torch.device) -> tuple[TinyPlateCenterNet, dict[str, Any]]:
    state = torch.load(path, map_location=device, weights_only=False)
    model_config = dict(state["model_config"])
    if model_config.get("architecture") != "tiny_anchor_free_centernet" or model_config.get("external_pretraining"):
        raise ValueError("the runner only accepts the frozen from-scratch TinyPlateCenterNet checkpoints")
    model = TinyPlateCenterNet(width=int(model_config["width"]))
    model.load_state_dict(state["model_state"])
    return model.to(device).eval(), state


def _topk_hit(row: dict[str, Any], target: str, width: int) -> bool:
    return target in {str(candidate["label"]) for candidate in row["candidates"][:width]}


def evaluate_one_plate(
    *,
    record: dict[str, Any],
    source: str,
    image_path: Path,
    detector: TinyPlateCenterNet,
    detector_config: dict[str, Any],
    vision: torch.nn.Module,
    vision_vocab: dict[str, Any],
    sequence: torch.nn.Module,
    device: torch.device,
    score_threshold: float,
    top_k: int,
    iou_threshold: float,
) -> tuple[dict[str, Any], Counter[str], Counter[str], Counter[str]]:
    """Run all frozen stages for one plate and return auditable local evidence."""
    plate_started = perf_counter()
    with Image.open(image_path) as handle:
        image = ImageOps.exif_transpose(handle).convert("RGB")
    annotations, member_status, member_to_annotation, annotation_audit = collect_visible_annotations(
        record, image.width, image.height
    )
    vocabulary_labels = set(str(label) for label in vision_vocab["labels"])
    groups, group_audit = eligible_reading_groups(
        record,
        member_status,
        member_to_annotation,
        annotations,
        vocabulary_labels,
        int(sequence.config.max_length),
    )
    detector_started = perf_counter()
    decoded = predict_plate_detections(
        detector,
        TF.pil_to_tensor(image).float().div(255.0),
        input_size=int(detector_config["input_size"]),
        score_threshold=score_threshold,
        top_k=int(detector_config["top_k"]),
        nms_threshold=float(detector_config["nms_threshold"]),
    )
    detector_seconds = perf_counter() - detector_started
    target_boxes = torch.tensor([entry["box_xyxy"] for entry in annotations], dtype=torch.float32).reshape(-1, 4)
    matches = greedy_iou_matches(decoded["boxes"], decoded["scores"], target_boxes, iou_threshold)
    match_rows: list[dict[str, Any]] = []
    labelled_matches: list[dict[str, Any]] = []
    for prediction_index, target_index, overlap in matches:
        annotation = annotations[target_index]
        row = {
            "prediction_index": prediction_index,
            "annotation_index": target_index,
            "iou": overlap,
            "score": float(decoded["scores"][prediction_index]),
            "predicted_box_xyxy": [float(value) for value in decoded["boxes"][prediction_index].tolist()],
            "target_box_xyxy": [float(value) for value in annotation["box_xyxy"]],
            "group_index": int(annotation["group_index"]),
            "char_index": int(annotation["char_index"]),
            "target_label": annotation["label"],
        }
        match_rows.append(row)
        if isinstance(annotation["label"], str) and annotation["label"] in vocabulary_labels:
            labelled_matches.append(row)

    # Recognize every detector candidate, not just boxes that happen to match a
    # target.  This is necessary for a genuine fixed-operating-point
    # detection-recognition F1: unmatched and wrongly labelled detections must
    # remain eligible false positives.
    recognition_started = perf_counter()
    prediction_crop_failures = 0
    prediction_crops: list[Image.Image] = []
    prediction_crop_rows: list[dict[str, Any]] = []
    for prediction_index, box in enumerate(decoded["boxes"].tolist()):
        try:
            prediction_crops.append(_square_crop(image, xyxy_to_xywh(box)))
            prediction_crop_rows.append({"prediction_index": prediction_index})
        except ValueError:
            prediction_crop_failures += 1
    prediction_recognition_by_index: dict[int, dict[str, Any]] = {}
    candidate_width = max(5, int(top_k))
    if prediction_crops:
        prediction_tokens, prediction_rows = recognize(
            vision, vision_vocab, prediction_crops, device, candidate_width, robust_tta=False
        )
        for row, token, recognition in zip(prediction_crop_rows, prediction_tokens, prediction_rows):
            prediction_recognition_by_index[int(row["prediction_index"])] = {
                "detected_token": int(token),
                "detected_label": str(recognition["candidates"][0]["label"]),
                "candidates": recognition["candidates"],
            }

    matched_crop_failures = 0
    recognisable: list[dict[str, Any]] = []
    oracle_crops: list[Image.Image] = []
    for row in labelled_matches:
        if int(row["prediction_index"]) not in prediction_recognition_by_index:
            matched_crop_failures += 1
            continue
        try:
            oracle_crops.append(_square_crop(image, xyxy_to_xywh(row["target_box_xyxy"])))
            recognisable.append(row)
        except ValueError:
            matched_crop_failures += 1

    recognized_by_annotation: dict[int, dict[str, Any]] = {}
    if recognisable:
        oracle_tokens, oracle_rows = recognize(
            vision, vision_vocab, oracle_crops, device, candidate_width, robust_tta=False
        )
        for row, oracle_token, oracle_recognition in zip(recognisable, oracle_tokens, oracle_rows):
            target_label = str(row["target_label"])
            detected = prediction_recognition_by_index[int(row["prediction_index"])]
            detected_label = str(detected["detected_label"])
            oracle_label = str(oracle_recognition["candidates"][0]["label"])
            recognized_by_annotation[int(row["annotation_index"])] = {
                **row,
                "detected_token": int(detected["detected_token"]),
                "detected_label": detected_label,
                "detected_top1": int(detected_label == target_label),
                "detected_top5": int(_topk_hit(detected, target_label, 5)),
                "oracle_crop_token": int(oracle_token),
                "oracle_crop_label": oracle_label,
                "oracle_crop_top1": int(oracle_label == target_label),
                "oracle_crop_top5": int(_topk_hit(oracle_recognition, target_label, 5)),
            }

    labelled_targets = [
        entry for entry in annotations if isinstance(entry["label"], str) and entry["label"] in vocabulary_labels
    ]
    ignored_targets = [
        entry
        for entry in annotations
        if not (isinstance(entry["label"], str) and entry["label"] in vocabulary_labels)
    ]
    labelled_target_boxes = (
        torch.tensor([entry["box_xyxy"] for entry in labelled_targets], dtype=torch.float32)
        if labelled_targets
        else torch.empty((0, 4), dtype=torch.float32)
    )
    ignored_target_boxes = (
        torch.tensor([entry["box_xyxy"] for entry in ignored_targets], dtype=torch.float32)
        if ignored_targets
        else torch.empty((0, 4), dtype=torch.float32)
    )
    dr_counts = detection_recognition_counts(
        decoded["boxes"],
        decoded["scores"],
        [
            prediction_recognition_by_index.get(prediction_index, {}).get("detected_label")
            for prediction_index in range(int(decoded["boxes"].shape[0]))
        ],
        labelled_target_boxes,
        [str(entry["label"]) for entry in labelled_targets],
        ignored_target_boxes,
        iou_threshold,
    )
    recognition_seconds = perf_counter() - recognition_started

    counts: Counter[str] = Counter()
    counts["plates"] = 1
    counts["visible_annotations"] = len(annotations)
    counts["visible_labelled_annotations"] = sum(
        isinstance(entry["label"], str) and entry["label"] in vocabulary_labels for entry in annotations
    )
    counts["detector_predictions"] = int(decoded["boxes"].shape[0])
    counts["matched_visible_annotations"] = len(matches)
    counts["matched_labelled_annotations"] = len(labelled_matches)
    counts["recognised_matched_labelled_annotations"] = len(recognized_by_annotation)
    counts["crop_failures"] = matched_crop_failures
    counts["detector_prediction_crop_failures"] = prediction_crop_failures
    for key, value in dr_counts.items():
        counts[f"dr_{key}"] = int(value)
    for entry in recognized_by_annotation.values():
        counts["detected_crop_top1"] += int(entry["detected_top1"])
        counts["detected_crop_top5"] += int(entry["detected_top5"])
        counts["oracle_crop_top1"] += int(entry["oracle_crop_top1"])
        counts["oracle_crop_top5"] += int(entry["oracle_crop_top5"])

    ordering_started = perf_counter()
    ordering_counts: Counter[str] = Counter()
    group_rows: list[dict[str, Any]] = []
    token_by_label = {str(label): int(token) for label, token in vision_vocab["label_to_token"].items()}
    for group in groups:
        ordering_counts["eligible_groups"] += 1
        annotation_indexes = list(group["annotation_indexes"])
        if not all(index in recognized_by_annotation for index in annotation_indexes):
            ordering_counts["incomplete_detection_groups"] += 1
            group_rows.append(
                {
                    "group_index": group["group_index"],
                    "category": group["category"],
                    "length": len(annotation_indexes),
                    "fully_detected": False,
                }
            )
            continue
        ordering_counts["fully_detected_groups"] += 1
        # The detector has no group head.  For a measurable validation probe,
        # annotations assign its matched boxes to their known group *after*
        # detection.  The sorter itself receives only recognised token IDs.
        ordered_gold_entries = [recognized_by_annotation[index] for index in annotation_indexes]
        input_entries = sorted(ordered_gold_entries, key=lambda entry: int(entry["prediction_index"]))
        true_positions = [annotation_indexes.index(int(entry["annotation_index"])) for entry in input_entries]
        detected_tokens = [int(entry["detected_token"]) for entry in input_entries]
        gold_tokens = [token_by_label[label] for label in group["gold_labels"]]
        gold_input_tokens = [gold_tokens[position] for position in true_positions]
        detected_order, _ = order_tokens(sequence, detected_tokens, device)
        gold_order, _ = order_tokens(sequence, gold_input_tokens, device)
        detected_output_positions = [true_positions[index] for index in detected_order]
        gold_output_positions = [true_positions[index] for index in gold_order]
        detected_pair_correct, detected_pair_total, detected_exact = reading_order_metrics(
            detected_output_positions, group["gold_labels"]
        )
        gold_pair_correct, gold_pair_total, gold_exact = reading_order_metrics(
            gold_output_positions, group["gold_labels"]
        )
        ordering_counts["detected_pair_correct"] += detected_pair_correct
        ordering_counts["detected_pair_total"] += detected_pair_total
        ordering_counts["detected_exact"] += detected_exact
        ordering_counts["gold_token_pair_correct"] += gold_pair_correct
        ordering_counts["gold_token_pair_total"] += gold_pair_total
        ordering_counts["gold_token_exact"] += gold_exact
        all_top1 = all(int(entry["detected_top1"]) for entry in ordered_gold_entries)
        strict_success = int(bool(all_top1) and bool(detected_exact))
        ordering_counts["strict_pipeline_exact"] += strict_success
        group_rows.append(
            {
                "group_index": group["group_index"],
                "category": group["category"],
                "length": len(annotation_indexes),
                "fully_detected": True,
                "detected_crop_pair_correct": detected_pair_correct,
                "detected_crop_pair_total": detected_pair_total,
                "detected_crop_exact_content": detected_exact,
                "gold_token_pair_correct": gold_pair_correct,
                "gold_token_pair_total": gold_pair_total,
                "gold_token_exact_content": gold_exact,
                "all_detected_crops_top1": bool(all_top1),
                "strict_pipeline_exact": strict_success,
            }
        )
    ordering_seconds = perf_counter() - ordering_started

    plate_result = {
        "plate": str(record["RubbingName"]),
        "source": source,
        "image": str(image_path.resolve()),
        "image_size": {"width": image.width, "height": image.height},
        "annotations": {
            "audit": dict(annotation_audit),
            "visible": len(annotations),
            "labelled_visible": counts["visible_labelled_annotations"],
        },
        "detector": {
            "predictions": counts["detector_predictions"],
            "matches_at_iou": len(matches),
            "labelled_matches": len(labelled_matches),
            "matched_crop_failures": matched_crop_failures,
            "all_prediction_crop_failures": prediction_crop_failures,
            "joint_detection_recognition": dr_counts,
        },
        "recognition_matches": list(recognized_by_annotation.values()),
        "reading_order": {"eligibility": dict(group_audit), "groups": group_rows},
        "timing_seconds": {
            "total": perf_counter() - plate_started,
            "detector": detector_seconds,
            "recognition_and_dr_f1": recognition_seconds,
            "reading_order": ordering_seconds,
        },
    }
    return plate_result, counts, ordering_counts, group_audit


def summarise_source(
    counts: Counter[str],
    ordering: Counter[str],
    group_audit: Counter[str],
    expected_plates: int,
    timing: dict[str, float],
) -> dict[str, Any]:
    labelled_total = int(counts["visible_labelled_annotations"])
    matched = int(counts["recognised_matched_labelled_annotations"])
    eligible = int(ordering["eligible_groups"])
    fully_detected = int(ordering["fully_detected_groups"])
    dr_true_positive = int(counts["dr_true_positive"])
    dr_false_positive = int(counts["dr_false_positive"])
    dr_false_negative = int(counts["dr_false_negative"])
    return {
        "counts": {
            "plates": int(counts["plates"]),
            "visible_annotations": int(counts["visible_annotations"]),
            "visible_labelled_annotations": labelled_total,
            "detector_predictions": int(counts["detector_predictions"]),
            "matched_visible_annotations": int(counts["matched_visible_annotations"]),
            "matched_labelled_annotations": int(counts["matched_labelled_annotations"]),
            "recognised_matched_labelled_annotations": matched,
            "crop_failures": int(counts["crop_failures"]),
            "detector_prediction_crop_failures": int(counts["detector_prediction_crop_failures"]),
        },
        "coverage_audit": {
            "expected_selected_plates": int(expected_plates),
            "processed_plates": int(counts["plates"]),
            "processed_plate_coverage": _finite_rate(int(counts["plates"]), int(expected_plates)),
            "labelled_visible_targets": labelled_total,
            "labelled_iou50_matched_targets": int(counts["matched_labelled_annotations"]),
            "labelled_iou50_matched_target_coverage": _finite_rate(
                int(counts["matched_labelled_annotations"]), labelled_total
            ),
        },
        "timing_seconds": {
            "total": float(timing["total"]),
            "detector": float(timing["detector"]),
            "recognition_and_dr_f1": float(timing["recognition_and_dr_f1"]),
            "reading_order": float(timing["reading_order"]),
            "mean_total_per_plate": _finite_rate(float(timing["total"]), int(counts["plates"])),
        },
        "recognition": {
            "visible_iou50_match_recall": _finite_rate(
                int(counts["matched_visible_annotations"]), int(counts["visible_annotations"])
            ),
            "labelled_iou50_match_recall": _finite_rate(
                int(counts["matched_labelled_annotations"]), labelled_total
            ),
            "matched_detected_box_top1": _finite_rate(int(counts["detected_crop_top1"]), matched),
            "matched_detected_box_top5": _finite_rate(int(counts["detected_crop_top5"]), matched),
            "matched_ground_truth_box_top1": _finite_rate(int(counts["oracle_crop_top1"]), matched),
            "matched_ground_truth_box_top5": _finite_rate(int(counts["oracle_crop_top5"]), matched),
            "detected_box_vs_ground_truth_box_top1_delta_pp": (
                100.0 * (counts["detected_crop_top1"] - counts["oracle_crop_top1"]) / matched
                if matched
                else None
            ),
            "detector_to_recognition_top1_recovery": _finite_rate(int(counts["detected_crop_top1"]), labelled_total),
            "detector_to_recognition_top5_recovery": _finite_rate(int(counts["detected_crop_top5"]), labelled_total),
        },
        "joint_detection_recognition": {
            "definition": "DR-F1@0.50: a TP requires one score-ranked detector candidate with IoU >= 0.50 and matching Top-1 canonical label; other evaluated candidates are FP and unmatched labelled visible targets are FN. Predictions that only overlap an unlabelled visible annotation are ignored.",
            "true_positive": dr_true_positive,
            "false_positive": dr_false_positive,
            "false_negative": dr_false_negative,
            "ignored_prediction": int(counts["dr_ignored_prediction"]),
            "precision_at_iou_0_50": _finite_rate(
                dr_true_positive, dr_true_positive + dr_false_positive
            ),
            "recall_at_iou_0_50": _finite_rate(
                dr_true_positive, dr_true_positive + dr_false_negative
            ),
            "dr_f1_at_iou_0_50": _finite_rate(
                2 * dr_true_positive,
                2 * dr_true_positive + dr_false_positive + dr_false_negative,
            ),
        },
        "reading_order": {
            "eligible_groups": eligible,
            "fully_detected_groups": fully_detected,
            "fully_detected_group_rate": _finite_rate(fully_detected, eligible),
            "gold_token_pair_accuracy_conditioned_on_full_detection": _finite_rate(
                int(ordering["gold_token_pair_correct"]), int(ordering["gold_token_pair_total"])
            ),
            "gold_token_exact_content_conditioned_on_full_detection": _finite_rate(
                int(ordering["gold_token_exact"]), fully_detected
            ),
            "detected_crop_to_order_pair_accuracy_conditioned_on_full_detection": _finite_rate(
                int(ordering["detected_pair_correct"]), int(ordering["detected_pair_total"])
            ),
            "detected_crop_to_order_exact_content_conditioned_on_full_detection": _finite_rate(
                int(ordering["detected_exact"]), fully_detected
            ),
            "strict_detection_to_recognition_to_order_exact_over_eligible_groups": _finite_rate(
                int(ordering["strict_pipeline_exact"]), eligible
            ),
            "strict_detection_to_recognition_to_order_exact_conditioned_on_full_detection": _finite_rate(
                int(ordering["strict_pipeline_exact"]), fully_detected
            ),
            "eligibility_audit": {key: int(value) for key, value in sorted(group_audit.items())},
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_started = perf_counter()
    if args.max_plates < 0:
        raise ValueError("--max-plates must be non-negative")
    if not 0.0 <= args.matching_iou <= 1.0:
        raise ValueError("--matching-iou must be in [0, 1]")
    if args.top_k < 5:
        raise ValueError("--top-k must be at least 5 so that Top-5 metrics remain defined")
    if args.detector_top_k is not None and args.detector_top_k <= 0:
        raise ValueError("--detector-top-k must be positive")
    if args.detector_nms_threshold is not None and not 0.0 <= args.detector_nms_threshold <= 1.0:
        raise ValueError("--detector-nms-threshold must be in [0, 1]")
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    detector_path = resolve_project_path(args.detector_checkpoint)
    vision_path = resolve_project_path(args.vision_checkpoint)
    sequence_path = resolve_project_path(args.sequence_checkpoint)
    detector, detector_state = load_detector(detector_path, device)
    vision, vision_vocab = _load_vision(vision_path, device)
    sequence, sequence_vocab = _load_sequence(sequence_path, device)
    if vision_vocab["labels"] != sequence_vocab["labels"]:
        raise ValueError("vision and sequence checkpoints use different vocabularies")
    detector_config = dict(detector_state["model_config"])
    score_threshold = (
        float(args.score_threshold) if args.score_threshold is not None else float(detector_config["score_threshold"])
    )
    decoder_config = {
        **detector_config,
        "score_threshold": score_threshold,
        "top_k": int(args.detector_top_k) if args.detector_top_k is not None else int(detector_config["top_k"]),
        "nms_threshold": (
            float(args.detector_nms_threshold)
            if args.detector_nms_threshold is not None
            else float(detector_config["nms_threshold"])
        ),
    }
    raw_dir = resolve_project_path(args.raw_dir)
    label_path = raw_dir / "标签.json"
    records: list[dict[str, Any]] = json.loads(label_path.read_text(encoding="utf-8"))
    assignments, _ = _assign_splits(records)
    validation_records = [record for record, split in zip(records, assignments) if split == "val"]
    record_by_name = {str(record["RubbingName"]): record for record in validation_records}
    selected_names = deterministic_plate_subset(list(record_by_name), args.max_plates, args.subset_seed)
    selected_records = [record_by_name[name] for name in selected_names]
    sources = [args.source] if args.source != "both" else ["rubbing", "facsimile"]

    source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    source_ordering: dict[str, Counter[str]] = defaultdict(Counter)
    source_group_audit: dict[str, Counter[str]] = defaultdict(Counter)
    source_timing: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    per_plate: list[dict[str, Any]] = []
    for record_index, record in enumerate(selected_records, start=1):
        for source in sources:
            source_directory, source_field = SOURCE_FIELDS[source]
            image_path = raw_dir / source_directory / Path(record[source_field]).name
            local, counts, ordering, group_audit = evaluate_one_plate(
                record=record,
                source=source,
                image_path=image_path,
                detector=detector,
                detector_config=decoder_config,
                vision=vision,
                vision_vocab=vision_vocab,
                sequence=sequence,
                device=device,
                score_threshold=score_threshold,
                top_k=args.top_k,
                iou_threshold=args.matching_iou,
            )
            source_counts[source].update(counts)
            source_ordering[source].update(ordering)
            source_group_audit[source].update(group_audit)
            for key, value in local["timing_seconds"].items():
                source_timing[source][key] += float(value)
            per_plate.append(local)
        print(
            json.dumps(
                {"validation_plate": record_index, "of": len(selected_records), "plate": record["RubbingName"]},
                ensure_ascii=False,
            ),
            flush=True,
        )

    result = {
        "protocol": {
            "name": "validation-only full-plate detector-to-crop-to-recognition-to-order smoke evaluation",
            "split": "val",
            "status": "development smoke / diagnostic; not a held-out end-to-end test",
            "group_assignment": "ground-truth group membership is applied only after IoU matching to make order scoring measurable; no detector group-segmentation claim is made",
            "visible_annotation_rule": "SeatFont=1 missing-character placeholders and malformed/duplicate boxes are excluded before detector matching",
            "matching": "score-ranked greedy one-to-one matching against every visible annotation, then labelled matches are recognised",
            "matching_iou": float(args.matching_iou),
            "decoder": {
                "provenance": str(args.decoder_provenance),
                "score_threshold": float(decoder_config["score_threshold"]),
                "top_k": int(decoder_config["top_k"]),
                "nms_threshold": float(decoder_config["nms_threshold"]),
            },
            "cropping": "predicted and ground-truth xyxy boxes are converted to xywh then use the deployed _square_crop with its 12% context margin",
            "sorter_input": "matched detections inside an oracle-assigned group are supplied in detector-score order; the sorter only receives canonical token IDs",
            "reading_group_filter": "all group characters must be visible, unique-box, labelled in the checkpoint vocabulary, unambiguous, and length 2..sequence max_length",
            "completion": "not evaluated here; SeatFont=1 denotes an unobserved placeholder and requires a separately pre-registered missing-glyph protocol",
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "inputs": {
            "runner_script": str(Path(__file__).resolve()),
            "runner_script_sha256": file_sha256(Path(__file__)),
            "matching_helper": str((ROOT / "src" / "oracle_vit" / "end_to_end_plate.py").resolve()),
            "matching_helper_sha256": file_sha256(ROOT / "src" / "oracle_vit" / "end_to_end_plate.py"),
            "raw_label_json": str(label_path.resolve()),
            "raw_label_json_sha256": file_sha256(label_path),
            "detector_checkpoint": str(detector_path.resolve()),
            "detector_checkpoint_sha256": file_sha256(detector_path),
            "detector_seed": int(detector_state["seed"]),
            "detector_best_epoch": int(detector_state["best_epoch"]),
            "detector_config_sha256": str(detector_state.get("config_sha256", "")),
            "vision_checkpoint": str(vision_path.resolve()),
            "vision_checkpoint_sha256": file_sha256(vision_path),
            "sequence_checkpoint": str(sequence_path.resolve()),
            "sequence_checkpoint_sha256": file_sha256(sequence_path),
        },
        "subset": {
            "available_validation_plates": len(validation_records),
            "selection": "SHA-256 rank of '{subset_seed}:{RubbingName}', no data-dependent selection",
            "subset_seed": int(args.subset_seed),
            "max_plates": int(args.max_plates),
            "selected_plates": selected_names,
        },
        "sources": {
            source: summarise_source(
                source_counts[source],
                source_ordering[source],
                source_group_audit[source],
                len(selected_records),
                source_timing[source],
            )
            for source in sources
        },
        "runtime_seconds": {
            "wall_clock_total": perf_counter() - run_started,
            "processed_plate_source_pairs": len(selected_records) * len(sources),
        },
    }
    output_path = _safe_output_path(args.output)
    atomic_json_dump(output_path, result)
    if args.per_plate_output:
        per_plate_path = _safe_output_path(args.per_plate_output)
        atomic_json_dump(
            per_plate_path,
            {"protocol": result["protocol"], "subset": result["subset"], "records": per_plate},
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validation-only full-plate detector -> crop -> recognizer -> reading-order smoke evaluation"
    )
    parser.add_argument("--raw-dir", default="甲骨文多模态数据集")
    parser.add_argument(
        "--detector-checkpoint",
        default="artifacts/experiments/plate_detector_center_20260907/seed_20260931/detector_best.pt",
    )
    parser.add_argument("--vision-checkpoint", default="artifacts/checkpoints/vision_best.pt")
    parser.add_argument("--sequence-checkpoint", default="artifacts/checkpoints/sequence_best.pt")
    parser.add_argument("--source", choices=("both", "rubbing", "facsimile"), default="both")
    parser.add_argument("--max-plates", type=int, default=12, help="0 evaluates all validation plates; default is a fixed smoke subset")
    parser.add_argument("--subset-seed", type=int, default=20260907)
    parser.add_argument("--matching-iou", type=float, default=0.50)
    parser.add_argument("--score-threshold", type=float, help="defaults to the frozen detector checkpoint configuration")
    parser.add_argument("--detector-top-k", type=int, help="optional decoder override; validation-selected settings are development-only")
    parser.add_argument("--detector-nms-threshold", type=float, help="optional decoder override; validation-selected settings are development-only")
    parser.add_argument(
        "--decoder-provenance",
        default="frozen_detector_checkpoint_config",
        help="auditable label for the decoder condition, e.g. validation_selected_sweep_20260907",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--output",
        default="artifacts/experiments/end_to_end_plate_smoke_20260907/validation_smoke_summary.json",
    )
    parser.add_argument(
        "--per-plate-output",
        default="artifacts/experiments/end_to_end_plate_smoke_20260907/validation_smoke_per_plate.json",
    )
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
