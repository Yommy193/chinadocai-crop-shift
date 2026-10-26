"""Plate-level character detection data and metrics.

The recognition benchmark is based on already cropped characters.  This module
keeps the plate-level detection task separate and auditable: it reuses the
existing plate split, treats every annotated rectangle as the single generic
``character`` class, and never needs a character-ID label for detector
training.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from .prepare import _assign_splits


SOURCE_FIELDS = {
    "rubbing": ("拓片", "Rubbing"),
    "facsimile": ("摹本", "Facsimile"),
}


def require_test_access_allowed(config: dict[str, Any]) -> None:
    """Reject configurations explicitly reserved for development-only runs.

    Validation-only detector pilots are intentionally allowed to share the
    training code with final runs, but they must never be passable to the test
    evaluator merely because their checkpoint SHA matches their configuration.
    Existing historical final configurations omit this marker and remain
    readable; new development configurations should set ``test_access`` to
    ``"prohibited"``.
    """
    access = str(config.get("test_access", "")).strip().casefold()
    protocol = str(config.get("protocol", "")).casefold()
    if access == "prohibited" or "development-only" in protocol:
        raise PermissionError("this detector configuration is development-only; test dataset construction is prohibited")


def is_visible_character_annotation(annotation: dict[str, Any]) -> bool:
    """A SeatFont=1 box denotes a missing-character placeholder, not ink to crop."""
    if not annotation.get("Position"):
        return False
    return int(annotation.get("SeatFont", 0) or 0) != 1


def sanitize_xywh_boxes(
    boxes: Iterable[Sequence[int | float]], image_width: int, image_height: int
) -> tuple[list[list[float]], dict[str, int]]:
    """Clip, validate, and de-duplicate annotation boxes in xywh format."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    clean: list[list[float]] = []
    seen: set[tuple[float, float, float, float]] = set()
    audit = {"invalid": 0, "clipped": 0, "duplicate": 0, "kept": 0}
    for raw in boxes:
        if len(raw) != 4:
            audit["invalid"] += 1
            continue
        try:
            x, y, width, height = (float(value) for value in raw)
        except (TypeError, ValueError):
            audit["invalid"] += 1
            continue
        if width <= 0 or height <= 0:
            audit["invalid"] += 1
            continue
        x1, y1, x2, y2 = x, y, x + width, y + height
        clipped = (max(0.0, x1), max(0.0, y1), min(float(image_width), x2), min(float(image_height), y2))
        if clipped != (x1, y1, x2, y2):
            audit["clipped"] += 1
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            audit["invalid"] += 1
            continue
        key = tuple(round(value, 5) for value in clipped)
        if key in seen:
            audit["duplicate"] += 1
            continue
        seen.add(key)
        clean.append(list(clipped))
        audit["kept"] += 1
    return clean, audit


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Return pairwise IoU for xyxy tensors without optional dependencies."""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=torch.float32)
    upper_left = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    lower_right = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    intersection = (lower_right - upper_left).clamp(min=0).prod(dim=-1)
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    return intersection / (area1[:, None] + area2[None, :] - intersection).clamp(min=torch.finfo(torch.float32).eps)


def average_precision_101(recall: torch.Tensor, precision: torch.Tensor) -> float:
    """COCO-style 101-point interpolated AP for a one-class detection task."""
    if recall.numel() == 0:
        return 0.0
    precision = torch.cummax(precision.flip(0), dim=0).values.flip(0)
    levels = torch.linspace(0.0, 1.0, 101)
    values = []
    for level in levels:
        eligible = precision[recall >= level]
        values.append(float(eligible.max()) if eligible.numel() else 0.0)
    return float(sum(values) / len(values))


def detection_metrics(
    predictions: Sequence[dict[str, torch.Tensor]],
    targets: Sequence[dict[str, torch.Tensor]],
    iou_thresholds: Sequence[float] = (0.5, 0.75),
    max_detections: int = 100,
) -> dict[str, Any]:
    """Compute one-class AP and recall from prediction/ground-truth lists.

    Scores are globally ranked across images; one ground-truth box can match
    only one prediction at every IoU threshold.  It is intentionally explicit
    rather than relying on a package whose evaluation details might vary.
    """
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have the same length")
    if max_detections <= 0:
        raise ValueError("max_detections must be positive")
    gt_boxes = [target["boxes"].detach().cpu().numpy().astype(np.float64, copy=False) for target in targets]
    prediction_boxes = [prediction["boxes"].detach().cpu().numpy().astype(np.float64, copy=False)[:max_detections] for prediction in predictions]
    prediction_scores = [prediction["scores"].detach().cpu().numpy().astype(np.float64, copy=False)[:max_detections] for prediction in predictions]
    total_ground_truth = sum(len(boxes) for boxes in gt_boxes)
    metrics: dict[str, Any] = {
        "images": len(targets),
        "ground_truth_boxes": total_ground_truth,
        "detections_considered": int(sum(len(scores) for scores in prediction_scores)),
        "max_detections_per_image": max_detections,
        "by_iou": {},
    }
    ap_values = []
    for threshold in iou_thresholds:
        score_parts, true_positive_parts = [], []
        for predicted_boxes, scores, truth in zip(prediction_boxes, prediction_scores, gt_boxes):
            order = np.argsort(-scores, kind="mergesort")
            ordered_boxes, ordered_scores = predicted_boxes[order], scores[order]
            matched = np.zeros(len(truth), dtype=bool)
            true_positive = np.zeros(len(ordered_scores), dtype=np.float64)
            if len(truth) and len(ordered_boxes):
                upper_left = np.maximum(ordered_boxes[:, None, :2], truth[None, :, :2])
                lower_right = np.minimum(ordered_boxes[:, None, 2:], truth[None, :, 2:])
                intersection = np.clip(lower_right - upper_left, a_min=0, a_max=None).prod(axis=-1)
                predicted_area = np.clip(ordered_boxes[:, 2] - ordered_boxes[:, 0], a_min=0, a_max=None) * np.clip(ordered_boxes[:, 3] - ordered_boxes[:, 1], a_min=0, a_max=None)
                truth_area = np.clip(truth[:, 2] - truth[:, 0], a_min=0, a_max=None) * np.clip(truth[:, 3] - truth[:, 1], a_min=0, a_max=None)
                overlaps = intersection / np.maximum(predicted_area[:, None] + truth_area[None, :] - intersection, np.finfo(np.float64).eps)
                for prediction_index, row in enumerate(overlaps):
                    available = np.where(~matched)[0]
                    if available.size:
                        candidate = available[int(np.argmax(row[available]))]
                        if row[candidate] >= threshold:
                            matched[candidate] = True
                            true_positive[prediction_index] = 1.0
            score_parts.append(ordered_scores)
            true_positive_parts.append(true_positive)
        scores = np.concatenate(score_parts) if score_parts else np.empty((0,), dtype=np.float64)
        true_positive = np.concatenate(true_positive_parts) if true_positive_parts else np.empty((0,), dtype=np.float64)
        if scores.size:
            order = np.argsort(-scores, kind="mergesort")
            true_positive = true_positive[order]
        false_positive = 1.0 - true_positive
        if total_ground_truth:
            tp = torch.from_numpy(true_positive.astype(np.float32, copy=False))
            fp = torch.from_numpy(false_positive.astype(np.float32, copy=False))
            recall = tp.cumsum(0) / total_ground_truth
            precision = tp.cumsum(0) / (tp.cumsum(0) + fp.cumsum(0)).clamp(min=1.0)
            ap = average_precision_101(recall, precision)
            final_recall = float(recall[-1]) if recall.numel() else 0.0
        else:
            ap, final_recall = 0.0, 0.0
        metrics["by_iou"][f"{threshold:.2f}"] = {"ap": ap, "recall": final_recall}
        ap_values.append(ap)
    metrics["mean_ap"] = float(sum(ap_values) / len(ap_values)) if ap_values else 0.0
    return metrics


class PlateDetectionDataset(Dataset[tuple[torch.Tensor, dict[str, torch.Tensor]]]):
    """One-class character boxes from full plates with an auditable plate split.

    ``records`` is an optional immutable, manifest-validated record cohort.  It
    exists so a provenance-disjoint outer-CV run can construct its datasets from
    its frozen subset rather than parse the full label corpus, which contains
    historical validation/test records that must stay out of that experiment.
    """

    def __init__(
        self,
        raw_dir: Path,
        split: str,
        source: str,
        plate_names: Iterable[str] | None = None,
        records: Sequence[dict[str, Any]] | None = None,
    ):
        allowed_splits = {"train", "val", "test", "inner_train", "inner_val", "outer_holdout"}
        if split not in allowed_splits:
            raise ValueError(f"unknown split: {split}")
        if source not in SOURCE_FIELDS:
            raise ValueError(f"unknown source: {source}")
        self.raw_dir = Path(raw_dir)
        source_directory, source_field = SOURCE_FIELDS[source]
        if records is None:
            loaded_records = json.loads((self.raw_dir / "标签.json").read_text(encoding="utf-8"))
            records_mode = "raw_labels"
        else:
            loaded_records = list(records)
            records_mode = "manifest_records_subset"
        if not isinstance(loaded_records, list) or not all(isinstance(record, dict) for record in loaded_records):
            raise ValueError("detector records must be a list of JSON objects")
        available_names = [str(record["RubbingName"]) for record in loaded_records]
        if len(available_names) != len(set(available_names)):
            raise ValueError("detector records contain duplicate RubbingName records")
        if plate_names is None:
            if records is not None:
                raise ValueError("manifest-backed detector records require explicit frozen plate membership")
            if split not in {"train", "val", "test"}:
                raise ValueError("inner/outer split names require explicit frozen plate membership")
            assignments, _ = _assign_splits(loaded_records)
            selected_records = [record for record, record_split in zip(loaded_records, assignments) if record_split == split]
            self.split_provenance = {"mode": "historical_assign_splits", "split": split}
        else:
            requested_names = [str(value) for value in plate_names]
            requested = set(requested_names)
            if not requested:
                raise ValueError("explicit plate_names must be non-empty")
            if len(requested_names) != len(requested):
                raise ValueError("explicit plate_names contain duplicates")
            available = set(available_names)
            missing = requested - available
            if missing:
                raise ValueError(f"explicit plate_names include {len(missing)} unknown plate(s)")
            selected_records = [record for record in loaded_records if str(record["RubbingName"]) in requested]
            if len(selected_records) != len(requested):
                raise AssertionError("explicit plate selection did not preserve one record per plate")
            self.split_provenance = {
                "mode": "explicit_plate_membership",
                "split_role": split,
                "plates": len(selected_records),
                "records_source": records_mode,
            }
        self.rows: list[dict[str, Any]] = []
        self.placeholder_boxes = 0
        for record in selected_records:
            filename = Path(record[source_field]).name
            image_path = self.raw_dir / source_directory / filename
            if not image_path.is_file():
                raise FileNotFoundError(f"missing {source} plate: {image_path}")
            annotations = [char for group in record["RecordUtilSentenceGroupVoList"] for char in group["RecordUtilOracleCharVoList"]]
            self.placeholder_boxes += sum(bool(char.get("Position")) and not is_visible_character_annotation(char) for char in annotations)
            boxes = [[int(part.strip()) for part in char["Position"].split(",")] for char in annotations if is_visible_character_annotation(char)]
            self.rows.append({"plate": record["RubbingName"], "image_path": image_path, "boxes_xywh": boxes})

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        row = self.rows[index]
        with Image.open(row["image_path"]) as handle:
            image = ImageOps.exif_transpose(handle).convert("RGB")
        boxes, _ = sanitize_xywh_boxes(row["boxes_xywh"], image.width, image.height)
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        target = {
            "boxes": boxes_tensor,
            "labels": torch.ones((len(boxes),), dtype=torch.int64),
            "image_id": torch.tensor([index]),
            "area": ((boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (boxes_tensor[:, 3] - boxes_tensor[:, 1])).clamp(min=0),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }
        return TF.pil_to_tensor(image).float().div(255.0), target

    def annotation_audit(self) -> dict[str, int]:
        """Audit all boxes with their actual image sizes; intentionally read-only."""
        summary = {"plates": len(self.rows), "placeholder_excluded": self.placeholder_boxes, "invalid": 0, "clipped": 0, "duplicate": 0, "kept": 0}
        for row in self.rows:
            with Image.open(row["image_path"]) as handle:
                boxes, audit = sanitize_xywh_boxes(row["boxes_xywh"], handle.width, handle.height)
            del boxes
            for key in ("invalid", "clipped", "duplicate", "kept"):
                summary[key] += audit[key]
        return summary


def collate_detection(batch: Sequence[tuple[torch.Tensor, dict[str, torch.Tensor]]]) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
    images, targets = zip(*batch)
    return list(images), list(targets)
