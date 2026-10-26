"""Export auditable per-target detector/crop-recognizer evidence.

This is an additive evaluator for the crop-shift alignment experiment.  It does
not alter the frozen plate evaluator or any prepared dataset.  Unlike the
existing per-plate diagnostic, it emits one row for every visible annotation,
including detector misses, and evaluates the corresponding ground-truth crop
even when no detector box was matched.

Validation is the default.  Held-out test access requires both ``--split test``
and the explicit ``--allow-test`` acknowledgement so a development command
cannot silently consume the test partition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import torch
from PIL import Image, ImageOps
from torchvision.transforms import functional as TF

from oracle_vit.data import _square_crop
from oracle_vit.end_to_end_plate import deterministic_plate_subset, greedy_iou_matches, xyxy_to_xywh
from oracle_vit.infer import _load_vision, recognize
from oracle_vit.plate_detection import SOURCE_FIELDS
from oracle_vit.plate_inference_strong import load_detector_any_family, predict_plate_detections
from oracle_vit.prepare import _assign_splits
from oracle_vit.utils import atomic_json_dump, file_sha256, resolve_project_path, write_jsonl

try:  # Package import in tests; direct import when executed as ``python scripts/...``.
    from scripts import evaluate_plate_to_order_smoke as plate_pipeline
except ModuleNotFoundError:  # pragma: no cover - exercised by the command-line entry point
    import evaluate_plate_to_order_smoke as plate_pipeline


ROOT = Path(__file__).resolve().parents[1]


def stable_target_id(plate: str, source: str, group_index: int, char_index: int) -> str:
    """Return a detector/model-independent identity for one source-specific target."""
    identity = json.dumps(
        [str(plate), str(source), int(group_index), int(char_index)],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return "target_sha256:" + hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _candidate_fields(
    recognition: dict[str, Any] | None,
    target_label: str | None,
    evaluation_eligible: bool,
) -> dict[str, Any]:
    candidates = list(recognition.get("candidates", [])) if recognition is not None else []
    labels = [str(candidate["label"]) for candidate in candidates[:5]]
    class_ids = [int(candidate["class_id"]) for candidate in candidates[:5]]
    probabilities = [float(candidate["probability"]) for candidate in candidates[:5]]
    return {
        "top1_label": labels[0] if labels else None,
        "top1_class_id": class_ids[0] if class_ids else None,
        "top1_probability": probabilities[0] if probabilities else None,
        "top5_labels": labels,
        "top5_class_ids": class_ids,
        "top5_probabilities": probabilities,
        "top1": int(labels[0] == target_label) if labels and evaluation_eligible else None,
        "top5": int(str(target_label) in labels) if labels and evaluation_eligible else None,
    }


def assemble_target_rows(
    *,
    plate: str,
    source: str,
    detector_label: str,
    image_width: int,
    image_height: int,
    annotations: list[dict[str, Any]],
    prediction_boxes: torch.Tensor,
    prediction_scores: torch.Tensor,
    matches: Iterable[tuple[int, int, float]],
    predicted_recognition: dict[int, dict[str, Any]],
    gold_recognition: dict[int, dict[str, Any]],
    vocabulary: dict[str, Any],
    class_counts: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Join detector matches and both crop-recognition paths by gold target."""
    match_by_target = {
        int(target_index): (int(prediction_index), float(overlap))
        for prediction_index, target_index, overlap in matches
    }
    label_to_class = {str(label): int(class_id) for label, class_id in vocabulary["label_to_class"].items()}
    if class_counts is not None and len(class_counts) != len(vocabulary["labels"]):
        raise ValueError("class_counts must align with the recognition vocabulary")

    rows: list[dict[str, Any]] = []
    image_area = max(1.0, float(image_width * image_height))
    for target_index, annotation in enumerate(annotations):
        raw_label = annotation.get("label")
        target_label = str(raw_label) if isinstance(raw_label, str) else None
        target_class_id = label_to_class.get(target_label) if target_label is not None else None
        evaluation_eligible = target_class_id is not None
        matched = target_index in match_by_target
        prediction_index: int | None = None
        overlap: float | None = None
        predicted_box: list[float] | None = None
        detector_score: float | None = None
        predicted_result: dict[str, Any] | None = None
        if matched:
            prediction_index, overlap = match_by_target[target_index]
            predicted_box = [float(value) for value in prediction_boxes[prediction_index].tolist()]
            detector_score = float(prediction_scores[prediction_index])
            predicted_result = predicted_recognition.get(prediction_index)

        gold_box = [float(value) for value in annotation["box_xyxy"]]
        width = max(0.0, gold_box[2] - gold_box[0])
        height = max(0.0, gold_box[3] - gold_box[1])
        predicted_fields = _candidate_fields(predicted_result, target_label, evaluation_eligible)
        gold_fields = _candidate_fields(gold_recognition.get(target_index), target_label, evaluation_eligible)
        train_count = int(class_counts[target_class_id]) if class_counts is not None and target_class_id is not None else None
        rows.append(
            {
                "target_id": stable_target_id(
                    plate,
                    source,
                    int(annotation["group_index"]),
                    int(annotation["char_index"]),
                ),
                "plate": str(plate),
                "source": str(source),
                "detector_label": str(detector_label),
                "annotation_index": int(annotation.get("annotation_index", target_index)),
                "group_index": int(annotation["group_index"]),
                "char_index": int(annotation["char_index"]),
                "target_label": target_label,
                "target_class_id": target_class_id,
                "target_train_count": train_count,
                "evaluation_eligible": bool(evaluation_eligible),
                "gold_box_xyxy": gold_box,
                "gold_box_width": width,
                "gold_box_height": height,
                "gold_box_area": width * height,
                "gold_box_area_fraction": width * height / image_area,
                "matched_at_iou": bool(matched),
                "matched_status": "matched" if matched else "unmatched",
                "matched_prediction_index": prediction_index,
                "predicted_box_xyxy": predicted_box,
                "matching_iou": overlap,
                "detector_score": detector_score,
                "predicted_crop_recognized": predicted_result is not None,
                "predicted_crop_top1_label": predicted_fields["top1_label"],
                "predicted_crop_top1_class_id": predicted_fields["top1_class_id"],
                "predicted_crop_top1_probability": predicted_fields["top1_probability"],
                "predicted_crop_top5_labels": predicted_fields["top5_labels"],
                "predicted_crop_top5_class_ids": predicted_fields["top5_class_ids"],
                "predicted_crop_top5_probabilities": predicted_fields["top5_probabilities"],
                "predicted_crop_top1": predicted_fields["top1"],
                "predicted_crop_top5": predicted_fields["top5"],
                "gold_crop_recognized": target_index in gold_recognition,
                "gold_crop_top1_label": gold_fields["top1_label"],
                "gold_crop_top1_class_id": gold_fields["top1_class_id"],
                "gold_crop_top1_probability": gold_fields["top1_probability"],
                "gold_crop_top5_labels": gold_fields["top5_labels"],
                "gold_crop_top5_class_ids": gold_fields["top5_class_ids"],
                "gold_crop_top5_probabilities": gold_fields["top5_probabilities"],
                "gold_crop_top1": gold_fields["top1"],
                "gold_crop_top5": gold_fields["top5"],
            }
        )
    return rows


def summarise_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Return count-based metrics without dropping unmatched or failed crops."""
    materialized = list(rows)
    eligible = [row for row in materialized if row["evaluation_eligible"]]
    matched = [row for row in eligible if row["matched_at_iou"]]
    predicted = [row for row in matched if row["predicted_crop_top1"] is not None]
    gold = [row for row in eligible if row["gold_crop_top1"] is not None]

    def rate(selected: list[dict[str, Any]], key: str) -> float | None:
        return sum(int(row[key]) for row in selected) / len(selected) if selected else None

    return {
        "visible_targets": len(materialized),
        "in_vocabulary_targets": len(eligible),
        "matched_in_vocabulary_targets": len(matched),
        "matched_target_recall": len(matched) / len(eligible) if eligible else None,
        "predicted_crop_evaluated_targets": len(predicted),
        "predicted_crop_top1_conditioned_on_evaluated_match": rate(predicted, "predicted_crop_top1"),
        "predicted_crop_top5_conditioned_on_evaluated_match": rate(predicted, "predicted_crop_top5"),
        "gold_crop_evaluated_targets": len(gold),
        "gold_crop_top1": rate(gold, "gold_crop_top1"),
        "gold_crop_top5": rate(gold, "gold_crop_top5"),
        "detector_to_recognition_top1_recovery": (
            sum(int(row["predicted_crop_top1"] or 0) for row in matched) / len(eligible) if eligible else None
        ),
        "detector_to_recognition_top5_recovery": (
            sum(int(row["predicted_crop_top5"] or 0) for row in matched) / len(eligible) if eligible else None
        ),
    }


def _safe_output_path(value: str, overwrite: bool) -> Path:
    path = resolve_project_path(value).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"output must remain inside {ROOT}: {path}") from error
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing output without --overwrite: {path}")
    return path


def _validate_access(split: str, allow_test: bool) -> None:
    if split == "test" and not allow_test:
        raise PermissionError("held-out test access requires the explicit --allow-test acknowledgement")


def _recognize_jobs(
    model: torch.nn.Module,
    vocabulary: dict[str, Any],
    jobs: list[tuple[int, Image.Image]],
    device: torch.device,
    batch_size: int,
) -> dict[int, dict[str, Any]]:
    results: dict[int, dict[str, Any]] = {}
    for offset in range(0, len(jobs), batch_size):
        chunk = jobs[offset : offset + batch_size]
        _, recognition = recognize(
            model,
            vocabulary,
            [image for _, image in chunk],
            device,
            top_k=5,
            robust_tta=False,
        )
        results.update({key: row for (key, _), row in zip(chunk, recognition)})
    return results


def _load_class_counts(path: Path | None, vocabulary: dict[str, Any]) -> tuple[list[int] | None, str | None]:
    if path is None:
        return None, None
    payload = json.loads(path.read_text(encoding="utf-8"))
    counts = payload.get("counts") if isinstance(payload, dict) else payload
    if not isinstance(counts, list) or any(not isinstance(value, int) or value < 0 for value in counts):
        raise ValueError("class-counts must contain a non-negative integer 'counts' list")
    if len(counts) != len(vocabulary["labels"]):
        raise ValueError("class-counts length differs from the recognition vocabulary")
    return [int(value) for value in counts], file_sha256(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    _validate_access(args.split, bool(args.allow_test))
    if args.max_plates < 0:
        raise ValueError("--max-plates must be non-negative")
    if not 0.0 <= args.matching_iou <= 1.0:
        raise ValueError("--matching-iou must be in [0, 1]")
    if args.recognition_batch_size <= 0:
        raise ValueError("--recognition-batch-size must be positive")
    rows_output = _safe_output_path(args.rows_output, bool(args.overwrite))
    summary_output = _safe_output_path(args.summary_output, bool(args.overwrite))
    if rows_output == summary_output:
        raise ValueError("rows and summary outputs must differ")

    started = perf_counter()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    detector_path = resolve_project_path(args.detector_checkpoint)
    vision_path = resolve_project_path(args.vision_checkpoint)
    detector, detector_state = load_detector_any_family(detector_path, device)
    vision, vocabulary = _load_vision(vision_path, device)
    class_counts_path = resolve_project_path(args.class_counts) if args.class_counts else None
    class_counts, class_counts_sha256 = _load_class_counts(class_counts_path, vocabulary)

    detector_config = dict(detector_state["model_config"])
    score_threshold = (
        float(args.score_threshold)
        if args.score_threshold is not None
        else float(detector_config["score_threshold"])
    )
    detector_top_k = int(args.detector_top_k) if args.detector_top_k is not None else int(detector_config["top_k"])
    detector_nms = (
        float(args.detector_nms_threshold)
        if args.detector_nms_threshold is not None
        else float(detector_config["nms_threshold"])
    )
    decoder = {
        "input_size": int(detector_config["input_size"]),
        "score_threshold": score_threshold,
        "top_k": detector_top_k,
        "nms_threshold": detector_nms,
    }

    raw_dir = resolve_project_path(args.raw_dir)
    label_path = raw_dir / "标签.json"
    records: list[dict[str, Any]] = json.loads(label_path.read_text(encoding="utf-8"))
    assignments, _ = _assign_splits(records)
    split_records = [record for record, split in zip(records, assignments) if split == args.split]
    record_by_name = {str(record["RubbingName"]): record for record in split_records}
    selected_names = deterministic_plate_subset(list(record_by_name), args.max_plates, args.subset_seed)
    sources = [args.source] if args.source != "both" else ["rubbing", "facsimile"]

    all_rows: list[dict[str, Any]] = []
    crop_failures: Counter[str] = Counter()
    for plate_number, plate in enumerate(selected_names, start=1):
        record = record_by_name[plate]
        for source in sources:
            directory, field = SOURCE_FIELDS[source]
            image_path = raw_dir / directory / Path(record[field]).name
            with Image.open(image_path) as handle:
                image = ImageOps.exif_transpose(handle).convert("RGB")
            annotations, _, _, _ = plate_pipeline.collect_visible_annotations(record, image.width, image.height)
            target_boxes = torch.tensor(
                [annotation["box_xyxy"] for annotation in annotations], dtype=torch.float32
            ).reshape(-1, 4)
            decoded = predict_plate_detections(
                detector,
                TF.pil_to_tensor(image).float().div(255.0),
                **decoder,
            )
            matches = greedy_iou_matches(
                decoded["boxes"], decoded["scores"], target_boxes, args.matching_iou
            )

            gold_jobs: list[tuple[int, Image.Image]] = []
            for target_index, annotation in enumerate(annotations):
                try:
                    gold_jobs.append(
                        (target_index, _square_crop(image, xyxy_to_xywh(annotation["box_xyxy"])))
                    )
                except ValueError:
                    crop_failures["gold"] += 1
            predicted_jobs: list[tuple[int, Image.Image]] = []
            for prediction_index, _, _ in matches:
                try:
                    predicted_jobs.append(
                        (
                            int(prediction_index),
                            _square_crop(
                                image,
                                xyxy_to_xywh(decoded["boxes"][prediction_index].tolist()),
                            ),
                        )
                    )
                except ValueError:
                    crop_failures["predicted"] += 1

            gold_recognition = _recognize_jobs(
                vision, vocabulary, gold_jobs, device, args.recognition_batch_size
            )
            predicted_recognition = _recognize_jobs(
                vision, vocabulary, predicted_jobs, device, args.recognition_batch_size
            )
            all_rows.extend(
                assemble_target_rows(
                    plate=plate,
                    source=source,
                    detector_label=args.detector_label,
                    image_width=image.width,
                    image_height=image.height,
                    annotations=annotations,
                    prediction_boxes=decoded["boxes"],
                    prediction_scores=decoded["scores"],
                    matches=matches,
                    predicted_recognition=predicted_recognition,
                    gold_recognition=gold_recognition,
                    vocabulary=vocabulary,
                    class_counts=class_counts,
                )
            )
        print(
            json.dumps(
                {"split": args.split, "plate": plate_number, "of": len(selected_names), "name": plate},
                ensure_ascii=False,
            ),
            flush=True,
        )

    write_jsonl(rows_output, all_rows)
    by_source = {
        source: summarise_rows(row for row in all_rows if row["source"] == source)
        for source in sources
    }
    summary = {
        "schema_version": 1,
        "protocol": {
            "name": "crop-shift per-target detector/crop-recognition evidence",
            "split": args.split,
            "test_access": "explicitly_acknowledged" if args.split == "test" else "not_accessed",
            "matching": "score-ranked greedy one-to-one IoU matching over every visible annotation",
            "matching_iou": float(args.matching_iou),
            "target_rows": "one row per visible target; detector misses and out-of-vocabulary targets are retained",
            "crop_rule": "deployed square crop with 12% context margin for both predicted and gold boxes",
            "top_k": 5,
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "detector_label": str(args.detector_label),
        "decoder": decoder,
        "subset": {
            "available_plates": len(split_records),
            "selected_plate_count": len(selected_names),
            "selection": "SHA-256 rank of '{subset_seed}:{RubbingName}'",
            "subset_seed": int(args.subset_seed),
            "max_plates": int(args.max_plates),
            "selected_plates": selected_names,
        },
        "metrics": {"overall": summarise_rows(all_rows), "by_source": by_source},
        "crop_failures": {key: int(value) for key, value in sorted(crop_failures.items())},
        "inputs": {
            "runner_script": str(Path(__file__).resolve()),
            "runner_script_sha256": file_sha256(Path(__file__)),
            "raw_labels": str(label_path.resolve()),
            "raw_labels_sha256": file_sha256(label_path),
            "detector_checkpoint": str(detector_path.resolve()),
            "detector_checkpoint_sha256": file_sha256(detector_path),
            "detector_architecture": str(detector_config.get("architecture")),
            "vision_checkpoint": str(vision_path.resolve()),
            "vision_checkpoint_sha256": file_sha256(vision_path),
            "class_counts": str(class_counts_path.resolve()) if class_counts_path else None,
            "class_counts_sha256": class_counts_sha256,
        },
        "outputs": {"target_rows_jsonl": str(rows_output)},
        "runtime_seconds": perf_counter() - started,
    }
    atomic_json_dump(summary_output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export per-target crop-shift evaluation rows")
    parser.add_argument("--raw-dir", default="甲骨文多模态数据集")
    parser.add_argument("--detector-checkpoint", required=True)
    parser.add_argument("--vision-checkpoint", required=True)
    parser.add_argument("--detector-label", required=True)
    parser.add_argument("--class-counts", help="optional JSON counts list aligned with the recognition vocabulary")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--source", choices=("both", "rubbing", "facsimile"), default="both")
    parser.add_argument("--max-plates", type=int, default=12, help="0 evaluates the entire selected split")
    parser.add_argument("--subset-seed", type=int, default=20261009)
    parser.add_argument("--matching-iou", type=float, default=0.50)
    parser.add_argument("--score-threshold", type=float)
    parser.add_argument("--detector-top-k", type=int)
    parser.add_argument("--detector-nms-threshold", type=float)
    parser.add_argument("--recognition-batch-size", type=int, default=128)
    parser.add_argument("--rows-output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
