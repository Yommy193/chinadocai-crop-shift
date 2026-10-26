"""Build detector-crop and size-matched gold-crop training manifests (P3-4).

Protocol: ``docs/ROBUST_RECOGNITION_TRAINING_PREREGISTRATION_20260913.md``.

Runs the frozen detector over the training partition only, matches predicted boxes to
gold visible labelled boxes one-to-one at ``IoU >= 0.50``, and writes two derived manifests
under a project-local processed directory that junctions to the real image and reference
trees.  Nothing here reads the validation or test partitions' boxes for training.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from PIL import Image, ImageOps
from torchvision.transforms import functional as TF

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import evaluate_plate_to_order_smoke as pipeline  # noqa: E402

from oracle_vit.end_to_end_plate import greedy_iou_matches  # noqa: E402
from oracle_vit.plate_detection import SOURCE_FIELDS  # noqa: E402
from oracle_vit.plate_inference import predict_plate_detections  # noqa: E402
from oracle_vit.prepare import _assign_splits  # noqa: E402
from oracle_vit.utils import atomic_json_dump, file_sha256, resolve_project_path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("rubbing", "facsimile")


def junction(link: Path, target: Path) -> None:
    if link.exists():
        return
    os.system(f'cmd /c mklink /J "{link}" "{target}" >nul')


def main() -> None:
    parser = argparse.ArgumentParser(description="Build detector-crop training manifests for the P3-4 experiment")
    parser.add_argument("--config", default="configs/end_to_end_plate_confirmation_20260913.json")
    parser.add_argument("--detector-checkpoint", default="artifacts/experiments/plate_detector_center_20260907/seed_20260931/detector_best.pt")
    parser.add_argument("--source-processed", default="data/processed")
    parser.add_argument("--output-root", default="artifacts/experiments/robust_recognition_20260913")
    parser.add_argument("--padding", type=float, default=0.12)
    parser.add_argument("--matching-iou", type=float, default=0.50)
    parser.add_argument("--sampling-seed", type=int, default=20260913)
    parser.add_argument("--split", default="train")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    config = json.loads(resolve_project_path(args.config).read_text(encoding="utf-8"))
    source_processed = resolve_project_path(args.source_processed)
    output_root = ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    entries = [json.loads(line) for line in (source_processed / "crops.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    train_entries = [entry for entry in entries if entry.get("split") == args.split]
    gold_by_plate: dict[str, list[dict[str, Any]]] = {}
    for entry in train_entries:
        gold_by_plate.setdefault(str(entry["plate"]), []).append(entry)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    detector, detector_state = pipeline.load_detector(resolve_project_path(args.detector_checkpoint), device)
    detector_config = dict(detector_state["model_config"])
    detector_config["score_threshold"] = float(config.get("score_threshold", detector_config["score_threshold"]))

    raw_dir = resolve_project_path(str(config["raw_dir"]))
    records: list[dict[str, Any]] = json.loads((raw_dir / "标签.json").read_text(encoding="utf-8"))
    assignments, _ = _assign_splits(records)
    partition = {str(record["RubbingName"]): record for record, split in zip(records, assignments) if split == args.split}

    matched: list[dict[str, Any]] = []
    started = perf_counter()
    plates = sorted(gold_by_plate)
    for index, plate in enumerate(plates, start=1):
        record = partition.get(plate)
        if record is None:
            continue
        for source in SOURCES:
            directory, field = SOURCE_FIELDS[source]
            image_path = raw_dir / directory / Path(record[field]).name
            if not image_path.is_file():
                continue
            with Image.open(image_path) as handle:
                image = ImageOps.exif_transpose(handle).convert("RGB")
            decoded = predict_plate_detections(
                detector,
                TF.pil_to_tensor(image).float().div(255.0),
                input_size=int(detector_config["input_size"]),
                score_threshold=float(detector_config["score_threshold"]),
                top_k=int(detector_config["top_k"]),
                nms_threshold=float(detector_config["nms_threshold"]),
            )
            boxes = decoded["boxes"]
            if not len(boxes):
                continue
            xywh = torch.tensor([entry["box"] for entry in gold_by_plate[plate]], dtype=torch.float32)
            truth_xyxy = torch.stack([
                xywh[:, 0], xywh[:, 1], xywh[:, 0] + xywh[:, 2], xywh[:, 1] + xywh[:, 3]
            ], dim=1)
            for prediction, target, overlap in greedy_iou_matches(boxes, decoded["scores"], truth_xyxy, float(args.matching_iou)):
                entry = gold_by_plate[plate][int(target)]
                box = [float(value) for value in boxes[int(prediction)].tolist()]
                matched.append({
                    "id": f"det::{plate}::{source}::{prediction}",
                    "split": args.split,
                    "plate": plate,
                    "source": source,
                    "facsimile": entry["facsimile"],
                    "rubbing": entry["rubbing"],
                    "box": [int(round(value)) for value in (box[0], box[1], box[2] - box[0], box[3] - box[1])],
                    "label": entry["label"],
                    "class_id": entry["class_id"],
                    "sub_label": entry.get("sub_label", entry["label"]),
                    "mark": entry.get("mark", -1),
                    "seat_font": 0,
                    "matched_iou": round(float(overlap), 6),
                })
        if index % 500 == 0:
            print(json.dumps({"plates_done": index, "matched": len(matched)}, ensure_ascii=False), flush=True)
    del detector
    if device.type == "cuda":
        torch.cuda.empty_cache()

    rng = random.Random(int(args.sampling_seed))
    control: list[dict[str, Any]] = []
    for plate, golds in sorted(gold_by_plate.items()):
        control.extend(golds)
    rng.shuffle(control)
    control = control[: len(matched)]
    control_entries = [
        {**entry, "id": f"gold::{entry['id']}", "split": args.split, "source": "gold"} for entry in control
    ]

    for arm, arm_entries in (("detected_crops", matched), ("gold_control", control_entries)):
        arm_dir = output_root / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        junction(arm_dir / "images", source_processed / "images")
        junction(arm_dir / "glyph_reference_full", source_processed / "glyph_reference_full")
        (arm_dir / "vocab.json").write_text((source_processed / "vocab.json").read_text(encoding="utf-8"), encoding="utf-8")
        written = 0
        with (arm_dir / "crops.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for entry in entries:
                if entry.get("split") == args.split:
                    continue
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            for entry in arm_entries:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
                written += 1
        summary = {
            "arm": arm,
            "train_entries_written": written,
            "processed_dir": str(arm_dir.relative_to(ROOT)).replace("\\", "/"),
            "source_manifest_sha256": file_sha256(source_processed / "crops.jsonl"),
        }
        atomic_json_dump(arm_dir / "arm_summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)

    payload = {
        "schema_version": 1,
        "scope": "detector-crop and size-matched gold-crop training manifests for the pre-registered P3-4 experiment",
        "preregistration": "docs/ROBUST_RECOGNITION_TRAINING_PREREGISTRATION_20260913.md",
        "split": args.split,
        "detector_checkpoint": {
            "path": str(resolve_project_path(args.detector_checkpoint).relative_to(ROOT)).replace("\\", "/"),
            "sha256": file_sha256(resolve_project_path(args.detector_checkpoint)),
        },
        "matching_iou": float(args.matching_iou),
        "padding_ratio": float(args.padding),
        "sampling_seed": int(args.sampling_seed),
        "plates": len(plates),
        "matched_crops": len(matched),
        "control_crops": len(control_entries),
        "label_coverage": {
            "matched_distinct_labels": len({entry["label"] for entry in matched}),
            "control_distinct_labels": len({entry["label"] for entry in control_entries}),
        },
        "elapsed_seconds": perf_counter() - started,
        "test_access": "no validation or test prediction was read; training manifests derive from the training partition only",
    }
    atomic_json_dump(output_root / "build_summary.json", payload)
    print(json.dumps({"matched": len(matched), "control": len(control_entries), "elapsed_s": round(payload["elapsed_seconds"], 1)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
