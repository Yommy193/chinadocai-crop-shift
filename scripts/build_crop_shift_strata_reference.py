"""Freeze train-only frequency and normalized-size strata for later analysis."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from PIL import Image

from oracle_vit.utils import atomic_json_dump, file_sha256, read_jsonl


ROOT = Path(__file__).resolve().parents[1]


def interpolated_quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot compute a quantile of an empty list")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = probability * (len(ordered) - 1)
    lower, upper = int(math.floor(position)), int(math.ceil(position))
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def load_frozen_counts(path: Path, vocabulary_size: int) -> tuple[list[int], str]:
    """Load the experiment's frozen training frequencies with strict checks."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("counts"), list):
        raise ValueError(f"class-counts file lacks a counts list: {path}")
    counts = payload["counts"]
    if len(counts) != vocabulary_size:
        raise ValueError(
            f"class-counts length {len(counts)} disagrees with vocabulary size {vocabulary_size}"
        )
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts):
        raise ValueError("class-counts must contain non-negative integers")
    return [int(value) for value in counts], file_sha256(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build train-only crop-shift strata reference")
    parser.add_argument("--experiment", default="artifacts/experiments/crop_shift_alignment_20261009")
    parser.add_argument("--source-processed", default="data/processed")
    parser.add_argument(
        "--class-counts",
        help="optional frozen training-frequency JSON; when supplied, use it instead of historical crops.jsonl counts",
    )
    parser.add_argument("--output")
    args = parser.parse_args()
    experiment = (ROOT / args.experiment).resolve()
    source_processed = (ROOT / args.source_processed).resolve()
    selected_manifest = experiment / "manifests" / "gold_count" / "crops.jsonl"
    selected_rows = [row for row in read_jsonl(selected_manifest) if row.get("split") == "train"]
    if any(row.get("split") == "test" for row in selected_rows):
        raise ValueError("selected training rows unexpectedly contain test data")

    full_train = [
        row for row in read_jsonl(source_processed / "crops.jsonl") if row.get("split") == "train"
    ]
    frequency = Counter(int(row["class_id"]) for row in full_train)
    image_sizes: dict[tuple[str, str], tuple[int, int]] = {}
    area_fractions: list[float] = []
    for row in selected_rows:
        source = str(row["crop_source"])
        field = "facsimile" if source == "facsimile" else "rubbing"
        filename = str(row[field])
        key = source, filename
        if key not in image_sizes:
            path = source_processed / "images" / source / filename
            with Image.open(path) as image:
                image_sizes[key] = (int(image.width), int(image.height))
        width, height = image_sizes[key]
        box = row["box"]
        area_fractions.append(float(box[2]) * float(box[3]) / max(1.0, float(width * height)))

    vocabulary = json.loads((source_processed / "vocab.json").read_text(encoding="utf-8"))
    counts_source = "historical_train_partition"
    counts_source_sha256 = None
    if args.class_counts:
        class_counts_path = Path(args.class_counts)
        if not class_counts_path.is_absolute():
            class_counts_path = ROOT / class_counts_path
        counts, counts_source_sha256 = load_frozen_counts(
            class_counts_path.resolve(), len(vocabulary["labels"])
        )
        counts_source = str(class_counts_path.resolve())
    else:
        counts = [int(frequency.get(class_id, 0)) for class_id in range(len(vocabulary["labels"]))]
    payload = {
        "schema_version": 1,
        "scope": "train-only fixed strata reference for the additive crop-shift experiment",
        "selected_pair_manifest": str(selected_manifest.relative_to(ROOT)).replace("\\", "/"),
        "selected_pair_manifest_sha256": file_sha256(selected_manifest),
        "source_manifest": str((source_processed / "crops.jsonl").relative_to(ROOT)).replace("\\", "/"),
        "source_manifest_sha256": file_sha256(source_processed / "crops.jsonl"),
        "counts_source": counts_source,
        "counts_source_sha256": counts_source_sha256,
        "selected_pairs": len(selected_rows),
        "unique_source_images": len(image_sizes),
        "class_train_counts": counts,
        "counts": counts,
        "frequency_bins": [
            {"name": "rare", "minimum": 1, "maximum": 20},
            {"name": "medium", "minimum": 21, "maximum": 100},
            {"name": "frequent", "minimum": 101, "maximum": None},
        ],
        "normalized_gold_box_area_tertiles": {
            "q33": interpolated_quantile(area_fractions, 1.0 / 3.0),
            "q67": interpolated_quantile(area_fractions, 2.0 / 3.0),
            "definition": "gold box width*height divided by source image width*height on the selected train pairs",
        },
        "iou_bins": [
            {"name": "0.50-0.60", "minimum": 0.50, "maximum_exclusive": 0.60},
            {"name": "0.60-0.75", "minimum": 0.60, "maximum_exclusive": 0.75},
            {"name": "0.75-0.90", "minimum": 0.75, "maximum_exclusive": 0.90},
            {"name": "0.90-1.00", "minimum": 0.90, "maximum_inclusive": 1.00},
        ],
        "test_access": "prohibited; only selected train rows and the original train partition are read",
    }
    output = Path(args.output) if args.output else experiment / "strata_reference.json"
    if not output.is_absolute():
        output = ROOT / output
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing strata reference: {output}")
    atomic_json_dump(output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

