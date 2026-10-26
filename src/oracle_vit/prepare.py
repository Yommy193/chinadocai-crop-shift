from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from .utils import atomic_json_dump, load_config, resolve_project_path, write_jsonl


SPECIAL_TOKENS = {"<pad>": 0, "<mask>": 1, "<unk>": 2}


def _initial_split(name: str) -> str:
    bucket = int(hashlib.sha1(name.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "val"
    return "test"


def _record_labels(record: dict[str, Any]) -> set[str]:
    return {
        char["Label"]
        for group in record["RecordUtilSentenceGroupVoList"]
        for char in group["RecordUtilOracleCharVoList"]
        if char.get("Label")
    }


def _assign_splits(records: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    """Create a plate-level split and move the minimum useful rare-label plates to train."""
    assignments = [_initial_split(record["RubbingName"]) for record in records]
    label_sets = [_record_labels(record) for record in records]
    all_labels = set().union(*label_sets)
    train_labels = set().union(*(labels for labels, split in zip(label_sets, assignments) if split == "train"))
    missing = all_labels - train_labels
    moved: list[str] = []

    while missing:
        candidates = [
            (len(labels & missing), index)
            for index, (labels, split) in enumerate(zip(label_sets, assignments))
            if split != "train" and labels & missing
        ]
        if not candidates:
            break
        _, best_index = max(candidates, key=lambda item: (item[0], records[item[1]]["RubbingName"]))
        assignments[best_index] = "train"
        missing -= label_sets[best_index]
        moved.append(records[best_index]["RubbingName"])

    if missing:
        raise RuntimeError(f"Could not cover {len(missing)} labels in the training split")
    return assignments, moved


def _parse_box(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",")]
    if len(values) != 4 or values[2] <= 0 or values[3] <= 0:
        raise ValueError(f"Invalid box: {value!r}")
    return values


def _extract_flat_zip(archive_path: Path, destination: Path) -> dict[str, int]:
    destination.mkdir(parents=True, exist_ok=True)
    extracted = 0
    skipped = 0
    with zipfile.ZipFile(archive_path) as archive:
        for entry in archive.infolist():
            if entry.is_dir():
                continue
            name = Path(entry.filename).name
            if name != entry.filename or Path(name).suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                raise ValueError(f"Unsafe or unexpected image entry: {entry.filename!r}")
            target = destination / name
            if target.exists() and target.stat().st_size == entry.file_size:
                skipped += 1
                continue
            temporary = target.with_suffix(target.suffix + ".tmp")
            with archive.open(entry) as source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            temporary.replace(target)
            extracted += 1
    return {"extracted": extracted, "unchanged": skipped}


def _extract_reference_glyphs(
    archive_path: Path,
    destination: Path,
    valid_labels: set[str],
) -> dict[str, int]:
    """Extract the supplemental canonical/variant PNG hierarchy despite its mojibake root name."""
    identifier = re.compile(r"^[a-z0-9]{10}$")
    destination.mkdir(parents=True, exist_ok=True)
    extracted = skipped = ignored = 0
    with zipfile.ZipFile(archive_path) as archive:
        for entry in archive.infolist():
            if entry.is_dir() or not entry.filename.lower().endswith(".png"):
                continue
            parts = entry.filename.replace("\\", "/").split("/")
            if len(parts) < 4:
                ignored += 1
                continue
            canonical = parts[-3]
            variant = Path(parts[-1]).stem
            if canonical not in valid_labels or not identifier.fullmatch(variant):
                ignored += 1
                continue
            target_dir = destination / canonical
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / f"{variant}.png"
            if target.exists() and target.stat().st_size == entry.file_size:
                skipped += 1
                continue
            temporary = target.with_suffix(".png.tmp")
            with archive.open(entry) as source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            temporary.replace(target)
            extracted += 1
    return {"extracted": extracted, "unchanged": skipped, "ignored": ignored}


def prepare(config_path: str, metadata_only: bool = False) -> dict[str, Any]:
    config = load_config(config_path)
    raw_dir = resolve_project_path(config["data"]["raw_dir"])
    processed_dir = resolve_project_path(config["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    label_path = raw_dir / "标签.json"
    with label_path.open("r", encoding="utf-8") as handle:
        records: list[dict[str, Any]] = json.load(handle)

    assignments, moved_records = _assign_splits(records)
    label_counts: Counter[str] = Counter()
    for record, split in zip(records, assignments):
        if split != "train":
            continue
        for group in record["RecordUtilSentenceGroupVoList"]:
            label_counts.update(
                char["Label"] for char in group["RecordUtilOracleCharVoList"] if char.get("Label")
            )

    labels = sorted(label_counts)
    label_to_class = {label: index for index, label in enumerate(labels)}
    label_to_token = {label: index + len(SPECIAL_TOKENS) for index, label in enumerate(labels)}
    vocabulary = {
        "special_tokens": SPECIAL_TOKENS,
        "labels": labels,
        "label_to_class": label_to_class,
        "label_to_token": label_to_token,
    }
    atomic_json_dump(processed_dir / "vocab.json", vocabulary)

    crop_rows: list[dict[str, Any]] = []
    sequence_rows: list[dict[str, Any]] = []
    statistics: dict[str, Any] = {
        "records": Counter(),
        "groups": Counter(),
        "characters": Counter(),
        "labeled_characters": Counter(),
        "ambiguous_order_groups": Counter(),
        "labels": len(labels),
        "rare_label_records_moved_to_train": moved_records,
    }

    for record_index, (record, split) in enumerate(zip(records, assignments)):
        statistics["records"][split] += 1
        facsimile_name = Path(record["Facsimile"]).name
        rubbing_name = Path(record["Rubbing"]).name
        for group_index, group in enumerate(record["RecordUtilSentenceGroupVoList"]):
            chars = group["RecordUtilOracleCharVoList"]
            order_numbers = [char.get("OrderNumber") for char in chars]
            ambiguous = any(value is None for value in order_numbers) or len(set(order_numbers)) != len(order_numbers)
            ordered_chars = sorted(
                enumerate(chars),
                key=lambda item: (
                    item[1].get("OrderNumber", 10**9),
                    _parse_box(item[1]["Position"])[1],
                    _parse_box(item[1]["Position"])[0],
                    item[0],
                ),
            )
            statistics["groups"][split] += 1
            statistics["characters"][split] += len(chars)
            statistics["ambiguous_order_groups"][split] += int(ambiguous)

            sequence_rows.append(
                {
                    "id": f"{record_index}:{group_index}",
                    "split": split,
                    "plate": record["RubbingName"],
                    "category": group.get("GroupCategory"),
                    "labels": [char.get("Label") for _, char in ordered_chars],
                    "sub_labels": [char.get("SubLabel") for _, char in ordered_chars],
                    "boxes": [_parse_box(char["Position"]) for _, char in ordered_chars],
                    "order_numbers": [char.get("OrderNumber") for _, char in ordered_chars],
                    "ambiguous_order": ambiguous,
                }
            )

            for char_index, char in enumerate(chars):
                label = char.get("Label")
                if label not in label_to_class:
                    continue
                statistics["labeled_characters"][split] += 1
                crop_rows.append(
                    {
                        "id": f"{record_index}:{group_index}:{char_index}",
                        "split": split,
                        "plate": record["RubbingName"],
                        "facsimile": facsimile_name,
                        "rubbing": rubbing_name,
                        "box": _parse_box(char["Position"]),
                        "label": label,
                        "class_id": label_to_class[label],
                        "sub_label": char.get("SubLabel"),
                        "mark": char.get("Mark"),
                        "seat_font": char.get("SeatFont"),
                    }
                )

    write_jsonl(processed_dir / "crops.jsonl", crop_rows)
    write_jsonl(processed_dir / "sequences.jsonl", sequence_rows)

    for key in (
        "records",
        "groups",
        "characters",
        "labeled_characters",
        "ambiguous_order_groups",
    ):
        statistics[key] = dict(statistics[key])
    statistics["raw_files"] = {
        path.name: {"bytes": path.stat().st_size, "modified_ns": path.stat().st_mtime_ns}
        for path in raw_dir.iterdir()
        if path.is_file()
    }

    if not metadata_only:
        image_root = processed_dir / "images"
        statistics["image_extraction"] = {
            "facsimile": _extract_flat_zip(raw_dir / "摹本.zip", image_root / "facsimile"),
            "rubbing": _extract_flat_zip(raw_dir / "拓片.zip", image_root / "rubbing"),
            "glyph_reference": _extract_reference_glyphs(
                raw_dir / "字头信息的补充材料.zip",
                processed_dir / "glyph_reference_full",
                set(labels),
            ),
        }

    atomic_json_dump(processed_dir / "stats.json", statistics)
    return statistics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare the oracle-bone multimodal dataset")
    parser.add_argument("--config", default="configs/baseline.json")
    parser.add_argument("--metadata-only", action="store_true", help="Build indices without extracting image ZIP files")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    stats = prepare(args.config, metadata_only=args.metadata_only)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
