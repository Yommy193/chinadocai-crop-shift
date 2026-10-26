"""Fail-closed validation for the additive crop-shift manifests."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from oracle_vit.utils import atomic_json_dump, file_sha256, read_jsonl


ROOT = Path(__file__).resolve().parents[1]
COUNT_ARMS = (
    "gold_count",
    "anchor_pred_count",
    "anchor_jitter_count",
    "frcnn_pred_count",
    "frcnn_jitter_count",
)
STEP_ARMS = (
    "gold_step",
    "anchor_pred_step",
    "anchor_mix025_step",
    "anchor_mix050_step",
    "anchor_mix075_step",
    "anchor_curriculum_step",
    "frcnn_pred_step",
    "frcnn_mix025_step",
    "frcnn_mix050_step",
    "frcnn_mix075_step",
    "frcnn_curriculum_step",
)


def _training_rows(path: Path) -> tuple[list[dict[str, Any]], Counter[str]]:
    rows = read_jsonl(path)
    splits = Counter(str(row.get("split")) for row in rows)
    if splits.get("test", 0):
        raise ValueError(f"forbidden test rows in {path}")
    train = [row for row in rows if row.get("split") == "train"]
    if len(train) != splits.get("train", 0):
        raise ValueError(f"malformed train split in {path}")
    return train, splits


def _composition(rows: list[dict[str, Any]]) -> Counter[tuple[Any, ...]]:
    return Counter(
        (
            str(row["pair_key"]),
            str(row["gold_id"]),
            str(row["plate"]),
            str(row["crop_source"]),
            int(row["class_id"]),
            str(row["label"]),
        )
        for row in rows
    )


def _sequence(rows: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    return [
        (str(row["pair_key"]), str(row["crop_source"]), int(row["class_id"]), str(row["plate"]))
        for row in rows
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate crop-shift alignment manifests")
    parser.add_argument("--root", default="artifacts/experiments/crop_shift_alignment_20261009")
    args = parser.parse_args()
    experiment = (ROOT / args.root).resolve()
    summary = json.loads((experiment / "build_summary.json").read_text(encoding="utf-8"))
    expected_hashes = {
        str((ROOT / item["path"]).resolve()): item["sha256"]
        for item in summary["detector_inputs"].values()
    }
    expected_hashes[str((ROOT / summary["source_manifest"]).resolve())] = summary["source_manifest_sha256"]
    for value, expected in expected_hashes.items():
        actual = file_sha256(Path(value))
        if actual != expected:
            raise ValueError(f"read-only input hash changed: {value}: {actual} != {expected}")

    loaded: dict[str, list[dict[str, Any]]] = {}
    split_counts: dict[str, Any] = {}
    for arm in (*COUNT_ARMS, *STEP_ARMS):
        path = experiment / "manifests" / arm / "crops.jsonl"
        rows, splits = _training_rows(path)
        loaded[arm] = rows
        split_counts[arm] = dict(sorted(splits.items()))
        expected = summary["arms"][arm]
        if len(rows) != int(expected["train_rows"]):
            raise ValueError(f"row-count mismatch for {arm}")
        if file_sha256(path) != str(expected["manifest_sha256"]):
            raise ValueError(f"manifest hash mismatch for {arm}")

    count_reference = _composition(loaded[COUNT_ARMS[0]])
    for arm in COUNT_ARMS[1:]:
        if _composition(loaded[arm]) != count_reference:
            raise ValueError(f"count-arm composition mismatch: {arm}")
    step_reference = _sequence(loaded[STEP_ARMS[0]])
    for arm in STEP_ARMS[1:]:
        if _sequence(loaded[arm]) != step_reference:
            raise ValueError(f"step-arm draw sequence mismatch: {arm}")

    for arm in (*COUNT_ARMS, *STEP_ARMS):
        for row in loaded[arm]:
            if row["crop_source"] not in {"rubbing", "facsimile"}:
                raise ValueError(f"invalid crop source in {arm}")
            if row["split"] != "train":
                raise ValueError(f"non-train row leaked into training list for {arm}")
    for arm in (name for name in STEP_ARMS if "mix" in name or "curriculum" in name):
        for row in loaded[arm]:
            if set(row.get("paired_boxes", {})) != {"gold", "predicted"}:
                raise ValueError(f"missing paired boxes in {arm}")
            if "crop_policy" not in row:
                raise ValueError(f"missing crop policy in {arm}")

    report = {
        "schema_version": 1,
        "status": "passed",
        "experiment": str(experiment.relative_to(ROOT)).replace("\\", "/"),
        "read_only_input_hashes_verified": len(expected_hashes),
        "count_arms_same_composition": list(COUNT_ARMS),
        "step_arms_same_draw_sequence": list(STEP_ARMS),
        "count_rows": len(loaded[COUNT_ARMS[0]]),
        "step_rows": len(loaded[STEP_ARMS[0]]),
        "split_counts": split_counts,
        "test_rows_across_all_manifests": 0,
    }
    output = experiment / "validation_report.json"
    atomic_json_dump(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

