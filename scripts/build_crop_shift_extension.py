"""Build the additive FCOS arm for the CSIG crop-shift supplement.

This script is intentionally separate from ``build_crop_shift_alignment_data.py``.
It only reads the historical manifests and the frozen FCOS OOF prediction cache, and
writes a new extension root.  The FCOS cache comes from the five-fold detector
comparison suite (one seed per fold); no validation or historical test plate is read
for training-manifest construction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from build_crop_shift_alignment_data import reconstruct_pairs  # noqa: E402
from oracle_vit.utils import file_sha256  # noqa: E402


def xywh_to_xyxy(box: list[float] | list[int]) -> tuple[float, float, float, float]:
    x, y, w, h = (float(v) for v in box)
    return x, y, x + w, y + h


def greedy_matches(pred_boxes: np.ndarray, gold_boxes: np.ndarray, threshold: float) -> list[tuple[int, int, float]]:
    """Greedy score-order matching; NPZ predictions are already score sorted."""
    if len(pred_boxes) == 0 or len(gold_boxes) == 0:
        return []
    px1, py1, px2, py2 = [pred_boxes[:, i, None] for i in range(4)]
    gx1, gy1, gx2, gy2 = [gold_boxes[:, i][None, :] for i in range(4)]
    inter = np.maximum(0.0, np.minimum(px2, gx2) - np.maximum(px1, gx1)) * np.maximum(
        0.0, np.minimum(py2, gy2) - np.maximum(py1, gy1)
    )
    p_area = np.maximum(0.0, px2 - px1) * np.maximum(0.0, py2 - py1)
    g_area = np.maximum(0.0, gx2 - gx1) * np.maximum(0.0, gy2 - gy1)
    ious = inter / np.maximum(1e-12, p_area + g_area - inter)
    used: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for prediction_index in range(len(pred_boxes)):
        order = np.argsort(-ious[prediction_index])
        for target_index in order.tolist():
            if target_index in used:
                continue
            overlap = float(ious[prediction_index, target_index])
            if overlap >= threshold:
                used.add(int(target_index))
                matches.append((prediction_index, int(target_index), overlap))
            break
    return matches


def hash_rank(keys: list[tuple[str, str]], seed: int) -> list[tuple[str, str]]:
    return sorted(
        keys,
        key=lambda key: (hashlib.sha256(f"{seed}|{key[0]}|{key[1]}".encode()).hexdigest(), key[0], key[1]),
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def junction(link: Path, target: Path) -> None:
    if link.exists():
        return
    if os.name == "nt":
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True, capture_output=True)
    else:
        link.symlink_to(target, target_is_directory=True)


def fcos_pairs_from_oof(
    gold_train: list[dict[str, Any]],
    prediction_root: Path,
    seed: int,
    iou_threshold: float,
) -> dict[tuple[str, str], dict[str, Any]]:
    gold_by_plate = defaultdict(list)
    for row in gold_train:
        gold_by_plate[str(row["plate"])].append(row)

    pairs: dict[tuple[str, str], dict[str, Any]] = {}
    seen_plates: set[str] = set()
    files = sorted(prediction_root.glob(f"fold_*/seed_{seed}/predictions_*.npz"))
    if len(files) != 10:
        raise FileNotFoundError(f"expected 10 FCOS OOF NPZ files for seed {seed}, found {len(files)}")
    for path in files:
        source = path.stem.replace("predictions_", "")
        payload = np.load(path, allow_pickle=True)
        for plate_index, plate_value in enumerate(payload["plate_ids"].tolist()):
            plate = str(plate_value)
            if plate not in gold_by_plate or plate in seen_plates:
                continue
            # Each plate appears once per fold, while each source is a separate NPZ.
            # Keep the plate guard only after processing both domains below.
            gold_rows = gold_by_plate[plate]
            gold_boxes = np.asarray([xywh_to_xyxy(row["box"]) for row in gold_rows], dtype=np.float32)
            start = int(payload["pred_offsets"][plate_index])
            stop = int(payload["pred_offsets"][plate_index + 1])
            pred_boxes = np.asarray(payload["pred_boxes"][start:stop], dtype=np.float32)
            for prediction_index, target_index, overlap in greedy_matches(pred_boxes, gold_boxes, iou_threshold):
                gold = gold_rows[target_index]
                box = pred_boxes[prediction_index]
                predicted_xywh = [
                    int(round(float(box[0]))),
                    int(round(float(box[1]))),
                    max(1, int(round(float(box[2] - box[0])))),
                    max(1, int(round(float(box[3] - box[1])))),
                ]
                key = (str(gold["id"]), source)
                if key in pairs:
                    raise ValueError(f"duplicate FCOS OOF match: {key}")
                pairs[key] = {
                    "pair_key": f"{gold['id']}::{source}",
                    "gold_id": str(gold["id"]),
                    "crop_source": source,
                    "gold": gold,
                    "predicted": {
                        "box": predicted_xywh,
                        "matched_iou": float(overlap),
                        "id": f"fcos::{plate}::{source}::{prediction_index}",
                    },
                    "matched_iou": float(overlap),
                    "rounded_box_iou": float(overlap),
                    "detector": "fcos_r18_fpn_oof",
                }
            if source == "rubbing":
                seen_plates.add(plate)
    return pairs


def paired_row(pair: dict[str, Any], arm: str, draw_index: int, variant: str, probability: float | None) -> dict[str, Any]:
    gold = pair["gold"]
    gold_box = [int(v) for v in gold["box"]]
    predicted_box = [int(v) for v in pair["predicted"]["box"]]
    row = {
        **gold,
        "id": f"{arm}::{draw_index:06d}::{pair['pair_key']}",
        "split": "train",
        "box": predicted_box if variant == "predicted" else gold_box,
        "crop_source": pair["crop_source"],
        "pair_key": pair["pair_key"],
        "gold_id": pair["gold_id"],
        "matched_iou": float(pair["matched_iou"]),
        "rounded_box_iou": float(pair["rounded_box_iou"]),
        "crop_arm": arm,
    }
    if variant == "mix":
        row["box"] = gold_box
        row["paired_boxes"] = {"gold": gold_box, "predicted": predicted_box}
        row["crop_policy"] = {"name": "stochastic_mix", "predicted_probability": float(probability)}
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Build additive FCOS crop-shift manifests")
    parser.add_argument("--source-processed", default="data/processed")
    parser.add_argument(
        "--anchor-manifest",
        default="artifacts/experiments/robust_recognition_20260913/detected_crops/crops.jsonl",
    )
    parser.add_argument(
        "--frcnn-manifest",
        default="artifacts/experiments/robust_recognition_frcnn_crops_20260919/detected_crops/crops.jsonl",
    )
    parser.add_argument(
        "--fcos-prediction-root",
        default="artifacts/experiments/plate_detector_baseline_comparison_provenance_disjoint_20260915/outer_eval/b1_fcos_r18_fpn",
    )
    parser.add_argument("--fcos-seed", type=int, default=20260971)
    parser.add_argument("--output-root", default="artifacts/experiments/crop_shift_alignment_extension_20261021")
    parser.add_argument("--base-count", type=int, default=75248)
    parser.add_argument("--step-budget", type=int, default=126188)
    parser.add_argument("--sampling-seed", type=int, default=20261021)
    parser.add_argument("--matching-iou", type=float, default=0.50)
    args = parser.parse_args()

    source_processed = (ROOT / args.source_processed).resolve()
    output_root = (ROOT / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    for child in (output_root / "manifests").glob("fcos_*"):
        if child.is_dir():
            raise FileExistsError(f"refusing to overwrite existing extension arm: {child}")

    source_rows = load_jsonl(source_processed / "crops.jsonl")
    gold_train = [row for row in source_rows if row.get("split") == "train"]
    validation_rows = [row for row in source_rows if row.get("split") == "val"]
    anchor_path = (ROOT / args.anchor_manifest).resolve()
    frcnn_path = (ROOT / args.frcnn_manifest).resolve()
    anchor = reconstruct_pairs(gold_train, load_jsonl(anchor_path), "anchor")
    frcnn = reconstruct_pairs(gold_train, load_jsonl(frcnn_path), "frcnn")
    fcos = fcos_pairs_from_oof(
        gold_train,
        (ROOT / args.fcos_prediction_root).resolve(),
        int(args.fcos_seed),
        float(args.matching_iou),
    )
    common = set(anchor) & set(frcnn) & set(fcos)
    ranked = hash_rank(sorted(common), int(args.sampling_seed))
    if int(args.base_count) > len(ranked):
        raise ValueError(f"base-count {args.base_count} exceeds common three-detector universe {len(ranked)}")
    selected_keys = ranked[: int(args.base_count)]
    step_draws = list(range(len(selected_keys)))
    rng = random.Random(int(args.sampling_seed))
    step_draws.extend(rng.randrange(len(selected_keys)) for _ in range(int(args.step_budget) - len(selected_keys)))
    rng.shuffle(step_draws)
    selected = [fcos[key] for key in selected_keys]

    vocab_path = source_processed / "vocab.json"
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    train_rows_by_arm = {}
    for arm, variant, probability in (
        ("fcos_pred_step", "predicted", None),
        ("fcos_mix075_step", "mix", 0.75),
    ):
        rows = [paired_row(selected[index], arm, draw_index, variant, probability) for draw_index, index in enumerate(step_draws)]
        train_rows_by_arm[arm] = rows
        arm_dir = output_root / "manifests" / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        junction(arm_dir / "images", source_processed / "images")
        junction(arm_dir / "glyph_reference_full", source_processed / "glyph_reference_full")
        (arm_dir / "vocab.json").write_text(vocab_path.read_text(encoding="utf-8"), encoding="utf-8")
        (arm_dir / "crops.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in [*validation_rows, *rows]),
            encoding="utf-8",
        )
        (arm_dir / "arm_summary.json").write_text(
            json.dumps(
                {
                    "arm": arm,
                    "detector_box_source": "fcos_r18_fpn_oof",
                    "variant": variant,
                    "train_rows": len(rows),
                    "validation_rows": len(validation_rows),
                    "test_rows": 0,
                    "unique_pair_keys": len(selected_keys),
                    "source_counts": dict(sorted(Counter(row["crop_source"] for row in rows).items())),
                    "class_coverage": len({int(row["class_id"]) for row in rows}),
                    "manifest_sha256": file_sha256(arm_dir / "crops.jsonl"),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    selected_payload = "\n".join(f"{key[0]}::{key[1]}" for key in selected_keys)
    summary = {
        "schema_version": 1,
        "scope": "additive CSIG supplement; FCOS source only; historical experiment files are read-only",
        "source_manifest": str((source_processed / "crops.jsonl").relative_to(ROOT)).replace("\\", "/"),
        "source_manifest_sha256": file_sha256(source_processed / "crops.jsonl"),
        "detector_inputs": {
            "anchor": {"path": str(anchor_path.relative_to(ROOT)).replace("\\", "/"), "sha256": file_sha256(anchor_path)},
            "frcnn": {"path": str(frcnn_path.relative_to(ROOT)).replace("\\", "/"), "sha256": file_sha256(frcnn_path)},
            "fcos_oof_root": str((ROOT / args.fcos_prediction_root).relative_to(ROOT)).replace("\\", "/"),
            "fcos_seed": int(args.fcos_seed),
        },
        "pair_universe": {"anchor": len(anchor), "frcnn": len(frcnn), "fcos": len(fcos), "common_three": len(common)},
        "base_count": len(selected_keys),
        "step_budget": len(step_draws),
        "sampling_seed": int(args.sampling_seed),
        "selected_pair_keys_sha256": hashlib.sha256(selected_payload.encode()).hexdigest(),
        "selected_composition": {
            "plates": len({fcos[key]["gold"]["plate"] for key in selected_keys}),
            "classes": len({int(fcos[key]["gold"]["class_id"]) for key in selected_keys}),
            "sources": dict(sorted(Counter(key[1] for key in selected_keys).items())),
        },
        "arms": {arm: json.loads((output_root / "manifests" / arm / "arm_summary.json").read_text(encoding="utf-8")) for arm in train_rows_by_arm},
        "test_access": "prohibited: only train rows are used for FCOS crop manifests; validation rows are copied as development metadata",
    }
    (output_root / "fcos_extension_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
