"""Build immutable, source-aligned crop-shift experiment manifests.

Historical data and experiment artifacts are read-only inputs.  Every output is
written below a new experiment root.  The primary universe is the intersection
of source-specific train matches made by the legacy anchor detector and the
Faster R-CNN detector.  A hash-ranked subset is shared byte-for-byte by all
arms, so only the crop box policy changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from oracle_vit.utils import atomic_json_dump, file_sha256, read_jsonl, write_jsonl


ROOT = Path(__file__).resolve().parents[1]
DETECTOR_INPUTS = {
    "anchor": Path("artifacts/experiments/robust_recognition_20260913/detected_crops/crops.jsonl"),
    "frcnn": Path("artifacts/experiments/robust_recognition_frcnn_crops_20260919/detected_crops/crops.jsonl"),
}
CURRICULUM_THRESHOLDS = [1.01, 0.90, 0.80, 0.70, 0.60, 0.50, 0.50, 0.50]


def xywh_iou(left: list[int], right: list[int]) -> float:
    lx, ly, lw, lh = (float(value) for value in left)
    rx, ry, rw, rh = (float(value) for value in right)
    intersection = max(0.0, min(lx + lw, rx + rw) - max(lx, rx)) * max(
        0.0, min(ly + lh, ry + rh) - max(ly, ry)
    )
    union = lw * lh + rw * rh - intersection
    return intersection / union if union > 0 else 0.0


def _prediction_index(row: dict[str, Any]) -> int:
    try:
        return int(str(row["id"]).rsplit("::", 1)[-1])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"prediction row has no parseable index: {row.get('id')}") from exc


def reconstruct_pairs(
    gold_rows: Iterable[dict[str, Any]],
    predicted_rows: Iterable[dict[str, Any]],
    detector: str,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Recover the source-specific gold target for each cached prediction row.

    The historical cache omitted ``gold_id``.  Its assigned class is retained,
    and within every plate/class the rounded predicted box has a unique maximum
    IoU target in the cached datasets.  We enforce that uniqueness and also
    enforce one target per source; any ambiguity fails closed.
    """

    candidates: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in gold_rows:
        if row.get("split") == "train":
            candidates[(str(row["plate"]), int(row["class_id"]))].append(row)
    pairs: dict[tuple[str, str], dict[str, Any]] = {}
    rows = sorted(
        (row for row in predicted_rows if row.get("split") == "train"),
        key=lambda row: (str(row["plate"]), str(row.get("source", "")), _prediction_index(row)),
    )
    for predicted in rows:
        source = str(predicted.get("source", ""))
        if source not in {"rubbing", "facsimile"}:
            raise ValueError(f"invalid prediction source: {source!r}")
        options = candidates.get((str(predicted["plate"]), int(predicted["class_id"])), [])
        if not options:
            raise ValueError(f"no gold candidate for prediction {predicted['id']}")
        scored = sorted(
            ((xywh_iou(predicted["box"], gold["box"]), str(gold["id"]), gold) for gold in options),
            key=lambda item: (-item[0], item[1]),
        )
        if len(scored) > 1 and abs(scored[0][0] - scored[1][0]) < 1e-12:
            raise ValueError(f"ambiguous reconstructed target for {predicted['id']}")
        rounded_iou, _, gold = scored[0]
        key = (str(gold["id"]), source)
        if key in pairs:
            raise ValueError(f"duplicate source-specific target in {detector}: {key}")
        if str(predicted["label"]) != str(gold["label"]):
            raise ValueError(f"label mismatch while reconstructing {predicted['id']}")
        stored_iou = float(predicted["matched_iou"])
        if stored_iou < 0.50:
            raise ValueError(f"cached match below IoU 0.50: {predicted['id']}")
        pairs[key] = {
            "pair_key": f"{gold['id']}::{source}",
            "gold_id": str(gold["id"]),
            "crop_source": source,
            "gold": gold,
            "predicted": predicted,
            "matched_iou": stored_iou,
            "rounded_box_iou": rounded_iou,
            "detector": detector,
        }
    return pairs


def hash_rank(keys: Iterable[tuple[str, str]], seed: int) -> list[tuple[str, str]]:
    def rank(key: tuple[str, str]) -> tuple[str, str, str]:
        payload = f"{seed}|{key[0]}|{key[1]}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest(), key[0], key[1]

    return sorted(keys, key=rank)


def paired_row(
    pair: dict[str, Any],
    *,
    arm: str,
    draw_index: int,
    box: list[int],
    paired_boxes: dict[str, list[int]] | None = None,
    crop_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    gold = pair["gold"]
    row = {
        **gold,
        "id": f"{arm}::{draw_index:06d}::{pair['pair_key']}",
        "split": "train",
        "box": [int(value) for value in box],
        "crop_source": pair["crop_source"],
        "pair_key": pair["pair_key"],
        "gold_id": pair["gold_id"],
        "matched_iou": float(pair["matched_iou"]),
        "rounded_box_iou": float(pair["rounded_box_iou"]),
        "crop_arm": arm,
    }
    if paired_boxes is not None:
        row["paired_boxes"] = paired_boxes
    if crop_policy is not None:
        row["crop_policy"] = crop_policy
    return row


def residual(gold: list[int], predicted: list[int]) -> tuple[float, float, float, float]:
    gx, gy, gw, gh = (float(value) for value in gold)
    px, py, pw, ph = (float(value) for value in predicted)
    return (px - gx) / gw, (py - gy) / gh, math.log(pw / gw), math.log(ph / gh)


def apply_residual(gold: list[int], value: tuple[float, float, float, float]) -> list[int]:
    gx, gy, gw, gh = (float(item) for item in gold)
    dx, dy, log_width, log_height = value
    width = max(1, int(round(gw * math.exp(log_width))))
    height = max(1, int(round(gh * math.exp(log_height))))
    x = max(0, int(round(gx + dx * gw)))
    y = max(0, int(round(gy + dy * gh)))
    return [x, y, width, height]


def empirical_jitter_boxes(
    pairs: list[dict[str, Any]], seed: int
) -> tuple[dict[str, list[int]], dict[str, int]]:
    """Transfer empirical residuals within source and gold-area quartile."""

    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        by_source[pair["crop_source"]].append(pair)
    for source, source_pairs in by_source.items():
        ordered = sorted(source_pairs, key=lambda item: (item["gold"]["box"][2] * item["gold"]["box"][3], item["pair_key"]))
        size = len(ordered)
        for position, pair in enumerate(ordered):
            quartile = min(3, position * 4 // max(1, size))
            groups[(source, quartile)].append(pair)

    output: dict[str, list[int]] = {}
    audit = Counter()
    for group_key, group in sorted(groups.items()):
        ordered = sorted(
            group,
            key=lambda item: hashlib.sha256(f"{seed}|jitter|{item['pair_key']}".encode("utf-8")).hexdigest(),
        )
        for position, target in enumerate(ordered):
            chosen: list[int] | None = None
            for offset in range(1, len(ordered) + 1):
                donor = ordered[(position + offset) % len(ordered)]
                if donor["gold_id"] == target["gold_id"]:
                    continue
                candidate = apply_residual(
                    target["gold"]["box"], residual(donor["gold"]["box"], donor["predicted"]["box"])
                )
                if xywh_iou(target["gold"]["box"], candidate) >= 0.50:
                    chosen = candidate
                    audit["transferred"] += 1
                    if offset > 1:
                        audit["donor_retries"] += offset - 1
                    break
                audit["rejected_below_iou50"] += 1
            if chosen is None:
                chosen = list(target["gold"]["box"])
                audit["gold_fallback"] += 1
            output[target["pair_key"]] = chosen
    return output, dict(audit)


def build_draws(size: int, budget: int, seed: int) -> list[int]:
    if size <= 0 or budget < size:
        raise ValueError("step budget must be at least the base-pair count")
    rng = random.Random(seed)
    draws = list(range(size))
    draws.extend(rng.randrange(size) for _ in range(budget - size))
    rng.shuffle(draws)
    return draws


def _quantiles(values: Iterable[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    result: dict[str, float] = {}
    for name, probability in (("min", 0.0), ("p10", 0.1), ("p25", 0.25), ("p50", 0.5), ("p75", 0.75), ("p90", 0.9), ("max", 1.0)):
        position = probability * (len(ordered) - 1)
        lower, upper = int(math.floor(position)), int(math.ceil(position))
        value = ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
        result[name] = value
    return result


def _junction(link: Path, target: Path) -> None:
    if link.exists():
        return
    if os.name == "nt":
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True, capture_output=True)
    else:
        link.symlink_to(target, target_is_directory=True)


def _reference_counts(source_processed: Path, labels: list[str]) -> tuple[list[int], list[str]]:
    counts = [0] * len(labels)
    paths: list[str] = []
    root = source_processed / "glyph_reference_full"
    for class_id in range(len(labels)):
        directory = root / labels[class_id]
        if not directory.exists():
            continue
        files = sorted(directory.glob("*.png"))
        counts[class_id] = len(files)
        paths.extend(str(path.relative_to(root)).replace("\\", "/") for path in files)
    return counts, paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fair source-aligned crop-shift manifests")
    parser.add_argument("--source-processed", default="data/processed")
    parser.add_argument("--anchor-manifest", default=str(DETECTOR_INPUTS["anchor"]))
    parser.add_argument("--frcnn-manifest", default=str(DETECTOR_INPUTS["frcnn"]))
    parser.add_argument("--output-root", default="artifacts/experiments/crop_shift_alignment_20261009")
    parser.add_argument("--base-count", type=int, default=75248)
    parser.add_argument("--step-budget", type=int, default=126188)
    parser.add_argument("--sampling-seed", type=int, default=20261009)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source_processed = (ROOT / args.source_processed).resolve()
    output_root = (ROOT / args.output_root).resolve()
    if output_root.exists() and any(output_root.iterdir()) and not args.force:
        raise FileExistsError(f"refusing to overwrite non-empty experiment root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    source_manifest = source_processed / "crops.jsonl"
    source_rows = read_jsonl(source_manifest)
    gold_train = [row for row in source_rows if row.get("split") == "train"]
    validation_rows = [row for row in source_rows if row.get("split") == "val"]
    paths = {
        "anchor": (ROOT / args.anchor_manifest).resolve(),
        "frcnn": (ROOT / args.frcnn_manifest).resolve(),
    }
    pairs = {
        detector: reconstruct_pairs(gold_train, read_jsonl(path), detector)
        for detector, path in paths.items()
    }
    common_keys = set(pairs["anchor"]) & set(pairs["frcnn"])
    ranked = hash_rank(common_keys, int(args.sampling_seed))
    if int(args.base_count) > len(ranked):
        raise ValueError(f"base-count {args.base_count} exceeds common pair universe {len(ranked)}")
    selected_keys = ranked[: int(args.base_count)]
    selected = {
        detector: [pairs[detector][key] for key in selected_keys]
        for detector in ("anchor", "frcnn")
    }
    count_draws = list(range(len(selected_keys)))
    step_draws = build_draws(len(selected_keys), int(args.step_budget), int(args.sampling_seed))
    jitter: dict[str, dict[str, list[int]]] = {}
    jitter_audit: dict[str, dict[str, int]] = {}
    for detector in ("anchor", "frcnn"):
        jitter[detector], jitter_audit[detector] = empirical_jitter_boxes(
            selected[detector], int(args.sampling_seed)
        )

    arm_specs: list[tuple[str, str, str, list[int], float | None]] = [
        ("gold_count", "anchor", "gold", count_draws, None),
        ("gold_step", "anchor", "gold", step_draws, None),
    ]
    for detector in ("anchor", "frcnn"):
        arm_specs.extend(
            [
                (f"{detector}_pred_count", detector, "predicted", count_draws, None),
                (f"{detector}_jitter_count", detector, "jitter", count_draws, None),
                (f"{detector}_pred_step", detector, "predicted", step_draws, None),
                (f"{detector}_mix025_step", detector, "mix", step_draws, 0.25),
                (f"{detector}_mix050_step", detector, "mix", step_draws, 0.50),
                (f"{detector}_mix075_step", detector, "mix", step_draws, 0.75),
                (f"{detector}_curriculum_step", detector, "curriculum", step_draws, None),
            ]
        )

    vocab = json.loads((source_processed / "vocab.json").read_text(encoding="utf-8"))
    covered = {int(pairs["anchor"][key]["gold"]["class_id"]) for key in selected_keys}
    pair_counts = Counter(int(pairs["anchor"][key]["gold"]["class_id"]) for key in selected_keys)
    reference_counts, reference_paths = _reference_counts(source_processed, list(vocab["labels"]))
    frozen_counts = [pair_counts.get(class_id, 0) + reference_counts[class_id] for class_id in range(len(vocab["labels"]))]
    class_count_payload = {
        "schema_version": 1,
        "definition": "base-count selected pair rows plus the full-vocabulary fixed reference glyph pool; shared by every arm",
        "counts": frozen_counts,
        "selected_pairs": len(selected_keys),
        "covered_classes": len(covered),
        "reference_samples": sum(reference_counts),
        "reference_paths_sha256": hashlib.sha256("\n".join(reference_paths).encode("utf-8")).hexdigest(),
    }
    atomic_json_dump(output_root / "frozen_class_counts.json", class_count_payload)

    arm_summaries: dict[str, Any] = {}
    for arm, detector, variant, draws, probability in arm_specs:
        train_rows: list[dict[str, Any]] = []
        for draw_index, pair_index in enumerate(draws):
            pair = selected[detector][pair_index]
            gold_box = [int(value) for value in pair["gold"]["box"]]
            predicted_box = [int(value) for value in pair["predicted"]["box"]]
            kwargs: dict[str, Any] = {"arm": arm, "draw_index": draw_index}
            if variant == "gold":
                box = gold_box
            elif variant == "predicted":
                box = predicted_box
            elif variant == "jitter":
                box = jitter[detector][pair["pair_key"]]
            elif variant == "mix":
                box = gold_box
                kwargs["paired_boxes"] = {"gold": gold_box, "predicted": predicted_box}
                kwargs["crop_policy"] = {"name": "stochastic_mix", "predicted_probability": probability}
            elif variant == "curriculum":
                box = gold_box
                kwargs["paired_boxes"] = {"gold": gold_box, "predicted": predicted_box}
                kwargs["crop_policy"] = {"name": "iou_curriculum", "thresholds": CURRICULUM_THRESHOLDS}
            else:
                raise AssertionError(variant)
            train_rows.append(paired_row(pair, box=box, **kwargs))

        arm_dir = output_root / "manifests" / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        _junction(arm_dir / "images", source_processed / "images")
        _junction(arm_dir / "glyph_reference_full", source_processed / "glyph_reference_full")
        shutil.copyfile(source_processed / "vocab.json", arm_dir / "vocab.json")
        # Development manifests physically exclude the historical test split.
        write_jsonl(arm_dir / "crops.jsonl", [*validation_rows, *train_rows])
        summary = {
            "arm": arm,
            "detector_box_source": detector if variant != "gold" else None,
            "variant": variant,
            "train_rows": len(train_rows),
            "validation_rows": len(validation_rows),
            "test_rows": 0,
            "unique_pair_keys": len({row["pair_key"] for row in train_rows}),
            "source_counts": dict(sorted(Counter(row["crop_source"] for row in train_rows).items())),
            "class_coverage": len({int(row["class_id"]) for row in train_rows}),
            "manifest_sha256": file_sha256(arm_dir / "crops.jsonl"),
        }
        atomic_json_dump(arm_dir / "arm_summary.json", summary)
        arm_summaries[arm] = summary

    selected_payload = "\n".join(f"{key[0]}::{key[1]}" for key in selected_keys)
    build_summary = {
        "schema_version": 2,
        "scope": "new additive crop-shift alignment experiment; all historical inputs remain read-only",
        "source_manifest": str(source_manifest.relative_to(ROOT)).replace("\\", "/"),
        "source_manifest_sha256": file_sha256(source_manifest),
        "detector_inputs": {
            name: {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": file_sha256(path)}
            for name, path in paths.items()
        },
        "full_pairs": {name: len(value) for name, value in pairs.items()},
        "common_pair_universe": len(common_keys),
        "base_count": len(selected_keys),
        "step_budget": len(step_draws),
        "sampling_seed": int(args.sampling_seed),
        "selected_pair_keys_sha256": hashlib.sha256(selected_payload.encode("utf-8")).hexdigest(),
        "selected_composition": {
            "sources": dict(sorted(Counter(key[1] for key in selected_keys).items())),
            "plates": len({pairs["anchor"][key]["gold"]["plate"] for key in selected_keys}),
            "classes": len(covered),
        },
        "iou_quantiles": {
            detector: _quantiles(pair["matched_iou"] for pair in selected[detector])
            for detector in ("anchor", "frcnn")
        },
        "jitter_audit": jitter_audit,
        "frozen_class_counts_sha256": file_sha256(output_root / "frozen_class_counts.json"),
        "arms": arm_summaries,
        "test_access": "prohibited: generated manifests contain train and validation rows only",
    }
    atomic_json_dump(output_root / "build_summary.json", build_summary)
    print(json.dumps(build_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

