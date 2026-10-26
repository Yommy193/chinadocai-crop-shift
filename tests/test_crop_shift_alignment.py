import json
import random

import pytest
import torch
from PIL import Image

from scripts.build_crop_shift_alignment_data import (
    apply_residual,
    build_draws,
    hash_rank,
    reconstruct_pairs,
    residual,
    xywh_iou,
)
from scripts.crop_shift_dataset import CropShiftDataset
from scripts.train_crop_shift_cnn import _assert_development_manifest


def _gold(identifier: str, class_id: int, box: list[int]) -> dict:
    return {
        "id": identifier,
        "split": "train",
        "plate": "P1",
        "facsimile": "sample.png",
        "rubbing": "sample.png",
        "box": box,
        "label": f"label-{class_id}",
        "class_id": class_id,
    }


def _pred(identifier: str, class_id: int, source: str, box: list[int], overlap: float) -> dict:
    return {
        "id": identifier,
        "split": "train",
        "plate": "P1",
        "source": source,
        "facsimile": "sample.png",
        "rubbing": "sample.png",
        "box": box,
        "label": f"label-{class_id}",
        "class_id": class_id,
        "matched_iou": overlap,
    }


def test_reconstruct_pairs_is_source_specific_and_order_invariant() -> None:
    gold = [_gold("g0", 0, [0, 0, 10, 10]), _gold("g1", 0, [30, 0, 10, 10])]
    predicted = [
        _pred("det::P1::rubbing::1", 0, "rubbing", [29, 0, 10, 10], 0.81),
        _pred("det::P1::rubbing::0", 0, "rubbing", [1, 0, 10, 10], 0.81),
        _pred("det::P1::facsimile::0", 0, "facsimile", [2, 0, 10, 10], 0.67),
    ]
    first = reconstruct_pairs(gold, predicted, "toy")
    second = reconstruct_pairs(gold, list(reversed(predicted)), "toy")
    assert list(first) == list(second)
    assert set(first) == {("g0", "rubbing"), ("g1", "rubbing"), ("g0", "facsimile")}
    assert first[("g1", "rubbing")]["predicted"]["id"].endswith("::1")


def test_hash_selection_and_step_draws_are_reproducible() -> None:
    keys = [(f"g{i}", "rubbing" if i % 2 else "facsimile") for i in range(20)]
    shuffled = list(keys)
    random.Random(8).shuffle(shuffled)
    assert hash_rank(keys, 42) == hash_rank(shuffled, 42)
    draws = build_draws(20, 35, 9)
    assert draws == build_draws(20, 35, 9)
    assert len(draws) == 35
    assert set(draws) == set(range(20))
    with pytest.raises(ValueError):
        build_draws(20, 19, 9)


def test_residual_round_trip_and_iou() -> None:
    gold = [20, 30, 40, 50]
    predicted = [24, 25, 36, 55]
    replayed = apply_residual(gold, residual(gold, predicted))
    assert replayed == predicted
    assert xywh_iou(gold, predicted) > 0.5


def test_crop_shift_dataset_binds_source_and_obeys_policies(tmp_path) -> None:
    image_root = tmp_path / "images"
    for source, colour in (("facsimile", 0), ("rubbing", 255)):
        directory = image_root / source
        directory.mkdir(parents=True)
        image = Image.new("L", (40, 20), colour)
        image.save(directory / "sample.png")
    manifest = tmp_path / "crops.jsonl"
    base = {
        "id": "pair",
        "split": "train",
        "plate": "P1",
        "facsimile": "sample.png",
        "rubbing": "sample.png",
        "box": [0, 0, 20, 20],
        "label": "glyph",
        "class_id": 0,
        "crop_source": "facsimile",
        "matched_iou": 0.75,
        "paired_boxes": {"gold": [0, 0, 20, 20], "predicted": [20, 0, 20, 20]},
    }
    row = {
        **base,
        "crop_policy": {"name": "stochastic_mix", "predicted_probability": 1.0},
    }
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    dataset = CropShiftDataset(manifest, image_root, "train", 16, train=True)
    dataset.transform = dataset.rubbing_transform = lambda image: torch.tensor(float(image.convert("L").getextrema()[0]))
    image, target = dataset[0]
    assert float(image) == 0.0  # facsimile is used even though rubbing is white
    assert target == 0

    row["crop_policy"] = {"name": "iou_curriculum", "thresholds": [1.01, 0.70]}
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    curriculum = CropShiftDataset(manifest, image_root, "train", 16, train=True)
    assert curriculum._box_for_row(curriculum.rows[0]) == [0, 0, 20, 20]
    curriculum.set_epoch(1)
    assert curriculum._box_for_row(curriculum.rows[0]) == [20, 0, 20, 20]


def test_crop_shift_dataset_uses_full_vocabulary_reference_pool(tmp_path) -> None:
    image_root = tmp_path / "images"
    for source in ("facsimile", "rubbing"):
        directory = image_root / source
        directory.mkdir(parents=True)
        Image.new("L", (20, 20), 255).save(directory / "sample.png")
    reference_root = tmp_path / "glyph_reference_full"
    for label in ("seen", "reference-only"):
        directory = reference_root / label
        directory.mkdir(parents=True)
        Image.new("L", (10, 10), 255).save(directory / "one.png")
    (tmp_path / "vocab.json").write_text(
        json.dumps({"labels": ["seen", "reference-only"]}), encoding="utf-8"
    )
    manifest = tmp_path / "crops.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "id": "pair",
                "split": "train",
                "plate": "P1",
                "facsimile": "sample.png",
                "rubbing": "sample.png",
                "box": [0, 0, 10, 10],
                "label": "seen",
                "class_id": 0,
                "crop_source": "rubbing",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = CropShiftDataset(
        manifest,
        image_root,
        "train",
        16,
        train=True,
        include_reference_glyphs=True,
    )
    assert dataset.reference_samples == 2
    assert dataset.class_counts[0] == 2
    assert dataset.class_counts[1] == 1


def test_development_manifest_rejects_test_rows(tmp_path) -> None:
    safe = tmp_path / "safe.jsonl"
    safe.write_text('{"split":"train"}\n{"split":"val"}\n', encoding="utf-8")
    _assert_development_manifest(safe)
    unsafe = tmp_path / "unsafe.jsonl"
    unsafe.write_text('{"split":"test"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden test row"):
        _assert_development_manifest(unsafe)

