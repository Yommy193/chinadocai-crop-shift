import json

from PIL import Image

from oracle_vit.data import OracleCropDataset


def test_paired_rubbing_dataset_returns_two_views(tmp_path) -> None:
    image_root = tmp_path / "images"
    for source in ("facsimile", "rubbing"):
        directory = image_root / source
        directory.mkdir(parents=True)
        Image.new("RGB", (20, 12), "white").save(directory / "sample.png")
    index_path = tmp_path / "crops.jsonl"
    row = {
        "split": "train",
        "label": "glyph-id",
        "class_id": 0,
        "facsimile": "sample.png",
        "rubbing": "sample.png",
        "box": [1, 1, 12, 8],
    }
    index_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    dataset = OracleCropDataset(
        index_path,
        image_root,
        "train",
        16,
        train=True,
        augmentation="rubbing_robust",
        paired_views=True,
    )
    source, paired, target, is_real_pair = dataset[0]
    assert source.shape == paired.shape == (3, 16, 16)
    assert target == 0
    assert is_real_pair is True
