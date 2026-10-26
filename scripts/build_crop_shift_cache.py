"""Build uint8 memmap crops for the additive FCOS crop-shift experiment.

The manifest is never modified.  Train rows retain both gold and predicted
boxes; validation rows are materialized for each validation source.  Images
are stored as HWC RGB uint8 arrays so the trainer can apply the historical
torchvision augmentations without reopening source files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from oracle_vit.data import _square_crop


def _rooted(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else Path.cwd() / value


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _crop(image_root: Path, row: dict[str, Any], source: str, box: list[int], image_size: int) -> np.ndarray:
    filename = row["facsimile" if source == "facsimile" else "rubbing"]
    with Image.open(image_root / source / filename) as image:
        image.load()
        crop = _square_crop(image.convert("RGB"), [int(v) for v in box])
        crop = crop.resize((image_size, image_size), Image.Resampling.LANCZOS)
        return np.asarray(crop, dtype=np.uint8)


def _write(path: Path, arrays: list[np.ndarray], image_size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mm = np.memmap(path, mode="w+", dtype=np.uint8, shape=(len(arrays), image_size, image_size, 3))
    if arrays:
        mm[:] = np.stack(arrays, axis=0)
    mm.flush()
    del mm


def main() -> None:
    parser = argparse.ArgumentParser(description="cache FCOS crop-shift crops as uint8 memmaps")
    parser.add_argument("--manifest", required=True, help="FCOS crops.jsonl (development manifest; no test rows)")
    parser.add_argument("--image-root", help="manifest image root; defaults to manifest sibling/images")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", type=int, default=144)
    parser.add_argument("--include-references", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    manifest = _rooted(args.manifest).resolve()
    output = _rooted(args.output_dir).resolve()
    image_root = _rooted(args.image_root).resolve() if args.image_root else manifest.parent / "images"
    rows = _rows(manifest)
    if any(row.get("split") == "test" for row in rows):
        raise ValueError("refusing a manifest containing test rows")
    if output.exists() and any(output.iterdir()) and not args.force:
        raise FileExistsError(f"non-empty output directory: {output}; use --force to replace cache files")
    output.mkdir(parents=True, exist_ok=True)
    train = [row for row in rows if row.get("split") == "train"]
    val = [row for row in rows if row.get("split") == "val"]
    image_size = int(args.image_size)
    if image_size < 2:
        raise ValueError("image-size must be >= 2")
    train_gold, train_pred = [], []
    train_meta = []
    for row_index, row in enumerate(train, start=1):
        paired = row.get("paired_boxes")
        if not isinstance(paired, dict) or "gold" not in paired or "predicted" not in paired:
            raise ValueError("every train row must contain paired_boxes gold and predicted")
        source = str(row.get("crop_source", ""))
        if source not in {"facsimile", "rubbing"}:
            raise ValueError("every train row must declare crop_source")
        train_gold.append(_crop(image_root, row, source, paired["gold"], image_size))
        train_pred.append(_crop(image_root, row, source, paired["predicted"], image_size))
        train_meta.append({k: row[k] for k in row if k not in {"paired_boxes"}} | {"paired_boxes": paired, "source": source})
        if row_index % 1000 == 0:
            print(f"cached train rows={row_index}/{len(train)}", flush=True)
    _write(output / "train_gold.dat", train_gold, image_size)
    _write(output / "train_predicted.dat", train_pred, image_size)
    val_meta = {}
    for source in ("facsimile", "rubbing"):
        # Validation is evaluated against both acquisition sources; the
        # development manifest rows carry both filenames and one gold box.
        source_rows = val
        arrays = [_crop(image_root, row, source, row["box"], image_size) for row in source_rows]
        _write(output / f"val_{source}.dat", arrays, image_size)
        val_meta[source] = [{k: row[k] for k in row if k != "paired_boxes"} for row in source_rows]
    reference_meta = []
    if args.include_references:
        vocabulary = json.loads((manifest.parent / "vocab.json").read_text(encoding="utf-8"))
        refs, reference_meta = [], []
        ref_root = manifest.parent / "glyph_reference_full"
        for class_id, label in enumerate(vocabulary["labels"]):
            for path in sorted((ref_root / str(label)).glob("*.png")) if (ref_root / str(label)).exists() else []:
                with Image.open(path) as image:
                    image.load()
                    crop = _square_crop(image.convert("RGB"), [0, 0, image.width, image.height], margin_ratio=0.0)
                    refs.append(np.asarray(crop.resize((image_size, image_size), Image.Resampling.LANCZOS), dtype=np.uint8))
                reference_meta.append({"class_id": class_id, "label": label, "path": str(path)})
        _write(output / "references.dat", refs, image_size)
    metadata = {"schema_version": 1, "dtype": "uint8", "layout": "HWC_RGB", "image_size": image_size,
                "manifest": str(manifest), "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "image_root": str(image_root), "train": train_meta, "validation": val_meta,
                "references": reference_meta, "files": {"train_gold": "train_gold.dat", "train_predicted": "train_predicted.dat"}}
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "train": len(train), "validation": {k: len(v) for k, v in val_meta.items()}, "references": len(reference_meta)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
