from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from .utils import atomic_json_dump, load_config, resolve_project_path


COLORS = ("#e63946", "#457b9d", "#2a9d8f", "#f4a261", "#8338ec", "#ff006e")


def _box(value: str) -> tuple[int, int, int, int]:
    x, y, width, height = (int(part) for part in value.split(","))
    return x, y, width, height


def _square_crop(image: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    x, y, width, height = box
    margin = max(2, int(max(width, height) * 0.12))
    crop = image.crop(
        (
            max(0, x - margin),
            max(0, y - margin),
            min(image.width, x + width + margin),
            min(image.height, y + height + margin),
        )
    )
    side = max(crop.size)
    horizontal, vertical = side - crop.width, side - crop.height
    return ImageOps.expand(
        crop,
        border=(horizontal // 2, vertical // 2, horizontal - horizontal // 2, vertical - vertical // 2),
        fill=(255, 255, 255),
    )


def inspect(config_path: str, plate_name: str, source: str, output_dir: str) -> dict[str, str]:
    config = load_config(config_path)
    raw_dir = resolve_project_path(config["data"]["raw_dir"])
    processed_dir = resolve_project_path(config["data"]["processed_dir"])
    destination = resolve_project_path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    with (raw_dir / "标签.json").open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    record = next((item for item in records if item["RubbingName"].lower() == plate_name.lower()), None)
    if record is None:
        raise ValueError(f"Unknown plate: {plate_name}")
    key = "Facsimile" if source == "facsimile" else "Rubbing"
    filename = Path(record[key]).name
    image_path = processed_dir / "images" / source / filename
    with Image.open(image_path) as handle:
        handle.load()
        plate = handle.convert("RGB")

    overlay = plate.copy()
    draw = ImageDraw.Draw(overlay)
    crop_rows = []
    first_group_boxes = []
    for group_index, group in enumerate(record["RecordUtilSentenceGroupVoList"]):
        color = COLORS[group_index % len(COLORS)]
        ordered = sorted(
            enumerate(group["RecordUtilOracleCharVoList"]),
            key=lambda item: (item[1].get("OrderNumber", 10**9), item[0]),
        )
        for local_index, (_, char) in enumerate(ordered):
            box = _box(char["Position"])
            x, y, width, height = box
            draw.rectangle((x, y, x + width, y + height), outline=color, width=max(2, plate.width // 500))
            draw.text((x, max(0, y - 12)), f"g{group_index}:{char.get('OrderNumber')}", fill=color)
            crop_rows.append((group_index, local_index, char, _square_crop(plate, box)))
            if group_index == 0:
                first_group_boxes.append(list(box))

    overlay_path = destination / f"{plate_name}_{source}_overlay.jpg"
    overlay.save(overlay_path, quality=92)
    cell = 150
    columns = min(8, max(1, len(crop_rows)))
    rows = (len(crop_rows) + columns - 1) // columns
    montage = Image.new("RGB", (columns * cell, rows * (cell + 32)), "white")
    montage_draw = ImageDraw.Draw(montage)
    for index, (group_index, local_index, char, crop) in enumerate(crop_rows):
        column, row = index % columns, index // columns
        resized = crop.convert("RGB").resize((cell, cell))
        left, top = column * cell, row * (cell + 32)
        montage.paste(resized, (left, top))
        label = char.get("Label") or "<missing>"
        montage_draw.text((left + 2, top + cell + 2), f"g{group_index}:{local_index} {label[:10]}", fill="black")
    montage_path = destination / f"{plate_name}_{source}_crops.jpg"
    montage.save(montage_path, quality=92)
    boxes_path = destination / f"{plate_name}_group0_boxes.json"
    atomic_json_dump(boxes_path, {"boxes": first_group_boxes})
    return {
        "plate": str(image_path.resolve()),
        "overlay": str(overlay_path.resolve()),
        "crops": str(montage_path.resolve()),
        "boxes": str(boxes_path.resolve()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a plate annotation and crop audit")
    parser.add_argument("--config", default="configs/baseline.json")
    parser.add_argument("--plate", default="H5")
    parser.add_argument("--source", choices=("facsimile", "rubbing"), default="rubbing")
    parser.add_argument("--output-dir", default="artifacts/data_audit")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(inspect(args.config, args.plate, args.source, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
