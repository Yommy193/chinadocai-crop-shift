from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageOps
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchvision import transforms

from .utils import read_jsonl


NORMALIZE_MEAN = (0.5, 0.5, 0.5)
NORMALIZE_STD = (0.5, 0.5, 0.5)


def _square_crop(image: Image.Image, box: list[int], margin_ratio: float = 0.12) -> Image.Image:
    image = ImageOps.exif_transpose(image)
    x, y, width, height = box
    margin = max(2, int(max(width, height) * margin_ratio))
    left = max(0, x - margin)
    top = max(0, y - margin)
    right = min(image.width, x + width + margin)
    bottom = min(image.height, y + height + margin)
    crop = image.crop((left, top, right, bottom))
    side = max(crop.width, crop.height)
    horizontal = side - crop.width
    vertical = side - crop.height
    return ImageOps.expand(
        crop,
        border=(horizontal // 2, vertical // 2, horizontal - horizontal // 2, vertical - vertical // 2),
        fill=(255, 255, 255) if crop.mode in {"RGB", "RGBA"} else 255,
    )


class RandomGaussianNoise:
    """Add mild sensor/scan noise after normalization with a fixed probability."""

    def __init__(self, probability: float = 0.18, std: float = 0.055) -> None:
        self.probability = probability
        self.std = std

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.probability:
            return image
        return (image + torch.randn_like(image) * self.std).clamp(-1.0, 1.0)


def build_image_transform(image_size: int, train: bool, robust_rubbing: bool = False) -> transforms.Compose:
    operations: list[Any] = [transforms.Grayscale(num_output_channels=3)]
    if train:
        operations.extend(
            [
                transforms.RandomAffine(
                    degrees=7,
                    translate=(0.06, 0.06),
                    scale=(0.9, 1.1),
                    fill=(255, 255, 255),
                ),
                transforms.RandomApply([transforms.ColorJitter(brightness=0.18, contrast=0.25)], p=0.5),
            ]
        )
        # These perturbations target scanning, wear and uneven rubbing contrast.
        # Horizontal flips are intentionally excluded: they can change the geometry
        # of a historical glyph rather than model an acquisition artifact.
        if robust_rubbing:
            operations.extend(
                [
                    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))], p=0.18),
                    transforms.RandomAutocontrast(p=0.12),
                ]
            )
    operations.extend(
        [
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        ]
    )
    if train and robust_rubbing:
        operations.append(RandomGaussianNoise())
    return transforms.Compose(operations)


class OracleCropDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self,
        index_path: Path,
        image_root: Path,
        split: str,
        image_size: int,
        source: str = "both",
        train: bool = False,
        include_reference_glyphs: bool = False,
        augmentation: str = "standard",
        paired_views: bool = False,
        paired_reference: bool = False,
    ) -> None:
        if source not in {"both", "facsimile", "rubbing"}:
            raise ValueError("source must be one of: both, facsimile, rubbing")
        if augmentation not in {"standard", "rubbing_robust"}:
            raise ValueError("augmentation must be one of: standard, rubbing_robust")
        if paired_views and (not train or source != "both"):
            raise ValueError("paired_views requires train=True and source='both'")
        if paired_reference and not train:
            raise ValueError("paired_reference requires train=True")
        if paired_views and paired_reference:
            raise ValueError("paired_views and paired_reference cannot be enabled together")
        self.rows = [row for row in read_jsonl(index_path) if row["split"] == split]
        self.image_root = image_root
        self.source = source
        self.train = train
        self.transform = build_image_transform(image_size, train=train)
        self.rubbing_transform = build_image_transform(
            image_size, train=train, robust_rubbing=augmentation == "rubbing_robust"
        )
        self.paired_views = paired_views
        self.paired_reference = paired_reference
        self.reference_samples = 0
        self.reference_by_class: dict[int, Path] = {}
        if train and include_reference_glyphs:
            reference_root = image_root.parent / "glyph_reference_full"
            class_by_label = {row["label"]: int(row["class_id"]) for row in self.rows}
            for label, class_id in class_by_label.items():
                directory = reference_root / label
                if not directory.exists():
                    continue
                paths = sorted(directory.glob("*.png"))
                if paths:
                    self.reference_by_class[class_id] = paths[0]
                for path in paths:
                    self.rows.append(
                        {
                            "_reference": str(path),
                            "label": label,
                            "class_id": class_id,
                        }
                    )
                    self.reference_samples += 1
        self.class_counts = Counter(int(row["class_id"]) for row in self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def _crop_from_source(self, row: dict[str, Any], source: str) -> Image.Image:
        filename_key = "facsimile" if source == "facsimile" else "rubbing"
        path = self.image_root / source / row[filename_key]
        with Image.open(path) as image:
            image.load()
            return _square_crop(image, row["box"])

    def _transform_for_source(self, crop: Image.Image, source: str) -> torch.Tensor:
        return self.rubbing_transform(crop) if source == "rubbing" else self.transform(crop)

    def _reference_tensor(self, row: dict[str, Any], fallback: torch.Tensor) -> tuple[torch.Tensor, bool]:
        path = Path(row["_reference"]) if "_reference" in row else self.reference_by_class.get(int(row["class_id"]))
        if path is None:
            return fallback.clone(), False
        with Image.open(path) as image:
            image.load()
            crop = _square_crop(image, [0, 0, image.width, image.height], margin_ratio=0.0)
        # The independent stochastic transform forms a second view even when
        # the row itself is a supplementary reference glyph.
        return self.transform(crop), True

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        if "_reference" in row:
            with Image.open(row["_reference"]) as image:
                image.load()
                crop = _square_crop(image, [0, 0, image.width, image.height], margin_ratio=0.0)
            tensor = self.transform(crop)
            if self.paired_views:
                return tensor, tensor.clone(), int(row["class_id"]), False
            if self.paired_reference:
                reference, available = self._reference_tensor(row, tensor)
                return tensor, reference, int(row["class_id"]), available
            return tensor, int(row["class_id"])
        source = self.source
        if source == "both":
            source = random.choice(("facsimile", "rubbing")) if self.train else "rubbing"
        crop = self._crop_from_source(row, source)
        tensor = self._transform_for_source(crop, source)
        if self.paired_views:
            paired_source = "rubbing" if source == "facsimile" else "facsimile"
            paired_crop = self._crop_from_source(row, paired_source)
            return tensor, self._transform_for_source(paired_crop, paired_source), int(row["class_id"]), True
        if self.paired_reference:
            reference, available = self._reference_tensor(row, tensor)
            return tensor, reference, int(row["class_id"]), available
        return tensor, int(row["class_id"])


class OracleSequenceDataset(Dataset[torch.Tensor]):
    def __init__(
        self,
        index_path: Path,
        label_to_token: dict[str, int],
        split: str,
        max_length: int,
        unk_id: int = 2,
    ) -> None:
        self.examples: list[list[int]] = []
        stride = max(1, max_length // 2)
        for row in read_jsonl(index_path):
            if row["split"] != split or row.get("ambiguous_order"):
                continue
            tokens = [label_to_token.get(label, unk_id) if label else unk_id for label in row["labels"]]
            if sum(token >= 3 for token in tokens) < 2:
                continue
            if len(tokens) <= max_length:
                self.examples.append(tokens)
                continue
            for start in range(0, len(tokens) - 1, stride):
                window = tokens[start : start + max_length]
                if sum(token >= 3 for token in window) >= 2:
                    self.examples.append(window)
                if start + max_length >= len(tokens):
                    break

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.tensor(self.examples[index], dtype=torch.long)


def collate_sequences(batch: list[torch.Tensor], pad_id: int = 0) -> dict[str, torch.Tensor]:
    tokens = pad_sequence(batch, batch_first=True, padding_value=pad_id)
    return {"tokens": tokens, "padding_mask": tokens.eq(pad_id)}
