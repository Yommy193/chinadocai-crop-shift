"""Experiment-local dataset for source-aligned crop-shift training.

This module is deliberately additive.  It does not change
``oracle_vit.data.OracleCropDataset`` or any historical manifest.  Development
manifests produced by ``build_crop_shift_alignment_data.py`` contain only train
and validation rows and use ``crop_source`` to bind every box to the image on
which the detector produced it.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from oracle_vit.data import OracleCropDataset


class CropShiftDataset(OracleCropDataset):
    """Oracle crop dataset with row-bound sources and paired crop policies."""

    def __init__(
        self,
        index_path: Path,
        image_root: Path,
        split: str,
        image_size: int,
        *,
        train: bool,
        validation_source: str = "rubbing",
        include_reference_glyphs: bool = False,
    ) -> None:
        if validation_source not in {"rubbing", "facsimile"}:
            raise ValueError("validation_source must be rubbing or facsimile")
        super().__init__(
            index_path,
            image_root,
            split,
            image_size,
            source=validation_source,
            train=train,
            include_reference_glyphs=False,
        )
        if train and include_reference_glyphs:
            vocabulary_path = index_path.parent / "vocab.json"
            vocabulary = json.loads(vocabulary_path.read_text(encoding="utf-8"))
            reference_root = image_root.parent / "glyph_reference_full"
            for class_id, label in enumerate(vocabulary["labels"]):
                directory = reference_root / str(label)
                if not directory.exists():
                    continue
                paths = sorted(directory.glob("*.png"))
                if paths:
                    self.reference_by_class[int(class_id)] = paths[0]
                for path in paths:
                    self.rows.append(
                        {
                            "_reference": str(path),
                            "label": str(label),
                            "class_id": int(class_id),
                        }
                    )
                    self.reference_samples += 1
            self.class_counts = Counter(int(row["class_id"]) for row in self.rows)
        self.crop_shift_train = bool(train)
        self.crop_shift_epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.crop_shift_epoch = int(epoch)

    def _box_for_row(self, row: dict[str, Any]) -> list[int]:
        paired = row.get("paired_boxes")
        if not self.crop_shift_train or paired is None:
            return list(row["box"])
        if not isinstance(paired, dict) or set(paired) != {"gold", "predicted"}:
            raise ValueError("paired_boxes must contain exactly gold and predicted")
        policy = row.get("crop_policy", {"name": "stochastic_mix", "predicted_probability": 0.5})
        name = str(policy.get("name", ""))
        if name == "stochastic_mix":
            probability = float(policy.get("predicted_probability", 0.5))
            if not 0.0 <= probability <= 1.0:
                raise ValueError("predicted_probability must be in [0, 1]")
            use_predicted = random.random() < probability
        elif name == "iou_curriculum":
            thresholds = [float(value) for value in policy.get("thresholds", [])]
            if not thresholds:
                raise ValueError("iou_curriculum requires thresholds")
            threshold = thresholds[min(self.crop_shift_epoch, len(thresholds) - 1)]
            use_predicted = float(row["matched_iou"]) >= threshold
        else:
            raise ValueError(f"unknown crop policy: {name}")
        key = "predicted" if use_predicted else "gold"
        box = paired[key]
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError(f"paired_boxes[{key!r}] must be a four-value list")
        return [int(value) for value in box]

    def __getitem__(self, index: int):  # noqa: ANN204 - mirrors the parent dataset variants
        row = self.rows[index]
        if "_reference" in row:
            return super().__getitem__(index)
        if not self.crop_shift_train:
            return super().__getitem__(index)
        source = str(row.get("crop_source", ""))
        if source not in {"rubbing", "facsimile"}:
            raise ValueError("training rows must declare crop_source as rubbing or facsimile")
        local = {**row, "box": self._box_for_row(row)}
        crop = self._crop_from_source(local, source)
        return self._transform_for_source(crop, source), int(row["class_id"])

