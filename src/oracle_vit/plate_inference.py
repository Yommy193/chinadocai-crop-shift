"""Inference helpers for the optional full-plate character detector.

The detector is deliberately kept separate from recognition: it proposes
visible-character crops on a full plate, while the existing recognition model
assigns character IDs to each crop.  Coordinates returned here are always in
the original image coordinate system.
"""

from __future__ import annotations

from typing import Any

import torch
from PIL import Image

from .anchor_free_detector import TinyPlateCenterNet, decode_detections, letterbox_image_and_boxes


def unletterbox_boxes(boxes: torch.Tensor, metadata: dict[str, float]) -> torch.Tensor:
    """Map xyxy boxes from the padded detector canvas back to the source image."""
    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError("boxes must have shape [N, 4]")
    scale = float(metadata["scale"])
    if scale <= 0:
        raise ValueError("letterbox scale must be positive")
    mapped = boxes.detach().clone().float()
    mapped[:, [0, 2]] = (mapped[:, [0, 2]] - float(metadata["left"])) / scale
    mapped[:, [1, 3]] = (mapped[:, [1, 3]] - float(metadata["top"])) / scale
    mapped[:, [0, 2]] = mapped[:, [0, 2]].clamp(0, float(metadata["original_width"]))
    mapped[:, [1, 3]] = mapped[:, [1, 3]].clamp(0, float(metadata["original_height"]))
    return mapped


def predict_plate_detections(
    model: TinyPlateCenterNet,
    image: torch.Tensor,
    input_size: int = 512,
    score_threshold: float = 0.05,
    top_k: int = 100,
    nms_threshold: float = 0.45,
) -> dict[str, Any]:
    """Run a frozen detector on one RGB tensor and return original-image boxes."""
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("image must be a 3xHxW tensor")
    device = next(model.parameters()).device
    canvas, _, metadata = letterbox_image_and_boxes(image.detach().cpu().float(), torch.empty((0, 4)), input_size)
    with torch.inference_mode():
        decoded = decode_detections(
            model(canvas.unsqueeze(0).to(device)),
            input_size,
            score_threshold=score_threshold,
            top_k=top_k,
            nms_threshold=nms_threshold,
        )[0]
    return {
        "boxes": unletterbox_boxes(decoded["boxes"].cpu(), metadata),
        "scores": decoded["scores"].detach().cpu(),
        "letterbox": metadata,
    }


def crop_xyxy(image: Image.Image, box: torch.Tensor, padding_fraction: float = 0.0) -> Image.Image:
    """Crop one xyxy box from a PIL image, with optional relative padding."""
    if box.numel() != 4:
        raise ValueError("box must contain four coordinates")
    if padding_fraction < 0:
        raise ValueError("padding_fraction must be non-negative")
    x1, y1, x2, y2 = (float(value) for value in box)
    pad_x, pad_y = (x2 - x1) * padding_fraction, (y2 - y1) * padding_fraction
    left = max(0, int(round(x1 - pad_x)))
    top = max(0, int(round(y1 - pad_y)))
    right = min(image.width, int(round(x2 + pad_x)))
    bottom = min(image.height, int(round(y2 + pad_y)))
    if right <= left or bottom <= top:
        raise ValueError("box is empty after clipping")
    return image.crop((left, top, right, bottom))
