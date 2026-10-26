"""A compact anchor-free detector for annotated character rectangles on plates."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision.ops import nms

from .plate_detection import PlateDetectionDataset


def letterbox_image_and_boxes(
    image: torch.Tensor, boxes: torch.Tensor, input_size: int
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Resize without changing aspect ratio, pad white, and map xyxy boxes."""
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("image must be a 3xHxW tensor")
    height, width = int(image.shape[1]), int(image.shape[2])
    scale = min(input_size / width, input_size / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    left, top = (input_size - resized_width) // 2, (input_size - resized_height) // 2
    resized = F.interpolate(image.unsqueeze(0), size=(resized_height, resized_width), mode="bilinear", align_corners=False, antialias=True).squeeze(0)
    canvas = torch.ones((3, input_size, input_size), dtype=image.dtype)
    canvas[:, top : top + resized_height, left : left + resized_width] = resized
    mapped = boxes.clone().float()
    if mapped.numel():
        mapped[:, [0, 2]] = mapped[:, [0, 2]] * scale + left
        mapped[:, [1, 3]] = mapped[:, [1, 3]] * scale + top
    metadata = {"scale": float(scale), "left": float(left), "top": float(top), "original_width": float(width), "original_height": float(height)}
    return canvas, mapped, metadata


def draw_center_targets(boxes: torch.Tensor, input_size: int, stride: int) -> dict[str, torch.Tensor]:
    """Build CenterNet-style heatmap, normalized size and subcell-offset targets."""
    if input_size % stride:
        raise ValueError("input_size must be divisible by stride")
    side = input_size // stride
    heatmap = torch.zeros((1, side, side), dtype=torch.float32)
    sizes = torch.zeros((2, side, side), dtype=torch.float32)
    offsets = torch.zeros((2, side, side), dtype=torch.float32)
    mask = torch.zeros((side, side), dtype=torch.bool)
    areas = torch.zeros((side, side), dtype=torch.float32)
    collisions = 0
    for box in boxes:
        x1, y1, x2, y2 = (float(value) for value in box)
        width, height = max(0.0, x2 - x1), max(0.0, y2 - y1)
        if width <= 0 or height <= 0:
            continue
        center_x, center_y = (x1 + x2) / 2.0 / stride, (y1 + y2) / 2.0 / stride
        grid_x, grid_y = min(side - 1, max(0, int(center_x))), min(side - 1, max(0, int(center_y)))
        radius = min(6, max(1, int(min(width, height) / stride / 5)))
        sigma = max(0.5, radius / 3.0)
        yy = torch.arange(max(0, grid_y - radius), min(side, grid_y + radius + 1), dtype=torch.float32)
        xx = torch.arange(max(0, grid_x - radius), min(side, grid_x + radius + 1), dtype=torch.float32)
        gaussian = torch.exp(-((yy[:, None] - grid_y).pow(2) + (xx[None, :] - grid_x).pow(2)) / (2.0 * sigma * sigma))
        heatmap[0, yy.long()[:, None], xx.long()[None, :]] = torch.maximum(heatmap[0, yy.long()[:, None], xx.long()[None, :]], gaussian)
        area = width * height
        if mask[grid_y, grid_x]:
            collisions += 1
        if not mask[grid_y, grid_x] or area >= float(areas[grid_y, grid_x]):
            sizes[:, grid_y, grid_x] = torch.tensor([width / input_size, height / input_size])
            offsets[:, grid_y, grid_x] = torch.tensor([center_x - grid_x, center_y - grid_y])
            areas[grid_y, grid_x] = area
            mask[grid_y, grid_x] = True
    return {"heatmap": heatmap, "sizes": sizes, "offsets": offsets, "mask": mask, "collisions": torch.tensor(collisions, dtype=torch.int64)}


class CenterPlateDataset(Dataset[tuple[torch.Tensor, dict[str, torch.Tensor]]]):
    """Letterboxed plate examples with dense targets for ``TinyPlateCenterNet``."""

    def __init__(
        self,
        raw_dir: str | Any,
        split: str,
        source: str,
        input_size: int = 512,
        plate_names: Sequence[str] | None = None,
        records: Sequence[dict[str, Any]] | None = None,
    ):
        self.base = PlateDetectionDataset(raw_dir, split, source, plate_names=plate_names, records=records)
        self.input_size = int(input_size)
        if self.input_size <= 0 or self.input_size % TinyPlateCenterNet.stride:
            raise ValueError("input_size must be positive and divisible by the detector stride")

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image, target = self.base[index]
        letterboxed, boxes, metadata = letterbox_image_and_boxes(image, target["boxes"], self.input_size)
        dense = draw_center_targets(boxes, self.input_size, TinyPlateCenterNet.stride)
        dense.update({"boxes": boxes, "image_id": target["image_id"], "letterbox": torch.tensor([metadata["scale"], metadata["left"], metadata["top"], metadata["original_width"], metadata["original_height"]], dtype=torch.float32)})
        return letterboxed, dense


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        groups = 8 if out_channels % 8 == 0 else 1
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.layers = nn.Sequential(ConvNormAct(channels, channels), nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.GroupNorm(8 if channels % 8 == 0 else 1, channels))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.silu(value + self.layers(value), inplace=True)


class TinyPlateCenterNet(nn.Module):
    """Output-stride-four one-class detector trained entirely from local plates."""

    stride = 4

    def __init__(self, width: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvNormAct(3, width // 2, stride=2),
            ResidualBlock(width // 2),
            ConvNormAct(width // 2, width, stride=2),
            ResidualBlock(width),
            ResidualBlock(width),
        )
        self.heatmap_head = nn.Sequential(ConvNormAct(width, width), nn.Conv2d(width, 1, 1))
        self.size_head = nn.Sequential(ConvNormAct(width, width), nn.Conv2d(width, 2, 1))
        self.offset_head = nn.Sequential(ConvNormAct(width, width), nn.Conv2d(width, 2, 1))
        nn.init.constant_(self.heatmap_head[-1].bias, -2.19)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encoder(images)
        return {"heatmap_logits": self.heatmap_head(features), "sizes_logits": self.size_head(features), "offsets_logits": self.offset_head(features)}


def detector_loss(
    outputs: dict[str, torch.Tensor],
    targets: Sequence[dict[str, torch.Tensor]],
    size_weight: float = 1.0,
    offset_weight: float = 1.0,
    box_iou_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Focal heatmap, center-only L1, and optional aligned-box IoU regression."""
    if box_iou_weight < 0:
        raise ValueError("box_iou_weight must be non-negative")
    heatmap_target = torch.stack([target["heatmap"] for target in targets])
    size_target = torch.stack([target["sizes"] for target in targets])
    offset_target = torch.stack([target["offsets"] for target in targets])
    mask = torch.stack([target["mask"] for target in targets])
    prediction = outputs["heatmap_logits"].sigmoid().clamp(1e-4, 1.0 - 1e-4)
    positive = heatmap_target.eq(1.0)
    negative = ~positive
    negative_weight = (1.0 - heatmap_target).pow(4)
    positive_loss = -(prediction.log() * (1.0 - prediction).pow(2) * positive).sum()
    negative_loss = -((1.0 - prediction).log() * prediction.pow(2) * negative_weight * negative).sum()
    normalizer = positive.sum().clamp(min=1).float()
    heatmap_loss = (positive_loss + negative_loss) / normalizer
    mask_expanded = mask[:, None].expand_as(size_target)
    size_prediction = outputs["sizes_logits"].sigmoid()
    offset_prediction = outputs["offsets_logits"].sigmoid()
    size_loss = F.l1_loss(size_prediction[mask_expanded], size_target[mask_expanded], reduction="sum") / normalizer
    offset_loss = F.l1_loss(offset_prediction[mask_expanded], offset_target[mask_expanded], reduction="sum") / normalizer
    box_iou_loss = heatmap_loss.new_zeros(())
    if box_iou_weight:
        # Both target size and offsets are normalized to the letterboxed canvas.
        # Reconstruct the single positive box at every center cell in FP32 so the
        # IoU term is stable under mixed-precision training.
        _, _, height, width = size_prediction.shape
        grid_y, grid_x = torch.meshgrid(
            torch.arange(height, device=size_prediction.device, dtype=torch.float32),
            torch.arange(width, device=size_prediction.device, dtype=torch.float32),
            indexing="ij",
        )
        pred_width, pred_height = size_prediction[:, 0].float(), size_prediction[:, 1].float()
        target_width, target_height = size_target[:, 0].float(), size_target[:, 1].float()
        pred_center_x = (grid_x + offset_prediction[:, 0].float()) / float(width)
        pred_center_y = (grid_y + offset_prediction[:, 1].float()) / float(height)
        target_center_x = (grid_x + offset_target[:, 0].float()) / float(width)
        target_center_y = (grid_y + offset_target[:, 1].float()) / float(height)
        pred_x1, pred_y1 = pred_center_x - pred_width / 2, pred_center_y - pred_height / 2
        pred_x2, pred_y2 = pred_center_x + pred_width / 2, pred_center_y + pred_height / 2
        target_x1, target_y1 = target_center_x - target_width / 2, target_center_y - target_height / 2
        target_x2, target_y2 = target_center_x + target_width / 2, target_center_y + target_height / 2
        intersection_width = (torch.minimum(pred_x2, target_x2) - torch.maximum(pred_x1, target_x1)).clamp(min=0)
        intersection_height = (torch.minimum(pred_y2, target_y2) - torch.maximum(pred_y1, target_y1)).clamp(min=0)
        intersection = intersection_width * intersection_height
        pred_area = (pred_x2 - pred_x1).clamp(min=0) * (pred_y2 - pred_y1).clamp(min=0)
        target_area = (target_x2 - target_x1).clamp(min=0) * (target_y2 - target_y1).clamp(min=0)
        aligned_iou = intersection / (pred_area + target_area - intersection).clamp(min=1e-6)
        box_iou_loss = (1.0 - aligned_iou)[mask].sum() / normalizer
    total = heatmap_loss + size_weight * size_loss + offset_weight * offset_loss + box_iou_weight * box_iou_loss
    return {"loss": total, "heatmap": heatmap_loss, "size": size_loss, "offset": offset_loss, "box_iou": box_iou_loss}


def decode_detections(outputs: dict[str, torch.Tensor], input_size: int, score_threshold: float = 0.05, top_k: int = 100, nms_threshold: float = 0.45) -> list[dict[str, torch.Tensor]]:
    """Decode local heatmap peaks to xyxy boxes on the letterboxed input canvas."""
    heatmaps = outputs["heatmap_logits"].sigmoid()
    heatmaps = heatmaps * heatmaps.eq(F.max_pool2d(heatmaps, kernel_size=3, stride=1, padding=1))
    sizes = outputs["sizes_logits"].sigmoid()
    offsets = outputs["offsets_logits"].sigmoid()
    batch, _, height, width = heatmaps.shape
    stride = input_size / height
    results: list[dict[str, torch.Tensor]] = []
    for index in range(batch):
        flat = heatmaps[index, 0].flatten()
        scores, positions = flat.topk(min(top_k, flat.numel()))
        keep = scores >= score_threshold
        scores, positions = scores[keep], positions[keep]
        if not scores.numel():
            results.append({"boxes": torch.empty((0, 4), device=flat.device), "scores": scores})
            continue
        grid_y, grid_x = torch.div(positions, width, rounding_mode="floor"), positions % width
        selected_sizes = sizes[index, :, grid_y, grid_x].transpose(0, 1) * input_size
        selected_offsets = offsets[index, :, grid_y, grid_x].transpose(0, 1)
        centers_x = (grid_x.float() + selected_offsets[:, 0]) * stride
        centers_y = (grid_y.float() + selected_offsets[:, 1]) * stride
        boxes = torch.stack((centers_x - selected_sizes[:, 0] / 2, centers_y - selected_sizes[:, 1] / 2, centers_x + selected_sizes[:, 0] / 2, centers_y + selected_sizes[:, 1] / 2), dim=1)
        boxes = boxes.clamp(min=0, max=float(input_size))
        selected = nms(boxes, scores, nms_threshold)
        results.append({"boxes": boxes[selected], "scores": scores[selected]})
    return results
