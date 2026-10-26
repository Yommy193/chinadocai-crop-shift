"""Detector-family-aware plate inference for the strong-detector end-to-end comparison.

The frozen pipeline module (``oracle_vit.plate_inference``) only accepts the project's
TinyPlateCenterNet checkpoints, and that restriction is deliberate - it is part of the frozen P0
confirmation. Rather than editing it, this module adds an adapter with exactly the same call signature and
return shape so the pre-registered comparison
(`docs/STRONG_DETECTOR_END_TO_END_PREREGISTRATION_20260918.md`) can feed a standard torchvision detector
through the identical cropping / recognition / ordering pipeline:

* ``load_detector_any_family`` accepts both the anchor family and the torchvision baselines
  (FCOS / Faster R-CNN) built by ``oracle_vit.detection_baselines``, still refusing any checkpoint that
  declares external pretraining;
* ``predict_plate_detections`` keeps the anchor path byte-identical by delegating to the frozen function,
  and implements the torchvision path with the same letterbox geometry, the same frozen decode contract
  (score 0.05 inside the model, NMS 0.45, at most 100 detections per image) and the same
  operating-point filtering (score threshold, top-k) applied *after* that contract, which is the order the
  anchor path uses as well.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from oracle_vit.anchor_free_detector import TinyPlateCenterNet, letterbox_image_and_boxes
from oracle_vit.detection_baselines import ARCHITECTURES, build_detector
from oracle_vit.plate_inference import predict_plate_detections as anchor_predict
from oracle_vit.plate_inference import unletterbox_boxes

ANCHOR_ARCHITECTURE = "tiny_anchor_free_centernet"


def load_detector_any_family(path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load a sealed checkpoint of either detector family without touching the frozen loader."""
    state = torch.load(path, map_location=device, weights_only=False)
    model_config = dict(state["model_config"])
    if bool(model_config.get("external_pretraining")):
        raise ValueError("the runner only accepts from-scratch checkpoints")
    architecture = str(model_config.get("architecture", ""))
    if architecture == ANCHOR_ARCHITECTURE:
        model: torch.nn.Module = TinyPlateCenterNet(width=int(model_config["width"]))
        family = "tiny_anchor_free_centernet"
    elif architecture in ARCHITECTURES:
        model = build_detector(architecture, 1)
        family = "torchvision_detection"
    else:
        raise ValueError(f"unsupported detector architecture: {architecture!r}")
    model.load_state_dict(state["model_state"])
    enriched = dict(state)
    enriched["detector_family"] = family
    return model.to(device).eval(), enriched


def predict_plate_detections(
    model: torch.nn.Module,
    image: torch.Tensor,
    input_size: int = 512,
    score_threshold: float = 0.05,
    top_k: int = 100,
    nms_threshold: float = 0.45,
) -> dict[str, Any]:
    """Same contract as the frozen predictor, with a torchvision branch for the standard baselines."""
    if isinstance(model, TinyPlateCenterNet):
        return anchor_predict(model, image, input_size=input_size, score_threshold=score_threshold,
                              top_k=top_k, nms_threshold=nms_threshold)
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("image must be a 3xHxW tensor")
    device = next(model.parameters()).device
    canvas, _, metadata = letterbox_image_and_boxes(image.detach().cpu().float(), torch.empty((0, 4)),
                                                    input_size)
    with torch.inference_mode():
        output = model([canvas.to(device)])[0]
    boxes = output["boxes"].detach().cpu().float()
    scores = output["scores"].detach().cpu().float()
    keep = scores >= float(score_threshold)
    boxes, scores = boxes[keep], scores[keep]
    if boxes.shape[0] > int(top_k):
        order = torch.argsort(scores, descending=True)[: int(top_k)]
        boxes, scores = boxes[order], scores[order]
    return {
        "boxes": unletterbox_boxes(boxes, metadata),
        "scores": scores,
        "letterbox": metadata,
    }
