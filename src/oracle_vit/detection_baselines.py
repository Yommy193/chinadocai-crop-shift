"""Fair strong detector baselines on the shared ResNet-18 + FPN P2--P6 backbone.

Pre-registration: ``docs/BASELINE_COMPARISON_PREREGISTRATION_20260909.md``.

Two frozen architectures, both from scratch (``weights=None``), single visible-character
class, identical 512 px letterbox input and identical decode budget:

``fcos_r18_fpn_p2_p6``
    Standard anchor-free FCOS with centerness, on P2--P6.
``fasterrcnn_r18_fpn_p2_p6``
    Two-stage Faster R-CNN with an RPN over P2--P6 (anchor areas 16--256, ratios
    0.5/1/2) and a 128-channel multi-scale ROI head.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn
from torchvision.models.detection import FCOS, FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.fcos import FCOSHead
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.resnet import resnet18
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops.feature_pyramid_network import ExtraFPNBlock

ARCHITECTURES = ("fcos_r18_fpn_p2_p6", "fasterrcnn_r18_fpn_p2_p6")
FPN_CHANNELS = 128
FPN_LEVELS = ("0", "1", "2", "3", "4")  # P2..P6
INPUT_SIZE = 512


class LastLevelP6(ExtraFPNBlock):
    """Append a single stride-2 level named ``4`` so the pyramid stays P2--P6."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.p6 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.out_channels = out_channels

    def forward(self, results, d, names):  # noqa: ANN001, ANN201
        results.append(self.p6(results[-1]))
        names.append(FPN_LEVELS[-1])
        return results, names


def build_backbone_p2_p6(seed_reset: bool = True) -> BackboneWithFPN:
    """ResNet-18 body + FPN returning P2--P6, all at ``FPN_CHANNELS`` channels."""
    body = resnet18(weights=None)
    if seed_reset:
        for module in body.modules():
            if isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
    return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
    in_channels_list = [64, 128, 256, 512]
    backbone = BackboneWithFPN(
        body,
        return_layers=return_layers,
        in_channels_list=in_channels_list,
        out_channels=FPN_CHANNELS,
        extra_blocks=LastLevelP6(FPN_CHANNELS, FPN_CHANNELS),
    )
    backbone.out_channels = FPN_CHANNELS
    return backbone


def build_detector(architecture: str, num_classes: int) -> nn.Module:
    """Build one pre-registered baseline. ``num_classes`` excludes background for FCOS."""
    if architecture not in ARCHITECTURES:
        raise ValueError(f"unsupported baseline architecture: {architecture}")
    backbone = build_backbone_p2_p6()
    if architecture == "fcos_r18_fpn_p2_p6":
        anchor_generator = AnchorGenerator(
            sizes=((8,), (16,), (32,), (64,), (128,)),
            aspect_ratios=((1.0,),) * len(FPN_LEVELS),
        )
        model = FCOS(
            backbone,
            num_classes=num_classes + 1,
            anchor_generator=anchor_generator,
            head=FCOSHead(FPN_CHANNELS, anchor_generator.num_anchors_per_location()[0], num_classes + 1),
            min_size=INPUT_SIZE,
            max_size=INPUT_SIZE,
            detections_per_img=100,
            score_thresh=0.05,
            nms_thresh=0.45,
        )
        return model
    anchor_generator = AnchorGenerator(
        sizes=((16,), (32,), (64,), (128,), (256,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * len(FPN_LEVELS),
    )
    box_roi_pool = MultiScaleRoIAlign(featmap_names=list(FPN_LEVELS), output_size=7, sampling_ratio=2)
    model = FasterRCNN(
        backbone,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=box_roi_pool,
        box_head=TwoMLPHead(FPN_CHANNELS * 7 * 7, 1024),
        box_predictor=FastRCNNPredictor(1024, num_classes + 1),
        min_size=INPUT_SIZE,
        max_size=INPUT_SIZE,
        box_detections_per_img=100,
        box_score_thresh=0.05,
        box_nms_thresh=0.45,
        rpn_post_nms_top_n_test=1000,
        rpn_pre_nms_top_n_test=2000,
    )
    return model


def checkpoint_config(architecture: str, num_classes: int) -> dict[str, Any]:
    return {
        "architecture": architecture,
        "num_classes": num_classes,
        "external_pretraining": False,
        "input_size": INPUT_SIZE,
        "fpn_channels": FPN_CHANNELS,
        "fpn_levels": list(FPN_LEVELS),
        "score_threshold": 0.05,
        "top_k": 100,
        "nms_threshold": 0.45,
    }
