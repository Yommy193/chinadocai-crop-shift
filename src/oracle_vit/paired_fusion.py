"""Small trainable head for aligned facsimile/rubbing ResNet features."""

from __future__ import annotations

import torch
from torch import nn


class PairedFeatureFusionHead(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int, dropout: float = 0.15) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(feature_dim * 2),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, facsimile_features: torch.Tensor, rubbing_features: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat((facsimile_features, rubbing_features), dim=-1))


def resnet18_features(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Extract frozen pre-classifier features from the project's torchvision ResNet-18."""
    if not all(hasattr(model, name) for name in ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4", "avgpool")):
        raise ValueError("paired feature fusion currently requires a torchvision ResNet-18 checkpoint")
    output = model.conv1(images)
    output = model.bn1(output)
    output = model.relu(output)
    output = model.maxpool(output)
    output = model.layer1(output)
    output = model.layer2(output)
    output = model.layer3(output)
    output = model.layer4(output)
    return torch.flatten(model.avgpool(output), 1)
