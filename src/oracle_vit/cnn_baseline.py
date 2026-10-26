"""Comparable convolutional image-classification baselines for OracleViT."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from torch import nn
from torchvision.models import convnext_tiny, resnet18


@dataclass(frozen=True)
class CnnConfig:
    architecture: str
    num_classes: int
    dropout: float = 0.1


def build_cnn(config: CnnConfig) -> nn.Module:
    """Create a standard CNN without external pretrained weights."""
    if config.architecture == "resnet18":
        model = resnet18(weights=None, num_classes=config.num_classes)
        model.fc = nn.Sequential(nn.Dropout(config.dropout), nn.Linear(model.fc.in_features, config.num_classes))
        return model
    if config.architecture == "convnext_tiny":
        model = convnext_tiny(weights=None, num_classes=config.num_classes)
        # torchvision's default head is norm -> flatten -> linear.  Add only a
        # dropout layer so output semantics stay that of the standard model.
        classifier = list(model.classifier.children())
        classifier.insert(-1, nn.Dropout(config.dropout))
        model.classifier = nn.Sequential(*classifier)
        return model
    raise ValueError(f"unsupported CNN architecture: {config.architecture}")


def checkpoint_config(config: CnnConfig) -> dict[str, object]:
    return asdict(config)
