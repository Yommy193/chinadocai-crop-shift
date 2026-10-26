"""Contract tests for the detector-family adapter used by the strong-detector end-to-end comparison.

These stay on CPU and use tiny synthetic inputs: they pin the adapter's public contract - which checkpoints
it accepts, that the anchor family still goes through the frozen predictor, and that the torchvision branch
applies the operating-point threshold and the top-k cap *after* the model's own decode budget.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from oracle_vit.anchor_free_detector import TinyPlateCenterNet  # noqa: E402
from oracle_vit.plate_inference_strong import (  # noqa: E402
    load_detector_any_family,
    predict_plate_detections,
)


class StubTorchvisionDetector(torch.nn.Module):
    """Mimics a torchvision detection model's eval-mode output (already NMS-filtered)."""

    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward(self, images: list[torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        return [{
            "boxes": torch.tensor([[10.0, 12.0, 50.0, 52.0], [200.0, 200.0, 260.0, 260.0]]),
            "scores": torch.tensor([0.90, 0.20]),
            "labels": torch.tensor([1, 1]),
        } for _ in images]


def write_checkpoint(path: Path, architecture: str, external_pretraining: bool, model: torch.nn.Module) -> Path:
    torch.save({"model_config": {"architecture": architecture, "width": 32,
                                 "external_pretraining": external_pretraining},
                "model_state": model.state_dict()}, path)
    return path


def test_loader_rejects_external_pretraining(tmp_path: Path) -> None:
    path = write_checkpoint(tmp_path / "ckpt.pt", "tiny_anchor_free_centernet", True, TinyPlateCenterNet(width=32))
    with pytest.raises(ValueError, match="from-scratch"):
        load_detector_any_family(path, torch.device("cpu"))


def test_loader_rejects_unknown_architecture(tmp_path: Path) -> None:
    path = write_checkpoint(tmp_path / "ckpt.pt", "not_a_detector", False, TinyPlateCenterNet(width=32))
    with pytest.raises(ValueError, match="unsupported detector architecture"):
        load_detector_any_family(path, torch.device("cpu"))


def test_loader_accepts_the_anchor_family(tmp_path: Path) -> None:
    model = TinyPlateCenterNet(width=32)
    path = write_checkpoint(tmp_path / "ckpt.pt", "tiny_anchor_free_centernet", False, model)
    loaded, state = load_detector_any_family(path, torch.device("cpu"))
    assert isinstance(loaded, TinyPlateCenterNet)
    assert state["detector_family"] == "tiny_anchor_free_centernet"
    assert not loaded.training


def test_anchor_family_delegates_to_the_frozen_predictor() -> None:
    model = TinyPlateCenterNet(width=32).eval()
    result = predict_plate_detections(model, torch.zeros(3, 64, 64), input_size=64)
    assert set(result) == {"boxes", "scores", "letterbox"}
    assert result["boxes"].shape[1] == 4
    assert result["scores"].shape[0] == result["boxes"].shape[0]


def test_torchvision_branch_applies_threshold_and_top_k() -> None:
    model = StubTorchvisionDetector().eval()
    result = predict_plate_detections(model, torch.zeros(3, 128, 128), input_size=128,
                                      score_threshold=0.5, top_k=100)
    assert result["scores"].tolist() == pytest.approx([0.90])
    assert result["boxes"].shape == (1, 4)
    capped = predict_plate_detections(model, torch.zeros(3, 128, 128), input_size=128,
                                      score_threshold=0.1, top_k=1)
    assert capped["scores"].tolist() == pytest.approx([0.90])


def test_torchvision_branch_keeps_boxes_inside_the_original_image() -> None:
    model = StubTorchvisionDetector().eval()
    image = torch.zeros(3, 200, 400)
    result = predict_plate_detections(model, image, input_size=128)
    boxes = result["boxes"]
    assert boxes.shape[0] >= 1
    assert float(boxes[:, 0].min()) >= 0.0
    assert float(boxes[:, 2].max()) <= 400.0
    assert float(boxes[:, 3].max()) <= 200.0
