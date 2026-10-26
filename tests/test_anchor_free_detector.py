from __future__ import annotations

import pytest
import torch
from PIL import Image

from oracle_vit.anchor_free_detector import decode_detections, detector_loss, draw_center_targets, letterbox_image_and_boxes
from oracle_vit.plate_inference import crop_xyxy, unletterbox_boxes


def test_letterbox_preserves_box_geometry() -> None:
    image = torch.zeros((3, 100, 200))
    boxes = torch.tensor([[20.0, 10.0, 120.0, 90.0]])
    _, mapped, metadata = letterbox_image_and_boxes(image, boxes, 200)
    assert metadata["scale"] == pytest.approx(1.0)
    assert metadata["top"] == 50.0
    assert mapped.tolist() == [[20.0, 60.0, 120.0, 140.0]]


def test_center_targets_marks_center_and_collision() -> None:
    targets = draw_center_targets(torch.tensor([[0.0, 0.0, 16.0, 16.0], [1.0, 1.0, 15.0, 15.0]]), 32, 4)
    assert targets["heatmap"][0, 2, 2] == 1.0
    assert targets["mask"].sum().item() == 1
    assert targets["collisions"].item() == 1


def test_detector_loss_and_decode_have_expected_shapes() -> None:
    target = draw_center_targets(torch.tensor([[8.0, 8.0, 24.0, 24.0]]), 32, 4)
    outputs = {
        "heatmap_logits": torch.full((1, 1, 8, 8), -8.0),
        "sizes_logits": torch.zeros((1, 2, 8, 8)),
        "offsets_logits": torch.zeros((1, 2, 8, 8)),
    }
    outputs["heatmap_logits"][0, 0, 4, 4] = 8.0
    losses = detector_loss(outputs, [target])
    assert torch.isfinite(losses["loss"])
    rows = decode_detections(outputs, 32, score_threshold=0.5, top_k=3)
    assert rows[0]["boxes"].shape[1] == 4
    assert rows[0]["scores"].numel() == 1


def test_aligned_box_iou_loss_is_zero_for_exact_geometry_and_positive_when_shifted() -> None:
    target = draw_center_targets(torch.tensor([[9.0, 9.0, 25.0, 25.0]]), 32, 4)
    grid_y, grid_x = target["mask"].nonzero(as_tuple=True)
    y, x = int(grid_y.item()), int(grid_x.item())
    outputs = {
        "heatmap_logits": torch.zeros((1, 1, 8, 8)),
        "sizes_logits": torch.zeros((1, 2, 8, 8)),
        "offsets_logits": torch.zeros((1, 2, 8, 8)),
    }
    outputs["sizes_logits"][0, :, y, x] = torch.logit(target["sizes"][:, y, x])
    outputs["offsets_logits"][0, :, y, x] = torch.logit(target["offsets"][:, y, x])
    exact = detector_loss(outputs, [target], box_iou_weight=1.0)
    assert exact["box_iou"].item() == pytest.approx(0.0, abs=1e-5)
    shifted = {key: value.clone() for key, value in outputs.items()}
    shifted["offsets_logits"][0, 0, y, x] = torch.logit(torch.tensor(0.75))
    shifted_loss = detector_loss(shifted, [target], box_iou_weight=1.0)
    assert shifted_loss["box_iou"].item() > 0.1
    assert shifted_loss["loss"].item() > exact["loss"].item()


def test_zero_iou_weight_is_strictly_equivalent_to_the_default_loss() -> None:
    target = draw_center_targets(torch.tensor([[8.0, 8.0, 24.0, 24.0]]), 32, 4)
    outputs = {
        "heatmap_logits": torch.randn((1, 1, 8, 8), generator=torch.Generator().manual_seed(7)),
        "sizes_logits": torch.randn((1, 2, 8, 8), generator=torch.Generator().manual_seed(8)),
        "offsets_logits": torch.randn((1, 2, 8, 8), generator=torch.Generator().manual_seed(9)),
    }
    default = detector_loss(outputs, [target])
    explicit_zero = detector_loss(outputs, [target], box_iou_weight=0.0)
    for key in default:
        assert explicit_zero[key].item() == pytest.approx(default[key].item(), abs=0.0)


def test_iou_loss_has_finite_gradients_for_nonoverlapping_aligned_boxes() -> None:
    target = draw_center_targets(torch.tensor([[8.02, 8.02, 8.10, 8.10]]), 32, 4)
    grid_y, grid_x = target["mask"].nonzero(as_tuple=True)
    y, x = int(grid_y.item()), int(grid_x.item())
    sizes = torch.zeros((1, 2, 8, 8), requires_grad=True)
    offsets = torch.zeros((1, 2, 8, 8), requires_grad=True)
    heatmap = torch.zeros((1, 1, 8, 8), requires_grad=True)
    with torch.no_grad():
        sizes[0, :, y, x] = torch.logit(target["sizes"][:, y, x])
        offsets[0, :, y, x] = torch.logit(torch.tensor([0.99, 0.99]))
    result = detector_loss({"heatmap_logits": heatmap, "sizes_logits": sizes, "offsets_logits": offsets}, [target], box_iou_weight=1.0)
    result["loss"].backward()
    assert result["box_iou"].item() == pytest.approx(1.0, abs=1e-5)
    assert all(torch.isfinite(value.grad).all() for value in (heatmap, sizes, offsets))


@pytest.mark.parametrize("input_size", [512, 640, 768])
def test_decode_preserves_normalized_geometry_across_input_resolutions(input_size: int) -> None:
    """Guard the resolution ablation against a decoder-coordinate scaling bug."""
    side = input_size // 4
    center_x, center_y, width, height = 0.503, 0.397, 0.10, 0.08
    grid_x, grid_y = int(center_x * side), int(center_y * side)
    offset = torch.tensor([center_x * side - grid_x, center_y * side - grid_y])
    outputs = {
        "heatmap_logits": torch.full((1, 1, side, side), -20.0),
        "sizes_logits": torch.zeros((1, 2, side, side)),
        "offsets_logits": torch.zeros((1, 2, side, side)),
    }
    outputs["heatmap_logits"][0, 0, grid_y, grid_x] = 20.0
    outputs["sizes_logits"][0, :, grid_y, grid_x] = torch.logit(torch.tensor([width, height]))
    outputs["offsets_logits"][0, :, grid_y, grid_x] = torch.logit(offset)
    decoded = decode_detections(outputs, input_size, score_threshold=0.5, top_k=1)[0]["boxes"][0]
    assert (decoded / input_size).tolist() == pytest.approx(
        [center_x - width / 2, center_y - height / 2, center_x + width / 2, center_y + height / 2],
        abs=1e-5,
    )


def test_unletterbox_and_crop_use_original_coordinates() -> None:
    metadata = {"scale": 2.0, "left": 10.0, "top": 20.0, "original_width": 100.0, "original_height": 80.0}
    boxes = unletterbox_boxes(torch.tensor([[10.0, 20.0, 50.0, 80.0]]), metadata)
    assert boxes.tolist() == [[0.0, 0.0, 20.0, 30.0]]
    image = Image.new("RGB", (100, 80), "white")
    crop = crop_xyxy(image, boxes[0], padding_fraction=0.1)
    assert crop.size == (22, 33)
