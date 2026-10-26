import torch
import pytest

from scripts.evaluate_cnn_checkpoint import metric_summary


def test_cnn_metric_summary_handles_macro_and_topk() -> None:
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0]])
    result = metric_summary(logits, torch.tensor([0, 1, 1]))
    assert result["top1"] == pytest.approx(2 / 3)
    assert result["top5"] == 1.0
    assert result["macro_top1"] == 0.75
