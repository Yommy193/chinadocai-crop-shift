import torch

from oracle_vit.cnn_baseline import CnnConfig, build_cnn, checkpoint_config
from scripts.train_cnn_baseline import build_parser


def test_resnet18_baseline_has_expected_logits_and_config() -> None:
    config = CnnConfig("resnet18", num_classes=7, dropout=0.1)
    model = build_cnn(config).eval()
    with torch.inference_mode():
        logits = model(torch.zeros(2, 3, 64, 64))
    assert logits.shape == (2, 7)
    assert checkpoint_config(config)["architecture"] == "resnet18"


def test_cnn_training_parser_exposes_domain_and_balanced_selection_controls() -> None:
    parser = build_parser()
    defaults = parser.parse_args(["--architecture", "resnet18", "--output-dir", "artifact"])
    assert defaults.source == "both"
    assert defaults.selection_source is None
    custom = parser.parse_args(
        [
            "--architecture",
            "resnet18",
            "--output-dir",
            "artifact",
            "--source",
            "facsimile",
            "--selection-source",
            "mean",
        ]
    )
    assert custom.source == "facsimile"
    assert custom.selection_source == "mean"
