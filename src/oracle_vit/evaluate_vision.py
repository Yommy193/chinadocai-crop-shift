from __future__ import annotations

import argparse
import json

import torch
from torch import nn

from .data import OracleCropDataset
from .models import OracleViT, VisionConfig
from .train_vision import _loader, evaluate
from .utils import atomic_json_dump, resolve_project_path


def run(args: argparse.Namespace) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    checkpoint_path = resolve_project_path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = OracleViT(VisionConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    processed_dir = resolve_project_path(args.processed_dir)
    dataset = OracleCropDataset(
        processed_dir / "crops.jsonl",
        processed_dir / "images",
        args.split,
        int(checkpoint["model_config"]["image_size"]),
        source=args.source,
        train=False,
    )
    loader = _loader(dataset, args.batch_size, args.num_workers, False, device)
    metrics = evaluate(model, loader, nn.CrossEntropyLoss(), device)
    if args.output:
        atomic_json_dump(resolve_project_path(args.output), metrics)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained oracle ViT on one image domain")
    parser.add_argument("--checkpoint", default="artifacts/checkpoints/vision_best.pt")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--source", choices=("facsimile", "rubbing"), default="rubbing")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--output")
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

