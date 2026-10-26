"""Evaluate a no-external-weights CNN checkpoint on a fixed image domain."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from oracle_vit.cnn_baseline import CnnConfig, build_cnn
from oracle_vit.data import OracleCropDataset
from oracle_vit.utils import atomic_json_dump, dataloader_worker_init, load_config, resolve_project_path


def metric_summary(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    predictions = logits.topk(min(5, logits.shape[-1]), dim=-1).indices
    correct1 = predictions[:, 0].eq(targets)
    correct5 = predictions.eq(targets[:, None]).any(dim=1)
    totals: defaultdict[int, int] = defaultdict(int)
    correct: defaultdict[int, int] = defaultdict(int)
    for target, value in zip(targets.tolist(), correct1.tolist()):
        totals[int(target)] += 1
        correct[int(target)] += int(value)
    macro = sum(correct[key] / count for key, count in totals.items()) / max(1, len(totals))
    return {"top1": float(correct1.float().mean()), "top5": float(correct5.float().mean()), "macro_top1": macro, "samples": float(targets.numel()), "classes_seen": float(len(totals))}


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, robust_tta: bool) -> dict[str, float]:
    logits_rows, target_rows = [], []
    model.eval()
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        views = [images]
        if robust_tta:
            views += [(images * 1.12).clamp(-1, 1), (images * 0.88).clamp(-1, 1), F.avg_pool2d(images, 3, 1, 1)]
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(torch.cat(views, dim=0))
        logits_rows.append(logits.float().reshape(len(views), images.shape[0], -1).mean(dim=0).cpu())
        target_rows.append(targets)
    return metric_summary(torch.cat(logits_rows), torch.cat(target_rows))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CNN baseline on a frozen split/domain")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source", choices=("rubbing", "facsimile"), required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--config", default="configs/cnn_baseline_20260829.json")
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--robust-tta", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    config = load_config(args.config)
    processed = resolve_project_path(config["data"]["processed_dir"])
    checkpoint_path = resolve_project_path(args.checkpoint)
    checkpoint: dict[str, Any] = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    with (processed / "vocab.json").open("r", encoding="utf-8") as handle:
        vocabulary = json.load(handle)
    if checkpoint["vocab"]["labels"] != vocabulary["labels"]:
        raise ValueError("checkpoint vocabulary does not match processed data")
    model = build_cnn(CnnConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model.to(device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
    dataset = OracleCropDataset(processed / "crops.jsonl", processed / "images", args.split, int(config["data"]["image_size"]), source=args.source)
    workers = int(args.num_workers)
    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=workers, pin_memory=device.type == "cuda", persistent_workers=workers > 0, worker_init_fn=dataloader_worker_init)
    result = {"checkpoint": str(checkpoint_path), "architecture": checkpoint["architecture"], "source": args.source, "split": args.split, "robust_tta": bool(args.robust_tta), "metrics": evaluate(model, loader, device, args.robust_tta)}
    if args.output:
        atomic_json_dump(resolve_project_path(args.output), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
