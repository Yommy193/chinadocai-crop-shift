"""Train a same-split, no-external-weights CNN baseline for OracleViT."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from oracle_vit.cnn_baseline import CnnConfig, build_cnn, checkpoint_config
from oracle_vit.data import OracleCropDataset
from oracle_vit.utils import atomic_json_dump, dataloader_worker_init, load_config, resolve_project_path, seed_everything


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def make_loader(dataset: OracleCropDataset, batch_size: int, workers: int, shuffle: bool, device: torch.device) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        worker_init_fn=dataloader_worker_init,
        drop_last=shuffle,
    )


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> dict[str, float]:
    model.eval()
    loss_sum = total = top1 = top5 = 0
    class_total: defaultdict[int, int] = defaultdict(int)
    class_correct: defaultdict[int, int] = defaultdict(int)
    for images, targets in loader:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, targets)
        predictions = logits.topk(min(5, logits.shape[-1]), dim=-1).indices
        correct1 = predictions[:, 0].eq(targets)
        correct5 = predictions.eq(targets[:, None]).any(dim=1)
        size = int(targets.numel())
        loss_sum += float(loss.item()) * size
        total += size
        top1 += int(correct1.sum().item())
        top5 += int(correct5.sum().item())
        for target, correct in zip(targets.cpu().tolist(), correct1.cpu().tolist()):
            class_total[int(target)] += 1
            class_correct[int(target)] += int(correct)
    macro = sum(class_correct[key] / value for key, value in class_total.items()) / max(1, len(class_total))
    return {
        "loss": loss_sum / max(1, total),
        "top1": top1 / max(1, total),
        "top5": top5 / max(1, total),
        "macro_top1": macro,
        "samples": float(total),
        "classes_seen": float(len(class_total)),
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    seed = int(args.seed if args.seed is not None else config["seed"])
    seed_everything(seed)
    data_config, train_config = config["data"], config["cnn"]
    processed = resolve_project_path(data_config["processed_dir"])
    output = resolve_project_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (processed / "vocab.json").open("r", encoding="utf-8") as handle:
        vocabulary = json.load(handle)
    image_size = int(args.image_size or data_config["image_size"])
    batch_size = int(args.batch_size or train_config["batch_size"])
    workers = int(args.num_workers if args.num_workers is not None else train_config["num_workers"])
    epochs = int(args.epochs or train_config["epochs"])
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    selection_source = args.selection_source or args.eval_source
    selection_sources = ("facsimile", "rubbing") if selection_source == "mean" else (selection_source,)
    train_set = OracleCropDataset(
        processed / "crops.jsonl", processed / "images", "train", image_size, source=args.source, train=True,
        include_reference_glyphs=bool(train_config.get("include_reference_glyphs", True)),
    )
    test_set = OracleCropDataset(processed / "crops.jsonl", processed / "images", "test", image_size, source=args.eval_source)
    loaders = {
        "train": make_loader(train_set, batch_size, workers, True, device),
        "test": make_loader(test_set, batch_size, workers, False, device),
    }
    validation_loaders = {
        source: make_loader(
            OracleCropDataset(processed / "crops.jsonl", processed / "images", "val", image_size, source=source),
            batch_size,
            workers,
            False,
            device,
        )
        for source in selection_sources
    }
    model_config = CnnConfig(args.architecture, len(vocabulary["labels"]), float(train_config["dropout"]))
    model = build_cnn(model_config).to(device)
    counts = torch.tensor([train_set.class_counts.get(index, 1) for index in range(len(vocabulary["labels"]))], dtype=torch.float32)
    weights = (counts.max() / counts).sqrt().clamp(max=5.0)
    weights = (weights / weights.mean()).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=float(train_config["label_smoothing"]))
    optimizer = AdamW(model.parameters(), lr=float(train_config["learning_rate"]), weight_decay=float(train_config["weight_decay"]), betas=(0.9, 0.95))
    steps = max(1, epochs * len(loaders["train"]))
    warmup = max(1, int(steps * 0.05))

    def schedule(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, schedule)
    best_top1, history = -1.0, []
    metadata = {
        "architecture": args.architecture,
        "seed": seed,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device),
        "image_size": image_size,
        "train_samples": len(train_set),
        "reference_samples": train_set.reference_samples,
        "epochs": epochs,
        "train_source": args.source,
        "eval_source": args.eval_source,
        "selection_source": selection_source,
        "external_pretraining": False,
    }
    print(json.dumps(metadata, ensure_ascii=False), flush=True)
    for epoch in range(epochs):
        model.train()
        started, running_loss, running_correct, seen = time.time(), 0.0, 0, 0
        for step, (images, targets) in enumerate(loaders["train"], start=1):
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running_loss += float(loss.item()) * targets.numel()
            running_correct += int(logits.detach().argmax(dim=-1).eq(targets).sum().item())
            seen += int(targets.numel())
            if step % args.log_every == 0 or step == len(loaders["train"]):
                print(f"epoch={epoch + 1}/{epochs} step={step}/{len(loaders['train'])} loss={running_loss / seen:.4f} top1={running_correct / seen:.4f} lr={scheduler.get_last_lr()[0]:.2e}", flush=True)
        validation_by_source = {
            source: evaluate(model, loader, criterion, device) for source, loader in validation_loaders.items()
        }
        validation = {
            key: sum(float(item[key]) for item in validation_by_source.values()) / len(validation_by_source)
            for key in next(iter(validation_by_source.values()))
        }
        epoch_result = {
            "epoch": epoch + 1,
            "train_loss": running_loss / max(1, seen),
            "train_top1": running_correct / max(1, seen),
            "val": validation,
            "val_by_source": validation_by_source,
            "seconds": time.time() - started,
        }
        history.append(epoch_result)
        print(json.dumps(epoch_result, ensure_ascii=False), flush=True)
        payload = {"model": model.state_dict(), "epoch": epoch, "best_top1": max(best_top1, validation["top1"]), "history": history, "architecture": args.architecture, "model_config": checkpoint_config(model_config), "vocab": vocabulary, "training": metadata}
        atomic_torch_save(output / "cnn_last.pt", payload)
        if validation["top1"] > best_top1:
            best_top1 = validation["top1"]
            payload["best_top1"] = best_top1
            atomic_torch_save(output / "cnn_best.pt", payload)
    best = torch.load(output / "cnn_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    summary = {"best_validation_top1": best_top1, "test": evaluate(model, loaders["test"], criterion, device), "history": history, "architecture": args.architecture, "model_config": checkpoint_config(model_config), "training": metadata}
    atomic_json_dump(output / "cnn_metrics.json", summary)
    print(json.dumps(summary["test"], ensure_ascii=False), flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train no-external-weights CNN baseline")
    parser.add_argument("--architecture", choices=("resnet18", "convnext_tiny"), required=True)
    parser.add_argument("--config", default="configs/cnn_baseline_20260829.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--seed", type=int, help="override the configuration seed for a controlled repeat")
    parser.add_argument("--source", choices=("both", "rubbing", "facsimile"), default="both", help="training image domain")
    parser.add_argument("--eval-source", choices=("rubbing", "facsimile"), default="rubbing", help="legacy single-domain validation/test output domain")
    parser.add_argument(
        "--selection-source",
        choices=("rubbing", "facsimile", "mean"),
        help="validation checkpoint-selection domain; mean averages facsimile and rubbing validation metrics",
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--log-every", type=int, default=100)
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
