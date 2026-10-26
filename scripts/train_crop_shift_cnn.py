"""Validation-only CNN trainer for the additive crop-shift experiment."""

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

try:
    from scripts.crop_shift_dataset import CropShiftDataset
except ModuleNotFoundError:  # direct ``python scripts/train_crop_shift_cnn.py`` execution
    from crop_shift_dataset import CropShiftDataset
from oracle_vit.cnn_baseline import CnnConfig, build_cnn, checkpoint_config
from oracle_vit.utils import (
    atomic_json_dump,
    dataloader_worker_init,
    file_sha256,
    load_config,
    resolve_project_path,
    seed_everything,
)


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def make_loader(
    dataset: CropShiftDataset,
    batch_size: int,
    workers: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    # Workers are recreated at every epoch so set_epoch() is visible to the
    # IoU curriculum without shared mutable state.
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
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


def _assert_development_manifest(path: Path) -> None:
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") == "test":
                raise ValueError(f"development manifest contains a forbidden test row at line {number}")


def train(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    seed_everything(int(args.seed))
    processed = resolve_project_path(args.processed_dir)
    output = resolve_project_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = processed / "crops.jsonl"
    _assert_development_manifest(manifest)
    with (processed / "vocab.json").open("r", encoding="utf-8") as handle:
        vocabulary = json.load(handle)
    class_count_path = resolve_project_path(args.class_counts)
    class_count_payload = json.loads(class_count_path.read_text(encoding="utf-8"))
    frozen_counts = [int(value) for value in class_count_payload["counts"]]
    if len(frozen_counts) != len(vocabulary["labels"]):
        raise ValueError("frozen class-count vector does not match the vocabulary")

    train_config = config["cnn"]
    image_size = int(args.image_size or config["data"]["image_size"])
    batch_size = int(args.batch_size or train_config["batch_size"])
    workers = int(args.num_workers if args.num_workers is not None else train_config["num_workers"])
    epochs = int(args.epochs or train_config["epochs"])
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    train_set = CropShiftDataset(
        manifest,
        processed / "images",
        "train",
        image_size,
        train=True,
        include_reference_glyphs=bool(train_config.get("include_reference_glyphs", True)),
    )
    validation_sets = {
        source: CropShiftDataset(
            manifest,
            processed / "images",
            "val",
            image_size,
            train=False,
            validation_source=source,
        )
        for source in ("facsimile", "rubbing")
    }
    train_loader = make_loader(train_set, batch_size, workers, True, device)
    validation_loaders = {
        source: make_loader(dataset, batch_size, workers, False, device)
        for source, dataset in validation_sets.items()
    }
    model_config = CnnConfig("resnet18", len(vocabulary["labels"]), float(train_config["dropout"]))
    model = build_cnn(model_config).to(device)
    counts = torch.tensor([max(1, value) for value in frozen_counts], dtype=torch.float32)
    weights = (counts.max() / counts).sqrt().clamp(max=5.0)
    weights = (weights / weights.mean()).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=float(train_config["label_smoothing"]))
    optimizer = AdamW(
        model.parameters(),
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
        betas=(0.9, 0.95),
    )
    total_steps = max(1, epochs * len(train_loader))
    warmup_steps = max(1, int(total_steps * 0.05))

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, schedule)
    metadata = {
        "experiment": "crop_shift_alignment_20261009",
        "architecture": "resnet18",
        "seed": int(args.seed),
        "device": str(device),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "image_size": image_size,
        "train_samples": len(train_set),
        "train_manifest_rows": len([row for row in train_set.rows if "_reference" not in row]),
        "reference_samples": train_set.reference_samples,
        "epochs": epochs,
        "steps_per_epoch": len(train_loader),
        "total_optimizer_steps": total_steps,
        "selection_source": "mean(facsimile,rubbing) on gold validation crops",
        "processed_dir": str(processed),
        "manifest_sha256": file_sha256(manifest),
        "class_counts_sha256": file_sha256(class_count_path),
        "test_access": "prohibited; the development manifest contains no test rows and no test dataset is constructed",
    }
    print(json.dumps(metadata, ensure_ascii=False), flush=True)

    best_top1 = -1.0
    history: list[dict[str, Any]] = []
    for epoch in range(epochs):
        train_set.set_epoch(epoch)
        model.train()
        started = time.time()
        running_loss = 0.0
        running_correct = 0
        seen = 0
        for step, (images, targets) in enumerate(train_loader, start=1):
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
            if step % int(args.log_every) == 0 or step == len(train_loader):
                print(
                    f"epoch={epoch + 1}/{epochs} step={step}/{len(train_loader)} "
                    f"loss={running_loss / seen:.4f} top1={running_correct / seen:.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e}",
                    flush=True,
                )
        by_source = {
            source: evaluate(model, loader, criterion, device)
            for source, loader in validation_loaders.items()
        }
        validation = {
            key: sum(float(metrics[key]) for metrics in by_source.values()) / len(by_source)
            for key in next(iter(by_source.values()))
        }
        result = {
            "epoch": epoch + 1,
            "train_loss": running_loss / max(1, seen),
            "train_top1": running_correct / max(1, seen),
            "val": validation,
            "val_by_source": by_source,
            "seconds": time.time() - started,
        }
        history.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        payload = {
            "model": model.state_dict(),
            "epoch": epoch,
            "best_top1": max(best_top1, validation["top1"]),
            "history": history,
            "architecture": "resnet18",
            "model_config": checkpoint_config(model_config),
            "vocab": vocabulary,
            "training": metadata,
        }
        atomic_torch_save(output / "cnn_last.pt", payload)
        if validation["top1"] > best_top1:
            best_top1 = validation["top1"]
            payload["best_top1"] = best_top1
            atomic_torch_save(output / "cnn_best.pt", payload)

    summary = {
        "best_validation_top1": best_top1,
        "test": None,
        "history": history,
        "architecture": "resnet18",
        "model_config": checkpoint_config(model_config),
        "training": metadata,
    }
    atomic_json_dump(output / "cnn_metrics.json", summary)
    print(json.dumps({"best_validation_top1": best_top1, "test_access": "prohibited"}, ensure_ascii=False))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a source-aligned crop-shift CNN without test access")
    parser.add_argument("--config", default="configs/cnn_baseline_20260829.json")
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--class-counts", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--log-every", type=int, default=100)
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())

