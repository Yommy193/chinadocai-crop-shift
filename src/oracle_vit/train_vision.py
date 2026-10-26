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
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, WeightedRandomSampler

from .data import OracleCropDataset
from .models import OracleViT, VisionConfig
from .utils import (
    atomic_json_dump,
    dataloader_worker_init,
    load_config,
    resolve_project_path,
    seed_everything,
    topk_accuracy,
)


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _loader(
    dataset: OracleCropDataset,
    batch_size: int,
    workers: int,
    shuffle: bool,
    device: torch.device,
    sampler: WeightedRandomSampler | None = None,
) -> DataLoader:
    if sampler is not None and shuffle:
        raise ValueError("sampler and shuffle cannot both be enabled")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        worker_init_fn=dataloader_worker_init,
        drop_last=shuffle or sampler is not None,
    )


@torch.inference_mode()
def evaluate(
    model: OracleViT,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    top1_correct = 0
    top5_correct = 0
    per_class_total: defaultdict[int, int] = defaultdict(int)
    per_class_correct: defaultdict[int, int] = defaultdict(int)
    autocast_enabled = device.type == "cuda"
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
            logits = model(images)
            loss = criterion(logits, targets)
        batch_size = targets.numel()
        total_loss += loss.item() * batch_size
        total += batch_size
        predictions = logits.topk(min(5, logits.shape[-1]), dim=-1).indices
        correct1 = predictions[:, 0].eq(targets)
        correct5 = predictions.eq(targets[:, None]).any(dim=1)
        top1_correct += int(correct1.sum().item())
        top5_correct += int(correct5.sum().item())
        for target, correct in zip(targets.cpu().tolist(), correct1.cpu().tolist()):
            per_class_total[target] += 1
            per_class_correct[target] += int(correct)
    macro_top1 = sum(per_class_correct[key] / count for key, count in per_class_total.items()) / max(
        1, len(per_class_total)
    )
    return {
        "loss": total_loss / max(1, total),
        "top1": top1_correct / max(1, total),
        "top5": top5_correct / max(1, total),
        "macro_top1": macro_top1,
        "samples": float(total),
        "classes_seen": float(len(per_class_total)),
    }


def reference_contrastive_loss(
    features: torch.Tensor,
    reference_features: torch.Tensor,
    targets: torch.Tensor,
    reference_mask: torch.Tensor,
    temperature: float,
    class_balanced: bool = False,
) -> torch.Tensor:
    """Multi-positive contrastive loss, optionally averaged equally over present classes."""
    if temperature <= 0:
        raise ValueError("reference contrastive temperature must be positive")
    if not bool(reference_mask.any()):
        return features.new_zeros(())
    selected_features = F.normalize(features[reference_mask], dim=-1)
    selected_references = F.normalize(reference_features[reference_mask], dim=-1)
    selected_targets = targets[reference_mask]
    similarities = selected_features @ selected_references.T / temperature
    same_class = selected_targets[:, None].eq(selected_targets[None, :])
    positive_logits = similarities.masked_fill(~same_class, float("-inf"))
    losses = -(torch.logsumexp(positive_logits, dim=1) - torch.logsumexp(similarities, dim=1))
    if not class_balanced:
        return losses.mean()
    # This is a narrow BCL-inspired intervention: every class represented in
    # the batch contributes one mean loss.  CE and sample frequencies stay fixed.
    _, inverse, counts = torch.unique(selected_targets, return_inverse=True, return_counts=True)
    by_class = torch.zeros_like(counts, dtype=losses.dtype).scatter_add_(0, inverse, losses)
    return (by_class / counts.to(losses.dtype)).mean()


def train(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    seed = int(config["seed"])
    seed_everything(seed)
    data_config = config["data"]
    train_config = config["vision"]
    processed_dir = resolve_project_path(data_config["processed_dir"])
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (processed_dir / "vocab.json").open("r", encoding="utf-8") as handle:
        vocabulary = json.load(handle)
    num_classes = len(vocabulary["labels"])
    image_root = processed_dir / "images"
    if not image_root.exists():
        raise FileNotFoundError(f"Prepared images not found: {image_root}. Run oracle-prepare first.")

    image_size = int(args.image_size or data_config["image_size"])
    batch_size = int(args.batch_size or train_config["batch_size"])
    workers = int(args.num_workers if args.num_workers is not None else train_config["num_workers"])
    epochs = int(args.epochs or train_config["epochs"])
    train_dataset = OracleCropDataset(
        processed_dir / "crops.jsonl",
        image_root,
        "train",
        image_size,
        source=args.source,
        train=True,
        include_reference_glyphs=bool(train_config.get("include_reference_glyphs", True))
        and not args.no_reference_glyphs,
        augmentation=args.augmentation,
        paired_views=args.consistency_weight > 0,
        paired_reference=args.reference_contrastive_weight > 0,
    )
    val_dataset = OracleCropDataset(
        processed_dir / "crops.jsonl", image_root, "val", image_size, source=args.eval_source, train=False
    )
    test_dataset = OracleCropDataset(
        processed_dir / "crops.jsonl", image_root, "test", image_size, source=args.eval_source, train=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
    model_config = VisionConfig(
        image_size=image_size,
        patch_size=int(train_config["patch_size"]),
        num_classes=num_classes,
        embed_dim=int(train_config["embed_dim"]),
        depth=int(train_config["depth"]),
        num_heads=int(train_config["num_heads"]),
        mlp_ratio=float(train_config["mlp_ratio"]),
        dropout=float(train_config["dropout"]),
    )
    model = OracleViT(model_config).to(device)

    if args.consistency_weight < 0:
        raise ValueError("consistency_weight must be non-negative")
    if args.reference_contrastive_weight < 0:
        raise ValueError("reference_contrastive_weight must be non-negative")
    if args.reference_contrastive_class_balanced and args.reference_contrastive_weight <= 0:
        raise ValueError("class-balanced reference contrastive loss requires a positive contrastive weight")
    if args.consistency_weight > 0 and args.source != "both":
        raise ValueError("跨模态一致性训练要求 --source both")
    if args.consistency_weight > 0 and args.reference_contrastive_weight > 0:
        raise ValueError("跨模态一致性与参考字形对比训练不能在同一轮同时启用")

    counts = torch.tensor([train_dataset.class_counts.get(index, 1) for index in range(num_classes)], dtype=torch.float32)
    class_weights: torch.Tensor | None
    if args.class_weighting == "sqrt":
        class_weights = (counts.max() / counts).sqrt().clamp(max=5.0)
        class_weights /= class_weights.mean()
        class_weights = class_weights.to(device)
    else:
        class_weights = None
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=float(train_config["label_smoothing"]),
    )
    optimizer = AdamW(
        model.parameters(),
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
        betas=(0.9, 0.95),
    )
    sampler: WeightedRandomSampler | None = None
    if args.sampling == "sqrt_balanced":
        sample_weights = torch.tensor(
            [1.0 / math.sqrt(max(1, train_dataset.class_counts[int(row["class_id"])])) for row in train_dataset.rows],
            dtype=torch.double,
        )
        generator = torch.Generator().manual_seed(seed)
        sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(train_dataset), replacement=True, generator=generator
        )
    train_loader = _loader(train_dataset, batch_size, workers, sampler is None, device, sampler=sampler)
    val_loader = _loader(val_dataset, batch_size, workers, False, device)
    test_loader = _loader(test_dataset, batch_size, workers, False, device)
    start_epoch = 0
    best_top1 = -1.0
    history: list[dict[str, Any]] = []
    resumed_from: str | None = None
    if args.resume:
        resume_path = resolve_project_path(args.resume)
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        if checkpoint.get("model_config") != model.checkpoint_config():
            raise ValueError("续训检查点的视觉模型结构与当前配置不一致")
        if checkpoint.get("vocab", {}).get("labels") != vocabulary["labels"]:
            raise ValueError("续训检查点的字头词表与当前数据不一致")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_top1 = float(checkpoint.get("best_top1", -1.0))
        history = list(checkpoint.get("history", []))
        resumed_from = str(resume_path)
        if not (output_dir / "vision_best.pt").exists():
            _atomic_torch_save(output_dir / "vision_best.pt", checkpoint)

    # Standard continuation preserves the global epoch horizon.  A controlled
    # fine-tuning experiment can instead retain model weights while restarting
    # AdamW and a cosine schedule over only the remaining epochs.
    scheduled_epochs = epochs if not args.restart_optimizer else max(1, epochs - start_epoch)
    total_steps = max(1, scheduled_epochs * len(train_loader))
    warmup_steps = max(1, int(total_steps * 0.05))

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    if args.restart_optimizer:
        optimizer = AdamW(
            model.parameters(),
            lr=float(train_config["learning_rate"]),
            weight_decay=float(train_config["weight_decay"]),
            betas=(0.9, 0.95),
        )
    scheduler = LambdaLR(optimizer, schedule)
    if args.resume and not args.restart_optimizer:
        # The target epoch count may be extended. Recalculate the cosine position
        # against the new horizon rather than retaining the old near-zero LR.
        resume_step = start_epoch * len(train_loader)
        scheduler.last_epoch = resume_step
        scheduler._step_count = resume_step + 1
        resumed_lrs = [base_lr * schedule(resume_step) for base_lr in scheduler.base_lrs]
        for group, learning_rate in zip(optimizer.param_groups, resumed_lrs):
            group["lr"] = learning_rate
        scheduler._last_lr = resumed_lrs

    print(
        json.dumps(
            {
                "device": str(device),
                "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "train_samples": len(train_dataset),
                "reference_samples": train_dataset.reference_samples,
                "val_samples": len(val_dataset),
                "test_samples": len(test_dataset),
                "classes": num_classes,
                "image_size": image_size,
                "target_epochs": epochs,
                "resumed_from": resumed_from,
                "augmentation": args.augmentation,
                "consistency_weight": args.consistency_weight,
                "consistency_temperature": args.consistency_temperature,
                "reference_contrastive_weight": args.reference_contrastive_weight,
                "reference_contrastive_temperature": args.reference_contrastive_temperature,
                "reference_contrastive_class_balanced": args.reference_contrastive_class_balanced,
                "sampling": args.sampling,
                "class_weighting": args.class_weighting,
                "restart_optimizer": args.restart_optimizer,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    for epoch in range(start_epoch, epochs):
        model.train()
        started = time.time()
        running_loss = 0.0
        running_correct = 0.0
        running_samples = 0
        for step, batch in enumerate(train_loader, start=1):
            if args.consistency_weight > 0:
                images, paired_images, targets, paired_mask = batch
                paired_images = paired_images.to(device, non_blocking=True)
                paired_mask = paired_mask.to(device, non_blocking=True)
                reference_images = None
                reference_mask = None
            elif args.reference_contrastive_weight > 0:
                images, reference_images, targets, reference_mask = batch
                reference_images = reference_images.to(device, non_blocking=True)
                reference_mask = reference_mask.to(device, non_blocking=True)
                paired_images = None
                paired_mask = None
            else:
                images, targets = batch
                paired_images = None
                paired_mask = None
                reference_images = None
                reference_mask = None
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                if reference_images is not None:
                    features = model.forward_features(images)
                    logits = model.head(features)
                else:
                    features = None
                    logits = model(images)
                loss = criterion(logits, targets)
                if paired_images is not None and paired_mask is not None and bool(paired_mask.any()):
                    paired_logits = model(paired_images)
                    temperature = float(args.consistency_temperature)
                    source_log_prob = F.log_softmax(logits[paired_mask] / temperature, dim=-1)
                    paired_log_prob = F.log_softmax(paired_logits[paired_mask] / temperature, dim=-1)
                    source_prob = source_log_prob.exp()
                    paired_prob = paired_log_prob.exp()
                    consistency = 0.5 * (
                        F.kl_div(source_log_prob, paired_prob, reduction="batchmean")
                        + F.kl_div(paired_log_prob, source_prob, reduction="batchmean")
                    ) * temperature**2
                    loss = loss + args.consistency_weight * consistency
                if reference_images is not None and reference_mask is not None and features is not None:
                    reference_features = model.forward_features(reference_images)
                    contrastive = reference_contrastive_loss(
                        features,
                        reference_features,
                        targets,
                        reference_mask,
                        float(args.reference_contrastive_temperature),
                        class_balanced=args.reference_contrastive_class_balanced,
                    )
                    loss = loss + args.reference_contrastive_weight * contrastive
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item() * targets.numel()
            running_correct += topk_accuracy(logits.detach(), targets, (1,))[0] * targets.numel()
            running_samples += targets.numel()
            if step % args.log_every == 0 or step == len(train_loader):
                print(
                    f"epoch={epoch + 1}/{epochs} step={step}/{len(train_loader)} "
                    f"loss={running_loss / running_samples:.4f} top1={running_correct / running_samples:.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e}",
                    flush=True,
                )

        validation = evaluate(model, val_loader, criterion, device)
        epoch_result = {
            "epoch": epoch + 1,
            "train_loss": running_loss / max(1, running_samples),
            "train_top1": running_correct / max(1, running_samples),
            "val": validation,
            "seconds": time.time() - started,
        }
        history.append(epoch_result)
        print(json.dumps(epoch_result, ensure_ascii=False), flush=True)
        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_top1": max(best_top1, validation["top1"]),
            "history": history,
            "model_config": model.checkpoint_config(),
            "vocab": vocabulary,
        }
        _atomic_torch_save(output_dir / "vision_last.pt", payload)
        if validation["top1"] > best_top1:
            best_top1 = validation["top1"]
            payload["best_top1"] = best_top1
            _atomic_torch_save(output_dir / "vision_best.pt", payload)

    best_checkpoint = torch.load(output_dir / "vision_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model"])
    test_metrics = evaluate(model, test_loader, criterion, device)
    summary = {
        "best_validation_top1": best_top1,
        "test": test_metrics,
        "history": history,
        "model_config": model.checkpoint_config(),
        "device": str(device),
        "training": {
            "target_epochs": epochs,
            "resumed_from": resumed_from,
            "augmentation": args.augmentation,
            "consistency_weight": args.consistency_weight,
            "consistency_temperature": args.consistency_temperature,
            "reference_contrastive_weight": args.reference_contrastive_weight,
            "reference_contrastive_temperature": args.reference_contrastive_temperature,
            "reference_contrastive_class_balanced": args.reference_contrastive_class_balanced,
            "sampling": args.sampling,
            "class_weighting": args.class_weighting,
            "restart_optimizer": args.restart_optimizer,
        },
    }
    atomic_json_dump(output_dir / "vision_metrics.json", summary)
    print(json.dumps(summary["test"], ensure_ascii=False), flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the oracle glyph ViT classifier")
    parser.add_argument("--config", default="configs/baseline.json")
    parser.add_argument("--output-dir", default="artifacts/checkpoints")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--image-size", type=int, help="override the configured square input size for a compatible checkpoint")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--source", choices=("both", "facsimile", "rubbing"), default="both")
    parser.add_argument("--eval-source", choices=("facsimile", "rubbing"), default="rubbing")
    parser.add_argument("--no-reference-glyphs", action="store_true")
    parser.add_argument(
        "--augmentation",
        choices=("standard", "rubbing_robust"),
        default="standard",
        help="rubbing_robust adds mild blur, autocontrast and noise to rubbing views only",
    )
    parser.add_argument(
        "--consistency-weight",
        type=float,
        default=0.0,
        help="symmetric KL weight between paired facsimile/rubbing crops; 0 disables it",
    )
    parser.add_argument("--consistency-temperature", type=float, default=2.0)
    parser.add_argument(
        "--reference-contrastive-weight",
        type=float,
        default=0.0,
        help="supervised contrastive weight between each crop and a same-class reference glyph; 0 disables it",
    )
    parser.add_argument("--reference-contrastive-temperature", type=float, default=0.1)
    parser.add_argument(
        "--reference-contrastive-class-balanced",
        action="store_true",
        help="average reference contrastive loss equally over classes present in each batch",
    )
    parser.add_argument(
        "--sampling",
        choices=("uniform", "sqrt_balanced"),
        default="uniform",
        help="uniform uses each source row once; sqrt_balanced tempers head-class dominance",
    )
    parser.add_argument(
        "--class-weighting",
        choices=("sqrt", "none"),
        default="sqrt",
        help="sqrt uses capped inverse-square-root loss weights; use none with balanced sampling",
    )
    parser.add_argument("--resume")
    parser.add_argument(
        "--restart-optimizer",
        action="store_true",
        help="load only model weights from --resume and restart AdamW/cosine over remaining epochs",
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--log-every", type=int, default=50)
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
