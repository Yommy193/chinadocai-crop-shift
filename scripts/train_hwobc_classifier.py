"""Train an independent closed-set classifier on the local HWOBC handwriting set.

The official ``data_train.lst`` and ``data_val.lst`` files define the split.  This
experiment is deliberately standalone: HWOBC labels are numeric Unicode-like IDs
and are not silently mapped to the anonymised 1,730-class vocabulary of the main
plate corpus.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from oracle_vit.cnn_baseline import CnnConfig, build_cnn, checkpoint_config
from oracle_vit.data import build_image_transform

ROOT = Path(__file__).resolve().parents[1]


def read_list(path: Path) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                raise ValueError(f"malformed list row {path}:{line_number}")
            label = int(float(parts[1]))
            relative = parts[2].replace("\\", "/")
            rows.append((relative, label))
    return rows


def prepare_cache(
    root: Path,
    rows: list[tuple[str, int]],
    label_to_class: dict[int, int],
    image_size: int,
    cache_dir: Path,
    split: str,
) -> tuple[Path, Path, int]:
    """Decode/resize once into an additive memory-mapped cache.

    The source HWOBC tree is never modified.  Keeping the cache in the extension
    directory removes repeated PNG decoding from every epoch, which is important
    on systems where the GPU otherwise waits for the input pipeline.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    image_path = cache_dir / f"{split}_{image_size}.npy"
    label_path = cache_dir / f"{split}_{image_size}_labels.npy"
    meta_path = cache_dir / f"{split}_{image_size}.json"
    existing = [(relative, label_to_class[label]) for relative, label in rows if label in label_to_class and (root / relative).is_file()]
    if image_path.exists() and label_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if int(meta.get("samples", -1)) == len(existing):
            return image_path, label_path, len(existing)
    images = np.lib.format.open_memmap(image_path, mode="w+", dtype=np.uint8, shape=(len(existing), image_size, image_size))
    labels = np.lib.format.open_memmap(label_path, mode="w+", dtype=np.int64, shape=(len(existing),))
    for index, (relative, target) in enumerate(existing):
        with Image.open(root / relative) as image:
            image = ImageOps.exif_transpose(image).convert("L").resize((image_size, image_size), Image.Resampling.LANCZOS)
            images[index] = np.asarray(image, dtype=np.uint8)
        labels[index] = int(target)
        if (index + 1) % 10000 == 0:
            print(json.dumps({"cache_split": split, "decoded": index + 1, "total": len(existing)}, ensure_ascii=False), flush=True)
    images.flush()
    labels.flush()
    meta_path.write_text(json.dumps({"samples": len(existing), "image_size": image_size, "source": str(root), "split": split}, ensure_ascii=False, indent=2), encoding="utf-8")
    return image_path, label_path, len(existing)


class CachedHWOBCDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, image_path: Path, label_path: Path):
        self.images = np.load(image_path, mmap_mode="r")
        self.labels = np.load(label_path, mmap_mode="r")

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        # The cache stores a single grayscale plane; replicate it to the three
        # channels expected by the unmodified ResNet-18 baseline and normalize to
        # the same [-1, 1] range as the main recognition pipeline.
        plane = torch.from_numpy(np.array(self.images[index], copy=True)).float().div_(127.5).sub_(1.0)
        return plane.unsqueeze(0).expand(3, -1, -1), int(self.labels[index])


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> dict[str, float]:
    model.eval()
    loss_sum = total = top1 = top5 = 0
    class_total: defaultdict[int, int] = defaultdict(int)
    class_correct: defaultdict[int, int] = defaultdict(int)
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, targets)
        k = min(5, logits.shape[-1])
        predictions = logits.topk(k, dim=-1).indices
        correct1 = predictions[:, 0].eq(targets)
        correct5 = predictions.eq(targets[:, None]).any(dim=1)
        size = int(targets.numel())
        total += size
        loss_sum += float(loss.item()) * size
        top1 += int(correct1.sum().item())
        top5 += int(correct5.sum().item())
        for target, correct in zip(targets.cpu().tolist(), correct1.cpu().tolist()):
            class_total[int(target)] += 1
            class_correct[int(target)] += int(correct)
    macro = sum(class_correct[c] / class_total[c] for c in class_total) / max(1, len(class_total))
    return {
        "loss": loss_sum / max(1, total),
        "top1": top1 / max(1, total),
        "top5": top5 / max(1, total),
        "macro_top1": macro,
        "samples": float(total),
        "classes_seen": float(len(class_total)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Independent HWOBC handwritten-character classifier")
    parser.add_argument("--data-root", default="data/hwobc/sample")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    data_root = Path(args.data_root).resolve()
    output = (ROOT / args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    train_rows_raw = read_list(data_root / "data_train.lst")
    val_rows_raw = read_list(data_root / "data_val.lst")
    train_rows = [(p, y) for p, y in train_rows_raw if (data_root / p).is_file()]
    val_rows = [(p, y) for p, y in val_rows_raw if (data_root / p).is_file()]
    labels = sorted({y for _, y in train_rows_raw} | {y for _, y in val_rows_raw})
    label_to_class = {label: index for index, label in enumerate(labels)}
    cache_dir = (ROOT / args.output_dir / "cache").resolve()
    train_image_path, train_label_path, train_cached = prepare_cache(data_root, train_rows, label_to_class, args.image_size, cache_dir, "train")
    val_image_path, val_label_path, val_cached = prepare_cache(data_root, val_rows, label_to_class, args.image_size, cache_dir, "val")
    train = CachedHWOBCDataset(train_image_path, train_label_path)
    val = CachedHWOBCDataset(val_image_path, val_label_path)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
    loader_kwargs = {"batch_size": args.batch_size, "num_workers": args.num_workers, "pin_memory": device.type == "cuda"}
    if args.num_workers > 0:
        loader_kwargs.update({"persistent_workers": True, "prefetch_factor": args.prefetch_factor})
    train_loader = DataLoader(train, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val, shuffle=False, **loader_kwargs)
    config = CnnConfig("resnet18", len(labels), 0.1)
    model = build_cnn(config).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.05, betas=(0.9, 0.95))
    metadata = {
        "dataset": "HWOBC-local",
        "data_root": str(data_root),
        "train_list": str(data_root / "data_train.lst"),
        "val_list": str(data_root / "data_val.lst"),
        "official_split": True,
        "train_rows_listed": len(train_rows_raw),
        "train_rows_existing": len(train_rows),
        "val_rows_listed": len(val_rows_raw),
        "val_rows_existing": len(val_rows),
        "skipped_train_missing": len(train_rows_raw) - len(train_rows),
        "skipped_val_missing": len(val_rows_raw) - len(val_rows),
        "classes": len(labels),
        "image_size": args.image_size,
        "seed": args.seed,
        "device": str(device),
        "external_pretraining": False,
        "cache_dir": str(cache_dir),
        "cached_train_samples": train_cached,
        "cached_val_samples": val_cached,
        "augmentation": "none after deterministic 96x96 cache; cache is additive and source data are read-only",
    }
    (output / "dataset_manifest.json").write_text(json.dumps({**metadata, "labels": labels}, ensure_ascii=False, indent=2), encoding="utf-8")
    history = []
    best = -1.0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = seen = correct = 0
        epoch_started = time.time()
        for step, (images, targets) in enumerate(train_loader, start=1):
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            size = int(targets.numel())
            running += float(loss.item()) * size
            seen += size
            correct += int(logits.detach().argmax(-1).eq(targets).sum().item())
            if step % 100 == 0 or step == len(train_loader):
                print(json.dumps({"epoch": epoch, "step": step, "steps": len(train_loader), "loss": running / seen, "train_top1": correct / seen}, ensure_ascii=False), flush=True)
        metrics = evaluate(model, val_loader, criterion, device)
        row = {"epoch": epoch, "train_loss": running / max(1, seen), "train_top1": correct / max(1, seen), "val": metrics, "seconds": time.time() - epoch_started}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        payload = {"model": model.state_dict(), "epoch": epoch, "best_val_top1": max(best, metrics["top1"]), "history": history, "architecture": "resnet18", "model_config": checkpoint_config(config), "vocab": {"labels": [str(label) for label in labels], "label_to_token": {str(label): i for i, label in enumerate(labels)}}, "training": metadata}
        torch.save(payload, output / "cnn_last.pt")
        if metrics["top1"] > best:
            best = metrics["top1"]
            payload["best_val_top1"] = best
            torch.save(payload, output / "cnn_best.pt")
    summary = {"dataset": "HWOBC-local", "best_validation_top1": best, "best_validation_top5": max(r["val"]["top5"] for r in history), "history": history, "training": {**metadata, "wall_seconds": time.time() - started}, "test": None, "test_access": "not_applicable; official train/val split only"}
    (output / "cnn_metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best_validation_top1": best, "best_validation_top5": summary["best_validation_top5"], "seconds": summary["training"]["wall_seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
