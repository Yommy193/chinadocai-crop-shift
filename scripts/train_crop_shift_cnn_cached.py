"""Validation-only ResNet18 trainer backed by crop uint8 memmaps."""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from oracle_vit.cnn_baseline import CnnConfig, build_cnn, checkpoint_config
from oracle_vit.data import build_image_transform
from oracle_vit.utils import atomic_json_dump, dataloader_worker_init, resolve_project_path, seed_everything


class CachedCropDataset(Dataset):
    def __init__(self, cache: Path, split: str, source: str | None = None, train: bool = False, references: bool = False, policy_override: str | None = None) -> None:
        payload = json.loads((cache / "metadata.json").read_text(encoding="utf-8"))
        self.cache, self.image_size, self.train, self.split, self.source = cache, int(payload["image_size"]), train, split, source
        self.policy_override = policy_override
        entries = payload["train"] if split == "train" else payload["validation"][str(source)]
        self.rows = list(entries)
        self.gold = np.memmap(cache / ("train_gold.dat" if split == "train" else f"val_{source}.dat"), mode="r", dtype=np.uint8, shape=(len(self.rows), self.image_size, self.image_size, 3))
        self.pred = np.memmap(cache / "train_predicted.dat", mode="r", dtype=np.uint8, shape=self.gold.shape) if split == "train" else None
        self.gold_transform = build_image_transform(self.image_size, train=train)
        self.rubbing_transform = build_image_transform(self.image_size, train=train)
        self.crop_shift_epoch = 0
        self.reference_meta = payload.get("references", []) if references and split == "train" else []
        self.references = None
        if self.reference_meta:
            self.references = np.memmap(cache / "references.dat", mode="r", dtype=np.uint8, shape=(len(self.reference_meta), self.image_size, self.image_size, 3))
            self.rows.extend({"_reference_index": i, "class_id": int(row["class_id"]), "label": row["label"]} for i, row in enumerate(self.reference_meta))

    def __len__(self): return len(self.rows)
    def set_epoch(self, epoch: int) -> None: self.crop_shift_epoch = int(epoch)
    def _tensor(self, array: np.ndarray, source: str = "facsimile") -> torch.Tensor:
        transform = self.rubbing_transform if source == "rubbing" else self.gold_transform
        return transform(Image.fromarray(np.asarray(array), mode="RGB"))
    def __getitem__(self, index: int):
        row = self.rows[index]
        if "_reference_index" in row:
            return self._tensor(self.references[int(row["_reference_index"])]), int(row["class_id"])
        source = str(row.get("source", row.get("crop_source", self.source or "rubbing")))
        array = self.gold[index]
        if self.split == "train" and row.get("paired_boxes"):
            policy = row.get("crop_policy", {"name": "stochastic_mix", "predicted_probability": 0.5})
            if self.policy_override == "predicted":
                use_pred = True
            elif self.policy_override == "gold":
                use_pred = False
            elif self.policy_override == "mix075":
                use_pred = random.random() < 0.75
            elif policy.get("name") == "stochastic_mix":
                use_pred = random.random() < float(policy.get("predicted_probability", 0.5))
            elif policy.get("name") == "iou_curriculum":
                thresholds = [float(v) for v in policy.get("thresholds", [])]
                use_pred = float(row.get("matched_iou", 0.0)) >= thresholds[min(self.crop_shift_epoch, len(thresholds) - 1)]
            else:
                raise ValueError(f"unknown crop policy: {policy.get('name')}")
            if use_pred: array = self.pred[index]
        return self._tensor(array, source), int(row["class_id"])


def _loader(dataset, batch_size, workers, shuffle, device):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers, pin_memory=device.type == "cuda", persistent_workers=False, worker_init_fn=dataloader_worker_init, drop_last=shuffle)


@torch.inference_mode()
def evaluate(model, loader, criterion, device):
    model.eval(); loss_sum = total = top1 = top5 = 0; ct = defaultdict(int); cc = defaultdict(int)
    for images, targets in loader:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"): logits = model(images); loss = criterion(logits, targets)
        pred = logits.topk(min(5, logits.shape[-1]), dim=-1).indices; c1 = pred[:, 0].eq(targets); c5 = pred.eq(targets[:, None]).any(dim=1); n = targets.numel()
        loss_sum += float(loss.item()) * n; total += n; top1 += int(c1.sum()); top5 += int(c5.sum())
        for target, correct in zip(targets.cpu().tolist(), c1.cpu().tolist()): ct[int(target)] += 1; cc[int(target)] += int(correct)
    return {"loss": loss_sum / max(total, 1), "top1": top1 / max(total, 1), "top5": top5 / max(total, 1), "macro_top1": sum(cc[k] / v for k, v in ct.items()) / max(len(ct), 1), "samples": float(total), "classes_seen": float(len(ct))}


def main():
    p = argparse.ArgumentParser(description="train crop-shift ResNet18 from uint8 memmaps")
    p.add_argument("--config", default="configs/cnn_baseline_20260829.json"); p.add_argument("--cache-dir", required=True); p.add_argument("--class-counts", required=True); p.add_argument("--output-dir", required=True); p.add_argument("--seed", type=int, required=True)
    p.add_argument("--policy", choices=("manifest", "predicted", "gold", "mix075"), default="manifest", help="training crop policy override for shared caches")
    p.add_argument("--epochs", type=int); p.add_argument("--batch-size", type=int); p.add_argument("--num-workers", type=int); p.add_argument("--cpu", action="store_true"); p.add_argument("--log-every", type=int, default=100)
    args = p.parse_args(); seed_everything(args.seed); config = json.loads(resolve_project_path(args.config).read_text(encoding="utf-8")); cache = resolve_project_path(args.cache_dir); output = resolve_project_path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    train_cfg = config["cnn"]; device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda"); cache_payload = json.loads((cache / "metadata.json").read_text(encoding="utf-8")); image_size = int(cache_payload["image_size"]); batch = int(args.batch_size or train_cfg["batch_size"]); workers = int(args.num_workers if args.num_workers is not None else train_cfg["num_workers"]); epochs = int(args.epochs or train_cfg["epochs"])
    policy_override = None if args.policy == "manifest" else args.policy
    train_set = CachedCropDataset(cache, "train", train=True, references=bool(train_cfg.get("include_reference_glyphs", True)), policy_override=policy_override); val_sets = {s: CachedCropDataset(cache, "val", source=s) for s in ("facsimile", "rubbing")}; train_loader = _loader(train_set, batch, workers, True, device); val_loaders = {s: _loader(d, batch, workers, False, device) for s, d in val_sets.items()}
    counts = [int(v) for v in json.loads(resolve_project_path(args.class_counts).read_text(encoding="utf-8"))["counts"]]; model_cfg = CnnConfig("resnet18", len(counts), float(train_cfg["dropout"])); model = build_cnn(model_cfg).to(device); weights = torch.tensor([max(1, v) for v in counts], dtype=torch.float32); weights = (weights.max() / weights).sqrt().clamp(max=5.0); weights = (weights / weights.mean()).to(device); criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=float(train_cfg["label_smoothing"])); optimizer = AdamW(model.parameters(), lr=float(train_cfg["learning_rate"]), weight_decay=float(train_cfg["weight_decay"]), betas=(0.9, 0.95)); total_steps = max(1, epochs * len(train_loader)); warmup = max(1, int(total_steps * .05)); scheduler = LambdaLR(optimizer, lambda step: (step + 1) / warmup if step < warmup else .5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup))))
    vocab_path = Path(str(cache_payload["manifest"])).parent / "vocab.json"
    vocabulary = json.loads(vocab_path.read_text(encoding="utf-8"))
    metadata = {"experiment": "crop_shift_alignment_cached", "architecture": "resnet18", "seed": args.seed, "device": str(device), "image_size": image_size, "train_samples": len(train_set), "epochs": epochs, "steps_per_epoch": len(train_loader), "batch_size": batch, "num_workers": workers, "parameters": sum(p.numel() for p in model.parameters()), "cache_dir": str(cache), "policy": args.policy, "vocab_path": str(vocab_path), "test_access": "prohibited"}; print(json.dumps(metadata, ensure_ascii=False), flush=True); history = []; best = -1.0
    for epoch in range(epochs):
        train_set.set_epoch(epoch); model.train(); started = time.time(); loss_sum = correct = seen = 0
        for step, (images, targets) in enumerate(train_loader, 1):
            images, targets = images.to(device), targets.to(device); optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"): logits = model(images); loss = criterion(logits, targets)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); scheduler.step(); n = targets.numel(); loss_sum += float(loss.item()) * n; correct += int(logits.detach().argmax(-1).eq(targets).sum()); seen += n
            if step % args.log_every == 0 or step == len(train_loader): print(f"epoch={epoch+1}/{epochs} step={step}/{len(train_loader)} loss={loss_sum/seen:.4f} top1={correct/seen:.4f} lr={scheduler.get_last_lr()[0]:.2e}", flush=True)
        by_source = {s: evaluate(model, loader, criterion, device) for s, loader in val_loaders.items()}; val = {k: sum(m[k] for m in by_source.values()) / 2 for k in by_source["facsimile"]}; result = {"epoch": epoch + 1, "train_loss": loss_sum / max(seen, 1), "train_top1": correct / max(seen, 1), "val": val, "val_by_source": by_source, "seconds": time.time() - started}; history.append(result); print(json.dumps(result, ensure_ascii=False), flush=True)
        payload = {"model": model.state_dict(), "epoch": epoch, "best_top1": max(best, val["top1"]), "history": history, "architecture": "resnet18", "model_config": checkpoint_config(model_cfg), "vocab": vocabulary, "training": metadata}; torch.save(payload, output / "cnn_last.pt")
        if val["top1"] > best: best = val["top1"]; payload["best_top1"] = best; torch.save(payload, output / "cnn_best.pt")
    atomic_json_dump(output / "cnn_metrics.json", {"best_validation_top1": best, "test": None, "history": history, "training": metadata, "vocab_path": str(vocab_path)})


if __name__ == "__main__": main()
