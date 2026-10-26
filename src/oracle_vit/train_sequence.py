from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from .data import OracleSequenceDataset, collate_sequences
from .models import OracleSequenceModel, SequenceConfig
from .utils import (
    atomic_json_dump,
    dataloader_worker_init,
    load_config,
    resolve_project_path,
    seed_everything,
)


IGNORE_INDEX = -100


def make_masked_batch(
    tokens: torch.Tensor,
    mask_probability: float,
    vocab_size: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    eligible = tokens.ge(3)
    selected = torch.rand(tokens.shape, generator=generator).lt(mask_probability) & eligible
    for row in range(tokens.shape[0]):
        if not selected[row].any() and eligible[row].any():
            choices = eligible[row].nonzero(as_tuple=False).flatten()
            choice = choices[torch.randint(len(choices), (1,), generator=generator).item()]
            selected[row, choice] = True
    targets = tokens.clone()
    targets[~selected] = IGNORE_INDEX
    masked = tokens.clone()
    replacement_draw = torch.rand(tokens.shape, generator=generator)
    masked[selected & replacement_draw.lt(0.8)] = 1
    random_positions = selected & replacement_draw.ge(0.8) & replacement_draw.lt(0.9)
    if random_positions.any():
        masked[random_positions] = torch.randint(3, vocab_size, (int(random_positions.sum()),), generator=generator)
    return masked, targets


def make_sort_batch(tokens: torch.Tensor, generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    token_rows: list[torch.Tensor] = []
    position_rows: list[torch.Tensor] = []
    for row in tokens:
        known = row[row.ge(3)]
        permutation = torch.randperm(len(known), generator=generator)
        token_rows.append(known[permutation])
        position_rows.append(torch.arange(len(known), dtype=torch.long)[permutation])
    max_length = max(len(row) for row in token_rows)
    shuffled = torch.zeros((len(token_rows), max_length), dtype=torch.long)
    positions = torch.full((len(token_rows), max_length), -1, dtype=torch.long)
    for index, (row_tokens, row_positions) in enumerate(zip(token_rows, position_rows)):
        shuffled[index, : len(row_tokens)] = row_tokens
        positions[index, : len(row_positions)] = row_positions
    return shuffled, positions


def pairwise_sort_loss(scores: torch.Tensor, positions: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    valid = positions.ge(0)
    earlier = positions[:, :, None] < positions[:, None, :]
    pair_mask = valid[:, :, None] & valid[:, None, :] & earlier & tokens[:, :, None].ne(tokens[:, None, :])
    if not pair_mask.any():
        return scores.sum() * 0.0
    logits = scores[:, None, :] - scores[:, :, None]
    return F.binary_cross_entropy_with_logits(logits[pair_mask], torch.ones_like(logits[pair_mask]))


def sort_metrics(scores: torch.Tensor, positions: torch.Tensor, tokens: torch.Tensor) -> tuple[int, int, int, int]:
    valid = positions.ge(0)
    earlier = positions[:, :, None] < positions[:, None, :]
    comparable = valid[:, :, None] & valid[:, None, :] & earlier & tokens[:, :, None].ne(tokens[:, None, :])
    predicted = scores[:, :, None] < scores[:, None, :]
    pair_correct = int((predicted & comparable).sum().item())
    pair_total = int(comparable.sum().item())
    exact = 0
    sequences = 0
    for row_scores, row_positions, row_tokens in zip(scores, positions, tokens):
        row_valid = row_positions.ge(0)
        predicted_order = row_tokens[row_valid][row_scores[row_valid].argsort()]
        true_order = row_tokens[row_valid][row_positions[row_valid].argsort()]
        exact += int(torch.equal(predicted_order, true_order))
        sequences += 1
    return pair_correct, pair_total, exact, sequences


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


@torch.inference_mode()
def evaluate(
    model: OracleSequenceModel,
    loader: DataLoader,
    device: torch.device,
    mask_probability: float,
) -> dict[str, float]:
    model.eval()
    generator = torch.Generator().manual_seed(424242)
    mlm_loss_sum = 0.0
    mlm_count = 0
    top1 = 0
    top5 = 0
    pair_correct = pair_total = exact = sequences = 0
    for batch in loader:
        tokens_cpu = batch["tokens"]
        masked_cpu, targets_cpu = make_masked_batch(
            tokens_cpu, mask_probability, model.config.vocab_size, generator=generator
        )
        shuffled_cpu, positions_cpu = make_sort_batch(tokens_cpu, generator=generator)
        masked = masked_cpu.to(device)
        targets = targets_cpu.to(device)
        padding_mask = tokens_cpu.eq(0).to(device)
        shuffled = shuffled_cpu.to(device)
        positions = positions_cpu.to(device)
        sort_padding = shuffled.eq(0)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model.forward_mlm(masked, padding_mask)
            sort_scores = model.forward_sort(shuffled, sort_padding)
        selected = targets.ne(IGNORE_INDEX)
        selected_logits = logits[selected]
        selected_targets = targets[selected]
        losses = F.cross_entropy(selected_logits, selected_targets, reduction="sum")
        mlm_loss_sum += float(losses.item())
        mlm_count += int(selected_targets.numel())
        predictions = selected_logits.topk(min(5, selected_logits.shape[-1]), dim=-1).indices
        top1 += int(predictions[:, 0].eq(selected_targets).sum().item())
        top5 += int(predictions.eq(selected_targets[:, None]).any(dim=1).sum().item())
        values = sort_metrics(sort_scores, positions, shuffled)
        pair_correct += values[0]
        pair_total += values[1]
        exact += values[2]
        sequences += values[3]
    return {
        "mlm_loss": mlm_loss_sum / max(1, mlm_count),
        "mlm_top1": top1 / max(1, mlm_count),
        "mlm_top5": top5 / max(1, mlm_count),
        "sort_pair_accuracy": pair_correct / max(1, pair_total),
        "sort_exact_content": exact / max(1, sequences),
        "masked_tokens": float(mlm_count),
        "sequences": float(sequences),
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    seed_everything(int(config["seed"]))
    data_config = config["data"]
    train_config = config["sequence"]
    processed_dir = resolve_project_path(data_config["processed_dir"])
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (processed_dir / "vocab.json").open("r", encoding="utf-8") as handle:
        vocabulary = json.load(handle)
    label_to_token = {key: int(value) for key, value in vocabulary["label_to_token"].items()}
    vocab_size = len(vocabulary["labels"]) + 3
    max_length = int(data_config["max_sequence_length"])

    datasets = {
        split: OracleSequenceDataset(
            processed_dir / "sequences.jsonl", label_to_token, split, max_length=max_length
        )
        for split in ("train", "val", "test")
    }
    batch_size = int(args.batch_size or train_config["batch_size"])
    workers = int(args.num_workers if args.num_workers is not None else train_config["num_workers"])
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=workers,
            pin_memory=device.type == "cuda",
            persistent_workers=workers > 0,
            worker_init_fn=dataloader_worker_init,
            collate_fn=collate_sequences,
            drop_last=split == "train",
        )
        for split, dataset in datasets.items()
    }
    model_config = SequenceConfig(
        vocab_size=vocab_size,
        max_length=max_length,
        embed_dim=int(train_config["embed_dim"]),
        depth=int(train_config["depth"]),
        num_heads=int(train_config["num_heads"]),
        mlp_ratio=float(train_config["mlp_ratio"]),
        dropout=float(train_config["dropout"]),
    )
    model = OracleSequenceModel(model_config).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
        betas=(0.9, 0.98),
    )
    epochs = int(args.epochs or train_config["epochs"])
    total_steps = max(1, epochs * len(loaders["train"]))
    warmup_steps = max(1, int(total_steps * 0.05))

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, schedule)
    mask_probability = float(train_config["mask_probability"])
    best_score = -1.0
    history: list[dict[str, Any]] = []
    start_epoch = 0
    resumed_from: str | None = None
    if args.resume:
        resume_path = resolve_project_path(args.resume)
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        if checkpoint.get("model_config") != model.checkpoint_config():
            raise ValueError("续训检查点的序列模型结构与当前配置不一致")
        if checkpoint.get("vocab", {}).get("labels") != vocabulary["labels"]:
            raise ValueError("续训检查点的字头词表与当前数据不一致")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint.get("best_score", -1.0))
        history = list(checkpoint.get("history", []))
        resumed_from = str(resume_path)
        if not (output_dir / "sequence_best.pt").exists():
            _atomic_torch_save(output_dir / "sequence_best.pt", checkpoint)

        resume_step = start_epoch * len(loaders["train"])
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
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "dataset_sizes": {key: len(value) for key, value in datasets.items()},
                "vocab_size": vocab_size,
                "target_epochs": epochs,
                "resumed_from": resumed_from,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    for epoch in range(start_epoch, epochs):
        model.train()
        started = time.time()
        loss_sum = mlm_sum = sort_sum = 0.0
        batch_count = 0
        for step, batch in enumerate(loaders["train"], start=1):
            tokens_cpu = batch["tokens"]
            masked_cpu, targets_cpu = make_masked_batch(tokens_cpu, mask_probability, vocab_size)
            shuffled_cpu, positions_cpu = make_sort_batch(tokens_cpu)
            masked = masked_cpu.to(device, non_blocking=True)
            targets = targets_cpu.to(device, non_blocking=True)
            padding_mask = tokens_cpu.eq(0).to(device, non_blocking=True)
            shuffled = shuffled_cpu.to(device, non_blocking=True)
            positions = positions_cpu.to(device, non_blocking=True)
            sort_padding = shuffled.eq(0)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model.forward_mlm(masked, padding_mask)
                mlm_loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=IGNORE_INDEX)
                sort_scores = model.forward_sort(shuffled, sort_padding)
                ordering_loss = pairwise_sort_loss(sort_scores, positions, shuffled)
                loss = mlm_loss + float(args.sort_weight) * ordering_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            batch_count += 1
            loss_sum += float(loss.item())
            mlm_sum += float(mlm_loss.item())
            sort_sum += float(ordering_loss.item())
            if step % args.log_every == 0 or step == len(loaders["train"]):
                print(
                    f"epoch={epoch + 1}/{epochs} step={step}/{len(loaders['train'])} "
                    f"loss={loss_sum / batch_count:.4f} mlm={mlm_sum / batch_count:.4f} "
                    f"sort={sort_sum / batch_count:.4f}",
                    flush=True,
                )
        validation = evaluate(model, loaders["val"], device, mask_probability)
        score = validation["mlm_top5"] + validation["sort_pair_accuracy"]
        result = {
            "epoch": epoch + 1,
            "train_loss": loss_sum / max(1, batch_count),
            "train_mlm_loss": mlm_sum / max(1, batch_count),
            "train_sort_loss": sort_sum / max(1, batch_count),
            "val": validation,
            "seconds": time.time() - started,
        }
        history.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_score": max(best_score, score),
            "history": history,
            "model_config": model.checkpoint_config(),
            "vocab": vocabulary,
        }
        _atomic_torch_save(output_dir / "sequence_last.pt", payload)
        if score > best_score:
            best_score = score
            payload["best_score"] = best_score
            _atomic_torch_save(output_dir / "sequence_best.pt", payload)

    best_checkpoint = torch.load(output_dir / "sequence_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model"])
    test_metrics = None if args.skip_final_test else evaluate(model, loaders["test"], device, mask_probability)
    summary = {
        "best_validation_score": best_score,
        "test": test_metrics,
        "history": history,
        "model_config": model.checkpoint_config(),
        "device": str(device),
        "training": {"target_epochs": epochs, "resumed_from": resumed_from},
    }
    atomic_json_dump(output_dir / "sequence_metrics.json", summary)
    if test_metrics is not None:
        print(json.dumps(test_metrics, ensure_ascii=False), flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train oracle sequence ordering and completion")
    parser.add_argument("--config", default="configs/baseline.json")
    parser.add_argument("--output-dir", default="artifacts/checkpoints")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--sort-weight", type=float, default=1.0)
    parser.add_argument("--resume", help="Checkpoint to resume; --epochs is the total target epoch count")
    parser.add_argument(
        "--skip-final-test",
        action="store_true",
        help="select and save the best checkpoint from validation only; defer all test reading to a later frozen evaluation",
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--log-every", type=int, default=25)
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
