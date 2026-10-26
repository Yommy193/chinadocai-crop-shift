from __future__ import annotations

import json
import hashlib
import os
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    import torch


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_project_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root() / path


def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    with resolve_project_path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dataloader_worker_init(worker_id: int) -> None:
    import numpy as np
    import torch

    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def read_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str | os.PathLike[str], rows: Iterable[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(destination)


def atomic_json_dump(path: str | os.PathLike[str], payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(destination)


def file_sha256(path: str | os.PathLike[str]) -> str:
    """Return a stable SHA-256 digest without loading an artifact into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def topk_accuracy(logits: "torch.Tensor", target: "torch.Tensor", ks: tuple[int, ...] = (1, 5)) -> list[float]:
    if target.numel() == 0:
        return [0.0 for _ in ks]
    max_k = min(max(ks), logits.shape[-1])
    predictions = logits.topk(max_k, dim=-1).indices
    return [predictions[:, : min(k, max_k)].eq(target[:, None]).any(dim=1).float().mean().item() for k in ks]
