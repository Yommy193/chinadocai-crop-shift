from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageOps
from torch import nn
from torch.nn import functional as F

from .cnn_baseline import CnnConfig, build_cnn
from .data import _square_crop, build_image_transform
from .joint_inference import JointRerankResult, rerank_visual_candidates
from .models import OracleSequenceModel, OracleViT, SequenceConfig, VisionConfig
from .utils import atomic_json_dump, resolve_project_path


def _load_vision(path: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if "architecture" in checkpoint:
        model = build_cnn(CnnConfig(**checkpoint["model_config"]))
        image_size = int(checkpoint["training"]["image_size"])
    else:
        model = OracleViT(VisionConfig(**checkpoint["model_config"]))
        image_size = int(checkpoint["model_config"]["image_size"])
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    # A uniform local attribute lets recognition use the same transform for
    # ViT and CNN checkpoints without changing either model's architecture.
    model.oracle_image_size = image_size
    return model, checkpoint["vocab"]


def _load_sequence(path: Path, device: torch.device) -> tuple[OracleSequenceModel, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = OracleSequenceModel(SequenceConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return model, checkpoint["vocab"]


def _standalone_square(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image)
    side = max(image.width, image.height)
    horizontal = side - image.width
    vertical = side - image.height
    return ImageOps.expand(
        image,
        border=(horizontal // 2, vertical // 2, horizontal - horizontal // 2, vertical - vertical // 2),
        fill=(255, 255, 255) if image.mode in {"RGB", "RGBA"} else 255,
    )


def _load_inputs(args: argparse.Namespace) -> tuple[list[Image.Image], list[str]]:
    if args.images:
        images: list[Image.Image] = []
        names: list[str] = []
        for value in args.images:
            path = resolve_project_path(value)
            with Image.open(path) as image:
                image.load()
                images.append(_standalone_square(image.convert("RGB")))
            names.append(str(path))
        return images, names

    plate_path = resolve_project_path(args.plate)
    boxes_path = resolve_project_path(args.boxes)
    with boxes_path.open("r", encoding="utf-8") as handle:
        box_payload = json.load(handle)
    boxes = box_payload["boxes"] if isinstance(box_payload, dict) else box_payload
    if not all(isinstance(box, list) and len(box) == 4 for box in boxes):
        raise ValueError("boxes JSON must be a list of [x, y, width, height]")
    with Image.open(plate_path) as plate:
        plate.load()
        plate = plate.convert("RGB")
        crops = [_square_crop(plate, [int(value) for value in box]) for box in boxes]
    return crops, [f"{plate_path}#box={index}" for index in range(len(boxes))]


@torch.inference_mode()
def _vision_logits(model: nn.Module, tensor: torch.Tensor, device: torch.device, robust_tta: bool) -> torch.Tensor:
    """Return standard logits or a glyph-safe four-view logit ensemble.

    The views only adjust contrast and apply symmetric blur.  They never flip,
    rotate, crop, or otherwise alter a glyph's potentially meaningful geometry.
    """
    views = [tensor]
    if robust_tta:
        views.extend(
            [
                (tensor * 1.12).clamp(-1.0, 1.0),
                (tensor * 0.88).clamp(-1.0, 1.0),
                F.avg_pool2d(tensor, kernel_size=3, stride=1, padding=1),
            ]
        )
    joined = torch.cat(views, dim=0)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits = model(joined)
    return logits.float().reshape(len(views), tensor.shape[0], -1).mean(dim=0)


@torch.inference_mode()
def recognize(
    model: nn.Module,
    vocabulary: dict[str, Any],
    images: list[Image.Image],
    device: torch.device,
    top_k: int,
    robust_tta: bool = False,
) -> tuple[list[int], list[dict[str, Any]]]:
    image_size = int(model.oracle_image_size) if hasattr(model, "oracle_image_size") else int(model.config.image_size)
    transform = build_image_transform(image_size, train=False)
    tensor = torch.stack([transform(image) for image in images]).to(device)
    probabilities = _vision_logits(model, tensor, device, robust_tta).softmax(dim=-1)
    values, indices = probabilities.topk(min(top_k, probabilities.shape[-1]), dim=-1)
    labels = vocabulary["labels"]
    top1_tokens: list[int] = []
    results: list[dict[str, Any]] = []
    label_to_token = vocabulary["label_to_token"]
    for row_values, row_indices in zip(values.cpu().tolist(), indices.cpu().tolist()):
        candidates = [
            {"label": labels[class_id], "class_id": class_id, "probability": probability}
            for probability, class_id in zip(row_values, row_indices)
        ]
        results.append({"candidates": candidates})
        top1_tokens.append(int(label_to_token[candidates[0]["label"]]))
    return top1_tokens, results


@torch.inference_mode()
def rerank_recognition_candidates(
    sequence: OracleSequenceModel,
    vocabulary: dict[str, Any],
    recognition: list[dict[str, Any]],
    device: torch.device,
    candidate_top_k: int = 3,
    beam_width: int = 8,
    sequence_weight: float = 0.6,
) -> JointRerankResult:
    """Choose among visual candidates with an explicit sorted-sequence prior.

    This intentionally cannot introduce a label that the visual network did
    not already propose.  Its frozen research settings are evaluated in
    ``configs/joint_candidate_rerank_20260830.json``; callers keep it opt-in
    until a modality-robust completion result is available.
    """
    if not 1 <= candidate_top_k:
        raise ValueError("candidate_top_k must be positive")
    label_to_token = vocabulary["label_to_token"]
    token_rows: list[list[int]] = []
    log_probability_rows: list[list[float]] = []
    for row in recognition:
        candidates = row.get("candidates", [])[:candidate_top_k]
        if not candidates:
            raise ValueError("every input image needs at least one visual candidate")
        token_rows.append([int(label_to_token[candidate["label"]]) for candidate in candidates])
        log_probability_rows.append([float(torch.log(torch.tensor(candidate["probability"])).item()) for candidate in candidates])
    return rerank_visual_candidates(
        sequence,
        token_rows,
        log_probability_rows,
        device,
        beam_width=beam_width,
        sequence_weight=sequence_weight,
    )


@torch.inference_mode()
def order_tokens(model: OracleSequenceModel, tokens: list[int], device: torch.device) -> tuple[list[int], list[float]]:
    if len(tokens) > model.config.max_length:
        raise ValueError(f"At most {model.config.max_length} glyphs can be sorted at once")
    tensor = torch.tensor(tokens, dtype=torch.long, device=device)[None, :]
    scores = model.forward_sort(tensor)[0]
    order = scores.argsort().cpu().tolist()
    return order, scores.float().cpu().tolist()


@torch.inference_mode()
def suggest_missing(
    model: OracleSequenceModel,
    ordered_tokens: list[int],
    token_to_label: dict[int, str],
    device: torch.device,
    top_k: int,
    fixed_position: int | None,
) -> list[dict[str, Any]]:
    positions = [fixed_position] if fixed_position is not None else list(range(len(ordered_tokens) + 1))
    if any(position is None or position < 0 or position > len(ordered_tokens) for position in positions):
        raise ValueError("missing-position must be between 0 and the number of input glyphs")
    if len(ordered_tokens) + 1 > model.config.max_length:
        raise ValueError("Sequence with the missing slot exceeds the model maximum length")
    candidates = []
    for position in positions:
        sequence = ordered_tokens[:position] + [1] + ordered_tokens[position:]
        tensor = torch.tensor(sequence, dtype=torch.long, device=device)[None, :]
        logits = model.forward_mlm(tensor)[0, position]
        label_logits = logits[3:]
        log_probabilities = label_logits.log_softmax(dim=-1)
        values, relative_ids = log_probabilities.topk(min(top_k, label_logits.numel()))
        for value, relative_id in zip(values.float().cpu().tolist(), relative_ids.cpu().tolist()):
            token_id = relative_id + 3
            candidates.append(
                {
                    "position": position,
                    "label": token_to_label[token_id],
                    "token_id": token_id,
                    "conditional_probability": float(torch.exp(torch.tensor(value)).item()),
                    "log_probability": value,
                }
            )
    candidates.sort(key=lambda row: row["log_probability"], reverse=True)
    return candidates[:top_k]


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    vision, vision_vocab = _load_vision(resolve_project_path(args.vision_checkpoint), device)
    sequence, sequence_vocab = _load_sequence(resolve_project_path(args.sequence_checkpoint), device)
    if vision_vocab["labels"] != sequence_vocab["labels"]:
        raise ValueError("Vision and sequence checkpoints use different vocabularies")
    images, names = _load_inputs(args)
    if not images:
        raise ValueError("No input images were supplied")
    requested_top_k = max(int(args.top_k), int(args.joint_candidate_top_k) if args.joint_candidate_rerank else 1)
    tokens, recognition = recognize(vision, vision_vocab, images, device, requested_top_k, robust_tta=args.robust_tta)
    for name, row in zip(names, recognition):
        row["input"] = name

    joint_result: JointRerankResult | None = None
    if args.joint_candidate_rerank and args.no_sort:
        raise ValueError("联合候选重排需要启用自动排序")
    if args.joint_candidate_rerank:
        joint_result = rerank_recognition_candidates(
            sequence,
            vision_vocab,
            recognition,
            device,
            candidate_top_k=args.joint_candidate_top_k,
            beam_width=args.joint_beam_width,
            sequence_weight=args.joint_sequence_weight,
        )
        tokens, order, sort_scores = joint_result.tokens, joint_result.order, joint_result.sort_scores
    elif args.no_sort:
        order = list(range(len(tokens)))
        sort_scores = [float(index) for index in order]
    else:
        order, sort_scores = order_tokens(sequence, tokens, device)
    token_to_label = {int(token): label for label, token in vision_vocab["label_to_token"].items()}
    reference_root = resolve_project_path(args.reference_dir)

    def reference_for(label: str) -> str | None:
        directory = reference_root / label
        if not directory.exists():
            return None
        match = next(iter(sorted(directory.glob("*.png"))), None)
        return str(match.resolve()) if match else None

    ordered_tokens = [tokens[index] for index in order]
    ordered = [
        {
            "rank": rank,
            "original_index": index,
            "input": names[index],
            "label": token_to_label[tokens[index]],
            "reference_image": reference_for(token_to_label[tokens[index]]),
            "sort_score": sort_scores[index],
        }
        for rank, index in enumerate(order)
    ]
    if joint_result:
        for token, row in zip(tokens, recognition):
            selected_label = token_to_label[token]
            row["joint_selected_label"] = selected_label
            for candidate in row["candidates"]:
                candidate["selected_by_joint"] = candidate["label"] == selected_label
    completion = []
    if args.suggest_missing:
        completion = suggest_missing(
            sequence,
            ordered_tokens,
            token_to_label,
            device,
            args.top_k,
            args.missing_position,
        )
        for suggestion in completion:
            suggestion["reference_image"] = reference_for(suggestion["label"])
    output = {
        "device": str(device),
        "robust_tta": bool(args.robust_tta),
        "recognition": recognition,
        "ordered": ordered,
        "missing_suggestions": completion,
        "joint_candidate_rerank": {
            "applied": joint_result is not None,
            "candidate_top_k": int(args.joint_candidate_top_k),
            "beam_width": int(args.joint_beam_width),
            "sequence_weight": float(args.joint_sequence_weight),
            "beam_rank_before_rerank": joint_result.beam_rank_before_rerank if joint_result else None,
            "candidate_count": joint_result.candidate_count if joint_result else None,
            "notes": "实验功能：两域排序有正的片级 bootstrap 区间；拓片缺字 Top-5 有提升，摹本缺字的差异尚不显著。",
        },
        "notes": [
            "label is the dataset's canonical oracle-glyph identifier, not a modern Chinese Unicode character",
            "missing-position is zero-based in the predicted ordered sequence",
            "when missing-position is omitted, gap ranking compares MLM confidence and is a heuristic, not a calibrated probability that a gap exists",
        ],
    }
    if args.output:
        atomic_json_dump(resolve_project_path(args.output), output)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recognize, order, and complete oracle glyph images")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--images", nargs="+", help="Pre-cropped single-glyph image paths")
    inputs.add_argument("--plate", help="Full plate image; requires --boxes")
    parser.add_argument("--boxes", help="JSON boxes for --plate, each [x, y, width, height]")
    parser.add_argument("--vision-checkpoint", default="artifacts/checkpoints/vision_best.pt")
    parser.add_argument("--sequence-checkpoint", default="artifacts/checkpoints/sequence_best.pt")
    parser.add_argument("--reference-dir", default="data/processed/glyph_reference_full")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--robust-tta", action="store_true", help="average four glyph-safe contrast/blur views")
    parser.add_argument("--joint-candidate-rerank", action="store_true", help="opt-in Top-3 visual candidate reranking with the sequence model")
    parser.add_argument("--joint-candidate-top-k", type=int, default=3)
    parser.add_argument("--joint-beam-width", type=int, default=8)
    parser.add_argument("--joint-sequence-weight", type=float, default=0.6)
    parser.add_argument("--no-sort", action="store_true", help="Keep the supplied image order")
    parser.add_argument("--suggest-missing", action="store_true")
    parser.add_argument("--missing-position", type=int, help="Known zero-based missing position; otherwise test every gap")
    parser.add_argument("--output")
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.plate and not args.boxes:
        raise SystemExit("--plate requires --boxes")
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
