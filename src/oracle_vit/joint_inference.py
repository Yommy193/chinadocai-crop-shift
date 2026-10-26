"""N-best visual candidate reranking with the project's sequence model.

The visual model remains the source of local glyph evidence.  This module only
selects among its top-k candidates, then scores the resulting *sorted* token
sequence with the independently trained masked sequence model.  Keeping the
two scores explicit makes this a reversible inference-time experiment rather
than a hidden change to either trained model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class JointRerankResult:
    """The best visual/sequence hypothesis and its auditable component scores."""

    tokens: list[int]
    order: list[int]
    sort_scores: list[float]
    visual_mean_log_probability: float
    sequence_mean_log_probability: float
    combined_score: float
    beam_rank_before_rerank: int
    candidate_count: int


@dataclass(frozen=True)
class JointCandidateLattice:
    """All n-best hypotheses with scores reusable across fusion weights."""

    hypotheses: torch.Tensor
    orders: torch.Tensor
    sort_scores: torch.Tensor
    visual_mean_scores: torch.Tensor
    sequence_scores: torch.Tensor


@dataclass(frozen=True)
class GapCompletionCandidateLattice:
    """Gap/token completions with local and complete-sequence evidence.

    ``masked_log_probabilities`` is the original missing-glyph score: a token
    is predicted at every possible gap while that gap is masked.  The optional
    pseudo-log-likelihood term is intentionally kept separate, because an MLM
    pseudo-likelihood is a candidate-ranking score rather than a calibrated
    probability distribution over gaps.
    """

    gaps: torch.Tensor
    tokens: torch.Tensor
    masked_log_probabilities: torch.Tensor
    completed_sequence_scores: torch.Tensor


def n_best_visual_candidates(
    candidate_tokens: Sequence[Sequence[int]],
    candidate_log_probabilities: Sequence[Sequence[float]],
    beam_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the highest-scoring independent visual hypotheses.

    The beam is exact for the independent visual score up to ``beam_width``.
    It avoids enumerating all K^L candidate combinations while retaining an
    auditable n-best list for later sequence reranking.
    """
    if beam_width < 1:
        raise ValueError("beam_width must be at least one")
    if not candidate_tokens or len(candidate_tokens) != len(candidate_log_probabilities):
        raise ValueError("candidate token and probability rows must be non-empty and aligned")

    beam: list[tuple[tuple[int, ...], float]] = [((), 0.0)]
    for token_row, score_row in zip(candidate_tokens, candidate_log_probabilities):
        if not token_row or len(token_row) != len(score_row):
            raise ValueError("every glyph needs at least one aligned candidate")
        if any(int(token) < 3 for token in token_row):
            raise ValueError("candidate tokens must be ordinary label tokens (>= 3)")
        expanded = [
            (prefix + (int(token),), score + float(log_probability))
            for prefix, score in beam
            for token, log_probability in zip(token_row, score_row)
        ]
        expanded.sort(key=lambda item: (-item[1], item[0]))
        beam = expanded[:beam_width]

    return (
        torch.tensor([tokens for tokens, _ in beam], dtype=torch.long),
        torch.tensor([score for _, score in beam], dtype=torch.float32),
    )


@torch.inference_mode()
def sorted_token_hypotheses(
    sequence_model: torch.nn.Module,
    token_hypotheses: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sort every candidate hypothesis and preserve original-image indices."""
    tokens = token_hypotheses.to(device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        scores = sequence_model.forward_sort(tokens)
    order = scores.argsort(dim=-1)
    return tokens.gather(1, order), order, scores.float()


@torch.inference_mode()
def masked_pseudo_log_likelihood(
    sequence_model: torch.nn.Module,
    ordered_tokens: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Score each ordered hypothesis by leave-one-out masked-token likelihood.

    The sequence model is bidirectional, so an autoregressive likelihood is
    unavailable.  Pseudo log-likelihood evaluates each content token with that
    token masked and all other sorted candidates visible.  It is length
    normalized, allowing a single weight to be selected on validation data.
    """
    if ordered_tokens.ndim != 2 or ordered_tokens.shape[1] < 1:
        raise ValueError("ordered_tokens must have shape [hypotheses, length >= 1]")
    if ordered_tokens.shape[1] > int(sequence_model.config.max_length):
        raise ValueError("candidate sequence exceeds the sequence model maximum length")

    hypotheses, length = ordered_tokens.shape
    tokens = ordered_tokens.to(device)
    masked = tokens[:, None, :].expand(hypotheses, length, length).clone().reshape(hypotheses * length, length)
    mask_positions = torch.arange(length, device=device).repeat(hypotheses)
    masked[torch.arange(hypotheses * length, device=device), mask_positions] = 1
    targets = tokens[:, None, :].expand(hypotheses, length, length)
    targets = targets[:, torch.arange(length, device=device), torch.arange(length, device=device)].reshape(-1)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits = sequence_model.forward_mlm(masked)
    log_probabilities = logits.float().log_softmax(dim=-1)
    selected = log_probabilities[
        torch.arange(hypotheses * length, device=device), mask_positions, targets
    ]
    return selected.reshape(hypotheses, length).mean(dim=1)


@torch.inference_mode()
def build_gap_completion_candidate_lattice(
    sequence_model: torch.nn.Module,
    visible_tokens: Sequence[int],
    device: torch.device,
    candidate_top_k: int,
) -> GapCompletionCandidateLattice:
    """Build all-gap completion candidates for later, validation-only rescoring.

    The local candidate list is exactly the same one used by the historical
    all-gap decoder: the Top-K masked-token predictions are retained at every
    insertion gap.  For each candidate we also form the completed sequence and
    obtain its length-normalized leave-one-out pseudo-log-likelihood.  Reusing
    this lattice makes weight selection deterministic and avoids changing the
    underlying image or sequence checkpoints.
    """
    if candidate_top_k < 1:
        raise ValueError("candidate_top_k must be at least one")
    if any(int(token) < 3 for token in visible_tokens):
        raise ValueError("visible completion tokens must be ordinary label tokens (>= 3)")
    completed_length = len(visible_tokens) + 1
    if completed_length > int(sequence_model.config.max_length):
        raise ValueError("completion candidate exceeds the sequence model maximum length")

    masked = [
        list(visible_tokens[:gap]) + [1] + list(visible_tokens[gap:])
        for gap in range(completed_length)
    ]
    inputs = torch.tensor(masked, dtype=torch.long, device=device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits = sequence_model.forward_mlm(inputs)
    positions = torch.arange(completed_length, device=device)
    local_scores = logits[positions, positions, 3:].float().log_softmax(dim=-1)
    values, classes = local_scores.topk(min(candidate_top_k, local_scores.shape[-1]), dim=-1)
    width = int(values.shape[1])
    gaps = torch.arange(completed_length, device=device).repeat_interleave(width)
    tokens = (classes.reshape(-1) + 3).long()
    completed = torch.tensor(
        [
            list(visible_tokens[:int(gap)]) + [int(token)] + list(visible_tokens[int(gap):])
            for gap, token in zip(gaps.cpu().tolist(), tokens.cpu().tolist())
        ],
        dtype=torch.long,
        device=device,
    )
    return GapCompletionCandidateLattice(
        gaps=gaps,
        tokens=tokens,
        masked_log_probabilities=values.reshape(-1),
        completed_sequence_scores=masked_pseudo_log_likelihood(sequence_model, completed, device),
    )


def select_gap_completion_candidates(
    lattice: GapCompletionCandidateLattice,
    pseudo_log_likelihood_weight: float,
    top_k: int,
) -> list[tuple[int, int]]:
    """Return globally ranked ``(gap, token)`` completions from a fixed lattice.

    Setting ``pseudo_log_likelihood_weight`` to zero recovers the historical
    masked-token ranking (up to deterministic tie handling).  Positive values
    add complete-sequence plausibility only after a validation-only choice.
    """
    if pseudo_log_likelihood_weight < 0:
        raise ValueError("pseudo_log_likelihood_weight must be non-negative")
    if top_k < 1:
        raise ValueError("top_k must be at least one")
    combined = lattice.masked_log_probabilities + float(pseudo_log_likelihood_weight) * lattice.completed_sequence_scores
    ranked = sorted(
        zip(
            lattice.gaps.detach().cpu().tolist(),
            lattice.tokens.detach().cpu().tolist(),
            combined.detach().cpu().tolist(),
        ),
        key=lambda item: (-float(item[2]), int(item[0]), int(item[1])),
    )
    return [(int(gap), int(token)) for gap, token, _ in ranked[:top_k]]


@torch.inference_mode()
def build_joint_candidate_lattice(
    sequence_model: torch.nn.Module,
    candidate_tokens: Sequence[Sequence[int]],
    candidate_log_probabilities: Sequence[Sequence[float]],
    device: torch.device,
    beam_width: int,
) -> JointCandidateLattice:
    """Construct one n-best lattice; fusion weights may then be tested cheaply."""
    hypotheses, visual_sum_scores = n_best_visual_candidates(
        candidate_tokens, candidate_log_probabilities, beam_width
    )
    ordered, orders, sort_scores = sorted_token_hypotheses(sequence_model, hypotheses, device)
    return JointCandidateLattice(
        hypotheses=hypotheses,
        orders=orders,
        sort_scores=sort_scores,
        visual_mean_scores=visual_sum_scores.to(device) / hypotheses.shape[1],
        sequence_scores=masked_pseudo_log_likelihood(sequence_model, ordered, device),
    )


def select_joint_candidate(lattice: JointCandidateLattice, sequence_weight: float) -> JointRerankResult:
    """Select a fused hypothesis from a precomputed lattice."""
    if sequence_weight < 0:
        raise ValueError("sequence_weight must be non-negative")
    combined_scores = lattice.visual_mean_scores + float(sequence_weight) * lattice.sequence_scores
    best_index = int(combined_scores.argmax().item())
    return JointRerankResult(
        tokens=lattice.hypotheses[best_index].tolist(),
        order=lattice.orders[best_index].cpu().tolist(),
        sort_scores=lattice.sort_scores[best_index].cpu().tolist(),
        visual_mean_log_probability=float(lattice.visual_mean_scores[best_index].item()),
        sequence_mean_log_probability=float(lattice.sequence_scores[best_index].item()),
        combined_score=float(combined_scores[best_index].item()),
        beam_rank_before_rerank=best_index + 1,
        candidate_count=int(lattice.hypotheses.shape[0]),
    )


@torch.inference_mode()
def rerank_visual_candidates(
    sequence_model: torch.nn.Module,
    candidate_tokens: Sequence[Sequence[int]],
    candidate_log_probabilities: Sequence[Sequence[float]],
    device: torch.device,
    beam_width: int,
    sequence_weight: float,
) -> JointRerankResult:
    """Jointly select visual candidates and their sorted order from an n-best beam."""
    lattice = build_joint_candidate_lattice(
        sequence_model, candidate_tokens, candidate_log_probabilities, device, beam_width
    )
    return select_joint_candidate(lattice, sequence_weight)
