"""Reusable, explicit post-processing for plate-detector candidate boxes.

The detector produces score-ranked local peaks before non-maximum suppression
(NMS).  Keeping the post-processing separate makes validation-only decoder
ablations reproducible without changing a trained model's weights.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torchvision.ops import nms


def postprocess_detection_candidates(
    candidates: Sequence[dict[str, torch.Tensor]],
    *,
    score_threshold: float,
    top_k: int,
    nms_threshold: float,
) -> list[dict[str, torch.Tensor]]:
    """Filter score-ranked candidates and apply class-agnostic NMS.

    ``top_k`` is deliberately applied before NMS, matching the detector's
    original decoder contract.  The returned tensors remain score-descending,
    which is required by the project-local AP implementation.
    """
    if not 0.0 <= score_threshold <= 1.0:
        raise ValueError("score_threshold must be in [0, 1]")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not 0.0 <= nms_threshold <= 1.0:
        raise ValueError("nms_threshold must be in [0, 1]")
    rows: list[dict[str, torch.Tensor]] = []
    for candidate in candidates:
        boxes, scores = candidate["boxes"], candidate["scores"]
        if boxes.ndim != 2 or boxes.shape[-1] != 4 or scores.ndim != 1 or boxes.shape[0] != scores.shape[0]:
            raise ValueError("candidate boxes/scores must be [N,4] and [N]")
        if not scores.numel():
            rows.append({"boxes": boxes, "scores": scores})
            continue
        ranked = torch.argsort(scores, descending=True, stable=True)
        ranked = ranked[scores[ranked] >= score_threshold][:top_k]
        if not ranked.numel():
            rows.append({"boxes": boxes.new_empty((0, 4)), "scores": scores.new_empty((0,))})
            continue
        selected = nms(boxes[ranked], scores[ranked], nms_threshold)
        selected_indices = ranked[selected]
        rows.append({"boxes": boxes[selected_indices], "scores": scores[selected_indices]})
    return rows
