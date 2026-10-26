"""Learned pairwise link model for sentence-group parsing.

Protocol: ``docs/LEARNED_GROUPING_READING_ORDER_PREREGISTRATION_20260919.md``.

This module is the single source of truth for the learned replacement of the
frozen geometric grouping rule.  It provides

* :func:`pairwise_features` -- features 1-12 of the pre-registration section 3,
  all scale-free, computed for every box pair of one plate given the plate's box
  list, the source domain and the plate's median box diagonal;
* :class:`LinkModel` -- feature standardisation plus a torch link model
  (logistic regression, MLP(64) or MLP(64->64)), with deterministic seeding,
  save/load and probability prediction;
* :func:`cluster_by_links` -- the deterministic union-find single-link
  clustering used by the geometric baseline, with the edge criterion replaced by
  ``link probability >= tau``;
* :func:`cluster_boxes_learned` -- the callable installed in place of the frozen
  ``cluster_boxes``; it ignores the ``radius`` argument that the frozen driver
  computes from the plate diagonal because the learned rule is scale-free;
* :func:`cluster_boxes_geometry` -- geometry mode, which calls the frozen
  ``scripts/evaluate_auto_grouping_read_order.cluster_boxes`` unchanged so that
  the wrapper's plumbing can be shown to reproduce the frozen path bit for bit.

Feature order (15 dimensions, feature *groups* 1-12 of the pre-registration):

==== ==========================================================
idx  feature
==== ==========================================================
0    1. normalised centre distance ``||ci - cj|| / median_diag``
1    2a. ``|dxi| / median_diag``
2    2b. ``|dyi| / median_diag``
3    3. ``log(width / height)``, averaged over the two boxes
4    4. ``min(area) / max(area)``
5    5. IoU
6    6. ``|dy| / mean height``
7    7. ``|dx| / mean width``
8    8a. mean of the two box scores
9    8b. minimum of the two box scores
10   9. number of boxes on the plate
11   10a. x centre of the left-most box, normalised by median diagonal
12   10b. y centre of the left-most box, normalised by median diagonal
13   11. source indicator (rubbing 0, facsimile 1)
14   12. mean normalised nearest-neighbour centre distance (density)
==== ==========================================================

Feature *groups* are numbered 1-12 exactly as in the pre-registration; groups 2,
8 and 10 each contribute more than one raw number, so the vector has 15
dimensions.  The pre-registered "geometry-only" control of section 5 uses
feature groups 1, 2 and 5 (indices ``0, 1, 2, 5``).
"""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import nn

SOURCE_CODES = {"rubbing": 0, "facsimile": 1}
MEDIAN_DIAGONAL_EPS = 1e-9
SINGLE_BOX_DENSITY = 2.0
MODEL_KINDS = ("logistic", "mlp64", "mlp64_64")
MODEL_SIMPLICITY = {"logistic": 0, "mlp64": 1, "mlp64_64": 2}
MODEL_HIDDEN = {"logistic": (), "mlp64": (64,), "mlp64_64": (64, 64)}


def source_code(source: str) -> int:
    """Return the numeric source indicator used by feature group 11."""
    if source not in SOURCE_CODES:
        raise ValueError(f"unknown source: {source!r}")
    return SOURCE_CODES[source]


def feature_indices(feature_mode: str) -> tuple[int, ...]:
    """Return the feature-vector indices used by one feature mode."""
    if feature_mode == "full":
        return tuple(range(FULL_FEATURE_DIM))
    if feature_mode == "geometry":
        return GEOMETRY_FEATURE_INDICES
    raise ValueError(f"unknown feature mode: {feature_mode!r}")


def feature_dim(feature_mode: str) -> int:
    return len(feature_indices(feature_mode))


def _pair_indices(count: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic upper-triangle pair order: (0,1), (0,2), ..., (n-2,n-1)."""
    if count < 2:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)
    left, right = np.triu_indices(count, k=1)
    return left.astype(np.int64), right.astype(np.int64)


def _box_matrix(boxes: Sequence[Sequence[float]]) -> np.ndarray:
    matrix = np.asarray(boxes, dtype=np.float64)
    if matrix.size == 0:
        return np.empty((0, 4), dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != 4:
        raise ValueError("boxes must be a sequence of four xyxy coordinates")
    return matrix


def pairwise_features(
    boxes: Sequence[Sequence[float]],
    source: str,
    median_diagonal: float,
    scores: Sequence[float] | None = None,
) -> np.ndarray:
    """Features 1-12 of the pre-registration for every pair of one plate.

    ``boxes`` are xyxy in pixels, ``median_diagonal`` is the median box diagonal
    of the same plate (computed by the caller exactly as the frozen geometric
    rule computes it), and ``source`` is ``"rubbing"`` or ``"facsimile"``.  The
    returned array has shape ``[n * (n - 1) / 2, 15]`` in the deterministic
    upper-triangle order used by the frozen ``cluster_boxes`` distance loop.

    Every dimension is a ratio of lengths, an angle-free alignment ratio, a
    bounded overlap/score/area ratio, a count, or a binary indicator, so scaling
    the boxes *and* ``median_diagonal`` by a common factor leaves the vector
    unchanged (contract test).
    """
    matrix = _box_matrix(boxes)
    count = int(matrix.shape[0])
    left, right = _pair_indices(count)
    diagonal = float(median_diagonal)
    if count and diagonal <= MEDIAN_DIAGONAL_EPS:
        raise ValueError("median_diagonal must be positive for a non-empty plate")
    if scores is None:
        score_values = np.zeros((count,), dtype=np.float64)
    else:
        score_values = np.asarray(scores, dtype=np.float64).reshape(-1)
        if score_values.shape[0] != count:
            raise ValueError("scores must align with boxes")
    features = np.zeros((left.shape[0], FULL_FEATURE_DIM), dtype=np.float64)
    if left.shape[0] == 0:
        return features

    width = matrix[:, 2] - matrix[:, 0]
    height = matrix[:, 3] - matrix[:, 1]
    area = np.maximum(width, 0.0) * np.maximum(height, 0.0)
    centre_x = (matrix[:, 0] + matrix[:, 2]) / 2.0
    centre_y = (matrix[:, 1] + matrix[:, 3]) / 2.0

    delta_x = np.abs(centre_x[left] - centre_x[right])
    delta_y = np.abs(centre_y[left] - centre_y[right])
    features[:, 0] = np.hypot(delta_x, delta_y) / diagonal
    features[:, 1] = delta_x / diagonal
    features[:, 2] = delta_y / diagonal

    safe_height = np.maximum(height, MEDIAN_DIAGONAL_EPS)
    safe_width = np.maximum(width, MEDIAN_DIAGONAL_EPS)
    # Feature group 3: shape of the two boxes.  The pre-registration names one
    # aspect-ratio log per pair; a pair has two boxes, so the feature is the mean
    # of their per-box ``log(width / height)`` values.
    log_aspect = np.log(safe_width / safe_height)
    features[:, 3] = (log_aspect[left] + log_aspect[right]) / 2.0

    area_left, area_right = area[left], area[right]
    features[:, 4] = np.minimum(area_left, area_right) / np.maximum(area_left, area_right)

    intersection_w = np.maximum(np.minimum(matrix[left, 2], matrix[right, 2]) - np.maximum(matrix[left, 0], matrix[right, 0]), 0.0)
    intersection_h = np.maximum(np.minimum(matrix[left, 3], matrix[right, 3]) - np.maximum(matrix[left, 1], matrix[right, 1]), 0.0)
    intersection = intersection_w * intersection_h
    features[:, 5] = intersection / np.maximum(area_left + area_right - intersection, MEDIAN_DIAGONAL_EPS)

    features[:, 6] = delta_y / np.maximum((height[left] + height[right]) / 2.0, MEDIAN_DIAGONAL_EPS)
    features[:, 7] = delta_x / np.maximum((width[left] + width[right]) / 2.0, MEDIAN_DIAGONAL_EPS)

    score_left, score_right = score_values[left], score_values[right]
    features[:, 8] = (score_left + score_right) / 2.0
    features[:, 9] = np.minimum(score_left, score_right)

    features[:, 10] = float(count)

    # Feature group 10: normalised centre coordinates of both boxes.  The pair
    # is ordered left-to-right so that the two raw coordinates are a
    # deterministic function of the unordered pair.
    first = np.where(centre_x[left] <= centre_x[right], left, right)
    features[:, 11] = centre_x[first] / diagonal
    features[:, 12] = centre_y[first] / diagonal

    features[:, 13] = float(source_code(source))

    # Feature group 12: density, the normalised distance to the nearest other box
    # of the plate, averaged over the two boxes of the pair.  A plate with a
    # single box is impossible here (no pairs); SINGLE_BOX_DENSITY is only the
    # finite fallback for a hypothetical isolated box and is documented so the
    # value is never guessed.
    distances = np.hypot(centre_x[:, None] - centre_x[None, :], centre_y[:, None] - centre_y[None, :])
    np.fill_diagonal(distances, np.inf)
    nearest = distances.min(axis=1) / diagonal
    nearest = np.where(np.isfinite(nearest), nearest, SINGLE_BOX_DENSITY)
    features[:, 14] = (nearest[left] + nearest[right]) / 2.0
    return features


FEATURE_NAMES = (
    "normalised_centre_distance",
    "normalised_delta_x",
    "normalised_delta_y",
    "log_aspect_ratio",
    "area_ratio",
    "iou",
    "row_alignment",
    "column_alignment",
    "mean_score",
    "min_score",
    "plate_box_count",
    "normalised_centre_x",
    "normalised_centre_y",
    "source_indicator",
    "mean_nearest_neighbour_distance",
)
FULL_FEATURE_DIM = len(FEATURE_NAMES)
GEOMETRY_FEATURE_INDICES = (0, 1, 2, 5)


def plate_median_diagonal(boxes: Sequence[Sequence[float]]) -> float:
    """Median xyxy box diagonal, identical to the frozen geometric rule."""
    matrix = _box_matrix(boxes)
    if matrix.shape[0] == 0:
        return 0.0
    diagonals = np.hypot(matrix[:, 2] - matrix[:, 0], matrix[:, 3] - matrix[:, 1])
    return float(np.median(diagonals))


def select_features(matrix: np.ndarray, feature_mode: str) -> np.ndarray:
    """Restrict a full feature matrix to one mode's columns."""
    indices = feature_indices(feature_mode)
    if matrix.shape[1] == len(indices):
        return np.asarray(matrix, dtype=np.float64)
    return np.asarray(matrix, dtype=np.float64)[:, list(indices)]


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cpu")
    return torch.device(device)


class LinkModel:
    """Standardised torch link model with deterministic serialisation.

    ``hidden`` is empty for logistic regression and ``(64,)`` / ``(64, 64)`` for
    the two pre-registered MLPs.  ``mean``/``scale`` are the training-set feature
    standardisation statistics; they travel with the weights so that prediction
    is exactly reproducible after a save/load round trip.
    """

    def __init__(
        self,
        *,
        feature_mode: str = "full",
        kind: str = "logistic",
        mean: Sequence[float] | np.ndarray | None = None,
        scale: Sequence[float] | np.ndarray | None = None,
        seed: int = 20260919,
    ) -> None:
        if feature_mode not in ("full", "geometry"):
            raise ValueError(f"unknown feature mode: {feature_mode!r}")
        if kind not in MODEL_KINDS:
            raise ValueError(f"unknown model kind: {kind!r}")
        self.feature_mode = feature_mode
        self.kind = kind
        self.seed = int(seed)
        dimension = feature_dim(feature_mode)
        self.mean = np.zeros((dimension,), dtype=np.float64) if mean is None else np.asarray(mean, dtype=np.float64)
        self.scale = np.ones((dimension,), dtype=np.float64) if scale is None else np.asarray(scale, dtype=np.float64)
        if self.mean.shape != (dimension,) or self.scale.shape != (dimension,):
            raise ValueError("standardisation statistics do not match the feature dimension")
        torch.manual_seed(self.seed)
        layers: list[nn.Module] = []
        previous = dimension
        for width in MODEL_HIDDEN[kind]:
            layers.append(nn.Linear(previous, int(width)))
            layers.append(nn.ReLU())
            previous = int(width)
        layers.append(nn.Linear(previous, 1))
        self.module = nn.Sequential(*layers)

    # ------------------------------------------------------------------ math
    def standardise(self, features: np.ndarray) -> torch.Tensor:
        """Standardise a feature matrix in either the full or the mode layout.

        A full 15-column matrix is restricted to the model's feature mode first,
        so callers can pass the same matrix to every candidate.
        """
        matrix = select_features(np.asarray(features, dtype=np.float64), self.feature_mode)
        if matrix.size == 0:
            return torch.zeros((0, self.mean.shape[0]), dtype=torch.float32)
        if matrix.ndim != 2 or matrix.shape[1] != self.mean.shape[0]:
            raise ValueError(f"expected features with {self.mean.shape[0]} columns, got {matrix.shape}")
        scaled = (matrix - self.mean[None, :]) / self.scale[None, :]
        return torch.from_numpy(scaled.astype(np.float32, copy=False))

    def fit_standardiser(self, features: np.ndarray) -> None:
        matrix = select_features(np.asarray(features, dtype=np.float64), self.feature_mode)
        if matrix.shape[0] == 0:
            raise ValueError("cannot standardise an empty feature matrix")
        self.mean = matrix.mean(axis=0)
        scale = matrix.std(axis=0)
        self.scale = np.where(scale > 0, scale, 1.0)

    def logits(self, features: np.ndarray) -> torch.Tensor:
        inputs = self.standardise(features)
        if inputs.shape[0] == 0:
            return torch.zeros((0,), dtype=torch.float32)
        self.module.eval()
        with torch.inference_mode():
            return self.module(inputs).reshape(-1)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Link probability for every row of a *full* or mode-restricted matrix."""
        return torch.sigmoid(self.logits(features)).numpy().astype(np.float64)

    def predict_pair_proba(
        self,
        boxes: Sequence[Sequence[float]],
        source: str,
        median_diagonal: float,
        scores: Sequence[float] | None = None,
    ) -> np.ndarray:
        return self.predict_proba(pairwise_features(boxes, source, median_diagonal, scores))

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        return self.module.parameters()

    # ------------------------------------------------------------------- io
    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "feature_mode": self.feature_mode,
            "kind": self.kind,
            "seed": self.seed,
            "feature_names": list(FEATURE_NAMES),
            "used_feature_indices": list(feature_indices(self.feature_mode)),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "module_state": copy.deepcopy(self.module.state_dict()),
        }

    @classmethod
    def from_state_dict(cls, payload: dict[str, Any]) -> "LinkModel":
        if int(payload.get("schema_version", 0)) != 1:
            raise ValueError("unsupported link-model schema version")
        model = cls(
            feature_mode=str(payload["feature_mode"]),
            kind=str(payload["kind"]),
            mean=payload["mean"],
            scale=payload["scale"],
            seed=int(payload.get("seed", 20260919)),
        )
        model.module.load_state_dict(payload["module_state"])
        model.module.eval()
        return model

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), destination)
        return destination

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device | None = None) -> "LinkModel":
        payload = torch.load(Path(path), map_location=_resolve_device(device), weights_only=False)
        model = cls.from_state_dict(payload)
        return model


# --------------------------------------------------------------------- clustering
def cluster_by_links(accept_matrix: np.ndarray, count: int) -> list[list[int]]:
    """Single-link union-find over an explicit boolean edge matrix.

    The traversal order, union direction and component ordering are identical to
    the frozen ``cluster_boxes`` so that only the edge criterion changes.
    """
    if count == 0:
        return []
    matrix = np.asarray(accept_matrix, dtype=bool)
    if matrix.shape != (count, count):
        raise ValueError("accept_matrix must be a square matrix matching count")
    parent = list(range(count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for i in range(count):
        for j in range(i + 1, count):
            if not bool(matrix[i, j]):
                continue
            root_i, root_j = find(i), find(j)
            if root_i != root_j:
                parent[max(root_i, root_j)] = min(root_i, root_j)
    grouped: dict[int, list[int]] = defaultdict(list)
    for index in range(count):
        grouped[find(index)].append(index)
    return [grouped[key] for key in sorted(grouped)]


def link_accept_matrix(probabilities: np.ndarray, count: int, tau: float) -> np.ndarray:
    """Boolean upper-triangular edge matrix for ``probability >= tau``."""
    left, right = _pair_indices(count)
    matrix = np.zeros((count, count), dtype=bool)
    if left.shape[0] == 0:
        return matrix
    values = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if values.shape[0] != left.shape[0]:
        raise ValueError("expected one probability per unordered box pair")
    accepted = values >= float(tau)
    matrix[left[accepted], right[accepted]] = True
    return matrix


def cluster_boxes_by_links(
    boxes: Sequence[Sequence[float]],
    probabilities: np.ndarray,
    tau: float,
) -> list[list[int]]:
    """Library entry point: cluster one plate from precomputed link probabilities."""
    return cluster_by_links(link_accept_matrix(probabilities, len(boxes), tau), len(boxes))


# --------------------------------------------------------------- frozen bridge
_CLUSTER_SOURCE: ContextVar[str] = ContextVar("learned_grouping_source", default="rubbing")

# Installed by the evaluation wrapper: the learned model and the frozen tau.
LEARNED_CLUSTER_STATE: dict[str, Any] = {"model": None, "tau": None, "rule": "geometry", "calls": 0}

# The untouched frozen ``cluster_boxes`` bound by the wrapper before any patch is
# installed.  Geometry mode calls this instead of re-importing the name, so a
# patched frozen module cannot make it recurse into itself.
FROZEN_CLUSTER_BOXES: Any = None


def bind_frozen_cluster_boxes(function: Any) -> None:
    """Remember the frozen ``cluster_boxes`` implementation before patching it."""
    if not callable(function):
        raise TypeError("the frozen cluster_boxes binding must be callable")
    global FROZEN_CLUSTER_BOXES
    FROZEN_CLUSTER_BOXES = function


def configure_cluster_source(source: str) -> None:
    """Set the source domain used by the frozen driver's next ``cluster_boxes`` call."""
    _CLUSTER_SOURCE.set(source)


def current_cluster_source() -> str:
    return _CLUSTER_SOURCE.get()


def configure_learned_clustering(model: LinkModel | None, tau: float | None, rule: str) -> None:
    """Install the learned rule (or geometry mode) for the frozen driver."""
    if rule not in ("geometry", "learned"):
        raise ValueError(f"unknown clustering rule: {rule!r}")
    if rule == "learned":
        if model is None or tau is None:
            raise ValueError("the learned rule needs both a model and a tau")
        LEARNED_CLUSTER_STATE["model"] = model
        LEARNED_CLUSTER_STATE["tau"] = float(tau)
    else:
        LEARNED_CLUSTER_STATE["model"] = None
        LEARNED_CLUSTER_STATE["tau"] = None
    LEARNED_CLUSTER_STATE["rule"] = rule
    LEARNED_CLUSTER_STATE["calls"] = 0


def cluster_boxes_learned(boxes: list[list[float]], radius: float) -> list[list[int]]:
    """Replacement for the frozen ``cluster_boxes`` driven by the learned rule.

    The frozen driver passes ``radius = T x median box diagonal``.  The learned
    rule is scale-free by construction, so ``radius`` is deliberately ignored:
    the accepted edges are exactly those with link probability ``>= tau`` for the
    frozen, validation-selected ``tau``, and the source domain comes from
    :func:`configure_cluster_source`.
    """
    del radius
    model: LinkModel | None = LEARNED_CLUSTER_STATE["model"]
    tau = LEARNED_CLUSTER_STATE["tau"]
    if model is None or tau is None:
        raise RuntimeError("learned clustering was not configured with a model and tau")
    LEARNED_CLUSTER_STATE["calls"] = int(LEARNED_CLUSTER_STATE["calls"]) + 1
    if len(boxes) < 2:
        return cluster_by_links(np.zeros((len(boxes), len(boxes)), dtype=bool), len(boxes))
    diagonal = plate_median_diagonal(boxes)
    probabilities = model.predict_pair_proba(boxes, current_cluster_source(), diagonal)
    return cluster_boxes_by_links(boxes, probabilities, float(tau))


def cluster_boxes_geometry(boxes: list[list[float]], radius: float) -> list[list[int]]:
    """Geometry mode: call the frozen ``cluster_boxes`` unchanged.

    The binding is taken from :func:`bind_frozen_cluster_boxes` when the wrapper
    has installed one; otherwise the frozen module is imported and its current
    ``cluster_boxes`` attribute is used.
    """
    function = FROZEN_CLUSTER_BOXES
    if function is None:
        from evaluate_auto_grouping_read_order import cluster_boxes as function  # noqa: PLC0415

    return function(boxes, radius)


__all__ = [
    "FEATURE_NAMES",
    "FULL_FEATURE_DIM",
    "GEOMETRY_FEATURE_INDICES",
    "MODEL_HIDDEN",
    "MODEL_KINDS",
    "MODEL_SIMPLICITY",
    "LinkModel",
    "bind_frozen_cluster_boxes",
    "cluster_boxes_by_links",
    "cluster_boxes_geometry",
    "cluster_boxes_learned",
    "cluster_by_links",
    "configure_cluster_source",
    "configure_learned_clustering",
    "current_cluster_source",
    "feature_dim",
    "feature_indices",
    "link_accept_matrix",
    "pairwise_features",
    "plate_median_diagonal",
    "select_features",
    "source_code",
]
