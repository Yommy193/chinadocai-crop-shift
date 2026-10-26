"""Feature hallucination for a rubbing-only inference path.

The module may see a matched facsimile feature only during supervised
training.  Its forward path consumes a rubbing feature alone, which keeps the
inference contract distinct from the research-only strict paired-fusion head.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .paired_fusion import PairedFeatureFusionHead


class RubbingPrivilegedHallucinationHead(nn.Module):
    """Hallucinate a facsimile-like feature from a rubbing feature.

    The residual initialization makes the ablation with alignment weight zero
    a well-defined rubbing-only control: the classifier receives two
    deterministic functions of the same rubbing feature, and no facsimile is
    required at validation or test time.
    """

    def __init__(self, feature_dim: int, num_classes: int, dropout: float = 0.15) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.input_norm = nn.LayerNorm(self.feature_dim)
        self.translator = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, self.feature_dim),
        )
        # At initialization, the privileged branch exactly equals the rubbing
        # representation.  This avoids turning an architectural random change
        # into an apparent effect of the facsimile-alignment objective.
        nn.init.zeros_(self.translator[-1].weight)
        nn.init.zeros_(self.translator[-1].bias)
        self.classifier = PairedFeatureFusionHead(self.feature_dim, int(num_classes), dropout)

    def hallucinate(self, rubbing_features: torch.Tensor) -> torch.Tensor:
        return rubbing_features + self.translator(self.input_norm(rubbing_features))

    def forward(self, rubbing_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hallucinated = self.hallucinate(rubbing_features)
        return self.classifier(hallucinated, rubbing_features), hallucinated


def normalized_feature_alignment(
    hallucinated_facsimile: torch.Tensor,
    true_facsimile: torch.Tensor,
) -> torch.Tensor:
    """Return cosine-distance feature supervision with a bounded scale."""

    if hallucinated_facsimile.shape != true_facsimile.shape:
        raise ValueError("hallucinated and true facsimile features must have the same shape")
    if torch.equal(hallucinated_facsimile, true_facsimile):
        return hallucinated_facsimile.sum() * 0.0
    # Floating-point round-off can make an exact match appear as a tiny
    # negative distance.  Keeping the objective non-negative is both
    # mathematically correct and makes the zero-alignment ablation exact.
    cosine = F.cosine_similarity(hallucinated_facsimile.float(), true_facsimile.float(), dim=-1).mean()
    return (1.0 - cosine).clamp_min(0.0)


def softened_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """KL distillation loss with the standard temperature-squared correction."""

    if temperature <= 0:
        raise ValueError("distillation temperature must be positive")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher logits must have the same shape")
    if torch.equal(student_logits, teacher_logits):
        return student_logits.sum() * 0.0
    scaled_student = F.log_softmax(student_logits.float() / temperature, dim=-1)
    scaled_teacher = F.softmax(teacher_logits.float() / temperature, dim=-1)
    # KL is non-negative analytically, but the fused floating-point kernel may
    # return a minute negative value for identical logits.  Clamp that purely
    # numerical artefact so a zero-distillation control remains well-defined.
    return (F.kl_div(scaled_student, scaled_teacher, reduction="batchmean") * (temperature**2)).clamp_min(0.0)
