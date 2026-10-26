"""Static, label-independent plate partitions for detector confirmation runs."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable


def assign_hash_fold(plate: str, folds: int) -> int:
    """Assign a plate to one stable fold using its UTF-8 identifier only."""
    if folds < 2:
        raise ValueError("at least two folds are required")
    name = str(plate)
    if not name:
        raise ValueError("plate identifier must be non-empty")
    return int(hashlib.sha256(name.encode("utf-8")).hexdigest()[:16], 16) % folds


def make_outer_cv_membership(plates: Iterable[str], folds: int = 5) -> dict[str, int]:
    """Return a complete, deterministic pure-plate outer-fold assignment."""
    names = [str(value) for value in plates]
    if len(names) != len(set(names)):
        raise ValueError("plate identifiers must be unique")
    membership = {name: assign_hash_fold(name, folds) for name in names}
    counts = [sum(value == fold for value in membership.values()) for fold in range(folds)]
    if any(count == 0 for count in counts):
        raise ValueError("every outer fold must receive at least one plate")
    return membership


def nested_fold_roles(membership: dict[str, int], outer_fold: int, folds: int = 5) -> dict[str, list[str]]:
    """Create a label-free 60/20/20 inner-train/inner-val/outer-test split.

    The fold immediately after the outer holdout is fixed as the inner
    validation fold.  The remaining folds form training data.  Thus no label
    coverage adjustment or metric-dependent reassignment occurs.
    """
    if folds < 3:
        raise ValueError("nested roles require at least three folds")
    if outer_fold not in range(folds):
        raise ValueError("outer_fold is outside the declared range")
    invalid = {value for value in membership.values() if value not in range(folds)}
    if invalid:
        raise ValueError("membership contains an invalid fold index")
    inner_val_fold = (outer_fold + 1) % folds
    roles = {"inner_train": [], "inner_val": [], "outer_test": []}
    for plate in sorted(membership):
        fold = membership[plate]
        if fold == outer_fold:
            roles["outer_test"].append(plate)
        elif fold == inner_val_fold:
            roles["inner_val"].append(plate)
        else:
            roles["inner_train"].append(plate)
    if not all(roles.values()):
        raise ValueError("nested role assignment produced an empty role")
    return roles
