"""Immutable provenance helpers for the detector's strict confirmation cohort.

The project's original preparation routine began with a pure SHA-1 plate
allocation, then moved some rare-label records from its nominal validation or
test buckets into training.  A later detector confirmation must not silently
reuse either of those historical selection/evaluation buckets.  This module
therefore exposes the *initial* training reservoir separately from the
label-adjusted historical split.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any

from .prepare import _assign_splits, _initial_split


INITIAL_TRAIN_RESERVOIR_POLICY = "legacy_initial_sha1_train_only_before_label_coverage_moves"


def plate_list_sha256(plates: list[str]) -> str:
    """Hash an ordered plate list using the project-wide manifest convention."""
    return hashlib.sha256("\n".join(plates).encode("utf-8")).hexdigest()


def _names(records: list[dict[str, Any]]) -> list[str]:
    names = [str(record["RubbingName"]) for record in records]
    if not names or len(names) != len(set(names)):
        raise ValueError("raw detector records must contain non-empty unique RubbingName values")
    return names


def derive_legacy_provenance(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Derive auditable legacy memberships without writing data or training a model.

    ``initial_train`` deliberately excludes every record whose original SHA-1
    bucket was validation/test, including records later moved to train solely
    to provide label coverage.  It is therefore disjoint from both historical
    metric-bearing partitions by construction.
    """
    names = _names(records)
    initial_assignments = [_initial_split(name) for name in names]
    final_assignments, moved_to_train = _assign_splits(records)
    initial_by_split = {
        split: sorted(name for name, assignment in zip(names, initial_assignments) if assignment == split)
        for split in ("train", "val", "test")
    }
    final_by_split = {
        split: sorted(name for name, assignment in zip(names, final_assignments) if assignment == split)
        for split in ("train", "val", "test")
    }
    moved = sorted(str(value) for value in moved_to_train)
    initial_train = initial_by_split["train"]
    if set(initial_train) & (set(initial_by_split["val"]) | set(initial_by_split["test"])):
        raise AssertionError("initial split buckets are not disjoint")
    if set(initial_train) & set(moved):
        raise AssertionError("label-coverage moves must originate outside the initial train reservoir")
    if set(moved) != set(final_by_split["train"]) - set(initial_train):
        raise AssertionError("legacy train expansion is inconsistent with moved-record provenance")
    return {
        "policy": INITIAL_TRAIN_RESERVOIR_POLICY,
        "initial_algorithm": "src.oracle_vit.prepare._initial_split: SHA1(RubbingName) first 8 hex digits mod 100; train when bucket < 80",
        "final_algorithm": "src.oracle_vit.prepare._assign_splits: initial allocation plus rare-label coverage moves into train",
        "initial_counts": dict(Counter(initial_assignments)),
        "final_counts": dict(Counter(final_assignments)),
        "initial_members": {
            split: {"plates": len(values), "plates_sha256": plate_list_sha256(values)}
            for split, values in initial_by_split.items()
        },
        "final_members": {
            split: {"plates": len(values), "plates_sha256": plate_list_sha256(values)}
            for split, values in final_by_split.items()
        },
        "label_coverage_moves": {"plates": len(moved), "plates_sha256": plate_list_sha256(moved), "members": moved},
        "eligible_initial_train": {"plates": len(initial_train), "plates_sha256": plate_list_sha256(initial_train), "members": initial_train},
        "excluded": {
            "initial_val": {"plates": len(initial_by_split["val"]), "plates_sha256": plate_list_sha256(initial_by_split["val"]), "members": initial_by_split["val"]},
            "initial_test": {"plates": len(initial_by_split["test"]), "plates_sha256": plate_list_sha256(initial_by_split["test"]), "members": initial_by_split["test"]},
            "label_coverage_moves_from_initial_val_or_test": {"plates": len(moved), "plates_sha256": plate_list_sha256(moved), "members": moved},
        },
    }


def initial_train_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return the raw-order record subset and its full provenance ledger."""
    provenance = derive_legacy_provenance(records)
    allowed = set(provenance["eligible_initial_train"]["members"])
    subset = [record for record in records if str(record["RubbingName"]) in allowed]
    if [str(record["RubbingName"]) for record in subset] != [
        str(record["RubbingName"])
        for record in records
        if str(record["RubbingName"]) in allowed
    ]:
        raise AssertionError("initial-train subset ordering is unstable")
    if {str(record["RubbingName"]) for record in subset} != allowed:
        raise AssertionError("initial-train subset membership is incomplete")
    return subset, provenance
