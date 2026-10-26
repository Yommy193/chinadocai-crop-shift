"""Validation helpers for the frozen detector nested outer-CV manifest.

The helper deliberately receives explicit member lists from a previously
frozen manifest.  It never calls the historical, label-adjusted split logic,
so a confirmation run cannot silently move a plate based on character labels.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .plate_detection import SOURCE_FIELDS


ROLE_NAMES = ("inner_train", "inner_val", "outer_test")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def plate_list_sha256(plates: list[str]) -> str:
    """Match the ordered member-list hash used by the manifest freezer."""
    return hashlib.sha256("\n".join(plates).encode("utf-8")).hexdigest()


def image_name_size_inventory(directory: Path) -> dict[str, Any]:
    """Recompute the immutable name/byte-size inventory stored at freeze time."""
    if not directory.is_dir():
        raise NotADirectoryError(f"raw image directory is missing: {directory}")
    rows: list[str] = []
    for path in sorted(directory.iterdir(), key=lambda value: value.name.casefold()):
        if path.is_file() and path.suffix.casefold() in {".jpg", ".jpeg", ".png"}:
            rows.append(f"{path.name}\t{path.stat().st_size}\n")
    return {
        "files": len(rows),
        "name_size_inventory_sha256": hashlib.sha256("".join(rows).encode("utf-8")).hexdigest(),
    }


def _safe_project_relative_path(value: object, field: str) -> Path:
    """Parse a manifest path without allowing an absolute/path-traversal escape."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty project-relative path")
    path = Path(value)
    if path.is_absolute() or path.drive or not path.parts or any(part in {".", ".."} for part in path.parts):
        raise ValueError(f"{field} must be a safe project-relative path")
    return path


def _project_root_from_manifest_raw_directory(payload: dict[str, Any], raw_dir: Path) -> Path:
    """Recover the project root used by a provenance-disjoint frozen manifest.

    The freezer stores ``raw_data.directory`` and ``records_subset.path`` as
    project-relative paths.  Deriving the root from the already caller-approved
    raw-data location means the loader never relies on the process CWD and never
    follows a subset reference outside that project.
    """
    raw_data = payload.get("raw_data")
    if not isinstance(raw_data, dict):
        raise ValueError("records-subset manifest lacks raw_data metadata")
    declared_raw_directory = _safe_project_relative_path(raw_data.get("directory"), "raw_data.directory")
    resolved_raw_dir = raw_dir.resolve()
    project_root = resolved_raw_dir
    for _ in declared_raw_directory.parts:
        project_root = project_root.parent
    if (project_root / declared_raw_directory).resolve() != resolved_raw_dir:
        raise ValueError("raw_data.directory does not resolve to the supplied raw-data directory")
    return project_root


def _load_records_subset(payload: dict[str, Any], raw_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    """Read and validate an immutable record subset when one is declared.

    The raw ``标签.json`` file is still byte-hashed by the caller, but is
    intentionally *not parsed* on this path.  That separation keeps the
    provenance-disjoint cohort from accidentally rebuilding a dataset from the
    full corpus (which contains historical validation/test records).
    """
    descriptor = payload.get("records_subset")
    if descriptor is None:
        return None
    if not isinstance(descriptor, dict):
        raise ValueError("records_subset must be an object in the frozen manifest")
    subset_relative_path = _safe_project_relative_path(descriptor.get("path"), "records_subset.path")
    project_root = _project_root_from_manifest_raw_directory(payload, raw_dir)
    subset_path = (project_root / subset_relative_path).resolve()
    if not subset_path.is_relative_to(project_root):
        raise ValueError("records_subset.path escapes the inferred project root")
    if not subset_path.is_file():
        raise FileNotFoundError(f"records subset is missing: {subset_path}")
    expected_sha256 = descriptor.get("sha256")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError("records_subset.sha256 must be a SHA-256 digest")
    actual_sha256 = sha256_file(subset_path)
    if actual_sha256 != expected_sha256:
        raise ValueError("records subset SHA-256 does not match the frozen outer-CV manifest")
    try:
        records = json.loads(subset_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("records subset is not valid JSON") from error
    if not isinstance(records, list):
        raise ValueError("records subset must contain a JSON list")
    expected_records = descriptor.get("records")
    if isinstance(expected_records, bool) or not isinstance(expected_records, int) or expected_records <= 0:
        raise ValueError("records_subset.records must be a positive integer")
    if len(records) != expected_records:
        raise ValueError("records subset count does not match the frozen outer-CV manifest")
    plates: list[str] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"records subset item {index} is not an object")
        plate = record.get("RubbingName")
        if not isinstance(plate, str) or not plate:
            raise ValueError(f"records subset item {index} lacks a non-empty RubbingName")
        plates.append(plate)
    if len(plates) != len(set(plates)):
        raise ValueError("records subset contains duplicate RubbingName records")
    expected_plate_hash = descriptor.get("plate_list_sha256")
    if not isinstance(expected_plate_hash, str) or len(expected_plate_hash) != 64:
        raise ValueError("records_subset.plate_list_sha256 must be a SHA-256 digest")
    actual_plate_hash = plate_list_sha256(sorted(plates))
    if actual_plate_hash != expected_plate_hash:
        raise ValueError("records subset plate member hash does not match the frozen outer-CV manifest")
    # The new freezer redundantly records the eligible universe outside the
    # subset descriptor.  Validate both representations before any role can use
    # this subset, without parsing the full raw label corpus.
    universe = payload.get("plate_universe")
    eligible = universe.get("eligible_plates") if isinstance(universe, dict) else None
    if not isinstance(eligible, dict):
        raise ValueError("records-subset manifest lacks plate_universe.eligible_plates")
    eligible_count = eligible.get("plates")
    if isinstance(eligible_count, bool) or not isinstance(eligible_count, int):
        raise ValueError("plate_universe.eligible_plates.plates must be a positive integer")
    if eligible_count != len(plates) or str(eligible.get("plates_sha256")) != actual_plate_hash:
        raise ValueError("records subset does not match the declared eligible plate universe")
    legacy = universe.get("legacy_provenance")
    initial_train = legacy.get("eligible_initial_train") if isinstance(legacy, dict) else None
    if not isinstance(initial_train, dict) or not isinstance(initial_train.get("members"), list):
        raise ValueError("records-subset manifest lacks the legacy initial-train membership ledger")
    initial_train_members = initial_train["members"]
    if not all(isinstance(plate, str) and plate for plate in initial_train_members):
        raise ValueError("legacy initial-train membership ledger contains an invalid plate")
    if len(initial_train_members) != len(set(initial_train_members)):
        raise ValueError("legacy initial-train membership ledger contains duplicate plates")
    initial_train_hash = plate_list_sha256(initial_train_members)
    initial_train_count = initial_train.get("plates")
    if (
        isinstance(initial_train_count, bool)
        or not isinstance(initial_train_count, int)
        or initial_train_count != len(initial_train_members)
        or str(initial_train.get("plates_sha256")) != initial_train_hash
    ):
        raise ValueError("legacy initial-train membership ledger count/hash is inconsistent")
    if set(plates) != set(initial_train_members) or actual_plate_hash != initial_train_hash:
        raise ValueError("records subset is not exactly the legacy initial-train reservoir")
    return records, {
        "path": str(subset_path),
        "sha256": actual_sha256,
        "records": len(records),
        "plate_list_sha256": actual_plate_hash,
    }


def _validate_duplicate_grouping(payload: dict[str, Any], membership: dict[str, int]) -> None:
    """Ensure every conservative duplicate-candidate edge remains within one fold."""
    grouping = payload.get("duplicate_grouping")
    if grouping is None:
        return
    if not isinstance(grouping, dict) or not isinstance(grouping.get("candidate_edges"), list):
        raise ValueError("duplicate_grouping must provide a candidate_edges list")
    for index, edge in enumerate(grouping["candidate_edges"]):
        if not isinstance(edge, dict):
            raise ValueError(f"duplicate grouping edge {index} is not an object")
        first, second = edge.get("first_plate"), edge.get("second_plate")
        if not isinstance(first, str) or not isinstance(second, str) or first not in membership or second not in membership:
            raise ValueError(f"duplicate grouping edge {index} references an unknown plate")
        if membership[first] != membership[second]:
            raise ValueError(f"duplicate grouping edge {index} crosses frozen outer folds")


def load_frozen_outer_cv_split(manifest_path: Path, raw_dir: Path, outer_fold: int) -> dict[str, Any]:
    """Load and validate one fixed nested-CV role assignment.

    Returned provenance intentionally includes member-list hashes/counts, not
    the full lists.  The full lists remain in the immutable input manifest.
    """
    if not manifest_path.is_file():
        raise FileNotFoundError(f"outer-CV manifest is missing: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    folds = int(payload.get("folds", 0))
    if folds != 5:
        raise ValueError("detector confirmation requires exactly five frozen outer folds")
    if outer_fold not in range(folds):
        raise ValueError("outer fold is outside the frozen five-fold range")
    labels_path = raw_dir / "标签.json"
    expected_labels = str(payload["raw_data"]["labels"]["sha256"])
    actual_labels = sha256_file(labels_path)
    if actual_labels != expected_labels:
        raise ValueError("raw label SHA-256 does not match the frozen outer-CV manifest")
    expected_inventories = payload["raw_data"].get("image_inventories")
    if not isinstance(expected_inventories, dict):
        raise ValueError("frozen outer-CV manifest lacks raw image inventories")
    actual_inventories: dict[str, dict[str, Any]] = {}
    for source, (directory_name, _) in SOURCE_FIELDS.items():
        expected_inventory = expected_inventories.get(source)
        if not isinstance(expected_inventory, dict):
            raise ValueError(f"frozen outer-CV manifest lacks {source} image inventory")
        actual_inventory = image_name_size_inventory(raw_dir / directory_name)
        if int(expected_inventory.get("files", -1)) != actual_inventory["files"] or str(expected_inventory.get("name_size_inventory_sha256")) != actual_inventory["name_size_inventory_sha256"]:
            raise ValueError(f"raw {source} image inventory does not match the frozen outer-CV manifest")
        actual_inventories[source] = actual_inventory
    records_subset = _load_records_subset(payload, raw_dir)
    if records_subset is None:
        # Legacy full-corpus manifests retain their exact historical behavior.
        # New provenance-disjoint manifests deliberately never take this path.
        records = json.loads(labels_path.read_text(encoding="utf-8"))
        plates = [str(row["RubbingName"]) for row in records]
        subset_records: list[dict[str, Any]] | None = None
        subset_provenance: dict[str, Any] | None = None
    else:
        subset_records, subset_provenance = records_subset
        plates = [str(row["RubbingName"]) for row in subset_records]
    if len(plates) != len(set(plates)):
        raise ValueError("loaded detector records contain duplicate RubbingName records")
    membership = {str(key): int(value) for key, value in payload["plate_to_outer_fold"].items()}
    if set(membership) != set(plates):
        raise ValueError("outer-CV manifest plate universe does not match the loaded detector records")
    _validate_duplicate_grouping(payload, membership)
    role_container = payload["nested_roles"].get(str(outer_fold))
    if not isinstance(role_container, dict):
        raise ValueError("requested outer fold has no frozen role definition")
    expected_inner_val = (outer_fold + 1) % folds
    if int(role_container["inner_validation_fold"]) != expected_inner_val:
        raise ValueError("frozen inner-validation fold violates the declared nested rule")
    members = {role: [str(value) for value in role_container["members"][role]] for role in ROLE_NAMES}
    union = members["inner_train"] + members["inner_val"] + members["outer_test"]
    if not all(members.values()) or len(union) != len(set(union)) or set(union) != set(plates):
        raise ValueError("frozen roles must be non-empty, disjoint, and cover every plate exactly once")
    expected_folds = {
        "inner_train": {fold for fold in range(folds) if fold not in {outer_fold, expected_inner_val}},
        "inner_val": {expected_inner_val},
        "outer_test": {outer_fold},
    }
    summaries = role_container["summary"]
    for role, role_plates in members.items():
        if {membership[plate] for plate in role_plates} != expected_folds[role]:
            raise ValueError(f"frozen {role} members do not match their declared folds")
        if int(summaries[role]["plates"]) != len(role_plates):
            raise ValueError(f"frozen {role} plate count mismatch")
        if str(summaries[role]["plates_sha256"]) != plate_list_sha256(role_plates):
            raise ValueError(f"frozen {role} member hash mismatch")
    result: dict[str, Any] = {
        "mode": "frozen_provenance_disjoint_initial_train_nested_outer_cv" if subset_records is not None else "frozen_pure_rubbingname_nested_outer_cv",
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "raw_labels_path": str(labels_path),
        "raw_labels_sha256": actual_labels,
        "raw_image_inventories": actual_inventories,
        "outer_fold": outer_fold,
        "inner_validation_fold": expected_inner_val,
        "members": members,
        "roles": {
            role: {
                "plates": len(members[role]),
                "plates_sha256": plate_list_sha256(members[role]),
                "visible_annotated_boxes": int(summaries[role]["visible_annotated_boxes"]),
                "placeholder_boxes_excluded": int(summaries[role]["placeholder_boxes_excluded"]),
            }
            for role in ROLE_NAMES
        },
    }
    if subset_records is not None and subset_provenance is not None:
        # ``records_subset_records`` is intentionally an in-memory-only value.
        # compact_split_provenance excludes it before writing checkpoints.
        result["records_subset"] = subset_provenance
        result["records_subset_records"] = subset_records
    return result


def compact_split_provenance(split: dict[str, Any]) -> dict[str, Any]:
    """Produce stable, checkpoint-safe provenance without expanding members."""
    compact = {
        "mode": str(split["mode"]),
        "manifest_path": str(split["manifest_path"]),
        "manifest_sha256": str(split["manifest_sha256"]),
        "raw_labels_path": str(split["raw_labels_path"]),
        "raw_labels_sha256": str(split["raw_labels_sha256"]),
        "raw_image_inventories": dict(split["raw_image_inventories"]),
        "outer_fold": int(split["outer_fold"]),
        "inner_validation_fold": int(split["inner_validation_fold"]),
        "roles": dict(split["roles"]),
    }
    if "records_subset" in split:
        compact["records_subset"] = dict(split["records_subset"])
    return compact
