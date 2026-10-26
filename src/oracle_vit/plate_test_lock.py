"""Immutable allow-list checks for the historical detector test reader."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any


def normalized_project_path(value: str) -> str:
    """Normalize a manifest path without resolving it outside the caller's root."""
    return Path(str(value)).as_posix().casefold()


def validate_historical_test_selection(
    manifest: dict[str, Any],
    *,
    config_path: str,
    config_sha256: str,
    checkpoints: Sequence[dict[str, Any]],
) -> None:
    """Require the exact frozen config and three checkpoint hashes in order."""
    if str(manifest.get("test_access")) != "historical_frozen_reference_only":
        raise PermissionError("selection manifest is not authorized for historical test access")
    expected_config = manifest.get("config", {})
    if normalized_project_path(str(expected_config.get("path", ""))) != normalized_project_path(config_path):
        raise PermissionError("requested config path is not the frozen historical test config")
    if str(expected_config.get("sha256", "")) != str(config_sha256):
        raise PermissionError("requested config hash is not the frozen historical test config")
    expected = list(manifest.get("checkpoints", []))
    if len(expected) != 3 or len(checkpoints) != 3:
        raise PermissionError("historical test access requires exactly three locked checkpoints")
    for index, (locked, requested) in enumerate(zip(expected, checkpoints), start=1):
        if normalized_project_path(str(locked.get("path", ""))) != normalized_project_path(str(requested.get("path", ""))):
            raise PermissionError(f"checkpoint {index} path is not in the frozen historical selection")
        if str(locked.get("sha256", "")) != str(requested.get("sha256", "")):
            raise PermissionError(f"checkpoint {index} hash is not in the frozen historical selection")
