"""Contract tests for the additive strong-detector validation wrapper."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_plate_to_order_smoke_strong as wrapper  # noqa: E402


def test_epoch_is_aliased_to_best_epoch_without_mutating_source(monkeypatch: pytest.MonkeyPatch) -> None:
    source = {"epoch": 10, "seed": 20260981}
    monkeypatch.setattr(wrapper, "load_detector_any_family", lambda path, device: ("model", source))

    model, state = wrapper.load_detector_with_legacy_metadata(Path("unused.pt"), "cpu")

    assert model == "model"
    assert state["best_epoch"] == 10
    assert "best_epoch" not in source


def test_existing_best_epoch_is_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        wrapper,
        "load_detector_any_family",
        lambda path, device: ("model", {"epoch": 10, "best_epoch": 7}),
    )

    _, state = wrapper.load_detector_with_legacy_metadata(Path("unused.pt"), "cpu")

    assert state["best_epoch"] == 7


def test_missing_epoch_metadata_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(wrapper, "load_detector_any_family", lambda path, device: ("model", {}))

    with pytest.raises(KeyError, match="neither 'best_epoch' nor 'epoch'"):
        wrapper.load_detector_with_legacy_metadata(Path("unused.pt"), "cpu")
