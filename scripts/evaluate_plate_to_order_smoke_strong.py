"""Validation-only plate pipeline using the family-aware detector adapter."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_plate_to_order_smoke as pipeline  # noqa: E402

from oracle_vit.plate_inference_strong import (  # noqa: E402
    load_detector_any_family,
    predict_plate_detections,
)


def load_detector_with_legacy_metadata(path, device):
    """Adapt newer strong-detector metadata to the frozen evaluator contract.

    The strong checkpoint records the selected epoch as ``epoch`` whereas the
    historical anchor checkpoint records it as ``best_epoch``.  This adapter
    only supplies that provenance alias; model construction and predictions
    remain delegated to the family-aware loader.
    """

    model, state = load_detector_any_family(path, device)
    enriched = dict(state)
    if "best_epoch" not in enriched:
        if "epoch" not in enriched:
            raise KeyError("detector checkpoint has neither 'best_epoch' nor 'epoch'")
        enriched["best_epoch"] = enriched["epoch"]
    return model, enriched


pipeline.load_detector = load_detector_with_legacy_metadata  # type: ignore[assignment]
pipeline.predict_plate_detections = predict_plate_detections  # type: ignore[assignment]


def main() -> None:
    print(
        "[validation strong-detector wrapper] family-aware detector loader enabled; "
        "the underlying evaluator remains validation-only",
        flush=True,
    )
    pipeline.main()


if __name__ == "__main__":
    main()

