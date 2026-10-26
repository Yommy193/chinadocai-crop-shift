"""Analyze the frozen 2 x 2 detector confirmation with paired bootstrap.

The input is an explicit confirmation matrix produced by
``run_crop_shift_confirmation.py``.  For each policy, the interaction is
computed as M_TT + M_FF - M_TF - M_FT, where the first index is the detector
whose crops were used for training and the second is the detector used at
evaluation.  Plate weights are synchronized across all policies, seeds,
detectors, modalities, and strata.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from oracle_vit.utils import atomic_json_dump, file_sha256
try:
    # ``python -m scripts.analyze_crop_shift_confirmation`` from the project
    # root and direct ``python scripts/analyze_crop_shift_confirmation.py``
    # are both supported by the additive helper.
    from scripts.analyze_crop_shift_targets import (
        CellSpec,
        _matched_eligible,
        _slice_specs,
        load_matrix,
        load_strata_reference,
        synchronized_plate_weights,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script invocation
    from analyze_crop_shift_targets import (  # type: ignore[no-redef]
        CellSpec,
        _matched_eligible,
        _slice_specs,
        load_matrix,
        load_strata_reference,
        synchronized_plate_weights,
    )


ROOT = Path(__file__).resolve().parents[1]
TRAIN_DETECTORS = ("anchor", "frcnn")
EVAL_DETECTORS = ("anchor", "frcnn")
POLICIES = {
    "pred_step": {"anchor": "anchor_pred_step", "frcnn": "frcnn_pred_step"},
    "mix075_step": {"anchor": "anchor_mix075_step", "frcnn": "frcnn_mix075_step"},
}
METRICS = ("predicted_crop_top1", "predicted_crop_top5")


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _interval(values: np.ndarray) -> tuple[float | None, float | None]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None, None
    q = np.quantile(finite, [0.025, 0.975], method="linear")
    return float(q[0]), float(q[1])


def _block(point: float | None, distribution: np.ndarray, denominator: int) -> dict[str, Any]:
    finite = distribution[np.isfinite(distribution)]
    low, high = _interval(finite)
    return {
        "matched_in_vocabulary_targets": int(denominator),
        "point_difference": point,
        "point_difference_pp": None if point is None else 100.0 * point,
        "bootstrap_mean_difference": float(finite.mean()) if finite.size else None,
        "interval_95": {"low": low, "high": high},
        "interval_95_pp": {
            "low": None if low is None else 100.0 * low,
            "high": None if high is None else 100.0 * high,
        },
        "effective_replicates": int(finite.size),
        "positive_replicates": int(np.count_nonzero(finite > 0.0)),
        "zero_replicates": int(np.count_nonzero(finite == 0.0)),
        "negative_replicates": int(np.count_nonzero(finite < 0.0)),
    }


def _load_specs(manifest: Path) -> list[CellSpec]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("confirmation matrix manifest must be a non-empty list")
    specs: list[CellSpec] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("confirmation matrix cell must be an object")
        specs.append(
            CellSpec(
                arm=str(item["arm"]),
                recognition_seed=int(item["recognition_seed"]),
                eval_detector=str(item["eval_detector"]),
                rows_path=_resolve(str(item["rows"])),
            )
        )
    return specs


def _slice_match(row: dict[str, Any], reference: Any, slice_spec: Any) -> bool:
    return bool(slice_spec.predicate(row, reference))


def _interaction_for_policy(
    *,
    policy: str,
    cells: dict[tuple[str, int, str], dict[str, dict[str, Any]]],
    seeds: list[int],
    plates: list[str],
    slices: list[Any],
    reference: Any,
    weights: np.ndarray,
) -> dict[str, Any]:
    plate_index = {plate: index for index, plate in enumerate(plates)}
    weight_float = weights.astype(np.float64, copy=False)
    per_seed_point: dict[tuple[int, str, str], np.ndarray] = {}
    per_seed_boot: dict[tuple[int, str, str], np.ndarray] = {}
    denominators: dict[tuple[int, str, str], np.ndarray] = {}

    # Build per-cell plate-by-slice numerator/denominator tensors.  Evaluation
    # detector geometry is shared across train arms, but we retain each cell's
    # denominator and verify equality below rather than assuming it.
    for train_detector in TRAIN_DETECTORS:
        arm = POLICIES[policy][train_detector]
        for seed in seeds:
            for eval_detector in EVAL_DETECTORS:
                rows = cells[(arm, seed, eval_detector)]
                den = np.zeros((len(plates), len(slices)), dtype=np.float64)
                num = np.zeros((len(plates), len(slices), len(METRICS)), dtype=np.float64)
                for row in rows.values():
                    if not _matched_eligible(row):
                        continue
                    plate = plate_index[str(row["plate"])]
                    for slice_index, spec in enumerate(slices):
                        if not _slice_match(row, reference, spec):
                            continue
                        den[plate, slice_index] += 1.0
                        for metric_index, metric in enumerate(METRICS):
                            num[plate, slice_index, metric_index] += int(row.get(metric) or 0)
                key = (seed, train_detector, eval_detector)
                denominators[key] = den
                boot_den = weight_float @ den
                boot_num = (weight_float @ num.reshape(len(plates), -1)).reshape(
                    len(weights), len(slices), len(METRICS)
                )
                point_den = den.sum(axis=0)
                point_num = num.sum(axis=0)
                with np.errstate(divide="ignore", invalid="ignore"):
                    per_seed_point[key] = np.divide(
                        point_num,
                        point_den[:, None],
                        out=np.full_like(point_num, np.nan),
                        where=point_den[:, None] > 0,
                    )
                    per_seed_boot[key] = np.divide(
                        boot_num,
                        boot_den[:, :, None],
                        out=np.full_like(boot_num, np.nan),
                        where=boot_den[:, :, None] > 0,
                    )

    # Detector geometry should give the same denominator for the two train
    # arms for each evaluation detector and seed.  Run this only after both
    # training arms have populated the denominator table.
    for seed in seeds:
        for eval_detector in EVAL_DETECTORS:
            left = denominators[(seed, "anchor", eval_detector)]
            right = denominators[(seed, "frcnn", eval_detector)]
            if not np.array_equal(left, right):
                raise ValueError(
                    f"matched-target denominators differ for policy={policy}, seed={seed}, "
                    f"eval_detector={eval_detector}"
                )

    slice_blocks: list[dict[str, Any]] = []
    for slice_index in range(len(slices)):
        metric_blocks: dict[str, Any] = {}
        for metric_index, metric in enumerate(METRICS):
            point_values: list[float] = []
            distributions: list[np.ndarray] = []
            for seed in seeds:
                tt = per_seed_point[(seed, "anchor", "anchor")][slice_index, metric_index]
                ff = per_seed_point[(seed, "frcnn", "frcnn")][slice_index, metric_index]
                tf = per_seed_point[(seed, "anchor", "frcnn")][slice_index, metric_index]
                ft = per_seed_point[(seed, "frcnn", "anchor")][slice_index, metric_index]
                values = np.asarray([tt, ff, tf, ft], dtype=np.float64)
                point_values.append(float(values[0] + values[1] - values[2] - values[3]))
                boot = (
                    per_seed_boot[(seed, "anchor", "anchor")][:, slice_index, metric_index]
                    + per_seed_boot[(seed, "frcnn", "frcnn")][:, slice_index, metric_index]
                    - per_seed_boot[(seed, "anchor", "frcnn")][:, slice_index, metric_index]
                    - per_seed_boot[(seed, "frcnn", "anchor")][:, slice_index, metric_index]
                )
                distributions.append(boot)
            distribution = np.mean(np.stack(distributions, axis=1), axis=1)
            denominator = int(denominators[(seeds[0], "anchor", "anchor")].sum(axis=0)[slice_index])
            point = float(np.mean(point_values)) if all(math.isfinite(value) for value in point_values) else None
            metric_blocks[metric] = _block(point, distribution, denominator)
        slice_blocks.append(metric_blocks)

    nested: dict[str, Any] = {}
    for spec, block in zip(slices, slice_blocks):
        if spec.family == "overall":
            nested["overall"] = block
        else:
            nested.setdefault(spec.family, {})[spec.name] = block
    return nested


def build_confirmation_analysis(
    *,
    manifest: str | Path,
    strata_reference: str | Path,
    output: str | Path,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 20261009,
) -> dict[str, Any]:
    manifest_path = _resolve(manifest)
    reference = load_strata_reference(strata_reference)
    specs = _load_specs(manifest_path)
    cells, arms, seeds, detectors = load_matrix(specs, reference)
    expected_arms = {
        "gold_step",
        "anchor_pred_step",
        "frcnn_pred_step",
        "anchor_mix075_step",
        "frcnn_mix075_step",
    }
    if set(arms) != expected_arms or set(detectors) != set(EVAL_DETECTORS):
        raise ValueError(f"unexpected confirmation matrix arms/detectors: {arms}, {detectors}")
    if len(seeds) != 3:
        raise ValueError(f"confirmation requires three seeds, got {seeds}")
    first_rows = next(iter(cells.values()))
    plates = sorted({str(row["plate"]) for row in first_rows.values()})
    weights, weights_hash = synchronized_plate_weights(plates, bootstrap_replicates, bootstrap_seed)
    slices = _slice_specs(reference)
    interactions = {
        policy: _interaction_for_policy(
            policy=policy,
            cells=cells,
            seeds=seeds,
            plates=plates,
            slices=slices,
            reference=reference,
            weights=weights,
        )
        for policy in POLICIES
    }
    output_path = _resolve(output)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite confirmation analysis: {output_path}")
    payload = {
        "schema_version": 1,
        "scope": "three-seed validation-only detector train/evaluation interaction",
        "metric": "matched in-vocabulary target conditional predicted-crop recognition",
        "interaction_definition": "M_TT + M_FF - M_TF - M_FT; first index train detector, second evaluation detector",
        "inputs": {
            "matrix_manifest": str(manifest_path),
            "matrix_manifest_sha256": file_sha256(manifest_path),
            "strata_reference": str(reference.path),
            "strata_reference_sha256": reference.sha256,
            "matrix_cells": len(specs),
            "arms": arms,
            "recognition_seeds": [int(seed) for seed in seeds],
            "evaluation_detectors": detectors,
            "target_count_per_cell": len(first_rows),
            "plate_count": len(plates),
        },
        "policies": POLICIES,
        "interactions": interactions,
        "bootstrap": {
            "replicates": int(bootstrap_replicates),
            "seed": int(bootstrap_seed),
            "resampling_unit": "plate",
            "plate_order": plates,
            "synchronized_weights_sha256": weights_hash,
            "synchronization": "one multinomial plate-weight matrix reused across policies, seeds, detectors, modalities, and strata",
        },
        "test_access": "prohibited; every matrix cell was evaluated with split=val",
    }
    atomic_json_dump(output_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the frozen crop-shift confirmation interaction")
    parser.add_argument("--matrix-manifest", required=True)
    parser.add_argument("--strata-reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261009)
    args = parser.parse_args()
    if args.bootstrap_replicates <= 0:
        raise ValueError("--bootstrap-replicates must be positive")
    report = build_confirmation_analysis(
        manifest=args.matrix_manifest,
        strata_reference=args.strata_reference,
        output=args.output,
        bootstrap_replicates=int(args.bootstrap_replicates),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    print(
        json.dumps(
            {
                "output": str(_resolve(args.output)),
                "cells": report["inputs"]["matrix_cells"],
                "seeds": report["inputs"]["recognition_seeds"],
                "bootstrap_replicates": report["bootstrap"]["replicates"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
