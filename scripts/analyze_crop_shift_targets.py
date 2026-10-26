"""Stratified, paired analysis for crop-shift per-target JSONL exports.

The inputs are the additive rows written by ``evaluate_crop_shift_targets.py``.
Every matrix cell is named explicitly by training arm, recognition seed, and
evaluation detector.  The analyzer is deliberately strict: matrix cells must
form a complete Cartesian product, every row must have a unique target ID, and
all cells must cover exactly the same targets.

Only recognition conditioned on an IoU-matched, in-vocabulary target is
reported here.  A crop-recognition failure remains in that conditional
denominator and contributes zero correct predictions.  Detection coverage is
therefore never relabelled as a target-stratum recognition score.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from oracle_vit.utils import atomic_json_dump, file_sha256


ROOT = Path(__file__).resolve().parents[1]
METRICS = (
    "predicted_crop_top1",
    "predicted_crop_top5",
    "gold_crop_top1",
    "gold_crop_top5",
)
MODALITIES = ("rubbing", "facsimile")


@dataclass(frozen=True)
class CellSpec:
    arm: str
    recognition_seed: int
    eval_detector: str
    rows_path: Path

    @property
    def key(self) -> tuple[str, int, str]:
        return self.arm, self.recognition_seed, self.eval_detector


@dataclass(frozen=True)
class NamedBin:
    name: str
    minimum: float
    maximum: float | None
    maximum_inclusive: bool

    def contains(self, value: float) -> bool:
        if value < self.minimum:
            return False
        if self.maximum is None:
            return True
        if self.maximum_inclusive:
            return value <= self.maximum
        return value < self.maximum


@dataclass(frozen=True)
class StrataReference:
    path: Path
    sha256: str
    payload: dict[str, Any]
    class_train_counts: tuple[int, ...]
    frequency_bins: tuple[NamedBin, ...]
    iou_bins: tuple[NamedBin, ...]
    size_q33: float
    size_q67: float


@dataclass(frozen=True)
class SliceSpec:
    family: str
    name: str
    predicate: Callable[[dict[str, Any], StrataReference], bool]


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _parse_frequency_bins(payload: Any) -> tuple[NamedBin, ...]:
    if not isinstance(payload, list) or not payload:
        raise ValueError("strata reference must define non-empty frequency_bins")
    bins: list[NamedBin] = []
    names: set[str] = set()
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"frequency_bins[{index}] must be an object")
        name = str(item.get("name", "")).strip()
        if not name or name in names:
            raise ValueError("frequency bin names must be non-empty and unique")
        minimum = item.get("minimum")
        maximum = item.get("maximum")
        if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 0:
            raise ValueError(f"frequency bin {name!r} has an invalid minimum")
        if maximum is not None and (
            isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < minimum
        ):
            raise ValueError(f"frequency bin {name!r} has an invalid maximum")
        bins.append(NamedBin(name, float(minimum), None if maximum is None else float(maximum), True))
        names.add(name)
    ordered = sorted(bins, key=lambda item: item.minimum)
    for previous, current in zip(ordered, ordered[1:]):
        if previous.maximum is None or current.minimum != previous.maximum + 1:
            raise ValueError("frequency bins must be ordered, disjoint, and contiguous")
    if ordered[-1].maximum is not None:
        raise ValueError("the last frequency bin must be open-ended")
    return tuple(ordered)


def _parse_iou_bins(payload: Any) -> tuple[NamedBin, ...]:
    if not isinstance(payload, list) or not payload:
        raise ValueError("strata reference must define non-empty iou_bins")
    bins: list[NamedBin] = []
    names: set[str] = set()
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"iou_bins[{index}] must be an object")
        name = str(item.get("name", "")).strip()
        if not name or name in names:
            raise ValueError("IoU bin names must be non-empty and unique")
        minimum = _finite_number(item.get("minimum"), f"IoU bin {name!r} minimum")
        has_exclusive = "maximum_exclusive" in item
        has_inclusive = "maximum_inclusive" in item
        if has_exclusive == has_inclusive:
            raise ValueError(
                f"IoU bin {name!r} must define exactly one of maximum_exclusive/maximum_inclusive"
            )
        maximum = _finite_number(
            item["maximum_inclusive"] if has_inclusive else item["maximum_exclusive"],
            f"IoU bin {name!r} maximum",
        )
        if not 0.0 <= minimum < maximum <= 1.0:
            raise ValueError(f"IoU bin {name!r} must remain inside [0, 1]")
        bins.append(NamedBin(name, minimum, maximum, has_inclusive))
        names.add(name)
    ordered = sorted(bins, key=lambda item: item.minimum)
    for previous, current in zip(ordered, ordered[1:]):
        if previous.maximum is None or not math.isclose(previous.maximum, current.minimum, abs_tol=1e-12):
            raise ValueError("IoU bins must be ordered and contiguous")
        if previous.maximum_inclusive:
            raise ValueError("only the final IoU bin may include its upper boundary")
    if not ordered[-1].maximum_inclusive:
        raise ValueError("the final IoU bin must include its upper boundary")
    return tuple(ordered)


def load_strata_reference(path: str | Path) -> StrataReference:
    resolved = _resolve_path(path)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or int(payload.get("schema_version", -1)) != 1:
        raise ValueError("unsupported or missing strata-reference schema_version")
    counts_a = payload.get("class_train_counts")
    counts_b = payload.get("counts")
    counts = counts_a if counts_a is not None else counts_b
    if counts_a is not None and counts_b is not None and counts_a != counts_b:
        raise ValueError("strata reference counts and class_train_counts disagree")
    if not isinstance(counts, list) or not counts or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts
    ):
        raise ValueError("strata reference must contain non-negative class_train_counts")
    tertiles = payload.get("normalized_gold_box_area_tertiles")
    if not isinstance(tertiles, dict):
        raise ValueError("strata reference lacks normalized_gold_box_area_tertiles")
    q33 = _finite_number(tertiles.get("q33"), "normalized size q33")
    q67 = _finite_number(tertiles.get("q67"), "normalized size q67")
    if not 0.0 <= q33 <= q67 <= 1.0:
        raise ValueError("normalized size tertiles must satisfy 0 <= q33 <= q67 <= 1")
    return StrataReference(
        path=resolved,
        sha256=file_sha256(resolved),
        payload=payload,
        class_train_counts=tuple(int(value) for value in counts),
        frequency_bins=_parse_frequency_bins(payload.get("frequency_bins")),
        iou_bins=_parse_iou_bins(payload.get("iou_bins")),
        size_q33=q33,
        size_q67=q67,
    )


def _named_bin(value: float, bins: Sequence[NamedBin], label: str) -> str:
    matches = [item.name for item in bins if item.contains(value)]
    if len(matches) != 1:
        raise ValueError(f"{label}={value} belongs to {len(matches)} frozen bins")
    return matches[0]


def frequency_stratum(row: dict[str, Any], reference: StrataReference) -> str | None:
    if not bool(row["evaluation_eligible"]):
        return None
    class_id = int(row["target_class_id"])
    return _named_bin(float(reference.class_train_counts[class_id]), reference.frequency_bins, "frequency")


def size_stratum(row: dict[str, Any], reference: StrataReference) -> str:
    value = float(row["gold_box_area_fraction"])
    if value < reference.size_q33:
        return "small"
    if value < reference.size_q67:
        return "medium"
    return "large"


def iou_stratum(row: dict[str, Any], reference: StrataReference) -> str | None:
    if not bool(row["matched_at_iou"]):
        return None
    return _named_bin(float(row["matching_iou"]), reference.iou_bins, "matching_iou")


def _binary_or_none(value: Any, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and value in (0, 1):
        return value
    raise ValueError(f"{label} must be 0, 1, or null")


def _validate_target_row(
    row: dict[str, Any], spec: CellSpec, reference: StrataReference, line_number: int
) -> None:
    location = f"{spec.rows_path}:{line_number}"
    target_id = row.get("target_id")
    if not isinstance(target_id, str) or not target_id.strip():
        raise ValueError(f"missing target_id at {location}")
    if not isinstance(row.get("plate"), str) or not str(row["plate"]).strip():
        raise ValueError(f"missing plate for {target_id} at {location}")
    if row.get("source") not in MODALITIES:
        raise ValueError(f"unsupported modality for {target_id} at {location}: {row.get('source')!r}")
    if row.get("detector_label") != spec.eval_detector:
        raise ValueError(
            f"detector label mismatch for {target_id}: row={row.get('detector_label')!r}, "
            f"cell={spec.eval_detector!r}"
        )
    if not isinstance(row.get("evaluation_eligible"), bool):
        raise ValueError(f"evaluation_eligible must be boolean for {target_id}")
    if not isinstance(row.get("matched_at_iou"), bool):
        raise ValueError(f"matched_at_iou must be boolean for {target_id}")
    area = _finite_number(row.get("gold_box_area_fraction"), f"gold_box_area_fraction for {target_id}")
    if not 0.0 <= area <= 1.0 + 1e-9:
        raise ValueError(f"gold_box_area_fraction is outside [0, 1] for {target_id}")

    if row["evaluation_eligible"]:
        class_id = row.get("target_class_id")
        if isinstance(class_id, bool) or not isinstance(class_id, int):
            raise ValueError(f"eligible target {target_id} lacks an integer target_class_id")
        if not 0 <= class_id < len(reference.class_train_counts):
            raise ValueError(f"target_class_id is outside the frozen count vector for {target_id}")
        frozen_count = reference.class_train_counts[class_id]
        if frozen_count <= 0:
            raise ValueError(f"eligible target {target_id} has zero frozen training frequency")
        recorded_count = row.get("target_train_count")
        if recorded_count is not None and recorded_count != frozen_count:
            raise ValueError(
                f"target_train_count disagrees with frozen reference for {target_id}: "
                f"row={recorded_count}, frozen={frozen_count}"
            )

    if row["matched_at_iou"]:
        overlap = _finite_number(row.get("matching_iou"), f"matching_iou for {target_id}")
        if not 0.0 <= overlap <= 1.0:
            raise ValueError(f"matching_iou is outside [0, 1] for {target_id}")
        iou_stratum(row, reference)
    elif row.get("matching_iou") is not None:
        raise ValueError(f"unmatched target {target_id} unexpectedly has matching_iou")
    if "matched_status" in row:
        expected_status = "matched" if row["matched_at_iou"] else "unmatched"
        if row["matched_status"] != expected_status:
            raise ValueError(f"matched_status disagrees with matched_at_iou for {target_id}")

    values = {metric: _binary_or_none(row.get(metric), f"{metric} for {target_id}") for metric in METRICS}
    if not row["evaluation_eligible"] and any(value is not None for value in values.values()):
        raise ValueError(f"out-of-vocabulary target {target_id} has correctness values")
    if not row["matched_at_iou"] and any(
        values[key] is not None for key in ("predicted_crop_top1", "predicted_crop_top5")
    ):
        raise ValueError(f"unmatched target {target_id} has predicted-crop correctness values")
    for prefix in ("predicted_crop", "gold_crop"):
        top1, top5 = values[f"{prefix}_top1"], values[f"{prefix}_top5"]
        if (top1 is None) != (top5 is None):
            raise ValueError(f"{prefix} Top-1/Top-5 availability differs for {target_id}")
        if top1 == 1 and top5 != 1:
            raise ValueError(f"{prefix} has Top-1 correct but Top-5 incorrect for {target_id}")
    size_stratum(row, reference)
    frequency_stratum(row, reference)


def load_cell_rows(spec: CellSpec, reference: StrataReference) -> dict[str, dict[str, Any]]:
    if not spec.rows_path.is_file():
        raise FileNotFoundError(f"matrix cell rows do not exist: {spec.rows_path}")
    rows: dict[str, dict[str, Any]] = {}
    with spec.rows_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL row must be an object at {spec.rows_path}:{line_number}")
            _validate_target_row(payload, spec, reference, line_number)
            target_id = str(payload["target_id"])
            if target_id in rows:
                raise ValueError(f"duplicate target_id {target_id!r} in {spec.rows_path}")
            rows[target_id] = payload
    if not rows:
        raise ValueError(f"matrix cell is empty: {spec.rows_path}")
    return rows


def _unique_in_order(values: Iterable[Any]) -> list[Any]:
    result: list[Any] = []
    seen: set[Any] = set()
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


def _same_float(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)


def _validate_static_targets(
    cells: dict[tuple[str, int, str], dict[str, dict[str, Any]]]
) -> None:
    first_key = next(iter(cells))
    first = cells[first_key]
    static_fields = (
        "plate",
        "source",
        "group_index",
        "char_index",
        "target_label",
        "target_class_id",
        "evaluation_eligible",
        "gold_box_xyxy",
    )
    for key, rows in cells.items():
        for target_id, reference_row in first.items():
            row = rows[target_id]
            for field in static_fields:
                if row.get(field) != reference_row.get(field):
                    raise ValueError(
                        f"static target field {field!r} differs for {target_id} between {first_key} and {key}"
                    )
            if not _same_float(row.get("gold_box_area_fraction"), reference_row.get("gold_box_area_fraction")):
                raise ValueError(
                    f"gold_box_area_fraction differs for {target_id} between {first_key} and {key}"
                )


def _validate_detector_geometry(
    cells: dict[tuple[str, int, str], dict[str, dict[str, Any]]], detectors: Sequence[str]
) -> None:
    for detector in detectors:
        keys = [key for key in cells if key[2] == detector]
        reference_key = keys[0]
        reference_rows = cells[reference_key]
        for key in keys[1:]:
            rows = cells[key]
            for target_id, reference_row in reference_rows.items():
                row = rows[target_id]
                if row["matched_at_iou"] != reference_row["matched_at_iou"] or not _same_float(
                    row.get("matching_iou"), reference_row.get("matching_iou")
                ):
                    raise ValueError(
                        f"detector match geometry differs for {target_id} between {reference_key} and {key}"
                    )


def load_matrix(
    specs: Sequence[CellSpec], reference: StrataReference
) -> tuple[
    dict[tuple[str, int, str], dict[str, dict[str, Any]]],
    list[str],
    list[int],
    list[str],
]:
    if not specs:
        raise ValueError("at least one --cell is required")
    duplicate_keys = [key for key in {spec.key for spec in specs} if sum(s.key == key for s in specs) > 1]
    if duplicate_keys:
        raise ValueError(f"duplicate matrix cell(s): {sorted(duplicate_keys)}")
    arms = _unique_in_order(spec.arm for spec in specs)
    seeds = sorted({spec.recognition_seed for spec in specs})
    detectors = _unique_in_order(spec.eval_detector for spec in specs)
    expected = set(itertools.product(arms, seeds, detectors))
    actual = {spec.key for spec in specs}
    missing = sorted(expected - actual)
    if missing:
        raise ValueError(f"incomplete arm x recognition-seed x detector matrix; missing cells: {missing}")

    cells = {spec.key: load_cell_rows(spec, reference) for spec in specs}
    reference_key = specs[0].key
    reference_ids = set(cells[reference_key])
    for key, rows in cells.items():
        ids = set(rows)
        if ids != reference_ids:
            missing_ids = sorted(reference_ids - ids)[:5]
            extra_ids = sorted(ids - reference_ids)[:5]
            raise ValueError(
                f"inconsistent target coverage for cell {key}; missing target IDs={missing_ids}, "
                f"extra target IDs={extra_ids}, expected={len(reference_ids)}, observed={len(ids)}"
            )
    _validate_static_targets(cells)
    _validate_detector_geometry(cells, detectors)
    return cells, arms, seeds, detectors


def _matched_eligible(row: dict[str, Any]) -> bool:
    return bool(row["evaluation_eligible"] and row["matched_at_iou"])


def _slice_specs(reference: StrataReference) -> list[SliceSpec]:
    slices = [SliceSpec("overall", "all", lambda row, _: True)]
    slices.extend(
        SliceSpec("by_modality", modality, lambda row, _, value=modality: row["source"] == value)
        for modality in MODALITIES
    )
    slices.extend(
        SliceSpec(
            "by_matching_iou",
            item.name,
            lambda row, ref, value=item.name: iou_stratum(row, ref) == value,
        )
        for item in reference.iou_bins
    )
    slices.extend(
        SliceSpec(
            "by_normalized_size_tertile",
            name,
            lambda row, ref, value=name: size_stratum(row, ref) == value,
        )
        for name in ("small", "medium", "large")
    )
    slices.extend(
        SliceSpec(
            "by_frequency",
            item.name,
            lambda row, ref, value=item.name: frequency_stratum(row, ref) == value,
        )
        for item in reference.frequency_bins
    )
    return slices


def matched_metric_block(
    rows: Iterable[dict[str, Any]],
    reference: StrataReference,
    predicate: Callable[[dict[str, Any], StrataReference], bool] = lambda row, ref: True,
) -> dict[str, Any]:
    selected = [row for row in rows if _matched_eligible(row) and predicate(row, reference)]
    denominator = len(selected)

    def crop_block(prefix: str) -> dict[str, Any]:
        top1_key, top5_key = f"{prefix}_top1", f"{prefix}_top5"
        recognized = sum(row.get(top1_key) is not None for row in selected)
        top1_correct = sum(int(row.get(top1_key) or 0) for row in selected)
        top5_correct = sum(int(row.get(top5_key) or 0) for row in selected)
        return {
            "recognized_targets": recognized,
            "recognition_coverage": recognized / denominator if denominator else None,
            "top1_correct": top1_correct,
            "top1": top1_correct / denominator if denominator else None,
            "top5_correct": top5_correct,
            "top5": top5_correct / denominator if denominator else None,
        }

    predicted = crop_block("predicted_crop")
    gold = crop_block("gold_crop")
    return {
        "matched_in_vocabulary_targets": denominator,
        "predicted_crop": predicted,
        "gold_crop": gold,
        "predicted_minus_gold": {
            "top1": predicted["top1"] - gold["top1"] if denominator else None,
            "top5": predicted["top5"] - gold["top5"] if denominator else None,
        },
    }


def stratified_cell_report(
    rows: dict[str, dict[str, Any]], reference: StrataReference, slices: Sequence[SliceSpec]
) -> dict[str, Any]:
    materialized = list(rows.values())
    report: dict[str, Any] = {}
    for item in slices:
        block = matched_metric_block(materialized, reference, item.predicate)
        if item.family == "overall":
            report["overall"] = block
        else:
            report.setdefault(item.family, {})[item.name] = block
    return report


def _validate_contrasts(
    contrasts: Sequence[tuple[str, str]] | None, arms: Sequence[str]
) -> list[tuple[str, str]]:
    if contrasts is None:
        return [(arms[right], arms[left]) for left in range(len(arms)) for right in range(left + 1, len(arms))]
    result: list[tuple[str, str]] = []
    seen: set[frozenset[str]] = set()
    for treatment, control in contrasts:
        if treatment == control:
            raise ValueError("a contrast must contain two different arms")
        if treatment not in arms or control not in arms:
            raise ValueError(f"contrast names unknown arm(s): {(treatment, control)}")
        unordered = frozenset((treatment, control))
        if unordered in seen:
            raise ValueError(f"duplicate or reversed contrast: {(treatment, control)}")
        seen.add(unordered)
        result.append((treatment, control))
    return result


def synchronized_plate_weights(
    plates: Sequence[str], replicates: int, seed: int
) -> tuple[np.ndarray, str]:
    if not plates:
        raise ValueError("cannot bootstrap an empty plate universe")
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    generator = np.random.default_rng(seed)
    probabilities = np.full(len(plates), 1.0 / len(plates), dtype=np.float64)
    weights = generator.multinomial(len(plates), probabilities, size=replicates).astype(np.int32)
    digest = hashlib.sha256()
    digest.update(json.dumps(list(plates), ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    digest.update(weights.tobytes(order="C"))
    return weights, digest.hexdigest()


def _interval(values: np.ndarray) -> tuple[float | None, float | None]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None, None
    quantiles = np.quantile(finite, [0.025, 0.975], method="linear")
    return float(quantiles[0]), float(quantiles[1])


def _bootstrap_block(point: float | None, values: np.ndarray, targets: int) -> dict[str, Any]:
    finite = values[np.isfinite(values)]
    low, high = _interval(finite)
    return {
        "matched_in_vocabulary_targets": int(targets),
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


def _nest_slice_blocks(
    slices: Sequence[SliceSpec], blocks: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    nested: dict[str, Any] = {}
    for item, block in zip(slices, blocks):
        if item.family == "overall":
            nested["overall"] = block
        else:
            nested.setdefault(item.family, {})[item.name] = block
    return nested


def paired_bootstrap(
    *,
    cells: dict[tuple[str, int, str], dict[str, dict[str, Any]]],
    arms: Sequence[str],
    seeds: Sequence[int],
    detectors: Sequence[str],
    contrasts: Sequence[tuple[str, str]],
    reference: StrataReference,
    slices: Sequence[SliceSpec],
    replicates: int,
    bootstrap_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    first_rows = next(iter(cells.values()))
    plates = sorted({str(row["plate"]) for row in first_rows.values()})
    plate_index = {plate: index for index, plate in enumerate(plates)}
    weights, weights_sha256 = synchronized_plate_weights(plates, replicates, bootstrap_seed)
    weights_float = weights.astype(np.float64, copy=False)
    metric_count = len(METRICS)
    comparisons: list[dict[str, Any]] = []

    for detector in detectors:
        reference_rows = cells[(arms[0], seeds[0], detector)]
        denominators = np.zeros((len(plates), len(slices)), dtype=np.float64)
        for row in reference_rows.values():
            if not _matched_eligible(row):
                continue
            plate = plate_index[str(row["plate"])]
            for slice_index, item in enumerate(slices):
                if item.predicate(row, reference):
                    denominators[plate, slice_index] += 1.0
        boot_denominators = weights_float @ denominators
        point_denominators = denominators.sum(axis=0)

        point_rates: dict[tuple[str, int], np.ndarray] = {}
        bootstrap_rates: dict[tuple[str, int], np.ndarray] = {}
        for arm in arms:
            for seed in seeds:
                numerators = np.zeros((len(plates), len(slices), metric_count), dtype=np.float64)
                for row in cells[(arm, seed, detector)].values():
                    if not _matched_eligible(row):
                        continue
                    plate = plate_index[str(row["plate"])]
                    for slice_index, item in enumerate(slices):
                        if not item.predicate(row, reference):
                            continue
                        for metric_index, metric in enumerate(METRICS):
                            numerators[plate, slice_index, metric_index] += int(row.get(metric) or 0)
                point_numerators = numerators.sum(axis=0)
                with np.errstate(divide="ignore", invalid="ignore"):
                    point_rates[(arm, seed)] = np.divide(
                        point_numerators,
                        point_denominators[:, None],
                        out=np.full_like(point_numerators, np.nan),
                        where=point_denominators[:, None] > 0,
                    )
                boot_numerators = (weights_float @ numerators.reshape(len(plates), -1)).reshape(
                    replicates, len(slices), metric_count
                )
                with np.errstate(divide="ignore", invalid="ignore"):
                    bootstrap_rates[(arm, seed)] = np.divide(
                        boot_numerators,
                        boot_denominators[:, :, None],
                        out=np.full_like(boot_numerators, np.nan),
                        where=boot_denominators[:, :, None] > 0,
                    )

        for treatment, control in contrasts:
            slice_blocks: list[dict[str, Any]] = []
            for slice_index, _ in enumerate(slices):
                metric_blocks: dict[str, Any] = {}
                for metric_index, metric in enumerate(METRICS):
                    seed_points = [
                        point_rates[(treatment, seed)][slice_index, metric_index]
                        - point_rates[(control, seed)][slice_index, metric_index]
                        for seed in seeds
                    ]
                    finite_points = [float(value) for value in seed_points if math.isfinite(float(value))]
                    point = sum(finite_points) / len(finite_points) if len(finite_points) == len(seeds) else None
                    distributions = np.stack(
                        [
                            bootstrap_rates[(treatment, seed)][:, slice_index, metric_index]
                            - bootstrap_rates[(control, seed)][:, slice_index, metric_index]
                            for seed in seeds
                        ],
                        axis=1,
                    )
                    distribution = np.mean(distributions, axis=1)
                    metric_blocks[metric] = _bootstrap_block(
                        point, distribution, int(point_denominators[slice_index])
                    )
                slice_blocks.append(metric_blocks)
            comparisons.append(
                {
                    "eval_detector": detector,
                    "treatment_arm": treatment,
                    "control_arm": control,
                    "difference_definition": "treatment minus control; mean of paired per-seed rates",
                    "recognition_seeds": [int(seed) for seed in seeds],
                    "matched_conditional": _nest_slice_blocks(slices, slice_blocks),
                }
            )

    metadata = {
        "replicates": int(replicates),
        "seed": int(bootstrap_seed),
        "resampling_unit": "plate",
        "plate_count": len(plates),
        "plate_order": plates,
        "synchronization": (
            "one multinomial plate-weight matrix is reused for every detector, arm contrast, "
            "recognition seed, modality, and frozen stratum"
        ),
        "synchronized_weights_sha256": weights_sha256,
        "aggregation": "within-seed matched-target rate difference, then arithmetic mean over seeds",
    }
    return comparisons, metadata


def build_analysis(
    *,
    strata_reference: str | Path,
    specs: Sequence[CellSpec],
    contrasts: Sequence[tuple[str, str]] | None = None,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 20261009,
) -> dict[str, Any]:
    reference = load_strata_reference(strata_reference)
    cells, arms, seeds, detectors = load_matrix(specs, reference)
    selected_contrasts = _validate_contrasts(contrasts, arms)
    slices = _slice_specs(reference)

    cell_reports: list[dict[str, Any]] = []
    spec_by_key = {spec.key: spec for spec in specs}
    for arm in arms:
        for seed in seeds:
            for detector in detectors:
                key = arm, seed, detector
                path = spec_by_key[key].rows_path
                cell_reports.append(
                    {
                        "arm": arm,
                        "recognition_seed": int(seed),
                        "eval_detector": detector,
                        "rows_path": str(path),
                        "rows_sha256": file_sha256(path),
                        "matched_conditional": stratified_cell_report(cells[key], reference, slices),
                    }
                )

    comparisons, bootstrap_metadata = paired_bootstrap(
        cells=cells,
        arms=arms,
        seeds=seeds,
        detectors=detectors,
        contrasts=selected_contrasts,
        reference=reference,
        slices=slices,
        replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    first_rows = next(iter(cells.values()))
    return {
        "schema_version": 1,
        "scope": "matched-target conditional crop-recognition analysis",
        "metric_definition": (
            "denominator is every IoU-matched in-vocabulary target in the named stratum; "
            "a missing crop-recognition result contributes zero correct predictions"
        ),
        "inputs": {
            "strata_reference": str(reference.path),
            "strata_reference_sha256": reference.sha256,
            "matrix_cells": [
                {
                    "arm": spec.arm,
                    "recognition_seed": int(spec.recognition_seed),
                    "eval_detector": spec.eval_detector,
                    "rows_path": str(spec.rows_path),
                    "rows_sha256": file_sha256(spec.rows_path),
                }
                for spec in specs
            ],
        },
        "matrix": {
            "arms": arms,
            "recognition_seeds": [int(seed) for seed in seeds],
            "eval_detectors": detectors,
            "complete_cartesian_product": True,
            "target_count_per_cell": len(first_rows),
            "plate_count_per_cell": len({row["plate"] for row in first_rows.values()}),
            "target_coverage_identical": True,
        },
        "strata": {
            "modality": list(MODALITIES),
            "matching_iou": [item.name for item in reference.iou_bins],
            "normalized_size_tertile": {
                "small": f"area_fraction < {reference.size_q33}",
                "medium": f"{reference.size_q33} <= area_fraction < {reference.size_q67}",
                "large": f"area_fraction >= {reference.size_q67}",
            },
            "frequency": [item.name for item in reference.frequency_bins],
        },
        "cell_reports": cell_reports,
        "pairwise_arm_differences": comparisons,
        "bootstrap": bootstrap_metadata,
    }


def _parse_specs(raw_cells: Sequence[Sequence[str]]) -> list[CellSpec]:
    specs: list[CellSpec] = []
    for arm, seed_text, detector, path_text in raw_cells:
        arm, detector = arm.strip(), detector.strip()
        if not arm or not detector:
            raise ValueError("arm and evaluation detector names must be non-empty")
        try:
            seed = int(seed_text)
        except ValueError as error:
            raise ValueError(f"recognition seed must be an integer: {seed_text!r}") from error
        specs.append(CellSpec(arm, seed, detector, _resolve_path(path_text)))
    return specs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze explicit crop-shift per-target matrix cells with frozen strata"
    )
    parser.add_argument("--strata-reference", required=True)
    parser.add_argument(
        "--cell",
        nargs=4,
        action="append",
        required=True,
        metavar=("ARM", "RECOGNITION_SEED", "EVAL_DETECTOR", "ROWS_JSONL"),
        help="repeat once for every Cartesian matrix cell",
    )
    parser.add_argument(
        "--contrast",
        nargs=2,
        action="append",
        metavar=("TREATMENT_ARM", "CONTROL_ARM"),
        help="repeat for desired treatment-minus-control contrasts; default: every arm pair",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261009)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = _resolve_path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing analysis: {output}")
    report = build_analysis(
        strata_reference=args.strata_reference,
        specs=_parse_specs(args.cell),
        contrasts=[(str(left), str(right)) for left, right in args.contrast] if args.contrast else None,
        bootstrap_replicates=int(args.bootstrap_replicates),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    atomic_json_dump(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
