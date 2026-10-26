"""Three-seed common-target sensitivity analysis for the CSIG supplement."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8") as handle:
        return {str(row["target_id"]): row for row in (json.loads(line) for line in handle if line.strip())}


def seed_cells(root: Path, seed: int) -> dict[str, dict[str, dict]]:
    names = {
        "AA": (f"anchor_pred_step_seed{seed}", "anchor"),
        "AF": (f"anchor_pred_step_seed{seed}", "frcnn"),
        "FA": (f"frcnn_pred_step_seed{seed}", "anchor"),
        "FF": (f"frcnn_pred_step_seed{seed}", "frcnn"),
    }
    return {key: load(root / arm / detector / "targets.jsonl") for key, (arm, detector) in names.items()}


def interaction_bootstrap(
    all_seed_cells: dict[int, dict[str, dict[str, dict]]],
    all_seed_targets: dict[int, set[str]],
    metric: str,
    reps: int,
    seed: int,
) -> dict[str, float]:
    """Bootstrap the four-cell interaction at the validation-plate level.

    Each seed contributes a plate-level mean for each cell.  The resampled
    interaction is computed within every seed and then averaged, matching the
    aggregation used for the target-level point estimate.
    """
    plate_rates: dict[int, dict[str, dict[str, float]]] = {}
    for run_seed, cells in all_seed_cells.items():
        grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for cell, rows in cells.items():
            for target_id in all_seed_targets[run_seed]:
                row = rows[target_id]
                grouped[str(row["plate"])][cell].append(float(row[f"predicted_crop_{metric}"]))
        plate_rates[run_seed] = {
            plate: {cell: float(np.mean(values)) for cell, values in cell_values.items()}
            for plate, cell_values in grouped.items()
        }
    plates = sorted(set.intersection(*(set(values) for values in plate_rates.values())))
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(reps):
        chosen = rng.choice(plates, size=len(plates), replace=True)
        seed_interactions = []
        for run_seed in sorted(plate_rates):
            means = {
                cell: float(np.mean([plate_rates[run_seed][plate][cell] for plate in chosen]))
                for cell in ("AA", "AF", "FA", "FF")
            }
            seed_interactions.append((means["AA"] + means["FF"] - means["AF"] - means["FA"]) * 100.0)
        samples.append(float(np.mean(seed_interactions)))
    values = np.asarray(samples)
    return {
        "mean": float(values.mean()),
        "lo": float(np.quantile(values, .025)),
        "hi": float(np.quantile(values, .975)),
        "plates": len(plates),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-root", default="artifacts/experiments/crop_shift_alignment_20261009/confirmation_20261010/target_rows")
    parser.add_argument("--output", default="artifacts/experiments/crop_shift_alignment_extension_20261021/results/common_target_intersection_all_seeds.json")
    parser.add_argument("--seeds", default="20261009,20261010,20261011")
    parser.add_argument("--replicates", type=int, default=5000)
    args = parser.parse_args()
    root = (ROOT / args.target_root).resolve()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    per_seed = {}
    all_seed_cells: dict[int, dict[str, dict[str, dict]]] = {}
    all_seed_targets: dict[int, set[str]] = {}
    interaction_rows = {"top1": [], "top5": []}
    for seed in seeds:
        cells = seed_cells(root, seed)
        common = set.intersection(*(set(rows) for rows in cells.values()))
        matched = {t for t in common if all(bool(cells[key][t].get("evaluation_eligible")) and bool(cells[key][t].get("matched_at_iou")) for key in cells)}
        all_seed_cells[seed] = cells
        all_seed_targets[seed] = matched
        rates = {}
        for cell, rows in cells.items():
            rates[cell] = {metric: float(np.mean([int(rows[t][f"predicted_crop_{metric}"]) for t in matched])) for metric in ("top1", "top5")}
        for metric in ("top1", "top5"):
            interaction_rows[metric].append((rates["AA"][metric] + rates["FF"][metric] - rates["AF"][metric] - rates["FA"][metric]) * 100.0)
        per_seed[str(seed)] = {"common_rows": len(common), "all_cells_matched": len(matched), "plates": len({cells["AA"][t]["plate"] for t in matched}), "rates": rates, "interaction_pp": {metric: interaction_rows[metric][-1] for metric in ("top1", "top5")}}
    first_cells = seed_cells(root, seeds[0])
    first_common = set.intersection(*(set(rows) for rows in first_cells.values()))
    first_matched = {t for t in first_common if all(bool(first_cells[key][t].get("evaluation_eligible")) and bool(first_cells[key][t].get("matched_at_iou")) for key in first_cells)}
    payload = {
        "schema_version": 1,
        "scope": "additive all-seed common-target sensitivity analysis",
        "seeds": seeds,
        "per_seed": per_seed,
        "mean_interaction_pp": {metric: float(np.mean(values)) for metric, values in interaction_rows.items()},
        "plate_bootstrap_interaction_pp": {
            metric: interaction_bootstrap(all_seed_cells, all_seed_targets, metric, args.replicates, 20261021 + index)
            for index, metric in enumerate(("top1", "top5"))
        },
        "common_target_reference": {"rows": len(first_common), "matched": len(first_matched)},
        "bootstrap": {"replicates": args.replicates, "unit": "validation plate; seed means averaged"},
    }
    out = (ROOT / args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
