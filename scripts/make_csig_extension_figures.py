"""Publication figures for the additive three-detector CSIG supplement.

The script profiles the available validation summaries, keeps detector recall
separate from conditional recognition, and exports heatmaps plus dot-and-
whisker plots through ``scipilot-figure-skill``.  It only reads extension
artifacts and writes new figure files; historical experiment directories are
never modified.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from figure_utils import audit_layout, export_figure, print_report, render_preview, setup_style

ROOT = Path(__file__).resolve().parents[1]


DET_ORDER = ("anchor", "frcnn", "fcos")
DET_LABELS = {"anchor": "Anchor", "frcnn": "Faster R-CNN", "fcos": "FCOS"}
ARM_ORDER = ("fcos_pred_step", "fcos_mix075_step")
ARM_LABELS = {"fcos_pred_step": "Pred-step", "fcos_mix075_step": "Mix .75-step"}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _detector_key(label: str) -> str | None:
    value = label.lower()
    if "anchor" in value or "centernet" in value:
        return "anchor"
    if "frcnn" in value or "faster" in value:
        return "frcnn"
    if "fcos" in value:
        return "fcos"
    return None


def _checkpoint_seed(checkpoint: Path) -> int | None:
    metrics = checkpoint.parent / "cnn_metrics.json"
    if metrics.exists():
        try:
            return int(_read_json(metrics)["training"]["seed"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass
    match = re.search(r"seed(\d+)", checkpoint.parent.name)
    return int(match.group(1)) if match else None


def _is_cached_extension_checkpoint(checkpoint: Path) -> bool:
    """Keep the multi-seed table on one optimizer/step-budget configuration.

    The first pilot FCOS runs used the historical manifest trainer.  The final
    extension uses the memory-mapped cached trainer (batch 1024, 133 steps per
    epoch).  Mixing those checkpoints would make a seed mean depend on the
    optimizer step budget, so discovery intentionally filters to the cached
    extension metadata while leaving the pilot files untouched on disk.
    """
    metrics_path = checkpoint.parent / "cnn_metrics.json"
    if not metrics_path.exists():
        return False
    try:
        payload = _read_json(metrics_path)
        training = payload.get("training") or {}
        return training.get("experiment") == "crop_shift_alignment_cached"
    except (OSError, json.JSONDecodeError):
        return False


def discover_records(root: Path) -> list[dict[str, Any]]:
    records: dict[tuple[str, int, str], dict[str, Any]] = {}
    for summary_path in (root / "results").glob("*_summary.json"):
        try:
            summary = _read_json(summary_path)
        except (OSError, json.JSONDecodeError):
            continue
        inputs = summary.get("inputs") or {}
        checkpoint_value = inputs.get("vision_checkpoint")
        if not checkpoint_value:
            continue
        checkpoint = Path(str(checkpoint_value))
        if not _is_cached_extension_checkpoint(checkpoint):
            continue
        name = str(checkpoint).replace("\\", "/")
        arm = next((candidate for candidate in ARM_ORDER if candidate in name), None)
        detector = _detector_key(str(summary.get("detector_label", "")))
        seed = _checkpoint_seed(checkpoint)
        if arm is None or detector is None or seed is None:
            continue
        rows_value = (summary.get("outputs") or {}).get("target_rows_jsonl")
        rows_path = Path(str(rows_value)) if rows_value else root / "results" / f"{summary_path.stem.replace('_summary', '_targets')}.jsonl"
        if not rows_path.is_absolute():
            rows_path = (ROOT / rows_path).resolve()
        metric = (summary.get("metrics") or {}).get("overall") or {}
        record = {
            "summary_path": summary_path,
            "rows_path": rows_path,
            "arm": arm,
            "detector": detector,
            "seed": seed,
            "metrics": metric,
            "summary": summary,
        }
        key = (arm, seed, detector)
        # Prefer the standardized seed-specific file if duplicate legacy and
        # new aliases point to the same checkpoint.
        if key not in records or "_seed" in summary_path.stem:
            records[key] = record
    return [records[key] for key in sorted(records)]


def _metric(record: dict[str, Any], name: str) -> float:
    value = record["metrics"].get(name)
    return float(value) if value is not None else float("nan")


def _save(fig, out: Path, stem: str, size: tuple[float, float]) -> None:
    issues = audit_layout(fig)
    print(f"[visual_qa] {stem}")
    print_report(issues)
    render_preview(fig, str(out / f"_{stem}_preview.png"), dpi=150)
    export_figure(
        fig,
        basename=str(out / stem),
        formats=("pdf", "svg", "png"),
        size_inches=size,
        dpi=600,
        grayscale_preview=True,
    )
    plt.close(fig)


def _heatmap(records: list[dict[str, Any]], out: Path) -> None:
    lookup: dict[tuple[str, str], list[float]] = defaultdict(list)
    for record in records:
        lookup[(record["arm"], record["detector"])].append(_metric(record, "predicted_crop_top1_conditioned_on_evaluated_match"))
    lookup5: dict[tuple[str, str], list[float]] = defaultdict(list)
    for record in records:
        lookup5[(record["arm"], record["detector"])].append(_metric(record, "predicted_crop_top5_conditioned_on_evaluated_match"))
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.85), constrained_layout=False)
    for ax, values, title, cmap in (
        (axes[0], lookup, "匹配条件 Top-1 (%)", "YlGnBu"),
        (axes[1], lookup5, "匹配条件 Top-5 (%)", "YlOrBr"),
    ):
        matrix = np.full((len(ARM_ORDER), len(DET_ORDER)), np.nan)
        for i, arm in enumerate(ARM_ORDER):
            for j, detector in enumerate(DET_ORDER):
                vals = [v for v in values.get((arm, detector), []) if np.isfinite(v)]
                if vals:
                    matrix[i, j] = np.mean(vals) * 100
        finite = matrix[np.isfinite(matrix)]
        if finite.size:
            lo = max(0, np.floor(finite.min() - 0.5))
            hi = np.ceil(finite.max() + 0.5)
        else:
            lo, hi = 0, 100
        im = ax.imshow(matrix, cmap=cmap, vmin=lo, vmax=hi, aspect="auto")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if np.isfinite(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8,
                            color="black" if matrix[i, j] < (lo + hi) / 2 else "white")
        ax.set_xticks(range(len(DET_ORDER)), ["Anchor", "FRCNN", "FCOS"])
        ax.set_yticks(range(len(ARM_ORDER)), [ARM_LABELS[a] for a in ARM_ORDER])
        ax.set_xlabel("Evaluator")
        ax.set_title(title, fontsize=9)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=7)
    axes[0].set_ylabel("Training crop source")
    fig.suptitle("Three-detector cross-evaluation matrix", fontsize=11)
    fig.subplots_adjust(left=0.18, right=0.96, bottom=0.25, top=0.80, wspace=0.55)
    _save(fig, out, "fig_detector_matrix_3det", (6.7, 2.85))


def _plate_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    eligible = [r for r in rows if r.get("evaluation_eligible")]
    matched = [r for r in eligible if r.get("matched_at_iou")]
    def _mean(key: str, selected: list[dict[str, Any]]) -> float:
        values = [int(r[key]) for r in selected if r.get(key) is not None]
        return float(np.mean(values)) if values else float("nan")
    return {
        "recall": len(matched) / len(eligible) if eligible else float("nan"),
        "top1": _mean("predicted_crop_top1", matched),
        "top5": _mean("predicted_crop_top5", matched),
        "recovery1": _mean("predicted_crop_top1", eligible),
        "recovery5": _mean("predicted_crop_top5", eligible),
    }


def _load_plate_metrics(record: dict[str, Any]) -> dict[str, dict[str, float]]:
    if not record["rows_path"].exists():
        return {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with record["rows_path"].open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                grouped[str(row["plate"])].append(row)
    return {plate: _plate_metrics(rows) for plate, rows in grouped.items()}


def _bootstrap_difference(records: list[dict[str, Any]], detector: str, metric: str, seed: int = 20261022, draws: int = 2000) -> tuple[float, float, float, int]:
    by_arm_seed = {(r["arm"], r["seed"]): r for r in records if r["detector"] == detector}
    common = sorted({s for a, s in by_arm_seed if a == "fcos_pred_step"} & {s for a, s in by_arm_seed if a == "fcos_mix075_step"})
    paired: list[tuple[dict[str, float], dict[str, float]]] = []
    for s in common:
        pred = _load_plate_metrics(by_arm_seed[("fcos_pred_step", s)])
        mix = _load_plate_metrics(by_arm_seed[("fcos_mix075_step", s)])
        plates = sorted(set(pred) & set(mix))
        if plates:
            paired.append(({p: pred[p][metric] for p in plates}, {p: mix[p][metric] for p in plates}))
    if not paired:
        return float("nan"), float("nan"), float("nan"), 0
    points = []
    for p, m in paired:
        points.append(float(np.nanmean(list(m.values())) - np.nanmean(list(p.values()))))
    point = float(np.nanmean(points))
    rng = np.random.default_rng(seed)
    boot = np.empty(draws, dtype=float)
    for b in range(draws):
        diffs = []
        for pred, mix in paired:
            plates = list(pred)
            idx = rng.integers(0, len(plates), size=len(plates))
            pv = np.asarray([pred[plates[i]] for i in idx], dtype=float)
            mv = np.asarray([mix[plates[i]] for i in idx], dtype=float)
            diffs.append(float(np.nanmean(mv) - np.nanmean(pv)))
        boot[b] = float(np.nanmean(diffs))
    return point, float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5)), len(paired)


def _effect_plot(records: list[dict[str, Any]], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 3.15), sharey=True, constrained_layout=False)
    for ax, metric, title, color in (
        (axes[0], "top1", "Mix .75 - Pred: Top-1", "#0072B2"),
        (axes[1], "top5", "Mix .75 - Pred: Top-5", "#D55E00"),
    ):
        for i, detector in enumerate(DET_ORDER):
            point, lo, hi, n = _bootstrap_difference(records, detector, metric)
            if not np.isfinite(point):
                continue
            ax.errorbar(point * 100, i, xerr=np.array([[point - lo], [hi - point]]) * 100,
                        fmt="o", color=color, ecolor=color, capsize=3, markersize=5, lw=1.5)
            ax.text(hi * 100 + 0.02, i, f"n={n}", va="center", fontsize=7)
        ax.axvline(0, color="#666666", lw=0.9, ls="--")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Difference (percentage points)")
        ax.grid(axis="x", color="#dddddd", lw=0.6)
    axes[0].set_yticks(range(len(DET_ORDER)), ["Anchor", "FRCNN", "FCOS"])
    axes[0].set_ylabel("Evaluator")
    fig.suptitle("Paired plate-level effects across detectors", fontsize=11)
    fig.subplots_adjust(left=0.15, right=0.97, bottom=0.20, top=0.82, wspace=0.16)
    _save(fig, out, "fig_effect_intervals_3det", (6.7, 3.15))


def _summary_stat(values: list[float]) -> tuple[float, float, float]:
    values = [v for v in values if np.isfinite(v)]
    if not values:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, mean, mean
    rng = np.random.default_rng(20261022 + len(values))
    boot = np.asarray([np.mean(rng.choice(values, size=len(values), replace=True)) for _ in range(2000)])
    return mean, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def _extension_plot(records: list[dict[str, Any]], out: Path) -> None:
    mix = [r for r in records if r["arm"] == "fcos_mix075_step"]
    metrics = (
        ("matched_target_recall", "Matched recall (%)"),
        ("predicted_crop_top1_conditioned_on_evaluated_match", "Matched Top-1 (%)"),
        ("detector_to_recognition_top1_recovery", "End-to-end Top-1 (%)"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(6.7, 2.8), sharey=False, constrained_layout=False)
    for ax, (metric, title) in zip(axes, metrics):
        for i, detector in enumerate(DET_ORDER):
            vals = [_metric(r, metric) * 100 for r in mix if r["detector"] == detector]
            vals = [v for v in vals if np.isfinite(v)]
            if not vals:
                continue
            mean, lo, hi = _summary_stat(vals)
            x = i
            jitter = np.linspace(-0.10, 0.10, len(vals)) if len(vals) > 1 else np.array([0.0])
            ax.scatter(np.full(len(vals), x) + jitter, vals, s=20, color="#999999", zorder=2)
            ax.errorbar(x, mean, yerr=[[mean - lo], [hi - mean]], fmt="o", color="#0072B2", capsize=3, zorder=3)
        ax.set_xticks(range(len(DET_ORDER)), ["Anchor", "FRCNN", "FCOS"])
        ax.set_title(title, fontsize=8.5)
        ax.grid(axis="y", color="#dddddd", lw=0.6)
        ax.set_xlim(-0.45, len(DET_ORDER) - 0.55)
    axes[0].set_ylabel("Percentage")
    fig.suptitle("FCOS Mix .75-step recognizer across three detectors", fontsize=10.5)
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.20, top=0.80, wspace=0.34)
    _save(fig, out, "fig_fcos_extension_3det", (6.7, 2.8))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="artifacts/experiments/crop_shift_alignment_extension_20261021")
    parser.add_argument("--out-dir", default="artifacts/paper/chinadocai2026_crop_shift/figures")
    args = parser.parse_args()
    setup_style(journal="general", lang="zh", use_sciplots=True, constrained_layout=False)
    # Noto Sans SC is a variable font on this host and emits a harmless
    # weight-resolution warning; YaHei avoids that warning while retaining
    # complete CJK coverage for any remaining Chinese annotations.
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "Arial", "DejaVu Sans"]
    root = (ROOT / args.root).resolve()
    out = (ROOT / args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    records = discover_records(root)
    (root / "results" / "figure_records.json").write_text(json.dumps([
        {k: str(v) if isinstance(v, Path) else v for k, v in r.items() if k not in {"summary", "metrics"}}
        for r in records
    ], ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"records": len(records), "keys": [[r["arm"], r["seed"], r["detector"]] for r in records]}, ensure_ascii=False))
    if records:
        _heatmap(records, out)
        _effect_plot(records, out)
        _extension_plot(records, out)


if __name__ == "__main__":
    main()
