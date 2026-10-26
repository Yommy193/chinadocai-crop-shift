"""Run the frozen target-strata analysis from a target-matrix manifest."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _root_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a validation-only crop-shift target matrix")
    parser.add_argument("--root", default="artifacts/experiments/crop_shift_alignment_20261009")
    parser.add_argument("--matrix-manifest", default="target_matrix_cells.json")
    parser.add_argument(
        "--strata-reference",
        default="strata_reference_frozen.json",
        help="frozen frequency/geometry strata reference (pair-only strata_reference.json is audit-only)",
    )
    parser.add_argument("--output", default="target_strata_analysis.json")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261009)
    parser.add_argument("--contrast", nargs=2, action="append", metavar=("TREATMENT", "CONTROL"))
    args = parser.parse_args()
    experiment = _root_path(args.root)
    manifest = _root_path(args.matrix_manifest) if Path(args.matrix_manifest).is_absolute() else experiment / args.matrix_manifest
    strata = _root_path(args.strata_reference) if Path(args.strata_reference).is_absolute() else experiment / args.strata_reference
    output = _root_path(args.output) if Path(args.output).is_absolute() else experiment / args.output
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing analysis: {output}")
    cells = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(cells, list) or not cells:
        raise ValueError("matrix manifest must contain a non-empty list")
    command = [
        sys.executable,
        "scripts/analyze_crop_shift_targets.py",
        "--strata-reference",
        str(strata),
        "--bootstrap-replicates",
        str(int(args.bootstrap_replicates)),
        "--bootstrap-seed",
        str(int(args.bootstrap_seed)),
        "--output",
        str(output),
    ]
    for cell in cells:
        required = ("arm", "recognition_seed", "eval_detector", "rows")
        if not all(key in cell for key in required):
            raise ValueError(f"matrix cell lacks required fields: {cell}")
        command.extend(
            [
                "--cell",
                str(cell["arm"]),
                str(int(cell["recognition_seed"])),
                str(cell["eval_detector"]),
                str(cell["rows"]),
            ]
        )
    for treatment, control in args.contrast or []:
        command.extend(["--contrast", str(treatment), str(control)])
    log = experiment / "logs" / "target_strata_analysis.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8", newline="\n") as handle:
        completed = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if completed.returncode:
        raise RuntimeError(f"target-strata analysis failed; see {log}")
    if not output.is_file():
        raise RuntimeError(f"analysis exited successfully but output is missing: {output}")
    print(json.dumps({"output": str(output), "log": str(log), "cells": len(cells)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
