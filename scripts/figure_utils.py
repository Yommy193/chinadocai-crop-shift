"""Small, portable figure helpers used by the CSIG aggregation script.

The original workstation used SciPilot for styling and visual QA. This module
keeps the repository script runnable without embedding a workstation path.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt

def setup_style(*, journal: str = "general", lang: str = "en", use_sciplots: bool = False, constrained_layout: bool = False) -> None:
    """Set a conservative, publication-friendly matplotlib baseline."""
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "figure.constrained_layout.use": constrained_layout,
    })

def audit_layout(fig) -> list[str]:
    """Return basic layout warnings; an empty list means no obvious warning."""
    warnings: list[str] = []
    for ax in fig.axes:
        if not ax.get_xlabel() and not ax.get_ylabel() and not ax.get_title():
            warnings.append("axis has no labels or title")
    return warnings

def print_report(issues: Iterable[str]) -> None:
    issues = list(issues)
    print("[visual_qa] " + ("ok" if not issues else "; ".join(issues)))

def render_preview(fig, path: str, *, dpi: int = 150) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")

def export_figure(fig, *, basename: str, formats: tuple[str, ...] = ("png",), size_inches: tuple[float, float] | None = None, dpi: int = 600, grayscale_preview: bool = False) -> None:
    if size_inches is not None:
        fig.set_size_inches(*size_inches)
    base = Path(basename)
    base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(base.with_suffix("." + fmt), dpi=dpi, bbox_inches="tight")
