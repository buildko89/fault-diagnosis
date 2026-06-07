"""Matplotlib figure builders for the GUI.

The topology / ΔV figures are reused from ``fault.reporter`` (single source of
truth for the styling; same figures the file-based report writes). This module
adds GUI-only figures (e.g. the evaluation sweep) and depends only on
matplotlib (not Streamlit), so it stays unit-testable on its own.
"""
from typing import Sequence

import matplotlib
matplotlib.use("Agg")  # Headless; safe under Streamlit's server process.
import matplotlib.pyplot as plt

# Re-exported so the GUI imports figures from one place.
from fault.reporter import topology_figure, delta_v_figure  # noqa: F401


def accuracy_sweep_figure(xs: Sequence[float], ys: Sequence[float],
                          xlabel: str, ylabel: str = "Top-1 accuracy [%]"):
    """Line plot of an accuracy metric swept over a parameter (tol or noise)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys, marker="o", color="steelblue")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(-2, 102)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.set_title(f"{ylabel} vs {xlabel}")
    fig.tight_layout()
    return fig
