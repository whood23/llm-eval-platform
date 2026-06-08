"""Matplotlib rendering helpers for eval reports (scaffolding, AI).

These functions only *render* values handed to them — they never compute any
statistic. The reliability diagram consumes a :class:`evalplatform.stats.ReliabilityCurve`;
the bar helpers consume already-computed points/CIs/biases. The ``Agg`` backend is
forced before ``pyplot`` is imported so plotting works headless (no display).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence, Union

import matplotlib

# Force a non-interactive backend BEFORE importing pyplot so this is headless-safe.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow backend selection)

PathLike = Union[str, Path]


def reliability_diagram(
    curve: Any,
    *,
    ax: "plt.Axes | None" = None,
    title: str = "Reliability diagram",
) -> "plt.Axes":
    """Plot a reliability diagram from a ``ReliabilityCurve`` (confidence vs accuracy).

    ``curve`` is an ``evalplatform.stats.ReliabilityCurve`` NamedTuple with
    ``bin_confidence`` / ``bin_accuracy`` arrays. We plot those points against the
    perfectly-calibrated diagonal. No statistics are computed here — the curve is
    produced by the hand-coded ``stats.reliability_curve``.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4.5))

    conf = list(curve.bin_confidence)
    acc = list(curve.bin_accuracy)

    # Perfect-calibration reference line y = x.
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="gray", label="perfect")
    ax.plot(conf, acc, marker="o", color="C0", label="judge")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("mean predicted confidence")
    ax.set_ylabel("empirical accuracy")
    ax.set_title(title)
    ax.legend(loc="best")
    return ax


def ci_bar(
    labels: Sequence[str],
    points: Sequence[float],
    lows: Sequence[float],
    highs: Sequence[float],
    *,
    ax: "plt.Axes | None" = None,
    title: str = "Point estimates with 95% CI",
) -> "plt.Axes":
    """Bar chart of point estimates with asymmetric error bars from CI bounds.

    ``points``/``lows``/``highs`` are already-computed CI fields (e.g. from the
    hand-coded ``stats.bootstrap_ci``); this only draws them.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(max(4.0, 0.8 * len(labels) + 2), 4.0))

    x = range(len(labels))
    pts = list(points)
    # Error bars are |point - bound|, clamped at 0 to avoid matplotlib negatives.
    lower_err = [max(0.0, p - lo) for p, lo in zip(pts, lows)]
    upper_err = [max(0.0, hi - p) for p, hi in zip(pts, highs)]

    ax.bar(list(x), pts, color="C0", alpha=0.8)
    ax.errorbar(
        list(x),
        pts,
        yerr=[lower_err, upper_err],
        fmt="none",
        ecolor="black",
        capsize=4,
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(list(labels), rotation=30, ha="right")
    ax.set_ylabel("estimate")
    ax.set_title(title)
    return ax


def bias_bar(
    labels: Sequence[str],
    values: Sequence[float],
    *,
    ax: "plt.Axes | None" = None,
    title: str = "Position bias",
) -> "plt.Axes":
    """Bar chart of bias values (e.g. flip rate / win-rate deltas) handed in pre-computed."""
    if ax is None:
        _, ax = plt.subplots(figsize=(max(4.0, 0.8 * len(labels) + 2), 4.0))

    x = range(len(labels))
    vals = list(values)
    # Color negative bars differently so sign reads at a glance.
    colors = ["C3" if v < 0 else "C0" for v in vals]
    ax.bar(list(x), vals, color=colors, alpha=0.8)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(list(labels), rotation=30, ha="right")
    ax.set_ylabel("value")
    ax.set_title(title)
    return ax


def savefig(fig: "plt.Figure", path: PathLike) -> Path:
    """Save ``fig`` to ``path`` (parent dirs created), close it, and return the Path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out
