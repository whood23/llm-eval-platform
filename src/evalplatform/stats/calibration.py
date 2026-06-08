"""Phase 2d — Calibration.   *** YOU HAND-CODE THIS. ***

Does a higher judge score actually mean a higher chance of being correct? Bin the judge's
confidence/score, compare the average confidence in each bin to the empirical accuracy in
that bin (the reliability diagram), and summarize the gap as Expected Calibration Error.

Verify: ``reliability_curve`` against ``sklearn.calibration.calibration_curve``; ``ece``
against the independent weighted-gap reference in handcode/verify_stats.py.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

import numpy as np


class ReliabilityCurve(NamedTuple):
    """Per-bin reliability data, ready to plot as a reliability diagram."""

    bin_edges: np.ndarray  # length n_bins + 1
    bin_confidence: np.ndarray  # mean predicted confidence in each bin
    bin_accuracy: np.ndarray  # empirical accuracy in each bin
    bin_count: np.ndarray  # number of samples in each bin


def expected_calibration_error(
    confidences: Sequence[float],
    correct: Sequence[bool],
    *,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE).

    Args:
        confidences: predicted confidence / normalized judge score in [0, 1], one per item.
        correct: ground-truth correctness (1/True or 0/False) for each item.
        n_bins: number of equal-width bins over [0, 1].

    Returns:
        ECE in [0, 1]: the sample-weighted average of ``|accuracy - confidence|`` across bins.

    Hint: partition items into ``n_bins`` equal-width confidence bins; for each non-empty bin
        compute mean confidence and accuracy; ECE = sum over bins of
        ``(n_bin / N) * |acc_bin - conf_bin|``.
    """
    raise NotImplementedError(
        "Hand-code ECE (Phase 2d), then run handcode/verify_stats.py."
    )


def reliability_curve(
    confidences: Sequence[float],
    correct: Sequence[bool],
    *,
    n_bins: int = 10,
) -> ReliabilityCurve:
    """Per-bin confidence vs accuracy, for the reliability diagram.

    Returns a :class:`ReliabilityCurve`. Verify ``bin_confidence``/``bin_accuracy`` against
    ``sklearn.calibration.calibration_curve(correct, confidences, n_bins=n_bins)`` (note
    sklearn drops empty bins; align before comparing).

    Hint: same binning as ``expected_calibration_error``; here you return the per-bin arrays
        instead of collapsing them to a single number.
    """
    raise NotImplementedError(
        "Hand-code the reliability curve (Phase 2d), then run handcode/verify_stats.py."
    )
