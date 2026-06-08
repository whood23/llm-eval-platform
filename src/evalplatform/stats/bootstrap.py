"""Phase 2a — Bootstrap confidence intervals.   *** YOU HAND-CODE THIS. ***

The signal: "my eval has error bars." Every metric the platform reports should ship with
a CI, not a bare mean. Verify against ``scipy.stats.bootstrap`` (see handcode/verify_stats.py).
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from ._types import CI


def bootstrap_ci(
    data: Sequence[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    *,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> CI:
    """Percentile-bootstrap confidence interval for ``statistic`` over ``data``.

    Args:
        data: 1-D sample of observations (e.g. per-item judge scores).
        statistic: function mapping a 1-D array -> scalar (default: the mean).
        n_resamples: number of bootstrap resamples.
        confidence: e.g. 0.95 for a 95% interval.
        seed: RNG seed for reproducibility.

    Returns:
        CI(point, low, high) — ``point`` is ``statistic(data)`` on the original sample;
        ``low``/``high`` are the percentile-bootstrap interval bounds.

    Verify: ``scipy.stats.bootstrap((data,), statistic, method='percentile')`` should give
        matching bounds (within a small tolerance, same seed-independent sample).

    Hint (the recipe from the build guide — implement it yourself):
        resample WITH replacement ``n_resamples`` times, recompute the statistic each time,
        then take the ``alpha/2`` and ``1 - alpha/2`` percentiles of the resampled values.
        Use ``np.random.default_rng(seed)`` and ``rng.integers``/``rng.choice`` for resampling.
    """
    raise NotImplementedError(
        "Hand-code the percentile bootstrap (Phase 2a), then run handcode/verify_stats.py."
    )
