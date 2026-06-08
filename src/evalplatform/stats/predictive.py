"""Phase 4 — Predictive validity (the killer question).   *** YOU HAND-CODE THIS. ***

"Does +0.1 on my offline metric mean anything real?" Correlate offline judge scores against
a proxy for real-world success (thumbs-up, task completion — synthetic is fine to show the
method) and report the correlation *with a confidence interval*.

Verify the point correlation against ``scipy.stats.pearsonr``; the CI reuses your own
``stats.bootstrap.bootstrap_ci`` (so implement that first).
"""

from __future__ import annotations

from typing import Sequence

from ._types import CI


def predictive_validity(
    offline_scores: Sequence[float],
    online_outcomes: Sequence[float],
    *,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> CI:
    """Correlation between offline scores and online outcomes, with a bootstrap CI.

    Args:
        offline_scores: the eval metric per item/system.
        online_outcomes: the real-world proxy per matching item/system.
        n_resamples, seed: forwarded to your bootstrap.

    Returns:
        CI where ``point`` is the correlation (e.g. Pearson r) and ``low``/``high`` bound it.

    Verify ``point`` against ``scipy.stats.pearsonr(offline_scores, online_outcomes)[0]``.

    Hint: compute the correlation on the full sample for ``point``; for the interval,
        bootstrap over *paired* (offline, online) observations — resample item indices with
        replacement and recompute the correlation each time — then take percentiles. Reuse
        ``stats.bootstrap.bootstrap_ci`` by bootstrapping over paired indices.
    """
    raise NotImplementedError(
        "Hand-code predictive validity (Phase 4), then run handcode/verify_stats.py."
    )
