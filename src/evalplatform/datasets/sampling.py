"""Phase 3 — Stratified sampling.   *** YOU HAND-CODE THIS. ***

To build an evalset that mirrors the real query distribution, sample items so each stratum
(category/segment, see ``EvalItem.stratum``) is represented in proportion to the population
— not just a uniform random draw. The *coverage measurement* that proves it worked lives in
``stats/diversity.py``; this is the sampler.
"""

from __future__ import annotations

from typing import Optional, Sequence

from ..models import EvalItem


def stratified_sample(
    items: Sequence[EvalItem],
    n: int,
    *,
    by: str = "stratum",
    seed: int = 0,
) -> list[EvalItem]:
    """Draw ``n`` items, preserving the population's per-stratum proportions.

    Args:
        items: the population to sample from.
        n: target sample size.
        by: the :class:`EvalItem` attribute defining strata (default ``"stratum"``).
        seed: RNG seed.

    Returns:
        a list of ``~n`` sampled items whose stratum distribution approximates the population's.

    Hint: bucket items by the stratum key; allocate the per-stratum quota proportional to each
        bucket's share of the population (decide how to round so the quotas sum to n); then
        sample without replacement within each bucket using ``random.Random(seed)``.
    """
    raise NotImplementedError(
        "Hand-code stratified sampling (Phase 3). See the docstring; coverage check is in stats/diversity.py."
    )
