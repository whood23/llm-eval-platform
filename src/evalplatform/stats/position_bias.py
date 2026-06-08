"""Phase 2b — Position-bias quantification.   *** YOU HAND-CODE THIS. ***

"Large Language Models are not Fair Evaluators": pairwise judges often prefer whichever
candidate is shown first. You presented every pair in BOTH orders (the swap logic in
``judge/pairwise.py``); these functions turn those swapped runs into a bias measurement
and a bias-corrected preference. Verify against the synthetic fixtures in
handcode/verify_stats.py (constructed with a known flip rate).
"""

from __future__ import annotations

from typing import Sequence

from ..models import PairwiseJudgment


def position_flip_rate(judgments: Sequence[PairwiseJudgment]) -> float:
    """Fraction of candidate pairs whose winner *changes* when only the order is swapped.

    A pair is "flipped" if the judge picks a different underlying candidate in the AB vs
    BA presentation (ties handled per your design choice — document it). 0.0 = perfectly
    order-invariant; higher = more position bias.

    Args:
        judgments: pairwise judgments containing both AB and BA presentations. Pair them up
            by (item_id, the unordered pair of candidate ids), using ``winner_candidate_id``.

    Returns:
        flip rate in [0, 1].

    Hint: group by (item_id, frozenset({candidate_a_id, candidate_b_id})); within each group
        find the AB row and the BA row; count the group as flipped if their
        ``winner_candidate_id`` disagree; divide by number of complete groups.
    """
    raise NotImplementedError(
        "Hand-code the position flip rate (Phase 2b), then run handcode/verify_stats.py."
    )


def consistent_winrate(judgments: Sequence[PairwiseJudgment]) -> dict[str, float]:
    """Bias-corrected win rate: a candidate wins a pair only if it wins in BOTH orders.

    Comparisons where the verdict flips with order are not credited as wins to either side
    (treat them as ties). Returns, per candidate id, the fraction of its complete pairings
    that it won consistently.

    Returns:
        mapping ``candidate_id -> consistent win rate in [0, 1]``.

    Hint: reuse the same (item_id, unordered-pair) grouping as ``position_flip_rate``; a
        candidate is credited a win in a group only if it is the ``winner_candidate_id`` in
        both the AB and BA rows.
    """
    raise NotImplementedError(
        "Hand-code the bias-corrected win rate (Phase 2b), then run handcode/verify_stats.py."
    )
