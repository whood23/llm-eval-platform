"""Phase 2c — Judge–human agreement (inter-rater reliability).   *** YOU HAND-CODE THIS. ***

You label a 50–100 item goldset yourself, then measure how well the judge agrees with you
(and how well two humans agree). Chance-corrected agreement is the right tool — raw percent
agreement is misleading when labels are imbalanced.

Verify: ``cohen_kappa`` against ``sklearn.metrics.cohen_kappa_score``; ``krippendorff_alpha``
against the known textbook value baked into handcode/verify_stats.py.
"""

from __future__ import annotations

from typing import Optional, Sequence


def cohen_kappa(labels_a: Sequence, labels_b: Sequence) -> float:
    """Cohen's kappa between two raters' categorical labels over the same items.

    Args:
        labels_a, labels_b: equal-length sequences of categorical labels (the i-th entries
            are the two raters' labels for item i).

    Returns:
        kappa in [-1, 1]: ``(p_o - p_e) / (1 - p_e)`` where ``p_o`` is observed agreement and
        ``p_e`` is agreement expected by chance from the marginal label frequencies.

    Verify against ``sklearn.metrics.cohen_kappa_score(labels_a, labels_b)``.

    Hint: build the confusion matrix of a-vs-b labels; ``p_o`` = trace / N; ``p_e`` from the
        outer product of the row/column marginals; then apply the formula above.
    """
    raise NotImplementedError(
        "Hand-code Cohen's kappa (Phase 2c), then run handcode/verify_stats.py."
    )


def krippendorff_alpha(
    reliability_data: Sequence[Sequence[Optional[float]]],
    *,
    level: str = "nominal",
) -> float:
    """Krippendorff's alpha — handles >2 raters and missing labels.

    Args:
        reliability_data: a raters x items matrix; ``reliability_data[r][i]`` is rater r's
            label for item i, or ``None`` if that rater did not label item i.
        level: measurement level for the distance function — start with ``"nominal"``
            (you may add ``"ordinal"``/``"interval"`` later).

    Returns:
        alpha in (-inf, 1]; 1 = perfect agreement, 0 = chance.

    Verify against the known reference value for the classic Krippendorff worked example in
        handcode/verify_stats.py (the ``krippendorff`` package may not be installed).

    Hint: alpha = ``1 - D_o / D_e`` where ``D_o`` is observed disagreement and ``D_e`` is
        expected disagreement, both built from coincidences within items using the chosen
        level's distance metric (for nominal: distance = 0 if equal else 1).
    """
    raise NotImplementedError(
        "Hand-code Krippendorff's alpha (Phase 2c), then run handcode/verify_stats.py."
    )
