"""Phase 3 — Evalset diversity / coverage metric.   *** YOU HAND-CODE THIS. ***

An evalset is only trustworthy if it covers the real query distribution. Given embeddings
of your eval items, quantify how spread out (diverse) they are, so you can show the set
isn't clustered in one corner of the space. (Stratified *sampling* lives in
``datasets/sampling.py``; this is the coverage *measurement*.)

Verify against ``scipy.spatial.distance.pdist`` in handcode/verify_stats.py.
"""

from __future__ import annotations

import numpy as np


def embedding_diversity(embeddings: np.ndarray) -> float:
    """A scalar diversity score for a set of item embeddings.

    Args:
        embeddings: array of shape (n_items, dim).

    Returns:
        a diversity score — e.g. the mean pairwise distance between distinct items. Document
        the exact distance you choose (cosine vs Euclidean) and why. Higher = more diverse.

    Verify: for Euclidean mean pairwise distance, compare to
        ``scipy.spatial.distance.pdist(embeddings).mean()``.

    Hint: compute all distinct pairwise distances and average them; decide explicitly how
        you normalize (e.g. cosine distance on L2-normalized rows) so the score is comparable
        across evalsets of different scale.
    """
    raise NotImplementedError(
        "Hand-code the diversity metric (Phase 3), then run handcode/verify_stats.py."
    )
