"""Judgment cache (scaffolding, AI).

A thin facade over the store's cache-key lookups. When disabled it always misses
(returns ``None``); when enabled it delegates to the unique-index queries in
:class:`evalplatform.store.db.Store`. No statistics, no judge logic here.
"""

from __future__ import annotations

from typing import Optional

from ..models import PairwiseJudgment, PointwiseJudgment, Position
from ..store.db import Store


class JudgmentCache:
    """Cache facade: returns a previously stored judgment for an identical key, or ``None``."""

    def __init__(self, store: Store, enabled: bool = True) -> None:
        self.store = store
        self.enabled = enabled

    def get_pointwise(
        self, candidate_id: str, judge_model: str, prompt_version: str
    ) -> Optional[PointwiseJudgment]:
        if not self.enabled:
            return None
        return self.store.get_cached_pointwise(candidate_id, judge_model, prompt_version)

    def get_pairwise(
        self,
        candidate_a_id: str,
        candidate_b_id: str,
        position: Position | str,
        judge_model: str,
        prompt_version: str,
    ) -> Optional[PairwiseJudgment]:
        if not self.enabled:
            return None
        return self.store.get_cached_pairwise(
            candidate_a_id, candidate_b_id, position, judge_model, prompt_version
        )
