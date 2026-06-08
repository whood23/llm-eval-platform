"""Tests for the JudgmentCache facade over the store (AI-written)."""

from __future__ import annotations

from evalplatform.judge.cache import JudgmentCache
from evalplatform.models import (
    Candidate,
    EvalItem,
    JudgeMode,
    PairwiseJudgment,
    PointwiseJudgment,
    Position,
    RunMeta,
)


def _seed_pointwise(store):
    store.upsert_item(EvalItem(id="i1", input="Q"))
    store.upsert_candidate(Candidate(id="c1", item_id="i1", system="a", output="o"))
    run = RunMeta(mode=JudgeMode.pointwise, judge_model="dummy", prompt_version="v1")
    store.insert_run(run)
    store.insert_pointwise(
        PointwiseJudgment(
            run_id=run.run_id,
            item_id="i1",
            candidate_id="c1",
            judge_model="dummy",
            prompt_version="v1",
            score=0.7,
        )
    )


def _seed_pairwise(store):
    store.upsert_item(EvalItem(id="i1", input="Q"))
    store.upsert_candidates(
        [
            Candidate(id="c1", item_id="i1", system="a", output="o1"),
            Candidate(id="c2", item_id="i1", system="b", output="o2"),
        ]
    )
    run = RunMeta(mode=JudgeMode.pairwise, judge_model="dummy", prompt_version="v1")
    store.insert_run(run)
    store.insert_pairwise(
        PairwiseJudgment(
            run_id=run.run_id,
            item_id="i1",
            candidate_a_id="c1",
            candidate_b_id="c2",
            position=Position.AB,
            winner_slot="A",
            winner_candidate_id="c1",
            judge_model="dummy",
            prompt_version="v1",
        )
    )


def test_cache_pointwise_hit_when_enabled(store):
    _seed_pointwise(store)
    cache = JudgmentCache(store, enabled=True)
    hit = cache.get_pointwise("c1", "dummy", "v1")
    assert hit is not None
    assert hit.score == 0.7


def test_cache_pointwise_miss_returns_none(store):
    _seed_pointwise(store)
    cache = JudgmentCache(store, enabled=True)
    assert cache.get_pointwise("c1", "dummy", "v2") is None


def test_cache_disabled_always_misses(store):
    _seed_pointwise(store)
    cache = JudgmentCache(store, enabled=False)
    # Even with a row present, a disabled cache must return None.
    assert cache.get_pointwise("c1", "dummy", "v1") is None


def test_cache_pairwise_hit_accepts_enum_and_str(store):
    _seed_pairwise(store)
    cache = JudgmentCache(store, enabled=True)
    by_enum = cache.get_pairwise("c1", "c2", Position.AB, "dummy", "v1")
    by_str = cache.get_pairwise("c1", "c2", "AB", "dummy", "v1")
    assert by_enum is not None and by_str is not None
    assert by_enum.id == by_str.id


def test_cache_pairwise_disabled_misses(store):
    _seed_pairwise(store)
    cache = JudgmentCache(store, enabled=False)
    assert cache.get_pairwise("c1", "c2", Position.AB, "dummy", "v1") is None
