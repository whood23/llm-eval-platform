"""Tests that the Runner gracefully surfaces unimplemented judge stubs (AI-written).

We build the *real* hand-coded ``PointwiseJudge`` / ``PairwiseJudge`` — whose rubric/
parse/swap methods are still ``NotImplementedError`` stubs — and assert the Runner raises
``JudgeNotImplemented`` instead of crashing. We never implement any judge method here.
"""

from __future__ import annotations

import pytest

from evalplatform.judge.cache import JudgmentCache
from evalplatform.judge.pairwise import PairwiseJudge
from evalplatform.judge.pointwise import PointwiseJudge
from evalplatform.judge.runner import JudgeNotImplemented, Runner
from evalplatform.models import Candidate, EvalItem
from evalplatform.providers.dummy import DummyProvider


@pytest.fixture
def runner(settings, store):
    provider = DummyProvider()
    cache = JudgmentCache(store, enabled=settings.cache_enabled)
    return Runner(settings=settings, store=store, provider=provider, cache=cache)


def _items_and_candidates():
    items = [EvalItem(id="i1", input="Q1"), EvalItem(id="i2", input="Q2")]
    candidates = {
        "i1": [
            Candidate(id="c1", item_id="i1", system="a", output="o1"),
            Candidate(id="c2", item_id="i1", system="b", output="o2"),
        ],
        "i2": [
            Candidate(id="c3", item_id="i2", system="a", output="o3"),
            Candidate(id="c4", item_id="i2", system="b", output="o4"),
        ],
    }
    return items, candidates


def test_judge_not_implemented_is_runtime_error():
    assert issubclass(JudgeNotImplemented, RuntimeError)


def test_run_pointwise_raises_judge_not_implemented(runner):
    items, candidates = _items_and_candidates()
    judge = PointwiseJudge(prompt_version="v1")  # real stub; build_prompt -> NotImplementedError
    with pytest.raises(JudgeNotImplemented):
        runner.run_pointwise(items, candidates, judge)


def test_run_pairwise_raises_judge_not_implemented(runner):
    items, candidates = _items_and_candidates()
    judge = PairwiseJudge(prompt_version="v1")  # real stub; iter_presentations -> NotImplementedError
    with pytest.raises(JudgeNotImplemented):
        runner.run_pairwise(items, candidates, judge)


def test_judge_not_implemented_message_names_a_file(runner):
    items, candidates = _items_and_candidates()
    judge = PointwiseJudge(prompt_version="v1")
    with pytest.raises(JudgeNotImplemented) as excinfo:
        runner.run_pointwise(items, candidates, judge)
    # The message should be actionable: point at the judge file to implement.
    assert "pointwise" in str(excinfo.value).lower()
