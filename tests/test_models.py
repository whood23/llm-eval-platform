"""Tests for the data contract models (scaffolding, AI-written)."""

from __future__ import annotations

from evalplatform.models import (
    Candidate,
    EvalItem,
    JudgeMode,
    PairwiseJudgment,
    PointwiseJudgment,
    Position,
    RunMeta,
    Trace,
    TraceStep,
    new_id,
)


def test_new_id_prefix_and_uniqueness():
    bare = new_id()
    assert "_" not in bare and len(bare) == 8

    prefixed = new_id("run")
    assert prefixed.startswith("run_")
    assert len(prefixed.split("_", 1)[1]) == 8

    # Two calls should (overwhelmingly) differ.
    assert new_id("run") != new_id("run")


def test_eval_item_defaults():
    item = EvalItem(id="i1", input="What is 2+2?")
    assert item.reference is None
    assert item.context is None
    assert item.stratum is None
    assert item.tags == []
    assert item.metadata == {}


def test_candidate_required_fields():
    c = Candidate(id="c1", item_id="i1", system="sys-a", output="4")
    assert c.item_id == "i1"
    assert c.system == "sys-a"
    assert c.metadata == {}


def test_judge_mode_and_position_enums():
    assert JudgeMode.pointwise.value == "pointwise"
    assert JudgeMode.pairwise.value == "pairwise"
    assert {p.value for p in Position} == {"AB", "BA", "NA"}
    # str-enum: comparable to the raw string.
    assert Position.AB == "AB"
    assert JudgeMode.pairwise == "pairwise"


def test_pointwise_judgment_defaults_and_autofields():
    j = PointwiseJudgment(
        run_id="run_1",
        item_id="i1",
        candidate_id="c1",
        judge_model="dummy",
        prompt_version="v1",
    )
    assert j.id.startswith("pw_")
    assert j.parse_ok is False
    assert j.cached is False
    assert j.score is None
    assert isinstance(j.created_at, float) and j.created_at > 0


def test_pairwise_judgment_position_typed():
    j = PairwiseJudgment(
        run_id="run_1",
        item_id="i1",
        candidate_a_id="c1",
        candidate_b_id="c2",
        position=Position.AB,
        judge_model="dummy",
        prompt_version="v1",
    )
    assert j.id.startswith("pr_")
    assert j.position is Position.AB
    assert j.winner_slot is None
    assert j.winner_candidate_id is None
    assert j.parse_ok is False


def test_run_meta_defaults():
    run = RunMeta(mode=JudgeMode.pointwise, judge_model="dummy", prompt_version="v1")
    assert run.run_id.startswith("run_")
    assert run.mode is JudgeMode.pointwise
    assert run.n_items == 0
    assert run.finished_at is None
    assert run.config_snapshot == {}


def test_trace_recorder_models_round_trip():
    steps = [
        TraceStep(index=0, kind="thought", content="thinking"),
        TraceStep(index=1, kind="tool_call", name="search", content="q"),
        TraceStep(index=2, kind="final", content="answer"),
    ]
    trace = Trace(item_id="i1", system="agent-x", steps=steps, final_output="answer")
    assert trace.id.startswith("trace_")
    assert [s.index for s in trace.steps] == [0, 1, 2]
    assert trace.steps[1].name == "search"
    assert trace.final_output == "answer"
