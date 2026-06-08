"""Tests for the SQLite store: schema init, upsert/insert/query, cache roundtrip (AI)."""

from __future__ import annotations

import time

from evalplatform.models import (
    Candidate,
    EvalItem,
    JudgeMode,
    PairwiseJudgment,
    PointwiseJudgment,
    Position,
    RunMeta,
)


def _seed_item_candidate(store):
    item = EvalItem(id="i1", input="Q1", stratum="math", tags=["t1"], metadata={"k": "v"})
    cand = Candidate(id="c1", item_id="i1", system="sys-a", output="A1")
    store.upsert_item(item)
    store.upsert_candidate(cand)
    return item, cand


def test_init_db_creates_tables(store):
    rows = store.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    names = {r["name"] for r in rows}
    expected = {
        "runs",
        "items",
        "candidates",
        "pointwise_judgments",
        "pairwise_judgments",
        "gold_labels",
    }
    assert expected <= names


def test_cache_unique_indexes_exist(store):
    rows = store.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index'"
    ).fetchall()
    names = {r["name"] for r in rows}
    assert "idx_pw_cache" in names
    assert "idx_pr_cache" in names


def test_run_insert_get_finish_latest(store):
    run = RunMeta(
        mode=JudgeMode.pointwise,
        judge_model="dummy",
        prompt_version="v1",
        config_snapshot={"a": 1},
    )
    store.insert_run(run)

    got = store.get_run(run.run_id)
    assert got is not None
    assert got.run_id == run.run_id
    assert got.mode is JudgeMode.pointwise
    assert got.config_snapshot == {"a": 1}
    assert got.finished_at is None

    store.finish_run(run.run_id, finished_at=time.time(), n_items=7)
    finished = store.get_run(run.run_id)
    assert finished.n_items == 7
    assert finished.finished_at is not None

    assert store.latest_run().run_id == run.run_id
    assert store.latest_run(mode="pointwise").run_id == run.run_id
    assert store.latest_run(mode="pairwise") is None


def test_item_and_candidate_roundtrip(store):
    item, cand = _seed_item_candidate(store)

    got_item = store.get_item("i1")
    assert got_item.input == "Q1"
    assert got_item.stratum == "math"
    assert got_item.tags == ["t1"]
    assert got_item.metadata == {"k": "v"}

    cands = store.get_candidates("i1")
    assert len(cands) == 1
    assert cands[0].id == "c1"
    assert cands[0].output == "A1"


def test_upsert_is_idempotent(store):
    _seed_item_candidate(store)
    # Re-upsert with a changed field; INSERT OR REPLACE keeps one row.
    store.upsert_candidate(Candidate(id="c1", item_id="i1", system="sys-a", output="A2"))
    cands = store.get_candidates("i1")
    assert len(cands) == 1
    assert cands[0].output == "A2"


def test_batch_upserts(store):
    items = [EvalItem(id=f"i{i}", input=f"Q{i}") for i in range(3)]
    store.upsert_items(items)
    cands = [Candidate(id=f"c{i}", item_id=f"i{i}", system="s", output="o") for i in range(3)]
    store.upsert_candidates(cands)
    for i in range(3):
        assert store.get_item(f"i{i}") is not None
        assert len(store.get_candidates(f"i{i}")) == 1


def test_pointwise_insert_and_query(store):
    _seed_item_candidate(store)
    run = RunMeta(mode=JudgeMode.pointwise, judge_model="dummy", prompt_version="v1")
    store.insert_run(run)
    j = PointwiseJudgment(
        run_id=run.run_id,
        item_id="i1",
        candidate_id="c1",
        judge_model="dummy",
        prompt_version="v1",
        score=0.8,
        parse_ok=True,
        raw_response="raw",
    )
    store.insert_pointwise(j)

    fetched = store.pointwise_for_run(run.run_id)
    assert len(fetched) == 1
    assert fetched[0].score == 0.8
    assert fetched[0].parse_ok is True
    assert fetched[0].raw_response == "raw"


def test_pointwise_cache_hit_roundtrip(store):
    _seed_item_candidate(store)
    run = RunMeta(mode=JudgeMode.pointwise, judge_model="dummy", prompt_version="v1")
    store.insert_run(run)
    store.insert_pointwise(
        PointwiseJudgment(
            run_id=run.run_id,
            item_id="i1",
            candidate_id="c1",
            judge_model="dummy",
            prompt_version="v1",
            score=0.5,
        )
    )

    hit = store.get_cached_pointwise("c1", "dummy", "v1")
    assert hit is not None
    assert hit.candidate_id == "c1"
    assert hit.score == 0.5

    # Miss on a different prompt_version.
    assert store.get_cached_pointwise("c1", "dummy", "v2") is None


def test_pairwise_insert_query_and_cache(store):
    store.upsert_item(EvalItem(id="i1", input="Q"))
    store.upsert_candidates(
        [
            Candidate(id="c1", item_id="i1", system="a", output="o1"),
            Candidate(id="c2", item_id="i1", system="b", output="o2"),
        ]
    )
    run = RunMeta(mode=JudgeMode.pairwise, judge_model="dummy", prompt_version="v1")
    store.insert_run(run)
    j = PairwiseJudgment(
        run_id=run.run_id,
        item_id="i1",
        candidate_a_id="c1",
        candidate_b_id="c2",
        position=Position.AB,
        winner_slot="A",
        winner_candidate_id="c1",
        judge_model="dummy",
        prompt_version="v1",
        parse_ok=True,
    )
    store.insert_pairwise(j)

    fetched = store.pairwise_for_run(run.run_id)
    assert len(fetched) == 1
    assert fetched[0].position is Position.AB
    assert fetched[0].winner_candidate_id == "c1"

    # Cache lookup accepts both the enum and its .value string.
    hit_enum = store.get_cached_pairwise("c1", "c2", Position.AB, "dummy", "v1")
    hit_str = store.get_cached_pairwise("c1", "c2", "AB", "dummy", "v1")
    assert hit_enum is not None and hit_str is not None
    assert hit_enum.id == hit_str.id == j.id

    # Position is part of the key: BA is a separate (missing) entry.
    assert store.get_cached_pairwise("c1", "c2", "BA", "dummy", "v1") is None


def test_gold_labels_roundtrip(store):
    _seed_item_candidate(store)
    gid = store.insert_gold_label(
        item_id="i1", labeler="human-1", label="pass", candidate_id="c1", metadata={"n": 1}
    )
    assert isinstance(gid, str) and gid.startswith("gold_")

    all_labels = store.gold_labels()
    assert len(all_labels) == 1
    assert all_labels[0]["label"] == "pass"
    assert all_labels[0]["metadata"] == {"n": 1}

    by_item = store.gold_labels(item_id="i1")
    assert len(by_item) == 1
    assert store.gold_labels(item_id="nope") == []


def test_store_context_manager(settings):
    from evalplatform.store.db import Store

    with Store.open(settings) as st:
        st.upsert_item(EvalItem(id="i1", input="Q"))
        assert st.get_item("i1") is not None
