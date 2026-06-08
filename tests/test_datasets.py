"""Tests for dataset I/O + versioning (AI-written). No sampling logic here (that's a stub)."""

from __future__ import annotations

from evalplatform.datasets.loader import (
    candidates_by_item,
    load_candidates,
    load_items,
    load_jsonl,
    write_candidates,
    write_items,
    write_jsonl,
)
from evalplatform.datasets.versioning import dataset_version, write_dataset_card
from evalplatform.models import Candidate, EvalItem


def _sample_items():
    return [
        EvalItem(id="i1", input="Q1", reference="R1", stratum="math", tags=["a"]),
        EvalItem(id="i2", input="Q2", context="ctx", stratum="prose"),
        EvalItem(id="i3", input="Q3", stratum="math"),
    ]


def test_jsonl_roundtrip(tmp_path):
    rows = [{"id": "i1", "x": 1}, {"id": "i2", "x": 2}]
    path = tmp_path / "rows.jsonl"
    write_jsonl(rows, path)
    assert load_jsonl(path) == rows


def test_items_roundtrip(tmp_path):
    items = _sample_items()
    path = tmp_path / "items.jsonl"
    write_items(items, path)
    loaded = load_items(path)
    assert [i.id for i in loaded] == ["i1", "i2", "i3"]
    assert loaded[0].reference == "R1"
    assert loaded[0].stratum == "math"
    assert loaded[1].context == "ctx"


def test_candidates_roundtrip_and_grouping(tmp_path):
    cands = [
        Candidate(id="c1", item_id="i1", system="a", output="o1"),
        Candidate(id="c2", item_id="i1", system="b", output="o2"),
        Candidate(id="c3", item_id="i2", system="a", output="o3"),
    ]
    path = tmp_path / "cands.jsonl"
    write_candidates(cands, path)
    loaded = load_candidates(path)
    assert len(loaded) == 3

    grouped = candidates_by_item(loaded)
    assert set(grouped) == {"i1", "i2"}
    assert [c.id for c in grouped["i1"]] == ["c1", "c2"]
    assert [c.id for c in grouped["i2"]] == ["c3"]


def test_dataset_version_stable_and_order_independent():
    items = _sample_items()
    v1 = dataset_version(items)
    v2 = dataset_version(list(reversed(items)))
    assert v1 == v2  # order of the input must not change the version
    assert v1.startswith("ds_")
    assert len(v1) == len("ds_") + 12


def test_dataset_version_changes_with_content():
    base = _sample_items()
    changed = list(base)
    changed[0] = EvalItem(id="i1", input="DIFFERENT", stratum="math")
    assert dataset_version(base) != dataset_version(changed)


def test_dataset_card_written_and_summary(tmp_path):
    items = _sample_items()
    card_path = tmp_path / "card.md"
    summary = write_dataset_card(items, card_path, name="My Eval Set", notes="hello")

    assert card_path.exists()
    assert summary["name"] == "My Eval Set"
    assert summary["n_items"] == 3
    # Two math items, one prose item.
    assert summary["strata"]["math"] == 2
    assert summary["strata"]["prose"] == 1
    assert summary["version"] == dataset_version(items)
    assert summary["n_with_reference"] == 1
    assert summary["n_with_context"] == 1

    text = card_path.read_text(encoding="utf-8")
    assert "My Eval Set" in text
    assert "math" in text
    assert "hello" in text
