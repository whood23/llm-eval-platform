"""End-to-end CLI plumbing test: init-db + smoke on a temp db (AI-written).

Drives the Typer app with ``CliRunner``. The ``smoke`` command exercises the
provider/store/cache/retry/ratelimit wiring WITHOUT the judge (no rubric), so it must
succeed on a bare offline checkout. We assert exit code 0 and that rows were persisted.
"""

from __future__ import annotations

import pytest

# Typer's CliRunner is part of typer (installed). Skip cleanly if somehow absent.
typer_testing = pytest.importorskip("typer.testing")
from typer.testing import CliRunner  # noqa: E402

from evalplatform.cli import app  # noqa: E402
from evalplatform.config import get_settings  # noqa: E402
from evalplatform.store.db import Store  # noqa: E402


@pytest.fixture
def env_db(tmp_path, monkeypatch):
    """Point the CLI's settings at a temp db via env vars (prefix EVAL_)."""
    db_path = tmp_path / "eval.db"
    monkeypatch.setenv("EVAL_PROVIDER", "dummy")
    monkeypatch.setenv("EVAL_DB_PATH", str(db_path))
    monkeypatch.setenv("EVAL_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("EVAL_REPORTS_DIR", str(tmp_path / "reports"))
    get_settings.cache_clear()
    return db_path


def test_init_db_command(env_db):
    runner = CliRunner()
    result = runner.invoke(app, ["init-db"])
    assert result.exit_code == 0, result.output
    assert env_db.exists()


def test_version_command():
    runner = CliRunner()
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0, result.output


def test_smoke_persists_rows_and_reports_cache(env_db):
    runner = CliRunner()
    result = runner.invoke(app, ["smoke", "--n", "5"])
    assert result.exit_code == 0, result.output

    # Rows persisted as pointwise judgments under the smoke sentinel keys.
    get_settings.cache_clear()
    store = Store.open(get_settings())
    try:
        rows = store.conn.execute(
            "SELECT * FROM pointwise_judgments WHERE judge_model = '__smoke__'"
        ).fetchall()
        assert len(rows) == 5
        # Smoke writes are diagnostic, not parsed rubric scores.
        assert all(r["parse_ok"] == 0 for r in rows)
        assert all(r["prompt_version"] == "__smoke__" for r in rows)
        assert all(r["raw_response"] for r in rows)
    finally:
        store.close()


def test_smoke_reports_cache_hits_on_second_pass(env_db):
    # The smoke command runs two passes over the SAME synthetic candidates: the second
    # pass must hit the cache and report it (cached=N). It exercises the cache wiring
    # without re-deriving any judge logic.
    runner = CliRunner()
    result = runner.invoke(app, ["smoke", "--n", "3"])
    assert result.exit_code == 0, result.output
    out = result.output.lower()
    assert "pass 2" in out
    assert "cached=3" in out  # every unit hit the cache on the second pass


def test_smoke_idempotent_rowcount_within_invocation(env_db):
    # One invocation writes exactly --n rows: the second pass replaces rows on the unique
    # cache key (INSERT OR REPLACE) rather than duplicating them.
    runner = CliRunner()
    result = runner.invoke(app, ["smoke", "--n", "4"])
    assert result.exit_code == 0, result.output

    get_settings.cache_clear()
    store = Store.open(get_settings())
    try:
        rows = store.conn.execute(
            "SELECT * FROM pointwise_judgments WHERE judge_model = '__smoke__'"
        ).fetchall()
        assert len(rows) == 4
        # After the cache-hit second pass, the persisted rows are marked cached.
        assert all(r["cached"] == 1 for r in rows)
    finally:
        store.close()
