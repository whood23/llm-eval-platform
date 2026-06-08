"""Shared pytest fixtures + path setup (scaffolding, AI-written).

Puts the src-layout package on ``sys.path`` so the suite runs without an editable
install, clears the cached settings singleton between tests, and provides tmp_path
based ``Settings``/``Store`` fixtures so every test gets an isolated SQLite db.

No hand-code core (rubric/parser/swap/sampling/scoring/statistic) is implemented here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# --- make the src-layout package importable without `pip install -e .` ---------------
# tests/conftest.py -> repo root is parents[1]; the package lives under <root>/src.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """Reset the ``get_settings`` lru_cache so per-test env/paths never leak."""
    from evalplatform.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings(tmp_path: Path):
    """A ``Settings`` instance whose db/data/reports all live under ``tmp_path``.

    Built directly (not via env) so the test is hermetic regardless of the caller's
    environment; the offline ``dummy`` provider is kept so no API keys are needed.
    """
    from evalplatform.config import Settings

    return Settings(
        provider="dummy",
        cache_enabled=True,
        db_path=tmp_path / "eval.db",
        data_dir=tmp_path / "data",
        reports_dir=tmp_path / "reports",
    )


@pytest.fixture
def store(settings):
    """An initialized :class:`Store` on the tmp_path db, closed on teardown."""
    from evalplatform.store.db import Store

    st = Store.open(settings)
    try:
        yield st
    finally:
        st.close()
