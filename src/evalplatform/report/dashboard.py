"""Optional Streamlit dashboard over the eval store (scaffolding, AI).

Run via ``eval-platform dashboard`` (which shells out to ``streamlit run`` on this file)
or directly with ``streamlit run src/evalplatform/report/dashboard.py``. Streamlit is an
optional dependency imported LAZILY: if it is missing, ``main`` prints an install hint and
returns instead of raising. Stats are read from the hand-coded ``evalplatform.stats`` layer
with ``try/except NotImplementedError`` tolerance — this module never computes statistics.
"""

from __future__ import annotations

from typing import Any

from ..config import get_settings
from ..models import JudgeMode
from ..store.db import Store


def main() -> None:
    """Entry point for ``streamlit run``. No-op (with hint) if Streamlit is unavailable."""
    try:
        import streamlit as st
    except ImportError:
        print(
            "Streamlit is not installed. Install the dashboard extra:\n"
            "    pip install .[dashboard]   # or: pip install streamlit\n"
            "then run:  eval-platform dashboard"
        )
        return

    st.set_page_config(page_title="llm-eval-platform", layout="wide")
    st.title("llm-eval-platform — runs dashboard")

    settings = get_settings()
    store = Store.open(settings)

    runs = _list_runs(store)
    if not runs:
        st.info("No runs found in the store yet. Run `eval-platform run ...` first.")
        return

    # Run picker (most recent first).
    labels = [
        f"{r.run_id}  ·  {r.mode.value}  ·  {r.judge_model}" for r in runs
    ]
    choice = st.sidebar.selectbox("Run", options=range(len(runs)), format_func=lambda i: labels[i])
    run = runs[choice]

    _render_run(st, store, run)


def _render_run(st: Any, store: Store, run: Any) -> None:
    """Render metadata, counts, sample judgments, and (tolerant) stats for one run."""
    st.subheader("Run metadata")
    st.json(
        {
            "run_id": run.run_id,
            "mode": run.mode.value,
            "judge_model": run.judge_model,
            "prompt_version": run.prompt_version,
            "dataset_id": run.dataset_id,
            "dataset_version": run.dataset_version,
            "n_items": run.n_items,
            "git_sha": run.git_sha,
            "started_at": run.started_at,
            "finished_at": run.finished_at,
            "notes": run.notes,
        }
    )

    is_pairwise = run.mode == JudgeMode.pairwise
    judgments = (
        store.pairwise_for_run(run.run_id)
        if is_pairwise
        else store.pointwise_for_run(run.run_id)
    )

    n_total = len(judgments)
    n_parse_ok = sum(1 for j in judgments if j.parse_ok)
    n_cached = sum(1 for j in judgments if j.cached)

    c1, c2, c3 = st.columns(3)
    c1.metric("judgments", n_total)
    c2.metric("parse-ok rate", f"{(n_parse_ok / n_total):.0%}" if n_total else "—")
    c3.metric("served from cache", n_cached)

    st.subheader("Statistics")
    if is_pairwise:
        _render_pairwise_stats(st, judgments)
    else:
        _render_pointwise_stats(st, judgments)

    st.subheader("Sample judgments")
    st.dataframe(_to_table(judgments[:25], is_pairwise))


def _render_pointwise_stats(st: Any, judgments: list[Any]) -> None:
    scores = [j.score for j in judgments if j.parse_ok and j.score is not None]

    st.markdown("**Mean score (bootstrap 95% CI)**")
    try:
        from .. import stats

        if scores:
            ci = stats.bootstrap_ci(scores)
            st.write(f"point = {ci.point:.4f}, 95% CI = [{ci.low:.4f}, {ci.high:.4f}]")
        else:
            st.caption("no parsed numeric scores available")
    except NotImplementedError:
        st.warning("not yet implemented — hand-code in `stats/bootstrap.py`")


def _render_pairwise_stats(st: Any, judgments: list[Any]) -> None:
    st.markdown("**Position flip rate**")
    try:
        from .. import stats

        flip = stats.position_flip_rate(judgments)
        st.write(f"flip rate = {flip:.4f}")
    except NotImplementedError:
        st.warning("not yet implemented — hand-code in `stats/position_bias.py`")

    st.markdown("**Bias-corrected win rate**")
    try:
        from .. import stats

        winrate = stats.consistent_winrate(judgments)
        if winrate:
            st.dataframe(
                [
                    {"candidate": cid, "consistent_win_rate": rate}
                    for cid, rate in sorted(winrate.items())
                ]
            )
        else:
            st.caption("no complete pairings available")
    except NotImplementedError:
        st.warning("not yet implemented — hand-code in `stats/position_bias.py`")


def _list_runs(store: Store) -> list[Any]:
    """All runs, newest first. Uses a raw query since the Store exposes only latest_run."""
    rows = store.conn.execute(
        "SELECT run_id FROM runs ORDER BY started_at DESC"
    ).fetchall()
    out: list[Any] = []
    for row in rows:
        run = store.get_run(row["run_id"])
        if run is not None:
            out.append(run)
    return out


def _to_table(judgments: list[Any], is_pairwise: bool) -> list[dict]:
    table: list[dict] = []
    for j in judgments:
        if is_pairwise:
            pos = j.position.value if hasattr(j.position, "value") else j.position
            table.append(
                {
                    "item_id": j.item_id,
                    "position": pos,
                    "winner_slot": j.winner_slot,
                    "winner_candidate_id": j.winner_candidate_id,
                    "parse_ok": j.parse_ok,
                    "cached": j.cached,
                    "raw_response": _short(j.raw_response),
                }
            )
        else:
            table.append(
                {
                    "item_id": j.item_id,
                    "candidate_id": j.candidate_id,
                    "score": j.score,
                    "label": j.label,
                    "parse_ok": j.parse_ok,
                    "cached": j.cached,
                    "raw_response": _short(j.raw_response),
                }
            )
    return table


def _short(value: Any, limit: int = 200) -> str:
    if value is None:
        return ""
    text = str(value)
    return text if len(text) <= limit else text[:limit] + "…"


# Streamlit executes the script top-to-bottom; invoke main() when run that way.
if __name__ == "__main__":
    main()
else:  # pragma: no cover - Streamlit runs this module as __main__ via runpy
    # When launched by `streamlit run`, the module name is "__main__" so the branch above
    # fires. Importing the module elsewhere should NOT auto-run the app.
    pass
