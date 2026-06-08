"""Self-contained static HTML report for a single run (scaffolding, AI).

Reads a run + its judgments from the :class:`~evalplatform.store.db.Store`, renders
run metadata, counts, parse-ok rate, and a few sample judgments. Wherever the report
would show a statistic (bootstrap CIs, calibration/ECE, position bias), it CALLS the
hand-coded ``evalplatform.stats`` functions but wraps each call in
``try/except NotImplementedError`` so an un-implemented stat degrades to a clear
placeholder instead of crashing the report. No statistics are computed here.
"""

from __future__ import annotations

import html
import time
from pathlib import Path
from typing import Any, Optional, Union

from ..models import JudgeMode
from ..store.db import Store

PathLike = Union[str, Path]

# Shown in place of any statistic whose hand-coded core still raises NotImplementedError.
_STAT_TODO = (
    "<em>not yet implemented &mdash; hand-code in "
    "<code>stats/{module}.py</code></em>"
)


def build_report(
    store: Store,
    run_id: str,
    *,
    out_dir: Optional[PathLike] = None,
) -> Path:
    """Write a static HTML report for ``run_id`` and return the path to the .html file.

    Plots (when their underlying stats are implemented) are saved as PNGs next to the
    HTML and embedded via ``<img>`` tags.
    """
    run = store.get_run(run_id)
    if run is None:
        raise ValueError(f"run_id not found in store: {run_id!r}")

    out_dir = Path(out_dir) if out_dir is not None else Path("reports") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    is_pairwise = run.mode == JudgeMode.pairwise
    if is_pairwise:
        judgments: list[Any] = store.pairwise_for_run(run_id)
    else:
        judgments = store.pointwise_for_run(run_id)

    n_total = len(judgments)
    n_parse_ok = sum(1 for j in judgments if j.parse_ok)
    n_cached = sum(1 for j in judgments if j.cached)
    parse_rate = (n_parse_ok / n_total) if n_total else 0.0

    # --- assemble sections ----------------------------------------------------------
    sections: list[str] = []
    sections.append(_render_meta(run))
    sections.append(
        _render_counts(
            n_total=n_total,
            n_parse_ok=n_parse_ok,
            n_cached=n_cached,
            parse_rate=parse_rate,
            mode=run.mode.value,
        )
    )

    # Stats sections (each tolerant of NotImplementedError) + any plots.
    plot_imgs: list[str] = []
    if is_pairwise:
        sections.append(_render_pairwise_stats(judgments, out_dir, plot_imgs))
    else:
        sections.append(_render_pointwise_stats(judgments, out_dir, plot_imgs))

    if plot_imgs:
        sections.append(
            "<section><h2>Plots</h2>" + "".join(plot_imgs) + "</section>"
        )

    sections.append(_render_samples(judgments, is_pairwise))

    body = "\n".join(sections)
    document = _HTML_SHELL.format(
        run_id=html.escape(run_id),
        body=body,
        generated_at=html.escape(_fmt_ts(time.time())),
    )

    out_path = out_dir / "report.html"
    out_path.write_text(document, encoding="utf-8")
    return out_path


# --- section renderers ---------------------------------------------------------------


def _render_meta(run: Any) -> str:
    rows = [
        ("run_id", run.run_id),
        ("mode", run.mode.value),
        ("judge_model", run.judge_model),
        ("prompt_version", run.prompt_version),
        ("dataset_id", run.dataset_id),
        ("dataset_version", run.dataset_version),
        ("n_items", run.n_items),
        ("git_sha", run.git_sha),
        ("started_at", _fmt_ts(run.started_at)),
        ("finished_at", _fmt_ts(run.finished_at)),
        ("notes", run.notes),
    ]
    body = "".join(
        f"<tr><th>{html.escape(k)}</th><td>{_cell(v)}</td></tr>" for k, v in rows
    )
    return f"<section><h2>Run metadata</h2><table>{body}</table></section>"


def _render_counts(
    *,
    n_total: int,
    n_parse_ok: int,
    n_cached: int,
    parse_rate: float,
    mode: str,
) -> str:
    cards = [
        ("judgments", n_total),
        ("parse-ok", f"{n_parse_ok} / {n_total}"),
        ("parse-ok rate", f"{parse_rate:.1%}"),
        ("served from cache", n_cached),
    ]
    body = "".join(
        f'<div class="card"><div class="val">{_cell(v)}</div>'
        f'<div class="lab">{html.escape(label)}</div></div>'
        for label, v in cards
    )
    return f'<section><h2>Counts ({html.escape(mode)})</h2><div class="cards">{body}</div></section>'


def _render_pointwise_stats(
    judgments: list[Any], out_dir: Path, plot_imgs: list[str]
) -> str:
    """Calibration / CI stats for a pointwise run — each call tolerant of NotImplementedError."""
    parts: list[str] = ["<section><h2>Statistics (pointwise)</h2>"]

    scores = [j.score for j in judgments if j.parse_ok and j.score is not None]

    # Bootstrap CI of mean score.
    parts.append("<h3>Mean score (bootstrap 95% CI)</h3>")
    try:
        from .. import stats

        if scores:
            ci = stats.bootstrap_ci(scores)
            parts.append(
                f"<p>point={ci.point:.4f}, 95% CI=[{ci.low:.4f}, {ci.high:.4f}]</p>"
            )
            self_img = _try_ci_plot(["mean score"], ci, out_dir)
            if self_img:
                plot_imgs.append(self_img)
        else:
            parts.append("<p><em>no parsed numeric scores available</em></p>")
    except NotImplementedError:
        parts.append("<p>" + _STAT_TODO.format(module="bootstrap") + "</p>")

    # Calibration: ECE + reliability diagram. Needs gold correctness; we surface the
    # placeholder when either the stat is unimplemented or correctness labels are absent.
    parts.append("<h3>Calibration</h3>")
    try:
        from .. import stats

        correct = _correctness_for(judgments)
        confidences = _confidences_for(judgments)
        if correct is not None and confidences:
            ece = stats.expected_calibration_error(confidences, correct)
            parts.append(f"<p>ECE = {ece:.4f}</p>")
            curve = stats.reliability_curve(confidences, correct)
            img = _try_reliability_plot(curve, out_dir)
            if img:
                plot_imgs.append(img)
        else:
            parts.append(
                "<p><em>no gold correctness labels available to assess calibration</em></p>"
            )
    except NotImplementedError:
        parts.append("<p>" + _STAT_TODO.format(module="calibration") + "</p>")

    parts.append("</section>")
    return "".join(parts)


def _render_pairwise_stats(
    judgments: list[Any], out_dir: Path, plot_imgs: list[str]
) -> str:
    """Position-bias stats for a pairwise run — each call tolerant of NotImplementedError."""
    parts: list[str] = ["<section><h2>Statistics (pairwise)</h2>"]

    # Position flip rate.
    parts.append("<h3>Position flip rate</h3>")
    try:
        from .. import stats

        flip = stats.position_flip_rate(judgments)
        parts.append(f"<p>flip rate = {flip:.4f}</p>")
        img = _try_bias_plot(["flip rate"], [flip], out_dir)
        if img:
            plot_imgs.append(img)
    except NotImplementedError:
        parts.append("<p>" + _STAT_TODO.format(module="position_bias") + "</p>")

    # Consistent (bias-corrected) win rate.
    parts.append("<h3>Bias-corrected win rate</h3>")
    try:
        from .. import stats

        winrate = stats.consistent_winrate(judgments)
        if winrate:
            rows = "".join(
                f"<tr><td>{html.escape(str(cid))}</td><td>{rate:.4f}</td></tr>"
                for cid, rate in sorted(winrate.items())
            )
            parts.append(
                f"<table><tr><th>candidate</th><th>consistent win rate</th></tr>{rows}</table>"
            )
        else:
            parts.append("<p><em>no complete pairings available</em></p>")
    except NotImplementedError:
        parts.append("<p>" + _STAT_TODO.format(module="position_bias") + "</p>")

    parts.append("</section>")
    return "".join(parts)


def _render_samples(judgments: list[Any], is_pairwise: bool) -> str:
    sample = judgments[:5]
    if not sample:
        return "<section><h2>Sample judgments</h2><p><em>none</em></p></section>"

    rows: list[str] = []
    if is_pairwise:
        header = (
            "<tr><th>item_id</th><th>position</th><th>winner_slot</th>"
            "<th>winner_candidate</th><th>parse_ok</th><th>raw_response</th></tr>"
        )
        for j in sample:
            pos = j.position.value if hasattr(j.position, "value") else j.position
            rows.append(
                "<tr>"
                f"<td>{_cell(j.item_id)}</td>"
                f"<td>{_cell(pos)}</td>"
                f"<td>{_cell(j.winner_slot)}</td>"
                f"<td>{_cell(j.winner_candidate_id)}</td>"
                f"<td>{_cell(j.parse_ok)}</td>"
                f"<td><pre>{_truncate(j.raw_response)}</pre></td>"
                "</tr>"
            )
    else:
        header = (
            "<tr><th>item_id</th><th>candidate_id</th><th>score</th><th>label</th>"
            "<th>parse_ok</th><th>raw_response</th></tr>"
        )
        for j in sample:
            rows.append(
                "<tr>"
                f"<td>{_cell(j.item_id)}</td>"
                f"<td>{_cell(j.candidate_id)}</td>"
                f"<td>{_cell(j.score)}</td>"
                f"<td>{_cell(j.label)}</td>"
                f"<td>{_cell(j.parse_ok)}</td>"
                f"<td><pre>{_truncate(j.raw_response)}</pre></td>"
                "</tr>"
            )
    return (
        "<section><h2>Sample judgments</h2>"
        f"<table>{header}{''.join(rows)}</table></section>"
    )


# --- plot helpers (each returns an <img> tag or None) --------------------------------


def _try_ci_plot(labels: list[str], ci: Any, out_dir: Path) -> Optional[str]:
    try:
        from . import plots

        fig, ax = plots.plt.subplots(figsize=(3.5, 4.0))
        plots.ci_bar(labels, [ci.point], [ci.low], [ci.high], ax=ax)
        path = plots.savefig(fig, out_dir / "ci.png")
        return _img_tag(path.name, "Bootstrap CI")
    except Exception:  # plotting must never break the report
        return None


def _try_reliability_plot(curve: Any, out_dir: Path) -> Optional[str]:
    try:
        from . import plots

        fig, ax = plots.plt.subplots(figsize=(4.5, 4.5))
        plots.reliability_diagram(curve, ax=ax)
        path = plots.savefig(fig, out_dir / "reliability.png")
        return _img_tag(path.name, "Reliability diagram")
    except Exception:
        return None


def _try_bias_plot(labels: list[str], values: list[float], out_dir: Path) -> Optional[str]:
    try:
        from . import plots

        fig, ax = plots.plt.subplots(figsize=(3.5, 4.0))
        plots.bias_bar(labels, values, ax=ax)
        path = plots.savefig(fig, out_dir / "bias.png")
        return _img_tag(path.name, "Position bias")
    except Exception:
        return None


# --- small data helpers --------------------------------------------------------------


def _correctness_for(judgments: list[Any]) -> Optional[list[bool]]:
    """Pull per-judgment correctness from metadata, if present (no stat computed).

    The report does not invent correctness labels. If a judgment carries a boolean
    ``metadata['correct']`` we surface it for calibration; otherwise we report that no
    gold labels are available. Returns ``None`` when not every parsed judgment is labeled.
    """
    out: list[bool] = []
    for j in judgments:
        if not j.parse_ok or j.score is None:
            continue
        meta = j.metadata or {}
        if "correct" not in meta:
            return None
        out.append(bool(meta["correct"]))
    return out or None


def _confidences_for(judgments: list[Any]) -> list[float]:
    """Parsed scores treated as confidences (clamped to [0,1]); no statistic computed."""
    out: list[float] = []
    for j in judgments:
        if not j.parse_ok or j.score is None:
            continue
        out.append(min(1.0, max(0.0, float(j.score))))
    return out


# --- formatting helpers --------------------------------------------------------------


def _cell(value: Any) -> str:
    if value is None:
        return "<span class='muted'>&mdash;</span>"
    return html.escape(str(value))


def _truncate(value: Any, limit: int = 300) -> str:
    if value is None:
        return ""
    text = str(value)
    if len(text) > limit:
        text = text[:limit] + "…"
    return html.escape(text)


def _fmt_ts(ts: Optional[float]) -> Optional[str]:
    if ts is None:
        return None
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(ts)))
    except (TypeError, ValueError, OSError):
        return str(ts)


def _img_tag(filename: str, alt: str) -> str:
    return (
        f'<figure><img src="{html.escape(filename)}" alt="{html.escape(alt)}">'
        f"<figcaption>{html.escape(alt)}</figcaption></figure>"
    )


_HTML_SHELL = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>eval report &mdash; {run_id}</title>
<style>
  body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
         margin: 2rem auto; max-width: 1000px; color: #1b1b1b; line-height: 1.45; }}
  h1 {{ font-size: 1.6rem; }}
  h2 {{ border-bottom: 1px solid #ddd; padding-bottom: .25rem; margin-top: 2rem; }}
  section {{ margin-bottom: 1.5rem; }}
  table {{ border-collapse: collapse; width: 100%; margin: .5rem 0; }}
  th, td {{ border: 1px solid #ddd; padding: .4rem .6rem; text-align: left;
            vertical-align: top; font-size: .9rem; }}
  th {{ background: #f5f5f5; white-space: nowrap; }}
  pre {{ margin: 0; white-space: pre-wrap; word-break: break-word; font-size: .8rem; }}
  .cards {{ display: flex; gap: 1rem; flex-wrap: wrap; }}
  .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 1rem 1.25rem;
           min-width: 8rem; text-align: center; background: #fafafa; }}
  .card .val {{ font-size: 1.5rem; font-weight: 600; }}
  .card .lab {{ font-size: .8rem; color: #666; margin-top: .25rem; }}
  .muted {{ color: #999; }}
  figure {{ display: inline-block; margin: .5rem 1rem .5rem 0; }}
  figure img {{ max-width: 420px; border: 1px solid #eee; }}
  figcaption {{ font-size: .8rem; color: #666; text-align: center; }}
  footer {{ margin-top: 2rem; color: #888; font-size: .8rem; }}
</style>
</head>
<body>
<h1>Eval report</h1>
<p class="muted">run <code>{run_id}</code></p>
{body}
<footer>Generated {generated_at} by evalplatform (scaffolding). Statistics are
hand-coded in <code>evalplatform.stats</code>; unimplemented metrics show a placeholder.</footer>
</body>
</html>
"""
