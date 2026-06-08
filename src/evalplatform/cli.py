"""Command-line interface for llm-eval-platform v2 (scaffolding, AI).

A thin Typer app that wires the scaffolded plumbing together: it opens the store, builds
a provider, and drives the run loop / reports / gate. It never implements a statistic or a
judge rubric -- the hand-coded cores are *called* (and any ``NotImplementedError`` they
raise is surfaced as an actionable message). ``smoke`` deliberately exercises the
provider/store/cache/retry/ratelimit path WITHOUT a judge so the plumbing is testable on a
bare checkout with the offline ``dummy`` provider.
"""

from __future__ import annotations

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import typer

from .config import get_settings
from .models import EvalItem, Candidate, JudgeMode, PointwiseJudgment, RunMeta, new_id

app = typer.Typer(help="llm-eval-platform v2", no_args_is_help=True)


# --- shared helpers -------------------------------------------------------------------


def _settings_with_overrides(
    *, provider: Optional[str] = None, judge_model: Optional[str] = None
):
    """Return a settings object, applying optional CLI overrides without mutating the cache.

    ``get_settings()`` is an ``lru_cache`` singleton; we ``model_copy`` it so per-invocation
    overrides (``--provider`` / ``--judge-model``) do not leak into the cached instance.
    """
    settings = get_settings()
    overrides: dict[str, object] = {}
    if provider is not None:
        overrides["provider"] = provider
    if judge_model is not None:
        overrides["judge_model"] = judge_model
    if overrides:
        settings = settings.model_copy(update=overrides)
    return settings


# --- commands -------------------------------------------------------------------------


@app.command("init-db")
def init_db() -> None:
    """Create (or migrate) the SQLite store from the bundled schema."""
    from .store.db import Store

    settings = get_settings()
    store = Store.open(settings)
    store.close()
    typer.echo(f"initialized store at {settings.db_path}")


@app.command()
def version() -> None:
    """Print the installed package version."""
    try:
        from importlib.metadata import version as _pkg_version

        ver = _pkg_version("llm-eval-platform")
    except Exception:  # noqa: BLE001 - fall back if metadata is unavailable (source tree)
        ver = "unknown"
    typer.echo(f"llm-eval-platform {ver}")


@app.command()
def smoke(
    n: int = typer.Option(5, "--n", help="number of synthetic items/candidates to send"),
) -> None:
    """Plumbing check: provider + store + cache + retry + ratelimit, WITHOUT the judge.

    Sends a fixed diagnostic prompt (not a rubric) for ``n`` synthetic candidates through
    the retry wrapper + rate limiter in a thread pool, persists each as a smoke-tagged
    ``PointwiseJudgment``, then re-runs to demonstrate cache hits.
    """
    from .judge.cache import JudgmentCache
    from .judge.ratelimit import RateLimiter
    from .judge.retry import retry_call
    from .providers.base import get_provider
    from .store.db import Store

    settings = get_settings()
    provider = get_provider(settings)
    store = Store.open(settings)

    SMOKE_MODEL = "__smoke__"
    SMOKE_PROMPT_VERSION = "__smoke__"

    try:
        cache = JudgmentCache(store, enabled=settings.cache_enabled)
        rate_limiter = RateLimiter(settings.rpm_limit)

        # Synthetic dataset: one candidate per item. Persisted so FK constraints hold and
        # the cache-key lookups have real rows to hit on the second pass.
        run = RunMeta(
            run_id=new_id("run"),
            mode=JudgeMode.pointwise,
            judge_model=SMOKE_MODEL,
            prompt_version=SMOKE_PROMPT_VERSION,
            notes="smoke: plumbing check (no judge)",
            config_snapshot={"provider": provider.name, "n": n},
        )
        store.insert_run(run)

        items: list[EvalItem] = []
        candidates: list[Candidate] = []
        for i in range(n):
            item = EvalItem(id=new_id("smk_item"), input=f"smoke item {i}")
            cand = Candidate(
                id=new_id("smk_cand"),
                item_id=item.id,
                system="__smoke__",
                output=f"smoke output {i}",
            )
            items.append(item)
            candidates.append(cand)
        store.upsert_items(items)
        store.upsert_candidates(candidates)

        # The Store shares one sqlite3 connection across threads (check_same_thread=False).
        # A single connection is NOT safe for *concurrent* execute/commit from multiple
        # worker threads (sqlite3 raises "bad parameter or other API misuse"), so serialize
        # all store access here while keeping the provider calls themselves concurrent.
        db_lock = threading.Lock()

        def _do_call(cand: Candidate) -> PointwiseJudgment:
            """One smoke unit of work: cache lookup, else a provider call via retry+ratelimit."""
            with db_lock:
                cached = cache.get_pointwise(cand.id, SMOKE_MODEL, SMOKE_PROMPT_VERSION)
            if cached is not None:
                # Persist a cached copy (idempotent on the cache key) so the result reflects
                # that no fresh provider call was needed.
                copy = cached.model_copy(update={"run_id": run.run_id, "cached": True})
                with db_lock:
                    store.insert_pointwise(copy)
                return copy

            prompt = f"plumbing smoke: echo id {cand.id}"
            rate_limiter.acquire()
            resp = retry_call(
                lambda: provider.complete(
                    prompt,
                    temperature=settings.temperature,
                    timeout=settings.request_timeout,
                ),
                max_retries=settings.max_retries,
            )
            judgment = PointwiseJudgment(
                run_id=run.run_id,
                item_id=cand.item_id,
                candidate_id=cand.id,
                judge_model=SMOKE_MODEL,
                prompt_version=SMOKE_PROMPT_VERSION,
                parse_ok=False,  # smoke never parses a rubric verdict
                cached=False,
                raw_response=resp.text,
                latency_ms=resp.latency_ms,
            )
            with db_lock:
                store.insert_pointwise(judgment)
            return judgment

        # First pass: fresh calls (some may already be cached from a prior smoke run).
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=settings.max_concurrency) as pool:
            first = list(pool.map(_do_call, candidates))
        total_latency = sum(j.latency_ms or 0.0 for j in first)
        n_cached_first = sum(1 for j in first if j.cached)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        typer.echo(
            f"smoke pass 1: provider={provider.name} n={len(first)} "
            f"cached={n_cached_first} total_latency_ms={total_latency:.1f} "
            f"wall_ms={elapsed_ms:.1f}"
        )

        # Second pass: identical inputs -> every unit should hit the cache (cached=True),
        # provided caching is enabled.
        with ThreadPoolExecutor(max_workers=settings.max_concurrency) as pool:
            second = list(pool.map(_do_call, candidates))
        n_cached_second = sum(1 for j in second if j.cached)
        typer.echo(
            f"smoke pass 2: n={len(second)} cached={n_cached_second}"
            + ("" if settings.cache_enabled else "  (cache disabled in settings)")
        )

        store.finish_run(run.run_id, finished_at=time.time(), n_items=len(items))
        typer.echo(f"smoke OK -> run_id={run.run_id}")
    finally:
        store.close()


@app.command()
def run(
    mode: str = typer.Option(
        "pointwise", "--mode", help="judging protocol: pointwise | pairwise"
    ),
    items: Path = typer.Option(..., "--items", help="path to items JSONL"),
    candidates: Path = typer.Option(
        ..., "--candidates", help="path to candidates JSONL"
    ),
    provider: Optional[str] = typer.Option(
        None, "--provider", help="override settings.provider (dummy | litellm)"
    ),
    judge_model: Optional[str] = typer.Option(
        None, "--judge-model", help="override settings.judge_model"
    ),
) -> None:
    """Run the judge over a dataset and persist judgments to the store.

    The judge rubric/parser (and pairwise swap logic) are hand-coded stubs. If they are not
    yet implemented the run loop raises ``JudgeNotImplemented``; this command turns that into
    a clean, actionable message and exits 1 (use ``eval-platform smoke`` to test plumbing).
    """
    from .datasets.loader import candidates_by_item, load_candidates, load_items
    from .judge.cache import JudgmentCache
    from .judge.ratelimit import RateLimiter
    from .judge.runner import JudgeNotImplemented, Runner
    from .providers.base import get_provider
    from .store.db import Store

    try:
        mode_enum = JudgeMode(mode)
    except ValueError:
        typer.echo(
            f"unknown --mode {mode!r}; expected 'pointwise' or 'pairwise'", err=True
        )
        raise typer.Exit(2)

    settings = _settings_with_overrides(provider=provider, judge_model=judge_model)
    eval_items = load_items(items)
    cands_by_item = candidates_by_item(load_candidates(candidates))

    store = Store.open(settings)
    try:
        prov = get_provider(settings)
        cache = JudgmentCache(store, enabled=settings.cache_enabled)
        rate_limiter = RateLimiter(settings.rpm_limit)
        runner = Runner(
            settings=settings,
            store=store,
            provider=prov,
            cache=cache,
            rate_limiter=rate_limiter,
        )

        try:
            if mode_enum is JudgeMode.pointwise:
                from .judge.pointwise import PointwiseJudge

                judge = PointwiseJudge(prompt_version=settings.prompt_version)
                run_meta = runner.run_pointwise(eval_items, cands_by_item, judge)
            else:
                from .judge.pairwise import PairwiseJudge

                judge = PairwiseJudge(prompt_version=settings.prompt_version)
                run_meta = runner.run_pairwise(eval_items, cands_by_item, judge)
        except JudgeNotImplemented as exc:
            typer.echo("judge is not yet implemented -- nothing was judged.", err=True)
            typer.echo(f"  {exc}", err=True)
            typer.echo(
                "  Hand-code the judge rubric/parser (and pairwise swap logic) in "
                "src/evalplatform/judge/, then re-run.",
                err=True,
            )
            typer.echo(
                "  To test the provider/store/cache plumbing meanwhile, run: "
                "eval-platform smoke",
                err=True,
            )
            raise typer.Exit(1)

        typer.echo(
            f"run complete: run_id={run_meta.run_id} mode={mode_enum.value} "
            f"provider={prov.name} judge_model={settings.judge_model} "
            f"n_items={run_meta.n_items}"
        )
    finally:
        store.close()


@app.command("verify-stats")
def verify_stats(
    strict: bool = typer.Option(
        False, "--strict", help="also fail while any stat remains a TODO stub"
    ),
) -> None:
    """Run the hand-coded stats verifier (handcode/verify_stats.py) and pass through its exit code."""
    # config.REPO_ROOT is .../llm-eval-platform (parents[2] of this file's dir lineage).
    from .config import REPO_ROOT

    repo_root = Path(REPO_ROOT)
    handcode_dir = repo_root / "handcode"
    verifier = handcode_dir / "verify_stats.py"
    if not verifier.exists():
        typer.echo(f"verifier not found at {verifier}", err=True)
        raise typer.Exit(2)

    # The verifier is a script (not a package module); import it by file path. Add the repo
    # root (and handcode dir) to sys.path so its own ``import evalplatform`` and module load work.
    import importlib.util

    for p in (str(repo_root), str(handcode_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)

    # Preserve the user's --strict by patching argv the verifier inspects.
    saved_argv = sys.argv
    sys.argv = [str(verifier)] + (["--strict"] if strict else [])
    try:
        spec = importlib.util.spec_from_file_location("evalplatform_verify_stats", verifier)
        if spec is None or spec.loader is None:
            typer.echo(f"could not load verifier module from {verifier}", err=True)
            raise typer.Exit(2)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        code = int(module.main())
    finally:
        sys.argv = saved_argv

    raise typer.Exit(code)


@app.command()
def report(
    run_id: Optional[str] = typer.Option(
        None, "--run-id", help="run to report on (defaults to the latest run)"
    ),
    out: Optional[Path] = typer.Option(
        None, "--out", help="output directory for the HTML report"
    ),
) -> None:
    """Build a self-contained static HTML report for a run."""
    from .report.report import build_report
    from .store.db import Store

    settings = get_settings()
    store = Store.open(settings)
    try:
        target = run_id
        if target is None:
            latest = store.latest_run()
            if latest is None:
                typer.echo("no runs found in the store; run one first.", err=True)
                raise typer.Exit(1)
            target = latest.run_id
            typer.echo(f"no --run-id given; using latest run {target}")

        path = build_report(store, target, out_dir=out)
        typer.echo(f"report written to {path}")
    finally:
        store.close()


@app.command()
def gate(
    current: Path = typer.Option(..., "--current", help="path to current metrics JSON"),
    baseline: Path = typer.Option(
        ..., "--baseline", help="path to baseline metrics JSON"
    ),
    thresholds: Optional[Path] = typer.Option(
        None, "--thresholds", help="optional per-metric threshold JSON"
    ),
) -> None:
    """Compare current metrics against a baseline; exit non-zero on regression."""
    from .gate.regression_gate import run_gate_cli

    code = run_gate_cli(current, baseline, thresholds)
    raise typer.Exit(code)


@app.command()
def dashboard() -> None:
    """Launch the Streamlit dashboard (requires the optional 'dashboard' extra)."""
    import importlib.util
    import subprocess

    if importlib.util.find_spec("streamlit") is None:
        typer.echo(
            "streamlit is not installed. Install it with: pip install '.[dashboard]'",
            err=True,
        )
        raise typer.Exit(1)

    # Launch the dashboard module as a Streamlit script via the current interpreter.
    dashboard_path = Path(__file__).parent / "report" / "dashboard.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(dashboard_path)]
    typer.echo(f"launching: {' '.join(cmd)}")
    raise typer.Exit(subprocess.call(cmd))


if __name__ == "__main__":
    app()
