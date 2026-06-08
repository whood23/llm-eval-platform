"""The run loop (scaffolding, AI): batching, concurrency, cache, retry, rate-limit, persist.

This is the orchestration glue that turns a dataset + a *hand-coded* judge into persisted
judgments. It owns nothing statistical and implements none of the judge's rubric/parsing/swap
logic — it merely *calls* the user's :class:`PointwiseJudge` / :class:`PairwiseJudge` methods
(``build_prompt``, ``parse_response``, ``iter_presentations``) and wraps each provider call in
the already-built reliability primitives (``retry_call``, ``RateLimiter``, ``JudgmentCache``,
``Store``).

When the judge methods are still stubs they raise ``NotImplementedError``; the runner detects
that up front and re-raises a clean :class:`JudgeNotImplemented` naming the file/method to fill
in. It never "helpfully completes" a judge method.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional

from ..config import Settings
from ..models import (
    Candidate,
    EvalItem,
    JudgeMode,
    PairwiseJudgment,
    ParsedPairwise,
    ParsedPointwise,
    Position,
    RunMeta,
    new_id,
)
from ..providers.base import JudgeProvider, ProviderResponse
from ..store.db import Store
from .cache import JudgmentCache
from .pairwise import PairwiseJudge, Presentation
from .pointwise import PointwiseJudge
from .ratelimit import RateLimiter
from .retry import retry_call


class JudgeNotImplemented(RuntimeError):
    """Raised when a hand-coded judge method is still a stub.

    The message names the exact file/method the user must implement; the CLI turns this into
    an actionable hint (and suggests ``eval-platform smoke`` to exercise the plumbing alone).
    """


def _now() -> float:
    return time.time()


class Runner:
    """Executes a judge over a dataset with concurrency, caching, retries and rate limiting."""

    def __init__(
        self,
        *,
        settings: Settings,
        store: Store,
        provider: JudgeProvider,
        cache: Optional[JudgmentCache] = None,
        rate_limiter: Optional[RateLimiter] = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.provider = provider
        self.cache = cache
        self.rate_limiter = rate_limiter

    # --- public entry points ------------------------------------------------------------

    def run_pointwise(
        self,
        items: list[EvalItem],
        candidates: dict[str, list[Candidate]],
        judge: PointwiseJudge,
        *,
        run: Optional[RunMeta] = None,
    ) -> RunMeta:
        """Score every candidate of every item with the pointwise ``judge``.

        Persists one :class:`PointwiseJudgment` per candidate; cache hits are copied through
        with ``cached=True`` and no fresh provider call.
        """
        run = self._prepare_run(run, judge, JudgeMode.pointwise, items, candidates)

        # Build the flat task list (one task per candidate of each item).
        tasks: list[tuple[EvalItem, Candidate]] = []
        for item in items:
            for cand in candidates.get(item.id, []):
                tasks.append((item, cand))

        # Probe the judge stub once so unimplemented methods fail fast and cleanly,
        # rather than surfacing NotImplementedError from inside worker threads.
        if tasks:
            self._probe_pointwise(judge, tasks[0][0], tasks[0][1])

        self._run_tasks(tasks, lambda t: self._judge_one_pointwise(run, judge, t[0], t[1]))

        self.store.finish_run(run.run_id, finished_at=_now(), n_items=len(items))
        run.finished_at = _now()
        run.n_items = len(items)
        return run

    def run_pairwise(
        self,
        items: list[EvalItem],
        candidates: dict[str, list[Candidate]],
        judge: PairwiseJudge,
        *,
        run: Optional[RunMeta] = None,
    ) -> RunMeta:
        """Compare candidate pairs with the pairwise ``judge``, in both orders.

        The swap logic (which presentations to emit, in which order) is the user's
        ``iter_presentations`` stub; the runner only executes whatever it yields and resolves
        the chosen slot back to the underlying candidate id.
        """
        run = self._prepare_run(run, judge, JudgeMode.pairwise, items, candidates)

        # Expand each item into its presentations via the user's swap harness. This call can
        # raise NotImplementedError (stub) — translate it up front into JudgeNotImplemented.
        tasks: list[tuple[EvalItem, Presentation]] = []
        for item in items:
            cands = candidates.get(item.id, [])
            try:
                presentations = list(judge.iter_presentations(item, cands))
            except NotImplementedError as exc:
                raise self._not_implemented(
                    "iter_presentations", "src/evalplatform/judge/pairwise.py", exc
                ) from exc
            for pres in presentations:
                tasks.append((item, pres))

        # Probe build_prompt/parse_response once so stub methods fail fast and cleanly.
        if tasks:
            self._probe_pairwise(judge, tasks[0][0], tasks[0][1])

        self._run_tasks(tasks, lambda t: self._judge_one_pairwise(run, judge, t[0], t[1]))

        self.store.finish_run(run.run_id, finished_at=_now(), n_items=len(items))
        run.finished_at = _now()
        run.n_items = len(items)
        return run

    # --- run setup ----------------------------------------------------------------------

    def _prepare_run(
        self,
        run: Optional[RunMeta],
        judge: Any,
        mode: JudgeMode,
        items: list[EvalItem],
        candidates: dict[str, list[Candidate]],
    ) -> RunMeta:
        """Create/insert the RunMeta and upsert the dataset (items + candidates)."""
        if run is None:
            run = RunMeta(
                run_id=new_id("run"),
                mode=mode,
                judge_model=self.settings.judge_model,
                prompt_version=judge.prompt_version,
                n_items=len(items),
            )
        else:
            # Stamp the canonical fields even if the caller supplied a partial RunMeta.
            run.mode = mode
            run.judge_model = self.settings.judge_model
            run.prompt_version = judge.prompt_version
            run.n_items = len(items)
        self.store.insert_run(run)

        self.store.upsert_items(items)
        all_cands: list[Candidate] = []
        for cs in candidates.values():
            all_cands.extend(cs)
        if all_cands:
            self.store.upsert_candidates(all_cands)
        return run

    # --- concurrency --------------------------------------------------------------------

    def _run_tasks(self, tasks: list[Any], work: Callable[[Any], None]) -> None:
        """Run ``work(task)`` over ``tasks`` in a bounded thread pool.

        ``max_workers`` is clamped to at least 1 and never exceeds the task count. Each task's
        result is consumed so any exception raised in a worker (notably
        :class:`JudgeNotImplemented`) propagates to the caller.
        """
        if not tasks:
            return
        workers = max(1, min(self.settings.max_concurrency, len(tasks)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(work, task) for task in tasks]
            for fut in futures:
                fut.result()  # surface worker exceptions (e.g. JudgeNotImplemented)

    # --- provider call (rate-limit + retry) ---------------------------------------------

    def _call_provider(
        self, prompt: str, *, system: Optional[str] = None
    ) -> ProviderResponse:
        """One reliability-wrapped provider call: rate-limit, then retry with backoff."""
        if self.rate_limiter is not None:
            self.rate_limiter.acquire()

        def _do() -> ProviderResponse:
            return self.provider.complete(
                prompt,
                system=system,
                temperature=self.settings.temperature,
                timeout=self.settings.request_timeout,
            )

        return retry_call(_do, max_retries=self.settings.max_retries)

    # --- pointwise work -----------------------------------------------------------------

    def _judge_one_pointwise(
        self,
        run: RunMeta,
        judge: PointwiseJudge,
        item: EvalItem,
        cand: Candidate,
    ) -> None:
        """Judge a single candidate and persist the :class:`PointwiseJudgment`."""
        judge_model = self.settings.judge_model
        prompt_version = run.prompt_version

        # Cache: a prior identical judgment short-circuits the provider call.
        if self.cache is not None:
            hit = self.cache.get_pointwise(cand.id, judge_model, prompt_version)
            if hit is not None:
                cached = hit.model_copy(
                    update={
                        "id": new_id("pw"),
                        "run_id": run.run_id,
                        "cached": True,
                        "created_at": _now(),
                    }
                )
                self.store.insert_pointwise(cached)
                return

        prompt = judge.build_prompt(item, cand)  # probed earlier; assumed implemented now
        resp = self._call_provider(prompt)

        parsed = ParsedPointwise()
        parse_ok = False
        try:
            parsed = judge.parse_response(resp.text)
            parse_ok = True
        except ValueError:
            # Unparseable judge output: record the failure, keep the raw response for auditing.
            parse_ok = False

        judgment = self._make_pointwise(
            run=run,
            item=item,
            cand=cand,
            parsed=parsed,
            parse_ok=parse_ok,
            resp=resp,
        )
        self.store.insert_pointwise(judgment)

    def _make_pointwise(
        self,
        *,
        run: RunMeta,
        item: EvalItem,
        cand: Candidate,
        parsed: ParsedPointwise,
        parse_ok: bool,
        resp: ProviderResponse,
    ):
        from ..models import PointwiseJudgment

        return PointwiseJudgment(
            id=new_id("pw"),
            run_id=run.run_id,
            item_id=item.id,
            candidate_id=cand.id,
            judge_model=self.settings.judge_model,
            prompt_version=run.prompt_version,
            score=parsed.score if parse_ok else None,
            label=parsed.label if parse_ok else None,
            rationale=parsed.rationale if parse_ok else None,
            raw_response=resp.text,
            parse_ok=parse_ok,
            cached=False,
            latency_ms=resp.latency_ms,
        )

    # --- pairwise work ------------------------------------------------------------------

    def _judge_one_pairwise(
        self,
        run: RunMeta,
        judge: PairwiseJudge,
        item: EvalItem,
        pres: Presentation,
    ) -> None:
        """Judge a single A-vs-B presentation and persist the :class:`PairwiseJudgment`."""
        judge_model = self.settings.judge_model
        prompt_version = run.prompt_version
        a_id, b_id = pres.cand_a.id, pres.cand_b.id

        # Cache keyed on (a_id, b_id, position, judge_model, prompt_version).
        if self.cache is not None:
            hit = self.cache.get_pairwise(
                a_id, b_id, pres.position, judge_model, prompt_version
            )
            if hit is not None:
                cached = hit.model_copy(
                    update={
                        "id": new_id("pr"),
                        "run_id": run.run_id,
                        "cached": True,
                        "created_at": _now(),
                    }
                )
                self.store.insert_pairwise(cached)
                return

        prompt = judge.build_prompt(item, pres.cand_a, pres.cand_b)  # probed earlier
        resp = self._call_provider(prompt)

        parsed = ParsedPairwise()
        parse_ok = False
        try:
            parsed = judge.parse_response(resp.text)
            parse_ok = True
        except ValueError:
            parse_ok = False

        winner_slot = parsed.winner_slot if parse_ok else None
        winner_candidate_id = self._resolve_winner(winner_slot, pres) if parse_ok else None

        judgment = self._make_pairwise(
            run=run,
            item=item,
            pres=pres,
            winner_slot=winner_slot,
            winner_candidate_id=winner_candidate_id,
            rationale=parsed.rationale if parse_ok else None,
            parse_ok=parse_ok,
            resp=resp,
        )
        self.store.insert_pairwise(judgment)

    @staticmethod
    def _resolve_winner(
        winner_slot: Optional[str], pres: Presentation
    ) -> Optional[str]:
        """Map the judge's chosen slot ("A"/"B"/"tie") back to the underlying candidate id.

        ``pres.cand_a`` occupies slot A and ``pres.cand_b`` slot B *as presented*, so the
        resolution is order-aware regardless of whether this is the AB or BA presentation.
        A tie (or any non-A/B slot) resolves to ``None``.
        """
        if winner_slot is None:
            return None
        slot = winner_slot.strip().upper()
        if slot == "A":
            return pres.cand_a.id
        if slot == "B":
            return pres.cand_b.id
        return None  # "tie" / unknown -> no single winner

    def _make_pairwise(
        self,
        *,
        run: RunMeta,
        item: EvalItem,
        pres: Presentation,
        winner_slot: Optional[str],
        winner_candidate_id: Optional[str],
        rationale: Optional[str],
        parse_ok: bool,
        resp: ProviderResponse,
    ) -> PairwiseJudgment:
        return PairwiseJudgment(
            id=new_id("pr"),
            run_id=run.run_id,
            item_id=item.id,
            candidate_a_id=pres.cand_a.id,
            candidate_b_id=pres.cand_b.id,
            position=pres.position if isinstance(pres.position, Position) else Position(pres.position),
            winner_slot=winner_slot,
            winner_candidate_id=winner_candidate_id,
            judge_model=self.settings.judge_model,
            prompt_version=run.prompt_version,
            rationale=rationale,
            raw_response=resp.text,
            parse_ok=parse_ok,
            cached=False,
            latency_ms=resp.latency_ms,
        )

    # --- stub probing / graceful NotImplemented handling --------------------------------

    def _probe_pointwise(
        self, judge: PointwiseJudge, item: EvalItem, cand: Candidate
    ) -> None:
        """Call build_prompt/parse_response once to fail fast if either is still a stub.

        We probe with the first real task so the prompt is well-formed. ``parse_response`` is
        probed with the *actual* prompt text rather than a fake reply: a stub raises
        ``NotImplementedError`` (caught here), while an implemented parser raising ``ValueError``
        on unexpected input is fine and is swallowed.
        """
        try:
            prompt = judge.build_prompt(item, cand)
        except NotImplementedError as exc:
            raise self._not_implemented(
                "build_prompt", "src/evalplatform/judge/pointwise.py", exc
            ) from exc

        try:
            judge.parse_response(prompt)
        except NotImplementedError as exc:
            raise self._not_implemented(
                "parse_response", "src/evalplatform/judge/pointwise.py", exc
            ) from exc
        except ValueError:
            # An implemented parser may legitimately reject this probe input — that's fine.
            pass

    def _probe_pairwise(
        self, judge: PairwiseJudge, item: EvalItem, pres: Presentation
    ) -> None:
        """Call build_prompt/parse_response once to fail fast if either is still a stub."""
        try:
            prompt = judge.build_prompt(item, pres.cand_a, pres.cand_b)
        except NotImplementedError as exc:
            raise self._not_implemented(
                "build_prompt", "src/evalplatform/judge/pairwise.py", exc
            ) from exc

        try:
            judge.parse_response(prompt)
        except NotImplementedError as exc:
            raise self._not_implemented(
                "parse_response", "src/evalplatform/judge/pairwise.py", exc
            ) from exc
        except ValueError:
            pass

    @staticmethod
    def _not_implemented(
        method: str, file: str, exc: BaseException
    ) -> "JudgeNotImplemented":
        """Build a clean, actionable JudgeNotImplemented for an unimplemented judge stub."""
        return JudgeNotImplemented(
            f"{file}:{method} is not implemented yet (Phase 1 hand-code). "
            f"Implement it, or run 'eval-platform smoke' to test the plumbing without a judge. "
            f"Original: {exc}"
        )
