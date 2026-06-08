"""Phase 1 — Pointwise judge: rubric prompt + output parser.   *** YOU HAND-CODE THIS. ***

The run loop (``judge/runner.py``, scaffolded) calls ``build_prompt`` to get the text to
send to the provider, then ``parse_response`` to turn the judge's reply into a structured
score. Designing the rubric and writing a robust parser are core eval skills you'll be
quizzed on, so they're yours. The plumbing around them (retries, caching, batching,
concurrency, persistence) is already built.

When these are still stubs, ``eval-platform run --mode pointwise`` will tell you exactly
what to implement; ``eval-platform smoke`` exercises the provider/store/cache plumbing
without needing a rubric.
"""

from __future__ import annotations

from ..models import Candidate, EvalItem, ParsedPointwise


class PointwiseJudge:
    """Scores a single candidate response against a rubric."""

    def __init__(self, prompt_version: str = "v1") -> None:
        self.prompt_version = prompt_version

    def build_prompt(self, item: EvalItem, candidate: Candidate) -> str:
        """Return the full judge prompt that scores ``candidate.output`` for ``item``.

        Design the rubric here: define the scoring scale, the criteria, and (recommended)
        ask the judge to return a parseable structured verdict (e.g. JSON with ``score`` and
        ``rationale``) so ``parse_response`` is robust. Use ``item.reference`` /
        ``item.context`` if present.
        """
        raise NotImplementedError(
            "Hand-code your pointwise rubric prompt (Phase 1). See the docstring."
        )

    def parse_response(self, raw: str) -> ParsedPointwise:
        """Parse the judge's raw reply into a :class:`ParsedPointwise`.

        Be defensive: judges wrap JSON in prose, add code fences, or drift format. On an
        unparseable reply, raise ``ValueError`` (the runner records it as ``parse_ok=False``
        and keeps the raw response) rather than guessing.
        """
        raise NotImplementedError(
            "Hand-code your pointwise output parser (Phase 1). See the docstring."
        )
