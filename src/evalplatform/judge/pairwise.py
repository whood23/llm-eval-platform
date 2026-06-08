"""Phase 1 — Pairwise judge: the swap harness + rubric prompt + parser. *** YOU HAND-CODE. ***

This is where position bias is *designed against*: ``iter_presentations`` must present every
candidate pair in BOTH orders (AB and BA) so Phase-2b can measure preference flips. The run
loop (scaffolded) executes whatever presentations you emit — with caching/retries/batching —
and persists each as a :class:`PairwiseJudgment` (recording its ``position``).
"""

from __future__ import annotations

from typing import Iterable, NamedTuple, Sequence

from ..models import Candidate, EvalItem, ParsedPairwise, Position


class Presentation(NamedTuple):
    """One pair shown to the judge in a specific physical order.

    ``cand_a`` fills slot A and ``cand_b`` fills slot B *as presented*; ``position`` records
    whether this is the AB or BA ordering of the underlying pair.
    """

    cand_a: Candidate
    cand_b: Candidate
    position: Position


class PairwiseJudge:
    """Compares candidate A vs candidate B for the same item, in both orders."""

    def __init__(self, prompt_version: str = "v1") -> None:
        self.prompt_version = prompt_version

    def iter_presentations(
        self, item: EvalItem, candidates: Sequence[Candidate]
    ) -> Iterable[Presentation]:
        """Yield the presentations to judge for one item — *both* orders of each pair.

        THE swap logic. For each unordered pair of candidates, emit two Presentations: one
        with ``position=Position.AB`` and one with the candidates swapped and
        ``position=Position.BA``. This is what makes position bias measurable downstream.
        """
        raise NotImplementedError(
            "Hand-code the pairwise swap logic: emit BOTH AB and BA per pair (Phase 1)."
        )

    def build_prompt(self, item: EvalItem, cand_a: Candidate, cand_b: Candidate) -> str:
        """Return the prompt asking the judge to pick between slot A and slot B for ``item``.

        Design the comparison rubric; ask for a parseable verdict (e.g. JSON ``{"winner":
        "A"|"B"|"tie", "rationale": ...}``). Refer to A/B by slot, never by system name, so
        the only thing that changes between AB and BA presentations is the order.
        """
        raise NotImplementedError(
            "Hand-code your pairwise rubric prompt (Phase 1). See the docstring."
        )

    def parse_response(self, raw: str) -> ParsedPairwise:
        """Parse the judge's reply into a :class:`ParsedPairwise` (``winner_slot`` in A/B/tie).

        Raise ``ValueError`` on an unparseable reply so the runner records ``parse_ok=False``.
        """
        raise NotImplementedError(
            "Hand-code your pairwise output parser (Phase 1). See the docstring."
        )
