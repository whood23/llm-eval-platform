"""The data contract for the platform.

These Pydantic models are the single source of truth shared by the store, the run loop,
the judge, and the stats layer. The SQLite schema in ``store/schema.sql`` mirrors them.

Design note (why these fields exist): Phase 2 of the build guide validates the judge as
an instrument. To do that later you must persist *now*, on every judgment: the raw judge
response, the judge model, the prompt version, the candidate position/order, and a
timestamp. Those fields are therefore first-class here, even though the statistics that
consume them are hand-coded later.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


def _now() -> float:
    """Wall-clock seconds since epoch; centralized so it is easy to stub in tests."""
    return time.time()


def new_id(prefix: str = "") -> str:
    """A short unique id, optionally namespaced (e.g. ``new_id('run')`` -> ``run_ab12cd34``)."""
    stub = uuid.uuid4().hex[:8]
    return f"{prefix}_{stub}" if prefix else stub


class JudgeMode(str, Enum):
    """Which judging protocol a run uses."""

    pointwise = "pointwise"  # score a single candidate against a rubric
    pairwise = "pairwise"  # compare candidate A vs candidate B


class Position(str, Enum):
    """The *physical* order a pairwise comparison was presented to the judge.

    Recording this is what makes Phase-2b position-bias measurement possible: the same
    candidate pair is judged once as ``AB`` and once as ``BA``; a judge that flips its
    preference when only the order changes is exhibiting position bias.
    """

    AB = "AB"  # candidate_a shown first, candidate_b second
    BA = "BA"  # candidate_b shown first, candidate_a second
    NA = "NA"  # not applicable (pointwise)


class EvalItem(BaseModel):
    """One task/question in an eval dataset (independent of any system's answer)."""

    id: str
    input: str  # the question/task posed to the candidate system
    reference: Optional[str] = None  # gold / reference answer, if one exists
    context: Optional[str] = None  # retrieved context, for RAG-style grounded items
    stratum: Optional[str] = None  # category/segment label, used by stratified sampling
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Candidate(BaseModel):
    """A system's response to an :class:`EvalItem` — the thing actually being judged."""

    id: str
    item_id: str
    system: str  # name/version of the system or model that produced ``output``
    output: str  # the response under evaluation
    metadata: dict[str, Any] = Field(default_factory=dict)


class PointwiseJudgment(BaseModel):
    """A judge's rubric score for a single candidate."""

    id: str = Field(default_factory=lambda: new_id("pw"))
    run_id: str
    item_id: str
    candidate_id: str
    judge_model: str
    prompt_version: str
    score: Optional[float] = None  # parsed numeric rubric score
    label: Optional[str] = None  # optional categorical verdict (e.g. pass/fail)
    rationale: Optional[str] = None  # judge's free-text justification
    raw_response: Optional[str] = None  # the unparsed judge output (kept for auditing)
    parse_ok: bool = False  # did output parsing succeed?
    cached: bool = False  # served from cache rather than a fresh call?
    latency_ms: Optional[float] = None
    created_at: float = Field(default_factory=_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PairwiseJudgment(BaseModel):
    """A judge's A-vs-B verdict for one *presentation* of a candidate pair.

    A single logical comparison of candidates X and Y produces *two* rows: one shown as
    ``AB`` and one as ``BA``. ``candidate_a_id`` / ``candidate_b_id`` are the candidates
    assigned to slots A and B *in this presentation*; ``position`` records the physical
    order. ``winner_slot`` is what the judge picked ("A"/"B"/"tie"); ``winner_candidate_id``
    resolves that back to the underlying candidate so downstream bias math is unambiguous.
    """

    id: str = Field(default_factory=lambda: new_id("pr"))
    run_id: str
    item_id: str
    candidate_a_id: str  # candidate in slot A of THIS presentation
    candidate_b_id: str  # candidate in slot B of THIS presentation
    position: Position  # AB or BA
    winner_slot: Optional[str] = None  # "A" | "B" | "tie"
    winner_candidate_id: Optional[str] = None  # resolved underlying candidate id
    judge_model: str
    prompt_version: str
    rationale: Optional[str] = None
    raw_response: Optional[str] = None
    parse_ok: bool = False
    cached: bool = False
    latency_ms: Optional[float] = None
    created_at: float = Field(default_factory=_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParsedPointwise(BaseModel):
    """What the user's hand-coded ``PointwiseJudge.parse_response`` returns.

    The run loop consumes this and copies the fields onto a :class:`PointwiseJudgment`.
    """

    score: Optional[float] = None
    label: Optional[str] = None
    rationale: Optional[str] = None


class ParsedPairwise(BaseModel):
    """What the user's hand-coded ``PairwiseJudge.parse_response`` returns."""

    winner_slot: Optional[str] = None  # "A" | "B" | "tie"
    rationale: Optional[str] = None


class TraceStep(BaseModel):
    """One step in an agent trajectory (Phase 5). Captured by ``trajectory/capture.py``."""

    index: int
    kind: str  # "thought" | "tool_call" | "tool_result" | "final" | "error" | ...
    name: Optional[str] = None  # tool name, when ``kind == 'tool_call'``
    content: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class Trace(BaseModel):
    """A full agent run: the ordered steps plus the final answer (Phase 5)."""

    id: str = Field(default_factory=lambda: new_id("trace"))
    item_id: Optional[str] = None
    system: str = ""
    steps: list[TraceStep] = Field(default_factory=list)
    final_output: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RunMeta(BaseModel):
    """Provenance for a single eval run — snapshot everything needed to reproduce it."""

    run_id: str = Field(default_factory=lambda: new_id("run"))
    mode: JudgeMode
    judge_model: str
    prompt_version: str
    dataset_id: Optional[str] = None
    dataset_version: Optional[str] = None
    n_items: int = 0
    git_sha: Optional[str] = None  # commit the run was executed against
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    started_at: float = Field(default_factory=_now)
    finished_at: Optional[float] = None
    notes: Optional[str] = None
