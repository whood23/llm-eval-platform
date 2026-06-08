"""Phase 5 — Trajectory scoring.   *** YOU HAND-CODE THIS. ***

Given a captured agent :class:`Trace` (steps + tool calls, recorded by ``capture.py``), score
the *process*, not just the final answer: were the right tools called, in a sensible order,
did the agent recover from errors — and *which step* is to blame when it failed. This is the
logic agent-eval screens probe; it's yours.
"""

from __future__ import annotations

from typing import Any, Optional

from ..models import Trace


def score_trajectory(trace: Trace, rubric: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Score one agent trajectory and localize its failing step.

    Args:
        trace: the captured run.
        rubric: optional knobs for your scoring scheme (expected tools, weights, etc.).

    Returns:
        a dict you define — e.g. ``{"score": float, "failed_step_index": int | None,
        "reasons": [...], "per_step": [...]}``. Keep it serializable so it can be persisted
        and reported.

    Hint: walk ``trace.steps``; reward correct/necessary tool calls and successful recovery,
        penalize wrong/redundant calls and unrecovered errors; when it fails, return the index
        of the earliest step that caused the failure.
    """
    raise NotImplementedError(
        "Hand-code trajectory scoring (Phase 5). See the docstring."
    )
