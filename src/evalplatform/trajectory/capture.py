"""Phase 5 — Trajectory capture (scaffolding, AI).

A small recorder that *captures* an agent run as an ordered list of :class:`TraceStep`
plus a final answer, producing a :class:`Trace`. This is plumbing only: it records what
the agent did (thoughts, tool calls, tool results, errors, final output) so that the
hand-coded ``trajectory/score.py`` can later score the *process*. Nothing here scores,
judges, or interprets a trajectory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Optional, Union

from ..models import Trace, TraceStep, new_id


class TraceRecorder:
    """Accumulates agent steps in call order and emits an immutable :class:`Trace`.

    Usage::

        rec = TraceRecorder(system="my-agent", item_id="item_1")
        rec.thought("I should look this up")
        rec.tool_call("search", "query=...")
        rec.tool_result("...results...", name="search")
        rec.final("the answer")
        trace = rec.build()

    Each ``thought``/``tool_call``/``tool_result``/``error``/``final`` appends one step
    whose ``index`` reflects its position in the recorded order. ``final`` additionally
    sets the trace's ``final_output``. Scoring is intentionally NOT done here.
    """

    def __init__(
        self,
        *,
        system: str = "",
        item_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        self.system = system
        self.item_id = item_id
        self.trace_id = trace_id or new_id("trace")
        self.metadata: dict[str, Any] = dict(metadata or {})
        self._steps: list[TraceStep] = []
        self._final_output: Optional[str] = None

    def _add(
        self,
        kind: str,
        content: str = "",
        *,
        name: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> TraceStep:
        """Append one step; ``index`` is its 0-based position in recorded order."""
        step = TraceStep(
            index=len(self._steps),
            kind=kind,
            name=name,
            content=content,
            metadata=dict(metadata or {}),
        )
        self._steps.append(step)
        return step

    def thought(self, content: str, *, metadata: Optional[dict[str, Any]] = None) -> TraceStep:
        """Record an internal reasoning step."""
        return self._add("thought", content, metadata=metadata)

    def tool_call(
        self,
        name: str,
        content: str = "",
        *,
        metadata: Optional[dict[str, Any]] = None,
    ) -> TraceStep:
        """Record the agent invoking a tool named ``name`` (``content`` = its args)."""
        return self._add("tool_call", content, name=name, metadata=metadata)

    def tool_result(
        self,
        content: str,
        name: Optional[str] = None,
        *,
        metadata: Optional[dict[str, Any]] = None,
    ) -> TraceStep:
        """Record the result returned by a tool (``name`` = which tool, if known)."""
        return self._add("tool_result", content, name=name, metadata=metadata)

    def error(self, content: str, *, metadata: Optional[dict[str, Any]] = None) -> TraceStep:
        """Record an error encountered during the run."""
        return self._add("error", content, metadata=metadata)

    def final(self, content: str, *, metadata: Optional[dict[str, Any]] = None) -> TraceStep:
        """Record the agent's final answer and set the trace's ``final_output``."""
        step = self._add("final", content, metadata=metadata)
        self._final_output = content
        return step

    @property
    def steps(self) -> list[TraceStep]:
        """A copy of the steps recorded so far (in order)."""
        return list(self._steps)

    def build(self) -> Trace:
        """Materialize the recorded steps into a :class:`Trace`.

        Steps are already indexed in recorded order; ``final_output`` is the content of
        the last ``final`` step (or ``None`` if none was recorded).
        """
        return Trace(
            id=self.trace_id,
            item_id=self.item_id,
            system=self.system,
            steps=[s.model_copy(deep=True) for s in self._steps],
            final_output=self._final_output,
            metadata=dict(self.metadata),
        )


def trace_to_jsonl(traces: Iterable[Trace], path: Union[str, Path]) -> Path:
    """Write traces as JSON Lines (one serialized :class:`Trace` per line)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for trace in traces:
            fh.write(json.dumps(trace.model_dump(), ensure_ascii=False))
            fh.write("\n")
    return out


def load_traces(path: Union[str, Path]) -> list[Trace]:
    """Read a JSON Lines file back into :class:`Trace` objects (blank lines skipped)."""
    src = Path(path)
    traces: list[Trace] = []
    with src.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            traces.append(Trace.model_validate(json.loads(line)))
    return traces
