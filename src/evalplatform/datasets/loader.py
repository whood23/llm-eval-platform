"""JSONL load/write helpers for eval items + candidates (scaffolding, AI).

These are pure I/O conveniences: they read/write the :class:`EvalItem` and
:class:`Candidate` Pydantic models to newline-delimited JSON. No sampling or
versioning logic lives here (see ``sampling`` and ``versioning``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Union

from ..models import Candidate, EvalItem

PathLike = Union[str, Path]


def load_jsonl(path: PathLike) -> list[dict]:
    """Read a UTF-8 JSONL file into a list of dicts, skipping blank lines."""
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(rows: Iterable[dict[str, Any]], path: PathLike) -> Path:
    """Write an iterable of dicts as JSONL (one compact JSON object per line)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False))
            fh.write("\n")
    return out


def load_items(path: PathLike) -> list[EvalItem]:
    """Load eval items from JSONL into validated :class:`EvalItem` models."""
    return [EvalItem(**row) for row in load_jsonl(path)]


def load_candidates(path: PathLike) -> list[Candidate]:
    """Load candidates from JSONL into validated :class:`Candidate` models."""
    return [Candidate(**row) for row in load_jsonl(path)]


def candidates_by_item(cands: Iterable[Candidate]) -> dict[str, list[Candidate]]:
    """Group candidates by their ``item_id`` (preserving input order per item)."""
    grouped: dict[str, list[Candidate]] = {}
    for c in cands:
        grouped.setdefault(c.item_id, []).append(c)
    return grouped


def write_items(items: Iterable[EvalItem], path: PathLike) -> Path:
    """Serialize :class:`EvalItem` models to JSONL."""
    return write_jsonl((item.model_dump() for item in items), path)


def write_candidates(cands: Iterable[Candidate], path: PathLike) -> Path:
    """Serialize :class:`Candidate` models to JSONL."""
    return write_jsonl((c.model_dump() for c in cands), path)
