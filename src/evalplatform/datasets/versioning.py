"""Content-hash dataset versions + human-readable dataset cards (scaffolding, AI).

A dataset's *version* is a stable hash of its canonicalized contents, so the same
items always produce the same id regardless of ordering — making runs reproducible
and diffable. The *dataset card* is a small Markdown artifact summarizing the set
(counts + per-stratum breakdown). Stratified sampling is NOT here — that is the
hand-coded ``datasets/sampling.py``.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from ..models import EvalItem

PathLike = Union[str, Path]


def dataset_version(items: Sequence[EvalItem]) -> str:
    """Stable short content hash for a set of items.

    The items are dumped to dicts, serialized with sorted keys, sorted by their
    canonical JSON string (so ordering of the input is irrelevant), then hashed with
    sha256. Returns ``'ds_' + first 12 hex chars``.
    """
    canon = sorted(
        json.dumps(item.model_dump(), sort_keys=True, ensure_ascii=False)
        for item in items
    )
    payload = json.dumps(canon, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return "ds_" + digest[:12]


def _stratum_counts(items: Sequence[EvalItem]) -> dict[str, int]:
    """Count items per stratum (``None`` stratum bucketed as ``'(none)'``)."""
    counts = Counter(item.stratum or "(none)" for item in items)
    return dict(sorted(counts.items()))


def write_dataset_card(
    items: Sequence[EvalItem],
    path: PathLike,
    *,
    name: str,
    version: Optional[str] = None,
    notes: Optional[str] = None,
) -> dict[str, Any]:
    """Write a Markdown dataset card and return its summary dict.

    Args:
        items: the dataset to describe.
        path: where to write the ``.md`` card.
        name: human-readable dataset name.
        version: dataset version; computed via :func:`dataset_version` if omitted.
        notes: optional free-text notes appended to the card.

    Returns:
        The summary dict embedded in the card (name, version, counts, strata, ...).
    """
    version = version or dataset_version(items)
    strata = _stratum_counts(items)
    n_with_reference = sum(1 for it in items if it.reference)
    n_with_context = sum(1 for it in items if it.context)

    summary: dict[str, Any] = {
        "name": name,
        "version": version,
        "n_items": len(items),
        "strata": strata,
        "n_with_reference": n_with_reference,
        "n_with_context": n_with_context,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    if notes:
        summary["notes"] = notes

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        f"# Dataset card: {name}",
        "",
        f"- **Version:** `{version}`",
        f"- **Items:** {len(items)}",
        f"- **With reference:** {n_with_reference}",
        f"- **With context:** {n_with_context}",
        f"- **Generated at:** {summary['generated_at']}",
        "",
        "## Per-stratum breakdown",
        "",
        "| Stratum | Count |",
        "| --- | ---: |",
    ]
    for stratum, count in strata.items():
        lines.append(f"| {stratum} | {count} |")
    if notes:
        lines += ["", "## Notes", "", notes]
    lines.append("")  # trailing newline

    out.write_text("\n".join(lines), encoding="utf-8")
    return summary
