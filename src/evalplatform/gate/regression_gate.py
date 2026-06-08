"""CI regression gate (Phase 6) — scaffolding (AI).

Pure comparison/threshold logic over metric dicts: take a *current* set of metrics and a
stored *baseline*, and decide whether anything regressed beyond an allowed threshold. The
metrics being compared are produced by the user's hand-coded stats layer; nothing here
computes a statistic — this module only diffs two flat ``{name: value}`` dictionaries and
applies per-metric thresholds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

# A metric value we know how to compare numerically.
Number = Union[int, float]


@dataclass
class GateResult:
    """Outcome of a gate comparison.

    ``passed`` is True iff no compared metric regressed beyond its threshold.
    ``regressions`` holds human-readable lines describing each violation.
    ``details`` maps each compared metric name to a per-metric record (current/baseline/
    delta/threshold/regressed) so callers can render or serialize the full diff.
    """

    passed: bool
    regressions: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


def _as_number(value: Any) -> Optional[float]:
    """Coerce ``value`` to float if it is a real number, else None (skip from comparison)."""
    # bool is a subclass of int; treat it as numeric (True/False -> 1.0/0.0).
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def gate(
    current: dict[str, Any],
    baseline: dict[str, Any],
    *,
    thresholds: Optional[dict[str, Number]] = None,
    higher_is_better: bool = True,
) -> GateResult:
    """Compare ``current`` metrics against ``baseline`` and flag regressions.

    A metric regresses if it moves in the "bad" direction by more than its threshold:

    * ``higher_is_better=True``  -> a *drop* of more than ``threshold`` regresses.
    * ``higher_is_better=False`` -> a *rise* of more than ``threshold`` regresses.

    ``thresholds`` is per-metric (default 0.0 for any metric not listed, i.e. *any* move in
    the bad direction is a regression). Only metrics present in *both* dicts and numeric in
    both are compared; metrics missing from one side, or with non-numeric values, are
    skipped (recorded in ``details`` with ``compared=False``).
    """
    thresholds = thresholds or {}
    regressions: list[str] = []
    details: dict[str, Any] = {}

    # Compare every metric the baseline knows about; this is what defines the contract a
    # candidate run must not regress against.
    for name in sorted(baseline.keys()):
        base_val = _as_number(baseline.get(name))
        cur_raw = current.get(name)
        cur_val = _as_number(cur_raw)

        # Skip metrics we cannot compare numerically on both sides.
        if base_val is None or cur_val is None:
            reason = (
                "missing in current"
                if name not in current
                else "non-numeric value(s)"
            )
            details[name] = {
                "compared": False,
                "reason": reason,
                "current": cur_raw,
                "baseline": baseline.get(name),
            }
            continue

        threshold = float(thresholds.get(name, 0.0))
        delta = cur_val - base_val  # positive => current is larger

        # The "bad-direction" magnitude: how far we moved the wrong way.
        if higher_is_better:
            drop = base_val - cur_val  # positive => we got worse (dropped)
        else:
            drop = cur_val - base_val  # positive => we got worse (rose)

        regressed = drop > threshold

        record = {
            "compared": True,
            "current": cur_val,
            "baseline": base_val,
            "delta": delta,
            "threshold": threshold,
            "higher_is_better": higher_is_better,
            "regressed": regressed,
        }
        details[name] = record

        if regressed:
            direction = "dropped" if higher_is_better else "rose"
            regressions.append(
                f"{name}: {direction} {drop:.6g} (current={cur_val:.6g}, "
                f"baseline={base_val:.6g}, allowed={threshold:.6g})"
            )

    return GateResult(passed=not regressions, regressions=regressions, details=details)


def load_baseline(path: Union[str, Path]) -> dict[str, Any]:
    """Load a baseline metrics dict from a JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_baseline(metrics: dict[str, Any], path: Union[str, Path]) -> Path:
    """Write ``metrics`` to ``path`` as pretty JSON; returns the path written."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return out


def run_gate_cli(
    current_path: Union[str, Path],
    baseline_path: Union[str, Path],
    thresholds_path: Optional[Union[str, Path]] = None,
) -> int:
    """Load metrics + (optional) thresholds from JSON, run the gate, print, return exit code.

    Returns 0 when the gate passes, 1 when any metric regresses. ``thresholds_path`` JSON
    may be a flat ``{metric: threshold}`` map; it may also carry the boolean
    ``higher_is_better`` (default True) used for the whole comparison.
    """
    current = load_baseline(current_path)
    baseline = load_baseline(baseline_path)

    thresholds: dict[str, Number] = {}
    higher_is_better = True
    if thresholds_path is not None:
        raw = load_baseline(thresholds_path)
        # Pull out the optional global flag; everything else is a per-metric threshold.
        higher_is_better = bool(raw.pop("higher_is_better", True))
        thresholds = {k: v for k, v in raw.items() if _as_number(v) is not None}

    result = gate(
        current,
        baseline,
        thresholds=thresholds,
        higher_is_better=higher_is_better,
    )

    if result.passed:
        n = sum(1 for d in result.details.values() if d.get("compared"))
        print(f"GATE PASS - {n} metric(s) compared, no regressions.")
    else:
        print(f"GATE FAIL - {len(result.regressions)} regression(s):")
        for line in result.regressions:
            print(f"  - {line}")

    return 0 if result.passed else 1
