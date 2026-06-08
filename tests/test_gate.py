"""Tests for the regression gate compare/threshold logic (AI-written).

The compared metric *values* come from the user's hand-coded stats; only the compare
and threshold logic is exercised here (with literal numbers, never a computed statistic).
"""

from __future__ import annotations

import json

from evalplatform.gate.regression_gate import (
    GateResult,
    gate,
    load_baseline,
    run_gate_cli,
    save_baseline,
)


def test_gate_passes_when_metrics_improve():
    baseline = {"accuracy": 0.80, "winrate": 0.50}
    current = {"accuracy": 0.85, "winrate": 0.55}
    res = gate(current, baseline)
    assert isinstance(res, GateResult)
    assert res.passed is True
    assert res.regressions == []


def test_gate_passes_when_unchanged():
    metrics = {"accuracy": 0.80}
    res = gate(dict(metrics), dict(metrics))
    assert res.passed is True
    assert res.regressions == []


def test_gate_fails_on_regression_higher_is_better():
    baseline = {"accuracy": 0.90}
    current = {"accuracy": 0.80}
    res = gate(current, baseline, higher_is_better=True)
    assert res.passed is False
    # The regressed metric is recorded in details and named in a regression line.
    assert res.details["accuracy"]["regressed"] is True
    assert any("accuracy" in line for line in res.regressions)


def test_threshold_tolerates_small_drop():
    baseline = {"accuracy": 0.90}
    current = {"accuracy": 0.88}  # 0.02 drop
    # A 0.05 tolerance means this is within budget -> pass.
    res = gate(current, baseline, thresholds={"accuracy": 0.05})
    assert res.passed is True
    # A tighter 0.01 tolerance flags it.
    res2 = gate(current, baseline, thresholds={"accuracy": 0.01})
    assert res2.passed is False
    assert res2.details["accuracy"]["regressed"] is True
    assert any("accuracy" in line for line in res2.regressions)


def test_lower_is_better_direction():
    # e.g. position_flip_rate / ECE: smaller is better, so a rise is a regression.
    baseline = {"flip_rate": 0.10}
    worse = {"flip_rate": 0.30}
    better = {"flip_rate": 0.05}
    assert gate(worse, baseline, higher_is_better=False).passed is False
    assert gate(better, baseline, higher_is_better=False).passed is True


def test_baseline_save_load_roundtrip(tmp_path):
    metrics = {"accuracy": 0.8, "winrate": 0.55}
    path = tmp_path / "baseline.json"
    save_baseline(metrics, path)
    assert path.exists()
    assert load_baseline(path) == metrics
    # File is valid JSON on disk.
    assert json.loads(path.read_text(encoding="utf-8")) == metrics


def test_run_gate_cli_pass_returns_zero(tmp_path):
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    save_baseline({"accuracy": 0.80}, baseline)
    save_baseline({"accuracy": 0.85}, current)
    assert run_gate_cli(current, baseline) == 0


def test_run_gate_cli_fail_returns_one(tmp_path):
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    save_baseline({"accuracy": 0.90}, baseline)
    save_baseline({"accuracy": 0.50}, current)
    assert run_gate_cli(current, baseline) == 1


def test_run_gate_cli_with_thresholds_file(tmp_path):
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    thresholds = tmp_path / "thresholds.json"
    save_baseline({"accuracy": 0.90}, baseline)
    save_baseline({"accuracy": 0.88}, current)  # 0.02 drop
    thresholds.write_text(json.dumps({"accuracy": 0.05}), encoding="utf-8")
    # Within the 0.05 budget -> pass.
    assert run_gate_cli(current, baseline, thresholds) == 0
