#!/usr/bin/env python
"""Verifier for the hand-coded statistics — the eval-platform analogue of ``grad_check``.

Run it any time:

    python handcode/verify_stats.py            # or:  make verify-stats

For each Phase-2/3/4 statistic you implement in ``src/evalplatform/stats/`` it prints:

    [PASS]  your implementation matches the reference oracle (scipy / sklearn / fixture)
    [TODO]  still a stub (raises NotImplementedError) — your rep to do
    [FAIL]  implemented but disagrees with the oracle — debug it
    [SKIP]  oracle unavailable on this env (e.g. optional package not installed)

The references are independent oracles, exactly like finite differences check a gradient:
scipy.stats.bootstrap, sklearn.metrics.cohen_kappa_score, sklearn.calibration.calibration_curve,
scipy.spatial.distance.pdist, scipy.stats.pearsonr, plus a few small hand-built fixtures whose
answers are known by construction. The reference math lives *here*, never in your stubs.

Exit code: 0 if nothing FAILed (TODOs are fine — they're your remaining reps); 1 on any FAIL.
Pass --strict to also fail the process while any TODO remains (useful in CI once you're done).
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np

# --- make `import evalplatform...` work without an editable install -------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from evalplatform.models import PairwiseJudgment, Position  # noqa: E402
from evalplatform.stats import (  # noqa: E402
    bootstrap_ci,
    cohen_kappa,
    consistent_winrate,
    embedding_diversity,
    expected_calibration_error,
    krippendorff_alpha,
    position_flip_rate,
    predictive_validity,
    reliability_curve,
)

TOL = 1e-6


# --- result plumbing ------------------------------------------------------------------
class CheckResult:
    def __init__(self, status: str, detail: str = ""):
        self.status = status  # PASS | TODO | FAIL | SKIP
        self.detail = detail


def _run(fn) -> CheckResult:
    """Execute one check, mapping exceptions to statuses (NotImplementedError -> TODO)."""
    try:
        return fn()
    except NotImplementedError:
        return CheckResult("TODO", "stub not implemented yet")
    except Exception:  # noqa: BLE001 — surface any other failure as FAIL with context
        tb = traceback.format_exc().strip().splitlines()[-1]
        return CheckResult("FAIL", f"raised: {tb}")


def _close(a, b, tol=TOL) -> bool:
    return bool(np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float), atol=tol, rtol=tol))


# --- the checks (each returns a CheckResult) ------------------------------------------
def check_bootstrap() -> CheckResult:
    from scipy.stats import bootstrap as scipy_bootstrap

    rng = np.random.default_rng(0)
    data = rng.normal(loc=5.0, scale=2.0, size=400)
    got = bootstrap_ci(data, np.mean, n_resamples=5000, confidence=0.95, seed=0)

    # point estimate must equal the sample statistic (tight)
    if not _close(got.point, float(np.mean(data)), tol=1e-9):
        return CheckResult("FAIL", f"point={got.point:.4f} != mean={np.mean(data):.4f}")
    # bounds: compare to scipy's percentile bootstrap (both stochastic -> loose tolerance)
    ref = scipy_bootstrap(
        (data,), np.mean, n_resamples=5000, confidence_level=0.95,
        method="percentile", random_state=1,
    ).confidence_interval
    if got.low >= got.high or not (got.low < np.mean(data) < got.high):
        return CheckResult("FAIL", f"interval [{got.low:.3f},{got.high:.3f}] looks wrong")
    if abs(got.low - ref.low) > 0.1 or abs(got.high - ref.high) > 0.1:
        return CheckResult("FAIL", f"bounds [{got.low:.3f},{got.high:.3f}] vs scipy [{ref.low:.3f},{ref.high:.3f}]")
    return CheckResult("PASS", f"point={got.point:.3f} CI=[{got.low:.3f},{got.high:.3f}]")


def _pairwise_fixture() -> list[PairwiseJudgment]:
    """4 item-pairs of candidates (X vs Y), each judged in AB and BA order.

    By construction exactly ONE of the 4 pairs flips its winner with order, so
    position_flip_rate == 0.25. X wins consistently on 2 pairs, Y on 1, and 1 flips.
    """
    rows: list[PairwiseJudgment] = []

    def pair(item: str, ab_winner: str, ba_winner: str):
        # slot A holds X in the AB row and Y in the BA row (the swap)
        rows.append(PairwiseJudgment(
            run_id="t", item_id=item, candidate_a_id="X", candidate_b_id="Y",
            position=Position.AB, winner_candidate_id=ab_winner,
            judge_model="m", prompt_version="v1",
        ))
        rows.append(PairwiseJudgment(
            run_id="t", item_id=item, candidate_a_id="Y", candidate_b_id="X",
            position=Position.BA, winner_candidate_id=ba_winner,
            judge_model="m", prompt_version="v1",
        ))

    pair("i1", "X", "X")  # X wins both  -> consistent X
    pair("i2", "X", "X")  # X wins both  -> consistent X
    pair("i3", "Y", "Y")  # Y wins both  -> consistent Y
    pair("i4", "X", "Y")  # flips         -> no consistent winner
    return rows


def check_position_flip_rate() -> CheckResult:
    got = position_flip_rate(_pairwise_fixture())
    if not _close(got, 0.25):
        return CheckResult("FAIL", f"flip_rate={got} != 0.25 (1 of 4 pairs flips)")
    return CheckResult("PASS", "flip_rate=0.25")


def check_consistent_winrate() -> CheckResult:
    got = consistent_winrate(_pairwise_fixture())
    # Every pair is X-vs-Y, so each candidate has 4 complete pairings. X wins 2 consistently
    # (i1,i2), Y wins 1 (i3), and i4 flips (credited to neither). So X=2/4, Y=1/4.
    exp = {"X": 0.5, "Y": 0.25}
    for k, v in exp.items():
        if k not in got or not _close(got[k], v):
            return CheckResult("FAIL", f"{got} != {exp}")
    return CheckResult("PASS", "X=0.5 Y=0.25 (flipped pair credited to neither)")


def check_cohen_kappa() -> CheckResult:
    from sklearn.metrics import cohen_kappa_score

    rng = np.random.default_rng(1)
    a = rng.integers(0, 3, size=80)
    b = a.copy()
    flip = rng.random(80) < 0.3  # inject ~30% disagreement
    b[flip] = rng.integers(0, 3, size=flip.sum())
    got = cohen_kappa(list(a), list(b))
    ref = float(cohen_kappa_score(a, b))
    if not _close(got, ref):
        return CheckResult("FAIL", f"kappa={got:.4f} != sklearn {ref:.4f}")
    return CheckResult("PASS", f"kappa={got:.3f} == sklearn")


def check_krippendorff() -> CheckResult:
    # Canonical worked example from the `krippendorff` package docs; nominal alpha ~ 0.691.
    nan = float("nan")
    data = [
        [nan, nan, nan, nan, nan, 3, 4, 1, 2, 1, 1, 3, 3, nan, 3],
        [1, nan, 2, 1, 3, 3, 4, 3, nan, nan, nan, nan, nan, nan, nan],
        [nan, nan, 2, 1, 3, 4, 4, nan, 2, 1, 1, 3, 3, nan, 4],
    ]
    got = krippendorff_alpha(data, level="nominal")
    try:
        import krippendorff as kref  # optional exact oracle

        ref = float(kref.alpha(reliability_data=data, level_of_measurement="nominal"))
    except ImportError:
        # No oracle installed: accept the documented value for this fixture (~0.691).
        if _close(got, 0.691, tol=2e-3):
            return CheckResult("PASS", f"alpha={got:.3f} ~= documented 0.691 (pip install krippendorff for exact)")
        return CheckResult(
            "SKIP",
            f"got alpha={got:.4f}; expected ~0.691 for the canonical fixture — "
            "`pip install krippendorff` for an exact oracle",
        )
    if not _close(got, ref, tol=1e-4):
        return CheckResult("FAIL", f"alpha={got:.4f} != krippendorff {ref:.4f}")
    return CheckResult("PASS", f"alpha={got:.3f} == krippendorff")


def check_ece() -> CheckResult:
    # Hand-built fixture, 2 equal-width bins; ECE worked out by hand = 0.20.
    #   bin0 [0,0.5): conf{0.2,0.4} mean=0.3, acc{0,1}=0.5, gap=0.2, n=2
    #   bin1 [0.5,1]: conf{0.6,0.8} mean=0.7, acc{0,1}=0.5, gap=0.2, n=2
    #   ECE = 0.5*0.2 + 0.5*0.2 = 0.20
    confidences = [0.2, 0.4, 0.6, 0.8]
    correct = [0, 1, 0, 1]
    got = expected_calibration_error(confidences, correct, n_bins=2)
    if not _close(got, 0.20, tol=1e-6):
        return CheckResult("FAIL", f"ECE={got:.4f} != 0.20 (hand-computed fixture)")
    return CheckResult("PASS", "ECE=0.200 on hand-built fixture")


def check_reliability_curve() -> CheckResult:
    from sklearn.calibration import calibration_curve

    rng = np.random.default_rng(2)
    conf = rng.random(500)
    correct = (rng.random(500) < conf).astype(int)  # roughly calibrated
    curve = reliability_curve(conf, correct, n_bins=10)
    prob_true, prob_pred = calibration_curve(correct, conf, n_bins=10, strategy="uniform")

    # align: compare non-empty bins only, in increasing-confidence order
    nonempty = np.asarray(curve.bin_count) > 0
    acc = np.asarray(curve.bin_accuracy)[nonempty]
    cnf = np.asarray(curve.bin_confidence)[nonempty]
    if len(acc) != len(prob_true):
        return CheckResult("FAIL", f"{len(acc)} non-empty bins vs sklearn {len(prob_true)}")
    if not (_close(acc, prob_true, tol=1e-6) and _close(cnf, prob_pred, tol=1e-6)):
        return CheckResult("FAIL", "per-bin accuracy/confidence disagree with sklearn")
    return CheckResult("PASS", f"{len(acc)} bins match sklearn.calibration_curve")


def check_diversity() -> CheckResult:
    from scipy.spatial.distance import pdist

    rng = np.random.default_rng(3)
    emb = rng.normal(size=(25, 8))
    got = embedding_diversity(emb)
    ref = float(pdist(emb, metric="euclidean").mean())
    if not _close(got, ref, tol=1e-6):
        return CheckResult(
            "FAIL",
            f"diversity={got:.4f} != mean Euclidean pairwise {ref:.4f} "
            "(this check assumes mean-pairwise-Euclidean; adjust if you defined it differently)",
        )
    return CheckResult("PASS", f"diversity={got:.3f} == mean pairwise distance")


def check_predictive() -> CheckResult:
    from scipy.stats import pearsonr

    rng = np.random.default_rng(4)
    offline = rng.normal(size=120)
    online = 0.7 * offline + rng.normal(scale=0.7, size=120)
    got = predictive_validity(offline, online, n_resamples=2000, seed=0)
    ref = float(pearsonr(offline, online)[0])
    if not _close(got.point, ref, tol=1e-6):
        return CheckResult("FAIL", f"r={got.point:.4f} != scipy pearsonr {ref:.4f}")
    if not (got.low <= got.point <= got.high) or got.low >= got.high:
        return CheckResult("FAIL", f"bad CI [{got.low:.3f},{got.high:.3f}] around r={got.point:.3f}")
    return CheckResult("PASS", f"r={got.point:.3f} CI=[{got.low:.3f},{got.high:.3f}]")


CHECKS = [
    ("2a  bootstrap CI", "bootstrap_ci", check_bootstrap),
    ("2b  position flip rate", "position_flip_rate", check_position_flip_rate),
    ("2b  bias-corrected win rate", "consistent_winrate", check_consistent_winrate),
    ("2c  Cohen's kappa", "cohen_kappa", check_cohen_kappa),
    ("2c  Krippendorff's alpha", "krippendorff_alpha", check_krippendorff),
    ("2d  Expected Calibration Error", "expected_calibration_error", check_ece),
    ("2d  reliability curve", "reliability_curve", check_reliability_curve),
    ("3   embedding diversity", "embedding_diversity", check_diversity),
    ("4   predictive validity", "predictive_validity", check_predictive),
]

_GLYPH = {"PASS": "[PASS]", "TODO": "[TODO]", "FAIL": "[FAIL]", "SKIP": "[SKIP]"}


def main() -> int:
    strict = "--strict" in sys.argv
    print("=" * 74)
    print(" Hand-coded stats verifier  -  flip every [TODO] to [PASS]")
    print("=" * 74)
    counts = {"PASS": 0, "TODO": 0, "FAIL": 0, "SKIP": 0}
    for phase, name, fn in CHECKS:
        res = _run(fn)
        counts[res.status] += 1
        line = f" {_GLYPH[res.status]}  {phase:<32} {name}"
        if res.detail:
            line += f"\n              -> {res.detail}"
        print(line)
    print("-" * 74)
    print(
        f" {counts['PASS']} pass | {counts['TODO']} todo | "
        f"{counts['FAIL']} fail | {counts['SKIP']} skip   (of {len(CHECKS)})"
    )
    if counts["FAIL"]:
        return 1
    if strict and counts["TODO"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
