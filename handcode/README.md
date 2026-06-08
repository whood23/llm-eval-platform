# `handcode/` — your reps

This platform is a learning artifact. The **plumbing is scaffolded** (providers, store,
run loop, caching, reporting, CLI) so you can spend your time on the part interviewers
actually probe: **the measurement science**. Those pieces are left as stubs that raise
`NotImplementedError`, with a docstring stating the contract, the reference to verify
against, and a hint — but never the implementation.

> The rule (from `CLAUDE.md`): if you couldn't whiteboard it tomorrow, you hand-code it
> today. Implement it yourself, then verify against a trusted reference — the way
> `grad_check` verifies a gradient with finite differences.

## The verifier

```bash
python handcode/verify_stats.py        # or: make verify-stats
```

It checks each statistic against an independent oracle (`scipy` / `sklearn` / a hand-built
fixture) and prints one line per metric:

| tag | meaning |
|-----|---------|
| `[PASS]` | matches the reference — rep complete |
| `[TODO]` | still a stub — your rep to do |
| `[FAIL]` | implemented but disagrees with the oracle — debug it |
| `[SKIP]` | oracle unavailable on this env (e.g. optional `krippendorff` not installed) |

Exit code is non-zero on any `[FAIL]`; pass `--strict` to also fail while any `[TODO]`
remains (handy in CI once you intend the stats to be done).

## What you hand-code, and where (suggested order — mirrors the build guide)

| Phase | Stub file | Function(s) | Verify against |
|-------|-----------|-------------|----------------|
| 2a | `src/evalplatform/stats/bootstrap.py` | `bootstrap_ci` | `scipy.stats.bootstrap` |
| 2b | `src/evalplatform/stats/position_bias.py` | `position_flip_rate`, `consistent_winrate` | hand-built fixture (known flip rate) |
| 2c | `src/evalplatform/stats/agreement.py` | `cohen_kappa`, `krippendorff_alpha` | `sklearn.metrics.cohen_kappa_score`; `krippendorff` pkg / canonical value |
| 2d | `src/evalplatform/stats/calibration.py` | `expected_calibration_error`, `reliability_curve` | hand fixture; `sklearn.calibration.calibration_curve` |
| 3  | `src/evalplatform/stats/diversity.py` | `embedding_diversity` | `scipy.spatial.distance.pdist` |
| 4  | `src/evalplatform/stats/predictive.py` | `predictive_validity` | `scipy.stats.pearsonr` (+ your bootstrap) |

These are **not** checked by `verify_stats.py` (no clean numeric oracle), but are equally
yours to hand-code — the docstrings carry the contract:

- `src/evalplatform/judge/pointwise.py` — pointwise rubric prompt + output parser.
- `src/evalplatform/judge/pairwise.py` — pairwise rubric prompt + parser + **swap logic**
  (present every pair in both orders; this is what makes Phase-2b possible).
- `src/evalplatform/datasets/sampling.py` — stratified sampling.
- `src/evalplatform/trajectory/score.py` — trajectory scoring + failure localization.

## How to work a rep

1. Read the stub's docstring — it states inputs, outputs, the oracle, and a hint.
2. Implement the body. Don't peek at a reference solution before a real attempt.
3. `python handcode/verify_stats.py` until the line flips to `[PASS]`.
4. Once green, wire it into a report/scorer (the "gray zone" glue — direct Claude, verify).

If you get stuck, ask Claude to *review* your attempt, give a *hint*, or write a *failing
test* — not to write the body. (See the working rule in `CLAUDE.md`.)
