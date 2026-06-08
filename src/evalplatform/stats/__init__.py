"""The measurement-science layer — *** THE DIFFERENTIATOR. YOU HAND-CODE EVERY STAT. ***

This is the part of the platform you will be quizzed on, so it is the part you implement.
Each function below is a stub that raises ``NotImplementedError`` with a docstring stating
the contract (inputs/outputs), the reference to verify against, and a hint — never the
implementation. Implement them, then run::

    python handcode/verify_stats.py        # flips each [TODO] -> [PASS]

The order mirrors the build guide: bootstrap CIs -> position bias -> agreement ->
calibration, then the Phase 3/4 metrics (diversity, predictive validity).
"""

from __future__ import annotations

from ._types import CI
from .agreement import cohen_kappa, krippendorff_alpha
from .bootstrap import bootstrap_ci
from .calibration import ReliabilityCurve, expected_calibration_error, reliability_curve
from .diversity import embedding_diversity
from .position_bias import consistent_winrate, position_flip_rate
from .predictive import predictive_validity

__all__ = [
    "CI",
    "bootstrap_ci",
    "position_flip_rate",
    "consistent_winrate",
    "cohen_kappa",
    "krippendorff_alpha",
    "expected_calibration_error",
    "reliability_curve",
    "ReliabilityCurve",
    "embedding_diversity",
    "predictive_validity",
]
