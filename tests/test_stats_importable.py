"""Smoke-import the stats layer and assert its public surface exists (AI-written).

These tests deliberately do NOT call the stat functions and do NOT assert
``NotImplementedError`` — so they keep passing unchanged once the user hand-codes the
statistics. They only check that the public callables/types are present and importable,
which is what sibling modules (runner, report, gate) compose against.
"""

from __future__ import annotations

import inspect


def test_stats_package_reexports():
    import evalplatform.stats as stats

    expected = {
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
    }
    assert expected <= set(stats.__all__)
    for name in expected:
        assert hasattr(stats, name), f"stats.{name} missing"


def test_ci_namedtuple_fields():
    from evalplatform.stats._types import CI

    assert CI._fields == ("point", "low", "high")


def test_bootstrap_module_surface():
    from evalplatform.stats import bootstrap

    assert callable(bootstrap.bootstrap_ci)
    sig = inspect.signature(bootstrap.bootstrap_ci)
    assert "data" in sig.parameters
    assert "n_resamples" in sig.parameters
    assert "confidence" in sig.parameters


def test_position_bias_module_surface():
    from evalplatform.stats import position_bias

    assert callable(position_bias.position_flip_rate)
    assert callable(position_bias.consistent_winrate)


def test_agreement_module_surface():
    from evalplatform.stats import agreement

    assert callable(agreement.cohen_kappa)
    assert callable(agreement.krippendorff_alpha)


def test_calibration_module_surface():
    from evalplatform.stats import calibration

    assert callable(calibration.expected_calibration_error)
    assert callable(calibration.reliability_curve)
    # ReliabilityCurve is the container the report/plots layer consumes.
    assert calibration.ReliabilityCurve._fields == (
        "bin_edges",
        "bin_confidence",
        "bin_accuracy",
        "bin_count",
    )


def test_diversity_module_surface():
    from evalplatform.stats import diversity

    assert callable(diversity.embedding_diversity)


def test_predictive_module_surface():
    from evalplatform.stats import predictive

    assert callable(predictive.predictive_validity)
