import pytest

from fedsiggraph.evaluation.paired import (
    exact_sign_flip_pvalue,
    paired_gain_summary,
)


def test_exact_sign_flip_pvalue_for_three_consistent_positive_gains():
    # Only all-positive and all-negative assignments are as extreme as the
    # observed mean: 2 / 2^3 = 0.25.
    assert exact_sign_flip_pvalue([0.1, 0.2, 0.3]) == pytest.approx(0.25)


def test_paired_gain_summary_reports_effect_direction_and_significance():
    summary = paired_gain_summary([0.1, 0.2, 0.3, 0.4])

    assert summary["n"] == 4
    assert summary["mean"] == pytest.approx(0.25)
    assert summary["median"] == pytest.approx(0.25)
    assert summary["positive_fraction"] == pytest.approx(1.0)
    assert summary["exact_sign_flip_pvalue"] == pytest.approx(0.125)


def test_paired_gain_summary_handles_no_finite_values():
    summary = paired_gain_summary([float("nan"), float("inf")])

    assert summary["n"] == 0
    assert summary["mean"] is None
    assert summary["exact_sign_flip_pvalue"] is None
