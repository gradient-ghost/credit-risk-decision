"""Unit & Integration Tests for Phase 5: Policy Engine & Threshold Optimization.

Verifies:
1. models/policy_thresholds.json exists and contains valid floats 0 < tau_low < tau_high < 1.
2. Monotonicity and boundary behavior of assign_decision_band.
3. compute_expected_loss mathematical correctness.
4. evaluate_policy return structure, non-negative counts, dollar sums, and finite EL reduction.
5. Threshold optimization logic under approval rate constraints.
"""

import json
from pathlib import Path
import numpy as np
import pytest

from src.policy.decision_engine import (
    BAND_AUTO_APPROVE,
    BAND_MANUAL_REVIEW,
    BAND_AUTO_REJECT,
    assign_decision_band,
    compute_expected_loss,
    evaluate_policy,
    optimize_thresholds,
)

POLICY_THRESHOLDS_PATH = Path("models/policy_thresholds.json")


def test_policy_thresholds_json_validity():
    """Assert models/policy_thresholds.json exists with valid floats 0 < tau_low < tau_high < 1."""
    assert POLICY_THRESHOLDS_PATH.exists(), f"File {POLICY_THRESHOLDS_PATH} does not exist"

    with open(POLICY_THRESHOLDS_PATH, "r") as f:
        data = json.load(f)

    assert "tau_low" in data, "tau_low key missing from policy_thresholds.json"
    assert "tau_high" in data, "tau_high key missing from policy_thresholds.json"

    tau_low = float(data["tau_low"])
    tau_high = float(data["tau_high"])

    assert 0.0 < tau_low < 1.0, f"tau_low ({tau_low}) must be strictly in (0, 1)"
    assert 0.0 < tau_high < 1.0, f"tau_high ({tau_high}) must be strictly in (0, 1)"
    assert tau_low < tau_high, f"tau_low ({tau_low}) must be strictly less than tau_high ({tau_high})"

    # Verify presence of evaluation payloads
    assert "validation_eval" in data, "validation_eval section missing"
    assert "test_eval" in data, "test_eval section missing"

    test_eval = data["test_eval"]
    assert "auto_approve" in test_eval
    assert "manual_review" in test_eval
    assert "auto_reject" in test_eval
    assert "headline" in test_eval


def test_assign_decision_band_monotonicity():
    """Assert monotonicity: probabilities < tau_low are AUTO_APPROVE, > tau_high are AUTO_REJECT."""
    tau_low = 0.20
    tau_high = 0.35

    probs = np.array([0.05, 0.15, 0.199, 0.20, 0.25, 0.35, 0.351, 0.50, 0.90])
    bands = assign_decision_band(probs, tau_low=tau_low, tau_high=tau_high)

    expected = np.array([
        BAND_AUTO_APPROVE,
        BAND_AUTO_APPROVE,
        BAND_AUTO_APPROVE,
        BAND_MANUAL_REVIEW,  # exact boundary p == tau_low is MANUAL_REVIEW
        BAND_MANUAL_REVIEW,
        BAND_MANUAL_REVIEW,  # exact boundary p == tau_high is MANUAL_REVIEW
        BAND_AUTO_REJECT,
        BAND_AUTO_REJECT,
        BAND_AUTO_REJECT,
    ])

    np.testing.assert_array_equal(bands, expected)

    # Monotonicity test: map categories to numeric ranks and check non-decreasing
    rank_map = {BAND_AUTO_APPROVE: 0, BAND_MANUAL_REVIEW: 1, BAND_AUTO_REJECT: 2}
    numeric_ranks = np.array([rank_map[b] for b in bands])
    assert np.all(np.diff(numeric_ranks) >= 0), "Decision bands must be monotonically non-decreasing with risk"


def test_assign_decision_band_edge_cases():
    """Assert boundary condition handling and input validation in assign_decision_band."""
    # Empty input
    empty_res = assign_decision_band([], tau_low=0.10, tau_high=0.30)
    assert len(empty_res) == 0

    # Invalid thresholds raise ValueError
    with pytest.raises(ValueError):
        assign_decision_band([0.2], tau_low=0.40, tau_high=0.20)  # tau_low > tau_high

    with pytest.raises(ValueError):
        assign_decision_band([0.2], tau_low=-0.05, tau_high=0.30)  # negative tau_low

    with pytest.raises(ValueError):
        assign_decision_band([0.2], tau_low=0.10, tau_high=1.50)  # tau_high > 1


def test_compute_expected_loss_math():
    """Assert compute_expected_loss calculates PD * EAD * LGD correctly."""
    probs = np.array([0.10, 0.25, 0.50])
    loans = np.array([10000.0, 20000.0, 30000.0])
    lgd = 1.0

    el = compute_expected_loss(probs, loans, lgd=lgd)
    expected_el = np.array([1000.0, 5000.0, 15000.0])

    np.testing.assert_allclose(el, expected_el)

    # LGD scaling
    el_half_lgd = compute_expected_loss(probs, loans, lgd=0.5)
    np.testing.assert_allclose(el_half_lgd, expected_el * 0.5)

    # Mismatch length raises ValueError
    with pytest.raises(ValueError):
        compute_expected_loss(probs, loans[:2])


def test_evaluate_policy_metrics():
    """Assert evaluate_policy returns non-negative counts, valid dollar sums, and finite EL reduction."""
    probs = np.array([0.05, 0.10, 0.22, 0.28, 0.45, 0.60])
    y_true = np.array([0, 0, 0, 1, 1, 1])
    loans = np.array([10000.0, 15000.0, 12000.0, 18000.0, 20000.0, 25000.0])

    res = evaluate_policy(
        probabilities=probs,
        y_true=y_true,
        loan_amounts=loans,
        tau_low=0.20,
        tau_high=0.35,
        lgd=1.0,
    )

    # Check structure
    assert res["total_applicants"] == len(probs)
    assert np.isclose(res["total_exposure"], np.sum(loans))

    # Counts add up
    total_counts = (
        res["auto_approve"]["count"]
        + res["manual_review"]["count"]
        + res["auto_reject"]["count"]
    )
    assert total_counts == len(probs)

    # Exposure adds up
    total_band_exposure = (
        res["auto_approve"]["total_exposure"]
        + res["manual_review"]["total_exposure"]
        + res["auto_reject"]["total_exposure"]
    )
    assert np.isclose(total_band_exposure, res["total_exposure"])

    # Non-negative counts and valid dollar sums
    for band_key in ["auto_approve", "manual_review", "auto_reject"]:
        band = res[band_key]
        assert band["count"] >= 0
        assert band["pct_portfolio"] >= 0.0
        assert band["total_exposure"] >= 0.0
        assert 0.0 <= band["realized_default_rate"] <= 100.0

    # Headline metric validity
    headline = res["headline"]
    assert headline["expected_loss_reduction"] >= 0.0
    assert 0.0 <= headline["pct_expected_loss_reduction"] <= 100.0
    assert np.isfinite(headline["expected_loss_reduction"])
    assert np.isfinite(headline["pct_expected_loss_reduction"])


def test_optimize_thresholds_synthetic():
    """Assert optimize_thresholds identifies valid thresholds under approval rate constraints."""
    rng = np.random.default_rng(42)
    n = 1000
    # Simulate calibrated probabilities skewed toward low risk
    probs = rng.beta(2, 5, size=n)
    y_true = (probs > 0.35).astype(int)
    loans = rng.uniform(5000, 35000, size=n)

    opt = optimize_thresholds(
        y_prob_val=probs,
        y_true_val=y_true,
        loan_amounts_val=loans,
        min_approval_rate=0.50,
        target_review_rate=0.15,
        tau_low_range=(0.10, 0.30, 0.02),
        tau_high_range=(0.20, 0.45, 0.02),
    )

    tl = opt["optimal_tau_low"]
    th = opt["optimal_tau_high"]

    assert 0.10 <= tl <= 0.30
    assert 0.20 <= th <= 0.45
    assert tl < th
    assert "val_evaluation" in opt
