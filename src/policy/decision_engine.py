"""Decision Policy & Threshold Optimization Engine.

Implements the 3-Band Credit Risk Decision Policy (ADR 1.4, ADR 4.3):
1. Auto-Approve (p < tau_low): High-confidence prime applicants originated instantly.
2. Manual Review (tau_low <= p <= tau_high): Ambiguous gray-zone applicants routed to underwriters.
3. Auto-Reject (p > tau_high): High-risk applicants declined with statutory Adverse Action notices.

Performs 2D grid search on out-of-time validation data to optimize (tau_low, tau_high)
subject to operational approval rate constraints, and evaluates the policy on the
untouched 2017-2018 Test Set.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Sequence, Tuple, Union

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("decision_engine")

BAND_AUTO_APPROVE = "AUTO_APPROVE"
BAND_MANUAL_REVIEW = "MANUAL_REVIEW"
BAND_AUTO_REJECT = "AUTO_REJECT"


def assign_decision_band(
    probabilities: Union[np.ndarray, pd.Series, Sequence[float]],
    tau_low: float,
    tau_high: float,
) -> np.ndarray:
    """Assign applicants into 3 decision tiers based on default probability.

    Parameters
    ----------
    probabilities : array-like of shape (n_samples,)
        Predicted probability of default in [0, 1].
    tau_low : float
        Lower threshold. Probabilities < tau_low are AUTO_APPROVE.
    tau_high : float
        Upper threshold. Probabilities > tau_high are AUTO_REJECT.
        Probabilities in [tau_low, tau_high] are MANUAL_REVIEW.

    Returns
    -------
    np.ndarray of shape (n_samples,) with dtype object
        Array of categorical decisions: 'AUTO_APPROVE', 'MANUAL_REVIEW', 'AUTO_REJECT'.

    Raises
    ------
    ValueError
        If thresholds do not satisfy 0.0 <= tau_low <= tau_high <= 1.0.
    """
    if not (0.0 <= tau_low <= tau_high <= 1.0):
        raise ValueError(
            f"Invalid thresholds: require 0.0 <= tau_low ({tau_low}) <= tau_high ({tau_high}) <= 1.0"
        )

    probs = np.asarray(probabilities, dtype=float).ravel()
    if len(probs) == 0:
        return np.array([], dtype=object)

    conditions = [
        probs < tau_low,
        probs <= tau_high,
    ]
    choices = [
        BAND_AUTO_APPROVE,
        BAND_MANUAL_REVIEW,
    ]
    return np.select(conditions, choices, default=BAND_AUTO_REJECT)


def compute_expected_loss(
    probabilities: Union[np.ndarray, pd.Series, Sequence[float]],
    loan_amounts: Union[np.ndarray, pd.Series, Sequence[float]],
    lgd: float = 1.0,
) -> np.ndarray:
    """Compute per-loan Expected Loss (EL = PD * EAD * LGD).

    Parameters
    ----------
    probabilities : array-like of shape (n_samples,)
        Probability of default (PD) in [0, 1].
    loan_amounts : array-like of shape (n_samples,)
        Exposure at Default (EAD) in dollars (loan_amnt).
    lgd : float, default=1.0
        Loss Given Default ratio in [0, 1]. Defaults to 1.0 (unsecured consumer debt).

    Returns
    -------
    np.ndarray of shape (n_samples,)
        Expected loss in dollars for each applicant.
    """
    probs = np.asarray(probabilities, dtype=float).ravel()
    exposure = np.asarray(loan_amounts, dtype=float).ravel()

    if len(probs) != len(exposure):
        raise ValueError(
            f"Dimension mismatch: probabilities length ({len(probs)}) != loan_amounts length ({len(exposure)})"
        )

    return probs * exposure * float(lgd)


def evaluate_policy(
    probabilities: Union[np.ndarray, pd.Series, Sequence[float]],
    y_true: Union[np.ndarray, pd.Series, Sequence[int]],
    loan_amounts: Union[np.ndarray, pd.Series, Sequence[float]],
    tau_low: float,
    tau_high: float,
    lgd: float = 1.0,
) -> Dict[str, Any]:
    """Compute comprehensive portfolio metrics across all 3 decision bands.

    Parameters
    ----------
    probabilities : array-like of shape (n_samples,)
        Predicted default probabilities.
    y_true : array-like of shape (n_samples,)
        True binary default outcomes {0, 1}.
    loan_amounts : array-like of shape (n_samples,)
        Loan exposure in dollars.
    tau_low : float
        Lower threshold for Auto-Approve.
    tau_high : float
        Upper threshold for Auto-Reject.
    lgd : float, default=1.0
        Loss Given Default assumption.

    Returns
    -------
    dict
        Detailed portfolio breakdown containing per-band statistics and headline metrics.
    """
    probs = np.asarray(probabilities, dtype=float).ravel()
    labels = np.asarray(y_true, dtype=int).ravel()
    loans = np.asarray(loan_amounts, dtype=float).ravel()
    n_samples = len(probs)

    if not (len(probs) == len(labels) == len(loans)):
        raise ValueError("Input arrays must have identical length.")

    # Calculate losses
    el_per_applicant = compute_expected_loss(probs, loans, lgd=lgd)
    realized_loss_per_applicant = labels * loans * float(lgd)

    # Baseline: Approve-Everyone
    baseline_expected_loss = float(np.sum(el_per_applicant))
    baseline_realized_loss = float(np.sum(realized_loss_per_applicant))
    total_exposure = float(np.sum(loans))
    baseline_default_rate = float(np.mean(labels) * 100.0) if n_samples > 0 else 0.0

    # Band assignments
    bands = assign_decision_band(probs, tau_low, tau_high)
    mask_approve = bands == BAND_AUTO_APPROVE
    mask_review = bands == BAND_MANUAL_REVIEW
    mask_reject = bands == BAND_AUTO_REJECT

    # Band 1: Auto-Approve
    count_approve = int(np.sum(mask_approve))
    pct_approve = float(count_approve / n_samples * 100.0) if n_samples > 0 else 0.0
    exposure_approve = float(np.sum(loans[mask_approve]))
    default_rate_approve = float(np.mean(labels[mask_approve]) * 100.0) if count_approve > 0 else 0.0
    el_approve = float(np.sum(el_per_applicant[mask_approve]))
    realized_loss_approve = float(np.sum(realized_loss_per_applicant[mask_approve]))

    # Band 2: Manual Review
    count_review = int(np.sum(mask_review))
    pct_review = float(count_review / n_samples * 100.0) if n_samples > 0 else 0.0
    exposure_review = float(np.sum(loans[mask_review]))
    default_rate_review = float(np.mean(labels[mask_review]) * 100.0) if count_review > 0 else 0.0
    el_review = float(np.sum(el_per_applicant[mask_review]))
    realized_loss_review = float(np.sum(realized_loss_per_applicant[mask_review]))

    # Band 3: Auto-Reject
    count_reject = int(np.sum(mask_reject))
    pct_reject = float(count_reject / n_samples * 100.0) if n_samples > 0 else 0.0
    exposure_reject = float(np.sum(loans[mask_reject]))
    default_rate_reject = float(np.mean(labels[mask_reject]) * 100.0) if count_reject > 0 else 0.0
    el_saved_reject = float(np.sum(el_per_applicant[mask_reject]))
    realized_loss_saved_reject = float(np.sum(realized_loss_per_applicant[mask_reject]))

    # Headline: Expected Loss Saved by Rejecting High-Risk Applicants
    expected_loss_reduction = el_saved_reject
    pct_el_reduction = (
        float(expected_loss_reduction / baseline_expected_loss * 100.0)
        if baseline_expected_loss > 0
        else 0.0
    )
    realized_loss_reduction = realized_loss_saved_reject
    pct_realized_reduction = (
        float(realized_loss_reduction / baseline_realized_loss * 100.0)
        if baseline_realized_loss > 0
        else 0.0
    )

    return {
        "tau_low": float(tau_low),
        "tau_high": float(tau_high),
        "lgd": float(lgd),
        "total_applicants": n_samples,
        "total_exposure": total_exposure,
        "baseline": {
            "expected_loss": baseline_expected_loss,
            "realized_loss": baseline_realized_loss,
            "default_rate_pct": baseline_default_rate,
            "expected_loss_rate_pct": float(baseline_expected_loss / total_exposure * 100.0)
            if total_exposure > 0
            else 0.0,
        },
        "auto_approve": {
            "count": count_approve,
            "pct_portfolio": pct_approve,
            "total_exposure": exposure_approve,
            "realized_default_rate": default_rate_approve,
            "total_expected_loss": el_approve,
            "realized_losses": realized_loss_approve,
            "expected_loss_rate_pct": float(el_approve / exposure_approve * 100.0)
            if exposure_approve > 0
            else 0.0,
        },
        "manual_review": {
            "count": count_review,
            "pct_portfolio": pct_review,
            "total_exposure": exposure_review,
            "realized_default_rate": default_rate_review,
            "total_expected_loss": el_review,
            "realized_losses": realized_loss_review,
            "expected_loss_rate_pct": float(el_review / exposure_review * 100.0)
            if exposure_review > 0
            else 0.0,
        },
        "auto_reject": {
            "count": count_reject,
            "pct_portfolio": pct_reject,
            "total_exposure": exposure_reject,
            "realized_default_rate": default_rate_reject,
            "expected_losses_saved": el_saved_reject,
            "realized_losses_saved": realized_loss_saved_reject,
            "expected_loss_rate_pct": float(el_saved_reject / exposure_reject * 100.0)
            if exposure_reject > 0
            else 0.0,
        },
        "headline": {
            "expected_loss_reduction": expected_loss_reduction,
            "pct_expected_loss_reduction": pct_el_reduction,
            "realized_loss_reduction": realized_loss_reduction,
            "pct_realized_loss_reduction": pct_realized_reduction,
        },
    }


def optimize_thresholds(
    y_prob_val: np.ndarray,
    y_true_val: np.ndarray,
    loan_amounts_val: np.ndarray,
    min_approval_rate: float = 0.60,
    target_review_rate: Optional[float] = 0.18,
    tau_low_range: Tuple[float, float, float] = (0.05, 0.25, 0.01),
    tau_high_range: Tuple[float, float, float] = (0.15, 0.45, 0.01),
    lgd: float = 1.0,
) -> Dict[str, Any]:
    """Perform a 2D grid search over (tau_low, tau_high) on validation data.

    Finds the optimal threshold pair that minimizes Expected Loss on the approved pool
    subject to maintaining >= min_approval_rate Auto-Approval rate (or combined auto+review capacity),
    while aligning the manual review band with realistic underwriting operational capacity.

    Parameters
    ----------
    y_prob_val : np.ndarray
        Calibrated probabilities on validation dataset.
    y_true_val : np.ndarray
        Ground-truth labels on validation dataset.
    loan_amounts_val : np.ndarray
        Loan amounts in validation dataset.
    min_approval_rate : float, default=0.60
        Minimum required auto-approval rate.
    target_review_rate : float or None, default=0.18
        Target operational manual review bandwidth (~18% of portfolio).
        If provided, selects tau_high closest to target capacity.
    tau_low_range : tuple of (start, end, step), default=(0.05, 0.25, 0.01)
        Search grid for tau_low.
    tau_high_range : tuple of (start, end, step), default=(0.15, 0.45, 0.01)
        Search grid for tau_high.
    lgd : float, default=1.0
        Loss Given Default.

    Returns
    -------
    dict
        Optimal policy configuration and evaluation metrics.
    """
    tl_start, tl_end, tl_step = tau_low_range
    th_start, th_end, th_step = tau_high_range

    tau_low_grid = np.round(np.arange(tl_start, tl_end + 1e-5, tl_step), 2)
    tau_high_grid = np.round(np.arange(th_start, th_end + 1e-5, th_step), 2)

    logger.info(
        f"Starting 2D grid search: tau_low in [{tl_start:.2f}, {tl_end:.2f}] (n={len(tau_low_grid)}), "
        f"tau_high in [{th_start:.2f}, {th_end:.2f}] (n={len(tau_high_grid)}), "
        f"min_approval_rate={min_approval_rate:.2f}, target_review_rate={target_review_rate}"
    )

    n_samples = len(y_prob_val)
    el_per_applicant = compute_expected_loss(y_prob_val, loan_amounts_val, lgd=lgd)

    candidates = []

    for tl in tau_low_grid:
        auto_mask = y_prob_val < tl
        auto_rate = float(np.mean(auto_mask))
        auto_el = float(np.sum(el_per_applicant[auto_mask]))
        auto_exposure = float(np.sum(loan_amounts_val[auto_mask]))
        auto_el_rate = (auto_el / auto_exposure) if auto_exposure > 0 else 0.0

        for th in tau_high_grid:
            if tl >= th:
                continue

            review_mask = (y_prob_val >= tl) & (y_prob_val <= th)
            reject_mask = y_prob_val > th

            review_rate = float(np.mean(review_mask))
            reject_rate = float(np.mean(reject_mask))
            combined_rate = auto_rate + review_rate

            el_saved = float(np.sum(el_per_applicant[reject_mask]))

            candidates.append({
                "tau_low": float(tl),
                "tau_high": float(th),
                "auto_approval_rate": auto_rate,
                "review_rate": review_rate,
                "combined_rate": combined_rate,
                "reject_rate": reject_rate,
                "auto_approved_el": auto_el,
                "auto_approved_el_rate": auto_el_rate,
                "expected_loss_saved": el_saved,
            })

    # Step 1: Filter candidates meeting min_approval_rate on Auto-Approve (or fallback to combined)
    eligible_auto = [c for c in candidates if c["auto_approval_rate"] >= min_approval_rate]
    if eligible_auto:
        # Minimum expected loss on approved pool chooses the lowest qualifying tau_low
        min_auto_el = min(c["auto_approved_el"] for c in eligible_auto)
        best_tl_candidates = [c for c in eligible_auto if np.isclose(c["auto_approved_el"], min_auto_el)]
        selected_tier = "Auto-Approval Constraint (auto_rate >= 0.60)"
    else:
        # Fallback to combined capacity
        eligible_comb = [c for c in candidates if c["combined_rate"] >= min_approval_rate]
        min_comb_el = min(c["auto_approved_el"] for c in eligible_comb)
        best_tl_candidates = [c for c in eligible_comb if np.isclose(c["auto_approved_el"], min_comb_el)]
        selected_tier = "Combined Capacity Fallback (combined_rate >= 0.60)"

    # Step 2: Choose tau_high
    if target_review_rate is not None:
        # Select tau_high closest to target review capacity
        best_tl_candidates.sort(key=lambda x: abs(x["review_rate"] - target_review_rate))
    else:
        # Maximize expected loss saved
        best_tl_candidates.sort(key=lambda x: -x["expected_loss_saved"])

    best = best_tl_candidates[0]
    best_tau_low = best["tau_low"]
    best_tau_high = best["tau_high"]

    logger.info(
        f"Selected optimal thresholds: tau_low={best_tau_low:.2f}, tau_high={best_tau_high:.2f} "
        f"via {selected_tier}"
    )

    # Full validation evaluation on optimal thresholds
    val_policy_eval = evaluate_policy(
        probabilities=y_prob_val,
        y_true=y_true_val,
        loan_amounts=loan_amounts_val,
        tau_low=best_tau_low,
        tau_high=best_tau_high,
        lgd=lgd,
    )

    return {
        "optimal_tau_low": best_tau_low,
        "optimal_tau_high": best_tau_high,
        "selected_tier": selected_tier,
        "min_approval_rate_constraint": min_approval_rate,
        "target_review_rate": target_review_rate,
        "val_evaluation": val_policy_eval,
    }


def main():
    """Main execution CLI for Decision Policy & Threshold Optimization."""
    logger.info("=" * 70)
    logger.info("PHASE 5: DECISION POLICY & THRESHOLD OPTIMIZATION ENGINE")
    logger.info("=" * 70)

    # Paths
    model_path = PROJECT_ROOT / "models" / "champion_model.joblib"
    val_data_path = PROJECT_ROOT / "data" / "processed" / "val.parquet"
    X_val_path = PROJECT_ROOT / "data" / "processed" / "X_val.parquet"
    y_val_path = PROJECT_ROOT / "data" / "processed" / "y_val.parquet"

    test_data_path = PROJECT_ROOT / "data" / "processed" / "test.parquet"
    X_test_path = PROJECT_ROOT / "data" / "processed" / "X_test.parquet"
    y_test_path = PROJECT_ROOT / "data" / "processed" / "y_test.parquet"

    output_policy_path = PROJECT_ROOT / "models" / "policy_thresholds.json"

    # Verify prerequisites
    for p in [model_path, val_data_path, X_val_path, y_val_path, test_data_path, X_test_path, y_test_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required artifact: {p}")

    logger.info(f"Loading calibrated champion model from {model_path}...")
    model = joblib.load(model_path)

    # Load Validation Set
    logger.info("Loading Validation Set (2016)...")
    val_df = pd.read_parquet(val_data_path, columns=["loan_amnt"])
    X_val = pd.read_parquet(X_val_path)
    y_val = pd.read_parquet(y_val_path)["target"].values
    loan_amnt_val = val_df["loan_amnt"].values

    logger.info(f"Computing validation probabilities for {len(X_val):,} applicants...")
    probs_val = model.predict_proba(X_val)[:, 1]

    # Run Threshold Optimization on Validation Fold
    logger.info("Optimizing decision thresholds on Validation fold...")
    opt_results = optimize_thresholds(
        y_prob_val=probs_val,
        y_true_val=y_val,
        loan_amounts_val=loan_amnt_val,
        min_approval_rate=0.60,
        target_review_rate=0.18,
    )

    tau_low_star = opt_results["optimal_tau_low"]
    tau_high_star = opt_results["optimal_tau_high"]
    logger.info(f"Optimal Thresholds Found: tau_low* = {tau_low_star:.2f}, tau_high* = {tau_high_star:.2f}")

    # Load Untouched Test Set (2017-2018)
    logger.info("Loading Untouched Test Set (2017-2018)...")
    test_df = pd.read_parquet(test_data_path, columns=["loan_amnt"])
    X_test = pd.read_parquet(X_test_path)
    y_test = pd.read_parquet(y_test_path)["target"].values
    loan_amnt_test = test_df["loan_amnt"].values

    logger.info(f"Computing test probabilities for {len(X_test):,} applicants...")
    probs_test = model.predict_proba(X_test)[:, 1]

    logger.info(f"Evaluating optimal policy on Test Set with tau_low={tau_low_star}, tau_high={tau_high_star}...")
    test_policy_eval = evaluate_policy(
        probabilities=probs_test,
        y_true=y_test,
        loan_amounts=loan_amnt_test,
        tau_low=tau_low_star,
        tau_high=tau_high_star,
        lgd=1.0,
    )

    # Prepare export dictionary
    export_payload = {
        "policy_name": "3-Band Credit Risk Decision Policy",
        "tau_low": tau_low_star,
        "tau_high": tau_high_star,
        "lgd_assumption": 1.0,
        "min_approval_rate_constraint": 0.60,
        "max_review_rate_constraint": 0.20,
        "selection_tier": opt_results["selected_tier"],
        "validation_eval": opt_results["val_evaluation"],
        "test_eval": test_policy_eval,
    }

    output_policy_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_policy_path, "w") as f:
        json.dump(export_payload, f, indent=2)

    logger.info(f"Serialized policy thresholds and metrics to {output_policy_path}")

    # Executive Output Summary
    head = test_policy_eval["headline"]
    auto_app = test_policy_eval["auto_approve"]
    man_rev = test_policy_eval["manual_review"]
    auto_rej = test_policy_eval["auto_reject"]
    base = test_policy_eval["baseline"]

    print("\n" + "=" * 70)
    print("🏆 OPTIMAL POLICY EVALUATION ON TEST SET (2017-2018)")
    print("=" * 70)
    print(f"Optimal Cutoffs: tau_low* = {tau_low_star:.2f} | tau_high* = {tau_high_star:.2f}")
    print(f"Total Applicants Evaluated: {test_policy_eval['total_applicants']:,}")
    print(f"Total Portfolio Exposure:   ${test_policy_eval['total_exposure']:,.2f}")
    print("-" * 70)
    print("1. AUTO-APPROVE BAND (p < tau_low):")
    print(f"   Count:              {auto_app['count']:,} ({auto_app['pct_portfolio']:.2f}% of portfolio)")
    print(f"   Total Exposure:     ${auto_app['total_exposure']:,.2f}")
    print(f"   Realized Def Rate:  {auto_app['realized_default_rate']:.2f}% (vs {base['default_rate_pct']:.2f}% baseline)")
    print(f"   Total Expected Loss: ${auto_app['total_expected_loss']:,.2f}")
    print(f"   Realized Losses:    ${auto_app['realized_losses']:,.2f}")
    print("-" * 70)
    print("2. MANUAL REVIEW BAND (tau_low <= p <= tau_high):")
    print(f"   Count:              {man_rev['count']:,} ({man_rev['pct_portfolio']:.2f}% of portfolio)")
    print(f"   Total Exposure:     ${man_rev['total_exposure']:,.2f}")
    print(f"   Realized Def Rate:  {man_rev['realized_default_rate']:.2f}%")
    print(f"   Total Expected Loss: ${man_rev['total_expected_loss']:,.2f}")
    print("-" * 70)
    print("3. AUTO-REJECT BAND (p > tau_high):")
    print(f"   Count:              {auto_rej['count']:,} ({auto_rej['pct_portfolio']:.2f}% of portfolio)")
    print(f"   Total Exposure:     ${auto_rej['total_exposure']:,.2f}")
    print(f"   Realized Def Rate:  {auto_rej['realized_default_rate']:.2f}% (High-Risk Cohort)")
    print(f"   Expected Loss Saved: ${auto_rej['expected_losses_saved']:,.2f}")
    print(f"   Realized Loss Saved: ${auto_rej['realized_losses_saved']:,.2f}")
    print("-" * 70)
    print("🌟 HEADLINE METRIC (Policy vs Approve-Everyone Baseline):")
    print(f"   Baseline Expected Loss:  ${base['expected_loss']:,.2f}")
    print(f"   Expected Loss Reduction: ${head['expected_loss_reduction']:,.2f} ({head['pct_expected_loss_reduction']:.2f}%)")
    print(f"   Realized Loss Reduction: ${head['realized_loss_reduction']:,.2f} ({head['pct_realized_loss_reduction']:.2f}%)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
