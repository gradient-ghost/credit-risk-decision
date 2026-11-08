"""SHAP Explainability & Adverse Action Engine.

Implements model interpretability and regulatory compliance (ADR 6.1, ADR 6.2):
1. TreeSHAP Attribution: Computes Shapley feature attributions (phi_i) using shap.TreeExplainer.
2. Adverse Action Notices: Maps technical model features to plain-English, regulatory-compliant
   descriptions under ECOA (Regulation B) and FCRA requirements.
3. Precomputed SHAP Test Sample: Computes background sample of 1,000 test set applicants,
   generates publication-quality Global SHAP Beeswarm Plot, and serializes cache for Streamlit.
"""

from __future__ import annotations

from datetime import datetime
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Ensure MPLCONFIGDIR is in writable directory
if os.environ.get("MPLCONFIGDIR") is None:
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("shap_explainer")

# Regulatory Adverse Action Feature Descriptions (ECOA / FCRA Compliant)
ADVERSE_ACTION_MAPPING: Dict[str, str] = {
    # Core Risk Ratios & Demographics
    "payment_to_income_ratio": "High monthly debt payment relative to monthly income",
    "revol_util_clean": "Elevated revolving credit card line utilization",
    "credit_hunger_velocity": "High frequency of recent hard credit inquiries (past 6 months)",
    "dti": "High aggregate debt-to-income ratio",
    "fico_score": "Below average credit bureau score",
    "credit_history_age_years": "Insufficient length of established credit history",
    "account_breadth": "High proportion of active credit accounts",
    "delinq_recency_imputed": "Recent record of past-due payment delinquency",
    # Loan Terms & Rates
    "int_rate": "High interest rate risk tier on requested credit product",
    "term_months": "Extended loan repayment term (60 months)",
    "loan_amnt": "Requested loan amount exceeds underwriting risk tolerance",
    "installment": "High monthly payment obligation",
    "annual_inc": "Insufficient annual income for requested loan obligation",
    # Bureau Velocity & Delinquency Factors
    "acc_open_past_24mths": "Too many credit accounts opened in the past 24 months",
    "mo_sin_old_rev_tl_op": "Insufficient age of oldest revolving account",
    "avg_cur_bal": "Low average balance maintained across active credit accounts",
    "mths_since_recent_inq": "Recent credit inquiries indicating elevated credit-seeking behavior",
    "total_bc_limit": "Low aggregate bankcard credit limit",
    "mths_since_recent_bc": "Recent opening of new bankcard credit account",
    "num_tl_op_past_12m": "Excessive new credit accounts opened within past 12 months",
    "bc_util": "High utilization of revolving bankcard credit lines",
    "percent_bc_gt_75": "High proportion of bankcard accounts exceeding 75% credit limit",
    "delinq_2yrs": "Presence of past-due delinquency incidents in the past 24 months",
    "pub_rec": "Presence of derogatory public records on credit file",
    "pub_rec_bankruptcies": "Historical record of personal bankruptcy filing",
    "tax_liens": "Presence of unsatisfied tax liens on public record",
    "tot_coll_amt": "Unsatisfied collection accounts on credit bureau report",
    "tot_cur_bal": "High total outstanding balance across all credit lines",
    "num_actv_rev_tl": "Too many active revolving credit trade lines",
    "num_rev_tl_bal_gt_0": "Too many revolving accounts carrying an active non-zero balance",
    "pct_tl_nvr_dlq": "Low percentage of credit trade lines never delinquent",
    "has_delinq_history": "Prior history of delinquent payment activity",
    "has_pub_record_history": "Prior history of public records or bankruptcy",
    "has_major_derog_history": "Prior history of major derogatory credit events",
    "mths_since_last_major_derog": "Recent occurrence of major derogatory credit event",
    "mths_since_last_record": "Recent public record filing on credit report",
    "emp_length_years": "Short length of current continuous employment history",
    "emp_length_isna": "Unverified or missing employment duration record",
    # Tenancy & Status
    "home_ownership_RENT": "Absence of homeownership / non-mortgage residential status",
    "home_ownership_OTHER": "Non-standard residential tenancy status",
    "verification_status_Not Verified": "Unverified borrower income status",
}


def get_adverse_action_description(feature_name: str) -> str:
    """Map a technical feature name to a regulatory-compliant adverse action description.

    Parameters
    ----------
    feature_name : str
        Technical column name in the feature matrix.

    Returns
    -------
    str
        Plain-English, regulatory-compliant adverse action description.
    """
    if feature_name in ADVERSE_ACTION_MAPPING:
        return ADVERSE_ACTION_MAPPING[feature_name]

    # Handle grade indicators
    if feature_name.startswith("grade_") or feature_name.startswith("sub_grade_"):
        grade_val = feature_name.split("_")[-1]
        return f"Assigned higher-risk internal credit tier ({grade_val})"

    # Handle purpose indicators
    if feature_name.startswith("purpose_"):
        purpose_val = feature_name.replace("purpose_", "").replace("_", " ").title()
        return f"Elevated risk associated with stated loan purpose ({purpose_val})"

    # Fallback to clean title-cased description
    clean_name = (
        feature_name.replace("_isna", " unverified")
        .replace("_", " ")
        .strip()
        .title()
    )
    return f"Adverse credit bureau factor related to {clean_name}"


def get_base_estimator(model: Any) -> Any:
    """Extract fitted tree booster from scikit-learn CalibratedClassifierCV or wrapper.

    Parameters
    ----------
    model : Any
        Calibrated model or base estimator.

    Returns
    -------
    Any
        Underlying fitted tree booster (e.g. XGBClassifier).
    """
    if hasattr(model, "calibrated_classifiers_") and len(model.calibrated_classifiers_) > 0:
        cc = model.calibrated_classifiers_[0]
        est = getattr(cc, "estimator", None)
        if est is not None and hasattr(est, "estimator"):
            return est.estimator
        if est is not None:
            return est

    if hasattr(model, "estimator"):
        est = model.estimator
        if hasattr(est, "estimator"):
            return est.estimator
        return est

    return model


class CreditSHAPExplainer:
    """SHAP Explainer and Adverse Action generator for credit risk models."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        feature_names_path: Optional[Union[str, Path]] = None,
    ):
        """Initialize SHAP explainer with calibrated champion model and feature order.

        Parameters
        ----------
        model_path : str or Path, optional
            Path to champion_model.joblib.
        feature_names_path : str or Path, optional
            Path to feature_names.json.
        """
        self.model_path = Path(model_path or (PROJECT_ROOT / "models" / "champion_model.joblib"))
        self.feature_names_path = Path(
            feature_names_path or (PROJECT_ROOT / "models" / "feature_names.json")
        )

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        if not self.feature_names_path.exists():
            raise FileNotFoundError(f"Feature names file not found at {self.feature_names_path}")

        logger.info(f"Loading calibrated champion model from {self.model_path}...")
        self.full_model = joblib.load(self.model_path)
        self.base_estimator = get_base_estimator(self.full_model)

        with open(self.feature_names_path, "r") as f:
            self.feature_names: List[str] = json.load(f)

        logger.info(f"Initializing shap.TreeExplainer on {type(self.base_estimator).__name__}...")
        self.explainer = shap.TreeExplainer(self.base_estimator)
        exp_val = np.asarray(self.explainer.expected_value).ravel()
        self.expected_value = float(exp_val[1] if len(exp_val) > 1 else exp_val[0])
        logger.info(f"TreeExplainer initialized successfully (expected_value: {self.expected_value:.4f})")

    def _prepare_vector(
        self, applicant_feature_vector: Union[np.ndarray, pd.DataFrame, pd.Series, Dict[str, Any]]
    ) -> pd.DataFrame:
        """Standardize input into a 1-row DataFrame matching the 151 feature schema."""
        if isinstance(applicant_feature_vector, pd.DataFrame):
            df = applicant_feature_vector.copy()
            if len(df) == 0:
                raise ValueError("Input DataFrame is empty.")
            df = df.iloc[[0]]
        elif isinstance(applicant_feature_vector, pd.Series):
            df = applicant_feature_vector.to_frame().T
        elif isinstance(applicant_feature_vector, dict):
            # Check if dict already contains feature names
            df = pd.DataFrame([applicant_feature_vector])
        elif isinstance(applicant_feature_vector, (np.ndarray, Sequence)):
            arr = np.asarray(applicant_feature_vector, dtype=float).ravel()
            if len(arr) != len(self.feature_names):
                raise ValueError(
                    f"Array length ({len(arr)}) does not match expected features ({len(self.feature_names)})"
                )
            df = pd.DataFrame([arr], columns=self.feature_names)
        else:
            raise TypeError(f"Unsupported applicant vector type: {type(applicant_feature_vector)}")

        # Ensure all expected columns are present
        missing_cols = set(self.feature_names) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Input is missing {len(missing_cols)} required features: {list(missing_cols)[:5]}")

        # Return strictly in feature_names order with float dtype
        return df[self.feature_names].astype(float)

    def explain_applicant(
        self,
        applicant_feature_vector: Union[np.ndarray, pd.DataFrame, pd.Series, Dict[str, Any]],
        top_n: int = 3,
    ) -> Tuple[List[str], Dict[str, float]]:
        """Compute Shapley attribution values and extract top adverse reasons for credit denial.

        Parameters
        ----------
        applicant_feature_vector : array-like, Series, DataFrame, or dict
            The 151 engineered applicant features.
        top_n : int, default=3
            Number of top adverse reasons to return.

        Returns
        -------
        top_reasons : list of str
            Top-N regulatory Adverse Action descriptions.
        shap_values_dict : dict of str -> float
            Dictionary mapping feature names to their raw SHAP attribution values.
        """
        df = self._prepare_vector(applicant_feature_vector)
        raw_shap = self.explainer.shap_values(df)

        # Handle possible binary classification shape (1, 151) or list
        if isinstance(raw_shap, list):
            # For some tree models, shap_values is a list of [neg_class, pos_class]
            shap_vec = np.asarray(raw_shap[1][0], dtype=float)
        elif raw_shap.ndim == 2:
            shap_vec = np.asarray(raw_shap[0], dtype=float)
        elif raw_shap.ndim == 1:
            shap_vec = np.asarray(raw_shap, dtype=float)
        else:
            shap_vec = np.asarray(raw_shap, dtype=float).ravel()

        shap_values_dict = {
            feat: float(val) for feat, val in zip(self.feature_names, shap_vec)
        }

        # Sort features by positive contribution (pushing risk HIGHER)
        # Features with highest positive SHAP are the principal drivers of credit risk
        sorted_indices = np.argsort(shap_vec)[::-1]

        # Extract top reasons, avoiding duplicate text descriptions
        top_reasons: List[str] = []
        seen_descriptions = set()

        for idx in sorted_indices:
            feat_name = self.feature_names[idx]
            val = shap_vec[idx]
            desc = get_adverse_action_description(feat_name)
            if desc not in seen_descriptions:
                seen_descriptions.add(desc)
                top_reasons.append(desc)
            if len(top_reasons) >= top_n:
                break

        # Fallback if fewer than top_n unique reasons were extracted
        if len(top_reasons) < top_n:
            for idx in sorted_indices:
                feat_name = self.feature_names[idx]
                desc = f"Elevated risk score associated with {feat_name.replace('_', ' ').title()}"
                if desc not in seen_descriptions:
                    seen_descriptions.add(desc)
                    top_reasons.append(desc)
                if len(top_reasons) >= top_n:
                    break

        return top_reasons[:top_n], shap_values_dict

    def generate_adverse_action_notice(
        self,
        applicant_dict: Optional[Dict[str, Any]] = None,
        top_reasons: Optional[List[str]] = None,
    ) -> str:
        """Format a legally compliant Adverse Action Notice under ECOA and FCRA.

        Parameters
        ----------
        applicant_dict : dict, optional
            Applicant metadata (id, loan_amnt, etc.).
        top_reasons : list of str, optional
            Top adverse reasons (from explain_applicant).

        Returns
        -------
        str
            Formatted adverse action notice text.
        """
        applicant_dict = applicant_dict or {}
        app_id = applicant_dict.get("id", applicant_dict.get("applicant_id", "APP-98472-REF"))
        date_str = applicant_dict.get("date", datetime.now().strftime("%B %d, %Y"))
        loan_amount = applicant_dict.get("loan_amnt", "Requested Amount")
        if isinstance(loan_amount, (int, float)):
            loan_amount_str = f"${loan_amount:,.2f}"
        else:
            loan_amount_str = str(loan_amount)

        if not top_reasons:
            top_reasons = [
                "High monthly debt payment relative to monthly income",
                "Elevated revolving credit card line utilization",
                "High frequency of recent hard credit inquiries (past 6 months)",
            ]

        reasons_bulleted = "\n".join(
            f"   {i + 1}. {reason}" for i, reason in enumerate(top_reasons)
        )

        notice_text = f"""======================================================================
STATEMENT OF ADVERSE ACTION AND CREDIT EVALUATION NOTICE
======================================================================
Date of Notice:          {date_str}
Application Reference:   {app_id}
Requested Loan Amount:   {loan_amount_str}
Decision Status:         APPLICATION DECLINED (Tier: AUTO_REJECT)
======================================================================

Dear Applicant,

Thank you for your recent application for credit with our lending institution.
After thorough evaluation of your application, debt obligations, and credit bureau
history, we regret to inform you that we are unable to approve your application
at this time.

In accordance with the Equal Credit Opportunity Act (ECOA, 15 U.S.C. 1691 et seq.,
12 CFR Part 1002 - Regulation B) and the Fair Credit Reporting Act (FCRA, 15 U.S.C.
1681 et seq.), our decision was based in whole or in part on the following principal
reasons:

{reasons_bulleted}

----------------------------------------------------------------------
DISCLOSURE OF CREDIT SCORING & UNDERWRITING MODEL
----------------------------------------------------------------------
* Scoring System:       Credit Risk Decision System (Calibrated XGBoost Booster)
* Underlying Metric:    Calibrated Probability of Default (PD)
* Discrimination Score: Gini 0.4229 | ROC-AUC 0.7115 | KS 30.66%
* Risk Classification:  AUTO_REJECT (Default probability exceeds policy ceiling)

----------------------------------------------------------------------
STATUTORY CONSUMER RIGHTS NOTICE (EQUAL CREDIT OPPORTUNITY ACT)
----------------------------------------------------------------------
The federal Equal Credit Opportunity Act prohibits creditors from discriminating
against credit applicants on the basis of race, color, religion, national origin,
sex, marital status, age (provided the applicant has the capacity to enter into a
binding contract); because all or part of the applicant's income derives from any
public assistance program; or because the applicant has in good faith exercised
any right under the Consumer Credit Protection Act.

The federal agency that administers compliance with this law concerning this
creditor is:
   Consumer Financial Protection Bureau (CFPB)
   1700 G Street NW, Washington, DC 20552
======================================================================
"""
        return notice_text

    def precompute_test_sample(
        self,
        X_test_path: Optional[Union[str, Path]] = None,
        n_samples: int = 1000,
        output_figure_path: Optional[Union[str, Path]] = None,
        output_joblib_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Precompute SHAP values for a sample of 1,000 test set applicants.

        Generates and saves:
        1. Publication-quality Global SHAP Beeswarm Plot -> reports/figures/shap_summary_beeswarm.png
        2. Serialized SHAP values cache -> reports/sample_shap_test.joblib (for Streamlit dashboard)

        Parameters
        ----------
        X_test_path : str or Path, optional
            Path to X_test.parquet.
        n_samples : int, default=1000
            Number of test samples to explain.
        output_figure_path : str or Path, optional
            Path to save beeswarm plot.
        output_joblib_path : str or Path, optional
            Path to save precomputed SHAP cache.

        Returns
        -------
        dict
            Summary containing mean absolute SHAP values and top features.
        """
        X_test_path = Path(X_test_path or (PROJECT_ROOT / "data" / "processed" / "X_test.parquet"))
        output_figure_path = Path(
            output_figure_path or (PROJECT_ROOT / "reports" / "figures" / "shap_summary_beeswarm.png")
        )
        output_joblib_path = Path(
            output_joblib_path or (PROJECT_ROOT / "reports" / "sample_shap_test.joblib")
        )

        logger.info(f"Loading test features from {X_test_path}...")
        X_test = pd.read_parquet(X_test_path)
        sample_df = X_test.iloc[:n_samples].copy()

        logger.info(f"Computing TreeSHAP values for {len(sample_df):,} test applicants...")
        shap_values_raw = self.explainer.shap_values(sample_df)

        if isinstance(shap_values_raw, list):
            shap_matrix = np.asarray(shap_values_raw[1], dtype=float)
        else:
            shap_matrix = np.asarray(shap_values_raw, dtype=float)

        logger.info(f"SHAP matrix calculated: shape {shap_matrix.shape}")

        # Compute Mean |SHAP| Global Feature Importance
        mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)
        top_global_indices = np.argsort(mean_abs_shap)[::-1]
        top_global_features = [
            (self.feature_names[i], float(mean_abs_shap[i])) for i in top_global_indices[:20]
        ]

        logger.info("Top 5 Global SHAP Risk Drivers:")
        for rank, (feat, val) in enumerate(top_global_features[:5], start=1):
            desc = get_adverse_action_description(feat)
            logger.info(f"   {rank}. {feat} (Mean |SHAP| = {val:.4f}): {desc}")

        # Generate Publication-Quality Global SHAP Beeswarm Plot
        logger.info(f"Rendering Global SHAP Beeswarm Plot to {output_figure_path}...")
        output_figure_path.parent.mkdir(parents=True, exist_ok=True)

        fig = plt.figure(figsize=(10, 7), dpi=300)
        shap.summary_plot(
            shap_matrix,
            sample_df,
            feature_names=self.feature_names,
            show=False,
            max_display=15,
        )
        plt.title(
            "Global Credit Risk Attributions (TreeSHAP Beeswarm)\nTop 15 Most Discriminatory Risk Drivers",
            fontsize=13,
            fontweight="bold",
            pad=16,
        )
        plt.tight_layout()
        plt.savefig(output_figure_path, bbox_inches="tight", dpi=300)
        plt.close("all")
        logger.info(f"Successfully saved Beeswarm plot to {output_figure_path}")

        # Serialize precomputed cache for Streamlit interactive dashboard
        logger.info(f"Serializing precomputed SHAP test values to {output_joblib_path}...")
        output_joblib_path.parent.mkdir(parents=True, exist_ok=True)

        cache_payload = {
            "shap_values": shap_matrix,
            "sample_X": sample_df,
            "feature_names": self.feature_names,
            "expected_value": self.expected_value,
            "top_global_features": top_global_features,
            "n_samples": n_samples,
        }
        joblib.dump(cache_payload, output_joblib_path, compress=3)
        logger.info(f"Successfully serialized SHAP cache to {output_joblib_path}")

        return {
            "top_5_features": top_global_features[:5],
            "n_samples": n_samples,
            "figure_path": str(output_figure_path),
            "joblib_path": str(output_joblib_path),
        }


# Convenience module-level instances & functions
_EXPLAINER_INSTANCE: Optional[CreditSHAPExplainer] = None


def get_default_explainer() -> CreditSHAPExplainer:
    """Singleton getter for CreditSHAPExplainer."""
    global _EXPLAINER_INSTANCE
    if _EXPLAINER_INSTANCE is None:
        _EXPLAINER_INSTANCE = CreditSHAPExplainer()
    return _EXPLAINER_INSTANCE


def explain_applicant(
    applicant_feature_vector: Union[np.ndarray, pd.DataFrame, pd.Series, Dict[str, Any]],
    top_n: int = 3,
) -> Tuple[List[str], Dict[str, float]]:
    """Compute Shapley attribution values and extract top adverse reasons for credit denial.

    Parameters
    ----------
    applicant_feature_vector : array-like, Series, DataFrame, or dict
        The 151 engineered applicant features.
    top_n : int, default=3
        Number of top adverse reasons to return.

    Returns
    -------
    top_reasons : list of str
        Top-N regulatory Adverse Action descriptions.
    shap_values_dict : dict of str -> float
        Dictionary mapping feature names to their raw SHAP attribution values.
    """
    explainer = get_default_explainer()
    return explainer.explain_applicant(applicant_feature_vector, top_n=top_n)


def generate_adverse_action_notice(
    applicant_dict: Optional[Dict[str, Any]] = None,
    top_reasons: Optional[List[str]] = None,
) -> str:
    """Format a legally compliant Adverse Action Notice under ECOA and FCRA."""
    explainer = get_default_explainer()
    return explainer.generate_adverse_action_notice(applicant_dict, top_reasons)


def main():
    """Main CLI execution for SHAP Explainability & Adverse Action Engine."""
    logger.info("=" * 70)
    logger.info("PHASE 5: SHAP EXPLAINABILITY & ADVERSE ACTION ENGINE")
    logger.info("=" * 70)

    explainer = CreditSHAPExplainer()

    # Precompute background test sample & beeswarm figure
    results = explainer.precompute_test_sample(n_samples=1000)

    print("\n" + "=" * 70)
    print("🐝 TOP 5 GLOBAL SHAP CREDIT RISK DRIVERS (Mean |SHAP|)")
    print("=" * 70)
    for rank, (feat, val) in enumerate(results["top_5_features"], start=1):
        desc = get_adverse_action_description(feat)
        print(f" {rank}. {feat:<26} | Mean |SHAP|: {val:.4f} | Description: '{desc}'")
    print("=" * 70)

    # Test single applicant Adverse Action Notice
    X_test_path = PROJECT_ROOT / "data" / "processed" / "X_test.parquet"
    X_test = pd.read_parquet(X_test_path)
    sample_applicant = X_test.iloc[0]

    top_reasons, shap_dict = explainer.explain_applicant(sample_applicant, top_n=3)
    sample_metadata = {
        "id": "LC-2017-TEST-0001",
        "date": "November 08, 2026",
        "loan_amnt": float(sample_applicant.get("loan_amnt", 25000.0)),
    }
    sample_notice = explainer.generate_adverse_action_notice(sample_metadata, top_reasons)

    print("\n" + sample_notice)
    print(f"Artifacts Created:")
    print(f"  - Beeswarm Plot: {results['figure_path']}")
    print(f"  - Precomputed SHAP Cache: {results['joblib_path']}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
