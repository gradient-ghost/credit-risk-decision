"""Unit & Integration Tests for Phase 5: SHAP Explainability & Adverse Action Engine.

Verifies:
1. reports/figures/shap_summary_beeswarm.png exists and is non-empty.
2. reports/sample_shap_test.joblib loads cleanly with valid schema.
3. explain_applicant returns exactly 3 adverse reasons as non-empty human-readable strings.
4. generate_adverse_action_notice formats a compliant notice.
5. Adverse action description mapping handles key features and fallbacks.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

from src.explainability.shap_explainer import (
    ADVERSE_ACTION_MAPPING,
    CreditSHAPExplainer,
    explain_applicant,
    generate_adverse_action_notice,
    get_adverse_action_description,
    get_default_explainer,
)

BEESWARM_PLOT_PATH = Path("reports/figures/shap_summary_beeswarm.png")
SHAP_CACHE_PATH = Path("reports/sample_shap_test.joblib")
X_TEST_PATH = Path("data/processed/X_test.parquet")


def test_shap_artifacts_exist():
    """Assert reports/figures/shap_summary_beeswarm.png exists and has valid size."""
    assert BEESWARM_PLOT_PATH.exists(), f"Missing file: {BEESWARM_PLOT_PATH}"
    assert BEESWARM_PLOT_PATH.stat().st_size > 10000, "Beeswarm plot file is suspiciously small or empty"


def test_shap_cache_loads_cleanly():
    """Assert reports/sample_shap_test.joblib loads cleanly with expected payload schema."""
    assert SHAP_CACHE_PATH.exists(), f"Missing file: {SHAP_CACHE_PATH}"

    cache = joblib.load(SHAP_CACHE_PATH)
    assert isinstance(cache, dict), "Cache payload must be a dictionary"

    assert "shap_values" in cache
    assert "sample_X" in cache
    assert "feature_names" in cache
    assert "expected_value" in cache
    assert "top_global_features" in cache

    shap_values = cache["shap_values"]
    sample_X = cache["sample_X"]
    feature_names = cache["feature_names"]

    assert shap_values.shape == (1000, 151), f"Expected shape (1000, 151), got {shap_values.shape}"
    assert len(sample_X) == 1000
    assert len(feature_names) == 151
    assert not np.isnan(shap_values).any(), "SHAP matrix must contain zero NaNs"


def test_explain_applicant_returns_three_reasons():
    """Pass a sample applicant vector to explain_applicant(vector, top_n=3) and assert 3 reasons."""
    assert X_TEST_PATH.exists(), f"Missing file: {X_TEST_PATH}"
    X_test = pd.read_parquet(X_TEST_PATH)
    sample_vector = X_test.iloc[0]

    top_reasons, shap_values = explain_applicant(sample_vector, top_n=3)

    # Asserts returns exactly 3 adverse reasons
    assert len(top_reasons) == 3, f"Expected exactly 3 adverse reasons, got {len(top_reasons)}"

    # Asserts reasons are non-empty human-readable strings
    for reason in top_reasons:
        assert isinstance(reason, str), f"Reason must be a string, got {type(reason)}"
        assert len(reason.strip()) > 5, f"Reason string '{reason}' is suspiciously short"

    # Raw SHAP values dictionary validation
    assert isinstance(shap_values, dict)
    assert len(shap_values) == 151
    assert all(isinstance(v, float) for v in shap_values.values())


def test_explain_applicant_numpy_and_series():
    """Assert explain_applicant supports both pandas Series and numpy 1D array."""
    X_test = pd.read_parquet(X_TEST_PATH)
    sample_series = X_test.iloc[10]
    sample_array = sample_series.values

    reasons_series, _ = explain_applicant(sample_series, top_n=3)
    reasons_array, _ = explain_applicant(sample_array, top_n=3)

    assert len(reasons_series) == 3
    assert len(reasons_array) == 3
    assert reasons_series == reasons_array, "Series and ndarray inputs should yield identical explanations"


def test_adverse_action_notice_generation():
    """Assert generate_adverse_action_notice creates a compliant, well-formatted notice."""
    applicant_dict = {
        "id": "TEST-APP-2018-9999",
        "loan_amnt": 18500.0,
        "date": "November 08, 2026",
    }
    reasons = [
        "High monthly debt payment relative to monthly income",
        "Elevated revolving credit card line utilization",
        "Below average credit bureau score",
    ]

    notice = generate_adverse_action_notice(applicant_dict, reasons)

    assert "STATEMENT OF ADVERSE ACTION" in notice
    assert "TEST-APP-2018-9999" in notice
    assert "$18,500.00" in notice
    assert "Equal Credit Opportunity Act" in notice
    assert "Fair Credit Reporting Act" in notice

    for r in reasons:
        assert r in notice


def test_adverse_action_description_mappings():
    """Assert adverse action mapping produces compliant strings for standard features."""
    expected_mappings = [
        ("payment_to_income_ratio", "High monthly debt payment relative to monthly income"),
        ("revol_util_clean", "Elevated revolving credit card line utilization"),
        ("credit_hunger_velocity", "High frequency of recent hard credit inquiries (past 6 months)"),
        ("dti", "High aggregate debt-to-income ratio"),
        ("fico_score", "Below average credit bureau score"),
        ("credit_history_age_years", "Insufficient length of established credit history"),
        ("account_breadth", "High proportion of active credit accounts"),
        ("delinq_recency_imputed", "Recent record of past-due payment delinquency"),
    ]

    for feat_name, expected_text in expected_mappings:
        assert feat_name in ADVERSE_ACTION_MAPPING
        assert get_adverse_action_description(feat_name) == expected_text

    # Test clean fallback
    fallback_desc = get_adverse_action_description("unknown_credit_metric")
    assert isinstance(fallback_desc, str)
    assert len(fallback_desc) > 0
    assert "Unknown Credit Metric" in fallback_desc
