"""Unit and Integration Tests for Phase 4: Model Training, Benchmarking & Calibration.

Verifies:
1. Champion model serialization, schema, and loadability.
2. Prediction probability validity (strictly in [0, 1], zero NaNs, zero Infs).
3. Benchmark superiority: Champion Gini > Logistic Regression baseline Gini.
4. Calibration quality: Champion Brier score < 0.20.
5. Runs log schema and completeness across all partitions.
6. Metric calculation functions (ROC-AUC, Gini, KS, Brier, Bootstrap CI).
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

from src.models.evaluate import (
    bootstrap_gini_ci,
    brier_score_loss,
    evaluate_predictions,
    gini_score,
    ks_statistic,
    roc_auc_score,
)

RUNS_LOG_PATH = Path("models/runs_log.csv")
CHAMPION_MODEL_PATH = Path("models/champion_model.joblib")
CALIBRATION_PLOT_PATH = Path("reports/figures/calibration_curves.png")


def test_metric_functions_behavior():
    """Verify evaluation metric calculations with known mathematical cases."""
    # Perfect separation case
    y_true_perfect = np.array([0, 0, 0, 1, 1, 1])
    y_prob_perfect = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

    auc_perf = roc_auc_score(y_true_perfect, y_prob_perfect)
    gini_perf = gini_score(y_true_perfect, y_prob_perfect)
    ks_perf = ks_statistic(y_true_perfect, y_prob_perfect)
    brier_perf = brier_score_loss(y_true_perfect, y_prob_perfect)

    assert np.isclose(auc_perf, 1.0)
    assert np.isclose(gini_perf, 1.0)
    assert np.isclose(ks_perf, 1.0)
    assert brier_perf < 0.10

    # Random coin flip case
    y_true_random = np.array([0, 1, 0, 1, 0, 1])
    y_prob_random = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

    auc_rand = roc_auc_score(y_true_random, y_prob_random)
    gini_rand = gini_score(y_true_random, y_prob_random)
    ks_rand = ks_statistic(y_true_random, y_prob_random)

    assert np.isclose(auc_rand, 0.5)
    assert np.isclose(gini_rand, 0.0)
    assert np.isclose(ks_rand, 0.0)

    # Dictionary helper
    summary = evaluate_predictions(y_true_perfect, y_prob_perfect)
    assert "roc_auc" in summary
    assert "gini" in summary
    assert "ks_stat" in summary
    assert "brier_score" in summary


def test_bootstrap_gini_ci():
    """Verify bootstrap confidence interval calculation."""
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=500)
    y_prob = rng.uniform(0, 1, size=500)

    lower_ci, upper_ci = bootstrap_gini_ci(y_true, y_prob, n_bootstraps=50, ci=0.95, random_state=42)
    assert isinstance(lower_ci, float)
    assert isinstance(upper_ci, float)
    assert lower_ci <= upper_ci
    assert -1.0 <= lower_ci <= 1.0
    assert -1.0 <= upper_ci <= 1.0


def test_champion_model_exists_and_loads():
    """Verify that models/champion_model.joblib exists and loads cleanly."""
    assert CHAMPION_MODEL_PATH.exists(), f"Missing champion model file: {CHAMPION_MODEL_PATH}"
    champion = joblib.load(CHAMPION_MODEL_PATH)
    assert hasattr(champion, "predict_proba"), "Champion model must implement predict_proba"
    assert hasattr(champion, "predict"), "Champion model must implement predict"


def test_calibration_plot_exists():
    """Verify reliability diagram plot is generated and non-empty."""
    assert CALIBRATION_PLOT_PATH.exists(), f"Missing calibration plot: {CALIBRATION_PLOT_PATH}"
    assert CALIBRATION_PLOT_PATH.stat().st_size > 1000, "Calibration plot file is too small"


def test_runs_log_integrity():
    """Verify models/runs_log.csv schema, completeness, and value ranges."""
    assert RUNS_LOG_PATH.exists(), f"Missing runs log: {RUNS_LOG_PATH}"
    df_runs = pd.read_csv(RUNS_LOG_PATH)

    required_cols = [
        "model_name",
        "split",
        "roc_auc",
        "gini",
        "ks_stat",
        "brier_score",
        "sample_count",
        "default_rate",
    ]
    for col in required_cols:
        assert col in df_runs.columns, f"Missing column {col} in runs_log.csv"

    # All splits represented
    expected_splits = {"train", "val", "test", "crisis_stress"}
    assert set(df_runs["split"].unique()) == expected_splits

    # Core models represented
    model_names = set(df_runs["model_name"].unique())
    expected_models = {
        "Approve-Everyone Baseline",
        "Logistic Regression Scorecard",
        "Lending Club Grade Classifier",
        "Random Forest",
        "XGBoost",
        "LightGBM",
        "Option Z Booster (No Grade)",
        "XGBoost (Platt Scaling)",
        "XGBoost (Isotonic)",
    }
    for model in expected_models:
        assert model in model_names, f"Model {model} missing from runs_log.csv"

    # Check for NaNs
    assert df_runs["gini"].isna().sum() == 0, "Gini scores contain NaNs"
    assert df_runs["brier_score"].isna().sum() == 0, "Brier scores contain NaNs"
    assert df_runs["ks_stat"].isna().sum() == 0, "KS statistics contain NaNs"


def test_champion_predict_proba_bounds_and_validity():
    """Verify output probabilities are strictly within [0, 1] with 0 NaNs."""
    champion = joblib.load(CHAMPION_MODEL_PATH)
    X_test = pd.read_parquet("data/processed/X_test.parquet").iloc[:1000]

    probas = champion.predict_proba(X_test)
    assert probas.shape == (1000, 2), f"Expected shape (1000, 2), got {probas.shape}"
    assert np.isnan(probas).sum() == 0, "Probabilities contain NaNs"
    assert np.isinf(probas).sum() == 0, "Probabilities contain Infs"
    assert (probas >= 0.0).all(), "Found negative probabilities"
    assert (probas <= 1.0).all(), "Found probabilities exceeding 1.0"

    # Probabilities row sum must equal 1.0
    row_sums = probas.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5), "Probabilities row sums do not equal 1.0"


def test_champion_outperforms_logistic_baseline_and_meets_brier_target():
    """Assert champion model Gini > Logistic Regression baseline Gini on test set, and Brier < 0.20."""
    df_runs = pd.read_csv(RUNS_LOG_PATH)
    df_test = df_runs[df_runs["split"] == "test"]

    lr_row = df_test[df_test["model_name"] == "Logistic Regression Scorecard"]
    champ_row = df_test[df_test["model_name"] == "XGBoost (Platt Scaling)"]

    assert not lr_row.empty, "Logistic Regression test entry missing"
    assert not champ_row.empty, "Champion test entry missing"

    lr_gini = float(lr_row["gini"].iloc[0])
    champ_gini = float(champ_row["gini"].iloc[0])
    champ_brier = float(champ_row["brier_score"].iloc[0])
    champ_ks = float(champ_row["ks_stat"].iloc[0])

    print(f"\n[Test Comparison] Champion Gini: {champ_gini:.4f} vs LR Gini: {lr_gini:.4f}")
    print(f"[Test Comparison] Champion Brier Score: {champ_brier:.4f}")
    print(f"[Test Comparison] Champion KS: {champ_ks:.4f}")

    # Gini superiority
    assert champ_gini > lr_gini, f"Champion Gini ({champ_gini:.4f}) must exceed LR Gini ({lr_gini:.4f})"
    assert champ_gini > 0.40, f"Champion Gini ({champ_gini:.4f}) should exceed 0.40"

    # Brier score target
    assert champ_brier < 0.20, f"Champion Brier score ({champ_brier:.4f}) must be < 0.20"

    # KS target
    assert champ_ks > 0.25, f"Champion KS ({champ_ks:.4f}) must be > 0.25"


def test_option_z_retains_discriminatory_power():
    """Verify Option Z ablation retains at least 95% of discriminatory power without grade."""
    df_runs = pd.read_csv(RUNS_LOG_PATH)
    df_test = df_runs[df_runs["split"] == "test"]

    xgb_full = float(df_test[df_test["model_name"] == "XGBoost"]["gini"].iloc[0])
    xgb_no_grade = float(df_test[df_test["model_name"] == "Option Z Booster (No Grade)"]["gini"].iloc[0])

    retention_pct = (xgb_no_grade / xgb_full) * 100.0
    print(f"\n[Option Z] Full Gini: {xgb_full:.4f} | No-Grade Gini: {xgb_no_grade:.4f} | Retention: {retention_pct:.2f}%")

    assert retention_pct >= 95.0, f"Option Z retained power ({retention_pct:.2f}%) fell below 95%"
