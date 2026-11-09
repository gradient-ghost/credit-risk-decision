"""Tests for Phase 7 Streamlit Dashboard and Components.

Validates:
1. Component, tab, and app module imports.
2. Cached data and model loader integrity.
3. Vectorized sub-10ms policy simulation math consistency.
4. Population Stability Index (PSI) calculation bounds and accuracy.
5. Visual card renderers and Plotly figure generation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_dashboard_imports():
    """Verify all dashboard components, tabs, and app orchestrator import cleanly."""
    import dashboard
    import dashboard.components.cards as cards
    import dashboard.components.data_loader as data_loader
    import dashboard.components.plots as plots
    import dashboard.tabs.tab1_executive as tab1
    import dashboard.tabs.tab2_policy_sim as tab2
    import dashboard.tabs.tab3_applicant as tab3
    import dashboard.tabs.tab4_stability as tab4
    import dashboard.tabs.tab5_monitoring as tab5
    import dashboard.app as app

    assert callable(app.main)
    assert callable(tab1.render_tab1)
    assert callable(tab2.render_tab2)
    assert callable(tab3.render_tab3)
    assert callable(tab4.render_tab4)
    assert callable(tab5.render_tab5)
    assert callable(data_loader.load_champion_model)
    assert callable(cards.render_kpi_card)
    assert callable(plots.plot_interactive_calibration_curve)


def test_load_champion_model():
    """Verify champion model and feature pipeline load with inference capabilities."""
    from dashboard.components.data_loader import load_champion_model

    loaded = load_champion_model()
    assert isinstance(loaded, dict)
    assert "model" in loaded
    assert "pipeline" in loaded
    assert hasattr(loaded["model"], "predict_proba")
    assert hasattr(loaded["pipeline"], "transform")


def test_load_policy_thresholds():
    """Verify policy thresholds JSON contains regulatory configuration and test metrics."""
    from dashboard.components.data_loader import load_policy_thresholds

    thresholds = load_policy_thresholds()
    assert isinstance(thresholds, dict)
    assert thresholds.get("tau_low") == 0.25
    assert thresholds.get("tau_high") == 0.35
    assert "test_eval" in thresholds
    test_eval = thresholds["test_eval"]
    assert test_eval["total_applicants"] == 225639
    assert test_eval["auto_approve"]["count"] == 138709
    assert test_eval["manual_review"]["count"] == 40620
    assert test_eval["auto_reject"]["count"] == 46310


def test_load_runs_log():
    """Verify model tournament runs log loads with expected schema and entries."""
    from dashboard.components.data_loader import load_runs_log

    runs = load_runs_log()
    assert isinstance(runs, pd.DataFrame)
    assert not runs.empty
    expected_cols = {"model_name", "split", "roc_auc", "gini", "ks_stat", "brier_score"}
    assert expected_cols.issubset(set(runs.columns))

    # Champion model presence
    models = runs["model_name"].unique()
    assert any("Platt Scaling" in m for m in models)


def test_load_shap_cache():
    """Verify precomputed SHAP cache contains 1,000 applicant attributions and schema."""
    from dashboard.components.data_loader import load_shap_cache

    cache = load_shap_cache()
    assert isinstance(cache, dict)
    assert "shap_values" in cache
    assert "sample_X" in cache
    assert "feature_names" in cache
    assert "expected_value" in cache

    shap_vals = cache["shap_values"]
    assert shap_vals.shape == (1000, 151)
    assert len(cache["feature_names"]) == 151
    assert len(cache["sample_X"]) == 1000


def test_load_test_summary_dataset():
    """Verify test summary dataset contains 225,639 loans with valid columns and ranges."""
    from dashboard.components.data_loader import load_test_summary_dataset

    summary = load_test_summary_dataset()
    assert isinstance(summary, pd.DataFrame)
    assert len(summary) == 225639

    expected_cols = [
        "loan_amnt",
        "target",
        "issue_year",
        "grade",
        "purpose",
        "home_ownership",
        "predicted_pd",
    ]
    for col in expected_cols:
        assert col in summary.columns

    # Verify probability bounds
    assert (summary["predicted_pd"] >= 0.0).all()
    assert (summary["predicted_pd"] <= 1.0).all()
    assert set(summary["target"].unique()).issubset({0, 1})


def test_vectorized_policy_simulation_accuracy():
    """Verify vectorized policy calculations match official test evaluation benchmarks."""
    from dashboard.components.data_loader import (
        load_policy_thresholds,
        load_test_summary_dataset,
    )

    summary = load_test_summary_dataset()
    thresholds = load_policy_thresholds()
    test_eval = thresholds["test_eval"]

    preds = summary["predicted_pd"].values
    loans = summary["loan_amnt"].values
    targets = summary["target"].values

    tau_low = 0.25
    tau_high = 0.35
    lgd = 1.0

    mask_approve = preds < tau_low
    mask_reject = preds > tau_high
    mask_review = (~mask_approve) & (~mask_reject)

    c_app = int(np.sum(mask_approve))
    c_rev = int(np.sum(mask_review))
    c_rej = int(np.sum(mask_reject))

    assert c_app == test_eval["auto_approve"]["count"]
    assert c_rev == test_eval["manual_review"]["count"]
    assert c_rej == test_eval["auto_reject"]["count"]

    # Verify financial loss calculations
    el_saved = float(np.sum(preds[mask_reject] * loans[mask_reject] * lgd))
    expected_saved = test_eval["auto_reject"]["expected_losses_saved"]
    assert abs(el_saved - expected_saved) < 1.0  # Within $1 tolerance


def test_psi_calculation_math_and_bounds():
    """Verify Population Stability Index (PSI) algorithm bounds and stability."""
    from dashboard.components.data_loader import calculate_psi, compute_psi_table

    rng = np.random.default_rng(42)

    # Identical distributions must yield PSI approx 0.0
    dist_a = rng.normal(100, 15, 5000)
    psi_identical = calculate_psi(dist_a, dist_a.copy(), num_bins=10)
    assert psi_identical == pytest.approx(0.0, abs=1e-4)

    # Distinct distributions must yield positive PSI
    dist_b = rng.normal(110, 20, 5000)
    psi_shifted = calculate_psi(dist_a, dist_b, num_bins=10)
    assert psi_shifted > 0.0

    # Test top-10 feature PSI table
    psi_table = compute_psi_table(num_bins=10)
    assert isinstance(psi_table, pd.DataFrame)
    assert len(psi_table) == 10
    assert "feature" in psi_table.columns
    assert "psi" in psi_table.columns
    assert "status" in psi_table.columns

    # All PSI values must be non-negative and finite
    assert (psi_table["psi"] >= 0.0).all()
    assert np.isfinite(psi_table["psi"]).all()


def test_cards_renderers():
    """Verify visual card renderers generate compliant HTML strings without indentation bugs."""
    from dashboard.components.cards import render_decision_badge, render_kpi_card
    from dashboard.components.theme import DARK_THEME, LIGHT_THEME, clean_html

    # Badges
    badge_app = render_decision_badge("AUTO_APPROVE")
    assert "AUTO-APPROVE" in badge_app
    assert "10B981" in badge_app or "16a34a" in badge_app.lower()

    badge_rev = render_decision_badge("MANUAL_REVIEW")
    assert "MANUAL REVIEW" in badge_rev
    assert "F59E0B" in badge_rev or "d97706" in badge_rev.lower()

    badge_rej = render_decision_badge("AUTO_REJECT")
    assert "AUTO-REJECT" in badge_rej
    assert "EF4444" in badge_rej or "dc2626" in badge_rej.lower()

    # KPI card generation must have zero lines with 4+ leading spaces
    card_html = render_kpi_card("Test Metric", "$100M", delta="+5%", status="positive")
    assert not any(line.startswith("    ") for line in card_html.splitlines())
    assert "<div" in card_html
    assert "$100M" in card_html

    # Theme tokens
    assert DARK_THEME.name == "Dark"
    assert LIGHT_THEME.name == "Light"
    assert clean_html("   <div>\n      <span>test</span>\n   </div>") == "<div><span>test</span></div>"


def test_plots_generation():
    """Verify Plotly figure constructors return valid figures."""
    import plotly.graph_objects as go
    from dashboard.components.plots import (
        plot_interactive_calibration_curve,
        plot_policy_donut,
        plot_psi_bar_chart,
        plot_shap_waterfall,
    )

    fig_cal = plot_interactive_calibration_curve()
    assert isinstance(fig_cal, go.Figure)
    assert len(fig_cal.data) >= 3

    fig_donut = plot_policy_donut({"AUTO_APPROVE": 100, "MANUAL_REVIEW": 30, "AUTO_REJECT": 20})
    assert isinstance(fig_donut, go.Figure)

    fig_shap = plot_shap_waterfall(
        feature_names=["f1", "f2"],
        shap_values=np.array([0.15, -0.25]),
        base_value=0.0,
        applicant_values=np.array([10.0, 20.0]),
    )
    assert isinstance(fig_shap, go.Figure)

    psi_df = pd.DataFrame(
        {
            "feature_name": ["Interest Rate", "Loan Term"],
            "psi": [0.14, 0.01],
            "status": ["🟡 Moderate", "🟢 Stable"],
        }
    )
    fig_psi = plot_psi_bar_chart(psi_df)
    assert isinstance(fig_psi, go.Figure)
