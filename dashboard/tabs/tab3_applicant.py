"""Tab 3: Single Applicant Underwriting Dossier & TreeSHAP Explainability.

Provides borrower factsheet dossier, interactive TreeSHAP attribution waterfall,
statutory adverse action notices for declined/reviewed applicants (ECOA / FCRA),
and global beeswarm summary.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.cards import render_adverse_action_box, render_decision_badge
from dashboard.components.data_loader import (
    load_applicant_sample_dossier,
    load_policy_thresholds,
    load_shap_cache,
)
from dashboard.components.plots import plot_shap_waterfall
from dashboard.components.theme import clean_html, get_current_theme
from src.explainability.shap_explainer import get_adverse_action_description

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


def render_tab3():
    """Render Tab 3 Single Applicant Underwriting Dossier."""
    t = get_current_theme()

    header_html = f"""
    <div style="margin-bottom: 20px;">
        <h2 style="color: {t.text_primary}; margin-bottom: 4px; font-weight: 700;">Individual Underwriting & SHAP Attribution</h2>
        <p style="color: {t.text_secondary}; font-size: 0.95rem;">
            Granular loan-level risk decomposition for <b>1,000 precomputed test applicants</b>. Explains local decision rationale and generates regulatory adverse action notices.
        </p>
    </div>
    """
    st.markdown(clean_html(header_html), unsafe_allow_html=True)

    dossier = load_applicant_sample_dossier()
    shap_cache = load_shap_cache()
    shap_matrix = shap_cache["shap_values"]
    feature_names = shap_cache["feature_names"]
    expected_value = float(shap_cache.get("expected_value", 0.0))
    sample_X = shap_cache["sample_X"]

    policy_data = load_policy_thresholds()
    tau_low = float(policy_data.get("tau_low", 0.25))
    tau_high = float(policy_data.get("tau_high", 0.35))

    # 1. Filters & Applicant Selector
    f_col1, f_col2, f_col3, f_col4 = st.columns([1.2, 1, 1, 1.8])

    with f_col1:
        tier_filter = st.selectbox(
            "Filter Decision Tier",
            options=["All Tiers", "AUTO_APPROVE", "MANUAL_REVIEW", "AUTO_REJECT"],
            index=0,
        )

    with f_col2:
        grade_filter = st.selectbox(
            "Filter Grade",
            options=["All Grades", "A", "B", "C", "D", "E", "F", "G"],
            index=0,
        )

    with f_col3:
        outcome_filter = st.selectbox(
            "Filter Outcome",
            options=["All Outcomes", "Non-Default (0)", "Defaulted (1)"],
            index=0,
        )

    # Filter dossier
    filtered_df = dossier.copy()
    if tier_filter != "All Tiers":
        filtered_df = filtered_df[filtered_df["decision_tier"] == tier_filter]
    if grade_filter != "All Grades":
        filtered_df = filtered_df[filtered_df["grade"] == grade_filter]
    if outcome_filter == "Non-Default (0)":
        filtered_df = filtered_df[filtered_df["target"] == 0]
    elif outcome_filter == "Defaulted (1)":
        filtered_df = filtered_df[filtered_df["target"] == 1]

    with f_col4:
        if len(filtered_df) == 0:
            st.error("No applicants match the selected filter criteria.")
            return

        applicant_options = [
            f"{row['applicant_id']} (Grade {row['grade']}, PD: {row['predicted_pd']*100:.1f}%, ${row['loan_amnt']:,.0f})"
            for _, row in filtered_df.iterrows()
        ]
        selected_option = st.selectbox(
            f"Select Applicant ({len(filtered_df):,} matching)",
            options=applicant_options,
            index=0,
        )

    selected_idx_in_filtered = applicant_options.index(selected_option)
    selected_row = filtered_df.iloc[selected_idx_in_filtered]
    app_idx = int(selected_row["applicant_idx"])
    applicant_id = str(selected_row["applicant_id"])
    pred_pd = float(selected_row["predicted_pd"])
    decision_tier = str(selected_row["decision_tier"])
    target_val = int(selected_row["target"])

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # 2. Split Screen: Factsheet Dossier & Underwriting Verdict Gauge
    card_col, gauge_col = st.columns([1.3, 0.7])

    with card_col:
        card_shadow = "0 4px 16px rgba(0, 0, 0, 0.25)" if t.name == "Dark" else "0 2px 10px rgba(0, 0, 0, 0.05)"
        dossier_html = f"""
        <div style="background: {t.bg_card}; border: 1px solid {t.border}; border-radius: 12px; padding: 18px 22px; box-shadow: {card_shadow};">
            <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid {t.border}; padding-bottom: 10px; margin-bottom: 14px;">
                <div>
                    <span style="color: {t.accent_primary}; font-weight: 800; font-size: 1.05rem;">{applicant_id}</span>
                    <span style="color: {t.text_secondary}; font-size: 0.8rem; margin-left: 8px;">Sub-Grade: <b style="color: {t.text_primary};">{selected_row['sub_grade']}</b></span>
                </div>
                <div>
                    {render_decision_badge(decision_tier)}
                </div>
            </div>

            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px 10px; font-size: 0.82rem;">
                <div>
                    <span style="color: {t.text_secondary};">Loan Amount:</span><br>
                    <b style="color: {t.text_primary}; font-size: 0.95rem;">${selected_row['loan_amnt']:,.0f}</b>
                </div>
                <div>
                    <span style="color: {t.text_secondary};">Term:</span><br>
                    <b style="color: {t.text_primary}; font-size: 0.95rem;">{selected_row['term']}</b>
                </div>
                <div>
                    <span style="color: {t.text_secondary};">Interest Rate:</span><br>
                    <b style="color: {t.text_primary}; font-size: 0.95rem;">{selected_row['int_rate']:.2f}%</b>
                </div>
                <div>
                    <span style="color: {t.text_secondary};">Monthly Payment:</span><br>
                    <b style="color: {t.text_primary}; font-size: 0.95rem;">${selected_row['installment']:,.2f}</b>
                </div>

                <div>
                    <span style="color: {t.text_secondary};">Annual Income:</span><br>
                    <b style="color: {t.text_primary}; font-size: 0.95rem;">${selected_row['annual_inc']:,.0f}</b>
                </div>
                <div>
                    <span style="color: {t.text_secondary};">FICO Bureau Score:</span><br>
                    <b style="color: {t.accent_primary}; font-size: 0.95rem;">{selected_row['fico_score']:.0f}</b>
                </div>
                <div>
                    <span style="color: {t.text_secondary};">Debt-to-Income (DTI):</span><br>
                    <b style="color: {t.text_primary}; font-size: 0.95rem;">{selected_row['dti']:.1f}%</b>
                </div>
                <div>
                    <span style="color: {t.text_secondary};">Revolving Utilization:</span><br>
                    <b style="color: {t.text_primary}; font-size: 0.95rem;">{selected_row['revol_util']:.1f}%</b>
                </div>

                <div style="grid-column: span 2;">
                    <span style="color: {t.text_secondary};">Stated Purpose:</span><br>
                    <b style="color: {t.text_primary};">{str(selected_row['purpose']).replace('_', ' ').title()}</b>
                </div>
                <div>
                    <span style="color: {t.text_secondary};">Residential Tenancy:</span><br>
                    <b style="color: {t.text_primary};">{selected_row['home_ownership']}</b>
                </div>
                <div>
                    <span style="color: {t.text_secondary};">Realized Outcome:</span><br>
                    <b style="color: {t.danger if target_val == 1 else t.success};">
                        {'🔴 Defaulted' if target_val == 1 else '🟢 Fully Repaid'}
                    </b>
                </div>
            </div>
        </div>
        """
        st.markdown(clean_html(dossier_html), unsafe_allow_html=True)

    with gauge_col:
        with st.container(border=True):
            st.markdown(
                clean_html(
                    f"""<div style="text-align: center; margin-bottom: 2px;">
                    <span style="color: {t.text_secondary}; font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;">
                        Calibrated Default Probability (PD)
                    </span>
                    </div>"""
                ),
                unsafe_allow_html=True,
            )

            gauge_fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=pred_pd * 100.0,
                    number={"suffix": "%", "font": {"color": t.text_primary, "size": 32, "family": "sans-serif"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": t.text_secondary, "tickfont": {"color": t.text_secondary, "size": 10}},
                        "bar": {"color": t.accent_primary, "thickness": 0.28},
                        "bgcolor": t.plot_bg,
                        "borderwidth": 1,
                        "bordercolor": t.border,
                        "steps": [
                            {"range": [0, tau_low * 100.0], "color": t.success_bg},
                            {"range": [tau_low * 100.0, tau_high * 100.0], "color": t.warning_bg},
                            {"range": [tau_high * 100.0, 100.0], "color": t.danger_bg},
                        ],
                        "threshold": {
                            "line": {"color": t.text_primary, "width": 3},
                            "thickness": 0.8,
                            "value": pred_pd * 100.0,
                        },
                    },
                )
            )
            gauge_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": t.text_primary, "family": "sans-serif"},
                height=160,
                margin=dict(l=20, r=20, t=10, b=10),
            )
            st.plotly_chart(gauge_fig, width="stretch")

            gauge_footer = f"""
            <div style="display: flex; justify-content: space-around; font-size: 0.72rem; color: {t.text_secondary}; border-top: 1px solid {t.border}; padding-top: 8px;">
                <span>Approve: &lt; {tau_low*100:.0f}%</span>
                <span>Review: {tau_low*100:.0f}-{tau_high*100:.0f}%</span>
                <span>Reject: &gt; {tau_high*100:.0f}%</span>
            </div>
            """
            st.markdown(clean_html(gauge_footer), unsafe_allow_html=True)

    # 3. Statutory Adverse Action Notice Box (For Non-Auto-Approve)
    app_shap = shap_matrix[app_idx]
    pos_indices = np.argsort(app_shap)[::-1]
    top_adverse_reasons = []
    for idx in pos_indices:
        if app_shap[idx] > 0:
            desc = get_adverse_action_description(feature_names[idx])
            if desc not in top_adverse_reasons:
                top_adverse_reasons.append(desc)
        if len(top_adverse_reasons) >= 3:
            break

    if decision_tier in ["MANUAL_REVIEW", "AUTO_REJECT"]:
        threshold_used = tau_high if decision_tier == "AUTO_REJECT" else tau_low
        render_adverse_action_box(
            applicant_id=applicant_id,
            reasons=top_adverse_reasons,
            pd_score=pred_pd,
            threshold=threshold_used,
            applicant_data={"loan_amnt": float(selected_row["loan_amnt"])},
        )
    else:
        approval_banner = f"""
        <div style="background: {t.success_bg}; border: 1px solid {t.success_border}; border-radius: 8px; padding: 12px 18px; margin: 14px 0; color: {t.success}; font-size: 0.85rem; display: flex; align-items: center; gap: 10px;">
            <span>🟢</span>
            <span><b>Instant Auto-Approval Granted:</b> Applicant PD ({pred_pd*100:.2f}%) clears prime threshold (τ_low = {tau_low*100:.1f}%). Zero adverse factors.</span>
        </div>
        """
        st.markdown(clean_html(approval_banner), unsafe_allow_html=True)

    # 4. Interactive TreeSHAP Waterfall Chart
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    applicant_values = sample_X.iloc[app_idx].values
    waterfall_fig = plot_shap_waterfall(
        feature_names=feature_names,
        shap_values=app_shap,
        base_value=expected_value,
        applicant_values=applicant_values,
        max_display=12,
    )
    st.plotly_chart(waterfall_fig, width="stretch")

    # 5. Global Explainability Beeswarm Expander
    with st.expander("🐝 View Global Population SHAP Beeswarm Plot (1,000 Applicants)", expanded=False):
        beeswarm_path = FIGURES_DIR / "shap_summary_beeswarm.png"
        if beeswarm_path.exists():
            st.image(
                str(beeswarm_path),
                caption="Global TreeSHAP Feature Attributions across 1,000 Out-of-Time Test Applicants",
                width="stretch",
            )
        else:
            st.info("Global SHAP Beeswarm figure not found on disk.")
