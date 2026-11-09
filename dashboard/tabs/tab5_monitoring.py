"""Tab 5: Model Monitoring & Basel III Risk Governance.

Presents Basel III regulatory compliance audit card, Population Stability Index (PSI)
feature-level monitoring, score distribution drift overlays, and SR 11-7 / OCC 2011-12
model risk governance triggers.
"""

from __future__ import annotations

import streamlit as st

from dashboard.components.data_loader import (
    compute_psi_table,
    load_baseline_score_sample,
    load_test_summary_dataset,
)
from dashboard.components.plots import (
    plot_pd_distribution_overlay,
    plot_psi_bar_chart,
)
from dashboard.components.theme import clean_html, get_current_theme


def render_tab5():
    """Render Tab 5 Model Monitoring & Basel III Risk Governance."""
    t = get_current_theme()

    header_html = f"""
    <div style="margin-bottom: 20px;">
        <h2 style="color: {t.text_primary}; margin-bottom: 4px; font-weight: 700;">Basel III Monitoring & Model Risk Governance</h2>
        <p style="color: {t.text_secondary}; font-size: 0.95rem;">
            Continuous population drift assessment, Basel III capital adequacy audit, and <b>SR 11-7 / OCC 2011-12</b> institutional governance protocols.
        </p>
    </div>
    """
    st.markdown(clean_html(header_html), unsafe_allow_html=True)

    # 1. Basel III Model Health Traffic-Light Card
    audit_bg = t.bg_card if t.name == "Dark" else "#F0FDF4"
    sub_box_bg = t.bg_card_secondary if t.name == "Dark" else "#FFFFFF"

    health_card_html = f"""
    <div style="
        background: {audit_bg};
        border: 1px solid {t.success};
        border-left: 6px solid {t.success};
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(16, 185, 129, 0.12);
    ">
        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid {t.border}; padding-bottom: 12px; margin-bottom: 14px;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <span style="font-size: 1.6rem;">🟢</span>
                <div>
                    <div style="color: {t.success}; font-size: 1.05rem; font-weight: 800; letter-spacing: 0.05em; text-transform: uppercase;">
                        BASEL III MODEL HEALTH: PASS / PRODUCTION READY
                    </div>
                    <div style="color: {t.text_secondary}; font-size: 0.8rem;">
                        Model Risk Governance (MRG) Tier 1 Comprehensive Validation Status
                    </div>
                </div>
            </div>
            <div>
                <span style="background: {t.success_bg}; color: {t.success}; border: 1px solid {t.success_border}; padding: 4px 12px; border-radius: 9999px; font-size: 0.78rem; font-weight: 700;">
                    AUDIT CLEARED
                </span>
            </div>
        </div>

        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; font-size: 0.82rem;">
            <div style="background: {sub_box_bg}; padding: 10px 14px; border-radius: 8px; border: 1px solid {t.border};">
                <span style="color: {t.text_secondary};">1. Discrimination Hurdle:</span><br>
                <b style="color: {t.success}; font-size: 0.95rem;">Gini 0.4229</b>
                <span style="color: {t.text_muted};">(Hurdle &gt; 0.4000)</span>
            </div>
            <div style="background: {sub_box_bg}; padding: 10px 14px; border-radius: 8px; border: 1px solid {t.border};">
                <span style="color: {t.text_secondary};">2. Calibration Error:</span><br>
                <b style="color: {t.success}; font-size: 0.95rem;">Brier 0.1525</b>
                <span style="color: {t.text_muted};">(-26.8% vs uncalibrated)</span>
            </div>
            <div style="background: {sub_box_bg}; padding: 10px 14px; border-radius: 8px; border: 1px solid {t.border};">
                <span style="color: {t.text_secondary};">3. Grade Monotonicity:</span><br>
                <b style="color: {t.success}; font-size: 0.95rem;">100% Strict Monotonic</b>
                <span style="color: {t.text_muted};">(Grades A to G)</span>
            </div>
            <div style="background: {sub_box_bg}; padding: 10px 14px; border-radius: 8px; border: 1px solid {t.border};">
                <span style="color: {t.text_secondary};">4. Population Drift:</span><br>
                <b style="color: {t.success}; font-size: 0.95rem;">Max PSI 0.1461</b>
                <span style="color: {t.text_muted};">(0/10 features &gt; 0.20)</span>
            </div>
        </div>
    </div>
    """
    st.markdown(clean_html(health_card_html), unsafe_allow_html=True)

    # 2. Main Monitoring Charts: PSI Bar Chart & Distribution Overlay
    m_col1, m_col2 = st.columns([1, 1])

    with m_col1:
        with st.container(border=True):
            psi_df = compute_psi_table()
            psi_fig = plot_psi_bar_chart(psi_df)
            st.plotly_chart(psi_fig, width="stretch")

    with m_col2:
        with st.container(border=True):
            baseline_sample = load_baseline_score_sample()
            summary_df = load_test_summary_dataset()
            holdout_2018_pds = summary_df[summary_df["issue_year"] == 2018]["predicted_pd"].values[
                : len(baseline_sample)
            ]
            overlay_fig = plot_pd_distribution_overlay(baseline_sample, holdout_2018_pds)
            st.plotly_chart(overlay_fig, width="stretch")

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # 3. Model Risk Governance (MRG) Protocol & Re-Training Triggers
    box_sub_bg = t.bg_card_secondary if t.name == "Dark" else "#F8FAFC"
    mrg_html = f"""
    <div style="background: {t.bg_card}; border: 1px solid {t.border}; border-radius: 12px; padding: 20px 24px;">
        <h4 style="color: {t.text_primary}; margin: 0 0 10px 0; font-size: 1rem; font-weight: 700;">
            Model Risk Governance (MRG) Policy & Re-Training Triggers
        </h4>
        <div style="color: {t.text_secondary}; font-size: 0.84rem; line-height: 1.6;">
            Compliant with Federal Reserve <b>SR 11-7</b> and OCC Bulletin <b>2011-12</b> guidance on model risk management:
        </div>

        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-top: 14px; font-size: 0.82rem;">
            <div style="background: {box_sub_bg}; border: 1px solid {t.border}; border-radius: 8px; padding: 14px 16px;">
                <div style="color: {t.success}; font-weight: 700; margin-bottom: 6px;">
                    🟢 GREEN: Routine Monitoring
                </div>
                <ul style="margin: 0; padding-left: 16px; color: {t.text_primary}; line-height: 1.5;">
                    <li>All feature PSIs &lt; 0.10</li>
                    <li>Gini degradation &lt; 2.0 points vs baseline</li>
                    <li>Quarterly validation review cycle</li>
                </ul>
            </div>

            <div style="background: {box_sub_bg}; border: 1px solid {t.warning_border}; border-radius: 8px; padding: 14px 16px;">
                <div style="color: {t.warning}; font-weight: 700; margin-bottom: 6px;">
                    🟡 YELLOW: Heightened Surveillance
                </div>
                <ul style="margin: 0; padding-left: 16px; color: {t.text_primary}; line-height: 1.5;">
                    <li>Feature PSI between 0.10 and 0.20 (e.g. <code>int_rate</code>)</li>
                    <li>Bi-weekly drift monitoring active</li>
                    <li>Assess macroeconomic interest rate trend shifts</li>
                </ul>
            </div>

            <div style="background: {box_sub_bg}; border: 1px solid {t.danger_border}; border-radius: 8px; padding: 14px 16px;">
                <div style="color: {t.danger}; font-weight: 700; margin-bottom: 6px;">
                    🔴 RED: Mandatory Retraining
                </div>
                <ul style="margin: 0; padding-left: 16px; color: {t.text_primary}; line-height: 1.5;">
                    <li>Any core feature PSI &ge; 0.20</li>
                    <li>Gini drops below 0.38 on rolling 90-day cohort</li>
                    <li>Trigger automatic Champion/Challenger recalibration</li>
                </ul>
            </div>
        </div>
    </div>
    """
    st.markdown(clean_html(mrg_html), unsafe_allow_html=True)
