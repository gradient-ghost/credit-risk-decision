"""Tab 1: Executive Summary & Champion Model Tournament Scorecard.

Presents portfolio-level financial metrics ($3.26B evaluated, $392.2M loss saved),
tournament model benchmark comparison, probability calibration validation,
and regulatory Selection Bias & FICO 660 invisible floor analysis.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.components.cards import render_kpi_card
from dashboard.components.data_loader import load_policy_thresholds, load_runs_log
from dashboard.components.plots import plot_interactive_calibration_curve
from dashboard.components.theme import clean_html, get_current_theme


def render_tab1():
    """Render Tab 1 Executive Summary."""
    t = get_current_theme()

    header_html = f"""
    <div style="margin-bottom: 24px;">
        <h2 style="color: {t.text_primary}; margin-bottom: 4px; font-weight: 700;">Executive Performance & Portfolio Governance</h2>
        <p style="color: {t.text_secondary}; font-size: 0.95rem;">
            Comprehensive underwriting evaluation across <b>225,639 out-of-time test applicants</b> (2017–2018 origination cohort).
        </p>
    </div>
    """
    st.markdown(clean_html(header_html), unsafe_allow_html=True)

    policy_data = load_policy_thresholds()
    test_eval = policy_data.get("test_eval", {})
    auto_approve = test_eval.get("auto_approve", {})
    baseline = test_eval.get("baseline", {})
    headline = test_eval.get("headline", {})

    total_exposure = test_eval.get("total_exposure", 3259579400.0)
    baseline_loss = baseline.get("realized_loss", 770237650.0)
    expected_loss_saved = headline.get("expected_loss_reduction", 392166307.37)
    auto_approve_pct = auto_approve.get("pct_portfolio", 61.47)
    auto_approve_def_rate = auto_approve.get("realized_default_rate", 13.045)

    # 1. Top 4 Institutional KPI Metric Cards
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

    with kpi_col1:
        render_kpi_card(
            title="Portfolio Evaluated",
            value=f"${total_exposure / 1e9:.2f}B",
            delta="225,639 loans",
            status="info",
            subtitle="2017–2018 OOT Test Horizon",
        )

    with kpi_col2:
        render_kpi_card(
            title="Baseline Realized Loss",
            value=f"${baseline_loss / 1e6:.1f}M",
            delta="21.29% Def Rate",
            status="warning",
            subtitle="Loss under Approve-Everyone baseline",
        )

    with kpi_col3:
        render_kpi_card(
            title="Expected Loss Saved",
            value=f"${expected_loss_saved / 1e6:.1f}M",
            delta="-47.51% Loss Saved",
            status="positive",
            subtitle="Protected via high-risk auto-rejection",
        )

    with kpi_col4:
        render_kpi_card(
            title="Instant Auto-Approval",
            value=f"{auto_approve_pct:.2f}%",
            delta=f"{auto_approve_def_rate:.2f}% Def Rate",
            status="positive",
            subtitle="Prime volume cleared with zero friction",
        )

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    # 2. Main Content Split: Tournament Scorecard & Calibration Curve
    col_left, col_right = st.columns([1.1, 0.9])

    with col_left:
        runs_log = load_runs_log()
        test_runs = runs_log[runs_log["split"] == "test"].copy()

        display_df = pd.DataFrame(
            {
                "Model Candidate": test_runs["model_name"],
                "ROC-AUC": test_runs["roc_auc"].map("{:.4f}".format),
                "Gini": test_runs["gini"].map("{:.4f}".format),
                "KS Stat": test_runs["ks_stat"].map("{:.4f}".format),
                "Brier Loss": test_runs["brier_score"].map("{:.4f}".format),
            }
        )

        rows_html = ""
        for _, row in display_df.iterrows():
            m_name = row["Model Candidate"]
            is_champ = "Platt" in m_name
            bg_color = t.info_bg if is_champ else "transparent"
            border_left = f"3px solid {t.accent_primary}" if is_champ else "3px solid transparent"
            text_color = t.accent_primary if is_champ else t.text_primary
            status_tag = (
                f'<span style="background: {t.accent_primary}; color: {t.bg_app}; font-weight: 700; '
                f'padding: 2px 6px; border-radius: 4px; font-size: 0.68rem;">CHAMPION</span>'
                if is_champ
                else f'<span style="color: {t.text_muted};">Evaluated</span>'
            )

            rows_html += (
                f'<tr style="border-bottom: 1px solid {t.border}; background: {bg_color};">'
                f'<td style="padding: 10px 8px; font-weight: {700 if is_champ else 500}; color: {text_color}; border-left: {border_left};">{m_name}</td>'
                f'<td style="padding: 10px 8px; font-family: monospace; color: {t.text_primary};">{row["ROC-AUC"]}</td>'
                f'<td style="padding: 10px 8px; font-family: monospace; color: {t.text_primary};">{row["Gini"]}</td>'
                f'<td style="padding: 10px 8px; font-family: monospace; color: {t.text_primary};">{row["KS Stat"]}</td>'
                f'<td style="padding: 10px 8px; font-family: monospace; color: {t.text_primary};">{row["Brier Loss"]}</td>'
                f'<td style="padding: 10px 8px;">{status_tag}</td>'
                f"</tr>"
            )

        card_shadow = "0 4px 16px rgba(0,0,0,0.25)" if t.name == "Dark" else "0 2px 10px rgba(0,0,0,0.05)"

        scorecard_html = f"""
        <div style="background: {t.bg_card}; border: 1px solid {t.border}; border-radius: 12px; padding: 20px 22px; height: 100%; box-shadow: {card_shadow};">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;">
                <div>
                    <h4 style="color: {t.text_primary}; margin: 0; font-size: 1.05rem; font-weight: 700;">Model Tournament Scorecard</h4>
                    <span style="color: {t.text_secondary}; font-size: 0.78rem;">Evaluated on Out-of-Time Test Set (225,639 loans)</span>
                </div>
                <span style="background: {t.info_bg}; color: {t.accent_primary}; border: 1px solid {t.info_border}; font-size: 0.72rem; font-weight: 700; padding: 3px 8px; border-radius: 9999px;">
                    🏆 Champion Selected
                </span>
            </div>
            <div style="overflow-x: auto;">
                <table style="width: 100%; border-collapse: collapse; font-size: 0.82rem;">
                    <thead>
                        <tr style="border-bottom: 2px solid {t.border_subtle}; text-align: left; color: {t.text_secondary}; text-transform: uppercase; font-size: 0.72rem;">
                            <th style="padding: 10px 8px;">Model Architecture</th>
                            <th style="padding: 10px 8px;">ROC-AUC</th>
                            <th style="padding: 10px 8px;">Gini</th>
                            <th style="padding: 10px 8px;">KS Stat</th>
                            <th style="padding: 10px 8px;">Brier Loss</th>
                            <th style="padding: 10px 8px;">Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows_html}
                    </tbody>
                </table>
            </div>
        </div>
        """
        st.markdown(clean_html(scorecard_html), unsafe_allow_html=True)

    with col_right:
        cal_fig = plot_interactive_calibration_curve(runs_log)
        st.plotly_chart(cal_fig, width="stretch")

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # 3. Selection Bias & FICO 660 "Invisible Floor" Container
    bias_bg = t.bg_card if t.name == "Dark" else "#FFFBEB"
    bias_border = t.warning

    bias_html = f"""
    <div style="
        background: {bias_bg};
        border: 1px solid {bias_border};
        border-left: 6px solid {bias_border};
        border-radius: 10px;
        padding: 18px 24px;
        box-shadow: 0 4px 16px rgba(245, 158, 11, 0.08);
    ">
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
            <span style="font-size: 1.2rem;">⚠️</span>
            <span style="color: {t.warning}; font-weight: 800; font-size: 0.85rem; letter-spacing: 0.06em; text-transform: uppercase;">
                Selection Bias Analysis: FICO 660 "Invisible Floor" & Survivorship Truncation
            </span>
        </div>
        <div style="color: {t.text_primary}; font-size: 0.85rem; line-height: 1.55;">
            <p style="margin-bottom: 8px;">
                <b>Institutional Data Notice:</b> Lending Club historically rejected applicants with credit scores below 
                <b>FICO 660</b> at the front gate. Consequently, all 1.25M observed loans represent a survivor-truncated population. 
                Standard credit scorecards trained solely on accepted loans underestimate subprime default rates due to unobserved rejection outcomes.
            </p>
            <div style="display: flex; gap: 24px; margin-top: 10px; padding-top: 10px; border-top: 1px solid {t.border}; font-size: 0.8rem;">
                <div>
                    <span style="color: {t.text_secondary};">Option Z Verification (No Grade):</span>
                    <b style="color: {t.accent_primary}; margin-left: 6px;">0.4218 Gini</b>
                    <span style="color: {t.text_muted};">(99.7% of full model discrimination preserved)</span>
                </div>
                <div>
                    <span style="color: {t.text_secondary};">Platt Brier Improvement:</span>
                    <b style="color: {t.accent_primary}; margin-left: 6px;">0.1525 vs 0.2084</b>
                    <span style="color: {t.success};">(-26.8% calibrated loss reduction)</span>
                </div>
            </div>
        </div>
    </div>
    """
    st.markdown(clean_html(bias_html), unsafe_allow_html=True)
