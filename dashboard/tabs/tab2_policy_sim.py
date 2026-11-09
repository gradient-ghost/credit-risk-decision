"""Tab 2: Dynamic Underwriting Policy Simulator.

Enables real-time (<10ms) vectorized policy stress testing across 225,639 test applicants.
Features dual cut-off sliders (tau_low, tau_high), LGD assumptions, one-click macroeconomic
scenario presets, dynamic KPI cards, and donut/trade-off visualizations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.cards import render_kpi_card
from dashboard.components.data_loader import load_test_summary_dataset
from dashboard.components.plots import get_plot_layout, plot_policy_donut
from dashboard.components.theme import clean_html, get_current_theme


def render_tab2():
    """Render Tab 2 Dynamic Policy Simulator."""
    t = get_current_theme()

    header_html = f"""
    <div style="margin-bottom: 20px;">
        <h2 style="color: {t.text_primary}; margin-bottom: 4px; font-weight: 700;">Dynamic Policy Simulation Engine</h2>
        <p style="color: {t.text_secondary}; font-size: 0.95rem;">
            Vectorized risk-return optimization on <b>225,639 test loans</b>. Adjust hurdle rates to simulate approval volumes, credit losses, and operational queue sizes.
        </p>
    </div>
    """
    st.markdown(clean_html(header_html), unsafe_allow_html=True)

    summary_df = load_test_summary_dataset()
    preds = summary_df["predicted_pd"].values
    loans = summary_df["loan_amnt"].values
    targets = summary_df["target"].values
    n_total = len(preds)

    # Initialize session state for sliders if not set
    if "sim_tau_low" not in st.session_state:
        st.session_state.sim_tau_low = 0.25
    if "sim_tau_high" not in st.session_state:
        st.session_state.sim_tau_high = 0.35
    if "sim_lgd" not in st.session_state:
        st.session_state.sim_lgd = 1.00

    # Macro Preset Buttons Container
    preset_header = f"""
    <div style="color: {t.text_secondary}; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px;">
        Macroeconomic Underwriting Presets:
    </div>
    """
    st.markdown(clean_html(preset_header), unsafe_allow_html=True)

    p_col1, p_col2, p_col3, p_col4 = st.columns([1, 1, 1, 1])

    with p_col1:
        if st.button("🟢 Economic Boom", width="stretch", help="Expansive: tau_low=0.28, tau_high=0.40, LGD=0.85"):
            st.session_state.sim_tau_low = 0.28
            st.session_state.sim_tau_high = 0.40
            st.session_state.sim_lgd = 0.85
            st.rerun()

    with p_col2:
        if st.button("🔵 Baseline Balanced", width="stretch", help="Standard: tau_low=0.25, tau_high=0.35, LGD=1.00"):
            st.session_state.sim_tau_low = 0.25
            st.session_state.sim_tau_high = 0.35
            st.session_state.sim_lgd = 1.00
            st.rerun()

    with p_col3:
        if st.button("🔴 Stagflation Squeeze", width="stretch", help="Conservative: tau_low=0.18, tau_high=0.28, LGD=1.00"):
            st.session_state.sim_tau_low = 0.18
            st.session_state.sim_tau_high = 0.28
            st.session_state.sim_lgd = 1.00
            st.rerun()

    with p_col4:
        if st.button("🔄 Reset to Default", width="stretch"):
            st.session_state.sim_tau_low = 0.25
            st.session_state.sim_tau_high = 0.35
            st.session_state.sim_lgd = 1.00
            st.rerun()

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Interactive Sliders in 3 columns
    s_col1, s_col2, s_col3 = st.columns(3)

    with s_col1:
        tau_low = st.slider(
            "Auto-Approve Hurdle (τ_low)",
            min_value=0.05,
            max_value=0.40,
            value=float(st.session_state.sim_tau_low),
            step=0.01,
            help="Applicants with predicted PD below this threshold are automatically approved without human intervention.",
            key="slider_tau_low",
        )
        st.session_state.sim_tau_low = tau_low

    with s_col2:
        tau_high = st.slider(
            "Auto-Reject Hurdle (τ_high)",
            min_value=0.20,
            max_value=0.60,
            value=float(st.session_state.sim_tau_high),
            step=0.01,
            help="Applicants with predicted PD above this threshold are automatically rejected with adverse action notices.",
            key="slider_tau_high",
        )
        st.session_state.sim_tau_high = tau_high

    with s_col3:
        lgd = st.slider(
            "Loss Given Default (LGD)",
            min_value=0.20,
            max_value=1.00,
            value=float(st.session_state.sim_lgd),
            step=0.05,
            help="Recovery rate assumption: LGD = 1.0 assumes zero recovery on defaulted unsecured debt.",
            key="slider_lgd",
        )
        st.session_state.sim_lgd = lgd

    # Threshold validation guard
    if tau_low > tau_high:
        st.warning(f"⚠️ Notice: τ_low ({tau_low:.2f}) exceeds τ_high ({tau_high:.2f}). Adjusting upper threshold.")
        tau_high = tau_low

    # Real-time Vectorized Calculation (<5ms)
    mask_approve = preds < tau_low
    mask_reject = preds > tau_high
    mask_review = (~mask_approve) & (~mask_reject)

    c_app = int(np.sum(mask_approve))
    c_rev = int(np.sum(mask_review))
    c_rej = int(np.sum(mask_reject))

    pct_app = (c_app / n_total * 100.0) if n_total > 0 else 0.0
    pct_rev = (c_rev / n_total * 100.0) if n_total > 0 else 0.0
    pct_rej = (c_rej / n_total * 100.0) if n_total > 0 else 0.0

    exp_app = float(np.sum(loans[mask_approve]))
    exp_rev = float(np.sum(loans[mask_review]))

    def_rate_app = float(np.mean(targets[mask_approve]) * 100.0) if c_app > 0 else 0.0
    def_rate_rev = float(np.mean(targets[mask_review]) * 100.0) if c_rev > 0 else 0.0

    el_saved = float(np.sum(preds[mask_reject] * loans[mask_reject] * lgd))
    realized_saved = float(np.sum(targets[mask_reject] * loans[mask_reject] * lgd))
    baseline_el = float(np.sum(preds * loans * lgd))
    pct_el_saved = (el_saved / baseline_el * 100.0) if baseline_el > 0 else 0.0

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # Dynamic Reaction Cards
    res_col1, res_col2, res_col3 = st.columns(3)

    with res_col1:
        render_kpi_card(
            title="Auto-Approve Tier",
            value=f"{pct_app:.1f}% ({c_app:,})",
            delta=f"{def_rate_app:.2f}% Realized Def Rate",
            status="positive",
            subtitle=f"${exp_app / 1e6:,.1f}M Approved Exposure",
        )

    with res_col2:
        render_kpi_card(
            title="Manual Review Queue",
            value=f"{pct_rev:.1f}% ({c_rev:,})",
            delta=f"{def_rate_rev:.2f}% Grey-Zone Def Rate",
            status="warning",
            subtitle=f"${exp_rev / 1e6:,.1f}M In Underwriter Review",
        )

    with res_col3:
        render_kpi_card(
            title="Auto-Reject Tier",
            value=f"{pct_rej:.1f}% ({c_rej:,})",
            delta=f"-{pct_el_saved:.1f}% Expected Loss Saved",
            status="negative",
            subtitle=f"${el_saved / 1e6:,.1f}M Loss Prevented (${realized_saved / 1e6:,.1f}M Realized)",
        )

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Visualizations: Donut Chart + Threshold Sensitivity Curve
    v_col1, v_col2 = st.columns([0.8, 1.2])

    with v_col1:
        with st.container(border=True):
            st.markdown(
                clean_html(
                    f"""<div>
                    <h4 style="color: {t.text_primary}; margin: 0 0 4px 0; font-size: 1rem; font-weight: 700;">Portfolio Allocation Breakdown</h4>
                    <span style="color: {t.text_secondary}; font-size: 0.78rem;">Distribution across underwriting decision tiers</span>
                    </div>"""
                ),
                unsafe_allow_html=True,
            )
            donut_fig = plot_policy_donut(
                {"AUTO_APPROVE": c_app, "MANUAL_REVIEW": c_rev, "AUTO_REJECT": c_rej}
            )
            st.plotly_chart(donut_fig, width="stretch")

    with v_col2:
        with st.container(border=True):
            st.markdown(
                clean_html(
                    f"""<div>
                    <h4 style="color: {t.text_primary}; margin: 0 0 4px 0; font-size: 1rem; font-weight: 700;">Approval Volume vs Expected Loss Trade-Off</h4>
                    <span style="color: {t.text_secondary}; font-size: 0.78rem;">Simulated frontier across potential cut-off thresholds</span>
                    </div>"""
                ),
                unsafe_allow_html=True,
            )

            threshold_range = np.linspace(0.10, 0.50, 25)
            sim_approval_rates = []
            sim_loss_reductions = []

            for tr in threshold_range:
                m_rej_sim = preds > tr
                el_s = float(np.sum(preds[m_rej_sim] * loans[m_rej_sim] * lgd))
                pct_s = (el_s / baseline_el * 100.0) if baseline_el > 0 else 0.0
                app_r = float(np.mean(preds < tr) * 100.0)
                sim_approval_rates.append(app_r)
                sim_loss_reductions.append(pct_s)

            curve_fig = go.Figure()
            curve_fig.add_trace(
                go.Scatter(
                    x=threshold_range,
                    y=sim_approval_rates,
                    name="Approval Rate (%)",
                    line=dict(color=t.accent_primary, width=3),
                    mode="lines",
                    hovertemplate="Hurdle: %{x:.2f}<br>Approval Rate: %{y:.1f}%<extra></extra>",
                )
            )
            curve_fig.add_trace(
                go.Scatter(
                    x=threshold_range,
                    y=sim_loss_reductions,
                    name="Expected Loss Saved (%)",
                    line=dict(color=t.danger, width=3, dash="dot"),
                    mode="lines",
                    hovertemplate="Hurdle: %{x:.2f}<br>Loss Saved: %{y:.1f}%<extra></extra>",
                )
            )

            # Current operating point vertical lines
            curve_fig.add_vline(
                x=tau_low,
                line=dict(color=t.success, width=2, dash="dash"),
                annotation_text=f"τ_low ({tau_low:.2f})",
                annotation_position="bottom left",
                annotation_font=dict(color=t.success, size=10),
            )
            curve_fig.add_vline(
                x=tau_high,
                line=dict(color=t.danger, width=2, dash="dash"),
                annotation_text=f"τ_high ({tau_high:.2f})",
                annotation_position="top right",
                annotation_font=dict(color=t.danger, size=10),
            )

            curve_fig.update_layout(
                get_plot_layout(
                    xaxis=dict(title="Probability Hurdle (τ)", range=[0.10, 0.50]),
                    yaxis=dict(title="Portfolio Percentage (%)", range=[0, 100]),
                    height=320,
                    margin=dict(l=40, r=40, t=20, b=40),
                )
            )
            st.plotly_chart(curve_fig, width="stretch")
