"""Tab 4: Model Stability & Out-of-Time Degradation Scorecard.

Presents walk-forward Gini degradation, macroeconomic Lehman crisis stress test,
Lending Club grade monotonicity validation (Grade A 6.33% -> Grade G 50.09%),
and sub-cohort segmentations by loan purpose and residential tenancy.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.cards import render_kpi_card
from dashboard.components.data_loader import load_test_summary_dataset
from dashboard.components.plots import (
    get_plot_layout,
    plot_grade_monotonicity,
    plot_walk_forward_gini,
)
from dashboard.components.theme import clean_html, get_current_theme


def render_tab4():
    """Render Tab 4 Model Stability & Degradation."""
    t = get_current_theme()

    header_html = f"""
    <div style="margin-bottom: 20px;">
        <h2 style="color: {t.text_primary}; margin-bottom: 4px; font-weight: 700;">Model Stability & Stress Resilience</h2>
        <p style="color: {t.text_secondary}; font-size: 0.95rem;">
            Temporal out-of-time degradation tracking, <b>2008 Great Financial Crisis Stress Testing</b>, and strict grade risk monotonicity proof.
        </p>
    </div>
    """
    st.markdown(clean_html(header_html), unsafe_allow_html=True)

    summary_df = load_test_summary_dataset()

    # 1. Top Stability Metrics
    s_col1, s_col2, s_col3, s_col4 = st.columns(4)

    with s_col1:
        render_kpi_card(
            title="Baseline 2016 Gini",
            value="0.4437",
            delta="AUC 0.7219",
            status="positive",
            subtitle="Validation Partition (293,105 loans)",
        )

    with s_col2:
        render_kpi_card(
            title="Holdout 2017–18 Gini",
            value="0.4229",
            delta="-0.0208 Drift (4.7%)",
            status="info",
            subtitle="Full Test Partition (225,639 loans)",
        )

    with s_col3:
        render_kpi_card(
            title="2008 Lehman Crisis Stress",
            value="0.3539",
            delta="Gini > 0.30 Hurdle",
            status="warning",
            subtitle="Macro shock stress test (39,786 loans)",
        )

    with s_col4:
        render_kpi_card(
            title="Grade Monotonicity",
            value="100.0%",
            delta="Strict Order (A → G)",
            status="positive",
            subtitle="6.33% (A) to 50.09% (G) def rate",
        )

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # 2. Walk-Forward Scorecard and Monotonicity Chart
    row1_left, row1_right = st.columns([1, 1])

    with row1_left:
        with st.container(border=True):
            years_df = pd.DataFrame(
                {
                    "period": [
                        "2016 (Val)",
                        "2017 (Test H1)",
                        "2018 (Test H2)",
                        "2008 Lehman Crisis",
                    ],
                    "gini": [0.4437, 0.4233, 0.4095, 0.3539],
                    "auc": [0.7219, 0.7117, 0.7047, 0.6770],
                    "default_rate": [23.29, 23.13, 15.76, 14.25],
                }
            )
            wf_fig = plot_walk_forward_gini(years_df)
            st.plotly_chart(wf_fig, width="stretch")

    with row1_right:
        with st.container(border=True):
            grade_order = ["A", "B", "C", "D", "E", "F", "G"]
            grade_stats = []
            for g in grade_order:
                sub = summary_df[summary_df["grade"] == g]
                cnt = len(sub)
                dr = float(sub["target"].mean() * 100.0) if cnt > 0 else 0.0
                mpd = float(sub["predicted_pd"].mean() * 100.0) if cnt > 0 else 0.0
                grade_stats.append({"grade": g, "count": cnt, "default_rate": dr, "mean_pd": mpd})

            grade_df = pd.DataFrame(grade_stats)
            mono_fig = plot_grade_monotonicity(grade_df)
            st.plotly_chart(mono_fig, width="stretch")

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # 3. Portfolio Subgroup Segmentations
    seg_header = f"""
    <div style="color: {t.text_primary}; font-size: 0.95rem; font-weight: 700; margin-bottom: 12px;">
        Portfolio Cohort Segmentations (Realized Default Rate vs Predicted Risk)
    </div>
    """
    st.markdown(clean_html(seg_header), unsafe_allow_html=True)

    seg_col1, seg_col2 = st.columns(2)

    with seg_col1:
        with st.container(border=True):
            st.markdown(
                clean_html(
                    f"""<div>
                    <h4 style="color: {t.text_primary}; margin: 0 0 4px 0; font-size: 0.95rem; font-weight: 700;">Segmentation by Stated Loan Purpose</h4>
                    <span style="color: {t.text_secondary}; font-size: 0.76rem;">Default rate calibration across borrower intent</span>
                    </div>"""
                ),
                unsafe_allow_html=True,
            )

            p_grp = (
                summary_df.groupby("purpose", observed=False)
                .agg(
                    count=("loan_amnt", "count"),
                    total_exposure=("loan_amnt", "sum"),
                    default_rate=("target", lambda x: x.mean() * 100.0),
                    mean_pd=("predicted_pd", lambda x: x.mean() * 100.0),
                )
                .sort_values(by="count", ascending=False)
                .head(6)
                .reset_index()
            )

            purpose_labels = [p.replace("_", " ").title() for p in p_grp["purpose"]]

            p_fig = go.Figure()
            p_fig.add_trace(
                go.Bar(
                    x=purpose_labels,
                    y=p_grp["default_rate"],
                    name="Realized Def Rate (%)",
                    marker_color=t.danger,
                    text=[f"{r:.1f}%" for r in p_grp["default_rate"]],
                    textposition="outside",
                )
            )
            p_fig.add_trace(
                go.Scatter(
                    x=purpose_labels,
                    y=p_grp["mean_pd"],
                    name="Mean Predicted PD (%)",
                    mode="lines+markers",
                    line=dict(color=t.accent_primary, width=2),
                    marker=dict(size=7),
                )
            )
            p_fig.update_layout(
                get_plot_layout(
                    yaxis=dict(title="Default Rate (%)", range=[0, 32]),
                    height=260,
                    margin=dict(l=30, r=20, t=10, b=40),
                )
            )
            st.plotly_chart(p_fig, width="stretch")

    with seg_col2:
        with st.container(border=True):
            st.markdown(
                clean_html(
                    f"""<div>
                    <h4 style="color: {t.text_primary}; margin: 0 0 4px 0; font-size: 0.95rem; font-weight: 700;">Segmentation by Residential Tenancy</h4>
                    <span style="color: {t.text_secondary}; font-size: 0.76rem;">Homeownership risk spread (Mortgage vs Rent vs Own)</span>
                    </div>"""
                ),
                unsafe_allow_html=True,
            )

            h_grp = (
                summary_df[summary_df["home_ownership"].isin(["MORTGAGE", "RENT", "OWN"])]
                .groupby("home_ownership", observed=False)
                .agg(
                    count=("loan_amnt", "count"),
                    default_rate=("target", lambda x: x.mean() * 100.0),
                    mean_pd=("predicted_pd", lambda x: x.mean() * 100.0),
                )
                .loc[["MORTGAGE", "OWN", "RENT"]]
                .reset_index()
            )

            h_fig = go.Figure()
            h_fig.add_trace(
                go.Bar(
                    x=h_grp["home_ownership"],
                    y=h_grp["default_rate"],
                    name="Realized Def Rate (%)",
                    marker_color=[t.success, t.warning, t.danger],
                    text=[f"{r:.1f}%" for r in h_grp["default_rate"]],
                    textposition="outside",
                )
            )
            h_fig.add_trace(
                go.Scatter(
                    x=h_grp["home_ownership"],
                    y=h_grp["mean_pd"],
                    name="Mean Predicted PD (%)",
                    mode="lines+markers",
                    line=dict(color=t.accent_primary, width=2),
                    marker=dict(size=7),
                )
            )
            h_fig.update_layout(
                get_plot_layout(
                    yaxis=dict(title="Default Rate (%)", range=[0, 32]),
                    height=260,
                    margin=dict(l=30, r=20, t=10, b=40),
                )
            )
            st.plotly_chart(h_fig, width="stretch")
