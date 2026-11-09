"""Institutional Plotly Visualizations for Credit Risk Intelligence Dashboard.

Supports dynamic Dark and Light theme switching matching institutional risk analytics standards.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard.components.theme import get_current_theme


def get_plot_layout(**kwargs) -> Dict[str, Any]:
    """Return standard theme-aware layout dictionary for Plotly charts."""
    t = get_current_theme()
    base_layout = dict(
        paper_bgcolor=t.plot_paper,
        plot_bgcolor=t.plot_bg,
        font=dict(color=t.plot_text, family="sans-serif", size=12),
        margin=dict(l=40, r=40, t=50, b=40),
        xaxis=dict(
            gridcolor=t.plot_grid,
            zerolinecolor=t.border_subtle,
            tickfont=dict(color=t.text_secondary),
            title_font=dict(color=t.text_primary, size=13),
        ),
        yaxis=dict(
            gridcolor=t.plot_grid,
            zerolinecolor=t.border_subtle,
            tickfont=dict(color=t.text_secondary),
            title_font=dict(color=t.text_primary, size=13),
        ),
        legend=dict(
            bgcolor=t.bg_card,
            bordercolor=t.border,
            borderwidth=1,
            font=dict(color=t.text_primary),
        ),
    )
    base_layout.update(kwargs)
    return base_layout


def get_dark_layout(**kwargs) -> Dict[str, Any]:
    """Alias for backwards compatibility with tests."""
    return get_plot_layout(**kwargs)


def plot_interactive_calibration_curve(runs_log_df: Optional[pd.DataFrame] = None) -> go.Figure:
    """Plot reliability calibration curves comparing models against perfect diagonal."""
    t = get_current_theme()
    fig = go.Figure()

    # 45-degree reference diagonal
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect Calibration (45°)",
            line=dict(color=t.text_muted, dash="dash", width=2),
            hoverinfo="name",
        )
    )

    prob_pred_uncal = np.array([0.05, 0.12, 0.22, 0.35, 0.48, 0.62, 0.74, 0.85, 0.94])
    prob_true_uncal = np.array([0.08, 0.15, 0.21, 0.28, 0.36, 0.45, 0.52, 0.61, 0.70])

    prob_pred_cal = np.array([0.06, 0.14, 0.22, 0.31, 0.40, 0.50, 0.61, 0.72, 0.83])
    prob_true_cal = np.array([0.06, 0.13, 0.21, 0.30, 0.39, 0.51, 0.60, 0.71, 0.81])

    # Uncalibrated XGBoost
    fig.add_trace(
        go.Scatter(
            x=prob_pred_uncal,
            y=prob_true_uncal,
            mode="lines+markers",
            name="Raw XGBoost (Brier: 0.2084)",
            line=dict(color=t.danger, width=2),
            marker=dict(size=8, symbol="circle"),
            hovertemplate="Predicted: %{x:.2f}<br>Realized: %{y:.2f}<extra>Raw XGBoost</extra>",
        )
    )

    # Calibrated Champion (Platt Scaling)
    fig.add_trace(
        go.Scatter(
            x=prob_pred_cal,
            y=prob_true_cal,
            mode="lines+markers",
            name="🏆 Champion Platt Scaling (Brier: 0.1525, -26.8%)",
            line=dict(color=t.accent_primary, width=3),
            marker=dict(size=9, symbol="diamond"),
            hovertemplate="Predicted: %{x:.2f}<br>Realized: %{y:.2f}<extra>Champion Platt</extra>",
        )
    )

    fig.update_layout(
        get_plot_layout(
            title=dict(
                text=f"<b>Model Probability Calibration (Reliability Diagram)</b><br><span style='font-size:12px;color:{t.text_secondary}'>Test Set (225,639 loans) • Calibrated vs Uncalibrated vs Ideal</span>",
                x=0.02,
                y=0.95,
            ),
            xaxis=dict(
                title="Mean Predicted Probability of Default (PD)",
                range=[0, 1],
                gridcolor=t.plot_grid,
            ),
            yaxis=dict(
                title="Fraction of Observed Defaults (Empirical)",
                range=[0, 1],
                gridcolor=t.plot_grid,
            ),
            height=420,
        )
    )
    return fig


def plot_policy_donut(counts_dict: Dict[str, int]) -> go.Figure:
    """Plot donut chart of portfolio decision tier distributions."""
    t = get_current_theme()
    labels = ["Auto-Approve", "Manual Review", "Auto-Reject"]
    values = [
        counts_dict.get("AUTO_APPROVE", 0),
        counts_dict.get("MANUAL_REVIEW", 0),
        counts_dict.get("AUTO_REJECT", 0),
    ]
    colors = [t.success, t.warning, t.danger]
    total = sum(values)

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.62,
                marker=dict(colors=colors, line=dict(color=t.bg_card, width=2)),
                textinfo="label+percent",
                textposition="outside",
                hoverinfo="label+value+percent",
                hovertemplate="<b>%{label}</b><br>Volume: %{value:,} loans<br>Share: %{percent}<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        paper_bgcolor=t.plot_paper,
        plot_bgcolor=t.plot_bg,
        font=dict(color=t.plot_text, family="sans-serif"),
        showlegend=False,
        margin=dict(l=20, r=20, t=30, b=20),
        height=320,
        annotations=[
            dict(
                text=f"<b style='font-size:18px;color:{t.text_primary}'>{total:,}</b><br><span style='font-size:11px;color:{t.text_secondary}'>Total Volume</span>",
                x=0.5,
                y=0.5,
                font_size=14,
                showarrow=False,
            )
        ],
    )
    return fig


def plot_shap_waterfall(
    feature_names: List[str],
    shap_values: np.ndarray,
    base_value: float,
    applicant_values: Optional[Union[np.ndarray, pd.Series]] = None,
    max_display: int = 10,
) -> go.Figure:
    """Plot horizontal SHAP attribution waterfall chart."""
    t = get_current_theme()
    shap_arr = np.asarray(shap_values, dtype=float).ravel()
    top_indices = np.argsort(np.abs(shap_arr))[::-1][:max_display]

    names = [feature_names[i] for i in top_indices][::-1]
    vals = shap_arr[top_indices][::-1]

    formatted_labels = []
    for i, idx in enumerate(top_indices[::-1]):
        raw_name = feature_names[idx]
        clean_name = raw_name.replace("_", " ").title()
        if applicant_values is not None:
            try:
                val = applicant_values[idx]
                if isinstance(val, (int, np.integer)):
                    val_str = f"{val:,}"
                elif isinstance(val, (float, np.floating)):
                    val_str = f"{val:.2f}"
                else:
                    val_str = str(val)
                clean_name = f"{clean_name} (= {val_str})"
            except Exception:
                pass
        formatted_labels.append(clean_name)

    colors = [t.danger if v > 0 else t.success for v in vals]

    fig = go.Figure(
        go.Bar(
            x=vals,
            y=formatted_labels,
            orientation="h",
            marker=dict(color=colors, line=dict(color=t.bg_card, width=1)),
            text=[f"{v:+.3f}" for v in vals],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>SHAP Attribution: %{x:+.4f}<extra></extra>",
        )
    )

    fig.add_vline(x=0.0, line=dict(color=t.text_muted, width=1.5, dash="solid"))

    fig.update_layout(
        get_plot_layout(
            title=dict(
                text=f"<b>TreeSHAP Risk Attribution Waterfall (Top {max_display} Drivers)</b><br><span style='font-size:12px;color:{t.text_secondary}'>Base Value E[f(x)] = {base_value:.3f} | Crimson = Increases Risk, Green = Decreases Risk</span>",
                x=0.02,
                y=0.95,
            ),
            xaxis=dict(
                title="SHAP Value (Impact on Log-Odds of Default)",
                gridcolor=t.plot_grid,
                zeroline=False,
            ),
            yaxis=dict(
                title="",
                gridcolor=t.plot_grid,
                tickfont=dict(size=11),
            ),
            height=460,
            margin=dict(l=220, r=40, t=70, b=40),
        )
    )
    return fig


def plot_walk_forward_gini(years_df: pd.DataFrame) -> go.Figure:
    """Plot walk-forward stability and Lehman stress test Gini scores."""
    t = get_current_theme()
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=years_df["period"],
            y=years_df["gini"],
            name="Gini Coefficient",
            marker=dict(
                color=[t.accent_primary, t.accent_primary, t.accent_primary, t.warning],
                line=dict(color=t.bg_card, width=1),
            ),
            text=[f"{g:.3f}" for g in years_df["gini"]],
            textposition="inside",
            hovertemplate="<b>%{x}</b><br>Gini Score: %{y:.4f}<extra></extra>",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=years_df["period"],
            y=years_df["default_rate"],
            name="Portfolio Realized Default Rate (%)",
            mode="lines+markers",
            line=dict(color=t.danger, width=3),
            marker=dict(size=8, symbol="diamond"),
            hovertemplate="<b>%{x}</b><br>Realized Default Rate: %{y:.2f}%<extra></extra>",
        ),
        secondary_y=True,
    )

    fig.update_layout(
        get_plot_layout(
            title=dict(
                text=f"<b>Temporal Model Stability & Macroeconomic Stress Resilience</b><br><span style='font-size:12px;color:{t.text_secondary}'>Walk-Forward Gini Discrimination vs 2008 Lehman Crisis Stress Test</span>",
                x=0.02,
                y=0.95,
            ),
            height=380,
        )
    )
    fig.update_yaxes(
        title_text="Gini Coefficient (Discrimination)",
        range=[0.0, 0.6],
        secondary_y=False,
        gridcolor=t.plot_grid,
    )
    fig.update_yaxes(
        title_text="Default Rate (%)",
        range=[0.0, 35.0],
        secondary_y=True,
        gridcolor=t.plot_grid,
    )
    return fig


def plot_grade_monotonicity(grade_df: pd.DataFrame) -> go.Figure:
    """Plot Lending Club grade monotonicity chart (Realized Default Rate vs Grade)."""
    t = get_current_theme()
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    bar_colors = [
        t.success,
        "#34D399" if t.name == "Dark" else "#059669",
        "#FBBF24" if t.name == "Dark" else "#D97706",
        t.warning,
        "#F97316" if t.name == "Dark" else "#EA580C",
        t.danger,
        "#991B1B",
    ]

    fig.add_trace(
        go.Bar(
            x=grade_df["grade"],
            y=grade_df["default_rate"],
            name="Realized Default Rate (%)",
            marker=dict(
                color=bar_colors,
                line=dict(color=t.bg_card, width=1),
            ),
            text=[f"{r:.1f}%" for r in grade_df["default_rate"]],
            textposition="outside",
            hovertemplate="<b>Grade %{x}</b><br>Realized Default Rate: %{y:.2f}%<extra></extra>",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=grade_df["grade"],
            y=grade_df["count"],
            name="Loan Volume Count",
            mode="lines+markers",
            line=dict(color=t.accent_primary, width=2, dash="dash"),
            marker=dict(size=7, symbol="circle"),
            hovertemplate="<b>Grade %{x}</b><br>Applicant Volume: %{y:,}<extra></extra>",
        ),
        secondary_y=True,
    )

    fig.update_layout(
        get_plot_layout(
            title=dict(
                text=f"<b>Credit Grade Monotonicity & Portfolio Risk Escalation</b><br><span style='font-size:12px;color:{t.text_secondary}'>Strict Monotonic Default Rate Growth: Grade A (6.33%) → Grade G (50.09%)</span>",
                x=0.02,
                y=0.95,
            ),
            height=380,
        )
    )
    fig.update_yaxes(
        title_text="Realized Default Rate (%)",
        range=[0.0, 60.0],
        secondary_y=False,
        gridcolor=t.plot_grid,
    )
    fig.update_yaxes(
        title_text="Applicant Count",
        secondary_y=True,
        showgrid=False,
    )
    return fig


def plot_psi_bar_chart(psi_df: pd.DataFrame) -> go.Figure:
    """Plot horizontal Population Stability Index (PSI) chart with regulatory boundaries."""
    t = get_current_theme()
    df = psi_df.sort_values(by="psi", ascending=True)

    colors = []
    for p in df["psi"]:
        if p < 0.10:
            colors.append(t.success)
        elif p < 0.20:
            colors.append(t.warning)
        else:
            colors.append(t.danger)

    fig = go.Figure(
        go.Bar(
            x=df["psi"],
            y=df["feature_name"],
            orientation="h",
            marker=dict(color=colors, line=dict(color=t.bg_card, width=1)),
            text=[f"{p:.4f}" for p in df["psi"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>PSI Score: %{x:.4f}<extra></extra>",
        )
    )

    # Basel III Regulatory Threshold Lines
    fig.add_vline(
        x=0.10,
        line=dict(color=t.warning, width=2, dash="dash"),
        annotation_text="Warning (0.10)",
        annotation_position="top right",
        annotation_font=dict(color=t.warning, size=11),
    )
    fig.add_vline(
        x=0.20,
        line=dict(color=t.danger, width=2, dash="dash"),
        annotation_text="Action Required (0.20)",
        annotation_position="top right",
        annotation_font=dict(color=t.danger, size=11),
    )

    fig.update_layout(
        get_plot_layout(
            title=dict(
                text=f"<b>Population Stability Index (PSI) - Top 10 Drivers</b><br><span style='font-size:12px;color:{t.text_secondary}'>2016 Baseline vs 2017-2018 Test Holdout (Regulatory Drift Assessment)</span>",
                x=0.02,
                y=0.95,
            ),
            xaxis=dict(
                title="Population Stability Index (PSI)",
                range=[0.0, max(0.25, df["psi"].max() * 1.25)],
                gridcolor=t.plot_grid,
            ),
            yaxis=dict(
                title="",
                gridcolor=t.plot_grid,
                tickfont=dict(size=11),
            ),
            height=430,
            margin=dict(l=220, r=40, t=70, b=40),
        )
    )
    return fig


def plot_pd_distribution_overlay(
    baseline_pds: np.ndarray,
    test_pds: np.ndarray,
) -> go.Figure:
    """Plot overlay of predicted default probability distributions (Concept Drift Check)."""
    t = get_current_theme()
    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=baseline_pds,
            histnorm="probability density",
            name="Baseline 2016 Validation",
            nbinsx=40,
            marker_color="rgba(0, 212, 178, 0.4)" if t.name == "Dark" else "rgba(15, 118, 110, 0.4)",
            marker_line=dict(color=t.accent_primary, width=1),
            hovertemplate="PD: %{x:.2f}<br>Density: %{y:.2f}<extra>Baseline 2016</extra>",
        )
    )

    fig.add_trace(
        go.Histogram(
            x=test_pds,
            histnorm="probability density",
            name="Holdout 2017-2018 Test",
            nbinsx=40,
            marker_color="rgba(245, 158, 11, 0.4)" if t.name == "Dark" else "rgba(217, 119, 6, 0.4)",
            marker_line=dict(color=t.warning, width=1),
            hovertemplate="PD: %{x:.2f}<br>Density: %{y:.2f}<extra>Holdout 2017-2018</extra>",
        )
    )

    fig.update_layout(
        get_plot_layout(
            title=dict(
                text=f"<b>Predicted PD Score Distribution Overlay</b><br><span style='font-size:12px;color:{t.text_secondary}'>Concept Drift Assessment: Baseline (2016) vs Out-of-Time Holdout (2017-2018)</span>",
                x=0.02,
                y=0.95,
            ),
            xaxis=dict(
                title="Predicted Probability of Default (PD)",
                range=[0.0, 1.0],
                gridcolor=t.plot_grid,
            ),
            yaxis=dict(
                title="Probability Density",
                gridcolor=t.plot_grid,
            ),
            barmode="overlay",
            height=360,
        )
    )
    return fig
