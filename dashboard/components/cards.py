"""Executive & Regulatory UI Cards for Streamlit Credit Risk Dashboard.

Constructs institutional elevated cards, delta badges, and statutory
ECOA/FCRA adverse action disclosure boxes supporting dynamic Dark and Light themes.
"""

from __future__ import annotations

import html
from typing import Any, Dict, List, Optional
import streamlit as st

from dashboard.components.theme import clean_html, get_current_theme


def render_kpi_card(
    title: str,
    value: str,
    delta: Optional[str] = None,
    status: str = "normal",
    subtitle: Optional[str] = None,
) -> str:
    """Render an institutional elevated KPI metric card.

    Parameters
    ----------
    title : str
        Metric title (e.g. 'PORTFOLIO EXPOSURE EVALUATED').
    value : str
        Headline metric value (e.g. '$3.26 Billion').
    delta : str, optional
        Delta text / badge (e.g. '-47.51% Expected Loss').
    status : str, default='normal'
        Status badge type: 'positive', 'negative', 'warning', 'info', or 'normal'.
    subtitle : str, optional
        Secondary explanatory text below value.

    Returns
    -------
    str
        The generated clean HTML string.
    """
    t = get_current_theme()

    badge_styles = {
        "positive": f"background: {t.success_bg}; color: {t.success}; border: 1px solid {t.success_border};",
        "negative": f"background: {t.danger_bg}; color: {t.danger}; border: 1px solid {t.danger_border};",
        "warning": f"background: {t.warning_bg}; color: {t.warning}; border: 1px solid {t.warning_border};",
        "info": f"background: {t.info_bg}; color: {t.info}; border: 1px solid {t.info_border};",
        "normal": f"background: {t.bg_card_secondary}; color: {t.text_secondary}; border: 1px solid {t.border};",
    }
    badge_style = badge_styles.get(status, badge_styles["normal"])

    delta_html = ""
    if delta:
        delta_html = (
            f'<div style="display: inline-block; padding: 2px 10px; border-radius: 9999px; '
            f'font-size: 0.72rem; font-weight: 600; {badge_style}">'
            f"{html.escape(delta)}</div>"
        )

    subtitle_html = ""
    if subtitle:
        subtitle_html = (
            f'<div style="font-size: 0.76rem; color: {t.text_muted}; margin-top: 6px; line-height: 1.3;">'
            f"{html.escape(subtitle)}</div>"
        )

    card_shadow = "0 4px 16px rgba(0, 0, 0, 0.25)" if t.name == "Dark" else "0 2px 10px rgba(0, 0, 0, 0.05)"

    raw_html = f"""
    <div style="
        background: {t.bg_card};
        border: 1px solid {t.border};
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: {card_shadow};
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    ">
        <div>
            <div style="
                color: {t.text_secondary};
                font-size: 0.72rem;
                font-weight: 700;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                margin-bottom: 8px;
            ">
                {html.escape(title)}
            </div>
            <div style="
                color: {t.text_primary};
                font-size: 1.75rem;
                font-weight: 700;
                line-height: 1.2;
                letter-spacing: -0.02em;
                margin-bottom: 8px;
            ">
                {html.escape(value)}
            </div>
        </div>
        <div style="margin-top: 4px;">
            {delta_html}
            {subtitle_html}
        </div>
    </div>
    """
    cleaned = clean_html(raw_html)
    st.markdown(cleaned, unsafe_allow_html=True)
    return cleaned


def render_decision_badge(decision_tier: str) -> str:
    """Render a semantic pill badge for credit decision tier.

    Parameters
    ----------
    decision_tier : str
        One of 'AUTO_APPROVE', 'MANUAL_REVIEW', 'AUTO_REJECT'.

    Returns
    -------
    str
        Clean HTML string of the badge.
    """
    t = get_current_theme()
    tier_upper = decision_tier.upper().strip()

    if "APPROVE" in tier_upper:
        color = t.success
        bg = t.success_bg
        border = t.success_border
        icon = "🟢"
        label = "AUTO-APPROVE"
    elif "REVIEW" in tier_upper:
        color = t.warning
        bg = t.warning_bg
        border = t.warning_border
        icon = "🟡"
        label = "MANUAL REVIEW"
    else:
        color = t.danger
        bg = t.danger_bg
        border = t.danger_border
        icon = "🔴"
        label = "AUTO-REJECT"

    raw_html = f"""
    <span style="
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 14px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        background: {bg};
        color: {color};
        border: 1px solid {border};
    ">
        <span>{icon}</span> {label}
    </span>
    """
    return clean_html(raw_html)


def render_adverse_action_box(
    applicant_id: str,
    reasons: List[str],
    pd_score: float,
    threshold: float,
    applicant_data: Optional[Dict[str, Any]] = None,
) -> None:
    """Render formal ECOA/FCRA statutory adverse action disclosure card.

    Parameters
    ----------
    applicant_id : str
        Unique applicant identifier.
    reasons : list of str
        Top 3 plain-English reasons for denial.
    pd_score : float
        Calculated probability of default.
    threshold : float
        Policy threshold that was exceeded.
    applicant_data : dict, optional
        Optional applicant metadata (e.g. date, loan amount).
    """
    t = get_current_theme()

    reasons_list_html = ""
    for idx, reason in enumerate(reasons[:3], start=1):
        reasons_list_html += f"""
        <div style="display: flex; align-items: flex-start; gap: 10px; margin-bottom: 8px;">
            <div style="
                background: {t.danger};
                color: #FFFFFF;
                font-weight: 700;
                font-size: 0.72rem;
                width: 20px;
                height: 20px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                flex-shrink: 0;
                margin-top: 2px;
            ">{idx}</div>
            <div style="color: {t.text_primary}; font-size: 0.88rem; line-height: 1.4;">
                {html.escape(reason)}
            </div>
        </div>
        """

    loan_amnt_str = ""
    if applicant_data and "loan_amnt" in applicant_data:
        loan_amnt_str = f"&nbsp;&nbsp;|&nbsp;&nbsp;Requested: <b>${applicant_data['loan_amnt']:,.0f}</b>"

    box_bg = "#0B111E" if t.name == "Dark" else "#FEF2F2"
    box_border = t.danger
    inner_summary_bg = t.bg_card

    raw_html = f"""
    <div style="
        background: {box_bg};
        border: 1px solid {box_border};
        border-left: 5px solid {box_border};
        border-radius: 10px;
        padding: 20px 24px;
        margin: 16px 0;
    ">
        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid {t.border}; padding-bottom: 12px; margin-bottom: 14px;">
            <div>
                <div style="color: {t.danger}; font-size: 0.75rem; font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase;">
                    STATUTORY ADVERSE ACTION NOTICE
                </div>
                <div style="color: {t.text_secondary}; font-size: 0.78rem;">
                    Pursuant to Equal Credit Opportunity Act (ECOA, Reg. B) § 701 & Fair Credit Reporting Act (FCRA) § 615
                </div>
            </div>
            <div style="text-align: right; color: {t.text_secondary}; font-size: 0.78rem;">
                Applicant Ref: <span style="color: {t.accent_primary}; font-weight: 600;">{html.escape(applicant_id)}</span>{loan_amnt_str}
            </div>
        </div>

        <div style="display: flex; gap: 20px; margin-bottom: 16px; background: {inner_summary_bg}; padding: 10px 16px; border-radius: 8px; border: 1px solid {t.border};">
            <div style="font-size: 0.8rem; color: {t.text_secondary};">
                Evaluated Default Probability: <b style="color: {t.danger}; font-size: 0.95rem;">{pd_score * 100:.2f}%</b>
            </div>
            <div style="font-size: 0.8rem; color: {t.text_secondary};">
                Policy Hurdle Threshold: <b style="color: {t.text_primary}; font-size: 0.95rem;">{threshold * 100:.2f}%</b>
            </div>
            <div style="font-size: 0.8rem; color: {t.text_secondary};">
                Underwriting Action: <b style="color: {t.danger};">DECLINED / REFERRED</b>
            </div>
        </div>

        <div style="color: {t.text_primary}; font-size: 0.82rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 10px;">
            Principal Risk Factors Influencing Decision:
        </div>

        <div style="margin-bottom: 16px;">
            {reasons_list_html}
        </div>

        <div style="border-top: 1px solid {t.border}; padding-top: 10px; color: {t.text_muted}; font-size: 0.72rem; line-height: 1.4;">
            <b>Federal Non-Discrimination Notice:</b> The Federal Equal Credit Opportunity Act prohibits creditors
            from discriminating against credit applicants on the basis of race, color, religion, national origin,
            sex, marital status, age (provided the applicant has the capacity to contract), or because all or part
            of the applicant's income derives from any public assistance program.
        </div>
    </div>
    """
    st.markdown(clean_html(raw_html), unsafe_allow_html=True)
