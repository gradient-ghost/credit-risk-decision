"""Credit Risk Underwriting & Policy Intelligence Dashboard.

Main application entry point. Implements an institutional interface
with sub-10ms dynamic policy simulation, single-applicant TreeSHAP explainability,
Basel III model risk governance, and full Dark/Light mode theme switching.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from dashboard.components.theme import clean_html, get_current_theme
from dashboard.tabs.tab1_executive import render_tab1
from dashboard.tabs.tab2_policy_sim import render_tab2
from dashboard.tabs.tab3_applicant import render_tab3
from dashboard.tabs.tab4_stability import render_tab4
from dashboard.tabs.tab5_monitoring import render_tab5


def setup_page():
    """Configure Streamlit page settings and inject dynamic theme CSS."""
    st.set_page_config(
        page_title="Credit Risk Underwriting & Policy Intelligence",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    if "theme_mode" not in st.session_state:
        st.session_state.theme_mode = "Dark"

    t = get_current_theme()

    # Dynamic Institutional CSS based on active theme
    css = f"""
    <style>
    /* App background & default text */
    .stApp {{
        background-color: {t.bg_app} !important;
        color: {t.text_primary} !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }}

    /* Global typography */
    h1, h2, h3, h4, h5, h6 {{
        color: {t.text_primary} !important;
    }}
    p, span, label, div {{
        color: {t.text_primary};
    }}

    /* Native containers */
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background-color: {t.bg_card};
        border-color: {t.border} !important;
        border-radius: 12px;
    }}

    /* Tab Bar Styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 12px;
        background-color: transparent;
        border-bottom: 2px solid {t.border};
        padding-bottom: 4px;
    }}

    .stTabs [data-baseweb="tab"] {{
        height: 44px;
        background-color: {t.bg_card};
        border: 1px solid {t.border};
        border-radius: 8px 8px 0 0;
        color: {t.text_secondary};
        font-weight: 600;
        font-size: 0.88rem;
        padding: 0 18px;
        transition: all 0.2s ease-in-out;
    }}

    .stTabs [data-baseweb="tab"]:hover {{
        color: {t.accent_primary};
        border-color: {t.border_subtle};
        background-color: {t.bg_card_secondary};
    }}

    .stTabs [aria-selected="true"] {{
        background-color: {t.bg_card_secondary} !important;
        border: 1px solid {t.accent_primary} !important;
        border-bottom: 2px solid {t.accent_primary} !important;
        color: {t.accent_primary} !important;
    }}

    /* Sliders */
    div[data-testid="stSlider"] > div {{
        color: {t.accent_primary};
    }}

    /* Button styling */
    .stButton button {{
        background: {t.bg_card};
        color: {t.text_primary};
        border: 1px solid {t.border};
        border-radius: 8px;
        font-size: 0.82rem;
        font-weight: 600;
        transition: all 0.2s;
    }}

    .stButton button:hover {{
        background: {t.bg_card_secondary};
        border-color: {t.accent_primary};
        color: {t.accent_primary};
    }}

    /* Metric cards */
    div[data-testid="stMetricValue"] {{
        color: {t.text_primary} !important;
        font-weight: 700;
    }}

    /* Selectbox */
    div[data-baseweb="select"] > div {{
        background-color: {t.bg_card} !important;
        color: {t.text_primary} !important;
        border-color: {t.border} !important;
    }}

    /* Expander */
    .streamlit-expanderHeader {{
        background-color: {t.bg_card};
        border: 1px solid {t.border};
        border-radius: 8px;
        color: {t.text_primary};
        font-weight: 600;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_header():
    """Render institutional header banner with integrated Theme Switcher."""
    t = get_current_theme()

    hdr_col_left, hdr_col_right = st.columns([0.82, 0.18])

    with hdr_col_left:
        header_html = f"""
        <div style="
            background: {t.bg_card};
            border: 1px solid {t.border};
            border-radius: 12px;
            padding: 14px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        ">
            <div>
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 1.5rem;">🛡️</span>
                    <span style="font-size: 1.15rem; font-weight: 800; color: {t.text_primary}; letter-spacing: -0.01em;">
                        CREDIT RISK UNDERWRITING & POLICY INTELLIGENCE
                    </span>
                    <span style="background: {t.info_bg}; color: {t.accent_primary}; border: 1px solid {t.info_border}; font-size: 0.70rem; font-weight: 700; padding: 2px 8px; border-radius: 9999px;">
                        v2.4 INSTITUTIONAL
                    </span>
                </div>
                <div style="color: {t.text_secondary}; font-size: 0.78rem; margin-top: 4px;">
                    Calibrated Machine Learning Engine • Basel III Compliance • TreeSHAP Explainability • ECOA Statutory Notices
                </div>
            </div>
            <div style="display: flex; gap: 16px; text-align: right;">
                <div>
                    <div style="color: {t.text_muted}; font-size: 0.68rem; text-transform: uppercase; font-weight: 700;">Champion Model</div>
                    <div style="color: {t.accent_primary}; font-size: 0.80rem; font-weight: 700;">XGBoost (Platt Calibrated)</div>
                </div>
                <div style="border-left: 1px solid {t.border}; padding-left: 14px;">
                    <div style="color: {t.text_muted}; font-size: 0.68rem; text-transform: uppercase; font-weight: 700;">Holdout Cohort</div>
                    <div style="color: {t.text_primary}; font-size: 0.80rem; font-weight: 700;">225,639 Loans ($3.26B)</div>
                </div>
                <div style="border-left: 1px solid {t.border}; padding-left: 14px;">
                    <div style="color: {t.text_muted}; font-size: 0.68rem; text-transform: uppercase; font-weight: 700;">Health Audit</div>
                    <div style="color: {t.success}; font-size: 0.80rem; font-weight: 700;">🟢 Basel III Pass</div>
                </div>
            </div>
        </div>
        """
        st.markdown(clean_html(header_html), unsafe_allow_html=True)

    with hdr_col_right:
        current_selection = "🌙 Dark" if st.session_state.theme_mode == "Dark" else "☀️ Light"
        selected_theme = st.segmented_control(
            "Theme Mode",
            options=["🌙 Dark", "☀️ Light"],
            default=current_selection,
            label_visibility="collapsed",
            key="theme_mode_selector",
        )
        if selected_theme:
            new_mode = "Light" if "Light" in selected_theme else "Dark"
            if new_mode != st.session_state.theme_mode:
                st.session_state.theme_mode = new_mode
                st.rerun()

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)


def main():
    """Application main orchestrator."""
    setup_page()
    render_header()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📊 Executive Summary",
            "⚙️ Policy Simulator",
            "🔍 Individual Underwriting",
            "📈 Portfolio Stability",
            "🛡️ Basel III Monitoring",
        ]
    )

    with tab1:
        render_tab1()

    with tab2:
        render_tab2()

    with tab3:
        render_tab3()

    with tab4:
        render_tab4()

    with tab5:
        render_tab5()


if __name__ == "__main__":
    main()
