"""Central Theme & Color Palette Engine for Credit Risk Dashboard.

Provides consistent styling tokens and layouts for both Dark and Light modes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict
import streamlit as st


@dataclass(frozen=True)
class ThemeTokens:
    name: str
    bg_app: str
    bg_card: str
    bg_card_secondary: str
    border: str
    border_subtle: str
    text_primary: str
    text_secondary: str
    text_muted: str
    accent_primary: str
    accent_secondary: str
    success: str
    success_bg: str
    success_border: str
    warning: str
    warning_bg: str
    warning_border: str
    danger: str
    danger_bg: str
    danger_border: str
    info: str
    info_bg: str
    info_border: str
    plot_paper: str
    plot_bg: str
    plot_grid: str
    plot_text: str


DARK_THEME = ThemeTokens(
    name="Dark",
    bg_app="#0A0E17",
    bg_card="#131B2A",
    bg_card_secondary="#1A2333",
    border="#1E293B",
    border_subtle="#2A374A",
    text_primary="#F8FAFC",
    text_secondary="#94A3B8",
    text_muted="#64748B",
    accent_primary="#00D4B2",
    accent_secondary="#38BDF8",
    success="#10B981",
    success_bg="rgba(16, 185, 129, 0.15)",
    success_border="rgba(16, 185, 129, 0.35)",
    warning="#F59E0B",
    warning_bg="rgba(245, 158, 11, 0.15)",
    warning_border="rgba(245, 158, 11, 0.35)",
    danger="#EF4444",
    danger_bg="rgba(239, 68, 68, 0.15)",
    danger_border="rgba(239, 68, 68, 0.35)",
    info="#00D4B2",
    info_bg="rgba(0, 212, 178, 0.15)",
    info_border="rgba(0, 212, 178, 0.35)",
    plot_paper="rgba(0,0,0,0)",
    plot_bg="#131B2A",
    plot_grid="#1E293B",
    plot_text="#E2E8F0",
)

LIGHT_THEME = ThemeTokens(
    name="Light",
    bg_app="#F8FAFC",
    bg_card="#FFFFFF",
    bg_card_secondary="#F1F5F9",
    border="#E2E8F0",
    border_subtle="#CBD5E1",
    text_primary="#0F172A",
    text_secondary="#334155",
    text_muted="#64748B",
    accent_primary="#0F766E",
    accent_secondary="#0284C7",
    success="#16A34A",
    success_bg="rgba(22, 163, 74, 0.10)",
    success_border="rgba(22, 163, 74, 0.30)",
    warning="#D97706",
    warning_bg="rgba(217, 119, 6, 0.10)",
    warning_border="rgba(217, 119, 6, 0.30)",
    danger="#DC2626",
    danger_bg="rgba(220, 38, 38, 0.10)",
    danger_border="rgba(220, 38, 38, 0.30)",
    info="#0284C7",
    info_bg="rgba(2, 132, 199, 0.10)",
    info_border="rgba(2, 132, 199, 0.30)",
    plot_paper="rgba(0,0,0,0)",
    plot_bg="#FFFFFF",
    plot_grid="#E2E8F0",
    plot_text="#0F172A",
)


def get_current_theme() -> ThemeTokens:
    """Retrieve active ThemeTokens based on st.session_state."""
    theme_name = st.session_state.get("theme_mode", "Dark")
    return LIGHT_THEME if theme_name == "Light" else DARK_THEME


def clean_html(html_str: str) -> str:
    """Remove multiline indentation to prevent CommonMark code block rendering."""
    lines = [line.strip() for line in html_str.strip().splitlines() if line.strip()]
    return "".join(lines)
