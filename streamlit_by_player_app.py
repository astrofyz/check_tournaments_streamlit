#!/usr/bin/env python3
"""Local Streamlit UI for the player-tournaments overlap check.

Uses logic from ``tourn_check_web_by_player`` (no Railway or FastAPI).

Install and run::

    pip install streamlit
    streamlit run streamlit_by_player_app.py
"""

from __future__ import annotations

import json
import os

import pandas as pd
import streamlit as st

from tourn_check_web_by_player import (
    DEFAULT_BASE,
    RatingAPIError,
    parse_player_ids_from_text,
    parse_tournament_lines_from_text,
    run_check,
)

_CLEAR_ROW_BG = "background-color: #e8f5e9"


def _style_summary_clear_green(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    def row_style(row: pd.Series) -> list[str]:
        if str(row.get("status", "")) == "clear":
            return [_CLEAR_ROW_BG] * len(row.index)
        return [""] * len(row.index)

    return df.style.apply(row_style, axis=1)

st.set_page_config(page_title="Tournament check (by player)", layout="wide")
st.title("Tournament check")
st.markdown(
    "Проверка заигранности турниров кем-то из игроков"
)

players = st.text_area("Players", height=100, placeholder="12345\n67890")
tournaments = st.text_area("Tournaments", height=100, placeholder="Substring or id per line")

c1, c2 = st.columns(2)
with c1:
    date_after = st.text_input(
        "dateEnd strictly after (optional)",
        placeholder="YYYY-MM-DD",
        help="Only affects name-based tournament lookup.",
    )
with c2:
    env_base = (os.environ.get("TOURN_CHECK_BASE_URL") or "").strip()
    base_url = st.text_input(
        "API base URL",
        value=env_base or DEFAULT_BASE,
    )

if st.button("Run check", type="primary"):
    try:
        player_ids = parse_player_ids_from_text(players)
        tournament_lines = parse_tournament_lines_from_text(tournaments)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    base = (base_url.strip() or DEFAULT_BASE).rstrip("/")
    date_end = date_after.strip() or None

    with st.spinner("Running check…"):
        try:
            report = run_check(
                player_ids,
                tournament_lines,
                base_url=base,
                date_end_after=date_end,
                verbose=False,
            )
        except RatingAPIError as exc:
            st.error(str(exc))
            st.stop()

    st.subheader("Summary")
    summary_df = pd.DataFrame(report["summary"])
    st.dataframe(
        _style_summary_clear_green(summary_df),
        use_container_width=True,
        hide_index=True,
    )

    warns = report.get("warnings") or []
    if warns:
        with st.expander(f"Warnings ({len(warns)})", expanded=True):
            st.json(warns)

    full_json = json.dumps(report, ensure_ascii=False, indent=2)
    st.download_button(
        "Download full report (JSON)",
        data=full_json,
        file_name="tourn_check_report.json",
        mime="application/json",
    )
    with st.expander("Full report (JSON)"):
        st.code(full_json, language="json")
