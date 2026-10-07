"""Elementer som vises på flere sider i appen."""

import pandas as pd
import streamlit as st


def captain_card(captain: pd.Series, vice: pd.Series) -> None:
    """Vis anbefalt kaptein og visekaptein på én linje."""
    with st.container(border=True, horizontal=True, vertical_alignment="center"):
        st.badge("Anbefalt kaptein", icon=":material/star:", color="green")
        st.markdown(f"**{captain['name']}** · {captain['team']} · {captain['fixture']}")
        st.markdown(f":green-badge[**{captain['predicted_points']:.1f}** xP]")
        st.caption(f"Visekaptein: {vice['name']} ({vice['predicted_points']:.1f} xP)")
