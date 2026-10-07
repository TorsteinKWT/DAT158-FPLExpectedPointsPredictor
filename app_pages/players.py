import streamlit as st

from src.data import POSITION_NAMES, POSITIONS

players = st.session_state.players

with st.sidebar:
    positions = st.pills(
        "Posisjoner", POSITIONS, selection_mode="multi", default=POSITIONS, format_func=POSITION_NAMES.get
    )
    search = st.text_input("Søk etter spiller eller lag")

columns = [
    "name", "team", "position", "fixture", "fdr", "price",
    "total_points_roll", "minutes_roll", "predicted_points",
]  # fmt: skip
if st.session_state.is_played:
    columns.append("total_points")
st.caption("Én rad per kamp, så spillere med dobbeltrunde kan stå to ganger.")

shown = players[players["position"].isin(positions)]
if search:
    matches = shown["name"].str.contains(search, case=False) | shown["team"].str.contains(search, case=False)
    shown = shown[matches]
shown = shown.sort_values("predicted_points", ascending=False)

st.dataframe(
    shown[columns].replace({"position": POSITION_NAMES}),
    column_config={
        "name": "Spiller",
        "team": "Lag",
        "position": "Posisjon",
        "fixture": "Kamp",
        "fdr": st.column_config.NumberColumn(
            "FDR", format="%d", help="FPLs vanskelighetsgrad for kampen, fra 1 (lett) til 5 (vanskelig)."
        ),
        "price": st.column_config.NumberColumn("Pris", format="£%.1fm"),
        "total_points_roll": st.column_config.NumberColumn("Poengsnitt (siste 5)", format="%.1f"),
        "minutes_roll": st.column_config.NumberColumn("Minutter i snitt (siste 5)", format="%.0f"),
        "predicted_points": st.column_config.ProgressColumn(
            "Forventede poeng",
            format="%.2f",
            min_value=0,
            max_value=float(players["predicted_points"].max()),
        ),
        "total_points": "Faktiske poeng",
    },
    hide_index=True,
    height=600,
    alt="Spillere rangert etter forventede poeng",
)
