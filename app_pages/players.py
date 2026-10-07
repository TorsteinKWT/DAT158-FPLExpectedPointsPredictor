import streamlit as st

from src.data import POSITIONS

players = st.session_state.players

with st.sidebar:
    positions = st.pills("Positions", POSITIONS, selection_mode="multi", default=POSITIONS)
    search = st.text_input("Search player or team")

columns = ["name", "team", "position", "price", "total_points_roll", "minutes_roll", "predicted_points"]
if st.session_state.is_played:
    columns.insert(3, "venue")
    columns.append("total_points")
    st.caption("One row per match, so players with a double gameweek can appear twice.")

shown = players[players["position"].isin(positions)]
if search:
    matches = shown["name"].str.contains(search, case=False) | shown["team"].str.contains(search, case=False)
    shown = shown[matches]
shown = shown.sort_values("predicted_points", ascending=False)

st.dataframe(
    shown[columns],
    column_config={
        "name": "Player",
        "team": "Team",
        "position": "Position",
        "venue": "Venue",
        "price": st.column_config.NumberColumn("Price", format="£%.1fm"),
        "total_points_roll": st.column_config.NumberColumn("Avg points (last 5)", format="%.1f"),
        "minutes_roll": st.column_config.NumberColumn("Avg minutes (last 5)", format="%.0f"),
        "predicted_points": st.column_config.ProgressColumn(
            "Expected points",
            format="%.2f",
            min_value=0,
            max_value=float(players["predicted_points"].max()),
        ),
        "total_points": "Actual points",
    },
    hide_index=True,
    height=600,
    alt="Players ranked by expected points",
)
