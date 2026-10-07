import streamlit as st

from src.data import POSITIONS
from src.team import pick_best_team

is_played = st.session_state.is_played

# En dobbeltrunde har én rad per kamp, så de summeres for å få spillerens total for runden.
totals = {"predicted_points": "sum"}
if is_played:
    totals["total_points"] = "sum"
players = st.session_state.players.groupby("element", as_index=False).agg(
    {"name": "first", "team": "first", "position": "first", "price": "first", **totals}
)

team = pick_best_team(players)
formation = "-".join(str((team["position"] == position).sum()) for position in POSITIONS[1:])

with st.container(horizontal=True):
    st.metric("Expected points", f"{team['predicted_points'].sum():.1f}", border=True)
    if is_played:
        st.metric("Actual points", int(team["total_points"].sum()), border=True)
        best_possible = pick_best_team(players, points="total_points")["total_points"].sum()
        st.metric("Best possible", int(best_possible), border=True, help="The eleven that actually scored the most.")
    st.metric("Formation", formation, border=True)
    st.metric("Total price", f"£{team['price'].sum():.1f}m", border=True)

for position in POSITIONS:
    with st.container(horizontal=True, horizontal_alignment="center"):
        for player in team[team["position"] == position].itertuples():
            with st.container(border=True, width=170):
                st.markdown(f"**{player.name}**")
                st.caption(f"{position} · {player.team} · £{player.price:.1f}m")
                points = f":green[**{player.predicted_points:.1f}**] expected"
                if is_played:
                    points += f" · **{int(player.total_points)}** actual"
                st.markdown(points)

st.caption("The eleven players with the highest expected points in a valid formation. No budget or per-club limit.")
