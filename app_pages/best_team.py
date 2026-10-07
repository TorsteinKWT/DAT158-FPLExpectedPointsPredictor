import streamlit as st

from src.data import POSITION_NAMES, POSITIONS
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
    st.metric("Forventede poeng", f"{team['predicted_points'].sum():.1f}", border=True)
    if is_played:
        st.metric("Faktiske poeng", int(team["total_points"].sum()), border=True)
        best_possible = pick_best_team(players, points="total_points")["total_points"].sum()
        st.metric("Best mulig", int(best_possible), border=True, help="De elleve som faktisk fikk flest poeng.")
    st.metric("Formasjon", formation, border=True)
    st.metric("Samlet pris", f"£{team['price'].sum():.1f}m", border=True)

for position in POSITIONS:
    with st.container(horizontal=True, horizontal_alignment="center"):
        for player in team[team["position"] == position].itertuples():
            with st.container(border=True, width=170):
                st.markdown(f"**{player.name}**")
                st.caption(f"{POSITION_NAMES[position]} · {player.team} · £{player.price:.1f}m")
                points = f":green[**{player.predicted_points:.1f}**] forventet"
                if is_played:
                    points += f" · **{int(player.total_points)}** faktisk"
                st.markdown(points)

st.caption("De elleve spillerne med flest forventede poeng i en gyldig formasjon. Uten budsjett og uten grense per klubb.")
