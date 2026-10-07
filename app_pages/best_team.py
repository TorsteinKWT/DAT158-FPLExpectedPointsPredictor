import streamlit as st

from src.data import POSITION_NAMES, POSITIONS
from src.team import pick_best_team, pick_captains
from src.ui import captain_card

is_played = st.session_state.is_played

# En dobbeltrunde har én rad per kamp, så de summeres for å få spillerens total for runden.
totals = {"predicted_points": "sum"}
if is_played:
    totals["total_points"] = "sum"
players = st.session_state.players.groupby("element", as_index=False).agg(
    {"name": "first", "team": "first", "position": "first", "price": "first", "fixture": ", ".join, **totals}
)

team = pick_best_team(players)
captain, vice = pick_captains(team)
formation = "-".join(str((team["position"] == position).sum()) for position in POSITIONS[1:])

with st.container(horizontal=True):
    # Kapteinen teller dobbelt, både i forventede og faktiske poeng.
    st.metric(
        "Forventede poeng",
        f"{team['predicted_points'].sum() + captain['predicted_points']:.1f}",
        border=True,
        help="Kapteinen teller dobbelt.",
    )
    if is_played:
        st.metric("Faktiske poeng", int(team["total_points"].sum() + captain["total_points"]), border=True)
        best_possible = pick_best_team(players, points="total_points")["total_points"]
        st.metric(
            "Best mulig",
            int(best_possible.sum() + best_possible.max()),
            border=True,
            help="De elleve som faktisk fikk flest poeng, med den beste som kaptein.",
        )
    st.metric("Formasjon", formation, border=True)
    st.metric("Samlet pris", f"£{team['price'].sum():.1f}m", border=True)

captain_card(captain, vice)

# Banen: én rad per posisjon, med keeperen øverst.
with st.container(border=True):
    for position in POSITIONS:
        st.caption(POSITION_NAMES[position].upper(), text_alignment="center")
        with st.container(horizontal=True, horizontal_alignment="center"):
            for player in team[team["position"] == position].itertuples():
                with st.container(border=True, width=170):
                    mark = {captain["element"]: " :violet-badge[K]", vice["element"]: " :gray-badge[V]"}
                    st.markdown(f"**{player.name}**{mark.get(player.element, '')}")
                    st.caption(f"{player.team} · {player.fixture} · £{player.price:.1f}m")
                    points = f":green-badge[**{player.predicted_points:.1f}** xP]"
                    if is_played:
                        points += f" :gray-badge[{int(player.total_points)} faktisk]"
                    st.markdown(points)

st.caption(
    "De elleve spillerne med flest forventede poeng (xP) i en gyldig formasjon. Uten budsjett og uten grense per klubb."
)
