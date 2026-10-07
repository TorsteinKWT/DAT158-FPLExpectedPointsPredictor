import requests
import streamlit as st

from src.data import CURRENT_SEASON
from src.loaders import load_entry, load_model, load_players
from src.team import pick_best_team

st.caption(
    f"Always uses {CURRENT_SEASON} and the next gameweek, whatever is chosen in the sidebar."
)

entry_id = st.text_input(
    "FPL team ID",
    key="entry_id",
    help="The number in the address of your Points page: fantasy.premierleague.com/entry/**1234567**/event/5",
    width=300,
)

if not entry_id:
    st.stop()
if not entry_id.isdigit():
    st.error("The team ID should be a number.", icon=":material/error:")
    st.stop()

try:
    team_name, picked_gameweek, picks = load_entry(int(entry_id))
except requests.HTTPError:
    st.error("Found no team with that ID.", icon=":material/error:")
    st.stop()

bundle = load_model()
_, upcoming = load_players(CURRENT_SEASON)
if upcoming["GW"].isna().all():
    st.info("There is no upcoming gameweek to predict.", icon=":material/info:")
    st.stop()
next_gameweek = int(upcoming["GW"].dropna().iat[0])

# Én rad per spiller per kamp i neste runde. Spillere uten kamp har en rad uten motstander og får null poeng.
squad = picks.merge(upcoming, on="element", how="left")
unknown = squad["name"].isna()
if unknown.any():
    st.warning(f"Left out {unknown.sum()} player(s) without any matches this season.", icon=":material/warning:")
    squad = squad[~unknown]

squad["predicted_points"] = bundle["model"].predict(squad[bundle["features"]])
squad.loc[squad["opponent"].isna(), "predicted_points"] = 0
squad["fixture"] = (squad["opponent"] + squad["was_home"].map({1: " (H)", 0: " (A)"})).fillna("No match")

squad = squad.groupby("element", as_index=False, sort=False).agg(
    {
        "name": "first",
        "team": "first",
        "position": "first",
        "is_starter": "first",
        "is_captain": "first",
        "is_vice_captain": "first",
        "fixture": ", ".join,
        "predicted_points": "sum",
    }
)
squad["role"] = "Starter"
squad.loc[squad["is_vice_captain"], "role"] = "Vice-captain"
squad.loc[squad["is_captain"], "role"] = "Captain"
squad.loc[~squad["is_starter"], "role"] = "Bench"

starters = squad[squad["is_starter"]]
# Kapteinen får doble poeng.
lineup_points = starters["predicted_points"].sum() + squad.loc[squad["is_captain"], "predicted_points"].sum()
best_lineup = pick_best_team(squad)
best_points = best_lineup["predicted_points"].sum() + best_lineup["predicted_points"].max()

st.subheader(team_name)
st.caption(f"Squad as picked in gameweek {picked_gameweek}. Later transfers and lineup changes are not visible.")

with st.container(horizontal=True):
    st.metric(f"Expected points, gameweek {next_gameweek}", f"{lineup_points:.1f}", border=True)
    st.metric(
        "Best lineup from squad",
        f"{best_points:.1f}",
        border=True,
        help="The best eleven of your fifteen in a valid formation, with the top player as captain.",
    )
    st.metric("On the bench", f"{squad.loc[~squad['is_starter'], 'predicted_points'].sum():.1f}", border=True)

st.dataframe(
    squad[["name", "team", "position", "fixture", "role", "predicted_points"]],
    column_config={
        "name": "Player",
        "team": "Team",
        "position": "Position",
        "fixture": "Fixture",
        "role": "Role",
        "predicted_points": st.column_config.ProgressColumn(
            "Expected points",
            format="%.2f",
            min_value=0,
            max_value=float(squad["predicted_points"].max()),
        ),
    },
    hide_index=True,
    height="content",
    alt="Your squad with expected points for the next gameweek",
)
