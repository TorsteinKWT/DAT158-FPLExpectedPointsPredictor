import requests
import streamlit as st

from src.data import CURRENT_SEASON, POSITION_NAMES
from src.loaders import load_entry, load_model, load_players
from src.team import pick_best_team, pick_captains
from src.ui import captain_card

st.caption(f"Bruker alltid {CURRENT_SEASON} og neste runde, uansett hva som er valgt i sidepanelet.")

entry_id = st.text_input(
    "Lag-ID i FPL",
    key="entry_id",
    help="Tallet i adressen til poengsiden din: fantasy.premierleague.com/entry/**1234567**/event/5",
    width=300,
)

if not entry_id:
    st.stop()
if not entry_id.isdigit():
    st.error("Lag-ID-en må være et tall.", icon=":material/error:")
    st.stop()

try:
    team_name, picked_gameweek, picks = load_entry(int(entry_id))
except requests.HTTPError:
    st.error("Fant ikke noe lag med den ID-en.", icon=":material/error:")
    st.stop()

bundle = load_model()
_, upcoming = load_players(CURRENT_SEASON)
if upcoming["GW"].isna().all():
    st.info("Det er ingen kommende runde å predikere.", icon=":material/info:")
    st.stop()
next_gameweek = int(upcoming["GW"].dropna().iat[0])

# Én rad per spiller per kamp i neste runde. Spillere uten kamp har en rad uten motstander og får null poeng.
squad = picks.merge(upcoming, on="element", how="left")
unknown = squad["name"].isna()
if unknown.any():
    st.warning(f"{unknown.sum()} spiller(e) uten kamper denne sesongen er utelatt.", icon=":material/warning:")
    squad = squad[~unknown]

squad["predicted_points"] = st.session_state.model.predict(squad[bundle["features"]])
squad.loc[squad["opponent"].isna(), "predicted_points"] = 0
squad["fixture"] = (squad["opponent"] + squad["was_home"].map({1: " (H)", 0: " (B)"})).fillna("Ingen kamp")

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
squad.loc[squad["is_vice_captain"], "role"] = "Visekaptein"
squad.loc[squad["is_captain"], "role"] = "Kaptein"
squad.loc[~squad["is_starter"], "role"] = "Benk"

starters = squad[squad["is_starter"]]
# Kapteinen får doble poeng.
lineup_points = starters["predicted_points"].sum() + squad.loc[squad["is_captain"], "predicted_points"].sum()
best_lineup = pick_best_team(squad)
# Kapteinen må stå i startelleveren, så anbefalingen velges blant dem som starter.
captain, vice = pick_captains(starters)
squad["recommended"] = squad["element"].map({captain["element"]: "Kaptein", vice["element"]: "Visekaptein"}).fillna("")
best_points = best_lineup["predicted_points"].sum() + best_lineup["predicted_points"].max()

st.subheader(team_name)
st.caption(f"Troppen slik den var satt opp i runde {picked_gameweek}. Senere bytter og endringer i laget vises ikke.")

with st.container(horizontal=True):
    st.metric(f"Forventede poeng, runde {next_gameweek}", f"{lineup_points:.1f}", border=True)
    st.metric(
        "Beste oppsett fra troppen",
        f"{best_points:.1f}",
        border=True,
        help="De beste elleve av dine femten i en gyldig formasjon, med den beste spilleren som kaptein.",
    )
    st.metric("På benken", f"{squad.loc[~squad['is_starter'], 'predicted_points'].sum():.1f}", border=True)

captain_card(captain, vice)
current = squad[squad["is_captain"]]
if not current.empty and current["element"].iat[0] != captain["element"]:
    difference = captain["predicted_points"] - current["predicted_points"].iat[0]
    st.caption(
        f"Du har {current['name'].iat[0]} som kaptein nå. "
        f"Med {captain['name']} er forventningen {difference:.1f} poeng høyere."
    )
elif not current.empty:
    st.caption("Det er samme kaptein som du har nå.")

st.dataframe(
    squad[["name", "team", "position", "fixture", "role", "recommended", "predicted_points"]].replace(
        {"position": POSITION_NAMES}
    ),
    column_config={
        "name": "Spiller",
        "team": "Lag",
        "position": "Posisjon",
        "fixture": "Kamp",
        "role": "Rolle nå",
        "recommended": "Anbefalt",
        "predicted_points": st.column_config.ProgressColumn(
            "Forventede poeng",
            format="%.2f",
            min_value=0,
            max_value=float(squad["predicted_points"].max()),
        ),
    },
    hide_index=True,
    height="content",
    alt="Troppen din med forventede poeng for neste runde",
)
