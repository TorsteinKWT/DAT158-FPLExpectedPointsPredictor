from pathlib import Path

import joblib
import streamlit as st

from src.data import SEASONS, build_prediction_frame, build_training_frame, load_season

MODEL_PATH = Path(__file__).parent / "model.joblib"
NEXT_MATCH = "Next match"

st.set_page_config(page_title="FPL expected points", page_icon=":material/sports_soccer:", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data(ttl="1h", max_entries=len(SEASONS))
def load_players(season: str):
    """Return per-match rows for played gameweeks, and one row per player for the next match."""
    raw = load_season(season, use_cache=False)
    return build_training_frame(raw), build_prediction_frame(raw), int(raw["GW"].max())


page = st.navigation(
    [
        st.Page("app_pages/players.py", title="Players", icon=":material/groups:"),
        st.Page("app_pages/best_team.py", title="Best team", icon=":material/trophy:"),
    ],
    position="top",
)

st.title("FPL expected points")

if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size == 0:
    st.error("No trained model found. Run `python -m src.train` first.", icon=":material/error:")
    st.stop()

bundle = load_model()
metrics = bundle["metrics"]

with st.sidebar:
    season = st.selectbox("Season", SEASONS, index=len(SEASONS) - 1)
    played, upcoming, latest_gw = load_players(season)
    # The first gameweek is missing here because there is no earlier form to predict from.
    gameweeks = sorted((int(gw) for gw in played["GW"].unique()), reverse=True)
    gameweek = st.selectbox(
        "Gameweek",
        [NEXT_MATCH, *gameweeks],
        format_func=lambda gw: gw if gw == NEXT_MATCH else f"Gameweek {gw}",
    )
    if gameweek == NEXT_MATCH:
        venue = st.segmented_control("Venue", ["Home", "Away"], default="Home")

if gameweek == NEXT_MATCH:
    players = upcoming.copy()
    players["was_home"] = int(venue == "Home")
    caption = f"Predictions for each player's next match, based on form through gameweek {latest_gw} of {season}."
else:
    players = played[played["GW"] == gameweek].copy()
    players["venue"] = players["was_home"].map({1: "Home", 0: "Away"})
    caption = (
        f"Predictions for gameweek {gameweek} of {season}, based on form before that gameweek. "
        "The model was trained on these matches, so it fits them better than it will fit new ones."
    )

players["predicted_points"] = bundle["model"].predict(players[bundle["features"]])
players["price"] = players["value"] / 10

# Shared with the pages, which only differ in how they present the same predictions.
st.session_state.players = players
st.session_state.is_played = gameweek != NEXT_MATCH

st.caption(f"{caption} Test MAE {metrics['mae']:.2f} points (baseline {metrics['baseline_mae']:.2f}).")

page.run()
