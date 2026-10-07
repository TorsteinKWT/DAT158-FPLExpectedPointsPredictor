import streamlit as st

from src.data import SEASONS
from src.loaders import MODEL_PATH, load_model, load_players

NEXT_MATCH = "Next match"

st.set_page_config(page_title="FPL expected points", page_icon=":material/sports_soccer:", layout="wide")

page = st.navigation(
    [
        st.Page("app_pages/players.py", title="Players", icon=":material/groups:"),
        st.Page("app_pages/best_team.py", title="Best team", icon=":material/trophy:"),
        st.Page("app_pages/my_team.py", title="My team", icon=":material/person:"),
    ],
    position="top",
)

# Streamlit sletter verdien til et felt som ikke vises, så lag-ID-en må holdes i live når man bytter side.
if "entry_id" in st.session_state:
    st.session_state.entry_id = st.session_state.entry_id

st.title("FPL expected points")

if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size == 0:
    st.error("No trained model found. Run `python -m src.train` first.", icon=":material/error:")
    st.stop()

bundle = load_model()
metrics = bundle["metrics"]

with st.sidebar:
    season = st.selectbox("Season", SEASONS, index=len(SEASONS) - 1)
    played, upcoming, latest_gw = load_players(season)
    # Første gameweek mangler i dataene, så vi må hente den fra de faktiske kampene.
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

# Delt med st.session_state for å kunne bruke dem i andre sider.
st.session_state.players = players
st.session_state.is_played = gameweek != NEXT_MATCH

st.caption(f"{caption} Test MAE {metrics['mae']:.2f} points (baseline {metrics['baseline_mae']:.2f}).")

page.run()
