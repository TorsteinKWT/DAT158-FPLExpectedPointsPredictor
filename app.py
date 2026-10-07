from pathlib import Path

import joblib
import streamlit as st

from src.data import POSITIONS, SEASONS, build_prediction_frame, load_season

MODEL_PATH = Path(__file__).parent / "model.joblib"

st.set_page_config(page_title="FPL expected points", page_icon=":material/sports_soccer:", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data(ttl="1h", max_entries=len(SEASONS))
def load_players(season: str):
    raw = load_season(season, use_cache=False)
    return build_prediction_frame(raw), int(raw["GW"].max())


st.title("FPL expected points")

if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size == 0:
    st.error("No trained model found. Run `python -m src.train` first.", icon=":material/error:")
    st.stop()

bundle = load_model()
metrics = bundle["metrics"]

with st.sidebar:
    season = st.selectbox("Season", SEASONS, index=len(SEASONS) - 1)
    venue = st.segmented_control("Next match", ["Home", "Away"], default="Home")
    positions = st.pills("Positions", POSITIONS, selection_mode="multi", default=POSITIONS)
    search = st.text_input("Search player or team")

players, latest_gw = load_players(season)
players["was_home"] = int(venue == "Home")
players["predicted_points"] = bundle["model"].predict(players[bundle["features"]])
players["price"] = players["value"] / 10

shown = players[players["position"].isin(positions)]
if search:
    matches = shown["name"].str.contains(search, case=False) | shown["team"].str.contains(search, case=False)
    shown = shown[matches]
shown = shown.sort_values("predicted_points", ascending=False)

st.caption(
    f"Predictions for each player's next match, based on form through gameweek {latest_gw} of {season}. "
    f"Test MAE {metrics['mae']:.2f} points (baseline {metrics['baseline_mae']:.2f})."
)

st.dataframe(
    shown[["name", "team", "position", "price", "total_points_roll", "minutes_roll", "predicted_points"]],
    column_config={
        "name": "Player",
        "team": "Team",
        "position": "Position",
        "price": st.column_config.NumberColumn("Price", format="£%.1fm"),
        "total_points_roll": st.column_config.NumberColumn("Avg points (last 5)", format="%.1f"),
        "minutes_roll": st.column_config.NumberColumn("Avg minutes (last 5)", format="%.0f"),
        "predicted_points": st.column_config.ProgressColumn(
            "Expected points",
            format="%.2f",
            min_value=0,
            max_value=float(players["predicted_points"].max()),
        ),
    },
    hide_index=True,
    height=600,
    alt="Players ranked by expected points in their next match",
)
