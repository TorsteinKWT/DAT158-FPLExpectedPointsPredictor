"""Cached loaders shared by the app pages."""

from pathlib import Path

import joblib
import streamlit as st

from src.data import SEASONS, build_prediction_frame, build_training_frame, load_season
from src.fpl import fetch_entry, fetch_next_fixtures

MODEL_PATH = Path(__file__).resolve().parent.parent / "model.joblib"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data(ttl="1h", max_entries=len(SEASONS))
def load_players(season: str):
    """Return per-match rows for played gameweeks, and one row per player for the next match."""
    raw = load_season(season, use_cache=False)
    return build_training_frame(raw), build_prediction_frame(raw), int(raw["GW"].max())


@st.cache_data(ttl="1h")
def load_next_fixtures():
    return fetch_next_fixtures()


@st.cache_data(ttl="5m", max_entries=100)
def load_entry(entry_id: int):
    return fetch_entry(entry_id)
