"""Cachede innlastere som deles av sidene i appen."""

from pathlib import Path

import joblib
import streamlit as st

from src.data import SEASONS, load_frames
from src.fpl import fetch_entry

MODEL_PATH = Path(__file__).resolve().parent.parent / "model.joblib"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data(ttl="1h", max_entries=len(SEASONS))
def load_players(season: str):
    """Returner rader per kamp for spilte runder, og rader per kamp for neste runde."""
    return load_frames(season, use_cache=False)


@st.cache_data(ttl="5m", max_entries=100)
def load_entry(entry_id: int):
    return fetch_entry(entry_id)
