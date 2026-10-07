"""Data loading and feature engineering shared by training and the app."""

from pathlib import Path

import pandas as pd

DATA_URL = (
    "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/"
    "master/data/{season}/gws/merged_gw.csv"
)
SEASONS = ["2024-25", "2025-26", "2026-27"]
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

WINDOW = 5
POSITIONS = ["GK", "DEF", "MID", "FWD"]
ROLLING_STATS = [
    "total_points",
    "minutes",
    "starts",
    "bps",
    "ict_index",
    "expected_goals",
    "expected_assists",
    "expected_goals_conceded",
]
TARGET = "total_points"
FEATURES = (
    [f"{stat}_roll" for stat in ROLLING_STATS]
    + ["value", "was_home"]
    + [f"pos_{pos}" for pos in POSITIONS]
)


def load_season(season: str, use_cache: bool = True) -> pd.DataFrame:
    """Load one season of per-match player data, one row per player per fixture."""
    cache_file = RAW_DIR / f"{season}.csv"
    if use_cache and cache_file.exists():
        df = pd.read_csv(cache_file)
    else:
        df = pd.read_csv(DATA_URL.format(season=season), on_bad_lines="skip")
        if use_cache:
            RAW_DIR.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache_file, index=False)

    df["season"] = season
    df["position"] = df["position"].replace({"GKP": "GK"})
    df = df[df["position"].isin(POSITIONS)].copy()
    df["kickoff_time"] = pd.to_datetime(df["kickoff_time"])
    df["was_home"] = df["was_home"].astype(int)
    return df.sort_values(["element", "kickoff_time"]).reset_index(drop=True)


def _add_position_dummies(df: pd.DataFrame) -> pd.DataFrame:
    for pos in POSITIONS:
        df[f"pos_{pos}"] = (df["position"] == pos).astype(int)
    return df


def build_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Add form features computed only from matches before the one being predicted."""
    df = df.copy()
    grouped = df.groupby(["season", "element"])
    for stat in ROLLING_STATS:
        df[f"{stat}_roll"] = grouped[stat].transform(
            lambda s: s.shift(1).rolling(WINDOW, min_periods=1).mean()
        )
    df = _add_position_dummies(df)
    # A player's first match of the season has no history to predict from.
    return df.dropna(subset=[f"{TARGET}_roll"])


def build_prediction_frame(df: pd.DataFrame) -> pd.DataFrame:
    """One row per player with form over their most recent matches, for the next match."""
    recent = df.groupby("element").tail(WINDOW)
    form = recent.groupby("element")[ROLLING_STATS].mean().add_suffix("_roll")
    latest = df.groupby("element").tail(1).set_index("element")
    players = latest[["name", "team", "position", "value"]].join(form)
    return _add_position_dummies(players).reset_index()
