"""Data loading and feature engineering shared by training and the app."""

from pathlib import Path

import pandas as pd
import requests

DATA_URL = (
    "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/"
    "master/data/{season}/gws/merged_gw.csv"
)
FPL_API = "https://fantasy.premierleague.com/api"
SEASONS = ["2024-25", "2025-26", "2026-27"]
# The official API only serves the season in progress, so this must be the last entry.
CURRENT_SEASON = SEASONS[-1]
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


def _fetch_current_season() -> pd.DataFrame:
    """Fetch every finished gameweek of the season in progress from the official FPL API.

    The API reports stats per gameweek, so a double gameweek becomes one row with the
    stats summed and the venue of the first match. Price is the current one, not the
    price at the time of the match.
    """
    session = requests.Session()
    # The API rejects requests without a browser-like user agent.
    session.headers["User-Agent"] = "Mozilla/5.0"

    def get(path: str):
        response = session.get(f"{FPL_API}/{path}/", timeout=30)
        response.raise_for_status()
        return response.json()

    bootstrap = get("bootstrap-static")
    teams = {team["id"]: team["name"] for team in bootstrap["teams"]}
    positions = {pos["id"]: pos["singular_name_short"] for pos in bootstrap["element_types"]}
    players = {player["id"]: player for player in bootstrap["elements"]}
    fixtures = {fixture["id"]: fixture for fixture in get("fixtures")}

    rows = []
    for event in bootstrap["events"]:
        if not event["finished"]:
            continue
        for element in get(f"event/{event['id']}/live")["elements"]:
            player = players.get(element["id"])
            # No fixture means the player's team had a blank gameweek.
            if player is None or not element["explain"]:
                continue
            fixture = fixtures[element["explain"][0]["fixture"]]
            rows.append(
                {
                    "name": f"{player['first_name']} {player['second_name']}",
                    "team": teams[player["team"]],
                    "position": positions[player["element_type"]],
                    "element": element["id"],
                    "value": player["now_cost"],
                    "GW": event["id"],
                    "kickoff_time": fixture["kickoff_time"],
                    "was_home": fixture["team_h"] == player["team"],
                    **{stat: element["stats"][stat] for stat in ROLLING_STATS},
                }
            )

    df = pd.DataFrame(rows)
    df[ROLLING_STATS] = df[ROLLING_STATS].apply(pd.to_numeric)
    return df


def load_season(season: str, use_cache: bool = True) -> pd.DataFrame:
    """Load one season of per-match player data, one row per player per fixture."""
    cache_file = RAW_DIR / f"{season}.csv"
    if season == CURRENT_SEASON:
        # Never cached on disk, since new gameweeks keep arriving.
        df = _fetch_current_season()
    elif use_cache and cache_file.exists():
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
