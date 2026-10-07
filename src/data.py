"""Datainnlasting og feature engineering, delt mellom trening og app."""

from pathlib import Path

import pandas as pd

from src.fpl import get

DATA_URL = (
    "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/"
    "master/data/{season}/gws/merged_gw.csv"
)
SEASONS = ["2024-25", "2025-26", "2026-27"]
# Det offisielle API-et gir bare sesongen som pågår, så denne må være den siste i listen.
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
            # Ingen kamp betyr at laget til spilleren hadde en blank runde.
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
    cache_file = RAW_DIR / f"{season}.csv"
    if season == CURRENT_SEASON:
        # Lagres aldri på disk, siden det stadig kommer nye runder.
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
    """Legg til form-features som bare bygger på kampene før den som skal predikeres."""
    df = df.copy()
    grouped = df.groupby(["season", "element"])
    for stat in ROLLING_STATS:
        df[f"{stat}_roll"] = grouped[stat].transform(
            lambda s: s.shift(1).rolling(WINDOW, min_periods=1).mean()
        )
    df = _add_position_dummies(df)
    # Den første kampen til en spiller i sesongen har ingen historikk å predikere fra.
    return df.dropna(subset=[f"{TARGET}_roll"])


def build_prediction_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Én rad per spiller med formen fra de siste kampene, for neste kamp."""
    recent = df.groupby("element").tail(WINDOW)
    form = recent.groupby("element")[ROLLING_STATS].mean().add_suffix("_roll")
    latest = df.groupby("element").tail(1).set_index("element")
    players = latest[["name", "team", "position", "value"]].join(form)
    return _add_position_dummies(players).reset_index()
