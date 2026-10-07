"""Datainnlasting og feature engineering, delt mellom trening og app."""

from pathlib import Path

import pandas as pd

from src.fpl import get

DATA_URL = (
    "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/"
    "master/data/{season}/gws/merged_gw.csv"
)
FIXTURES_URL = (
    "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/"
    "master/data/{season}/fixtures.csv"
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
# FDR er FPLs egen vanskelighetsgrad for kampen (1–5). De to andre er hvor mange FPL-poeng
# motstanderen har tatt og sluppet til per kamp i sine siste kamper.
OPPONENT_FEATURES = ["fdr", "opp_points_for_roll", "opp_points_against_roll"]
FEATURES = (
    [f"{stat}_roll" for stat in ROLLING_STATS]
    + ["value", "was_home"]
    + OPPONENT_FEATURES
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
                    "fixture": fixture["id"],
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


def load_fixtures(season: str, use_cache: bool = True) -> pd.DataFrame:
    """Last inn kampoppsettet for en sesong, én rad per kamp, også kamper som ikke er spilt."""
    cache_file = RAW_DIR / f"{season}_fixtures.csv"
    if season == CURRENT_SEASON:
        df = pd.DataFrame(get("fixtures"))
    elif use_cache and cache_file.exists():
        df = pd.read_csv(cache_file)
    else:
        df = pd.read_csv(FIXTURES_URL.format(season=season))
        if use_cache:
            RAW_DIR.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache_file, index=False)

    df = df.rename(columns={"id": "fixture", "event": "GW"})
    # Utsatte kamper som ikke har fått ny runde ennå, har ingen runde.
    df = df.dropna(subset=["GW"])
    df["kickoff_time"] = pd.to_datetime(df["kickoff_time"])
    return df[["fixture", "GW", "kickoff_time", "team_h", "team_a", "team_h_difficulty", "team_a_difficulty"]]


def build_sides(matches: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    """Én rad per lag per kamp, med FDR og motstanderens form før kampen.

    Kamper som ikke er spilt er også med, slik at de kan brukes til å predikere neste runde.
    """
    home = fixtures.rename(columns={"team_h": "team_id", "team_a": "opponent_id", "team_h_difficulty": "fdr"})
    away = fixtures.rename(columns={"team_a": "team_id", "team_h": "opponent_id", "team_a_difficulty": "fdr"})
    columns = ["fixture", "GW", "kickoff_time", "team_id", "opponent_id", "fdr", "was_home"]
    sides = pd.concat([home.assign(was_home=1)[columns], away.assign(was_home=0)[columns]], ignore_index=True)

    # FPL-poengene laget tok i kampen, og poengene motstanderen tok i samme kamp.
    points = matches.groupby(["fixture", "was_home"])["total_points"].sum().rename("points_for")
    sides = sides.join(points, on=["fixture", "was_home"])
    other_side = sides.assign(was_home=1 - sides["was_home"])
    sides = sides.merge(
        other_side[["fixture", "was_home", "points_for"]].rename(columns={"points_for": "points_against"}),
        on=["fixture", "was_home"],
    )

    sides = sides.sort_values(["team_id", "kickoff_time"])
    grouped = sides.groupby("team_id")
    for column in ["points_for", "points_against"]:
        sides[f"{column}_roll"] = grouped[column].transform(
            lambda s: s.shift(1).rolling(WINDOW, min_periods=1).mean()
        )

    # Hent formen til laget på den andre siden av kampen.
    other_side = sides.assign(was_home=1 - sides["was_home"]).rename(
        columns={"points_for_roll": "opp_points_for_roll", "points_against_roll": "opp_points_against_roll"}
    )
    sides = sides.merge(
        other_side[["fixture", "was_home", "opp_points_for_roll", "opp_points_against_roll"]],
        on=["fixture", "was_home"],
    )

    # Kampoppsettet har bare lag-ID-er, så navnene hentes fra spillerne som spilte for laget.
    names = (
        matches.merge(sides[["fixture", "was_home", "team_id"]], on=["fixture", "was_home"])
        .groupby("team_id")["team"]
        .agg(lambda s: s.mode().iat[0])
    )
    sides["team"] = sides["team_id"].map(names)
    sides["opponent"] = sides["opponent_id"].map(names)
    return sides[["fixture", "GW", "was_home", "team", "opponent", *OPPONENT_FEATURES]]


def _add_position_dummies(df: pd.DataFrame) -> pd.DataFrame:
    for pos in POSITIONS:
        df[f"pos_{pos}"] = (df["position"] == pos).astype(int)
    return df


def build_training_frame(matches: pd.DataFrame, sides: pd.DataFrame) -> pd.DataFrame:
    """Legg til form-features som bare bygger på kampene før den som skal predikeres."""
    df = matches.copy()
    grouped = df.groupby(["season", "element"])
    for stat in ROLLING_STATS:
        df[f"{stat}_roll"] = grouped[stat].transform(
            lambda s: s.shift(1).rolling(WINDOW, min_periods=1).mean()
        )
    df = df.merge(sides.drop(columns=["GW", "team"]), on=["fixture", "was_home"], how="left")
    df = _add_position_dummies(df)
    # Den første kampen til en spiller i sesongen har ingen historikk å predikere fra.
    return df.dropna(subset=[f"{TARGET}_roll"])


def build_prediction_frame(matches: pd.DataFrame, sides: pd.DataFrame) -> pd.DataFrame:
    """Én rad per spiller per kamp i neste runde, med formen fra de siste kampene.

    Neste runde er den første som ikke finnes i kampdataene. Spillere uten kamp i den
    runden får én rad uten motstander, og en avsluttet sesong gir bare slike rader.
    """
    recent = matches.groupby("element").tail(WINDOW)
    form = recent.groupby("element")[ROLLING_STATS].mean().add_suffix("_roll")
    latest = matches.groupby("element").tail(1).set_index("element")
    players = latest[["name", "team", "position", "value"]].join(form).reset_index()

    upcoming = sides[sides["GW"] == matches["GW"].max() + 1]
    players = players.merge(upcoming, on="team", how="left")
    return _add_position_dummies(players)


def load_frames(season: str, use_cache: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returner treningsrader for spilte kamper og prediksjonsrader for neste runde."""
    matches = load_season(season, use_cache)
    sides = build_sides(matches, load_fixtures(season, use_cache))
    return build_training_frame(matches, sides), build_prediction_frame(matches, sides)
