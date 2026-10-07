"""Client for the official FPL API."""

import pandas as pd
import requests

API_URL = "https://fantasy.premierleague.com/api"
STARTERS = 11

_session = requests.Session()
# API-et avviser forespørsler uten en nettleser-lignende user agent.
_session.headers["User-Agent"] = "Mozilla/5.0"


def get(path: str):
    response = _session.get(f"{API_URL}/{path}/", timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_next_fixtures() -> tuple[int, pd.DataFrame]:
    """Return the next gameweek and one row per team per match in it.

    A team with a double gameweek gets two rows, and a team with a blank gameweek gets none.
    """
    bootstrap = get("bootstrap-static")
    teams = {team["id"]: team["name"] for team in bootstrap["teams"]}
    gameweek = next(event["id"] for event in bootstrap["events"] if event["is_next"])

    rows = []
    for fixture in get("fixtures"):
        if fixture["event"] != gameweek:
            continue
        home, away = teams[fixture["team_h"]], teams[fixture["team_a"]]
        rows.append({"team": home, "opponent": away, "was_home": 1})
        rows.append({"team": away, "opponent": home, "was_home": 0})
    return gameweek, pd.DataFrame(rows)


def fetch_entry(entry_id: int) -> tuple[str, int, pd.DataFrame]:
    """Return a manager's team name, and their squad as picked in their latest gameweek.

    The API only shows picks for gameweeks that have started, so transfers and lineup
    changes made for the next gameweek are not included.
    """
    entry = get(f"entry/{entry_id}")
    gameweek = entry["current_event"]
    picks = get(f"entry/{entry_id}/event/{gameweek}/picks")
    # Et free hit-lag gjelder bare én runde, så laget fra runden før er det som kommer tilbake.
    if picks["active_chip"] == "freehit" and gameweek > entry["started_event"]:
        gameweek -= 1
        picks = get(f"entry/{entry_id}/event/{gameweek}/picks")
    picks = pd.DataFrame(picks["picks"])
    picks["is_starter"] = picks["position"] <= STARTERS
    return entry["name"], gameweek, picks[["element", "is_starter", "is_captain", "is_vice_captain"]]
