"""Lagre prediksjonene for neste runde før den spilles. Kjøres fra prosjektroten: python -m src.snapshot"""

from pathlib import Path

import joblib
import pandas as pd

from src.data import CURRENT_SEASON, POSITIONS, load_fixtures, load_frames, load_season

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "model.joblib"
PREDICTIONS_DIR = ROOT / "predictions"
SELECTION_PATH = PREDICTIONS_DIR / "utvalg.csv"
SELECTED_PER_POSITION = 5


def snapshot_path(gameweek: int) -> Path:
    return PREDICTIONS_DIR / f"runde_{gameweek:02d}.csv"


def pick_selection(matches: pd.DataFrame) -> pd.DataFrame:
    """Velg spillerne med flest minutter så langt i sesongen, fem per posisjon."""
    totals = matches.groupby("element", as_index=False).agg(
        name=("name", "last"),
        team=("team", "last"),
        position=("position", "last"),
        minutes=("minutes", "sum"),
        total_points=("total_points", "sum"),
    )
    # Mange har spilt alle minuttene, så poeng og til slutt ID avgjør ved likt antall.
    totals = totals.sort_values(["minutes", "total_points", "element"], ascending=[False, False, True])
    selection = totals.groupby("position").head(SELECTED_PER_POSITION)
    order = selection["position"].map(POSITIONS.index)
    return selection.assign(order=order).sort_values(["order", "minutes"], ascending=[True, False]).drop(columns="order")


def save_snapshot() -> str:
    """Lagre hva hver modell forventer av hver spiller i neste runde.

    En runde lagres bare én gang, og aldri etter at første kamp har startet, slik at
    prediksjonene ikke kan endres når fasiten begynner å komme. Skriptet kan derfor
    kjøres daglig. Returnerer en melding om hva som ble gjort.
    """
    _, upcoming = load_frames(CURRENT_SEASON)
    upcoming = upcoming.dropna(subset=["opponent"])
    if upcoming.empty:
        return "Det er ingen kommende runde å predikere."
    gameweek = int(upcoming["GW"].iat[0])
    path = snapshot_path(gameweek)
    if path.exists():
        return f"Runde {gameweek} er allerede lagret i {path.name}. Slett filen for å lagre den på nytt."

    fixtures = load_fixtures(CURRENT_SEASON)
    kickoff = fixtures.loc[fixtures["GW"] == gameweek, "kickoff_time"].min()
    now = pd.Timestamp.now(tz="UTC")
    if now >= kickoff:
        return f"Runde {gameweek} startet {kickoff:%d.%m.%Y %H:%M} UTC, så det er for sent å lagre den."

    bundle = joblib.load(MODEL_PATH)

    predictions = upcoming[["element", "name", "team", "position"]].copy()
    predictions["fixture"] = upcoming["opponent"] + upcoming["was_home"].map({1: " (H)", 0: " (B)"})
    for name, model in bundle["models"].items():
        predictions[name] = model.predict(upcoming[bundle["features"]])
    # En dobbeltrunde har én rad per kamp, så de summeres til én rad per spiller.
    predictions = predictions.groupby("element", as_index=False).agg(
        {
            "name": "first",
            "team": "first",
            "position": "first",
            "fixture": ", ".join,
            **{name: "sum" for name in bundle["models"]},
        }
    )
    predictions.insert(0, "GW", gameweek)
    predictions["created_at"] = now.isoformat(timespec="seconds")

    PREDICTIONS_DIR.mkdir(exist_ok=True)
    # Utvalget bestemmes én gang, første gang det lagres en runde, og ligger fast etterpå.
    if not SELECTION_PATH.exists():
        pick_selection(load_season(CURRENT_SEASON)).to_csv(SELECTION_PATH, index=False)
    predictions.round(3).to_csv(path, index=False)
    return f"Lagret prediksjonene for runde {gameweek} i {path.name}."


def load_snapshots() -> pd.DataFrame:
    """Alle lagrede runder samlet, én rad per spiller per runde."""
    files = sorted(PREDICTIONS_DIR.glob("runde_*.csv"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(file) for file in files], ignore_index=True)


def load_selection() -> pd.DataFrame:
    return pd.read_csv(SELECTION_PATH) if SELECTION_PATH.exists() else pd.DataFrame()


if __name__ == "__main__":
    print(save_snapshot())
