"""Tren modellene for forventede poeng. Kjøres fra prosjektroten: python -m src.train"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error

from src.data import CURRENT_SEASON, FEATURES, SEASONS, TARGET, load_frames
from src.models import MODELS

MODEL_PATH = Path(__file__).resolve().parent.parent / "model.joblib"
TEST_FRACTION = 0.2


def evaluate(model, test: pd.DataFrame) -> dict:
    """Snittfeil for alle spillere, og for dem som faktisk spilte."""
    predicted = model.predict(test[FEATURES])
    played = test["minutes"] > 0
    return {
        "mae": mean_absolute_error(test[TARGET], predicted),
        "mae_played": mean_absolute_error(test[TARGET][played], predicted[played]),
    }


def main() -> None:
    data = pd.concat([load_frames(season)[0] for season in SEASONS], ignore_index=True)
    data = data.sort_values("kickoff_time")

    # Split basert på tidspunkt, slik at vi ikke får "lekkasje" fra fremtidige kamper inn i treningsdataene.
    split = int(len(data) * (1 - TEST_FRACTION))
    train, test = data.iloc[:split], data.iloc[split:]
    print(f"Treningsrader: {len(train)}, testrader: {len(test)}")

    metrics = {}
    for name, make_model in MODELS.items():
        metrics[name] = evaluate(make_model().fit(train[FEATURES], train[TARGET]), test)
        print(f"{name:<20} MAE {metrics[name]['mae']:.3f}   spilte {metrics[name]['mae_played']:.3f}")

    # Hver runde i år predikeres av modeller som bare er trent på kampene før runden.
    backtest = []
    current = data[data["season"] == CURRENT_SEASON]
    for gameweek, matches in current.groupby("GW"):
        earlier = data[data["kickoff_time"] < matches["kickoff_time"].min()]
        for name, make_model in MODELS.items():
            model = make_model().fit(earlier[FEATURES], earlier[TARGET])
            backtest.append({"GW": int(gameweek), "model": name, **evaluate(model, matches)})

    models = {name: make_model().fit(data[FEATURES], data[TARGET]) for name, make_model in MODELS.items()}
    bundle = {
        "models": models,
        "features": FEATURES,
        "metrics": pd.DataFrame(metrics).T,
        "backtest": pd.DataFrame(backtest),
        "n_train": len(train),
        "n_test": len(test),
    }
    joblib.dump(bundle, MODEL_PATH, compress=3)
    print(f"Lagret modellene i {MODEL_PATH}")


if __name__ == "__main__":
    main()
