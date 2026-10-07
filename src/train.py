"""Tren modellen for forventede poeng. Kjøres fra prosjektroten: python -m src.train"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from src.data import FEATURES, SEASONS, TARGET, build_training_frame, load_season

MODEL_PATH = Path(__file__).resolve().parent.parent / "model.joblib"
TEST_FRACTION = 0.2


def main() -> None:
    raw = pd.concat([load_season(season) for season in SEASONS], ignore_index=True)
    data = build_training_frame(raw).sort_values("kickoff_time")

    # Split basert på tidspunkt, slik at vi ikke får "lekkasje" fra fremtidige kamper inn i treningsdataene.
    split = int(len(data) * (1 - TEST_FRACTION))
    train, test = data.iloc[:split], data.iloc[split:]

    model = HistGradientBoostingRegressor(random_state=42)
    model.fit(train[FEATURES], train[TARGET])

    metrics = {
        "mae": mean_absolute_error(test[TARGET], model.predict(test[FEATURES])),
        # Baseline: predicte gjennomsnittet av de siste 3 kampene for hver spiller.
        "baseline_mae": mean_absolute_error(test[TARGET], test[f"{TARGET}_roll"]),
        "n_train": len(train),
        "n_test": len(test),
    }
    print(f"Train rows: {metrics['n_train']}, test rows: {metrics['n_test']}")
    print(f"Model MAE:    {metrics['mae']:.3f}")
    print(f"Baseline MAE: {metrics['baseline_mae']:.3f}")

    model.fit(data[FEATURES], data[TARGET])
    joblib.dump({"model": model, "features": FEATURES, "metrics": metrics}, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
