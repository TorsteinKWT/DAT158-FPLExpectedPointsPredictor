"""Modellene som trenes og sammenlignes."""

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data import TARGET

BASELINE = "Snitt siste kamper"


class RecentAverage(RegressorMixin, BaseEstimator):
    """Baseline uten læring: gjetter spillerens poengsnitt fra de siste kampene."""

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return X[f"{TARGET}_roll"].to_numpy()


# Lineær regresjon og random forest får manglende verdier fylt inn med medianen.
# Gradient boosting håndterer dem selv.
MODELS = {
    BASELINE: RecentAverage,
    "Lineær regresjon": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge()),
    "Random forest": lambda: make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=25, n_jobs=-1, random_state=42),
    ),
    "Gradient boosting": lambda: HistGradientBoostingRegressor(random_state=42),
}
