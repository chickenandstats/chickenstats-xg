"""IsotonicCalibrator — sklearn-compatible isotonic regression calibrator."""

import numpy as np
from sklearn.isotonic import IsotonicRegression


class IsotonicCalibrator:
    """Isotonic regression calibrator with a predict_proba interface.

    Wraps sklearn IsotonicRegression as a drop-in replacement for LogisticRegression
    (Platt) calibrators: same joblib serialization, same predict_proba(X)[:, 1] call
    pattern. Use for states where Platt's sigmoid ceiling can't reach near-1.0 goal
    rates (e.g. empty_against, ~99% actual rate in the top prediction decile).
    """

    def __init__(self) -> None:
        self._iso = IsotonicRegression(out_of_bounds="clip")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        self._iso.fit(X.ravel(), y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = self._iso.predict(X.ravel())
        return np.stack([1 - p, p], axis=1)
