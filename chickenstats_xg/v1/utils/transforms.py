"""Feature transform utilities shared across all three model tiers."""

import numpy as np
import pandas as pd
from chickenstats.chicken_nhl._game_utils import (
    POSITIONS,
    PRIOR_EVENT_TYPES,
    SHOT_TYPES,
    STRENGTH_STATE_CATS,
)

_BM_EPS = 1e-7


def apply_fixed_categoricals(X: pd.DataFrame, strength: str) -> pd.DataFrame:
    """Cast categorical columns to pd.Categorical with fixed category lists."""
    X = X.copy()
    cat_map = {
        "shot_type": SHOT_TYPES,
        "position": POSITIONS,
        "strength_state": STRENGTH_STATE_CATS[strength],
        "prior_event_same": PRIOR_EVENT_TYPES,
        "prior_event_opp": PRIOR_EVENT_TYPES,
    }
    for col, cats in cat_map.items():
        if col in X.columns:
            X[col] = pd.Categorical(X[col], categories=cats)
    return X


def logit(p: np.ndarray) -> np.ndarray:
    """Convert probability to log-odds, clipped to avoid ±inf."""
    p = np.clip(p, _BM_EPS, 1 - _BM_EPS)
    return np.log(p / (1 - p))
