"""Shared training constants imported by experiments.py and finalize scripts.

No heavy dependencies — pure Python only. Add constants here when they appear
in more than one script so there is a single source of truth.
"""

SEED: int = 615
DPI: int = 100
FIGSIZE: tuple[int, int] = (6, 4)
SHAP_FIGSIZE: tuple[int, int] = (8, 6)  # wider than standard charts

STRENGTHS: list[str] = ["even_strength", "powerplay", "shorthanded", "empty_for", "empty_against"]
MODELS: list[str] = ["base_xg", "context_xg", "pred_goal"]

# Minimum player TOI (minutes) to be included in each RAPM regression.
# EV regular season is strictest — enough sample for stable coefficients.
# Playoff and non-EV situations use lower floors due to smaller samples.
RAPM_TOI_LIMITS: dict[str, int] = {
    "ev_r": 10,
    "ev_p": 5,
    "other_r": 5,
    "other_p": 1,
}

# Hold-out split boundary — the 2024-25 season is held out across all three tiers.
# Single source of truth: base_xg/context_xg/pred_goal process_data scripts all import this.
HOLD_OUT_SEASON: int = 20242025
PASSTHROUGH_COLS: list[str] = [
    "event_idx",
    "game_id",
    "player_1_api_id",
    "opp_goalie_api_id",
    "home_on_api_id",
    "away_on_api_id",
    "session",
]

# Tier 1 — pure spatial prior. Only puck physics and ice coordinates. Identical
# distributions for GOAL and SHOT events so XGBoost cannot fingerprint outcomes.
# Game-state features (score_diff, period, period_seconds, is_home, position,
# strength_state) live in Tier 2 where depth-2 + interaction constraints prevent
# them from combining into era-specific fingerprint paths.
# pred_goal/process_data.py drops this entire list so pred_goal sees only talent features.
BASE_XG_FEATURE_COLUMNS: list[str] = [
    # Pure geometry
    "event_distance",
    "event_angle",
    "coords_x",
    "coords_y",
    "abs_y_distance",
    # Derived danger zones
    "danger",
    "high_danger",
    # Physical puck action
    "shot_type",
]

# Tier 2 — sequence context + game-state modifiers for the gbtree middle tier.
# logit_base_xg carries the T1 spatial prior as a learnable feature and is paired
# individually with each state/flag so that depth-2 trees answer exactly one question:
# "given this spatial prior, how much does THIS game-state factor shift danger?"
# Combining two state features on one path is structurally impossible at depth=2.
# prior_event_same and prior_event_opp are passed as pd.Categorical.
# position and strength_state are also pd.Categorical (handled by _apply_fixed_categoricals).
CONTEXT_XG_FEATURE_COLUMNS: list[str] = [
    "logit_base_xg",       # T1 spatial prior — present in every interaction group
    # Binary event flags — each paired individually with logit_base_xg (groups 0-3)
    "is_rebound",          # interaction group 0
    "is_scramble",         # interaction group 1
    "rush_attempt",        # interaction group 2
    "prior_face",          # interaction group 3
    # Game-state modifiers — each paired individually with logit_base_xg (groups 4-7)
    "is_home",             # interaction group 4: home-ice tactical proxy
    "position",            # interaction group 5: shooter personnel (F vs D)
    "strength_state",      # interaction group 6: ice openness (5v5, 3v3, etc.)
    "score_diff",          # interaction group 7: score effects / era-drift isolated here
    # Continuous sequence block — all interact together with logit_base_xg (group 8)
    "play_speed",
    "seconds_since_last",
    "distance_from_last",
    "prior_event_angle",
    "prior_event_distance",
    "seconds_since_stoppage",
    "period",              # ice degradation / desperation proxy
    "period_seconds",      # end-of-period timing
    "prior_event_same",    # categorical
    "prior_event_opp",     # categorical
]

# Interaction constraint groups for context_xg gbtree (depth=2).
# Each group limits which features can share a single tree path.
# State/flag features are isolated in their own pair with logit_base_xg —
# this structurally prevents any two game-state features from combining on one path.
# The continuous block includes logit_base_xg so sequence features can ask
# "how does rush speed modify the spatial prior?" at depth 2.
CONTEXT_XG_INTERACTION_GROUPS: list[list[str]] = [
    ["logit_base_xg", "is_rebound"],
    ["logit_base_xg", "is_scramble"],
    ["logit_base_xg", "rush_attempt"],
    ["logit_base_xg", "prior_face"],
    ["logit_base_xg", "is_home"],
    ["logit_base_xg", "position"],
    ["logit_base_xg", "strength_state"],
    ["logit_base_xg", "score_diff"],
    [
        "logit_base_xg",
        "play_speed", "seconds_since_last", "distance_from_last",
        "prior_event_angle", "prior_event_distance", "seconds_since_stoppage",
        "period", "period_seconds",
        "prior_event_same", "prior_event_opp",
    ],
]

# XGBoost architecture — shared between tuning (experiments.py) and finalize scripts
N_ESTIMATORS: int = 500
EARLY_STOPPING_ROUNDS: int = 50

# TimeSeriesSplit folds: fewer during tuning (speed), more for OOF calibration (coverage)
CV_TUNE_FOLDS: int = 3
CV_CALIBRATE_FOLDS: int = 5

# Monotone constraints applied where columns are present in the feature matrix
MONOTONE_CONSTRAINTS: dict[str, int] = {
    "event_distance": -1,
    "event_angle": -1,
}

# Interaction constraints for base_xg. With geometry-only features, GOAL and SHOT events
# have identical feature distributions, so XGBoost cannot fingerprint outcomes — no
# constraints are needed. Kept as an empty list for script compatibility.
BASE_XG_INTERACTION_GROUPS: list[list[str]] = []

# context_xg discriminates the same events as base_xg, so PR-AUC scale is the same.
# These thresholds gate artifact uploads exactly as in base_xg.
CONTEXT_XG_PERF_THRESHOLDS: dict[str, dict[str, float]] = {
    "even_strength": {"very_high": 0.30, "high": 0.23, "medium": 0.17, "cutoff": 0.12},
    "powerplay":     {"very_high": 0.36, "high": 0.28, "medium": 0.21, "cutoff": 0.14},
    "shorthanded":   {"very_high": 0.28, "high": 0.22, "medium": 0.16, "cutoff": 0.11},
    "empty_for":     {"very_high": 0.43, "high": 0.35, "medium": 0.27, "cutoff": 0.19},
    "empty_against": {"very_high": 0.88, "high": 0.81, "medium": 0.73, "cutoff": 0.65},
}

# Per-strength PR-AUC thresholds for performance tagging.
# Goal rates vary widely across states (EV ~7%, PP ~15%, EA ~80%+), so flat thresholds would
# either over-upload artifacts on high-rate states or never fire on low-rate ones.
# Log-loss is logged for monitoring but excluded from the gate: label smoothing (eps=0.05)
# systematically inflates log-loss on binary targets, causing well-performing trials to
# be mis-tagged "none" despite good PR-AUC. PR-AUC alone matches Optuna's objective.
PERF_THRESHOLDS: dict[str, dict[str, float]] = {
    "even_strength": {"very_high": 0.25, "high": 0.18, "medium": 0.13, "cutoff": 0.08},
    "powerplay":     {"very_high": 0.28, "high": 0.21, "medium": 0.15, "cutoff": 0.10},
    "shorthanded":   {"very_high": 0.22, "high": 0.17, "medium": 0.12, "cutoff": 0.08},
    "empty_for":     {"very_high": 0.35, "high": 0.27, "medium": 0.20, "cutoff": 0.13},
    "empty_against": {"very_high": 0.75, "high": 0.68, "medium": 0.60, "cutoff": 0.52},
}


def compute_performance_tag(prauc: float, log_loss_val: float, strength: str) -> str:
    """Return a performance tier tag based on per-strength PR-AUC thresholds.

    log_loss_val is accepted for call-site compatibility but not used for gating —
    label smoothing inflates log-loss on binary targets, making it an unreliable gate.
    """
    t = PERF_THRESHOLDS.get(strength, PERF_THRESHOLDS["even_strength"])
    if prauc < t["cutoff"]:
        return "none"
    if prauc >= t["very_high"]:
        return "very high"
    if prauc >= t["high"]:
        return "high"
    if prauc >= t["medium"]:
        return "medium"
    return "low"