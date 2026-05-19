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
# logit_base_xg is both the base_margin (T1 spatial prior) and a learnable feature.
# max_depth=2 limits each path to at most 2 features — the structural protection against
# complex fingerprint paths. No interaction_constraints are set (they blocked binary flags
# from competing with continuous features; at depth=2 they add no additional safety).
# prior_event_same and prior_event_opp are passed as pd.Categorical.
# position and strength_state are also pd.Categorical (handled by _apply_fixed_categoricals).
CONTEXT_XG_FEATURE_COLUMNS: list[str] = [
    "logit_base_xg",  # T1 spatial prior — present in every interaction group
    # Binary event flags — each paired individually with logit_base_xg (groups 0-3)
    "is_rebound",  # interaction group 0
    "is_scramble",  # interaction group 1
    "rush_attempt",  # interaction group 2
    "prior_face",  # interaction group 3
    # Game-state modifiers — each paired individually with logit_base_xg (groups 4-7)
    "is_home",  # interaction group 4: home-ice tactical proxy
    "position",  # interaction group 5: shooter personnel (F vs D)
    "strength_state",  # interaction group 6: ice openness (5v5, 3v3, etc.)
    "score_diff",  # interaction group 7: score effects / era-drift isolated here
    # Continuous sequence block — all interact together with logit_base_xg (group 8)
    "play_speed",
    "seconds_since_last",
    "distance_from_last",
    "prior_event_angle",
    "prior_event_distance",
    "seconds_since_stoppage",
    "seconds_since_event_team_change",  # interaction group 8: time since shooting team's last line change
    "seconds_since_opp_team_change",  # interaction group 8: time since opposing team's last line change
    "period",  # ice degradation / desperation proxy
    "period_seconds",  # end-of-period timing
    "prior_event_same",  # categorical
    "prior_event_opp",  # categorical
]

# Interaction constraint groups — retained for reference only; no longer passed to XGBoost.
# The original per-flag isolation design prevented binary flags (is_rebound, is_scramble,
# strength_state) from competing against the continuous block for tree allocation: all
# 100 boosting rounds went to group 8, leaving flag groups with zero importance despite
# real signal (rebound goal rate 15% vs base_xg 11.3%). Removing constraints at depth=2
# is safe because each path can hold at most 2 features regardless.
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
        "play_speed",
        "seconds_since_last",
        "distance_from_last",
        "prior_event_angle",
        "prior_event_distance",
        "seconds_since_stoppage",
        "seconds_since_event_team_change",
        "seconds_since_opp_team_change",
        "period",
        "period_seconds",
        "prior_event_same",
        "prior_event_opp",
    ],
]

# XGBoost architecture — shared between tuning (experiments.py) and finalize scripts
N_ESTIMATORS: int = 500
EARLY_STOPPING_ROUNDS: int = 50

# context_xg-specific budget constraints to prevent GOAL-side fingerprinting.
# Capping n_estimators at 100 with early_stopping_rounds=20 limits best_iter to ≤80.
# Combined with lr ceiling 0.10 in experiments.py: max accumulated log-odds ≈ 3.1
# (lr × best_iter × effective_fraction ≈ 0.10 × 80 × 0.385), which prevents low-geometry
# shots (base_xg ≈ 5%) from being pushed above ~58% by contextual features alone.
N_ESTIMATORS_CONTEXT_XG: int = 100
EARLY_STOPPING_ROUNDS_CONTEXT_XG: int = 20

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

# Per-strength composite-score thresholds for performance tagging (controls artifact upload).
# Composite = PR-AUC − 0.5 × log_loss, matching the Pareto ranking in select_top_trials().
# Thresholds are anchored ~0.04–0.20 above each strength's null composite
# (null_composite ≈ base_rate − 0.5 × H(base_rate)) so near-random models always tag "none".
# Goal rates vary widely (EV ~7%, PP ~15%, EA ~80%+), so per-strength calibration is required.
# These are initial estimates — adjust after the first full v1.0.1 tuning run if tiers shift.
PERF_THRESHOLDS: dict[str, dict[str, float]] = {
    "even_strength": {"very_high": 0.12, "high": 0.06, "medium": 0.00, "cutoff": -0.04},
    "powerplay": {"very_high": 0.14, "high": 0.08, "medium": 0.02, "cutoff": -0.02},
    "shorthanded": {"very_high": 0.13, "high": 0.07, "medium": 0.01, "cutoff": -0.03},
    "empty_for": {"very_high": 0.17, "high": 0.11, "medium": 0.05, "cutoff": 0.01},
    "empty_against": {"very_high": 0.75, "high": 0.69, "medium": 0.63, "cutoff": 0.59},
}


def compute_performance_tag(prauc: float, log_loss_val: float, strength: str) -> str:
    """Return a performance tier tag based on per-strength composite-score thresholds.

    Composite = PR-AUC − 0.5 × log_loss. Using both objectives ensures that models with
    high discrimination but poor calibration don't gate artifact uploads as "high" or better.
    """
    composite = prauc - 0.5 * log_loss_val
    t = PERF_THRESHOLDS.get(strength, PERF_THRESHOLDS["even_strength"])
    if composite < t["cutoff"]:
        return "none"
    if composite >= t["very_high"]:
        return "very high"
    if composite >= t["high"]:
        return "high"
    if composite >= t["medium"]:
        return "medium"
    return "low"
