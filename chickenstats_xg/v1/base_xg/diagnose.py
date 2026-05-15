
"""base_xg overfitting / calibration diagnostic.

Run from repo root:
    uv run diagnose-base-xg
    uv run diagnose-base-xg --strength even_strength

Checks (per strength state):
  1. GOAL vs SHOT prediction distribution — GOAL p90 / SHOT p90 ratio, target ≤ 5×
  2. High-confidence event rate           — % of events with base_xg > HIGH_CONF threshold
  3. Precision/recall balance             — fingerprinting risk at base-rate threshold
  4. Calibration by decile               — mean(pred) vs actual goal rate per decile
  5. OOF (training) vs hold-out PR-AUC   — gap indicates generalisation failure
  6. PR-AUC by season                    — spots temporal overfitting or degradation
  7. Feature gain dominance              — flags if a single non-geometry feature
                                           monopolises model gain (overfitting cause)
"""

import argparse
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path as _Path

from chickenstats_xg.v1.config import STRENGTHS
from chickenstats_xg.v1.utils.diagnose_utils import (
    PASS, WARN, FAIL,
    status_icon, pct,
    check_calibration,
    check_precision_recall_balance,
    check_oof_vs_holdout,
    compute_holdout_metrics,
    print_holdout_metrics,
    extract_model_hyperparams,
)

# ── paths ─────────────────────────────────────────────────────────────────────
_BASE = _Path(__file__).parent.parent
DATA_DIR   = _BASE / "data"   / "base_xg"
MODELS_DIR = _BASE / "models" / "base_xg"

# ── thresholds ─────────────────────────────────────────────────────────────────
P90_RATIO_WARN = 3.5
P90_RATIO_FAIL = 5.0
HIGH_CONF = 0.80
HIGH_CONF_GOAL_WARN = 10.0
HIGH_CONF_GOAL_FAIL = 20.0
HIGH_CONF_SHOT_WARN = 1.0

GEOMETRY_FEATURES: frozenset[str] = frozenset([
    "event_distance", "event_angle", "coords_x", "coords_y",
    "abs_y_distance", "danger", "high_danger",
])
GAIN_TOP_N = 10
GAIN_DOMINANT_WARN = 0.40
GAIN_DOMINANT_FAIL = 0.60

_PR_FIX = "check max_depth and min_child_weight; raise lambda floor"


# ── Check 1: GOAL vs SHOT distribution ────────────────────────────────────────

def check_distribution(df: pd.DataFrame, strength: str) -> tuple[str, float]:
    """GOAL vs SHOT percentile table + p90 ratio verdict."""
    shots = df.loc[df.goal == 0, "base_xg"]
    goals = df.loc[df.goal == 1, "base_xg"]

    shot_p90 = shots.quantile(0.90)
    goal_p90 = goals.quantile(0.90)
    ratio = goal_p90 / shot_p90 if shot_p90 > 1e-9 else float("inf")

    if ratio <= P90_RATIO_WARN:
        status = PASS
    elif ratio <= P90_RATIO_FAIL:
        status = WARN
    else:
        status = FAIL

    print(f"\n{'─' * 62}")
    print(f"  {strength}")
    print(f"{'─' * 62}")
    print(f"  {'':6s}  {'n':>9}  {'mean':>6}  {'p50':>6}  {'p75':>6}  {'p90':>6}  {'p95':>6}  {'max':>6}")
    for label, sub in [("SHOT", shots), ("GOAL", goals)]:
        print(
            f"  {label:<6s}  {len(sub):>9,}  {sub.mean():>6.3f}  "
            f"{sub.quantile(0.50):>6.3f}  {sub.quantile(0.75):>6.3f}  "
            f"{sub.quantile(0.90):>6.3f}  {sub.quantile(0.95):>6.3f}  "
            f"{sub.max():>6.3f}"
        )
    print(f"\n  {status_icon(status)} Distribution — GOAL p90 / SHOT p90 = {ratio:.2f}×  [{status}]")
    return status, ratio


# ── Check 2: high-confidence event rate ───────────────────────────────────────

def check_high_confidence(df: pd.DataFrame) -> str:
    """% of events above HIGH_CONF threshold, split by GOAL / SHOT."""
    n_goals = (df.goal == 1).sum()
    n_shots = (df.goal == 0).sum()

    high_goals = ((df.goal == 1) & (df.base_xg > HIGH_CONF)).sum()
    high_shots = ((df.goal == 0) & (df.base_xg > HIGH_CONF)).sum()

    pct_g = pct(high_goals, n_goals)
    pct_s = pct(high_shots, n_shots)

    goal_status = PASS if pct_g < HIGH_CONF_GOAL_WARN else (WARN if pct_g < HIGH_CONF_GOAL_FAIL else FAIL)
    shot_status = PASS if pct_s < HIGH_CONF_SHOT_WARN else WARN
    overall = FAIL if FAIL in (goal_status, shot_status) else (WARN if WARN in (goal_status, shot_status) else PASS)

    print(f"\n  High-confidence events (base_xg > {HIGH_CONF})")
    print(f"    GOAL: {high_goals:>6,} / {n_goals:,}  ({pct_g:5.1f}%)  {status_icon(goal_status)} {goal_status}")
    print(f"    SHOT: {high_shots:>6,} / {n_shots:,}  ({pct_s:5.2f}%)  {status_icon(shot_status)} {shot_status}")
    return overall


# ── Check 5: PR-AUC by season ─────────────────────────────────────────────────

def check_season_prauc(scored_df: pd.DataFrame) -> None:
    """PR-AUC broken out by season. No pass/fail — informational only."""
    from sklearn.metrics import average_precision_score
    print(f"\n  PR-AUC by season")
    print(f"  {'Season':>8}  {'n_shots':>9}  {'n_goals':>9}  {'goal%':>7}  {'PR-AUC':>8}")
    for season in sorted(scored_df.season.unique()):
        sub = scored_df[scored_df.season == season]
        n_goals = int((sub.goal == 1).sum())
        n_shots = int((sub.goal == 0).sum())
        if n_goals < 10:
            continue
        prauc = average_precision_score(sub.goal, sub.base_xg)
        goal_rate = 100 * n_goals / len(sub)
        print(f"  {season:>8}  {n_shots:>9,}  {n_goals:>9,}  {goal_rate:>6.1f}%  {prauc:>8.4f}")


# ── Check 6: feature gain dominance ───────────────────────────────────────────

def check_feature_dominance(strength: str) -> str:
    """Gain distribution across features — flags contextual memorisation."""
    model_path = MODELS_DIR / strength / "model.ubj"
    if not model_path.exists():
        print(f"\n  Feature gain: model not found at {model_path} — skip")
        return WARN

    booster = xgb.Booster()
    booster.load_model(str(model_path))
    importance = booster.get_score(importance_type="gain")
    if not importance:
        print("\n  Feature gain: no scores returned (model may have no splits) — skip")
        return WARN

    total_gain = sum(importance.values())
    ranked = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    non_geo_ranked = [(f, g) for f, g in ranked if f not in GEOMETRY_FEATURES]
    top_non_geo_feat, top_non_geo_gain = non_geo_ranked[0] if non_geo_ranked else (None, 0.0)
    top_non_geo_pct = top_non_geo_gain / total_gain if total_gain > 0 else 0.0

    if top_non_geo_feat and top_non_geo_pct >= GAIN_DOMINANT_FAIL:
        status = FAIL
    elif top_non_geo_feat and top_non_geo_pct >= GAIN_DOMINANT_WARN:
        status = WARN
    else:
        status = PASS

    n_show = min(GAIN_TOP_N, len(ranked))
    print(f"\n  Feature gain dominance (top {n_show} of {len(ranked)})")
    print(f"  {'Feature':<26}  {'Gain%':>6}  {'':>10}")
    for feat, gain in ranked[:n_show]:
        tag = "(geometry)" if feat in GEOMETRY_FEATURES else ""
        marker = " ←" if feat == top_non_geo_feat and status != PASS else ""
        print(f"  {feat:<26}  {gain / total_gain:>5.1%}  {tag:<10}{marker}")

    if top_non_geo_feat:
        print(
            f"\n  {status_icon(status)} Feature dominance — top non-geometry: "
            f"{top_non_geo_feat} ({top_non_geo_pct:.1%})  [{status}]"
        )
    else:
        print(f"\n  {status_icon(status)} Feature dominance — geometry features dominate  [{status}]")
    return status


# ── Check 7: hyperparameter assessment ────────────────────────────────────────

def check_hyperparameters(strength: str) -> None:
    """Extract hyperparameters from the frozen base_xg booster and opine on their appropriateness."""
    import json as _json
    model_path = MODELS_DIR / strength / "model.ubj"
    meta_path  = MODELS_DIR / strength / "meta.json"

    params = extract_model_hyperparams(model_path)
    if params is None:
        print(f"\n  Hyperparameters: model not found at {model_path} — skip")
        return

    trial_num = trial_val = None
    if meta_path.exists():
        meta = _json.loads(meta_path.read_text())
        trial_num = meta.get("trial_num")
        trial_val = meta.get("trial_value")

    print(f"\n  Hyperparameters (from saved booster + meta.json)")
    if trial_num is not None:
        tv_str = f"  (CV PR-AUC: {trial_val:.4f})" if trial_val is not None else ""
        print(f"    Optuna trial {trial_num}{tv_str}")

    rows = [
        ("max_depth",         params["max_depth"]),
        ("min_child_weight",  params["min_child_weight"]),
        ("max_delta_step",    params["max_delta_step"]),
        ("eta (learning_rate)", params["eta"]),
        ("gamma",             params["gamma"]),
        ("lambda",            params["lambda_"]),
        ("alpha",             params["alpha"]),
        ("subsample",         params["subsample"]),
        ("colsample_bytree",  params["colsample_bytree"]),
        ("colsample_bylevel", params["colsample_bylevel"]),
        ("scale_pos_weight",  params["scale_pos_weight"]),
        ("best_iteration",    params["best_iteration"]),
        ("n_trees",           params["n_trees"]),
    ]
    print(f"  {'Parameter':<24}  {'Value':>10}")
    for name, val in rows:
        if val is None:
            continue
        val_str = f"{int(val)}" if isinstance(val, float) and val == int(val) else (
            f"{val:.4f}" if isinstance(val, float) else str(val)
        )
        print(f"  {name:<24}  {val_str:>10}")

    issues = []
    md = params["max_depth"]
    mcw = params["min_child_weight"]
    lam = params["lambda_"]
    bi = params["best_iteration"]
    nt = params["n_trees"]

    if md is not None and md >= 6:
        issues.append("WARN  max_depth at cap (6) — marginal fingerprinting risk; confirm feature gain shows geometry dominance")
    if mcw is not None and mcw < 30:
        issues.append(f"WARN  min_child_weight={int(mcw)} below floor of 30 — leaf-size risk at ~8% positive rate")
    if lam is not None and lam < 0.15:
        issues.append(f"WARN  lambda={lam:.4f} — very light L2 regularisation; leaf weights may inflate")
    if bi is not None and nt is not None and bi >= nt - 10:
        issues.append(f"INFO  best_iteration={bi} near n_trees limit — early stopping may not have fired; consider n_estimators > 500")
    if bi is not None and bi < 20:
        issues.append(f"INFO  best_iteration={bi} — very early convergence; model may be over-regularised or data volume is low")

    print()
    if issues:
        for issue in issues:
            print(f"    ⚠  {issue}")
    else:
        print("    ✓  No hyperparameter concerns for base_xg.")


# ── Check 8: hold-out performance metrics ─────────────────────────────────────

def check_holdout_metrics(
    strength: str,
    scored_df: pd.DataFrame,
    oof_gap: float = float("nan"),
) -> dict[str, float]:
    """Full advanced metrics on the hold-out season."""
    hold_out_path = DATA_DIR / "hold_out" / f"{strength}.parquet"
    if not hold_out_path.exists():
        print(f"\n  Hold-out metrics: {hold_out_path.name} not found — skip")
        return {}

    hold_seasons = set(pd.read_parquet(hold_out_path, columns=["season"]).season.unique())
    df = scored_df[scored_df.season.isin(hold_seasons)]
    if len(df) == 0:
        print("\n  Hold-out metrics: no scored events match hold-out seasons — skip")
        return {}

    y = df.goal.to_numpy()
    p = df.base_xg.to_numpy()
    season_label = sorted(hold_seasons)[-1]

    metrics = compute_holdout_metrics(y, p, oof_gap=oof_gap)
    print_holdout_metrics(metrics, season_label, len(df))
    return metrics


# ── main ───────────────────────────────────────────────────────────────────────

def run_strength(strength: str) -> tuple[dict[str, str], dict[str, float]]:
    scored_path = DATA_DIR / "scored" / f"{strength}.parquet"
    if not scored_path.exists():
        print(f"\n  {strength}: scored parquet missing — skip")
        return {}, {}

    df = pd.read_parquet(scored_path, columns=["game_id", "event_idx", "goal", "base_xg", "season"])

    dist_status, _  = check_distribution(df, strength)
    hc_status       = check_high_confidence(df)
    pr_status       = check_precision_recall_balance(df, "base_xg", fix_message=_PR_FIX)
    cal_status      = check_calibration(df, "base_xg")
    oof_status, oof_gap = check_oof_vs_holdout(strength, df, "base_xg", DATA_DIR, MODELS_DIR)
    gain_status     = check_feature_dominance(strength)
    check_season_prauc(df)
    check_hyperparameters(strength)
    metrics = check_holdout_metrics(strength, df, oof_gap=oof_gap)

    statuses = {
        "distribution":   dist_status,
        "high_confidence": hc_status,
        "pr_balance":     pr_status,
        "calibration":    cal_status,
        "oof_vs_holdout": oof_status,
        "feature_gain":   gain_status,
    }
    overall = FAIL if FAIL in statuses.values() else (WARN if WARN in statuses.values() else PASS)
    print(f"\n  {status_icon(overall)} Overall [{strength}]: {overall}")
    return statuses, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="base_xg overfitting diagnostic")
    parser.add_argument("--strength", "-s", choices=STRENGTHS, default=None,
                        help="Single strength state (default: all 5)")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")

    strengths = [args.strength] if args.strength else STRENGTHS
    all_statuses: dict[str, dict[str, str]] = {}
    all_metrics:  dict[str, dict[str, float]] = {}

    print("=" * 62)
    print("  base_xg overfitting / calibration diagnostic")
    print("=" * 62)

    for s in strengths:
        statuses, metrics = run_strength(s)
        all_statuses[s] = statuses
        all_metrics[s]  = metrics

    # Pass/fail summary table
    print(f"\n{'=' * 62}")
    print(f"  Summary")
    print(f"{'─' * 62}")
    checks  = ["distribution", "high_confidence", "pr_balance", "calibration", "oof_vs_holdout", "feature_gain"]
    headers = ["distribut",    "high_conf",        "pr_bal",     "calibrat",    "oof_gap",        "feat_gain"]
    print(f"  {'Strength':<20}  " + "  ".join(f"{h:>9}" for h in headers))
    for s, statuses in all_statuses.items():
        if not statuses:
            continue
        row = "  ".join(f"{status_icon(statuses.get(c, '—'))+'':>9}" for c in checks)
        print(f"  {s:<20}  {row}")

    # Advanced metrics summary table
    W = 125
    print(f"\n{'─' * W}")
    print(f"  Hold-out advanced metrics  (precision/recall threshold = base rate)")
    print(f"{'─' * W}")
    print(
        f"  {'Strength':<20}  {'Base%':>6}  {'PR AUC':>7}  {'PR×':>5}  {'ROC AUC':>8}"
        f"  {'LogLoss':>8}  {'NullLL':>7}  {'ΔLL%':>6}"
        f"  {'Brier':>7}  {'NullBr':>7}  {'ΔBr%':>6}"
        f"  {'ECE':>7}  {'MaxCal':>7}  {'OOFGap':>7}"
        f"  {'Prec':>6}  {'Recall':>7}"
    )
    for s, m in all_metrics.items():
        if not m:
            continue
        oof_str = f"{m['OOF Gap']:>7.4f}" if not np.isnan(m.get("OOF Gap", float("nan"))) else f"{'—':>7}"
        print(
            f"  {s:<20}  {100 * m['Base Rate']:>5.1f}%  {m['PR AUC']:>7.4f}  {m['PR AUC Multiplier']:>5.2f}×  {m['ROC AUC']:>8.4f}"
            f"  {m['Log Loss']:>8.4f}  {m['Null Log Loss']:>7.4f}  {m['Log Loss Improvement %']:>+5.1f}%"
            f"  {m['Brier Score']:>7.4f}  {m['Null Brier']:>7.4f}  {m['Brier Improvement %']:>+5.1f}%"
            f"  {m['ECE']:>7.4f}  {m['Max Calibration Error']:>7.4f}  {oof_str}"
            f"  {m['Precision']:>6.4f}  {m['Recall']:>7.4f}"
        )
    print(f"{'=' * W}\n")


if __name__ == "__main__":
    main()
