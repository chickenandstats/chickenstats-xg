
"""pred_goal calibration / quality diagnostic.

Run from repo root:
    uv run diagnose-pred-goal
    uv run diagnose-pred-goal --strength even_strength

Checks (per strength state):
  1. Distribution         — GOAL vs SHOT pred_goal ratio (same as base_xg check 1)
  2. Precision/recall     — fingerprinting risk at base-rate threshold
  3. Calibration          — mean(pred) vs actual per decile (Platt scaling should help)
  4. OOF vs hold-out gap  — unbiased OOF PR-AUC vs hold-out (generalisation check)
  5. Lift over base_xg    — pred_goal PR-AUC must exceed base_xg PR-AUC on hold-out
  6. RAPM null rate       — % of train shots with non-null shooter_rapm_career_xg_off
  7. Feature gain         — context leak guard: BASE_XG_FEATURE_COLUMNS must not appear
"""

import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score

from chickenstats_xg.v1.config import BASE_XG_FEATURE_COLUMNS, PASSTHROUGH_COLS, STRENGTHS
from chickenstats_xg.v1.utils.transforms import apply_fixed_categoricals, logit
from chickenstats_xg.v1.utils.diagnose_utils import (
    PASS, WARN, FAIL,
    status_icon,
    check_calibration,
    check_precision_recall_balance,
    compute_holdout_metrics,
    print_holdout_metrics,
    OOF_GAP_WARN, OOF_GAP_FAIL,
    extract_model_hyperparams,
)

_NON_FEATURE_COLS = frozenset(["goal", "season", "base_xg"] + PASSTHROUGH_COLS)
_CONTEXT_FEATS    = frozenset(BASE_XG_FEATURE_COLUMNS)

_BASE      = Path(__file__).parent.parent
DATA_DIR   = _BASE / "data"   / "pred_goal"
MODELS_DIR = _BASE / "models" / "pred_goal"

P90_RATIO_WARN = 6.0
P90_RATIO_FAIL = 10.0
HIGH_CONF      = 0.80
HIGH_CONF_WARN = 10.0
HIGH_CONF_FAIL = 20.0

RAPM_NULL_WARN = 50.0
RAPM_NULL_FAIL = 70.0

CONTEXT_GAIN_FAIL = 0.01
CONTEXT_GAIN_WARN = 0.001

_PR_FIX = "check RAPM null rate and rolling window sizes; raise lambda floor"


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_model(strength: str) -> tuple | None:
    """Return (XGBClassifier, calibrator) or None if artifacts are missing."""
    mp = MODELS_DIR / strength / "model.ubj"
    cp = MODELS_DIR / strength / "calibrator.joblib"
    if not mp.exists() or not cp.exists():
        return None
    model = xgb.XGBClassifier(enable_categorical=True)
    model.load_model(str(mp))
    calibrator = joblib.load(cp)
    return model, calibrator


def _predict(model, calibrator, df: pd.DataFrame, strength: str) -> np.ndarray:
    """Full base-model → Platt-calibrator pipeline. Returns calibrated probabilities."""
    bm   = logit(df["base_xg"].to_numpy()) if "base_xg" in df.columns else None
    drop = [c for c in _NON_FEATURE_COLS if c in df.columns]
    X    = apply_fixed_categoricals(df.drop(columns=drop), strength)
    raw  = model.predict_proba(X, base_margin=bm)[:, 1]
    return calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]


# ── Check 1: distribution ─────────────────────────────────────────────────────

def check_distribution(scored_df: pd.DataFrame, strength: str) -> tuple[str, float]:
    shots = scored_df.loc[scored_df.goal == 0, "pred_goal"]
    goals = scored_df.loc[scored_df.goal == 1, "pred_goal"]

    shot_p90 = shots.quantile(0.90)
    goal_p90 = goals.quantile(0.90)
    ratio    = goal_p90 / shot_p90 if shot_p90 > 1e-9 else float("inf")

    status = PASS if ratio <= P90_RATIO_WARN else (WARN if ratio <= P90_RATIO_FAIL else FAIL)

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


# ── Check 4: OOF vs hold-out PR-AUC ───────────────────────────────────────────

def check_oof_gap(strength: str, hold_df: pd.DataFrame, hold_preds: np.ndarray) -> tuple[str, float]:
    oof_path   = MODELS_DIR / strength / "oof.parquet"
    train_path = DATA_DIR   / "train"  / f"{strength}.parquet"

    if not oof_path.exists() or not train_path.exists():
        print(f"\n  OOF gap: missing files — skip")
        return WARN, float("nan")

    train_raw  = pd.read_parquet(train_path, columns=["game_id", "event_idx", "goal"])
    oof_raw    = pd.read_parquet(oof_path)
    merged     = train_raw.merge(oof_raw, on=["game_id", "event_idx"], how="inner")
    valid      = merged.dropna(subset=["pred_goal"])
    n_dropped  = len(merged) - len(valid)

    oof_prauc  = float(average_precision_score(valid.goal, valid.pred_goal)) if len(valid) > 0 else float("nan")
    hold_prauc = float(average_precision_score(hold_df.goal, hold_preds))
    gap        = abs(oof_prauc - hold_prauc) if not (np.isnan(oof_prauc) or np.isnan(hold_prauc)) else float("nan")

    status = PASS if gap < OOF_GAP_WARN else (WARN if gap < OOF_GAP_FAIL else FAIL)

    print(f"\n  OOF (training) vs hold-out PR-AUC")
    print(f"    Training OOF  PR-AUC: {oof_prauc:.4f}  (n={len(valid):,}; {n_dropped:,} earliest-fold rows excluded)")
    print(f"    Hold-out      PR-AUC: {hold_prauc:.4f}  (n={len(hold_df):,})")
    print(f"    Gap: {gap:.4f}  {status_icon(status)} [{status}]")
    return status, gap


# ── Check 5: lift over base_xg ────────────────────────────────────────────────

def check_lift(strength: str, hold_df: pd.DataFrame, hold_preds: np.ndarray) -> str:
    """pred_goal PR-AUC must exceed base_xg PR-AUC on the same hold-out data."""
    if "base_xg" not in hold_df.columns:
        print(f"\n  Lift over base_xg: base_xg column missing — skip")
        return WARN

    base_prauc = float(average_precision_score(hold_df.goal, hold_df.base_xg))
    pred_prauc = float(average_precision_score(hold_df.goal, hold_preds))
    lift       = pred_prauc - base_prauc

    status = PASS if lift > 0 else (WARN if lift > -0.005 else FAIL)

    print(f"\n  Lift over base_xg (hold-out)")
    print(f"    base_xg   PR-AUC: {base_prauc:.4f}")
    print(f"    pred_goal PR-AUC: {pred_prauc:.4f}  (lift = {lift:+.4f})")
    print(f"  {status_icon(status)} Lift over base_xg [{status}]")
    return status


# ── Check 6: RAPM null rate ────────────────────────────────────────────────────

def check_rapm_nulls(strength: str) -> str:
    train_path = DATA_DIR / "train" / f"{strength}.parquet"
    rapm_col   = "shooter_rapm_career_xg_off"

    if not train_path.exists():
        print(f"\n  RAPM null rate: train parquet missing — skip")
        return WARN

    df      = pd.read_parquet(train_path, columns=["season", rapm_col])
    n       = len(df)
    n_null  = int(df[rapm_col].isna().sum())
    pct_null = 100 * n_null / n if n else 0.0
    status  = FAIL if pct_null > RAPM_NULL_FAIL else (WARN if pct_null > RAPM_NULL_WARN else PASS)

    print(f"\n  RAPM null rate (shooter_rapm_career_xg_off in train)")
    print(f"    n_shots: {n:,}   null: {n_null:,} ({pct_null:.1f}%)")

    per_season = df.groupby("season")[rapm_col].apply(lambda x: 100 * x.isna().mean()).reset_index()
    per_season.columns = ["season", "null_pct"]
    print(f"\n    {'Season':>8}  {'null%':>7}")
    for _, row in per_season.iterrows():
        print(f"    {int(row.season):>8}  {row.null_pct:>6.1f}%")

    print(f"  {status_icon(status)} RAPM null rate [{status}]")
    return status


# ── Check 7: feature gain (context leak guard) ────────────────────────────────

def check_feature_gain(strength: str) -> str:
    model_path = MODELS_DIR / strength / "model.ubj"
    if not model_path.exists():
        print(f"\n  Feature gain: model missing — skip")
        return WARN

    booster    = xgb.Booster()
    booster.load_model(str(model_path))
    importance = booster.get_score(importance_type="gain")
    if not importance:
        print(f"\n  Feature gain: no scores returned — skip")
        return WARN

    total_gain    = sum(importance.values())
    ranked        = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    n_show        = min(12, len(ranked))
    context_feats = [(f, g) for f, g in ranked if f in _CONTEXT_FEATS]
    top_ctx_share = context_feats[0][1] / total_gain if context_feats else 0.0

    status = (
        FAIL if top_ctx_share > CONTEXT_GAIN_FAIL
        else WARN if top_ctx_share > CONTEXT_GAIN_WARN
        else PASS
    )

    print(f"\n  Feature gain dominance (top {n_show} of {len(ranked)})")
    print(f"  {'Feature':<35}  {'Gain%':>6}  {'note':>12}")
    for feat, gain in ranked[:n_show]:
        note = "⚠️  CONTEXT LEAK" if feat in _CONTEXT_FEATS else ""
        print(f"  {feat:<35}  {gain / total_gain:>5.1%}  {note}")

    if context_feats:
        names = ", ".join(f"{f}({g/total_gain:.1%})" for f, g in context_feats[:3])
        print(f"\n    Context features with nonzero gain: {names}")
    else:
        print(f"\n    No context leakage detected — all base_xg features have 0 gain.")

    print(f"  {status_icon(status)} Feature gain [{status}]")
    return status


# ── Check 7b: hyperparameter assessment ───────────────────────────────────────

def check_hyperparameters(strength: str) -> None:
    """Extract hyperparameters from the frozen pred_goal booster and opine on appropriateness.

    Note: full Optuna params (lambda, gamma, alpha, max_depth, etc.) may not be available if
    the study was deleted before meta.json storage was implemented. The booster config always
    provides best_iteration and n_trees; other params depend on what XGBoost serialised.
    """
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
    md  = params["max_depth"]
    bi  = params["best_iteration"]
    nt  = params["n_trees"]
    mcw = params["min_child_weight"]

    if bi is not None and bi < 10:
        issues.append(f"INFO  best_iteration={bi} — thin adjustment layer on top of context_xg prior (expected when talent signal is weak)")
    elif bi is not None and bi > 100:
        issues.append(f"INFO  best_iteration={bi} — substantive talent-layer learning detected; check lift vs context_xg")
    if bi is not None and nt is not None and bi >= nt - 10:
        issues.append(f"INFO  best_iteration={bi} near n_trees limit — early stopping may not have fired; consider n_estimators > 500")
    if md is not None and md >= 5:
        issues.append(f"WARN  max_depth={int(md)} — deep trees with talent features can overfit to training-era RAPM patterns; check OOF gap")
    if mcw is not None and mcw < 30:
        issues.append(f"WARN  min_child_weight={int(mcw)} below floor of 30 — leaf memorisation risk with sparse RAPM features")

    print()
    if issues:
        for issue in issues:
            print(f"    ⚠  {issue}")
    else:
        print("    ✓  No hyperparameter concerns for pred_goal.")


# ── Check 8: hold-out advanced metrics ────────────────────────────────────────

def check_holdout_metrics(
    strength: str,
    hold_df: pd.DataFrame,
    hold_preds: np.ndarray,
    oof_gap: float = float("nan"),
) -> dict[str, float]:
    """Full advanced metrics on the hold-out set (pred_goal predictions)."""
    y       = hold_df.goal.to_numpy()
    p       = hold_preds
    prior_p = hold_df.base_xg.to_numpy() if "base_xg" in hold_df.columns else None
    season_label = sorted(hold_df.season.unique())[-1] if "season" in hold_df.columns else "—"

    metrics = compute_holdout_metrics(y, p, oof_gap=oof_gap, prior_p=prior_p)
    print_holdout_metrics(metrics, season_label, len(hold_df), lift_label="Lift over context_xg")
    return metrics


# ── Check 8b: PR-AUC by season ────────────────────────────────────────────────

def check_season_prauc(
    train_scored: pd.DataFrame,
    hold_df: pd.DataFrame,
    hold_preds: np.ndarray,
) -> None:
    """PR-AUC by season for pred_goal and ctx_xg. Informational only."""
    hold_scored = hold_df[["season", "goal"]].copy() if "season" in hold_df.columns else hold_df[["goal"]].copy()
    hold_scored["pred_goal"] = hold_preds
    if "base_xg" in hold_df.columns:
        hold_scored["base_xg"] = hold_df["base_xg"].values

    combined = pd.concat([train_scored, hold_scored], ignore_index=True) if not train_scored.empty else hold_scored
    if "season" not in combined.columns:
        return

    has_base = "base_xg" in combined.columns
    print(f"\n  PR-AUC by season")
    print(f"  {'Season':>8}  {'n_shots':>9}  {'n_goals':>9}  {'goal%':>7}  {'ctx_xg':>8}  {'pred_goal':>10}")
    for season in sorted(combined.season.unique()):
        sub     = combined[combined.season == season]
        n_goals = int((sub.goal == 1).sum())
        n_shots = int((sub.goal == 0).sum())
        if n_goals < 10:
            continue
        pred_prauc = average_precision_score(sub.goal, sub.pred_goal)
        goal_rate  = 100 * n_goals / len(sub)
        base_str   = f"{average_precision_score(sub.goal, sub.base_xg):>8.4f}" if has_base else f"{'—':>8}"
        print(f"  {season:>8}  {n_shots:>9,}  {n_goals:>9,}  {goal_rate:>6.1f}%  {base_str}  {pred_prauc:>10.4f}")


# ── per-strength runner ────────────────────────────────────────────────────────

def run_strength(strength: str) -> tuple[dict[str, str], dict[str, float]]:
    hold_path  = DATA_DIR / "hold_out" / f"{strength}.parquet"
    train_path = DATA_DIR / "train"    / f"{strength}.parquet"

    artifacts = _load_model(strength)
    if artifacts is None:
        print(f"\n{'─' * 62}")
        print(f"  {strength}: model artifacts missing — run pred_goal/finalize.py first")
        print(f"{'─' * 62}")
        return {}, {}

    if not hold_path.exists() or not train_path.exists():
        print(f"\n{'─' * 62}")
        print(f"  {strength}: train/hold_out parquets missing — run pred_goal/process_data.py first")
        print(f"{'─' * 62}")
        return {}, {}

    model, calibrator = artifacts
    hold_df    = pd.read_parquet(hold_path)
    hold_preds = _predict(model, calibrator, hold_df, strength)

    # Build combined df for distribution/calibration checks (OOF + hold-out)
    oof_path = MODELS_DIR / strength / "oof.parquet"
    if oof_path.exists():
        try:
            train_meta = pd.read_parquet(train_path, columns=["game_id", "event_idx", "goal", "season", "base_xg"])
        except Exception:
            train_meta = pd.read_parquet(train_path, columns=["game_id", "event_idx", "goal"])
        oof_raw = pd.read_parquet(oof_path)
        merged  = train_meta.merge(oof_raw, on=["game_id", "event_idx"], how="inner")
        no_oof  = merged["pred_goal"].isna()
        if no_oof.any():
            fallback_df = pd.read_parquet(train_path).iloc[np.where(no_oof.values)[0]].reset_index(drop=True)
            merged.loc[no_oof.values, "pred_goal"] = _predict(model, calibrator, fallback_df, strength)
        keep_cols    = [c for c in ["goal", "pred_goal", "season", "base_xg"] if c in merged.columns]
        train_scored = merged[keep_cols].copy()
    else:
        train_scored = pd.DataFrame()

    hold_scored = hold_df[["goal"]].copy()
    hold_scored["pred_goal"] = hold_preds

    scored_df = (
        pd.concat([train_scored[["goal", "pred_goal"]], hold_scored], ignore_index=True)
        if not train_scored.empty else hold_scored
    )

    dist_status, _       = check_distribution(scored_df, strength)
    pr_status            = check_precision_recall_balance(scored_df, "pred_goal", fix_message=_PR_FIX)
    cal_status           = check_calibration(scored_df, "pred_goal")
    oof_status, oof_gap  = check_oof_gap(strength, hold_df, hold_preds)
    lift_status          = check_lift(strength, hold_df, hold_preds)
    null_status          = check_rapm_nulls(strength)
    gain_status          = check_feature_gain(strength)
    check_season_prauc(train_scored, hold_df, hold_preds)
    check_hyperparameters(strength)
    metrics              = check_holdout_metrics(strength, hold_df, hold_preds, oof_gap=oof_gap)

    statuses = {
        "distribution": dist_status,
        "pr_balance":   pr_status,
        "calibration":  cal_status,
        "oof_gap":      oof_status,
        "lift":         lift_status,
        "rapm_nulls":   null_status,
        "feat_gain":    gain_status,
    }
    overall = FAIL if FAIL in statuses.values() else (WARN if WARN in statuses.values() else PASS)
    print(f"\n  {status_icon(overall)} Overall [{strength}]: {overall}")
    return statuses, metrics


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="pred_goal calibration / quality diagnostic")
    parser.add_argument("--strength", "-s", choices=STRENGTHS, default=None,
                        help="Single strength state (default: all 5)")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")

    strengths = [args.strength] if args.strength else STRENGTHS
    all_statuses: dict[str, dict[str, str]] = {}
    all_metrics:  dict[str, dict[str, float]] = {}

    print("=" * 62)
    print("  pred_goal quality diagnostic")
    print("=" * 62)

    for s in strengths:
        statuses, metrics = run_strength(s)
        all_statuses[s] = statuses
        all_metrics[s]  = metrics

    # Pass/fail summary table
    print(f"\n{'=' * 62}")
    print(f"  Summary")
    print(f"{'─' * 62}")
    checks  = ["distribution", "pr_balance", "calibration", "oof_gap", "lift", "rapm_nulls", "feat_gain"]
    headers = ["distribut",    "pr_bal",     "calibrat",    "oof_gap", "lift", "rapm_null",  "feat_gain"]
    print(f"  {'Strength':<20}  " + "  ".join(f"{h:>9}" for h in headers))
    for s, statuses in all_statuses.items():
        if not statuses:
            continue
        row = "  ".join(f"{status_icon(statuses.get(c, '—'))+'':>9}" for c in checks)
        print(f"  {s:<20}  {row}")

    # Advanced metrics summary table
    W = 135
    print(f"\n{'─' * W}")
    print(f"  Hold-out advanced metrics  (precision/recall threshold = base rate)")
    print(f"{'─' * W}")
    print(
        f"  {'Strength':<20}  {'Base%':>6}  {'PR AUC':>7}  {'PR×':>5}  {'ROC AUC':>8}"
        f"  {'LogLoss':>8}  {'NullLL':>7}  {'ΔLL%':>6}"
        f"  {'Brier':>7}  {'NullBr':>7}  {'ΔBr%':>6}"
        f"  {'ECE':>7}  {'MaxCal':>7}  {'OOFGap':>7}"
        f"  {'Lift':>7}  {'Prec':>6}  {'Recall':>7}"
    )
    for s, m in all_metrics.items():
        if not m:
            continue
        oof_str  = f"{m['OOF Gap']:>7.4f}" if not np.isnan(m.get("OOF Gap", float("nan"))) else f"{'—':>7}"
        lift_str = f"{m['Lift']:>+7.4f}"   if not np.isnan(m.get("Lift",    float("nan"))) else f"{'—':>7}"
        print(
            f"  {s:<20}  {100 * m['Base Rate']:>5.1f}%  {m['PR AUC']:>7.4f}  {m['PR AUC Multiplier']:>5.2f}×  {m['ROC AUC']:>8.4f}"
            f"  {m['Log Loss']:>8.4f}  {m['Null Log Loss']:>7.4f}  {m['Log Loss Improvement %']:>+5.1f}%"
            f"  {m['Brier Score']:>7.4f}  {m['Null Brier']:>7.4f}  {m['Brier Improvement %']:>+5.1f}%"
            f"  {m['ECE']:>7.4f}  {m['Max Calibration Error']:>7.4f}  {oof_str}"
            f"  {lift_str}  {m['Precision']:>6.4f}  {m['Recall']:>7.4f}"
        )
    print(f"{'=' * W}\n")


if __name__ == "__main__":
    main()
