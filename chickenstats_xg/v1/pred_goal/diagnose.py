
"""pred_goal calibration / quality diagnostic.

Run from repo root:
    uv run python 1_0_0/pred_goal/diagnose.py
    uv run python 1_0_0/pred_goal/diagnose.py --strength even_strength

Checks (per strength state):
  1. Distribution         — GOAL vs SHOT pred_goal ratio (same as base_xg check 1)
  2. Calibration          — mean(pred) vs actual per decile (Platt scaling should help)
  3. OOF vs hold-out gap  — unbiased OOF PR-AUC vs hold-out (generalisation check)
  4. Lift over base_xg    — pred_goal PR-AUC must exceed base_xg PR-AUC on hold-out
  5. RAPM null rate       — % of train shots with non-null shooter_rapm_career_xg_off
  6. Feature gain         — context leak guard: BASE_XG_FEATURE_COLUMNS must not appear

What to watch for:
  - Distribution ratio >> 3×: model still bifurcating GOAL vs SHOT events
  - OOF gap > 0.05: talent model overfitting to training-era player quality
  - Lift ≤ 0: pred_goal adds no discriminative value over base_xg alone
  - Any base_xg context feature with >0 gain: context leak (process_data.py drop failed)
  - RAPM null rate > 50%: too few players have RAPM → talent signal too sparse
"""

import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    log_loss,
    brier_score_loss,
    precision_score,
    recall_score,
)

from chickenstats_xg.v1.config import BASE_XG_FEATURE_COLUMNS, PASSTHROUGH_COLS, STRENGTHS
from chickenstats_xg.v1.experiments import _apply_fixed_categoricals, _logit

_NON_FEATURE_COLS = frozenset(["goal", "season", "base_xg"] + PASSTHROUGH_COLS)

_BASE = _Path(__file__).parent.parent
DATA_DIR  = _BASE / "data" / "pred_goal"
MODELS_DIR = _BASE / "models" / "pred_goal"

# Thresholds — pred_goal amplifies talent signal on top of base_xg, so higher
# distribution ratios are expected and legitimate (elite shooters in high-danger
# situations genuinely score at much higher rates). Feature gain confirms extreme
# predictions come from talent features, not context leakage.
P90_RATIO_WARN = 6.0
P90_RATIO_FAIL = 10.0
HIGH_CONF        = 0.80
HIGH_CONF_WARN   = 10.0
HIGH_CONF_FAIL   = 20.0
CAL_ERR_WARN = 0.05
CAL_ERR_FAIL = 0.10
OOF_GAP_WARN = 0.03
OOF_GAP_FAIL = 0.05

# RAPM null rate thresholds
RAPM_NULL_WARN = 50.0   # > 50% null → WARN
RAPM_NULL_FAIL = 70.0   # > 70% null → FAIL

# Feature gain: any context feature with > 0 gain is flagged.
# A nonzero FAIL threshold below handles numerical noise (tiny residual splits).
CONTEXT_GAIN_FAIL = 0.01   # single context feature share > 1% → FAIL
CONTEXT_GAIN_WARN = 0.001  # > 0.1% → WARN

# Fingerprinting thresholds — precision/recall balance at base-rate threshold
FINGERPRINT_PREC_WARN = 0.70
FINGERPRINT_PREC_FAIL = 0.85
FINGERPRINT_REC_WARN  = 0.25
FINGERPRINT_REC_FAIL  = 0.15

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
_CONTEXT_FEATS = frozenset(BASE_XG_FEATURE_COLUMNS)


def _icon(s: str) -> str:
    return {"PASS": "✅", "WARN": "⚠️ ", "FAIL": "❌"}.get(s, "—")


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
    bm = _logit(df["base_xg"].to_numpy()) if "base_xg" in df.columns else None
    drop = [c for c in _NON_FEATURE_COLS if c in df.columns]
    X = _apply_fixed_categoricals(df.drop(columns=drop), strength)
    raw = model.predict_proba(X, base_margin=bm)[:, 1]
    return calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]


# ── Check 1: distribution ──────────────────────────────────────────────────────

def check_distribution(scored_df: pd.DataFrame, strength: str) -> tuple[str, float]:
    shots = scored_df.loc[scored_df.goal == 0, "pred_goal"]
    goals = scored_df.loc[scored_df.goal == 1, "pred_goal"]

    shot_p90 = shots.quantile(0.90)
    goal_p90 = goals.quantile(0.90)
    ratio = goal_p90 / shot_p90 if shot_p90 > 1e-9 else float("inf")

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
    print(f"\n  {_icon(status)} Distribution — GOAL p90 / SHOT p90 = {ratio:.2f}×  [{status}]")
    return status, ratio


# ── Check 1b: precision/recall balance (fingerprinting risk) ──────────────────

def check_precision_recall_balance(scored_df: pd.DataFrame) -> str:
    """Precision/recall balance at base-rate threshold — fingerprinting risk check.

    Fingerprinting manifests as extreme precision + low recall: the model finds
    a narrow event cluster it memorised in training, calls those shots goals at
    very high confidence, and ignores everything else.

    Threshold: precision >= 0.85 AND recall <= 0.15 → FAIL (textbook fingerprinting)
               precision >= 0.70 AND recall <= 0.25 → WARN (early warning)
    pred_goal stacks on context_xg's logit base_margin; a fingerprinting signal
    here typically means RAPM null rate is high or rolling windows are too short.
    """
    y = scored_df.goal.to_numpy()
    p = scored_df.pred_goal.to_numpy()
    base_rate = float(y.mean())
    y_pred = (p >= base_rate).astype(int)

    prec = float(precision_score(y, y_pred, zero_division=0))
    rec  = float(recall_score(y, y_pred, zero_division=0))

    if prec >= FINGERPRINT_PREC_FAIL and rec <= FINGERPRINT_REC_FAIL:
        status = FAIL
        note = "Textbook fingerprinting — model memorised specific player combinations"
    elif prec >= FINGERPRINT_PREC_WARN and rec <= FINGERPRINT_REC_WARN:
        status = WARN
        note = "Early fingerprinting signal — check RAPM null rate and window sizes"
    else:
        status = PASS
        note = f"Balance OK  (prec {prec:.3f} / recall {rec:.3f} at base-rate threshold {base_rate:.4f})"

    print(f"\n  Precision/recall balance (threshold = base rate {base_rate:.4f})")
    print(f"    Precision: {prec:.4f}    Recall: {rec:.4f}")
    if status != PASS:
        print(f"    ⚠  {note}")
        print(f"    ⚠  Fix: check RAPM null rate and rolling window sizes; raise lambda floor")
    print(f"\n  {_icon(status)} Precision/recall balance  [{status}]")
    return status


# ── Check 2: calibration ───────────────────────────────────────────────────────

def check_calibration(scored_df: pd.DataFrame) -> str:
    df2 = scored_df.copy()
    try:
        df2["decile"] = pd.qcut(df2.pred_goal, q=10, labels=False, duplicates="drop")
    except Exception:
        print("\n  Calibration: degenerate distribution — skip")
        return WARN

    cal = (
        df2.groupby("decile", observed=True)
        .agg(n=("goal", "size"), mean_pred=("pred_goal", "mean"), actual=("goal", "mean"))
        .reset_index()
    )
    cal["abs_err"] = (cal.mean_pred - cal.actual).abs()
    max_err = cal.abs_err.max()
    status = PASS if max_err < CAL_ERR_WARN else (WARN if max_err < CAL_ERR_FAIL else FAIL)

    print(f"\n  Calibration by decile")
    print(f"  {'Dec':>4}  {'n':>8}  {'mean_pred':>10}  {'actual':>8}  {'abs_err':>8}")
    for _, r in cal.iterrows():
        flag = " ← max" if r.abs_err == max_err else ""
        print(f"  {int(r.decile):>4}  {int(r.n):>8,}  {r.mean_pred:>10.4f}  {r.actual:>8.4f}  {r.abs_err:>8.4f}{flag}")
    print(f"\n  {_icon(status)} Calibration — max abs error = {max_err:.4f}  [{status}]")
    return status


# ── Check 3: OOF vs hold-out PR-AUC ───────────────────────────────────────────

def check_oof_gap(strength: str, hold_df: pd.DataFrame, hold_preds: np.ndarray) -> tuple[str, float]:
    oof_path = MODELS_DIR / strength / "oof.parquet"
    train_path = DATA_DIR / "train" / f"{strength}.parquet"

    if not oof_path.exists() or not train_path.exists():
        print(f"\n  OOF gap: missing files — skip")
        return WARN, float("nan")

    train_raw = pd.read_parquet(train_path, columns=["game_id", "event_idx", "goal"])
    oof_raw   = pd.read_parquet(oof_path)
    merged    = train_raw.merge(oof_raw, on=["game_id", "event_idx"], how="inner")
    valid     = merged.dropna(subset=["pred_goal"])
    n_dropped = len(merged) - len(valid)

    oof_prauc  = float(average_precision_score(valid.goal, valid.pred_goal)) if len(valid) > 0 else float("nan")
    hold_prauc = float(average_precision_score(hold_df.goal, hold_preds))
    gap        = abs(oof_prauc - hold_prauc) if not (np.isnan(oof_prauc) or np.isnan(hold_prauc)) else float("nan")

    status = PASS if gap < OOF_GAP_WARN else (WARN if gap < OOF_GAP_FAIL else FAIL)

    print(f"\n  OOF (training) vs hold-out PR-AUC")
    print(f"    Training OOF  PR-AUC: {oof_prauc:.4f}  (n={len(valid):,}; {n_dropped:,} earliest-fold rows excluded)")
    print(f"    Hold-out      PR-AUC: {hold_prauc:.4f}  (n={len(hold_df):,})")
    print(f"    Gap: {gap:.4f}  {_icon(status)} [{status}]")
    return status, gap


# ── Check 4: lift over base_xg ────────────────────────────────────────────────

def check_lift(strength: str, hold_df: pd.DataFrame, hold_preds: np.ndarray) -> str:
    """pred_goal PR-AUC must exceed base_xg PR-AUC on the same hold-out data."""
    if "base_xg" not in hold_df.columns:
        print(f"\n  Lift over base_xg: base_xg column missing — skip")
        return WARN

    base_prauc = float(average_precision_score(hold_df.goal, hold_df.base_xg))
    pred_prauc = float(average_precision_score(hold_df.goal, hold_preds))
    lift = pred_prauc - base_prauc

    status = PASS if lift > 0 else (WARN if lift > -0.005 else FAIL)

    print(f"\n  Lift over base_xg (hold-out)")
    print(f"    base_xg   PR-AUC: {base_prauc:.4f}")
    print(f"    pred_goal PR-AUC: {pred_prauc:.4f}  (lift = {lift:+.4f})")
    print(f"  {_icon(status)} Lift over base_xg [{status}]")
    return status


# ── Check 5: RAPM null rate ────────────────────────────────────────────────────

def check_rapm_nulls(strength: str) -> str:
    train_path = DATA_DIR / "train" / f"{strength}.parquet"
    rapm_col = "shooter_rapm_career_xg_off"

    if not train_path.exists():
        print(f"\n  RAPM null rate: train parquet missing — skip")
        return WARN

    df = pd.read_parquet(train_path, columns=["season", rapm_col])
    n = len(df)
    n_null = int(df[rapm_col].isna().sum())
    pct_null = 100 * n_null / n if n else 0.0
    status = FAIL if pct_null > RAPM_NULL_FAIL else (WARN if pct_null > RAPM_NULL_WARN else PASS)

    print(f"\n  RAPM null rate (shooter_rapm_career_xg_off in train)")
    print(f"    n_shots: {n:,}   null: {n_null:,} ({pct_null:.1f}%)")

    # Per-season null rates (informational)
    per_season = df.groupby("season")[rapm_col].apply(lambda x: 100 * x.isna().mean()).reset_index()
    per_season.columns = ["season", "null_pct"]
    print(f"\n    {'Season':>8}  {'null%':>7}")
    for _, row in per_season.iterrows():
        print(f"    {int(row.season):>8}  {row.null_pct:>6.1f}%")

    print(f"  {_icon(status)} RAPM null rate [{status}]")
    return status


# ── Check 6: feature gain (context leak guard) ────────────────────────────────

def check_feature_gain(strength: str) -> str:
    model_path = MODELS_DIR / strength / "model.ubj"
    if not model_path.exists():
        print(f"\n  Feature gain: model missing — skip")
        return WARN

    booster = xgb.Booster()
    booster.load_model(str(model_path))
    importance = booster.get_score(importance_type="gain")
    if not importance:
        print(f"\n  Feature gain: no scores returned — skip")
        return WARN

    total_gain  = sum(importance.values())
    ranked      = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    n_show      = min(12, len(ranked))

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
        pct  = gain / total_gain
        note = "⚠️  CONTEXT LEAK" if feat in _CONTEXT_FEATS else ""
        print(f"  {feat:<35}  {pct:>5.1%}  {note}")

    if context_feats:
        names = ", ".join(f"{f}({g/total_gain:.1%})" for f, g in context_feats[:3])
        print(f"\n    Context features with nonzero gain: {names}")
    else:
        print(f"\n    No context leakage detected — all base_xg features have 0 gain.")

    print(f"  {_icon(status)} Feature gain [{status}]")
    return status


# ── Check 7: hold-out advanced metrics ────────────────────────────────────────

def check_holdout_metrics(
    strength: str,
    hold_df: pd.DataFrame,
    hold_preds: np.ndarray,
    oof_gap: float = float("nan"),
) -> dict[str, float]:
    """Full advanced metrics on the hold-out set (pred_goal predictions).

    Mirrors context_xg/diagnose.py check_holdout_metrics() with two differences:
    prediction source is hold_preds array (not a scored parquet column), and
    lift is reported as 'Lift over context_xg' since base_xg in pred_goal
    parquets contains the context_xg output renamed at process_data.py time.
    """
    y = hold_df.goal.to_numpy()
    p = hold_preds
    base_rate = float(y.mean())

    prauc      = float(average_precision_score(y, p))
    prauc_mult = prauc / base_rate if base_rate > 0 else float("nan")
    roc        = float(roc_auc_score(y, p))
    ll         = float(log_loss(y, p))
    brier      = float(brier_score_loss(y, p))

    eps = 1e-15
    null_ll        = float(-(base_rate * np.log(base_rate + eps) + (1 - base_rate) * np.log(1 - base_rate + eps)))
    ll_impr_pct    = 100.0 * (null_ll - ll) / null_ll if null_ll > 0 else 0.0
    null_brier     = float(base_rate * (1 - base_rate))
    brier_impr_pct = 100.0 * (null_brier - brier) / null_brier if null_brier > 0 else 0.0

    bins   = np.linspace(0.0, 1.0, 11)
    binids = np.digitize(p, bins) - 1
    ece = 0.0
    max_cal_err = 0.0
    for i in range(10):
        mask = binids == i
        if np.any(mask):
            bin_err = abs(float(p[mask].mean()) - float(y[mask].mean()))
            ece += (mask.sum() / len(p)) * bin_err
            if bin_err > max_cal_err:
                max_cal_err = bin_err

    y_pred = (p >= base_rate).astype(int)
    prec   = float(precision_score(y, y_pred, zero_division=0))
    rec    = float(recall_score(y, y_pred, zero_division=0))

    lift = float("nan")
    if "base_xg" in hold_df.columns:
        base_prauc = float(average_precision_score(y, hold_df.base_xg.to_numpy()))
        lift = prauc - base_prauc

    season_label = sorted(hold_df.season.unique())[-1] if "season" in hold_df.columns else "—"
    print(f"\n  Hold-out metrics  (season {season_label},  n={len(hold_df):,})")
    print(f"    Base rate:             {base_rate:.4f}  ({100 * base_rate:.1f}%)")
    print(f"    PR AUC:                {prauc:.4f}  (×{prauc_mult:.2f} vs null)")
    print(f"    ROC AUC:               {roc:.4f}")
    print(f"    Log loss:              {ll:.4f}  (null {null_ll:.4f},  {ll_impr_pct:+.1f}% vs null)")
    print(f"    Brier score:           {brier:.4f}  (null {null_brier:.4f},  {brier_impr_pct:+.1f}% vs null)")
    print(f"    ECE:                   {ece:.4f}")
    print(f"    Max cal error:         {max_cal_err:.4f}  (uniform bins)")
    if not np.isnan(oof_gap):
        print(f"    OOF gap:               {oof_gap:.4f}")
    else:
        print(f"    OOF gap:               —")
    if not np.isnan(lift):
        print(f"    Lift over context_xg:  {lift:+.4f}")
    print(f"    Precision:             {prec:.4f}  (threshold = base rate {base_rate:.4f})")
    print(f"    Recall:                {rec:.4f}  (threshold = base rate {base_rate:.4f})")

    return {
        "Base Rate":              base_rate,
        "PR AUC":                 prauc,
        "PR AUC Multiplier":      prauc_mult,
        "ROC AUC":                roc,
        "Log Loss":               ll,
        "Null Log Loss":          null_ll,
        "Log Loss Improvement %": ll_impr_pct,
        "Brier Score":            brier,
        "Null Brier":             null_brier,
        "Brier Improvement %":    brier_impr_pct,
        "ECE":                    ece,
        "Max Calibration Error":  max_cal_err,
        "OOF Gap":                oof_gap,
        "Lift":                   lift,
        "Precision":              prec,
        "Recall":                 rec,
    }


# ── Check 8: PR-AUC by season ─────────────────────────────────────────────────

def check_season_prauc(
    train_scored: pd.DataFrame,
    hold_df: pd.DataFrame,
    hold_preds: np.ndarray,
) -> None:
    """PR-AUC by season for pred_goal and ctx_xg (base_xg = context_xg proxy). Informational only."""
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
        sub = combined[combined.season == season]
        n_goals = int((sub.goal == 1).sum())
        n_shots = int((sub.goal == 0).sum())
        if n_goals < 10:
            continue
        pred_prauc = average_precision_score(sub.goal, sub.pred_goal)
        goal_rate = 100 * n_goals / len(sub)
        base_str = f"{average_precision_score(sub.goal, sub.base_xg):>8.4f}" if has_base else f"{'—':>8}"
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
    hold_df   = pd.read_parquet(hold_path)
    hold_preds = _predict(model, calibrator, hold_df, strength)

    # Build a combined df for distribution/calibration checks (OOF + hold-out predictions)
    oof_path = MODELS_DIR / strength / "oof.parquet"
    if oof_path.exists():
        try:
            train_meta = pd.read_parquet(train_path, columns=["game_id", "event_idx", "goal", "season", "base_xg"])
        except Exception:
            train_meta = pd.read_parquet(train_path, columns=["game_id", "event_idx", "goal"])
        oof_raw    = pd.read_parquet(oof_path)
        merged     = train_meta.merge(oof_raw, on=["game_id", "event_idx"], how="inner")
        # Fallback for earliest-fold rows without OOF coverage
        no_oof = merged["pred_goal"].isna()
        if no_oof.any():
            fallback_df = pd.read_parquet(train_path).iloc[np.where(no_oof.values)[0]].reset_index(drop=True)
            merged.loc[no_oof.values, "pred_goal"] = _predict(model, calibrator, fallback_df, strength)
        keep_cols = [c for c in ["goal", "pred_goal", "season", "base_xg"] if c in merged.columns]
        train_scored = merged[keep_cols].copy()
    else:
        train_scored = pd.DataFrame()

    hold_scored = hold_df[["goal"]].copy()
    hold_scored["pred_goal"] = hold_preds

    if not train_scored.empty:
        scored_df = pd.concat([train_scored[["goal", "pred_goal"]], hold_scored], ignore_index=True)
    else:
        scored_df = hold_scored

    dist_status, _ = check_distribution(scored_df, strength)
    cal_status           = check_calibration(scored_df)
    pr_status            = check_precision_recall_balance(scored_df)
    oof_status, oof_gap  = check_oof_gap(strength, hold_df, hold_preds)
    lift_status          = check_lift(strength, hold_df, hold_preds)
    null_status          = check_rapm_nulls(strength)
    gain_status          = check_feature_gain(strength)
    check_season_prauc(train_scored, hold_df, hold_preds)
    metrics              = check_holdout_metrics(strength, hold_df, hold_preds, oof_gap=oof_gap)

    statuses = {
        "distribution": dist_status,
        "calibration":  cal_status,
        "pr_balance":   pr_status,
        "oof_gap":      oof_status,
        "lift":         lift_status,
        "rapm_nulls":   null_status,
        "feat_gain":    gain_status,
    }
    overall = FAIL if FAIL in statuses.values() else (WARN if WARN in statuses.values() else PASS)
    print(f"\n  {_icon(overall)} Overall [{strength}]: {overall}")
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
    all_metrics: dict[str, dict[str, float]] = {}

    print("=" * 62)
    print("  pred_goal quality diagnostic")
    print("=" * 62)

    for s in strengths:
        statuses, metrics = run_strength(s)
        all_statuses[s] = statuses
        all_metrics[s] = metrics

    # Pass/fail summary table
    print(f"\n{'=' * 62}")
    print(f"  Summary")
    print(f"{'─' * 62}")
    checks  = ["distribution", "calibration", "pr_balance", "oof_gap", "lift", "rapm_nulls", "feat_gain"]
    headers = ["distribut",    "calibrat",    "pr_bal",     "oof_gap", "lift", "rapm_null",  "feat_gain"]
    print(f"  {'Strength':<20}  " + "  ".join(f"{h:>9}" for h in headers))
    for s, statuses in all_statuses.items():
        if not statuses:
            continue
        row = "  ".join(f"{_icon(statuses.get(c, '—'))+'':>9}" for c in checks)
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
        lift_str = f"{m['Lift']:>+7.4f}" if not np.isnan(m.get("Lift", float("nan"))) else f"{'—':>7}"
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