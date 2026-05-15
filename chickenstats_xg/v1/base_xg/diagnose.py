
"""base_xg overfitting / calibration diagnostic.

Run from repo root:
    uv run python 1_0_0/base_xg/diagnose.py
    uv run python 1_0_0/base_xg/diagnose.py --strength even_strength

Checks (per strength state):
  1. GOAL vs SHOT prediction distribution — GOAL p90 / SHOT p90 ratio, target ≤ 2×
  2. High-confidence event rate           — % of events with base_xg > HIGH_CONF threshold
  3. Calibration by decile               — mean(pred) vs actual goal rate per decile
  4. OOF (training) vs hold-out PR-AUC   — gap indicates generalisation failure
  5. PR-AUC by season                    — spots temporal overfitting or degradation
  6. Feature gain dominance              — flags if a single non-geometry feature
                                           monopolises model gain (overfitting cause)

Overfitting signals to watch for:
  - GOAL p90 / SHOT p90 ratio >> 2 → model memorising GOAL event feature patterns
  - High % of GOAL events at >HIGH_CONF with low % of SHOT events → bifurcation
  - Large OOF vs hold-out PR-AUC gap → model exploits training-era feature correlations
  - Calibration error > 0.05 in high-probability deciles → overconfident predictions
  - Non-geometry feature > 40% of total gain → contextual feature dominating over location
"""

import argparse
import warnings
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

from chickenstats_xg.v1.config import STRENGTHS

# ── paths ─────────────────────────────────────────────────────────────────────
from pathlib import Path as _Path
_BASE = _Path(__file__).parent.parent
DATA_DIR = _BASE / "data" / "base_xg"
MODELS_DIR = _BASE / "models" / "base_xg"

# ── thresholds ─────────────────────────────────────────────────────────────────
P90_RATIO_WARN = 3.5     # GOAL p90 / SHOT p90 — warn above this
P90_RATIO_FAIL = 5.0     # GOAL p90 / SHOT p90 — fail above this
# Rationale: rush shots from the high danger zone genuinely score at 50-75% rates in
# NHL data. A 2× threshold conflates real physics with memorisation; 5× is where the
# ratio becomes implausible regardless of shot quality (no shot type scores at 90%+).
HIGH_CONF = 0.80         # threshold for "high-confidence" check
HIGH_CONF_GOAL_WARN = 10.0   # % of goals above HIGH_CONF — warn above this
HIGH_CONF_GOAL_FAIL = 20.0   # % of goals above HIGH_CONF — fail above this
HIGH_CONF_SHOT_WARN = 1.0    # % of shots above HIGH_CONF — warn above this
CAL_MAX_ERR_WARN = 0.05  # calibration max abs error — warn above this
CAL_MAX_ERR_FAIL = 0.10  # calibration max abs error — fail above this
OOF_GAP_WARN = 0.03      # OOF vs hold-out PR-AUC gap — warn above this
OOF_GAP_FAIL = 0.05      # OOF vs hold-out PR-AUC gap — fail above this

# Geometry features — if these dominate gain the model is doing the right thing.
# Non-geometry features dominating gain signals contextual memorisation.
GEOMETRY_FEATURES: frozenset[str] = frozenset([
    "event_distance", "event_angle", "coords_x", "coords_y",
    "abs_y_distance", "danger", "high_danger",
])
GAIN_TOP_N = 10                # features to show in the printout
GAIN_DOMINANT_WARN = 0.40     # top non-geometry gain share — warn above this
GAIN_DOMINANT_FAIL = 0.60     # top non-geometry gain share — fail above this

# Fingerprinting thresholds — precision/recall balance at base-rate threshold
FINGERPRINT_PREC_WARN = 0.70
FINGERPRINT_PREC_FAIL = 0.85
FINGERPRINT_REC_WARN  = 0.25
FINGERPRINT_REC_FAIL  = 0.15

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"


def _status_icon(status: str) -> str:
    return {"PASS": "✅", "WARN": "⚠️ ", "FAIL": "❌"}[status]


def _pct(n: int, total: int) -> float:
    return 100 * n / total if total > 0 else 0.0


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

    icon = _status_icon(status)
    print(f"\n  {icon} Distribution — GOAL p90 / SHOT p90 = {ratio:.2f}×  [{status}]")
    return status, ratio


# ── Check 2: high-confidence event rate ───────────────────────────────────────

def check_high_confidence(df: pd.DataFrame) -> str:
    """% of events above HIGH_CONF threshold, split by GOAL / SHOT."""
    n_goals = (df.goal == 1).sum()
    n_shots = (df.goal == 0).sum()

    high_goals = ((df.goal == 1) & (df.base_xg > HIGH_CONF)).sum()
    high_shots = ((df.goal == 0) & (df.base_xg > HIGH_CONF)).sum()

    pct_g = _pct(high_goals, n_goals)
    pct_s = _pct(high_shots, n_shots)

    goal_status = PASS if pct_g < HIGH_CONF_GOAL_WARN else (WARN if pct_g < HIGH_CONF_GOAL_FAIL else FAIL)
    shot_status = PASS if pct_s < HIGH_CONF_SHOT_WARN else WARN

    overall = FAIL if FAIL in (goal_status, shot_status) else (WARN if WARN in (goal_status, shot_status) else PASS)

    print(f"\n  High-confidence events (base_xg > {HIGH_CONF})")
    print(f"    GOAL: {high_goals:>6,} / {n_goals:,}  ({pct_g:5.1f}%)  {_status_icon(goal_status)} {goal_status}")
    print(f"    SHOT: {high_shots:>6,} / {n_shots:,}  ({pct_s:5.2f}%)  {_status_icon(shot_status)} {shot_status}")
    return overall


# ── Check 2b: precision/recall balance (fingerprinting risk) ──────────────────

def check_precision_recall_balance(df: pd.DataFrame) -> str:
    """Precision/recall balance at base-rate threshold — fingerprinting risk check.

    Fingerprinting manifests as extreme precision + low recall: the model finds
    a narrow event cluster it memorised in training, calls those shots goals at
    very high confidence, and ignores everything else.

    Threshold: precision >= 0.85 AND recall <= 0.15 → FAIL (textbook fingerprinting)
               precision >= 0.70 AND recall <= 0.25 → WARN (early warning)
    base_xg uses monotone constraints and no interaction groups, making it
    structurally resistant to fingerprinting, but the check acts as a safety net.
    """
    y = df.goal.to_numpy()
    p = df.base_xg.to_numpy()
    base_rate = float(y.mean())
    y_pred = (p >= base_rate).astype(int)

    prec = float(precision_score(y, y_pred, zero_division=0))
    rec  = float(recall_score(y, y_pred, zero_division=0))

    if prec >= FINGERPRINT_PREC_FAIL and rec <= FINGERPRINT_REC_FAIL:
        status = FAIL
        note = "Textbook fingerprinting — model memorised specific event patterns"
    elif prec >= FINGERPRINT_PREC_WARN and rec <= FINGERPRINT_REC_WARN:
        status = WARN
        note = "Early fingerprinting signal — check max_depth and min_child_weight"
    else:
        status = PASS
        note = f"Balance OK  (prec {prec:.3f} / recall {rec:.3f} at base-rate threshold {base_rate:.4f})"

    print(f"\n  Precision/recall balance (threshold = base rate {base_rate:.4f})")
    print(f"    Precision: {prec:.4f}    Recall: {rec:.4f}")
    if status != PASS:
        print(f"    ⚠  {note}")
        print(f"    ⚠  Fix: check max_depth and min_child_weight; raise lambda floor")
    print(f"\n  {_status_icon(status)} Precision/recall balance  [{status}]")
    return status


# ── Check 3: calibration by decile ────────────────────────────────────────────

def check_calibration(df: pd.DataFrame) -> str:
    """Mean predicted probability vs actual goal rate per decile."""
    df2 = df.copy()
    try:
        df2["decile"] = pd.qcut(df2.base_xg, q=10, labels=False, duplicates="drop")
    except Exception:
        print("\n  Calibration: could not compute deciles (likely degenerate distribution)")
        return WARN

    cal = (
        df2.groupby("decile", observed=True)
        .agg(n=("goal", "size"), mean_pred=("base_xg", "mean"), actual=("goal", "mean"))
        .reset_index()
    )
    cal["abs_err"] = (cal.mean_pred - cal.actual).abs()
    max_err = cal.abs_err.max()

    status = PASS if max_err < CAL_MAX_ERR_WARN else (WARN if max_err < CAL_MAX_ERR_FAIL else FAIL)

    print(f"\n  Calibration by decile")
    print(f"  {'Dec':>4}  {'n':>8}  {'mean_pred':>10}  {'actual':>8}  {'abs_err':>8}")
    for _, r in cal.iterrows():
        flag = " ← max" if r.abs_err == max_err else ""
        print(f"  {int(r.decile):>4}  {int(r.n):>8,}  {r.mean_pred:>10.4f}  {r.actual:>8.4f}  {r.abs_err:>8.4f}{flag}")
    print(f"\n  {_status_icon(status)} Calibration — max abs error = {max_err:.4f}  [{status}]")
    return status


# ── Check 4: OOF (training) vs hold-out PR-AUC ────────────────────────────────

def check_oof_vs_holdout(strength: str, scored_df: pd.DataFrame) -> tuple[str, float]:
    """Compare unbiased training PR-AUC (OOF file) vs hold-out PR-AUC.

    OOF file has NaN for the earliest training fold (never in any validation set);
    those rows are excluded from the OOF PR-AUC to keep the estimate unbiased.
    """
    oof_path = MODELS_DIR / strength / "oof.parquet"
    train_path = DATA_DIR / "train" / f"{strength}.parquet"
    hold_out_path = DATA_DIR / "hold_out" / f"{strength}.parquet"

    missing = [p for p in (oof_path, train_path, hold_out_path) if not p.exists()]
    if missing:
        print(f"\n  OOF vs hold-out PR-AUC: missing files {[p.name for p in missing]} — skip")
        return WARN, float("nan")

    oof_raw = pd.read_parquet(oof_path)
    train_raw = pd.read_parquet(train_path, columns=["game_id", "event_idx", "goal"])
    merged = train_raw.merge(oof_raw, on=["game_id", "event_idx"], how="inner")
    valid = merged.dropna(subset=["base_xg"])
    oof_prauc = float(average_precision_score(valid.goal, valid.base_xg)) if len(valid) > 0 else float("nan")
    n_oof_dropped = len(merged) - len(valid)

    hold_seasons = set(pd.read_parquet(hold_out_path, columns=["season"]).season.unique())
    hold_scored = scored_df[scored_df.season.isin(hold_seasons)]
    hold_prauc = float(average_precision_score(hold_scored.goal, hold_scored.base_xg)) if len(hold_scored) > 0 else float("nan")

    gap = abs(hold_prauc - oof_prauc) if not (np.isnan(oof_prauc) or np.isnan(hold_prauc)) else float("nan")
    status = PASS if gap < OOF_GAP_WARN else (WARN if gap < OOF_GAP_FAIL else FAIL)

    print(f"\n  OOF (training) vs hold-out PR-AUC")
    print(f"    Training OOF  PR-AUC: {oof_prauc:.4f}  (n={len(valid):,}; {n_oof_dropped:,} earliest-fold rows excluded)")
    print(f"    Hold-out      PR-AUC: {hold_prauc:.4f}  (n={len(hold_scored):,})")
    print(f"    Gap: {gap:.4f}  {_status_icon(status)} [{status}]")
    return status, gap


# ── Check 5: PR-AUC by season ─────────────────────────────────────────────────

def check_season_prauc(scored_df: pd.DataFrame) -> None:
    """PR-AUC broken out by season. No pass/fail — informational only."""
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
    """Gain distribution across features — flags contextual memorisation at the cause level.

    A healthy model has spatial geometry (event_distance, event_angle) doing the heavy
    lifting, with contextual features (is_rebound, rush_attempt) acting as modifiers.
    If a single non-geometry feature consumes >40% of total gain, the model has latched
    onto it as a proxy for GOAL events rather than learning true shot quality.
    """
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
        pct = gain / total_gain
        tag = "(geometry)" if feat in GEOMETRY_FEATURES else ""
        marker = " ←" if feat == top_non_geo_feat and status != PASS else ""
        print(f"  {feat:<26}  {pct:>5.1%}  {tag:<10}{marker}")

    icon = _status_icon(status)
    if top_non_geo_feat:
        print(
            f"\n  {icon} Feature dominance — top non-geometry: "
            f"{top_non_geo_feat} ({top_non_geo_pct:.1%})  [{status}]"
        )
    else:
        print(f"\n  {icon} Feature dominance — geometry features dominate  [{status}]")
    return status


# ── Check 7: hold-out performance metrics ─────────────────────────────────────

def check_holdout_metrics(
    strength: str,
    scored_df: pd.DataFrame,
    oof_gap: float = float("nan"),
) -> dict[str, float]:
    """Full advanced metrics on the hold-out season.

    Metrics match the calculate_advanced_metrics() reference implementation:
      PR AUC, PR AUC Multiplier, ROC AUC, Log Loss, Null Log Loss,
      Log Loss Improvement %, Brier Score, ECE (uniform bins), Max Calibration Error,
      OOF Gap (threaded in from check_oof_vs_holdout).
    Precision and recall are computed at the goal-rate threshold.
    """
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
    base_rate = float(y.mean())

    prauc      = float(average_precision_score(y, p))
    prauc_mult = prauc / base_rate if base_rate > 0 else float("nan")
    roc        = float(roc_auc_score(y, p))
    ll         = float(log_loss(y, p))
    brier      = float(brier_score_loss(y, p))

    # Null model: always predict the base rate
    eps = 1e-15
    null_ll     = float(-(base_rate * np.log(base_rate + eps) + (1 - base_rate) * np.log(1 - base_rate + eps)))
    ll_impr_pct = 100.0 * (null_ll - ll) / null_ll if null_ll > 0 else 0.0
    null_brier      = float(base_rate * (1 - base_rate))
    brier_impr_pct  = 100.0 * (null_brier - brier) / null_brier if null_brier > 0 else 0.0

    # ECE and max calibration error — uniform probability bins (10 bins over [0, 1])
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

    # Precision / recall at goal-rate threshold
    y_pred = (p >= base_rate).astype(int)
    prec   = float(precision_score(y, y_pred, zero_division=0))
    rec    = float(recall_score(y, y_pred, zero_division=0))

    season_label = sorted(hold_seasons)[-1]
    print(f"\n  Hold-out metrics  (season {season_label},  n={len(df):,})")
    print(f"    Base rate:         {base_rate:.4f}  ({100 * base_rate:.1f}%)")
    print(f"    PR AUC:            {prauc:.4f}  (×{prauc_mult:.2f} vs null)")
    print(f"    ROC AUC:           {roc:.4f}")
    print(f"    Log loss:          {ll:.4f}  (null {null_ll:.4f},  {ll_impr_pct:+.1f}% vs null)")
    print(f"    Brier score:       {brier:.4f}  (null {null_brier:.4f},  {brier_impr_pct:+.1f}% vs null)")
    print(f"    ECE:               {ece:.4f}")
    print(f"    Max cal error:     {max_cal_err:.4f}  (uniform bins)")
    print(f"    OOF gap:           {oof_gap:.4f}" if not np.isnan(oof_gap) else f"    OOF gap:           —")
    print(f"    Precision:         {prec:.4f}  (threshold = base rate {base_rate:.4f})")
    print(f"    Recall:            {rec:.4f}  (threshold = base rate {base_rate:.4f})")

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
        "Precision":              prec,
        "Recall":                 rec,
    }


# ── main ───────────────────────────────────────────────────────────────────────

def run_strength(strength: str) -> tuple[dict[str, str], dict[str, float]]:
    scored_path = DATA_DIR / "scored" / f"{strength}.parquet"
    if not scored_path.exists():
        print(f"\n  {strength}: scored parquet missing — skip")
        return {}, {}

    df = pd.read_parquet(scored_path, columns=["game_id", "event_idx", "goal", "base_xg", "season"])

    dist_status, ratio = check_distribution(df, strength)
    hc_status = check_high_confidence(df)
    pr_status = check_precision_recall_balance(df)
    cal_status = check_calibration(df)
    oof_status, oof_gap = check_oof_vs_holdout(strength, df)
    gain_status = check_feature_dominance(strength)
    check_season_prauc(df)
    metrics = check_holdout_metrics(strength, df, oof_gap=oof_gap)

    statuses = {
        "distribution": dist_status,
        "high_confidence": hc_status,
        "pr_balance": pr_status,
        "calibration": cal_status,
        "oof_vs_holdout": oof_status,
        "feature_gain": gain_status,
    }
    overall = FAIL if FAIL in statuses.values() else (WARN if WARN in statuses.values() else PASS)
    print(f"\n  {_status_icon(overall)} Overall [{strength}]: {overall}")
    return statuses, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="base_xg overfitting diagnostic")
    parser.add_argument("--strength", "-s", choices=STRENGTHS, default=None,
                        help="Single strength state (default: all 5)")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")

    strengths = [args.strength] if args.strength else STRENGTHS
    all_statuses: dict[str, dict[str, str]] = {}
    all_metrics: dict[str, dict[str, float]] = {}

    print("=" * 62)
    print("  base_xg overfitting / calibration diagnostic")
    print("=" * 62)

    for s in strengths:
        statuses, metrics = run_strength(s)
        all_statuses[s] = statuses
        all_metrics[s] = metrics

    # Pass/fail summary table
    print(f"\n{'=' * 62}")
    print(f"  Summary")
    print(f"{'─' * 62}")
    checks = ["distribution", "high_confidence", "pr_balance", "calibration", "oof_vs_holdout", "feature_gain"]
    headers = ["distribut", "high_conf", "pr_bal", "calibrat", "oof_gap", "feat_gain"]
    print(f"  {'Strength':<20}  " + "  ".join(f"{h:>9}" for h in headers))
    for s, statuses in all_statuses.items():
        if not statuses:
            continue
        row = "  ".join(f"{_status_icon(statuses.get(c, '—'))+'':>9}" for c in checks)
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