"""context_xg overfitting / calibration diagnostic.

Run from repo root:
    uv run diagnose-context-xg
    uv run diagnose-context-xg --strength even_strength

Checks (per strength state):
  1. SHOT p90 / base_rate distribution    — bimodal cliff check (SHOT p90 >> base_rate)
  2. High-confidence event rate           — % of events with context_xg > HIGH_CONF threshold
  3. Precision/recall balance             — fingerprinting risk at base-rate threshold
  4. Calibration by decile               — mean(pred) vs actual goal rate per decile
  5. OOF (training) vs hold-out PR-AUC   — gap indicates generalisation failure
  6. Lift over base_xg                   — context_xg PR-AUC vs base_xg PR-AUC on same data
  7. PR-AUC by season                    — spots temporal overfitting or degradation
  8. Feature gain concentration          — flags if a single feature dominates gbtree gain
"""

import argparse
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path as _Path
from sklearn.metrics import average_precision_score

from chickenstats_xg.v1.config import STRENGTHS
from chickenstats_xg.v1.utils.diagnose_utils import (
    PASS,
    WARN,
    FAIL,
    status_icon,
    pct,
    check_calibration,
    check_precision_recall_balance,
    check_oof_vs_holdout,
    compute_holdout_metrics,
    print_holdout_metrics,
    extract_model_hyperparams,
)

# ── paths ─────────────────────────────────────────────────────────────────────
_BASE = _Path(__file__).parent.parent
DATA_DIR = _BASE / "data" / "context_xg"
MODELS_DIR = _BASE / "models" / "context_xg"

# ── thresholds ─────────────────────────────────────────────────────────────────
# Distribution: SHOT p90 / base_rate. Bimodal cliff → non-goal shots spike.
SHOT_P90_BASE_RATIO_WARN = 2.5
SHOT_P90_BASE_RATIO_FAIL = 5.0

_BASE_RATE_REF = 0.07
HIGH_CONF = 0.80
HIGH_CONF_GOAL_WARN = 10.0
HIGH_CONF_GOAL_FAIL = 20.0
HIGH_CONF_SHOT_WARN = 1.0

LIFT_WARN = 0.003
LIFT_FAIL = 0.0
GAIN_DOMINANT_WARN = 0.60
GAIN_DOMINANT_FAIL = 0.80

_PR_FIX = "enforce max_delta_step=1 in Optuna search space; raise lambda floor to ≥1.0"


# ── Check 1: GOAL vs SHOT distribution ────────────────────────────────────────


def check_distribution(df: pd.DataFrame, strength: str) -> tuple[str, float]:
    """SHOT p90 / base_rate bimodal check.

    A bimodal cliff pushes non-goal shot predictions to high probability, so
    SHOT p90 >> base_rate. Well-calibrated models have SHOT p90 ≈ 1–2× base_rate.
    """
    shots = df.loc[df.goal == 0, "context_xg"]
    goals = df.loc[df.goal == 1, "context_xg"]

    shot_p90 = shots.quantile(0.90)
    base_rate = len(goals) / (len(goals) + len(shots))
    ratio = shot_p90 / base_rate if base_rate > 1e-9 else float("inf")

    status = PASS if ratio < SHOT_P90_BASE_RATIO_WARN else (WARN if ratio < SHOT_P90_BASE_RATIO_FAIL else FAIL)

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
    print(f"\n  {status_icon(status)} Distribution — SHOT p90 / base_rate = {ratio:.2f}×  [{status}]")
    return status, ratio


# ── Check 2: high-confidence event rate ───────────────────────────────────────


def check_high_confidence(df: pd.DataFrame) -> str:
    """% of events above HIGH_CONF threshold, split by GOAL / SHOT.

    Thresholds scale with base rate so high-base-rate states (e.g. empty_against
    ~57%) are not penalised for naturally elevated predictions.
    """
    n_goals = (df.goal == 1).sum()
    n_shots = (df.goal == 0).sum()
    base_rate = n_goals / (n_goals + n_shots) if (n_goals + n_shots) > 0 else 0.0

    scale = max(1.0, base_rate / _BASE_RATE_REF)
    adj_goal_warn = min(HIGH_CONF_GOAL_WARN * scale, 50.0)
    adj_goal_fail = min(HIGH_CONF_GOAL_FAIL * scale, 80.0)
    adj_shot_warn = min(HIGH_CONF_SHOT_WARN * scale, 10.0)

    high_goals = ((df.goal == 1) & (df.context_xg > HIGH_CONF)).sum()
    high_shots = ((df.goal == 0) & (df.context_xg > HIGH_CONF)).sum()

    pct_g = pct(high_goals, n_goals)
    pct_s = pct(high_shots, n_shots)

    goal_status = PASS if pct_g < adj_goal_warn else (WARN if pct_g < adj_goal_fail else FAIL)
    shot_status = PASS if pct_s < adj_shot_warn else WARN
    overall = FAIL if FAIL in (goal_status, shot_status) else (WARN if WARN in (goal_status, shot_status) else PASS)

    print(f"\n  High-confidence events (context_xg > {HIGH_CONF})")
    print(f"    GOAL: {high_goals:>6,} / {n_goals:,}  ({pct_g:5.1f}%)  {status_icon(goal_status)} {goal_status}")
    print(f"    SHOT: {high_shots:>6,} / {n_shots:,}  ({pct_s:5.2f}%)  {status_icon(shot_status)} {shot_status}")
    return overall


# ── Check 5: lift over base_xg ────────────────────────────────────────────────


def check_lift(df: pd.DataFrame) -> str:
    """context_xg PR-AUC vs base_xg PR-AUC on the same scored data."""
    if "base_xg" not in df.columns:
        print("\n  Lift over base_xg: base_xg column not present in scored parquet — skip")
        return WARN

    valid = df.dropna(subset=["base_xg", "context_xg"])
    base_prauc = float(average_precision_score(valid.goal, valid.base_xg))
    ctx_prauc = float(average_precision_score(valid.goal, valid.context_xg))
    lift = ctx_prauc - base_prauc

    status = PASS if lift >= LIFT_WARN else (WARN if lift > LIFT_FAIL else FAIL)

    print(f"\n  Lift over base_xg")
    print(f"    base_xg    PR-AUC: {base_prauc:.4f}")
    print(f"    context_xg PR-AUC: {ctx_prauc:.4f}")
    print(f"    Lift: {lift:+.4f}  {status_icon(status)} [{status}]")
    return status


# ── Check 7: PR-AUC by season ─────────────────────────────────────────────────


def check_season_prauc(scored_df: pd.DataFrame) -> None:
    """PR-AUC broken out by season. No pass/fail — informational only."""
    has_base = "base_xg" in scored_df.columns
    print(f"\n  PR-AUC by season")
    print(f"  {'Season':>8}  {'n_shots':>9}  {'n_goals':>9}  {'goal%':>7}  {'base_xg':>8}  {'ctx_xg':>8}")
    for season in sorted(scored_df.season.unique()):
        sub = scored_df[scored_df.season == season]
        n_goals = int((sub.goal == 1).sum())
        n_shots = int((sub.goal == 0).sum())
        if n_goals < 10:
            continue
        ctx_prauc = average_precision_score(sub.goal, sub.context_xg)
        goal_rate = 100 * n_goals / len(sub)
        base_str = f"{average_precision_score(sub.goal, sub.base_xg):>8.4f}" if has_base else f"{'—':>8}"
        print(f"  {season:>8}  {n_shots:>9,}  {n_goals:>9,}  {goal_rate:>6.1f}%  {base_str}  {ctx_prauc:>8.4f}")


# ── Check 8: feature gain concentration ───────────────────────────────────────


def check_feature_gain_concentration(strength: str) -> str:
    """gbtree feature gain distribution — flags single-feature dominance."""
    model_path = MODELS_DIR / strength / "model.ubj"
    if not model_path.exists():
        print(f"\n  Feature gain: model not found at {model_path} — skip")
        return WARN

    booster = xgb.Booster()
    booster.load_model(str(model_path))
    try:
        importance = booster.get_score(importance_type="gain")
    except Exception:
        print("\n  Feature gain: get_score() failed — skip")
        return WARN

    if not importance:
        print("\n  Feature gain: no gains returned — skip")
        return WARN

    total = sum(importance.values())
    ranked = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    top_feat, top_gain = ranked[0]
    top_pct = top_gain / total if total > 0 else 0.0

    status = PASS if top_pct < GAIN_DOMINANT_WARN else (WARN if top_pct < GAIN_DOMINANT_FAIL else FAIL)

    n_show = min(10, len(ranked))
    print(f"\n  Feature gain concentration (top {n_show} of {len(ranked)}, by gain)")
    print(f"  {'Feature':<30}  {'Gain':>12}  {'gain%':>6}")
    for feat, g in ranked[:n_show]:
        marker = " ←" if feat == top_feat and status != PASS else ""
        print(f"  {feat:<30}  {g:>12.1f}  {g / total:>5.1%}{marker}")

    print(f"\n  {status_icon(status)} Feature gain concentration — top feature: {top_feat} ({top_pct:.1%})  [{status}]")
    return status


# ── Check 8b: hyperparameter assessment ───────────────────────────────────────


def check_hyperparameters(strength: str) -> None:
    """Extract hyperparameters from the frozen context_xg booster and opine on appropriateness.

    The CRITICAL check is max_delta_step: mds=1 prevents the bimodal cliff; mds≥2 produces
    large leaf weights that cluster non-goal shots at high predicted probability. This was the
    root cause of the calibration failures documented in Issues 11 and 12.
    """
    import json as _json

    model_path = MODELS_DIR / strength / "model.ubj"
    meta_path = MODELS_DIR / strength / "meta.json"

    params = extract_model_hyperparams(model_path)
    if params is None:
        print(f"\n  Hyperparameters: model not found at {model_path} — skip")
        return

    trial_num = trial_vals = None
    if meta_path.exists():
        meta = _json.loads(meta_path.read_text())
        trial_num = meta.get("optuna_trial_number")
        trial_vals = meta.get("optuna_trial_values")

    print(f"\n  Hyperparameters (from saved booster + meta.json)")
    if trial_num is not None:
        if trial_vals is not None:
            tv_str = f"  (PR-AUC: {trial_vals[0]:.4f}  log-loss: {trial_vals[1]:.4f})"
        else:
            tv_str = ""
        print(f"    Optuna trial {trial_num}{tv_str}")

    rows = [
        ("max_depth", params["max_depth"]),
        ("min_child_weight", params["min_child_weight"]),
        ("max_delta_step", params["max_delta_step"]),
        ("eta (learning_rate)", params["eta"]),
        ("gamma", params["gamma"]),
        ("lambda", params["lambda_"]),
        ("alpha", params["alpha"]),
        ("subsample", params["subsample"]),
        ("scale_pos_weight", params["scale_pos_weight"]),
        ("best_iteration", params["best_iteration"]),
        ("n_trees", params["n_trees"]),
    ]
    print(f"  {'Parameter':<24}  {'Value':>10}")
    for name, val in rows:
        if val is None:
            continue
        val_str = (
            f"{int(val)}"
            if isinstance(val, float) and val == int(val)
            else (f"{val:.4f}" if isinstance(val, float) else str(val))
        )
        print(f"  {name:<24}  {val_str:>10}")

    issues = []
    mds = params["max_delta_step"]
    mcw = params["min_child_weight"]
    lam = params["lambda_"]
    bi = params["best_iteration"]
    nt = params["n_trees"]

    if mds is not None and mds > 1:
        issues.append(
            f"FAIL  max_delta_step={int(mds)} > 1 — BIMODAL CLIFF RISK: large leaf weights will cluster "
            f"non-goal shots at high predicted probability. Re-finalize with --top-n 150."
        )
    elif mds is not None and mds == 1:
        issues.append("PASS  max_delta_step=1 confirmed — bimodal cliff structurally prevented.")
    if lam is not None and lam < 1.0:
        issues.append(f"WARN  lambda={lam:.4f} below expected floor of ~1.0 for constrained-feature dataset")
    if lam is not None and lam > 40:
        issues.append(
            f"INFO  lambda={lam:.1f} — strong L2 shrinkage (expected for depth-2 gbtree with isolated constraint groups)"
        )
    if mcw is not None and mcw < 50 and strength in ("shorthanded", "empty_for"):
        issues.append(f"WARN  min_child_weight={int(mcw)} below recommended ≥50 for {strength} (sparse flag groups)")
    if bi is not None and nt is not None and bi >= nt - 10:
        issues.append(f"INFO  best_iteration={bi} near n_trees limit — early stopping may not have fired")
    if bi is not None and bi < 30:
        issues.append(
            f"INFO  best_iteration={bi} — expected range for mds=1 + moderate eta; low tree count confirmed non-bimodal"
        )

    print()
    if issues:
        for issue in issues:
            print(f"    {'❌' if 'FAIL' in issue else '✅' if 'PASS' in issue else '⚠ '}  {issue}")
    else:
        print("    ✓  No hyperparameter concerns for context_xg.")


# ── Check 5b: hold-out performance metrics ────────────────────────────────────


def check_holdout_metrics(
    strength: str,
    scored_df: pd.DataFrame,
    oof_gap: float = float("nan"),
) -> dict[str, float]:
    """Full advanced metrics on the hold-out season (context_xg predictions)."""
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
    p = df.context_xg.to_numpy()
    prior_p = df.base_xg.to_numpy() if "base_xg" in df.columns else None
    season_label = sorted(hold_seasons)[-1]

    metrics = compute_holdout_metrics(y, p, oof_gap=oof_gap, prior_p=prior_p)
    print_holdout_metrics(metrics, season_label, len(df), lift_label="Lift over base_xg")
    return metrics


# ── main ───────────────────────────────────────────────────────────────────────


def run_strength(strength: str) -> tuple[dict[str, str], dict[str, float]]:
    scored_path = DATA_DIR / "scored" / f"{strength}.parquet"
    if not scored_path.exists():
        print(f"\n  {strength}: scored parquet missing — skip")
        return {}, {}

    df = pd.read_parquet(scored_path)
    keep = [c for c in ["game_id", "event_idx", "goal", "context_xg", "base_xg", "season"] if c in df.columns]
    df = df[keep]

    dist_status, _ = check_distribution(df, strength)
    hc_status = check_high_confidence(df)
    pr_status = check_precision_recall_balance(df, "context_xg", fix_message=_PR_FIX)
    cal_status = check_calibration(df, "context_xg")
    oof_status, oof_gap = check_oof_vs_holdout(strength, df, "context_xg", DATA_DIR, MODELS_DIR)
    lift_status = check_lift(df)
    gain_status = check_feature_gain_concentration(strength)
    check_season_prauc(df)
    check_hyperparameters(strength)
    metrics = check_holdout_metrics(strength, df, oof_gap=oof_gap)

    statuses = {
        "distribution": dist_status,
        "high_confidence": hc_status,
        "pr_balance": pr_status,
        "calibration": cal_status,
        "oof_vs_holdout": oof_status,
        "lift": lift_status,
        "gain_conc": gain_status,
    }
    overall = FAIL if FAIL in statuses.values() else (WARN if WARN in statuses.values() else PASS)
    print(f"\n  {status_icon(overall)} Overall [{strength}]: {overall}")
    return statuses, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="context_xg overfitting / calibration diagnostic")
    parser.add_argument(
        "--strength", "-s", choices=STRENGTHS, default=None, help="Single strength state (default: all 5)"
    )
    args = parser.parse_args()

    warnings.filterwarnings("ignore")

    strengths = [args.strength] if args.strength else STRENGTHS
    all_statuses: dict[str, dict[str, str]] = {}
    all_metrics: dict[str, dict[str, float]] = {}

    print("=" * 62)
    print("  context_xg overfitting / calibration diagnostic")
    print("=" * 62)

    for s in strengths:
        statuses, metrics = run_strength(s)
        all_statuses[s] = statuses
        all_metrics[s] = metrics

    # Pass/fail summary table
    print(f"\n{'=' * 62}")
    print(f"  Summary")
    print(f"{'─' * 62}")
    checks = ["distribution", "high_confidence", "pr_balance", "calibration", "oof_vs_holdout", "lift", "gain_conc"]
    headers = ["distribut", "high_conf", "pr_bal", "calibrat", "oof_gap", "lift", "gain_conc"]
    print(f"  {'Strength':<20}  " + "  ".join(f"{h:>9}" for h in headers))
    for s, statuses in all_statuses.items():
        if not statuses:
            continue
        row = "  ".join(f"{status_icon(statuses.get(c, '—')) + '':>9}" for c in checks)
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
        oof_str = f"{m['OOF Gap']:>7.4f}" if not np.isnan(m.get("OOF Gap", float("nan"))) else f"{'—':>7}"
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
