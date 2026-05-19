"""Shared diagnostic utilities for base_xg, context_xg, and pred_goal diagnose.py files."""

import json
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"

FINGERPRINT_PREC_WARN = 0.70
FINGERPRINT_PREC_FAIL = 0.85
FINGERPRINT_REC_WARN = 0.25
FINGERPRINT_REC_FAIL = 0.15

CAL_MAX_ERR_WARN = 0.05
CAL_MAX_ERR_FAIL = 0.10
OOF_GAP_WARN = 0.03
OOF_GAP_FAIL = 0.05


def status_icon(s: str) -> str:
    return {"PASS": "✅", "WARN": "⚠️ ", "FAIL": "❌"}.get(s, "—")


def pct(n: int, total: int) -> float:
    return 100 * n / total if total > 0 else 0.0


def check_calibration(
    df: pd.DataFrame,
    pred_col: str,
    warn: float = CAL_MAX_ERR_WARN,
    fail: float = CAL_MAX_ERR_FAIL,
) -> str:
    """Mean predicted probability vs actual goal rate per decile."""
    df2 = df.copy()
    try:
        df2["decile"] = pd.qcut(df2[pred_col], q=10, labels=False, duplicates="drop")
    except Exception:
        print("\n  Calibration: could not compute deciles (likely degenerate distribution)")
        return WARN

    cal = (
        df2.groupby("decile", observed=True)
        .agg(n=("goal", "size"), mean_pred=(pred_col, "mean"), actual=("goal", "mean"))
        .reset_index()
    )
    cal["abs_err"] = (cal.mean_pred - cal.actual).abs()
    max_err = cal.abs_err.max()
    status = PASS if max_err < warn else (WARN if max_err < fail else FAIL)

    print(f"\n  Calibration by decile")
    print(f"  {'Dec':>4}  {'n':>8}  {'mean_pred':>10}  {'actual':>8}  {'abs_err':>8}")
    for _, r in cal.iterrows():
        flag = " ← max" if r.abs_err == max_err else ""
        print(f"  {int(r.decile):>4}  {int(r.n):>8,}  {r.mean_pred:>10.4f}  {r.actual:>8.4f}  {r.abs_err:>8.4f}{flag}")
    print(f"\n  {status_icon(status)} Calibration — max abs error = {max_err:.4f}  [{status}]")
    return status


def check_precision_recall_balance(
    df: pd.DataFrame,
    pred_col: str,
    fix_message: str = "check regularization and model depth",
) -> str:
    """Precision/recall balance at base-rate threshold — fingerprinting risk check.

    Fingerprinting manifests as extreme precision + low recall: the model finds
    a narrow event cluster it memorised in training, calls those shots goals at
    very high confidence, and ignores everything else.

    Threshold: precision >= 0.85 AND recall <= 0.15 → FAIL (textbook fingerprinting)
               precision >= 0.70 AND recall <= 0.25 → WARN (early warning)
    """
    y = df.goal.to_numpy()
    p = df[pred_col].to_numpy()
    base_rate = float(y.mean())
    y_pred = (p >= base_rate).astype(int)

    prec = float(precision_score(y, y_pred, zero_division=0))
    rec = float(recall_score(y, y_pred, zero_division=0))

    if prec >= FINGERPRINT_PREC_FAIL and rec <= FINGERPRINT_REC_FAIL:
        status = FAIL
        note = "Textbook fingerprinting — model memorised specific event sequences"
    elif prec >= FINGERPRINT_PREC_WARN and rec <= FINGERPRINT_REC_WARN:
        status = WARN
        note = "Early fingerprinting signal"
    else:
        status = PASS
        note = f"Balance OK  (prec {prec:.3f} / recall {rec:.3f} at base-rate threshold {base_rate:.4f})"

    print(f"\n  Precision/recall balance (threshold = base rate {base_rate:.4f})")
    print(f"    Precision: {prec:.4f}    Recall: {rec:.4f}")
    if status != PASS:
        print(f"    ⚠  {note}")
        print(f"    ⚠  Fix: {fix_message}")
    print(f"\n  {status_icon(status)} Precision/recall balance  [{status}]")
    return status


def check_oof_vs_holdout(
    strength: str,
    scored_df: pd.DataFrame,
    pred_col: str,
    data_dir: Path,
    models_dir: Path,
    warn: float = OOF_GAP_WARN,
    fail: float = OOF_GAP_FAIL,
) -> tuple[str, float]:
    """Compare unbiased training PR-AUC (OOF file) vs hold-out PR-AUC.

    OOF file has NaN for the earliest training fold (never in any validation set);
    those rows are excluded from the OOF PR-AUC to keep the estimate unbiased.
    """
    oof_path = models_dir / strength / "oof.parquet"
    train_path = data_dir / "train" / f"{strength}.parquet"
    hold_out_path = data_dir / "hold_out" / f"{strength}.parquet"

    missing = [p for p in (oof_path, train_path, hold_out_path) if not p.exists()]
    if missing:
        print(f"\n  OOF vs hold-out PR-AUC: missing files {[p.name for p in missing]} — skip")
        return WARN, float("nan")

    oof_raw = pd.read_parquet(oof_path)
    train_raw = pd.read_parquet(train_path, columns=["game_id", "event_idx", "goal"])
    merged = train_raw.merge(oof_raw, on=["game_id", "event_idx"], how="inner")
    valid = merged.dropna(subset=[pred_col])
    oof_prauc = float(average_precision_score(valid.goal, valid[pred_col])) if len(valid) > 0 else float("nan")
    n_dropped = len(merged) - len(valid)

    hold_seasons = set(pd.read_parquet(hold_out_path, columns=["season"]).season.unique())
    hold_scored = scored_df[scored_df.season.isin(hold_seasons)]
    hold_prauc = (
        float(average_precision_score(hold_scored.goal, hold_scored[pred_col]))
        if len(hold_scored) > 0
        else float("nan")
    )

    gap = abs(hold_prauc - oof_prauc) if not (np.isnan(oof_prauc) or np.isnan(hold_prauc)) else float("nan")
    status = PASS if gap < warn else (WARN if gap < fail else FAIL)

    print(f"\n  OOF (training) vs hold-out PR-AUC")
    print(f"    Training OOF  PR-AUC: {oof_prauc:.4f}  (n={len(valid):,}; {n_dropped:,} earliest-fold rows excluded)")
    print(f"    Hold-out      PR-AUC: {hold_prauc:.4f}  (n={len(hold_scored):,})")
    print(f"    Gap: {gap:.4f}  {status_icon(status)} [{status}]")
    return status, gap


def compute_holdout_metrics(
    y: np.ndarray,
    p: np.ndarray,
    oof_gap: float = float("nan"),
    prior_p: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute advanced hold-out metrics. Call print_holdout_metrics() to display.

    Args:
        y:       True labels.
        p:       Predicted probabilities.
        oof_gap: Pre-computed OOF gap from check_oof_vs_holdout (or nan).
        prior_p: Prior-tier predictions for lift computation (base_xg for context_xg,
                 context_xg for pred_goal). None → lift = nan.
    """
    base_rate = float(y.mean())
    prauc = float(average_precision_score(y, p))
    prauc_mult = prauc / base_rate if base_rate > 0 else float("nan")
    roc = float(roc_auc_score(y, p))
    ll = float(log_loss(y, p))
    brier = float(brier_score_loss(y, p))

    eps = 1e-15
    null_ll = float(-(base_rate * np.log(base_rate + eps) + (1 - base_rate) * np.log(1 - base_rate + eps)))
    ll_impr_pct = 100.0 * (null_ll - ll) / null_ll if null_ll > 0 else 0.0
    null_brier = float(base_rate * (1 - base_rate))
    brier_impr_pct = 100.0 * (null_brier - brier) / null_brier if null_brier > 0 else 0.0

    bins = np.linspace(0.0, 1.0, 11)
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
    prec = float(precision_score(y, y_pred, zero_division=0))
    rec = float(recall_score(y, y_pred, zero_division=0))

    if prior_p is not None:
        lift = prauc - float(average_precision_score(y, prior_p))
    else:
        lift = float("nan")

    return {
        "Base Rate": base_rate,
        "PR AUC": prauc,
        "PR AUC Multiplier": prauc_mult,
        "ROC AUC": roc,
        "Log Loss": ll,
        "Null Log Loss": null_ll,
        "Log Loss Improvement %": ll_impr_pct,
        "Brier Score": brier,
        "Null Brier": null_brier,
        "Brier Improvement %": brier_impr_pct,
        "ECE": ece,
        "Max Calibration Error": max_cal_err,
        "OOF Gap": oof_gap,
        "Lift": lift,
        "Precision": prec,
        "Recall": rec,
    }


def print_holdout_metrics(
    metrics: dict[str, float],
    season_label,
    n: int,
    lift_label: str | None = None,
) -> None:
    """Print the standard hold-out metrics block. Pass lift_label to show the lift line."""
    m = metrics
    oof_gap = m.get("OOF Gap", float("nan"))
    lift = m.get("Lift", float("nan"))

    print(f"\n  Hold-out metrics  (season {season_label},  n={n:,})")
    print(f"    Base rate:         {m['Base Rate']:.4f}  ({100 * m['Base Rate']:.1f}%)")
    print(f"    PR AUC:            {m['PR AUC']:.4f}  (×{m['PR AUC Multiplier']:.2f} vs null)")
    print(f"    ROC AUC:           {m['ROC AUC']:.4f}")
    print(
        f"    Log loss:          {m['Log Loss']:.4f}  (null {m['Null Log Loss']:.4f},  {m['Log Loss Improvement %']:+.1f}% vs null)"
    )
    print(
        f"    Brier score:       {m['Brier Score']:.4f}  (null {m['Null Brier']:.4f},  {m['Brier Improvement %']:+.1f}% vs null)"
    )
    print(f"    ECE:               {m['ECE']:.4f}")
    print(f"    Max cal error:     {m['Max Calibration Error']:.4f}  (uniform bins)")
    if not np.isnan(oof_gap):
        print(f"    OOF gap:           {oof_gap:.4f}")
    else:
        print(f"    OOF gap:           —")
    if lift_label and not np.isnan(lift):
        print(f"    {lift_label}: {lift:+.4f}")
    print(f"    Precision:         {m['Precision']:.4f}  (threshold = base rate {m['Base Rate']:.4f})")
    print(f"    Recall:            {m['Recall']:.4f}  (threshold = base rate {m['Base Rate']:.4f})")


def extract_model_hyperparams(model_path: Path) -> dict | None:
    """Extract training hyperparameters for a frozen XGBoost model.

    Prefers params.json sidecar (actual Python-side XGBClassifier constructor values).
    Falls back to booster save_config() only when params.json is absent — that path
    returns XGBoost internal defaults (eta=0.30, max_depth=6, lambda=1) which do NOT
    reflect actual training values and should never be used for analysis.

    Returns a dict with keys: max_depth, min_child_weight, max_delta_step, eta, gamma,
    lambda_, alpha, subsample, colsample_bytree, colsample_bylevel, colsample_bynode,
    scale_pos_weight, best_iteration, n_trees. Returns None if model file is missing.
    """
    if not model_path.exists():
        return None

    # best_iteration and n_trees always come from the booster (not in params.json).
    best_iter = None
    n_trees = None
    try:
        booster = xgb.Booster()
        booster.load_model(str(model_path))
        try:
            best_iter = int(booster.best_iteration) if booster.best_iteration is not None else None
        except Exception:
            pass
        try:
            n_trees = booster.num_trees()
        except Exception:
            pass
    except Exception:
        return None

    # Prefer params.json — written by save_model_artifacts() with actual training values.
    params_json_path = model_path.parent / "params.json"
    if params_json_path.exists():
        try:
            p = json.loads(params_json_path.read_text())
            return {
                "max_depth": float(p["max_depth"]) if "max_depth" in p else None,
                "min_child_weight": float(p["min_child_weight"]) if "min_child_weight" in p else None,
                "max_delta_step": float(p["max_delta_step"]) if "max_delta_step" in p else None,
                "eta": float(p["learning_rate"]) if "learning_rate" in p else None,
                "gamma": float(p["gamma"]) if "gamma" in p else None,
                "lambda_": float(p["lambda"]) if "lambda" in p else None,
                "alpha": float(p["alpha"]) if "alpha" in p else None,
                "subsample": float(p["subsample"]) if "subsample" in p else None,
                "colsample_bytree": float(p["colsample_bytree"]) if "colsample_bytree" in p else None,
                "colsample_bylevel": float(p["colsample_bylevel"]) if "colsample_bylevel" in p else None,
                "colsample_bynode": float(p["colsample_bynode"]) if "colsample_bynode" in p else None,
                "scale_pos_weight": float(p["scale_pos_weight"]) if "scale_pos_weight" in p else None,
                "best_iteration": best_iter,
                "n_trees": n_trees,
            }
        except Exception:
            pass

    # Fallback: booster save_config() — returns XGBoost internal defaults, not actual params.
    try:
        config = json.loads(booster.save_config())
        tp = config.get("learner", {}).get("gradient_booster", {}).get("tree_train_param", {})

        def _float(d: dict, key: str) -> float | None:
            v = d.get(key)
            try:
                return float(v) if v is not None else None
            except (TypeError, ValueError):
                return None

        spw = None
        for path in [
            ["learner", "objective", "reg_loss_param", "scale_pos_weight"],
            ["learner", "objective", "binary_classification_param", "scale_pos_weight"],
            ["learner", "learner_model_param", "scale_pos_weight"],
        ]:
            obj: dict | str = config
            for key in path:
                obj = obj.get(key, {}) if isinstance(obj, dict) else {}
            if obj and isinstance(obj, str):
                try:
                    spw = float(obj)
                    break
                except ValueError:
                    pass

        return {
            "max_depth": _float(tp, "max_depth"),
            "min_child_weight": _float(tp, "min_child_weight"),
            "max_delta_step": _float(tp, "max_delta_step"),
            "eta": _float(tp, "eta"),
            "gamma": _float(tp, "min_split_loss"),
            "lambda_": _float(tp, "reg_lambda"),
            "alpha": _float(tp, "reg_alpha"),
            "subsample": _float(tp, "subsample"),
            "colsample_bytree": _float(tp, "colsample_bytree"),
            "colsample_bylevel": _float(tp, "colsample_bylevel"),
            "colsample_bynode": _float(tp, "colsample_bynode"),
            "scale_pos_weight": spw,
            "best_iteration": best_iter,
            "n_trees": n_trees,
        }
    except Exception:
        return None
