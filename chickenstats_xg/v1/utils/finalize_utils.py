"""Shared finalize-pipeline utilities for base_xg, context_xg, and pred_goal finalize.py.

Covers OOF prediction generation, Optuna trial selection, and composite screening.
These functions are never called by the Optuna training loop — they belong to the
post-tuning model selection and calibration pipeline.
"""

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.metrics import log_loss as sklearn_log_loss
from sklearn.model_selection import TimeSeriesSplit

from chickenstats_xg.v1.config import N_ESTIMATORS

# Trials with max_depth > this cap are excluded from finalize selection — deep trees
# overfit to GOAL event feature patterns, producing near-1.0 predictions with minimal
# PR-AUC gain (< 0.5% relative vs max_depth ≤ 6).
MAX_DEPTH_CAP = 6

# Composite score weights for screen_trials() and context_xg Optuna objective:
#   composite = cal_prauc − (_ECE_WEIGHT × ece) − (_STRUCTURAL_FLAW_WEIGHT × struct_penalty) − dist_penalty
# ECE penalty: 1.5 deducts ~0.015 PR-AUC points per 1% absolute calibration error.
# Structural flaw penalty: 2.0 heavily discounts models where Isotonic >> Platt (bimodal signal).
# Distribution penalty: max(0, SHOT_p90/base_rate − 2.5) × 0.02 — catches asymmetric bimodal
#   that Platt compression masks from structural_flaw_penalty (at ratio=10.73: −0.165).
_ECE_WEIGHT = 1.5
_STRUCTURAL_FLAW_WEIGHT = 2.0

# Hard cap for dist_ratio hard rejection in screen_trials().
# Working 1.0.0 models have dist_ratio 1.3–2.0 on 2024-25 hold-out.
# Bimodal 1.0.1 models have dist_ratio 6.5–13.4. Threshold of 3.0 cleanly separates them.
_DIST_RATIO_HARD_CAP = 3.0


def select_top_trials(
    study: optuna.Study,
    n: int,
    max_depth_cap: int | None = None,
) -> list[tuple[dict, int, optuna.Trial]]:
    """Return top N completed trials by Optuna objective value as (params, trial_num, trial) triples.

    For context_xg, trial.value is the composite calibration score (cal_prauc − ECE penalty −
    structural flaw penalty), so calibrated models naturally rank above bimodal ones.
    For base_xg and pred_goal, trial.value is raw hold-out PR-AUC.

    If max_depth_cap is given, trials with max_depth > cap are excluded (fallback to all
    completed if no shallow trials exist).
    """
    all_completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]
    if not all_completed:
        raise RuntimeError("No completed trials in study — run experiments.py first.")
    if max_depth_cap is not None:
        shallow = [t for t in all_completed if t.params.get("max_depth", 0) <= max_depth_cap]
        pool = shallow if shallow else all_completed
    else:
        pool = all_completed
    top = sorted(pool, key=lambda t: t.value, reverse=True)[:n]
    return [(t.params, t.number, t) for t in top]


def calculate_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error using quantile bins.

    Quantile bins put equal weight on each probability decile rather than equal-width
    probability intervals — better for skewed distributions like hockey xG where most
    predictions cluster near the base goal rate (~10%).
    """
    bin_edges = np.percentile(y_prob, np.linspace(0, 100, n_bins + 1))
    bin_ids = np.digitize(y_prob, bin_edges[1:-1])
    ece = 0.0
    for i in range(n_bins):
        mask = bin_ids == i
        if np.any(mask):
            ece += (np.sum(mask) / len(y_prob)) * abs(float(np.mean(y_prob[mask])) - float(np.mean(y_true[mask])))
    return ece


def screen_trials(
    candidates: list[tuple[dict, int, optuna.Trial]],
    fixed_params: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_hold_out: pd.DataFrame,
    y_hold_out: pd.Series,
    bm_train: np.ndarray | None = None,
    bm_hold_out: np.ndarray | None = None,
    max_depth_cap: int | None = None,
) -> tuple[dict, int, optuna.Trial]:
    """Retrain each candidate and select best by composite score. Returns (params, trial_num, trial).

    Composite score = cal_prauc − (_ECE_WEIGHT × ece) − (_STRUCTURAL_FLAW_WEIGHT × structural_flaw_penalty)

    structural_flaw_penalty = max(0, platt_ll − iso_ll): Platt (linear log-odds transform) cannot
    fix bimodal raw probabilities; IsotonicRegression (monotone step function) can. A large gap
    means the raw distribution is structurally flawed, not merely shifted.

    Hard rejection: candidates with cal_ll > 2× null_ll are excluded before composite ranking.
    If all candidates fail, RuntimeError is raised — no silent fallback to least-bad bimodal.
    """
    y_np = y_hold_out.to_numpy()
    base_rate = float(y_np.mean())
    # Null model log loss: always predicting the base rate.
    null_ll = float(sklearn_log_loss(y_np, np.full(len(y_np), base_rate)))
    bimodal_threshold = 2.0 * null_ll
    print(f"    screening {len(candidates)} candidates using composite scoring...")
    print(f"    bimodal threshold: cal_ll > {bimodal_threshold:.4f} (2 × null_ll={null_ll:.4f})")

    results: list[tuple[int, dict, float, float, float, optuna.Trial]] = []
    n_bimodal = 0
    for trial_params, trial_num, trial_obj in candidates:
        screen_params = {**fixed_params, **trial_params}
        if max_depth_cap is not None:
            screen_params["max_depth"] = min(screen_params.get("max_depth", max_depth_cap), max_depth_cap)
        model = xgb.XGBClassifier(**screen_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_hold_out, y_hold_out)],
            base_margin=bm_train,
            base_margin_eval_set=[bm_hold_out] if bm_hold_out is not None else None,
            verbose=False,
        )
        raw = model.predict_proba(X_hold_out, base_margin=bm_hold_out)[:, 1]

        # Platt calibration — linear log-odds transform, sigmoid ceiling prevents 1.0 outputs.
        platt = LogisticRegression(C=1.0, max_iter=1000).fit(raw.reshape(-1, 1), y_np)
        prob_platt = platt.predict_proba(raw.reshape(-1, 1))[:, 1]
        cal_prauc = float(average_precision_score(y_np, prob_platt))
        cal_ll    = float(sklearn_log_loss(y_np, prob_platt))
        ece       = calculate_ece(y_np, prob_platt)

        # Isotonic regression — monotone step function, can reach 0/1 → clip before log_loss.
        iso = IsotonicRegression(out_of_bounds="clip").fit(raw, y_np)
        prob_iso = np.clip(iso.predict(raw), 1e-7, 1 - 1e-7)
        iso_ll   = float(sklearn_log_loss(y_np, prob_iso))

        # Structural flaw penalty: how much better Isotonic is than Platt at log loss.
        # A large gap means the raw distribution is bimodal or non-monotone — uncorrectable by Platt.
        structural_flaw_penalty = max(0.0, cal_ll - iso_ll)

        # Distribution penalty on raw probs — catches asymmetric bimodal that Platt masks.
        shot_mask = y_np == 0
        if shot_mask.sum() > 0 and base_rate > 0:
            dist_ratio = float(np.quantile(raw[shot_mask], 0.90)) / base_rate
        else:
            dist_ratio = 0.0
        distribution_penalty = max(0.0, dist_ratio - 2.5) * 0.02

        composite = cal_prauc - (_ECE_WEIGHT * ece) - (_STRUCTURAL_FLAW_WEIGHT * structural_flaw_penalty) - distribution_penalty

        # Hard bimodal rejection: either Platt cal_ll > 2× null (severe miscalibration) OR
        # raw SHOT p90 > 3× base_rate (non-goal shots clustering in high-prob region).
        # The cal_ll check alone is insufficient: in-sample Platt always passes for any model
        # with reasonable discrimination (ROC-AUC > 0.65), masking bimodal raw distributions.
        # dist_ratio operates on RAW probs before Platt — cannot be masked by recalibration.
        is_bimodal = cal_ll > bimodal_threshold or dist_ratio > _DIST_RATIO_HARD_CAP
        if is_bimodal:
            n_bimodal += 1
        print(
            f"    trial {trial_num:4d}: {'❌ BIMODAL ' if is_bimodal else '✅ pass     '}"
            f"composite={composite:+.4f} | "
            f"PR={cal_prauc:.4f}  ECE={ece:.4f}  Platt_LL={cal_ll:.4f}  Iso_LL={iso_ll:.4f}  "
            f"struct={structural_flaw_penalty:.4f}  dist_ratio={dist_ratio:.2f}  dist_pen={distribution_penalty:.4f}"
        )
        if not is_bimodal:
            results.append((trial_num, trial_params, composite, cal_prauc, ece, trial_obj))

    if not results:
        raise RuntimeError(
            f"All {len(candidates)} screened candidates are bimodal "
            f"(Platt cal_ll > 2× null_ll={null_ll:.4f}). "
            "Run more trials or check the learning_rate ceiling before finalizing."
        )

    results.sort(key=lambda x: x[2], reverse=True)
    best = results[0]
    print(
        f"    → Selected trial {best[0]} from {len(results)} non-bimodal candidates "
        f"({n_bimodal} bimodal rejected) | composite={best[2]:+.4f}  cal_prauc={best[3]:.4f}  ECE={best[4]:.4f}"
    )
    return best[1], best[0], best[5]


def compute_oof_predictions(
    model: xgb.XGBClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict,
    n_splits: int,
    bm_train: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run TimeSeriesSplit OOF predictions, returning (oof_prob, oof_mask).

    Fold models are pinned to model.best_iteration so the OOF probability scale
    matches the early-stopped final model.
    """
    best_iter = model.best_iteration or N_ESTIMATORS
    kfold = TimeSeriesSplit(n_splits=n_splits)
    oof_prob = np.zeros(len(y_train))
    oof_mask = np.zeros(len(y_train), dtype=bool)
    fold_params = {k: v for k, v in params.items() if k not in ("eval_metric", "early_stopping_rounds")}
    fold_params["n_estimators"] = best_iter
    for tr_idx, val_idx in kfold.split(X_train):
        bm_tr  = bm_train[tr_idx]  if bm_train is not None else None
        bm_val = bm_train[val_idx] if bm_train is not None else None
        fold_m = xgb.XGBClassifier(**fold_params)
        fold_m.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx], base_margin=bm_tr, verbose=False)
        oof_prob[val_idx] = fold_m.predict_proba(X_train.iloc[val_idx], base_margin=bm_val)[:, 1]
        oof_mask[val_idx] = True
    return oof_prob, oof_mask
