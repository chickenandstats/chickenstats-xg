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

from chickenstats.utilities import ChickenProgress
from chickenstats_xg.v1.config import N_ESTIMATORS

# Trials with max_depth > this cap are excluded from finalize selection — deep trees
# overfit to GOAL event feature patterns, producing near-1.0 predictions with minimal
# PR-AUC gain (< 0.5% relative vs max_depth ≤ 6).
MAX_DEPTH_CAP = 6

# Composite score weights for screen_trials() and context_xg Optuna objective:
#   composite = cal_prauc − (_ECE_WEIGHT × ece) − (_STRUCTURAL_FLAW_WEIGHT × struct_penalty)
#               − dist_penalty − goal_fp_penalty
# ECE penalty: 1.5 deducts ~0.015 PR-AUC points per 1% absolute calibration error.
# Structural flaw penalty: 2.0 heavily discounts models where Isotonic >> Platt (bimodal signal).
# Distribution penalty: max(0, SHOT_p90/base_rate − 2.5) × 0.02 — catches asymmetric bimodal
#   that Platt compression masks from structural_flaw_penalty (at ratio=10.73: −0.165).
# GOAL-side fingerprint penalty: max(0, goal_fraction_high − 0.02) × 3.0 — soft signal mirroring
#   screen_trials() hard-reject (>5%). At the 5% hard-reject threshold: (0.05−0.02)×3.0 = 0.09.
_ECE_WEIGHT = 1.5
_STRUCTURAL_FLAW_WEIGHT = 2.0
_GOAL_FP_WEIGHT = 3.0
_GOAL_FP_RAMP_START = 0.02

# dist_ratio (SHOT_p90 / base_rate) is computed and printed per trial for diagnostics but is
# NOT used as a hard rejection gate. For context_xg, contextual features legitimately push
# SHOT_p90 to 30–40%+ (rebounds, rushes), making any fixed threshold either too tight
# (rejects healthy models) or too loose (passes bimodal ones).

# structural_flaw_penalty relative cap for screen_trials bimodal rejection.
# structural_flaw_penalty = max(0, Platt_cal_ll − Isotonic_cal_ll): the gap measures how much
# better Isotonic (monotone step function) beats Platt (linear log-odds) at log-loss.
# Smooth/heavy-tailed: both calibrators perform similarly; bimodal → large gap.
#
# The cap is expressed as a fraction of null_ll so it scales across strength states:
#   even_strength: null_ll=0.226, healthy struct=0.007 (3.1%); cap=8.8% → 0.020
#   empty_against: null_ll=0.684, healthy struct=0.025 (3.7%); cap=8.8% → 0.060
# A fixed absolute cap of 0.02 would reject all healthy empty_against models because
# Isotonic's extra DOF produce a larger absolute gap on its 1,002-event holdout vs
# even_strength's 97,552 — even though the relative gap is the same ~3-4%.
# 8.8% = 2.8× headroom over even_strength's observed healthy ratio of 3.1%.
#
# NOTE: For base_margin models (pred_goal), the struct hard gate is DISABLED —
# see screen_trials() comment for rationale. This constant only applies when
# bm_train is None (base_xg and context_xg).
_STRUCT_PENALTY_REL_CAP = 0.088


def select_top_trials(
    study: optuna.Study,
    n: int,
    max_depth_cap: int | None = None,
    pareto_tradeoff_weight: float = 0.5,
    goal_fp_cap: float | None = None,
) -> list[tuple[dict, int, optuna.Trial]]:
    """Return top N completed trials as (params, trial_num, trial) triples.

    Pareto-optimal trials are preferred first: they are non-dominated, so any model
    selected from the front is guaranteed optimal on at least one objective. Non-Pareto
    trials fill remaining slots only when the front is smaller than N.

    Within each tier (Pareto vs non-Pareto), trials are ranked by
    values[0] − pareto_tradeoff_weight × values[1].
    For base_xg/pred_goal: values[0]=PR-AUC, values[1]=log_loss.
    For context_xg: values[0]=composite (PR-AUC minus goal_fp/ECE/structural penalties), values[1]=cal_ll.

    If max_depth_cap is given, trials with max_depth > cap are excluded before ranking
    (fallback to all completed if no shallow trials exist).
    """
    all_completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.values is not None]
    if not all_completed:
        raise RuntimeError("No completed trials in study — run experiments.py first.")
    if max_depth_cap is not None:
        shallow = [t for t in all_completed if t.params.get("max_depth", 0) <= max_depth_cap]
        pool = shallow if shallow else all_completed
    else:
        pool = all_completed

    if goal_fp_cap is not None:
        fp_filtered = [t for t in pool if t.user_attrs.get("goal_fp", float("inf")) < goal_fp_cap]
        if fp_filtered:
            pool = fp_filtered
            print(f"    select_top_trials: goal_fp_cap={goal_fp_cap} → {len(pool)} non-bimodal candidates")
        else:
            print(f"    select_top_trials: WARNING — no trials with goal_fp < {goal_fp_cap}; using all {len(pool)}")

    def composite(t: optuna.Trial) -> float:
        return t.values[0] - pareto_tradeoff_weight * t.values[1]

    pareto_ids = {t.number for t in study.best_trials}
    pareto_pool = sorted([t for t in pool if t.number in pareto_ids], key=composite, reverse=True)
    non_pareto_pool = sorted([t for t in pool if t.number not in pareto_ids], key=composite, reverse=True)

    top = (pareto_pool + non_pareto_pool)[:n]
    print(
        f"    select_top_trials: {len(pareto_pool)} Pareto-optimal, "
        f"{len(non_pareto_pool)} non-Pareto in pool — "
        f"selected {len(top)} ({min(len(pareto_pool), n)} from Pareto front)"
    )
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

    Hard rejection triggers: cal_ll > 2× null_ll (catastrophic miscalibration) always;
    structural_flaw_penalty > struct_cap only when bm_train is None (base_xg / context_xg).
    For pred_goal (bm_train is not None), the struct hard gate is disabled: mds=1 + lambda≥10
    clamps prevent structural bimodality by construction, and adversarial testing (SH holdout,
    187 positives) showed bimodal params (mds=5, lambda=1) produce the same struct ~11% of
    null_ll as healthy params — the gate cannot discriminate, it only causes false rejections.
    struct_penalty is still used as a soft composite-score penalty regardless.
    goal_fp and dist_ratio are diagnostics only. struct_cap is relative to null_ll so it
    scales across strength states with different base rates and holdout sizes.
    If all candidates fail, RuntimeError is raised — no silent fallback to least-bad model.
    """
    y_np = y_hold_out.to_numpy()
    base_rate = float(y_np.mean())
    # Null model log loss: always predicting the base rate.
    null_ll = float(sklearn_log_loss(y_np, np.full(len(y_np), base_rate)))
    bimodal_threshold = 2.0 * null_ll
    # Structural flaw cap relative to null_ll: scales across base rates and holdout sizes.
    # Even_strength: 0.088 × 0.226 = 0.020. Empty_against: 0.088 × 0.684 = 0.060.
    struct_cap = _STRUCT_PENALTY_REL_CAP * null_ll
    print(f"    screening {len(candidates)} candidates using composite scoring...")
    print(
        f"    bimodal threshold: cal_ll > {bimodal_threshold:.4f} (2 × null_ll={null_ll:.4f}), struct_cap={struct_cap:.4f}"
    )

    results: list[tuple[int, dict, float, float, float, optuna.Trial]] = []
    n_bimodal = 0
    with ChickenProgress(transient=True) as progress:
        task = progress.add_task(f"Screening trial 0/{len(candidates)}...", total=len(candidates))
        for trial_params, trial_num, trial_obj in candidates:
            progress.update(task, description=f"Screening trial {trial_num}...", refresh=True)
            screen_params = {**fixed_params, **trial_params}
            if max_depth_cap is not None:
                screen_params["max_depth"] = min(screen_params.get("max_depth", max_depth_cap), max_depth_cap)
            if bm_train is not None:
                screen_params["max_delta_step"] = 1  # mds>=2 causes bimodal cliff with base_margin
                screen_params["lambda"] = max(
                    screen_params.get("lambda", 10.0), 10.0
                )  # lambda<10 bimodal even with mds=1
            model = xgb.XGBClassifier(**screen_params)
            model.fit(
                X_train,
                y_train,
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
            cal_ll = float(sklearn_log_loss(y_np, prob_platt))
            ece = calculate_ece(y_np, prob_platt)

            # Isotonic regression — monotone step function, can reach 0/1 → clip before log_loss.
            iso = IsotonicRegression(out_of_bounds="clip").fit(raw, y_np)
            prob_iso = np.clip(iso.predict(raw), 1e-7, 1 - 1e-7)
            iso_ll = float(sklearn_log_loss(y_np, prob_iso))

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

            composite = (
                cal_prauc
                - (_ECE_WEIGHT * ece)
                - (_STRUCTURAL_FLAW_WEIGHT * structural_flaw_penalty)
                - distribution_penalty
            )

            # GOAL-side fingerprint check: fraction of GOAL events with post-Platt prob > 0.85.
            # Skipped for high base-rate states (base_rate >= 0.5, i.e. empty_against ~80%):
            # at 80% goal rate, most goal events legitimately exceed 0.85 — that's correct
            # calibration, not fingerprinting. The 0.85 threshold is only meaningful when it
            # represents a large multiple of the base rate (e.g. 12× at even_strength).
            goal_mask = y_np == 1
            if base_rate < 0.5 and goal_mask.sum() > 0:
                goal_fraction_high = float(np.mean(prob_platt[goal_mask] > 0.85))
                is_goal_fingerprinting = goal_fraction_high > 0.05
            else:
                goal_fraction_high = 0.0
                is_goal_fingerprinting = False

            # Hard bimodal rejection: Platt cal_ll > 2× null (catastrophic miscalibration).
            # struct gate (Isotonic >> Platt gap) only applies to non-base-margin models
            # (base_xg, context_xg). For pred_goal (bm_train is not None):
            #   1. mds=1 + lambda≥10 clamps prevent structural bimodality by construction.
            #   2. SH (187 hold-out positives) adversarial test: mds=5/lambda=1 produces the
            #      same struct ~11% of null_ll as healthy params — gate has zero discrimination.
            #      Root cause: extreme right-tail predictions in SH's small holdout create a
            #      Platt-Iso gap that is a data artifact, not a bimodal signal.
            # struct_penalty is still used in the composite soft-penalty score.
            struct_hard_fail = (bm_train is None) and (structural_flaw_penalty > struct_cap)
            is_bimodal = cal_ll > bimodal_threshold or struct_hard_fail
            if is_bimodal:
                n_bimodal += 1
            reject_reason = ""
            if cal_ll > bimodal_threshold:
                reject_reason = "cal_ll"
            elif struct_hard_fail:
                reject_reason = "struct"
            progress.console.print(
                f"    trial {trial_num:4d}: {'❌ ' + reject_reason + (' ' * (7 - len(reject_reason))) if is_bimodal else '✅ pass     '}"
                f"composite={composite:+.4f} | "
                f"PR={cal_prauc:.4f}  ECE={ece:.4f}  Platt_LL={cal_ll:.4f}  Iso_LL={iso_ll:.4f}  "
                f"struct={structural_flaw_penalty:.4f}  dist_ratio={dist_ratio:.2f}  dist_pen={distribution_penalty:.4f}  "
                f"goal_fp={goal_fraction_high:.3f}"
            )
            if not is_bimodal:
                results.append((trial_num, trial_params, composite, cal_prauc, ece, trial_obj))
            progress.advance(task)

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
    best_iter = model.best_iteration or params.get("n_estimators", N_ESTIMATORS)
    kfold = TimeSeriesSplit(n_splits=n_splits)
    oof_prob = np.zeros(len(y_train))
    oof_mask = np.zeros(len(y_train), dtype=bool)
    fold_params = {k: v for k, v in params.items() if k not in ("eval_metric", "early_stopping_rounds")}
    fold_params["n_estimators"] = best_iter
    for tr_idx, val_idx in kfold.split(X_train):
        bm_tr = bm_train[tr_idx] if bm_train is not None else None
        bm_val = bm_train[val_idx] if bm_train is not None else None
        fold_m = xgb.XGBClassifier(**fold_params)
        fold_m.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx], base_margin=bm_tr, verbose=False)
        oof_prob[val_idx] = fold_m.predict_proba(X_train.iloc[val_idx], base_margin=bm_val)[:, 1]
        oof_mask[val_idx] = True
    return oof_prob, oof_mask
