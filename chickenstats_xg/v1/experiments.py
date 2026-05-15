
import argparse
import functools
import json
import os
import subprocess
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import numpy as np
import optuna
import pandas as pd
import sklearn
import sklearn.metrics
import xgboost as xgb
from dotenv import load_dotenv
from matplotlib.figure import Figure
from mlflow.data.pandas_dataset import PandasDataset, from_pandas
from mlflow.entities import LoggedModelInput, Metric, RunInfo
from mlflow.models.signature import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss as sklearn_log_loss
from sklearn.model_selection import TimeSeriesSplit

from chickenstats_xg.v1.config import (
    BASE_XG_FEATURE_COLUMNS,
    CONTEXT_XG_FEATURE_COLUMNS,
    CONTEXT_XG_INTERACTION_GROUPS,
    CV_TUNE_FOLDS,
    DPI,
    EARLY_STOPPING_ROUNDS,
    FIGSIZE,
    MODELS,
    MONOTONE_CONSTRAINTS,
    N_ESTIMATORS,
    PASSTHROUGH_COLS,
    SEED,
    STRENGTHS,
    compute_performance_tag,
)
from chickenstats_xg.utilities.charts import all_classifier_charts
from chickenstats_xg.utilities.style import set_style
from sklearn.isotonic import IsotonicRegression
from chickenstats_xg.v1.utils.artifacts import load_model_artifacts, params_from_run_name, save_model_metadata
from chickenstats_xg.v1.utils.calibration import IsotonicCalibrator
from chickenstats_xg.v1.utils.finalize_utils import (
    MAX_DEPTH_CAP,
    _ECE_WEIGHT,
    _STRUCTURAL_FLAW_WEIGHT,
    calculate_ece,
    compute_oof_predictions,
    screen_trials,
    select_top_trials,
)
from chickenstats_xg.v1.utils.transforms import apply_fixed_categoricals, logit

set_style()


# ── Data container ─────────────────────────────────────────────────────────────

@dataclass
class ExperimentData:
    """Holds all inputs needed by the Optuna objective for one tuning run."""
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    scale_pos_weight: float
    pd_dataset: PandasDataset
    study_name: str
    parent_info: RunInfo
    pip_requirements: list[str] | None = None
    model: str = "base_xg"
    strength: str = "even_strength"


# ── Metric and visualization helpers ───────────────────────────────────────────

def model_metrics(y: pd.Series, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> dict[str, float]:
    """Compute a standard set of binary classification metrics."""
    return {
        "accuracy":           sklearn.metrics.accuracy_score(y, y_pred),
        "average_precision":  sklearn.metrics.average_precision_score(y, y_pred_proba),
        "balanced_accuracy":  sklearn.metrics.balanced_accuracy_score(y, y_pred),
        "f1":                 sklearn.metrics.f1_score(y, y_pred, zero_division=0),
        "f1_weighted":        sklearn.metrics.f1_score(y, y_pred, average="weighted", zero_division=0),
        "precision":          sklearn.metrics.precision_score(y, y_pred, zero_division=0),
        "precision_weighted": sklearn.metrics.precision_score(y, y_pred, average="weighted", zero_division=0),
        "recall":             sklearn.metrics.recall_score(y, y_pred, zero_division=0),
        "recall_weighted":    sklearn.metrics.recall_score(y, y_pred, average="weighted", zero_division=0),
        "roc_auc":            sklearn.metrics.roc_auc_score(y, y_pred_proba),
        "log_loss":           sklearn.metrics.log_loss(y, y_pred_proba),
    }


def _importance_as_array(
    booster: xgb.Booster, feature_names: list[str], importance_type: str = "weight"
) -> np.ndarray:
    """Convert booster importance dict to a numpy array aligned to feature_names order."""
    score = booster.get_score(importance_type=importance_type)
    return np.array([score.get(f, 0.0) for f in feature_names], dtype=float)


def model_viz(
    model: xgb.XGBClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    run_name: str,
    base_margin: np.ndarray | None = None,
) -> dict[str, Figure | None]:
    """Generate all classifier charts via the utilities.charts module."""
    y_prob = model.predict_proba(X_test, base_margin=base_margin)[:, 1]
    # Use base-rate threshold, not 0.5 — at 6% goal rate, model.predict() flags only ~0.3%
    # of shots as goals (recall ≈ 6%), making classification/confusion charts uninformative.
    # Base-rate threshold matches _run_cv_folds, _objective_body, and the HTML report.
    base_rate = float(y_test.mean())
    y_pred = (y_prob >= base_rate).astype(int)

    feat_names = list(X_train.columns)
    booster = model.get_booster()
    # Gain-based: most informative for feature evaluation (which features provide new signal).
    imp_gain   = _importance_as_array(booster, feat_names, importance_type="gain")
    # Weight-based: shows which features are actually split on (logit_base_xg dominates here).
    imp_weight = _importance_as_array(booster, feat_names, importance_type="weight")

    from chickenstats_xg.utilities.charts import feature_importances as fi_chart
    charts = all_classifier_charts(
        y_true=y_test.to_numpy(),
        y_pred=y_pred,
        y_proba=y_prob,
        importances=imp_gain,
        feature_names=feat_names,
        classes=["no goal", "goal"],
        title_prefix=run_name,
        figsize=FIGSIZE,
        dpi=DPI,
    )
    charts["feature_importances_weight"] = fi_chart(
        imp_weight, feat_names,
        title=f"{run_name} — Feature Importances (weight/splits)",
        relative=False, topn=10, figsize=FIGSIZE, dpi=DPI,
    )
    charts["relative_feature_importances_weight"] = fi_chart(
        imp_weight, feat_names,
        title=f"{run_name} — Relative Feature Importances (weight/splits)",
        relative=True, topn=10, figsize=FIGSIZE, dpi=DPI,
    )
    return charts


def log_viz(charts: dict[str, Figure | None]) -> None:
    """Log visualization figures from model_viz to MLflow."""
    mlflow_paths = {
        "classification_report":              "viz/classification_report.png",
        "roc_auc":                            "viz/roc_auc.png",
        "precision_recall_curve":             "viz/precision_recall_curve.png",
        "class_prediction_error":             "viz/class_prediction_error.png",
        "confusion_matrix":                   "viz/confusion_matrix.png",
        "feature_importances":                "viz/feature_importance_gain.png",
        "relative_feature_importances":       "viz/relative_feature_importance_gain.png",
        "feature_importances_weight":         "viz/feature_importance_weight.png",
        "relative_feature_importances_weight": "viz/relative_feature_importance_weight.png",
    }
    for key, path in mlflow_paths.items():
        fig = charts.get(key)
        if fig is not None:
            mlflow.log_figure(fig, path)
            plt.close(fig)


def _build_context_interaction_constraints(feat_cols: list[str]) -> list[list[str]]:
    """Filter CONTEXT_XG_INTERACTION_GROUPS to columns present in the feature matrix.

    XGBClassifier (sklearn API) resolves interaction_constraints by feature name,
    not integer index. Groups with no present columns are silently dropped.
    """
    feat_set = set(feat_cols)
    return [
        [c for c in group if c in feat_set]
        for group in CONTEXT_XG_INTERACTION_GROUPS
        if any(c in feat_set for c in group)
    ]


# ── Data loading ────────────────────────────────────────────────────────────────

def load_data(
    strength: str, study_name: str, model: str = "base_xg"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, float, PandasDataset]:
    """Load and chronologically split processed training data for one strength state."""
    filepath = Path(__file__).parent / "data" / model / "train" / f"{strength}.parquet"

    df = pd.read_parquet(filepath)
    df = df.sort_values("season").reset_index(drop=True)
    season = df.pop("season")

    passthrough = [c for c in PASSTHROUGH_COLS if c in df.columns]
    if passthrough:
        df = df.drop(columns=passthrough)

    pd_dataset = from_pandas(df, source=str(filepath), name=study_name, targets="goal")

    X = df.drop(columns="goal")
    y = df["goal"].copy()

    if model == "base_xg":
        X = X[[c for c in BASE_XG_FEATURE_COLUMNS if c in X.columns]]
    elif model == "context_xg":
        X = X[[c for c in CONTEXT_XG_FEATURE_COLUMNS if c in X.columns]]
    # pred_goal: all remaining columns are talent features (base_xg handled via base_margin)

    # Chronological split: train through 2022-23, test on 2023-24
    train_mask = season <= 20222023
    X_train = apply_fixed_categoricals(X.loc[train_mask].reset_index(drop=True), strength)
    X_test  = apply_fixed_categoricals(X.loc[~train_mask].reset_index(drop=True), strength)
    y_train = y.loc[train_mask].reset_index(drop=True)
    y_test  = y.loc[~train_mask].reset_index(drop=True)

    # empty_against goal rate ~80% → no upweighting needed
    scale_pos_weight = 1.0 if strength == "empty_against" else (y_train == 0).sum() / (y_train == 1).sum()

    return X_train, X_test, y_train, y_test, scale_pos_weight, pd_dataset


# ── Per-model param builders ────────────────────────────────────────────────────

def _params_base_xg(
    trial: optuna.Trial, spw: float, X_train: pd.DataFrame
) -> dict[str, Any]:
    """Optuna param space for base_xg (geometry-only gbtree, no fingerprint risk)."""
    return {
        "objective": "binary:logistic",
        "verbosity": 0,
        "random_state": SEED,
        "n_estimators": N_ESTIMATORS,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "enable_categorical": True,
        "monotone_constraints": {
            col: d for col, d in MONOTONE_CONSTRAINTS.items() if col in X_train.columns
        },
        "max_depth":         trial.suggest_int("max_depth", 3, 6),
        "min_child_weight":  trial.suggest_int("min_child_weight", 20, 200, log=True),
        "max_delta_step":    trial.suggest_int("max_delta_step", 1, 10),
        "scale_pos_weight":  spw,
        "learning_rate":     trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "gamma":             trial.suggest_float("gamma", 0.0, 5.0),
        "lambda":            trial.suggest_float("lambda", 0.1, 10.0, log=True),
        "alpha":             trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        "subsample":         trial.suggest_float("subsample", 0.4, 1.0, step=0.05),
        "colsample_bytree":  trial.suggest_float("colsample_bytree",  0.6, 1.0, step=0.05),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0, step=0.05),
        "colsample_bynode":  trial.suggest_float("colsample_bynode",  0.6, 1.0, step=0.05),
    }


def _params_context_xg(
    trial: optuna.Trial, spw: float, X_train: pd.DataFrame
) -> dict[str, Any]:
    """Optuna param space for context_xg (depth-2 gbtree with flag isolation constraints).

    max_depth=2 is fixed — it is the core structural constraint that prevents multi-flag
    paths. colsample_* params are omitted because column subsampling runs before interaction
    constraints are applied, which can silently degenerate a constraint group.
    """
    constraints = _build_context_interaction_constraints(list(X_train.columns))
    return {
        "objective": "binary:logistic",
        "booster": "gbtree",
        "verbosity": 0,
        "random_state": SEED,
        "n_estimators": N_ESTIMATORS,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "enable_categorical": True,
        "scale_pos_weight": spw,
        "max_depth": 2,
        "interaction_constraints": constraints,
        # Raised regularisation floors vs base_xg: the base_margin anchor means leaf weights
        # are learning residuals from logit_base_xg, not the full sigmoid range. Small-sample
        # states (SH: 36K shots, tiny constraint-group leaves) need lambda >> 1 to prevent
        # the denominator from being dominated by the per-leaf hessian sum alone.
        # gamma > 0 enforces a minimum gain threshold — prevents trivial splits on flag groups.
        # alpha > 0.1 adds L1 sparsity that shrinks small context boosts toward zero.
        # Floor raised from 50 → 100: SH/EF production models landed at the 50 floor,
        # causing low-decile overestimation (model can't reach very low predictions).
        "min_child_weight": trial.suggest_int("min_child_weight", 100, 500, log=True),
        "gamma":            trial.suggest_float("gamma", 1.0, 10.0),
        # Ceiling raised from 100 → 200: EA production lambda was 95.62 (hit the ceiling).
        # Optimal EA regularization may require lambda > 100 for the ~9K-event dataset.
        # Floor raised from 1.0 → 10.0: passing models all had lambda ≥ 8.5 (PP minimum).
        # lambda < 10 produces bimodal output even with mds=1 — flag groups accumulate
        # enough positive weight across iterations to push ~10% of shots to p≈0.60.
        "lambda":           trial.suggest_float("lambda", 10.0, 200.0, log=True),
        "alpha":            trial.suggest_float("alpha", 0.1, 10.0, log=True),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0, step=0.05),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.30, log=True),
        # max_delta_step=1 is fixed — not tunable. mds >= 2 causes bimodal cliff regardless
        # of other params. Fixed here so CV folds match the final model in finalize.py.
        "max_delta_step":   1,
    }


# pred_goal uses the same param space as base_xg — talent features have the same
# relaxed regularisation budget (no fingerprint risk, no monotone constraints needed).
_params_pred_goal = _params_base_xg

_PARAM_BUILDERS: dict[str, Any] = {
    "base_xg":    _params_base_xg,
    "context_xg": _params_context_xg,
    "pred_goal":  _params_pred_goal,
}


# ── CV fold runner ──────────────────────────────────────────────────────────────

def _run_cv_folds(
    params: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    kfold: TimeSeriesSplit,
    bm_train: np.ndarray | None,
    trial: optuna.Trial,
) -> list[dict[str, float]]:
    """Run TimeSeriesSplit CV, log per-fold metrics to MLflow, and prune weak trials.

    Fold models run to full n_estimators (no eval_set → early stopping is inapplicable).
    early_stopping_rounds is stripped from fold_params so XGBoost doesn't warn.
    """
    fold_params = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
    fold_results: list[dict[str, float]] = []

    for fold_idx, (tr_idx, val_idx) in enumerate(kfold.split(X_train)):
        X_tr,  X_val  = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr,  y_val  = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        bm_tr  = bm_train[tr_idx]  if bm_train is not None else None
        bm_val = bm_train[val_idx] if bm_train is not None else None

        fold_m = xgb.XGBClassifier(**fold_params)
        fold_m.fit(X_tr, y_tr, base_margin=bm_tr, verbose=False)
        y_prob = fold_m.predict_proba(X_val, base_margin=bm_val)[:, 1]
        # Use base-rate threshold, not 0.5 — at 6% goal rate almost nothing exceeds 0.5
        # so precision/recall at 0.5 are noise. Base-rate threshold gives the fingerprinting
        # signal: healthy model has recall ~0.6+; fingerprinted model has recall ~0.05.
        y_pred = (y_prob >= float(y_val.mean())).astype(int)

        prauc = sklearn.metrics.average_precision_score(y_val, y_prob)
        n_cls = len(np.unique(y_val))
        result: dict[str, float] = {
            "roc_auc":           sklearn.metrics.roc_auc_score(y_val, y_prob) if n_cls > 1 else float("nan"),
            "average_precision": prauc,
            "f1":                sklearn.metrics.f1_score(y_val, y_pred, zero_division=0),
            "log_loss":          sklearn.metrics.log_loss(y_val, y_prob),
            "precision":         sklearn.metrics.precision_score(y_val, y_pred, zero_division=0),
            "recall":            sklearn.metrics.recall_score(y_val, y_pred, zero_division=0),
            "accuracy":          sklearn.metrics.accuracy_score(y_val, y_pred),
        }
        fold_results.append(result)
        mlflow.log_metrics({f"fold{fold_idx}_{k}": v for k, v in result.items() if not np.isnan(v)})

        # Percentile pruning: store this fold's PR-AUC and prune if bottom-25% across
        # all trials that reached this fold (not just completed ones — avoids selection
        # bias where completed-only medians creep upward over time).
        attr_key = f"fold_{fold_idx}_prauc"
        trial.set_user_attr(attr_key, prauc)
        if fold_idx < kfold.n_splits - 1:
            all_vals = [t.user_attrs[attr_key] for t in trial.study.trials if attr_key in t.user_attrs]
            if len(all_vals) >= 5 and prauc < float(np.percentile(all_vals, 25)):
                raise optuna.TrialPruned()

    return fold_results


# ── Optuna objective ────────────────────────────────────────────────────────────

def _objective(trial: optuna.Trial, data: ExperimentData) -> float:
    """Optuna objective — wraps _objective_body with MLflow run context."""
    warnings.filterwarnings("ignore")
    with mlflow.start_run(run_id=data.parent_info.run_id):
        with mlflow.start_run(nested=True) as current_run:
            trial.set_user_attr("mlflow_run_id",          current_run.info.run_id)
            trial.set_user_attr("mlflow_run_name",         current_run.info.run_name)
            trial.set_user_attr("mlflow_parent_run_id",    data.parent_info.run_id)
            trial.set_user_attr("mlflow_parent_run_name",  data.parent_info.run_name)
            try:
                return _objective_body(trial, data, current_run)
            except optuna.TrialPruned:
                mlflow.set_tag("trial_outcome", "pruned")
                mlflow.set_tag("performance", "pruned")
                mlflow.end_run("FINISHED")
                raise
            except Exception as exc:
                mlflow.set_tag("trial_outcome", "failed")
                mlflow.set_tag("failure_reason", f"{type(exc).__name__}: {str(exc)[:250]}")
                raise


def _objective_body(
    trial: optuna.Trial, data: ExperimentData, current_run: Any
) -> float:
    # pred_goal: logit(context_xg) as base_margin; base_xg col dropped (it IS the base_margin).
    # context_xg: logit_base_xg as base_margin (residual learning from T1 prior); logit_base_xg
    #   stays in the feature matrix so interaction constraint groups remain intact.
    # base_xg: no base_margin.
    if data.model == "pred_goal" and "base_xg" in data.X_train.columns:
        bm_train = logit(data.X_train["base_xg"].to_numpy())
        bm_test  = logit(data.X_test["base_xg"].to_numpy())
        X_train  = data.X_train.drop(columns=["base_xg"])
        X_test   = data.X_test.drop(columns=["base_xg"])
    elif data.model == "context_xg" and "logit_base_xg" in data.X_train.columns:
        bm_train = data.X_train["logit_base_xg"].to_numpy()
        bm_test  = data.X_test["logit_base_xg"].to_numpy()
        X_train  = data.X_train   # logit_base_xg stays in feature matrix
        X_test   = data.X_test
    else:
        bm_train, bm_test = None, None
        X_train, X_test   = data.X_train, data.X_test

    # scale_pos_weight: capped per model type.
    # context_xg cap is 3.0 (not 10.0): the base_margin anchor already shifts gradients toward
    # the T1 prior, so heavy upweighting of goal events amplifies residual leaf values and
    # causes bimodal collapse — especially for small-sample states (SH, EF, EA) where the
    # per-leaf hessian sum is small and lambda must do more work.
    # empty_against is fixed at 1.0 for all models (goal rate ~57-80% — no upweighting needed).
    spw_cap = 3.0 if data.model == "context_xg" else 10.0
    spw_high = min(data.scale_pos_weight, spw_cap) if data.scale_pos_weight > 1.0 else 1.0
    spw = trial.suggest_float("scale_pos_weight", 1.0, spw_high) if spw_high > 1.0 else 1.0

    params = _PARAM_BUILDERS[data.model](trial, spw, X_train)

    # Log params before adding eval_metric (list values aren't valid MLflow param types).
    log_params_dict: dict[str, Any] = {**params}
    for key in ("monotone_constraints", "interaction_constraints"):
        if key in log_params_dict:
            log_params_dict[key] = str(log_params_dict[key])
    mlflow.log_params(log_params_dict)

    # Add eval_metric separately — used only at fit time, not for logging.
    # logloss is last so XGBoost uses it for early stopping (calibration criterion).
    # aucpr is tracked but does not drive early stopping — ranking-only stopping
    # caused bimodal collapse in context_xg (model stopped before calibration settled).
    fit_params = {**params, "eval_metric": ["aucpr", "logloss"]}

    kfold = TimeSeriesSplit(n_splits=CV_TUNE_FOLDS)
    fold_results = _run_cv_folds(fit_params, X_train, data.y_train, kfold, bm_train, trial)

    # Aggregate CV metrics
    cv_summary: dict[str, float] = {}
    for metric in fold_results[0]:
        vals = np.array([r[metric] for r in fold_results])
        cv_summary[f"train_{metric}_mean"] = float(np.nanmean(vals))
        cv_summary[f"train_{metric}_std"]  = float(np.nanstd(vals))
    mlflow.log_metrics(cv_summary)

    evals_df = pd.DataFrame(fold_results)
    evals_df.index = pd.RangeIndex(1, len(evals_df) + 1, name="kfold")
    mlflow.log_text(
        evals_df.to_html(na_rep="", float_format=lambda x: str(round(x, 3))),
        "performance/train_cross_validation.html",
    )

    # Final model — fit on all training data with early stopping on hold-out.
    model = xgb.XGBClassifier(**fit_params)
    model.fit(
        X_train, data.y_train,
        base_margin=bm_train,
        eval_set=[(X_test, data.y_test)],
        base_margin_eval_set=[bm_test] if bm_test is not None else None,
        verbose=False,
    )

    # Log any XGBoost default params not already in the user-specified set.
    extra_params = {k: str(v) for k, v in model.get_xgb_params().items() if k not in params}
    if extra_params:
        mlflow.log_params(extra_params)

    # Upload per-iteration boosting metrics in batches (avoid file descriptor exhaustion).
    ts = int(pd.Timestamp.now().timestamp() * 1000)
    boosting_metrics = [
        Metric(f"boosting_{name}", value, ts, step)
        for name, values in model.evals_result()["validation_0"].items()
        for step, value in enumerate(values)
    ]
    client = mlflow.tracking.MlflowClient()
    for i in range(0, len(boosting_metrics), 1000):
        client.log_batch(current_run.info.run_id, metrics=boosting_metrics[i: i + 1000])

    booster    = model.get_booster()
    feat_names = list(X_train.columns)
    fi_weight  = booster.get_score(importance_type="weight")
    fi_gain    = booster.get_score(importance_type="gain")
    mlflow.log_dict(fi_weight, "artifacts/feature_importance_weight.json")
    mlflow.log_dict(fi_gain,   "artifacts/feature_importance_gain.json")
    fi_df = (
        pd.DataFrame({
            "feature":         feat_names,
            "weight (splits)": [fi_weight.get(f, 0) for f in feat_names],
            "gain (avg)":      [round(fi_gain.get(f, 0.0), 4) for f in feat_names],
        })
        .query("`weight (splits)` > 0")
        .sort_values("weight (splits)", ascending=False)
        .reset_index(drop=True)
    )
    mlflow.log_text(
        fi_df.to_html(index=False, float_format=lambda x: f"{x:.4f}"),
        "artifacts/feature_importance.html",
    )

    y_probs = model.predict_proba(X_test, base_margin=bm_test)[:, 1]
    # Base-rate threshold for threshold-dependent metrics — see _run_cv_folds for rationale.
    y_preds_br = (y_probs >= float(data.y_test.mean())).astype(int)
    degenerate = len(np.unique(y_preds_br)) < 2

    test_metrics = {f"test_{k}": float(v) for k, v in model_metrics(data.y_test, y_preds_br, y_probs).items()}
    mlflow.log_metrics(test_metrics)

    prauc       = test_metrics["test_average_precision"]
    log_loss_val = test_metrics["test_log_loss"]
    performance_tag = compute_performance_tag(prauc, log_loss_val, data.strength)

    # Upload model artifact only for trials above the performance cutoff — uploading
    # every trial exhausts file descriptors after hundreds of runs.
    run_name = current_run.info.run_name
    signature = infer_signature(X_test, y_preds_br)
    logged_model = None
    if not degenerate and performance_tag in ("medium", "high", "very high"):
        model_info = mlflow.xgboost.log_model(
            model, name=run_name, signature=signature, pip_requirements=data.pip_requirements
        )
        logged_model = LoggedModelInput(model_id=model_info.model_id) if model_info.model_id else None

    mlflow.log_input(data.pd_dataset, context="training", model=logged_model)

    mlflow.set_tags({
        "performance":      performance_tag,
        "experiment_name":  data.study_name,
        "experiment_id":    current_run.info.experiment_id,
        "estimator_name":   model.__class__.__name__,
        "parent_id":        data.parent_info.run_id,
        "parent_name":      data.parent_info.run_name,
        "level":            "child",
        "optuna_trial_num": str(trial.number),
    })

    class_report = sklearn.metrics.classification_report(
        data.y_test, y_preds_br, labels=[0, 1], target_names=["no goal", "goal"],
        output_dict=True, zero_division=0,
    )
    mlflow.log_text(
        pd.DataFrame(class_report).to_html(na_rep="", float_format=lambda x: str(round(x, 3))),
        "performance/test_classification_report.html",
    )

    goals_mask = data.y_test.to_numpy() == 1
    shots_mask = ~goals_mask
    pct_labels = [5, 10, 25, 50, 75, 90, 95]
    pred_dist_df = pd.DataFrame({
        "percentile": pct_labels,
        "all_shots":  np.percentile(y_probs, pct_labels).round(4),
        "goals":      np.percentile(y_probs[goals_mask], pct_labels).round(4) if goals_mask.sum() > 0 else [float("nan")] * len(pct_labels),
        "non_goals":  np.percentile(y_probs[shots_mask], pct_labels).round(4) if shots_mask.sum() > 0 else [float("nan")] * len(pct_labels),
    })
    mlflow.log_text(pred_dist_df.to_html(index=False), "artifacts/prediction_distribution.html")

    if not degenerate and performance_tag in ("medium", "high", "very high"):
        log_viz(model_viz(model, X_train, data.y_train, X_test, data.y_test, run_name, base_margin=bm_test))

    mlflow.set_tag("trial_outcome", "completed")

    # context_xg objective: composite score (same formula as screen_trials) so Optuna
    # gets a calibration signal and actively steers toward well-regularised configs.
    # base_xg and pred_goal use raw hold-out PR-AUC — no bimodal cliff risk there.
    if data.model == "context_xg":
        y_np = data.y_test.to_numpy()
        platt = LogisticRegression(C=1.0, max_iter=1000).fit(y_probs.reshape(-1, 1), y_np)
        prob_platt = platt.predict_proba(y_probs.reshape(-1, 1))[:, 1]
        cal_prauc = float(average_precision_score(y_np, prob_platt))
        cal_ll    = float(sklearn_log_loss(y_np, prob_platt))
        ece       = calculate_ece(y_np, prob_platt)
        iso = IsotonicRegression(out_of_bounds="clip").fit(y_probs, y_np)
        prob_iso  = np.clip(iso.predict(y_probs), 1e-7, 1 - 1e-7)
        iso_ll    = float(sklearn_log_loss(y_np, prob_iso))
        structural_flaw_penalty = max(0.0, cal_ll - iso_ll)

        bin_edges = np.percentile(y_probs, np.linspace(0, 100, 11))
        bin_edges[0] -= 1e-9
        bin_ids = np.clip(np.digitize(y_probs, bin_edges) - 1, 0, 9)
        cal_rows = []
        for b in range(10):
            mask = bin_ids == b
            if not mask.any():
                continue
            cal_rows.append({
                "decile":      b + 1,
                "n":           int(mask.sum()),
                "actual_rate": round(float(y_np[mask].mean()), 4),
                "raw_prob":    round(float(y_probs[mask].mean()), 4),
                "platt_prob":  round(float(prob_platt[mask].mean()), 4),
                "iso_prob":    round(float(prob_iso[mask].mean()), 4),
                "raw_p10":     round(float(np.percentile(y_probs[mask], 10)), 4),
                "raw_p90":     round(float(np.percentile(y_probs[mask], 90)), 4),
            })
        mlflow.log_text(
            pd.DataFrame(cal_rows).to_html(index=False),
            "artifacts/calibration_deciles.html",
        )

        # Distribution penalty on raw probs — catches asymmetric bimodal that Platt masks.
        # SHOT p90 / base_rate > 2.5 means non-goal shots are reaching high-probability territory.
        # Weight 0.02: at ratio=10.73 (bimodal ES), penalty=0.165; at ratio=1.79 (passing ES), 0.
        # Gap of ~0.30 composite points is unambiguous signal for TPE.
        shot_mask = y_np == 0
        base_rate = float(y_np.mean())
        if shot_mask.sum() > 0 and base_rate > 0:
            shot_p90 = float(np.quantile(y_probs[shot_mask], 0.90))
            dist_ratio = shot_p90 / base_rate
        else:
            shot_p90 = 0.0
            dist_ratio = 0.0
        distribution_penalty = max(0.0, dist_ratio - 2.5) * 0.02

        composite = cal_prauc - (_ECE_WEIGHT * ece) - (_STRUCTURAL_FLAW_WEIGHT * structural_flaw_penalty) - distribution_penalty
        mlflow.log_metrics({
            "objective_cal_prauc":             cal_prauc,
            "objective_ece":                   ece,
            "objective_platt_ll":              cal_ll,
            "objective_iso_ll":               iso_ll,
            "objective_structural_penalty":    structural_flaw_penalty,
            "objective_distribution_ratio":    dist_ratio,
            "objective_distribution_penalty":  distribution_penalty,
            "objective_composite":             composite,
        })
        return composite

    return prauc


# ── Study management ────────────────────────────────────────────────────────────

def tune_model(
    strength: str,
    version: str,
    storage: optuna.storages.RDBStorage,
    max_trials: int,
    run: str | None = None,
    model: str = "base_xg",
    n_startup_trials: int = 100,
) -> optuna.Study:
    """Create (or resume) an Optuna study and run Optuna + MLflow tuning."""
    model_suffix = model.replace("_xg", "")
    study_name = f"{strength}-{version}-{model_suffix}"

    mlflow.enable_system_metrics_logging()
    experiment = mlflow.set_experiment(study_name)
    experiment_id = experiment.experiment_id

    X_train, X_test, y_train, y_test, scale_pos_weight, pd_dataset = load_data(
        strength, study_name, model=model
    )

    tags = {
        "experiment_name": study_name,
        "experiment_id":   experiment_id,
        "level":           "parent",
        "model":           model,
    }
    if run is not None:
        parent_ctx = mlflow.start_run(run_id=run)
    else:
        parent_ctx = mlflow.start_run(tags=tags)

    with parent_ctx as parent_run:
        parent_info = parent_run.info

    pip_requirements = mlflow.models.infer_pip_requirements(None, "xgboost")

    data = ExperimentData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        scale_pos_weight=scale_pos_weight,
        pd_dataset=pd_dataset,
        study_name=study_name,
        parent_info=parent_info,
        pip_requirements=pip_requirements,
        model=model,
        strength=strength,
    )

    sampler = optuna.samplers.TPESampler(multivariate=True, seed=SEED, n_startup_trials=n_startup_trials)
    try:
        study = optuna.create_study(
            study_name=study_name,
            sampler=sampler,
            load_if_exists=True,
            storage=storage,
            direction="maximize",
        )
    except optuna.exceptions.StorageInternalError:
        study = optuna.load_study(study_name=study_name, storage=storage)

    study.set_metric_names(["average_precision"])
    study.optimize(
        functools.partial(_objective, data=data),
        n_trials=max_trials,
        show_progress_bar=True,
    )

    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]
    with mlflow.start_run(run_id=parent_info.run_id):
        mlflow.set_tags({
            "optuna_study_id":    str(study._study_id),
            "optuna_study_name":  study_name,
            "optuna_n_trials":    str(len(study.trials)),
        })
        if completed:
            best = max(completed, key=lambda t: t.value)
            mlflow.log_metrics({"best_pr_auc": best.value})
            mlflow.log_params({"best_" + k: str(v) for k, v in best.params.items()})
            mlflow.set_tag("best_trial_num", str(best.number))

    return study


# ── CLI ─────────────────────────────────────────────────────────────────────────

def load_optuna_storage() -> optuna.storages.RDBStorage:
    """Create an RDBStorage from DB_* environment variables."""
    return optuna.storages.RDBStorage(
        url=(
            f"postgresql+psycopg2://{os.environ['DB_USER']}:{os.environ['DB_PASSWORD']}"
            f"@{os.environ.get('DB_HOST')}:{os.environ['DB_PORT']}/{os.environ['DB_NAME']}"
        ),
        skip_compatibility_check=True,
    )


def main():
    """Main function."""
    parser = argparse.ArgumentParser(prog="xG Training", description="Tune an xG model tier with Optuna + MLflow.")
    parser.add_argument("--strength", "-s", type=str, required=True, choices=STRENGTHS)
    parser.add_argument("--version", "-v", type=str, required=True)
    parser.add_argument("--model", "-m", type=str, required=False, default="base_xg", choices=MODELS)
    parser.add_argument("--run", "-r", type=str, required=False)
    parser.add_argument("--trials", "-t", type=int, required=False, default=100)
    parser.add_argument("--startup-trials", "-st", type=int, required=False, default=100,
                        help="Number of random startup trials before TPE model kicks in (default: 100).")
    parser.add_argument("--delete", "-d", action="store_true", help="Delete the Optuna study and exit.")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    load_dotenv()

    storage = load_optuna_storage()

    if args.delete:
        study_name = f"{args.strength}-{args.version}-{args.model.replace('_xg', '')}"
        optuna.delete_study(study_name=study_name, storage=storage)
    else:
        tune_model(
            strength=args.strength,
            version=args.version,
            storage=storage,
            max_trials=args.trials,
            run=args.run,
            model=args.model,
            n_startup_trials=args.startup_trials,
        )

if __name__ == "__main__":
    main()
