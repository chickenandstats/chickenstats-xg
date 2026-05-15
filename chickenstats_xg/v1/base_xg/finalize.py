
"""Retrain best base_xg model on all training data, score all historical PBP, and freeze.

Reads the best Optuna trial (by PR-AUC) from the given study, retrains on the full
training parquet, scores both train and hold_out shots, then writes:

  data/base_xg/scored/{strength}.parquet  — all shots with base_xg appended
  data/base_xg/models/{strength}.ubj      — frozen XGBoost booster

Usage:
    python base_xg/finalize.py --strength even_strength --version v1
    python base_xg/finalize.py --strength even_strength --version v1 --no-log
"""

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import mlflow
import mlflow.xgboost
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from datetime import datetime, timezone

from dotenv import load_dotenv
from mlflow.models.signature import infer_signature

from chickenstats_xg.v1.config import (
    BASE_XG_FEATURE_COLUMNS,
    BASE_XG_INTERACTION_GROUPS,
    CV_CALIBRATE_FOLDS,
    EARLY_STOPPING_ROUNDS,
    MONOTONE_CONSTRAINTS,
    N_ESTIMATORS,
    PASSTHROUGH_COLS,
    SEED,
    STRENGTHS,
)
from chickenstats_xg.v1.experiments import (
    load_optuna_storage,
    log_viz,
    model_metrics,
    model_viz,
)
from chickenstats_xg.v1.utils.artifacts import (
    load_model_artifacts,
    params_from_run_name,
    save_model_artifacts,
    save_model_metadata,
)
from chickenstats_xg.v1.utils.finalize_utils import (
    MAX_DEPTH_CAP,
    compute_oof_predictions,
    screen_trials,
    select_top_trials,
)
from chickenstats_xg.v1.utils.calibration import IsotonicCalibrator
from chickenstats_xg.v1.utils.transforms import apply_fixed_categoricals

NON_FEATURE_COLS = ["goal", "season"] + PASSTHROUGH_COLS


def _split_df(df: pd.DataFrame, strength: str) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X_features, y) selecting only BASE_XG_FEATURE_COLUMNS."""
    y = df["goal"].copy()
    feat_cols = [c for c in BASE_XG_FEATURE_COLUMNS if c in df.columns]
    X = apply_fixed_categoricals(df[feat_cols], strength)
    return X, y


def _interaction_constraints(columns: list[str]) -> list[list[str]]:
    """Filter BASE_XG_INTERACTION_GROUPS to features present in the training matrix."""
    return [
        [f for f in group if f in columns]
        for group in BASE_XG_INTERACTION_GROUPS
        if any(f in columns for f in group)
    ]



def _finalize_one(
    strength: str,
    version: str,
    storage: optuna.storages.RDBStorage,
    no_log: bool,
    run_name: str | None = None,
    top_n: int = 15,
) -> None:
    """Retrain, score with OOF, and save artifacts for a single strength state."""
    study_name = f"{strength}-{version}-base"
    study = optuna.load_study(study_name=study_name, storage=storage)

    data_dir = Path(__file__).parent.parent / "data" / "base_xg"
    train_df = pd.read_parquet(data_dir / "train" / f"{strength}.parquet")
    hold_out_df = pd.read_parquet(data_dir / "hold_out" / f"{strength}.parquet")

    X_train, y_train = _split_df(train_df, strength)
    X_hold_out, y_hold_out = _split_df(hold_out_df, strength)

    fixed_params = {
        "objective": "binary:logistic",
        "verbosity": 0,
        "random_state": SEED,
        "n_estimators": N_ESTIMATORS,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "enable_categorical": True,
        # logloss last so XGBoost uses it for early stopping (calibration criterion).
        # aucpr-only stopping caused bimodal collapse in context_xg; same fix applied here.
        "eval_metric": ["aucpr", "logloss"],
        "monotone_constraints": {
            col: direction for col, direction in MONOTONE_CONSTRAINTS.items() if col in X_train.columns
        },
        "interaction_constraints": _interaction_constraints(list(X_train.columns)),
    }

    if run_name:
        best_params, best_trial_num = params_from_run_name(run_name, study_name, study)
        best_trial = next((t for t in study.trials if t.number == best_trial_num), None)
        print(f"  [{strength}] using params from run {run_name!r} (trial {best_trial_num})")
    else:
        print(f"  [{strength}] screening top {top_n} trials by calibrated hold-out log loss...")
        candidates = select_top_trials(study, top_n, max_depth_cap=MAX_DEPTH_CAP)
        best_params, best_trial_num, best_trial = screen_trials(
            candidates, fixed_params, X_train, y_train, X_hold_out, y_hold_out,
            max_depth_cap=MAX_DEPTH_CAP,
        )

    params = {**fixed_params, **best_params}
    params["max_depth"] = min(params.get("max_depth", 6), MAX_DEPTH_CAP)

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_hold_out, y_hold_out)], verbose=False)

    # OOF predictions — unbiased estimates for training shots, used to fit the calibrator
    # and written to the scored parquet so pred_goal/process_data.py gets unbiased base_margin.
    oof_prob, oof_mask = compute_oof_predictions(model, X_train, y_train, params, CV_CALIBRATE_FOLDS)

    # Geometry-only base_xg has no memorisation risk, so isotonic calibration is safe
    # for all strengths. empty_against still uses IsotonicCalibrator (genuine near-certain
    # goals at the high end). All others use Platt (logistic) for its sigmoid ceiling,
    # which prevents the calibrator from mapping any raw prediction to 1.0.
    oof_X = oof_prob[oof_mask].reshape(-1, 1)
    if strength == "empty_against":
        calibrator = IsotonicCalibrator().fit(oof_X, y_train.to_numpy()[oof_mask])
    else:
        calibrator = LogisticRegression(C=1.0, max_iter=1000).fit(oof_X, y_train.to_numpy()[oof_mask])

    # Calibrated predictions: OOF where available, final model fallback for earliest fold
    raw_train = np.where(oof_mask, oof_prob, model.predict_proba(X_train)[:, 1])
    train_base_xg = calibrator.predict_proba(raw_train.reshape(-1, 1))[:, 1]
    hold_out_base_xg = calibrator.predict_proba(model.predict_proba(X_hold_out)[:, 1].reshape(-1, 1))[:, 1]

    scored_train = train_df.reset_index(drop=True).assign(base_xg=train_base_xg)
    scored_hold_out = hold_out_df.reset_index(drop=True).assign(base_xg=hold_out_base_xg)
    all_df = (
        pd.concat([scored_train, scored_hold_out], ignore_index=True)
        .sort_values(["season", "game_id", "period", "period_seconds"])
        .reset_index(drop=True)
    )

    scored_dir = data_dir / "scored"
    scored_dir.mkdir(parents=True, exist_ok=True)
    all_df.to_parquet(scored_dir / f"{strength}.parquet", index=False)

    models_dir = Path(__file__).parent.parent / "models" / "base_xg"
    strength_dir = models_dir / strength
    save_model_artifacts(model, calibrator, strength_dir, train_df, oof_prob, oof_mask, "base_xg", params=params)
    print(f"  [{strength}] saved → {strength_dir}/model.ubj + calibrator.joblib + oof.parquet + scored parquet")

    save_model_metadata(
        models_dir, strength, "base_xg", version, study_name, best_trial_num,
        trial_value=best_trial.value if best_trial is not None else None,
        trial=best_trial,
    )

    if no_log:
        return

    mlflow.enable_system_metrics_logging()

    y_probs = model.predict_proba(X_hold_out)[:, 1]
    y_preds = (y_probs >= 0.5).astype(int)
    hold_out_metrics = {f"hold_out_{k}": float(v) for k, v in model_metrics(y_hold_out, y_preds, y_probs).items()}
    signature = infer_signature(X_hold_out, y_probs)
    run_name = f"base_xg-{strength}-final"

    tuning_run_id = best_trial.user_attrs.get("mlflow_run_id") if best_trial else None
    if tuning_run_id:
        run_ctx = mlflow.start_run(run_id=tuning_run_id)
    else:
        mlflow.set_experiment(study_name)
        run_ctx = mlflow.start_run(
            tags={"type": "finalize", "strength": strength, "version": version, "best_trial": str(best_trial_num)}
        )

    with run_ctx:
        mlflow.set_tags({
            "finalized": "true",
            "finalized_at": datetime.now(timezone.utc).isoformat(),
            "finalized_version": version,
        })
        mlflow.log_params({**params, "monotone_constraints": str(params["monotone_constraints"])})
        mlflow.log_metric("best_iteration", float(model.best_iteration or 0))
        mlflow.log_metrics(hold_out_metrics)
        log_viz(model_viz(model, X_train, y_train, X_hold_out, y_hold_out, run_name))
        model_info = mlflow.xgboost.log_model(model, name=run_name, signature=signature)
        mlflow.register_model(model_uri=model_info.model_uri, name=f"base_xg_{strength}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain and freeze best base_xg model")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--strength", "-s", type=str, choices=STRENGTHS, help="Single strength state to finalize.")
    group.add_argument("--all", "-a", action="store_true", help="Finalize all 5 strength states in sequence.")
    parser.add_argument("--version", "-v", type=str, required=True)
    parser.add_argument("--run", "-r", type=str, default=None, help="MLflow run name to use instead of Optuna auto-selection. Cannot be combined with --all.")
    parser.add_argument("--no-log", action="store_true", help="Skip all MLflow logging — just retrain, score, and save files.")
    parser.add_argument(
        "--top-n", "-n", type=int, default=15,
        help="Number of top CV PR-AUC trials to screen by calibrated hold-out log loss (default: 15). Ignored when --run is specified.",
    )
    args = parser.parse_args()

    if args.run and args.all:
        parser.error("--run cannot be combined with --all — specify a single --strength when pinning a run.")

    warnings.filterwarnings("ignore")
    load_dotenv()

    storage = load_optuna_storage()

    strengths_to_run = STRENGTHS if args.all else [args.strength]

    for strength in strengths_to_run:
        print(f"Finalizing {strength}...")
        _finalize_one(strength, args.version, storage, no_log=args.no_log, run_name=args.run, top_n=args.top_n)


if __name__ == "__main__":
    main()