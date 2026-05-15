
"""Retrain best context_xg gbtree model on all training data and freeze.

Reads the best Optuna trial (by PR-AUC) from the given study, retrains a
gbtree XGBoost model (depth=2, flag isolation constraints) with logit_base_xg
as both base_margin (residual learning from T1 prior) and a learnable feature
(preserving interaction constraint groups), calibrates with pooled OOF + hold-out
Platt scaling, and writes:

  models/context_xg/{strength}.ubj           — frozen gbtree booster
  models/context_xg/{strength}_calibrator.joblib
  models/context_xg/{strength}_oof.parquet   — calibrated OOF predictions
  data/context_xg/scored/{strength}.parquet  — all train+hold_out shots

Usage:
    python context_xg/finalize.py --strength even_strength --version 1.0.0 --no-log
    python context_xg/finalize.py --all --version 1.0.0 --no-log
"""

import argparse
import warnings
from pathlib import Path

import joblib
import mlflow
import mlflow.xgboost
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from datetime import datetime, timezone

from dotenv import load_dotenv
from mlflow.models.signature import infer_signature
from sklearn.linear_model import LogisticRegression

from chickenstats_xg.v1.config import (
    CONTEXT_XG_FEATURE_COLUMNS,
    CONTEXT_XG_INTERACTION_GROUPS,
    CV_CALIBRATE_FOLDS,
    EARLY_STOPPING_ROUNDS,
    N_ESTIMATORS,
    PASSTHROUGH_COLS,
    SEED,
    STRENGTHS,
    compute_performance_tag,
)
from chickenstats_xg.v1.experiments import (
    _apply_fixed_categoricals,
    _build_context_interaction_constraints,
    compute_oof_predictions,
    load_optuna_storage,
    log_viz,
    model_metrics,
    model_viz,
    save_model_metadata,
    screen_trials,
    select_top_trials,
)

NON_FEATURE_COLS = ["goal", "season", "base_xg"] + PASSTHROUGH_COLS


def _split_df(df: pd.DataFrame, strength: str) -> tuple[pd.DataFrame, pd.Series, np.ndarray | None]:
    """Return (X_context_features, y, base_margin). logit_base_xg is in both X and base_margin."""
    y = df["goal"].copy()
    feat_cols = [c for c in CONTEXT_XG_FEATURE_COLUMNS if c in df.columns]
    X = df[feat_cols].copy()
    X = _apply_fixed_categoricals(X, strength)
    bm = df["logit_base_xg"].to_numpy() if "logit_base_xg" in df.columns else None
    return X, y, bm



def _finalize_one(
    strength: str,
    version: str,
    storage: optuna.storages.RDBStorage,
    no_log: bool,
    top_n: int = 15,
) -> None:
    study_name = f"{strength}-{version}-context"
    study = optuna.load_study(study_name=study_name, storage=storage)

    data_dir = Path(__file__).parent.parent / "data" / "context_xg"
    train_df = pd.read_parquet(data_dir / "train" / f"{strength}.parquet")
    hold_out_df = pd.read_parquet(data_dir / "hold_out" / f"{strength}.parquet")

    X_train, y_train, bm_train = _split_df(train_df, strength)
    X_hold_out, y_hold_out, bm_hold_out = _split_df(hold_out_df, strength)

    fixed_params = {
        "objective": "binary:logistic",
        "booster": "gbtree",
        "verbosity": 0,
        "random_state": SEED,
        "n_estimators": N_ESTIMATORS,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "enable_categorical": True,
        "max_depth": 2,
        "interaction_constraints": _build_context_interaction_constraints(list(X_train.columns)),
        # logloss is last so XGBoost uses it for early stopping (calibration criterion).
        # aucpr-only stopping caused bimodal collapse — model stopped when ranking plateaued,
        # before calibration settled. max_delta_step=1 is a fixed structural constraint;
        # it is also set in _params_context_xg() so CV folds match this value.
        "eval_metric": ["aucpr", "logloss"],
        "max_delta_step": 1,
    }

    print(f"  [{strength}] screening top {top_n} trials by calibrated hold-out log loss...")
    candidates = select_top_trials(study, top_n)
    best_params, best_trial_num, best_trial = screen_trials(
        candidates, fixed_params, X_train, y_train, X_hold_out, y_hold_out,
        bm_train=bm_train, bm_hold_out=bm_hold_out,
    )

    # max_delta_step=1 is a structural constraint, not a tunable param.
    # Strip it so fixed_params wins — trial values of 2-5 cause bimodal cliff.
    best_params.pop("max_delta_step", None)

    params = {**fixed_params, **best_params}

    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_hold_out, y_hold_out)],
        base_margin=bm_train,
        base_margin_eval_set=[bm_hold_out] if bm_hold_out is not None else None,
        verbose=False,
    )

    # OOF loop — fold models trained to model.best_iteration (same tree count
    # as the final model, no early stopping). Matching the tree count aligns
    # the fold-model probability scale with the final model, reducing the
    # distribution mismatch that caused decile 7-8 over-prediction when the
    # calibrator was anchored to hold-out data alone.
    # These OOF predictions also serve as honest training-era predictions for
    # _oof.parquet (score.py replaces in-sample predictions with them).
    oof_prob, oof_mask = compute_oof_predictions(
        model, X_train, y_train, params, CV_CALIBRATE_FOLDS, bm_train=bm_train
    )

    # Pool training OOF + hold-out for Platt calibration. Pooling spans the
    # full 15-season range, preventing the temporal mismatch that arises when
    # calibrating on only the 2-season hold-out. Platt (logistic regression,
    # 2 parameters) is stable at any dataset size and its sigmoid ceiling
    # (max < 1.0) is desirable: logit(1.0) = +inf would break pred_goal's
    # base_margin. Issue 5's Platt ceiling concern applies to base_xg EA
    # (~99% top-decile actual rate), not context_xg EA (~89%).
    raw_hold_out = model.predict_proba(X_hold_out, base_margin=bm_hold_out)[:, 1]
    calib_probs  = np.concatenate([oof_prob[oof_mask], raw_hold_out])
    calib_labels = np.concatenate([y_train.to_numpy()[oof_mask], y_hold_out.to_numpy()])
    calibrator = LogisticRegression(C=1.0, max_iter=1000).fit(
        calib_probs.reshape(-1, 1), calib_labels,
    )

    # Training shots use OOF probs (honest, not in-sample), falling back to
    # full-model probs for the earliest fold not covered by OOF.
    raw_train = np.where(oof_mask, oof_prob, model.predict_proba(X_train, base_margin=bm_train)[:, 1])
    train_context_xg = calibrator.predict_proba(raw_train.reshape(-1, 1))[:, 1]
    hold_out_context_xg = calibrator.predict_proba(raw_hold_out.reshape(-1, 1))[:, 1]

    scored_train = train_df.reset_index(drop=True).assign(context_xg=train_context_xg)
    scored_hold_out = hold_out_df.reset_index(drop=True).assign(context_xg=hold_out_context_xg)
    all_df = (
        pd.concat([scored_train, scored_hold_out], ignore_index=True)
        .sort_values(["season", "game_id", "event_idx"])
        .reset_index(drop=True)
    )

    scored_dir = Path(__file__).parent.parent / "data" / "context_xg" / "scored"
    scored_dir.mkdir(parents=True, exist_ok=True)
    all_df.to_parquet(scored_dir / f"{strength}.parquet", index=False)

    models_dir = Path(__file__).parent.parent / "models" / "context_xg"
    strength_dir = models_dir / strength
    strength_dir.mkdir(parents=True, exist_ok=True)
    model.get_booster().save_model(str(strength_dir / "model.ubj"))
    joblib.dump(calibrator, strength_dir / "calibrator.joblib")

    oof_df = train_df[["game_id", "event_idx"]].reset_index(drop=True).copy()
    oof_df["context_xg"] = np.where(
        oof_mask,
        calibrator.predict_proba(oof_prob.reshape(-1, 1))[:, 1],
        np.nan,
    )
    oof_df.to_parquet(strength_dir / "oof.parquet", index=False)
    print(f"  [{strength}] saved → {strength_dir}/model.ubj + calibrator.joblib + oof.parquet + scored parquet")

    save_model_metadata(
        models_dir, strength, "context_xg", version, study_name, best_trial_num,
        trial_value=best_trial.value if best_trial is not None else None,
        trial=best_trial,
    )

    if no_log:
        return

    mlflow.enable_system_metrics_logging()

    y_probs = model.predict_proba(X_hold_out, base_margin=bm_hold_out)[:, 1]
    y_preds = (y_probs >= 0.5).astype(int)
    hold_out_metrics = {f"hold_out_{k}": float(v) for k, v in model_metrics(y_hold_out, y_preds, y_probs).items()}
    signature = infer_signature(X_hold_out, y_probs)
    run_name = f"context_xg-{strength}-final"

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
        mlflow.log_params(params)
        mlflow.log_metric("best_iteration", float(model.best_iteration or 0))
        mlflow.log_metrics(hold_out_metrics)
        log_viz(model_viz(model, X_train, y_train, X_hold_out, y_hold_out, run_name, base_margin=bm_hold_out))
        model_info = mlflow.xgboost.log_model(model, name=run_name, signature=signature)
        mlflow.register_model(model_uri=model_info.model_uri, name=f"context_xg_{strength}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain and freeze best context_xg gbtree model.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--strength", "-s", type=str, choices=STRENGTHS)
    group.add_argument("--all", "-a", action="store_true")
    parser.add_argument("--version", "-v", type=str, required=True)
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument(
        "--top-n", "-n", type=int, default=15,
        help="Number of top CV PR-AUC trials to screen by calibrated hold-out log loss (default: 15).",
    )
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    load_dotenv()

    storage = load_optuna_storage()

    strengths_to_run = STRENGTHS if args.all else [args.strength]
    for strength in strengths_to_run:
        print(f"Finalizing {strength}...")
        _finalize_one(strength, args.version, storage, no_log=args.no_log, top_n=args.top_n)


if __name__ == "__main__":
    main()
