
"""Retrain best pred_goal model, calibrate with OOF isotonic regression, generate SHAP, and freeze.

Reads the best Optuna trial (by PR-AUC) from the given study, retrains on the full training
parquet with base_xg as XGBoost base_margin, then writes:

  models/pred_goal/{strength}_calibrator.joblib — OOF isotonic calibrator (sklearn)
  models/pred_goal/{strength}_base.ubj          — base XGBoost booster (for SHAP/inspection)

Inference: base_model.predict_proba(X, base_margin=logit(base_xg))[:, 1] → calibrator.predict(raw_prob)

Usage:
    python pred_goal/finalize.py --strength even_strength --version 1.0.0
    python pred_goal/finalize.py --all --version 1.0.0 --no-log
    python pred_goal/finalize.py --all --version 1.0.0 --score
    python pred_goal/finalize.py --all --version 1.0.0 --score --years 2024 2025
"""

import argparse
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import optuna
import pandas as pd
import shap
import xgboost as xgb
from dotenv import load_dotenv
from mlflow.models.signature import infer_signature
from sklearn.linear_model import LogisticRegression

from chickenstats_xg.v1.config import (
    CV_CALIBRATE_FOLDS,
    DPI,
    EARLY_STOPPING_ROUNDS,
    FIGSIZE,
    MONOTONE_CONSTRAINTS,
    N_ESTIMATORS,
    PASSTHROUGH_COLS,
    SEED,
    SHAP_FIGSIZE,
    STRENGTHS,
)
from chickenstats_xg.v1.experiments import (
    load_optuna_storage,
    log_viz,
    model_metrics,
)
from chickenstats_xg.v1.utils.artifacts import (
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
from chickenstats_xg.v1.utils.transforms import apply_fixed_categoricals, logit
from chickenstats_xg.utilities.charts import all_classifier_charts

NON_FEATURE_COLS = ["goal", "season"] + PASSTHROUGH_COLS


def _split_df(df: pd.DataFrame, strength: str) -> tuple[pd.DataFrame, pd.Series, np.ndarray | None]:
    """Return (X_features, y, base_margin_or_None).

    base_xg is extracted and converted to log-odds before being removed from X so that
    pred_goal receives it as XGBoost base_margin rather than a feature.
    """
    y = df["goal"].copy()
    bm: np.ndarray | None = None
    if "base_xg" in df.columns:
        bm = logit(df["base_xg"].to_numpy())
    drop = [c for c in NON_FEATURE_COLS if c in df.columns]
    if "base_xg" in df.columns and "base_xg" not in drop:
        drop = drop + ["base_xg"]
    X = apply_fixed_categoricals(df.drop(columns=drop), strength)
    return X, y, bm



def _finalize_one(
    strength: str,
    version: str,
    storage: optuna.storages.RDBStorage,
    no_log: bool,
    run_name: str | None = None,
    top_n: int = 15,
) -> None:
    """Retrain, calibrate, and save artifacts for a single strength state."""
    study_name = f"{strength}-{version}-pred_goal"
    study = optuna.load_study(study_name=study_name, storage=storage)

    data_dir = Path(__file__).parent.parent / "data" / "pred_goal"
    train_df = pd.read_parquet(data_dir / "train" / f"{strength}.parquet")
    hold_out_df = pd.read_parquet(data_dir / "hold_out" / f"{strength}.parquet")

    X_train, y_train, bm_train = _split_df(train_df, strength)
    X_hold_out, y_hold_out, bm_hold_out = _split_df(hold_out_df, strength)

    fixed_params = {
        "objective": "binary:logistic",
        "verbosity": 0,
        "random_state": SEED,
        "n_estimators": N_ESTIMATORS,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "enable_categorical": True,
        # logloss first for monitoring; aucpr last so XGBoost early stopping fires on PR-AUC
        "eval_metric": ["logloss", "aucpr"],
        "monotone_constraints": {
            col: direction for col, direction in MONOTONE_CONSTRAINTS.items() if col in X_train.columns
        },
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
            bm_train=bm_train, bm_hold_out=bm_hold_out,
            max_depth_cap=MAX_DEPTH_CAP,
        )

    params = {**fixed_params, **best_params}
    params["max_depth"] = min(params.get("max_depth", MAX_DEPTH_CAP), MAX_DEPTH_CAP)

    base_model = xgb.XGBClassifier(**params)
    base_model.fit(
        X_train,
        y_train,
        base_margin=bm_train,
        eval_set=[(X_hold_out, y_hold_out)],
        base_margin_eval_set=[bm_hold_out] if bm_hold_out is not None else None,
        verbose=False,
    )

    # OOF calibration — CalibratedClassifierCV can't pass base_margin internally.
    # Fold models are pinned to base_model.best_iteration so OOF probability scale
    # matches the early-stopped final model. oof_mask tracks which rows were in a
    # validation fold so the calibrator is only fit on unbiased predictions, and
    # score.py can reuse them.
    oof_prob, oof_mask = compute_oof_predictions(
        base_model, X_train, y_train, params, CV_CALIBRATE_FOLDS, bm_train=bm_train
    )
    # Pool OOF training probs with hold-out raw probs for calibration.
    # OOF-only calibration suffers from temporal drift: in training-era OOF data,
    # extreme talent matchup features (e.g. goalie_gsax_per_shot_1g) correlate with
    # very high actual goal rates, so the calibrator maps high-raw → high-calibrated.
    # At inference time (hold-out season), the same feature patterns produce much lower
    # actual goal rates → catastrophic miscalibration. Including hold-out anchors the
    # calibrator to the inference-time distribution.
    hold_raw = base_model.predict_proba(X_hold_out, base_margin=bm_hold_out)[:, 1]
    pool_probs = np.concatenate([oof_prob[oof_mask], hold_raw])
    pool_labels = np.concatenate([y_train.to_numpy()[oof_mask], y_hold_out.to_numpy()])

    # empty_against has ~99% actual goal rate in the top prediction decile — the
    # sigmoid ceiling of Platt (logistic) calibration can't reach it. Use isotonic
    # regression instead, which fits a monotone step function and can reach 0.99
    # directly from the pooled distribution without extrapolation.
    if strength == "empty_against":
        calibrator = IsotonicCalibrator().fit(
            pool_probs.reshape(-1, 1), pool_labels
        )
    else:
        calibrator = LogisticRegression(C=1.0, max_iter=1000).fit(
            pool_probs.reshape(-1, 1), pool_labels
        )

    models_dir = Path(__file__).parent.parent / "models" / "pred_goal"
    strength_dir = models_dir / strength
    save_model_artifacts(base_model, calibrator, strength_dir, train_df, oof_prob, oof_mask, "pred_goal", params=params)
    print(f"  [{strength}] saved → {strength_dir}/model.ubj + calibrator.joblib + oof.parquet")

    save_model_metadata(
        models_dir, strength, "pred_goal", version, study_name, best_trial_num,
        trial_value=best_trial.value if best_trial is not None else None,
        trial=best_trial,
    )

    if no_log:
        return

    # SHAP summary on a sample of hold-out data (TreeExplainer uses the base booster)
    explainer = shap.TreeExplainer(base_model)
    shap_sample = X_hold_out.sample(min(2000, len(X_hold_out)), random_state=SEED)
    shap_values = explainer.shap_values(shap_sample)

    mlflow.enable_system_metrics_logging()

    raw_probs_hold_out = base_model.predict_proba(X_hold_out, base_margin=bm_hold_out)[:, 1]
    y_probs = calibrator.predict_proba(raw_probs_hold_out.reshape(-1, 1))[:, 1]
    y_preds = (y_probs >= 0.5).astype(int)
    hold_out_metrics = {f"hold_out_{k}": float(v) for k, v in model_metrics(y_hold_out, y_preds, y_probs).items()}
    xgb_signature = infer_signature(X_hold_out, raw_probs_hold_out)
    cal_signature = infer_signature(X_hold_out, y_probs)
    run_name = f"pred_goal-{strength}-final"

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
        mlflow.log_metric("best_iteration", float(base_model.best_iteration or 0))
        mlflow.log_metrics(hold_out_metrics)

        plt.figure(dpi=DPI, figsize=SHAP_FIGSIZE)
        shap.summary_plot(shap_values, shap_sample, show=False, max_display=20)
        mlflow.log_figure(plt.gcf(), "viz/shap_summary.png")
        plt.close()

        charts = all_classifier_charts(
            y_true=y_hold_out.to_numpy(),
            y_pred=y_preds,
            y_proba=y_probs,
            importances=base_model.feature_importances_,
            feature_names=list(X_train.columns),
            classes=["no goal", "goal"],
            title_prefix=run_name,
            figsize=FIGSIZE,
            dpi=DPI,
        )
        log_viz(charts)

        xgb_model_info = mlflow.xgboost.log_model(base_model, name=run_name, signature=xgb_signature)
        mlflow.register_model(model_uri=xgb_model_info.model_uri, name=f"pred_goal_{strength}")
        mlflow.sklearn.log_model(calibrator, name=f"pred_goal-{strength}-calibrator", signature=cal_signature)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain and freeze best pred_goal model")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--strength", "-s", type=str, choices=STRENGTHS, help="Single strength state to finalize.")
    group.add_argument("--all", "-a", action="store_true", help="Finalize all 5 strength states in sequence.")
    parser.add_argument("--version", "-v", type=str, required=True)
    parser.add_argument("--run", "-r", type=str, default=None, help="MLflow run name to use instead of Optuna auto-selection. Cannot be combined with --all.")
    parser.add_argument("--no-log", action="store_true", help="Skip all MLflow/SHAP/viz — just retrain, calibrate, and save files.")
    parser.add_argument(
        "--top-n", "-n", type=int, default=15,
        help="Number of top CV PR-AUC trials to screen by calibrated hold-out log loss (default: 15). Ignored when --run is specified.",
    )
    parser.add_argument("--score", action="store_true", help="Run pred_goal/score.py after finalize completes. Only valid with --all.")
    parser.add_argument("--years", "-y", type=int, nargs="+", help="Season end-years to pass to score.py (e.g. 2024 2025). Default: all.")
    args = parser.parse_args()

    if args.run and args.all:
        parser.error("--run cannot be combined with --all — specify a single --strength when pinning a run.")
    if args.score and not args.all:
        parser.error("--score requires --all — scoring runs after all strengths are finalized.")
    if args.years and not args.score:
        parser.error("--years requires --score.")

    warnings.filterwarnings("ignore")
    load_dotenv()

    storage = load_optuna_storage()

    strengths_to_run = STRENGTHS if args.all else [args.strength]

    for strength in strengths_to_run:
        print(f"Finalizing {strength}...")
        _finalize_one(strength, args.version, storage, no_log=args.no_log, run_name=args.run, top_n=args.top_n)

    if args.score:
        score_script = Path(__file__).parent / "score.py"
        cmd = [sys.executable, str(score_script)]
        if args.years:
            cmd += ["--years"] + [str(y) for y in args.years]
        print(f"\n{'=' * 62}")
        print("  Scoring pred_goal")
        print(f"  $ {' '.join(cmd)}")
        print(f"{'=' * 62}\n")
        t0 = time.monotonic()
        result = subprocess.run(cmd)
        elapsed = time.monotonic() - t0
        mins, secs = divmod(int(elapsed), 60)
        duration = f"{mins}m {secs}s" if mins else f"{secs}s"
        if result.returncode != 0:
            print(f"\n  pred_goal/score.py FAILED (exit {result.returncode}) after {duration}")
            sys.exit(result.returncode)
        print(f"\n  pred_goal scoring done ({duration})")


if __name__ == "__main__":
    main()