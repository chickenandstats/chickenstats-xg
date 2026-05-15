
"""Shared model artifact helpers for base_xg, context_xg, and pred_goal finalize.py."""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb


def save_model_artifacts(
    model,
    calibrator,
    strength_dir: Path,
    train_df: pd.DataFrame,
    oof_prob: np.ndarray,
    oof_mask: np.ndarray,
    pred_col: str,
    params: dict | None = None,
) -> None:
    """Save model.ubj, calibrator.joblib, oof.parquet, and params.json to strength_dir.

    oof.parquet stores calibrated OOF predictions (NaN for earliest-fold rows not
    covered by any validation fold) so score.py can override in-sample predictions
    with unbiased, calibrated values without re-loading the calibrator.

    params.json stores actual Python-side XGBClassifier constructor params. XGBoost .ubj
    serialization does not preserve these in human-readable form — save_config() returns
    internal defaults (eta=0.30, max_depth=6) regardless of actual training values.
    """
    strength_dir.mkdir(parents=True, exist_ok=True)
    model.get_booster().save_model(str(strength_dir / "model.ubj"))
    joblib.dump(calibrator, strength_dir / "calibrator.joblib")
    oof_df = train_df[["game_id", "event_idx"]].reset_index(drop=True).copy()
    oof_df[pred_col] = np.where(
        oof_mask,
        calibrator.predict_proba(oof_prob.reshape(-1, 1))[:, 1],
        np.nan,
    )
    oof_df.to_parquet(strength_dir / "oof.parquet", index=False)
    if params is not None:
        serializable = {
            k: (list(v) if hasattr(v, "__iter__") and not isinstance(v, (str, bytes)) else v)
            for k, v in params.items()
            if k not in ("verbosity", "enable_categorical")
        }
        if "interaction_constraints" in serializable:
            serializable["interaction_constraints"] = [list(g) for g in serializable["interaction_constraints"]]
        if "monotone_constraints" in serializable:
            serializable["monotone_constraints"] = list(serializable["monotone_constraints"])
        (strength_dir / "params.json").write_text(json.dumps(serializable, indent=2))


def params_from_run_name(run_name: str, experiment_name: str, study: optuna.Study) -> tuple[dict, int]:
    """Fetch Optuna trial params for a specific MLflow run name.

    Looks up the run by name, reads the optuna_trial_num tag, then returns that trial's
    params from Optuna directly (avoiding string type conversion from MLflow tags).
    """
    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        filter_string=f"tags.`mlflow.runName` = '{run_name}'",
        max_results=1,
    )
    if runs.empty:
        raise ValueError(f"No run named {run_name!r} found in MLflow experiment {experiment_name!r}")
    trial_num_val = runs.iloc[0].get("tags.optuna_trial_num")
    if trial_num_val is None or pd.isna(trial_num_val):
        raise ValueError(f"Run {run_name!r} has no optuna_trial_num tag — is it a tuning child run?")
    trial_num = int(trial_num_val)
    trial = next((t for t in study.trials if t.number == trial_num), None)
    if trial is None:
        raise ValueError(f"Trial {trial_num} not found in Optuna study {study.study_name!r}")
    return trial.params, trial_num


def save_model_metadata(
    models_dir: Path,
    strength: str,
    tier: str,
    version: str,
    study_name: str,
    trial_num: int,
    trial_value: float | None,
    trial: optuna.Trial | None = None,
) -> None:
    """Write a .meta.json sidecar alongside the frozen model for provenance tracing.

    Captures enough information to trace any .ubj back to its Optuna trial and
    MLflow tuning run without querying the database.
    """
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).parent),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        git_commit = "unknown"

    tuning_run_id = trial.user_attrs.get("mlflow_run_id") if trial is not None else None

    meta = {
        "strength": strength,
        "tier": tier,
        "version": version,
        "optuna_study_name": study_name,
        "optuna_trial_number": trial_num,
        "optuna_trial_value": round(trial_value, 6) if trial_value is not None else None,
        "mlflow_tuning_run_id": tuning_run_id,
        "git_commit": git_commit,
        "finalized_at": datetime.now(timezone.utc).isoformat(),
    }

    strength_dir = models_dir / strength
    strength_dir.mkdir(parents=True, exist_ok=True)
    meta_path = strength_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  [{strength}] metadata → {meta_path}")


def load_model_artifacts(
    models_dir: Path, strength: str
) -> tuple[xgb.XGBClassifier, Any]:
    """Load frozen XGBoost model and calibrator for one strength state.

    Expects models at models_dir/{strength}/model.ubj and calibrator.joblib.
    Returns (model, calibrator). calibrator is None if the file doesn't exist.
    """
    strength_dir = models_dir / strength
    model = xgb.XGBClassifier(enable_categorical=True)
    model.load_model(str(strength_dir / "model.ubj"))
    cal_path = strength_dir / "calibrator.joblib"
    calibrator = joblib.load(cal_path) if cal_path.exists() else None
    return model, calibrator
