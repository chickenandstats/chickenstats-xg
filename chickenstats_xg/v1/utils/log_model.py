"""Log a locally saved model to MLflow and register it in the Model Registry.

Reads the meta.json sidecar written by finalize scripts, reopens the original
trial's MLflow run (or creates a new standalone run if the run ID is absent),
logs the frozen model, and registers it under {tier}_{strength}.

Useful when a model was finalized with --no-log, or when MLflow was unavailable
at finalize time. Does NOT retrain — the local .ubj and calibrator.joblib files
are the source of truth.

Usage:
    log-model --tier base_xg --strength even_strength
    log-model --tier context_xg --all
    log-model --all                         # all tiers × all strengths
"""

import argparse
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
from dotenv import load_dotenv
from mlflow.models.signature import infer_signature

from chickenstats_xg.v1.config import (
    BASE_XG_FEATURE_COLUMNS,
    CONTEXT_XG_FEATURE_COLUMNS,
    PASSTHROUGH_COLS,
    STRENGTHS,
)
from chickenstats_xg.v1.utils.artifacts import load_model_artifacts
from chickenstats_xg.v1.utils.transforms import apply_fixed_categoricals, logit

TIERS = ["base_xg", "context_xg", "pred_goal"]

# Maps tier → Optuna study name suffix (matches finalize scripts)
_STUDY_SUFFIX = {
    "base_xg": "base",
    "context_xg": "context",
    "pred_goal": "pred_goal",
}

# NON_FEATURE_COLS for pred_goal (mirrors pred_goal/finalize.py)
_PRED_GOAL_DROP = ["goal", "season", "base_xg"] + PASSTHROUGH_COLS


def _build_signature(
    tier: str,
    strength: str,
    base_dir: Path,
    model,
    calibrator,
):
    """Infer MLflow signature from hold-out data. Returns None if data is missing."""
    hold_path = base_dir / "data" / tier / "hold_out" / f"{strength}.parquet"
    if not hold_path.exists():
        print(f"  [{tier}/{strength}] Hold-out parquet not found — logging without signature.")
        return None

    df = pd.read_parquet(hold_path)

    try:
        if tier == "base_xg":
            feat_cols = [c for c in BASE_XG_FEATURE_COLUMNS if c in df.columns]
            X = apply_fixed_categoricals(df[feat_cols], strength)
            y_prob = model.predict_proba(X)[:, 1]

        elif tier == "context_xg":
            feat_cols = [c for c in CONTEXT_XG_FEATURE_COLUMNS if c in df.columns]
            X = apply_fixed_categoricals(df[feat_cols], strength)
            bm = df["logit_base_xg"].to_numpy() if "logit_base_xg" in df.columns else None
            y_prob = model.predict_proba(X, base_margin=bm)[:, 1]

        elif tier == "pred_goal":
            drop = [c for c in _PRED_GOAL_DROP if c in df.columns]
            X = apply_fixed_categoricals(df.drop(columns=drop), strength)
            bm = logit(df["base_xg"].to_numpy()) if "base_xg" in df.columns else None
            y_prob = model.predict_proba(X, base_margin=bm)[:, 1]
            if calibrator is not None:
                y_prob = calibrator.predict_proba(y_prob.reshape(-1, 1))[:, 1]

        else:
            return None

        return infer_signature(X, y_prob)

    except Exception as exc:
        print(f"  [{tier}/{strength}] Signature inference failed ({exc}) — logging without signature.")
        return None


def _log_one(tier: str, strength: str, base_dir: Path) -> None:
    """Log and register one tier/strength model from local files."""
    strength_dir = base_dir / "models" / tier / strength
    meta_path = strength_dir / "meta.json"

    if not meta_path.exists():
        tier_cmd = {"base_xg": "finalize-base", "context_xg": "finalize-context", "pred_goal": "finalize-pred"}
        print(f"  [{tier}/{strength}] No meta.json — run {tier_cmd.get(tier, 'finalize')} first.")
        return

    meta = json.loads(meta_path.read_text())
    tuning_run_id = meta.get("mlflow_tuning_run_id")
    version = meta.get("version", "unknown")

    models_dir = base_dir / "models" / tier
    model, calibrator = load_model_artifacts(models_dir, strength)

    signature = _build_signature(tier, strength, base_dir, model, calibrator)

    run_name = f"{tier}-{strength}-final"

    if tuning_run_id:
        run_ctx = mlflow.start_run(run_id=tuning_run_id)
    else:
        study_name = f"{strength}-{version}-{_STUDY_SUFFIX[tier]}"
        mlflow.set_experiment(study_name)
        run_ctx = mlflow.start_run(
            tags={"type": "finalize", "strength": strength, "version": version}
        )

    with run_ctx:
        mlflow.set_tags({
            "finalized": "true",
            "logged_from": "local",
            "logged_at": datetime.now(timezone.utc).isoformat(),
            "finalized_version": version,
        })

        model_info = mlflow.xgboost.log_model(model, name=run_name, signature=signature)
        mlflow.register_model(model_uri=model_info.model_uri, name=f"{tier}_{strength}")

        if tier == "pred_goal" and calibrator is not None:
            mlflow.sklearn.log_model(
                calibrator,
                name=f"{tier}-{strength}-calibrator",
                signature=signature,
            )

    print(f"  [{tier}/{strength}] logged → Model Registry: {tier}_{strength}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Log locally saved model artifacts to MLflow and register in the Model Registry."
    )
    tier_group = parser.add_mutually_exclusive_group(required=True)
    tier_group.add_argument("--tier", "-t", type=str, choices=TIERS, help="Single tier to log.")
    tier_group.add_argument("--all", "-a", action="store_true", help="Log all tiers × all strengths.")
    parser.add_argument(
        "--strength", "-s", type=str, choices=STRENGTHS,
        help="Single strength state (required unless --all is used with --tier, or --all is used alone).",
    )
    parser.add_argument(
        "--strengths-all", action="store_true",
        help="Log all strength states for the specified --tier.",
    )
    args = parser.parse_args()

    if args.all and args.strength:
        parser.error("--strength cannot be combined with --all.")
    if args.tier and not args.strength and not args.strengths_all:
        parser.error("--tier requires --strength or --strengths-all.")

    warnings.filterwarnings("ignore")
    load_dotenv()
    mlflow.enable_system_metrics_logging()

    base_dir = Path(__file__).parent.parent

    if args.all:
        targets = [(tier, strength) for tier in TIERS for strength in STRENGTHS]
    elif args.strengths_all:
        targets = [(args.tier, s) for s in STRENGTHS]
    else:
        targets = [(args.tier, args.strength)]

    print(f"Logging {len(targets)} model(s) to MLflow...")
    for tier, strength in targets:
        _log_one(tier, strength, base_dir)

    print("Done.")


if __name__ == "__main__":
    main()
