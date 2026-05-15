
"""Score raw PBP with the finalized pred_goal model (base_xg cascade).

Training shots use the OOF predictions saved by finalize.py to avoid the
in-sample bias of scoring a model on its own training data. Hold-out shots
use the final model directly (they were never in the training set).

The small fraction of training shots from the earliest fold (never in a
TimeSeriesSplit validation set) fall back to the final model.

Output:
    1_0_0/data/pred_goal/scored/pbp_{year}.parquet  — one file per NHL season

Non-fenwick events have NaN for both base_xg and pred_goal.

Usage:
    python pred_goal/score.py
    python pred_goal/score.py --years 2023 2024 2025
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from chickenstats.utilities import ChickenProgress
from rich.progress import TaskID

from chickenstats_xg.v1.config import PASSTHROUGH_COLS, STRENGTHS
from chickenstats_xg.v1.utils.transforms import apply_fixed_categoricals, logit

NON_FEATURE_COLS = ["goal", "season"] + PASSTHROUGH_COLS

# Columns in raw PBP replaced by fresh model scores.
_STALE_XG_COLS = {"base_xg", "pred_goal", "pred_goal_adj"}


def _split_df(df: pd.DataFrame, strength: str) -> tuple[pd.DataFrame, pd.Series, np.ndarray | None]:
    """Return (X_features, y, base_margin_or_None), mirroring finalize.py."""
    y = df["goal"].copy()
    bm: np.ndarray | None = None
    if "base_xg" in df.columns:
        bm = logit(df["base_xg"].to_numpy())
    drop = [c for c in NON_FEATURE_COLS if c in df.columns]
    if "base_xg" in df.columns and "base_xg" not in drop:
        drop = drop + ["base_xg"]
    X = apply_fixed_categoricals(df.drop(columns=drop), strength)
    return X, y, bm


def _model_predict(
    model: xgb.XGBClassifier,
    calibrator,
    df: pd.DataFrame,
    strength: str,
) -> np.ndarray:
    """Run the full base-model → Platt-calibrator pipeline on df."""
    X, _, bm = _split_df(df, strength)
    raw = model.predict_proba(X, base_margin=bm)[:, 1]
    return calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]


def _score_strengths(
    pred_goal_dir: Path,
    models_dir: Path,
    strengths: list[str],
) -> pd.DataFrame:
    """Return DataFrame with (game_id, event_idx, base_xg, pred_goal) for all strengths.

    Training shots use OOF predictions (unbiased). Hold-out shots and the
    small earliest-fold portion without OOF coverage use the final model.
    """
    frames: list[pd.DataFrame] = []

    for strength in strengths:
        model_path = models_dir / strength / "model.ubj"
        cal_path = models_dir / strength / "calibrator.joblib"
        oof_path = models_dir / strength / "oof.parquet"

        if not model_path.exists() or not cal_path.exists():
            print(f"  [{strength}] Missing model artifacts — run pred_goal/finalize.py first.")
            continue

        model = xgb.XGBClassifier(enable_categorical=True)
        model.load_model(str(model_path))
        calibrator = joblib.load(cal_path)

        # --- Training shots: prefer OOF predictions ---
        train_df = pd.read_parquet(pred_goal_dir / "train" / f"{strength}.parquet")
        train_scores = train_df[["game_id", "event_idx", "base_xg"]].reset_index(drop=True).copy()

        if oof_path.exists():
            oof_df = pd.read_parquet(oof_path)
            train_scores = train_scores.merge(oof_df, on=["game_id", "event_idx"], how="left")
        else:
            train_scores["pred_goal"] = np.nan

        # Fallback for earliest-fold shots that were never in a validation set
        no_oof = train_scores["pred_goal"].isna()
        if no_oof.any():
            fallback_df = train_df.iloc[np.where(no_oof.values)[0]].reset_index(drop=True)
            train_scores.loc[no_oof.values, "pred_goal"] = _model_predict(model, calibrator, fallback_df, strength)

        # --- Hold-out shots: final model is unbiased here ---
        hold_df = pd.read_parquet(pred_goal_dir / "hold_out" / f"{strength}.parquet")
        hold_scores = hold_df[["game_id", "event_idx", "base_xg"]].reset_index(drop=True).copy()
        hold_scores["pred_goal"] = _model_predict(model, calibrator, hold_df, strength)

        combined = pd.concat(
            [train_scores[["game_id", "event_idx", "base_xg", "pred_goal"]],
             hold_scores[["game_id", "event_idx", "base_xg", "pred_goal"]]],
            ignore_index=True,
        )
        frames.append(combined)
        n_oof = (~no_oof).sum()
        print(f"  [{strength}] {len(train_df):,} train ({n_oof:,} OOF) + {len(hold_df):,} hold-out scored.")

    if not frames:
        raise RuntimeError("No strengths scored — check model artifacts in models/pred_goal/.")

    result = pd.concat(frames, ignore_index=True)
    dupes = result.duplicated(subset=["game_id", "event_idx"]).sum()
    if dupes:
        print(f"  Warning: {dupes} duplicate (game_id, event_idx) pairs across strengths — keeping first.")
    return result.drop_duplicates(subset=["game_id", "event_idx"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Score raw PBP with finalized pred_goal models.")
    parser.add_argument(
        "--years", "-y", type=int, nargs="+",
        help="NHL season end-years to include (e.g. 2024 for 2023-24). Defaults to all available.",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    raw_dir = base_dir.parent / "raw_data" / "pbp"
    pred_goal_dir = base_dir / "data" / "pred_goal"
    models_dir = base_dir / "models" / "pred_goal"
    scored_dir = pred_goal_dir / "scored"
    scored_dir.mkdir(parents=True, exist_ok=True)

    print("Scoring pred_goal across all strength states...")
    xg_map = _score_strengths(pred_goal_dir, models_dir, STRENGTHS)
    print(f"  xg_map: {len(xg_map):,} fenwick events mapped.")

    available = sorted(raw_dir.glob("pbp_*.parquet"))
    if args.years:
        available = [p for p in available if int(p.stem.split("_")[1]) in args.years]
    if not available:
        print(f"No raw PBP parquets found in {raw_dir}.")
        return

    print(f"Joining onto {len(available)} raw PBP file(s)...")
    with ChickenProgress(speed_estimate_period=300, transient=True) as progress:
        task: TaskID = progress.add_task("Writing scored PBP...", total=len(available))
        for path in available:
            progress.update(task, description=f"Joining {path.stem}...", refresh=True)
            raw = pd.read_parquet(path)
            drop_cols = [c for c in raw.columns if c in _STALE_XG_COLS]
            if drop_cols:
                raw = raw.drop(columns=drop_cols)
            scored = raw.merge(xg_map, on=["game_id", "event_idx"], how="left")
            out_path = scored_dir / path.name
            scored.to_parquet(out_path, index=False)
            progress.update(task, advance=1, refresh=True)
            print(f"  {path.stem}: {len(raw):,} rows → {out_path.name}")

    print("Done.")


if __name__ == "__main__":
    main()