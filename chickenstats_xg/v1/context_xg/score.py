
"""Score all PBP with the frozen context_xg gbtree model.

Reads base_xg scored parquets (which contain calibrated base_xg + all sequence
features), computes logit_base_xg, passes it as both base_margin (residual
learning from T1 prior) and a feature (preserving interaction constraint groups),
applies Platt calibration, overrides training-era shots with OOF predictions,
and writes:

    data/context_xg/scored/{strength}.parquet

This is the primary input to pred_goal/process_data.py (as the base_margin
prior that replaces the old single-tier base_xg).

Usage:
    python context_xg/score.py --strength even_strength
    python context_xg/score.py --all
    python context_xg/score.py --all --years 2024 2025
"""

import argparse
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
from chickenstats.utilities import ChickenProgress
from rich.progress import TaskID

from chickenstats_xg.v1.config import CONTEXT_XG_FEATURE_COLUMNS, PASSTHROUGH_COLS, STRENGTHS
from chickenstats_xg.v1.experiments import _apply_fixed_categoricals, _logit

_StrengthArg = Literal["even", "powerplay", "shorthanded", "empty_for", "empty_against"]


def score_strength(
    strength: str,
    base_xg_scored_dir: Path,
    models_dir: Path,
    scored_dir: Path,
) -> bool:
    model_path = models_dir / strength / "model.ubj"
    if not model_path.exists():
        print(f"  [{strength}] No model at {model_path} — run context_xg/finalize.py first.")
        return False

    src_path = base_xg_scored_dir / f"{strength}.parquet"
    if not src_path.exists():
        print(f"  [{strength}] No base_xg scored parquet at {src_path} — run base_xg/score.py first.")
        return False

    df = pd.read_parquet(src_path)

    # Compute logit_base_xg — used as both base_margin (residual learning) and
    # a feature (preserving interaction constraint groups).
    logit_bm = _logit(df["base_xg"].to_numpy())
    df = df.assign(logit_base_xg=logit_bm)

    feat_cols = [c for c in CONTEXT_XG_FEATURE_COLUMNS if c in df.columns]
    X = df[feat_cols].copy()
    X = _apply_fixed_categoricals(X, strength)

    booster = xgb.Booster()
    booster.load_model(str(model_path))
    dmat = xgb.DMatrix(X, enable_categorical=True, base_margin=logit_bm)
    raw_probs = 1 / (1 + np.exp(-booster.predict(dmat)))

    cal_path = models_dir / strength / "calibrator.joblib"
    calibrator = joblib.load(cal_path) if cal_path.exists() else None
    if calibrator is None:
        print(f"  [{strength}] WARNING: no calibrator found — context_xg will be uncalibrated.")

    context_xg = calibrator.predict_proba(raw_probs.reshape(-1, 1))[:, 1] if calibrator else raw_probs

    df = (
        df.assign(context_xg=context_xg)
        .sort_values(["season", "game_id", "period", "period_seconds"])
        .reset_index(drop=True)
    )

    # Override training-era shots with calibrated OOF predictions
    oof_path = models_dir / strength / "oof.parquet"
    if oof_path.exists():
        oof_df = pd.read_parquet(oof_path).dropna(subset=["context_xg"])
        oof_map = oof_df.set_index(["game_id", "event_idx"])["context_xg"]
        idx = df.set_index(["game_id", "event_idx"]).index
        in_oof = idx.isin(oof_map.index)
        df.loc[in_oof, "context_xg"] = oof_map.reindex(idx[in_oof]).values
        print(f"  [{strength}] {in_oof.sum():,} training shots replaced with OOF context_xg predictions.")

    scored_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(scored_dir / f"{strength}.parquet", index=False)
    print(f"  [{strength}] {len(df):,} shots scored → {scored_dir / f'{strength}.parquet'}")
    return True


def _load_scored_xg(scored_dir: Path) -> pl.DataFrame:
    """Combine all five strength-state context_xg scored parquets into a (game_id, event_idx) map."""
    frames = [
        pl.scan_parquet(scored_dir / f"{s}.parquet")
        .select(["game_id", "event_idx", "context_xg"])
        .with_columns(pl.col("context_xg").cast(pl.Float64))
        for s in STRENGTHS
        if (scored_dir / f"{s}.parquet").exists()
    ]
    return (
        pl.concat(frames)
        .group_by(["game_id", "event_idx"])
        .agg(pl.col("context_xg").sum())
        .collect()
    )


def _enrich_rapm_year(year_file: Path, scored_xg: pl.DataFrame, out_dir: Path) -> None:
    """Add context_xg column to a single per-year RAPM PBP file."""
    df = pl.read_parquet(year_file)
    if "context_xg" in df.columns:
        df = df.drop("context_xg")
    df = (
        df
        .join(scored_xg, on=["game_id", "event_idx"], how="left")
        .with_columns(pl.col("context_xg").fill_null(0.0))
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_dir / year_file.name)


def prep_rapm(scored_dir: Path) -> None:
    """Enrich RAPM PBP parquets with context_xg column (additive — base_xg column is kept)."""
    rapm_dir = scored_dir.parent.parent / "rapm" / "pbp"

    pbp_files = sorted(rapm_dir.glob("pbp_*.parquet"))
    if not pbp_files:
        print("  [rapm] No RAPM PBP files found — run base_xg/score.py first.")
        return

    print("  [rapm] Loading scored context_xg for RAPM enrichment...")
    scored_xg = _load_scored_xg(scored_dir)
    print(f"  [rapm] {len(scored_xg):,} scored fenwick events loaded.")

    with ChickenProgress(speed_estimate_period=300, transient=True) as progress:
        task: TaskID = progress.add_task("Enriching RAPM PBP with context_xg...", total=len(pbp_files))
        for pbp_file in pbp_files:
            progress.update(task, description=f"Enriching {pbp_file.stem}...", refresh=True)
            _enrich_rapm_year(pbp_file, scored_xg, rapm_dir)
            progress.update(task, advance=1, refresh=True)

    print(f"  [rapm] {len(pbp_files)} file(s) enriched with context_xg → {rapm_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score PBP with the frozen context_xg model.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--strength", "-s", type=str, choices=STRENGTHS)
    group.add_argument("--all", "-a", action="store_true")
    parser.add_argument(
        "--no-rapm", action="store_true",
        help="Skip RAPM PBP enrichment after scoring (only applies with --all).",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    base_xg_scored_dir = base_dir / "data" / "base_xg" / "scored"
    models_dir = base_dir / "models" / "context_xg"
    scored_dir = base_dir / "data" / "context_xg" / "scored"

    targets = STRENGTHS if args.all else [args.strength]
    print(f"Scoring {len(targets)} strength state(s)...")
    scored_count = sum(
        score_strength(strength, base_xg_scored_dir, models_dir, scored_dir)
        for strength in targets
    )

    if args.all and not args.no_rapm:
        if scored_count == 0:
            print("  [rapm] No strengths scored — skipping RAPM PBP enrichment.")
        else:
            prep_rapm(scored_dir)

    print("Done.")


if __name__ == "__main__":
    main()
