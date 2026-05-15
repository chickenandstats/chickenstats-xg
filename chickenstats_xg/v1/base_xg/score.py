
"""Score raw PBP with the frozen base_xg model.

Reads raw PBP parquets for the requested years, applies base_xg feature
engineering via prep_data(), loads the frozen XGBoost model for each
strength state, appends base_xg predictions, and writes scored parquets to:

    1_0_0/data/base_xg/scored/{strength}.parquet

The scored output is the primary input to pred_goal/process_data.py.

Usage:
    # Score a single strength state (all available years)
    python score.py --strength even_strength

    # Score all strength states
    python score.py --all

    # Limit to specific NHL season end-years (e.g. 2024 = 2023-24 season)
    python score.py --all --years 2023 2024 2025
"""

import argparse
from pathlib import Path
from typing import Literal, cast

import pandas as pd
import polars as pl
from chickenstats.utilities import ChickenProgress
from rich.progress import TaskID

from chickenstats_xg.v1.config import BASE_XG_FEATURE_COLUMNS, PASSTHROUGH_COLS, STRENGTHS
from chickenstats_xg.v1.experiments import _apply_fixed_categoricals, load_model_artifacts
from chickenstats_xg.v1.xg_utils import prep_data

_StrengthArg = Literal["even", "powerplay", "shorthanded", "empty_for", "empty_against"]

STRENGTH_FILE_ARGS: list[tuple[str, _StrengthArg]] = [
    ("even_strength", "even"),
    ("powerplay", "powerplay"),
    ("shorthanded", "shorthanded"),
    ("empty_for", "empty_for"),
    ("empty_against", "empty_against"),
]

READ_COLS = [
    "event_idx", "event", "strength_state", "coords_x", "coords_y",
    "season", "game_id", "period", "period_seconds", "game_seconds",
    "event_team", "event_distance", "event_angle", "shot_type",
    "player_1_position", "is_home", "score_diff", "zone",
    "danger", "high_danger", "goal",
    "player_1_api_id", "opp_goalie_api_id", "session",
    "home_on_api_id", "away_on_api_id",
]

NON_FEATURE_COLS = ["goal", "season"] + PASSTHROUGH_COLS


def _feature_matrix(df: pd.DataFrame, strength: str) -> pd.DataFrame:
    """Select BASE_XG_FEATURE_COLUMNS and apply fixed categorical encoding."""
    feat_cols = [c for c in BASE_XG_FEATURE_COLUMNS if c in df.columns]
    return _apply_fixed_categoricals(df[feat_cols], strength)


def score_strength(
    strength: str,
    strength_arg: _StrengthArg,
    raw_dir: Path,
    models_dir: Path,
    scored_dir: Path,
    years: list[int] | None,
) -> None:
    """Score all available raw PBP for one strength state and write parquet.

    Training-era shots are overridden with OOF predictions from finalize.py
    where available, avoiding in-sample bias. Hold-out and future shots use
    the final model directly.
    """
    model_path = models_dir / strength / "model.ubj"
    if not model_path.exists():
        print(f"  [{strength}] No model at {model_path} — run finalize-base first.")
        return

    available = sorted(raw_dir.glob("pbp_*.parquet"))
    if years:
        available = [p for p in available if int(p.stem.split("_")[1]) in years]
    if not available:
        print(f"  [{strength}] No raw parquets found for the requested years in {raw_dir}.")
        return

    raw_frames: list[pl.DataFrame] = []
    with ChickenProgress(speed_estimate_period=300, transient=True) as progress:
        task: TaskID = progress.add_task(f"Reading raw PBP...", total=len(available))
        for path in available:
            progress.update(task, description=f"Reading {path.stem}...", refresh=True)
            raw_frames.append(pl.read_parquet(path, columns=READ_COLS))
            progress.update(task, advance=1, refresh=True)

    combined = (
        cast(pl.DataFrame, pl.concat(raw_frames, how="diagonal_relaxed"))
        .lazy()
        .sort(["season", "game_id", "game_seconds"])
        .collect()
    )
    del raw_frames

    processed = prep_data(combined, strength_arg)
    del combined

    df = processed.to_pandas()
    X = _feature_matrix(df, strength)

    model, calibrator = load_model_artifacts(models_dir, strength)
    if calibrator is None:
        print(f"  [{strength}] WARNING: no calibrator found — base_xg will be uncalibrated (scale_pos_weight inflated).")

    raw_probs = model.predict_proba(X)[:, 1]
    base_xg = calibrator.predict_proba(raw_probs.reshape(-1, 1))[:, 1] if calibrator is not None else raw_probs

    df = (
        df.assign(base_xg=base_xg)
        .sort_values(["season", "game_id", "period", "period_seconds"])
        .reset_index(drop=True)
    )

    # Override training-era shots with calibrated OOF predictions from finalize.py
    oof_path = models_dir / strength / "oof.parquet"
    if oof_path.exists():
        oof_df = pd.read_parquet(oof_path).dropna(subset=["base_xg"])
        oof_map = oof_df.set_index(["game_id", "event_idx"])["base_xg"]
        idx = df.set_index(["game_id", "event_idx"]).index
        in_oof = idx.isin(oof_map.index)
        df.loc[in_oof, "base_xg"] = oof_map.reindex(idx[in_oof]).values
        print(f"  [{strength}] {in_oof.sum():,} training shots replaced with calibrated OOF predictions.")

    scored_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(scored_dir / f"{strength}.parquet", index=False)
    print(f"  [{strength}] {len(df):,} shots scored → {scored_dir / f'{strength}.parquet'}")


def _load_scored_xg(scored_dir: Path) -> pl.DataFrame:
    """Combine all five strength-state scored parquets into a (game_id, event_idx) → base_xg map."""
    frames = [
        pl.scan_parquet(scored_dir / f"{s}.parquet").select(["game_id", "event_idx", "base_xg"])
        for s in STRENGTHS
        if (scored_dir / f"{s}.parquet").exists()
    ]
    return (
        pl.concat(frames)
        .group_by(["game_id", "event_idx"])
        .agg(pl.col("base_xg").sum())
        .collect()
    )


def _enrich_rapm_year(year_file: Path, scored_xg: pl.DataFrame, out_dir: Path) -> None:
    """Join base_xg onto a single per-year raw PBP file and write the RAPM input parquet."""
    df = pl.read_parquet(year_file)
    if "base_xg" in df.columns:
        df = df.drop("base_xg")
    df = (
        df
        .join(scored_xg, on=["game_id", "event_idx"], how="left")
        .with_columns(pl.col("base_xg").fill_null(0.0))
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_dir / year_file.name)


def prep_rapm(raw_dir: Path, scored_dir: Path, years: list[int] | None) -> None:
    """Build RAPM input parquets from scored base_xg — equivalent to rapm/prep_pbp.py."""
    rapm_dir = scored_dir.parent.parent / "rapm" / "pbp"

    pbp_files = sorted(raw_dir.glob("pbp_*.parquet"))
    if years:
        pbp_files = [p for p in pbp_files if int(p.stem.split("_")[1]) in years]
    if not pbp_files:
        print("  [rapm] No raw PBP files found — skipping RAPM prep.")
        return

    print("  [rapm] Loading scored base_xg for RAPM enrichment...")
    scored_xg = _load_scored_xg(scored_dir)
    print(f"  [rapm] {len(scored_xg):,} scored fenwick events loaded.")

    with ChickenProgress(speed_estimate_period=300, transient=True) as progress:
        task: TaskID = progress.add_task("Enriching RAPM PBP...", total=len(pbp_files))
        for pbp_file in pbp_files:
            progress.update(task, description=f"Enriching {pbp_file.stem}...", refresh=True)
            _enrich_rapm_year(pbp_file, scored_xg, rapm_dir)
            progress.update(task, advance=1, refresh=True)

    print(f"  [rapm] {len(pbp_files)} file(s) written to {rapm_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score raw PBP with the frozen base_xg model.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--strength", "-s", type=str, choices=STRENGTHS, help="Single strength state to score.")
    group.add_argument("--all", "-a", action="store_true", help="Score all five strength states.")
    parser.add_argument(
        "--years", "-y", type=int, nargs="+",
        help="NHL season end-years to include (e.g. 2024 for 2023-24). Defaults to all available.",
    )
    parser.add_argument(
        "--no-rapm", action="store_true",
        help="Skip RAPM PBP enrichment after scoring (only applies with --all).",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    raw_dir = base_dir.parent / "raw_data" / "pbp"
    models_dir = base_dir / "models" / "base_xg"
    scored_dir = base_dir / "data" / "base_xg" / "scored"

    strength_arg_map: dict[str, _StrengthArg] = {name: arg for name, arg in STRENGTH_FILE_ARGS}
    targets = STRENGTH_FILE_ARGS if args.all else [(args.strength, strength_arg_map[args.strength])]

    print(f"Scoring {len(targets)} strength state(s)..." + (f" (years: {args.years})" if args.years else ""))
    for strength, strength_arg in targets:
        score_strength(strength, strength_arg, raw_dir, models_dir, scored_dir, args.years)

    if args.all and not args.no_rapm:
        prep_rapm(raw_dir, scored_dir, args.years)

    print("Done.")


if __name__ == "__main__":
    main()