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
from chickenstats_xg.v1.utils.artifacts import load_model_artifacts
from chickenstats_xg.v1.utils.transforms import apply_fixed_categoricals
from chickenstats_xg.v1.utils.rapm import prep_rapm
from chickenstats_xg.v1.utils.scoring import apply_oof_predictions
from chickenstats_xg.v1.utils.shot_features import prep_data

_StrengthArg = Literal["even", "powerplay", "shorthanded", "empty_for", "empty_against"]

STRENGTH_FILE_ARGS: list[tuple[str, _StrengthArg]] = [
    ("even_strength", "even"),
    ("powerplay", "powerplay"),
    ("shorthanded", "shorthanded"),
    ("empty_for", "empty_for"),
    ("empty_against", "empty_against"),
]

READ_COLS = [
    "event_idx",
    "event",
    "strength_state",
    "coords_x",
    "coords_y",
    "season",
    "game_id",
    "period",
    "period_seconds",
    "game_seconds",
    "event_team",
    "event_distance",
    "event_angle",
    "shot_type",
    "player_1_position",
    "is_home",
    "score_diff",
    "zone",
    "danger",
    "high_danger",
    "goal",
    "player_1_api_id",
    "opp_goalie_api_id",
    "session",
    "home_on_api_id",
    "away_on_api_id",
]

NON_FEATURE_COLS = ["goal", "season"] + PASSTHROUGH_COLS


def _feature_matrix(df: pd.DataFrame, strength: str) -> pd.DataFrame:
    """Select BASE_XG_FEATURE_COLUMNS and apply fixed categorical encoding."""
    feat_cols = [c for c in BASE_XG_FEATURE_COLUMNS if c in df.columns]
    return apply_fixed_categoricals(df[feat_cols], strength)


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
        print(
            f"  [{strength}] WARNING: no calibrator found — base_xg will be uncalibrated (scale_pos_weight inflated)."
        )

    raw_probs = model.predict_proba(X)[:, 1]
    base_xg = calibrator.predict_proba(raw_probs.reshape(-1, 1))[:, 1] if calibrator is not None else raw_probs

    df = (
        df.assign(base_xg=base_xg).sort_values(["season", "game_id", "period", "period_seconds"]).reset_index(drop=True)
    )

    # Override training-era shots with calibrated OOF predictions from finalize.py
    df = apply_oof_predictions(df, models_dir, strength, "base_xg")

    scored_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(scored_dir / f"{strength}.parquet", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score raw PBP with the frozen base_xg model.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--strength", "-s", type=str, choices=STRENGTHS, help="Single strength state to score.")
    group.add_argument("--all", "-a", action="store_true", help="Score all five strength states.")
    parser.add_argument(
        "--years",
        "-y",
        type=int,
        nargs="+",
        help="NHL season end-years to include (e.g. 2024 for 2023-24). Defaults to all available.",
    )
    parser.add_argument(
        "--no-rapm",
        action="store_true",
        help="Skip RAPM PBP enrichment after scoring (only applies with --all).",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    raw_dir = base_dir.parent.parent / "raw_data" / "pbp"
    models_dir = base_dir / "models" / "base_xg"
    scored_dir = base_dir / "data" / "base_xg" / "scored"

    strength_arg_map: dict[str, _StrengthArg] = {name: arg for name, arg in STRENGTH_FILE_ARGS}
    targets = STRENGTH_FILE_ARGS if args.all else [(args.strength, strength_arg_map[args.strength])]

    with ChickenProgress() as progress:
        task = progress.add_task(f"Scoring {targets[0][0]}...", total=len(targets))
        for strength, strength_arg in targets:
            progress.update(task, description=f"Scoring {strength}...", refresh=True)
            score_strength(strength, strength_arg, raw_dir, models_dir, scored_dir, args.years)
            progress.update(task, advance=1, refresh=True)
        progress.update(task, description="Finished scoring all strength states", refresh=True)

    if args.all and not args.no_rapm:
        prep_rapm(raw_dir, scored_dir, "base_xg", args.years)


if __name__ == "__main__":
    main()
