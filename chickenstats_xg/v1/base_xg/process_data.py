from pathlib import Path
from typing import Literal, cast

from rich.progress import TaskID

import polars as pl

from chickenstats_xg.v1.utils.shot_features import prep_data
from chickenstats.utilities import ChickenProgress
from chickenstats_xg.v1.utils.data_splitting import write_train_holdout_split

HOLD_OUT_YEAR = 2024  # raw files are named by season start-year (pbp_2024.parquet = 2024-25)

_StrengthArg = Literal["even", "powerplay", "shorthanded", "empty_for", "empty_against"]

STRENGTH_FILE_ARGS: list[tuple[str, _StrengthArg]] = [
    ("even_strength", "even"),
    ("powerplay", "powerplay"),
    ("shorthanded", "shorthanded"),
    ("empty_for", "empty_for"),
    ("empty_against", "empty_against"),
]

# Columns needed by prep_data from the raw PBP
_PREP_DATA_COLS = [
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
    "event_idx",  # passthrough — needed for OOF join in finalize.py / score.py
    "player_1_api_id",  # passthrough — needed by pred_goal for GxG/RAPM join
    "opp_goalie_api_id",  # passthrough — needed by pred_goal for GSAx/RAPM join
    "session",  # needed for hold-out split logic
    "home_on_api_id",  # passthrough — needed by process_data_inf.py for teammates RAPM
    "away_on_api_id",  # passthrough — needed by process_data_inf.py for teammates RAPM
]

READ_COLS = list(dict.fromkeys(_PREP_DATA_COLS))


def main():
    """Builds stateless base_xg training data from raw PBP CSVs.

    Features are entirely stateless — no player IDs or rolling metrics are used.
    Passthrough columns (game_id, player_1_api_id, opp_goalie_api_id) are kept in
    the output parquets for use by the pred_goal pipeline but are excluded from
    the training feature matrix in experiments.py.

    Output: data/base_xg/train/ and data/base_xg/hold_out/
    """
    years: list[int] = list(range(HOLD_OUT_YEAR, 2009, -1))

    raw_by_year: dict[int, pl.DataFrame] = {}

    with ChickenProgress(speed_estimate_period=300, transient=True) as progress:
        progress_task: TaskID = progress.add_task("Reading raw play-by-play data...", total=len(years))

        for year in years:
            progress.update(progress_task, description=f"Reading {year}...", refresh=True)
            filepath: Path = Path(__file__).parent.parent.parent.parent / "raw_data" / "pbp" / f"pbp_{year}.parquet"
            raw_by_year[year] = pl.read_parquet(filepath, columns=READ_COLS)
            progress.update(progress_task, advance=1, refresh=True)

        progress.update(progress_task, description="Finished reading raw data", refresh=True)

    combined = cast(
        pl.DataFrame,
        pl.concat(
            [pbp.with_columns(pl.lit(year).alias("_year")) for year, pbp in raw_by_year.items()], how="diagonal_relaxed"
        )
        .lazy()
        .sort(["season", "game_id", "game_seconds"])
        .collect(),
    )
    del raw_by_year

    accumulators: dict[str, list[pl.DataFrame]] = {name: [] for name, _ in STRENGTH_FILE_ARGS}

    with ChickenProgress(speed_estimate_period=300, transient=True) as progress:
        progress_task = progress.add_task("Prepping base_xg features...", total=len(years))

        for year in years:
            progress.update(progress_task, description=f"Prepping {year}...", refresh=True)

            for file_name, strength_arg in STRENGTH_FILE_ARGS:
                accumulators[file_name].append(
                    prep_data(combined.filter(pl.col("_year") == year).drop("_year"), strength_arg)
                )

            progress.update(progress_task, advance=1, refresh=True)

        progress.update(progress_task, description="Finished prepping base_xg data", refresh=True)

    del combined

    dfs: dict[str, pl.DataFrame] = {
        name: pl.concat(year_dfs, how="diagonal") for name, year_dfs in accumulators.items()
    }

    data_dir = Path(__file__).parent.parent / "data" / "base_xg"
    for name, df in dfs.items():
        write_train_holdout_split(
            df.sort(["season", "game_id", "period", "period_seconds"]),
            data_dir / "train",
            data_dir / "hold_out",
            name,
        )


if __name__ == "__main__":
    main()
