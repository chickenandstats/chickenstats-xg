"""Shared train/holdout split helper for base_xg, context_xg, and pred_goal process_data.py."""

from pathlib import Path

import polars as pl

from chickenstats_xg.v1.config import HOLD_OUT_SEASON


def write_train_holdout_split(
    df: pl.DataFrame,
    train_dir: Path,
    hold_out_dir: Path,
    filename: str,
) -> None:
    """Write a Polars DataFrame split by HOLD_OUT_SEASON to train and hold_out directories.

    Args:
        df:           DataFrame containing a "season" column.
        train_dir:    Destination for the training split (season != HOLD_OUT_SEASON).
        hold_out_dir: Destination for the hold-out split (season == HOLD_OUT_SEASON).
        filename:     Stem used for the output parquet names (e.g. "even_strength").
    """
    train_dir.mkdir(parents=True, exist_ok=True)
    hold_out_dir.mkdir(parents=True, exist_ok=True)
    df.filter(pl.col("season") != HOLD_OUT_SEASON).write_parquet(train_dir / f"{filename}.parquet")
    df.filter(pl.col("season") == HOLD_OUT_SEASON).write_parquet(hold_out_dir / f"{filename}.parquet")
