"""Assemble pred_goal training data from scored context_xg parquets.

Pipeline:
  1. Load all 5 scored context_xg parquets (from data/context_xg/scored/), tagged with RAPM situation
  2. Sort chronologically and call compute_rolling_stats → adds GxG/GSAx columns
  3. Load RAPM table (from data/rapm/rapm_by_season.parquet)
  4. Join shooter RAPM — prior-season (lagged) + career × 6 dims → 12 columns
  5. Compute teammates RAPM — mean of on-ice teammates (shooter excluded) × prior + career × 6 dims → 12 columns
  6. Compute opponent RAPM — mean of opposing on-ice skaters × prior + career × 6 dims → 12 columns
  7. Add shooter-vs-teammates differentials (xg_off, corsi_off, goals_off) × prior + career → 6 columns
  8. Split by hold-out season and save to data/pred_goal/train/ + hold_out/

Usage:
    python process_data.py
"""

from pathlib import Path
from typing import cast

import polars as pl
from rich.progress import track

from chickenstats.utilities import ChickenProgressIndeterminate

from chickenstats_xg.v1.pred_goal.compute_rolling_stats import compute_rolling_stats
from chickenstats_xg.v1.config import BASE_XG_FEATURE_COLUMNS, CONTEXT_XG_FEATURE_COLUMNS
from chickenstats_xg.v1.utils.data_splitting import write_train_holdout_split

# Map each scored parquet → RAPM situation used for joins
STRENGTH_RAPM_SITUATION: dict[str, str] = {
    "even_strength": "EV",
    "powerplay": "PP",
    "shorthanded": "SH",
    "empty_for": "EV",
    "empty_against": "EV",
}

# Minimum teammates / opponents with RAPM required to compute a group mean
MIN_WITH_RAPM = 2

# RAPM coefficient columns in the parquet → short dim names used in feature columns
_RAPM_COEFF_COLS: list[tuple[str, str]] = [
    ("off_coeff_context_xg", "xg_off"),
    ("def_coeff_context_xg", "xg_def"),
]
_RAPM_DIMS: list[str] = [dst for _, dst in _RAPM_COEFF_COLS]
_RAPM_RENAME: dict[str, str] = {src: dst for src, dst in _RAPM_COEFF_COLS}

# Off-metric dims used for shooter-vs-teammates differentials
_OFF_METRICS: list[str] = ["xg"]


def _load_rapm(rapm_path: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load RAPM table, returning (per_season, career) DataFrames.

    per_season: one row per (player, season, situation), regular season only,
                deduplicated by max TOI for multi-team seasons.
    career:     one row per (player, situation), regular season only
                (season=0 sentinel rows from regressions.py).
    """
    rapm = pl.read_parquet(rapm_path)

    src_cols = [src for src, _ in _RAPM_COEFF_COLS]

    rapm = rapm.filter(pl.col("session") == "R")

    per_season = (
        rapm.filter(pl.col("season") != 0)
        .select(["player", "season", "situation", "toi_minutes"] + src_cols)
        .sort("toi_minutes", descending=True)
        .unique(subset=["player", "season", "situation"], keep="first")
        .drop("toi_minutes")
        .rename(_RAPM_RENAME)
        .with_columns(pl.col("player").cast(pl.Int64))
    )

    career = (
        rapm.filter(pl.col("season") == 0)
        .select(["player", "situation"] + src_cols)
        .unique(subset=["player", "situation"], keep="first")
        .rename(_RAPM_RENAME)
        .with_columns(pl.col("player").cast(pl.Int64))
    )

    return per_season, career


def _join_shooter_rapm(df: pl.DataFrame, per_season: pl.DataFrame, career: pl.DataFrame) -> pl.DataFrame:
    """Join shooter prior-season (lagged) and career RAPM for all 6 dims → 12 columns."""
    # Prior season: join season S event to season S-1 RAPM
    prior = per_season.rename({"player": "player_1_api_id"} | {d: f"shooter_rapm_{d}" for d in _RAPM_DIMS})
    df = df.with_columns((pl.col("season") - 10001).alias("_prev_season"))
    df = df.join(
        prior,
        left_on=["player_1_api_id", "_prev_season", "_rapm_situation"],
        right_on=["player_1_api_id", "season", "situation"],
        how="left",
    ).drop("_prev_season")

    # Career: join on player + situation only (no season lag)
    career_j = career.rename({"player": "player_1_api_id"} | {d: f"shooter_rapm_career_{d}" for d in _RAPM_DIMS})
    df = df.join(
        career_j,
        left_on=["player_1_api_id", "_rapm_situation"],
        right_on=["player_1_api_id", "situation"],
        how="left",
    )

    return df


def _join_aggregate_rapm(
    on_ice: pl.DataFrame,
    per_season: pl.DataFrame,
    career: pl.DataFrame,
    entity_col: str,
    prior_raw_prefix: str,
    career_raw_prefix: str,
    out_prior_prefix: str,
    out_career_prefix: str,
    count_prior_col: str,
    count_career_col: str,
) -> pl.DataFrame:
    """Join RAPM (prior + career), aggregate per _row_idx, apply MIN_WITH_RAPM threshold.

    Returns a DataFrame with columns (_row_idx, {out_prior_prefix}*, {out_career_prefix}*).
    on_ice must already contain: _row_idx, season, _rapm_situation, {entity_col}.
    """
    prior = per_season.rename({"player": entity_col} | {d: f"{prior_raw_prefix}{d}" for d in _RAPM_DIMS})
    on_ice = on_ice.with_columns((pl.col("season") - 10001).alias("_prev_season"))
    on_ice = on_ice.join(
        prior,
        left_on=[entity_col, "_prev_season", "_rapm_situation"],
        right_on=[entity_col, "season", "situation"],
        how="left",
    ).drop("_prev_season")

    career_j = career.rename({"player": entity_col} | {d: f"{career_raw_prefix}{d}" for d in _RAPM_DIMS})
    on_ice = on_ice.join(
        career_j,
        left_on=[entity_col, "_rapm_situation"],
        right_on=[entity_col, "situation"],
        how="left",
    )

    first_dim = _RAPM_DIMS[0]
    prior_agg = [pl.col(f"{prior_raw_prefix}{d}").drop_nulls().mean().alias(f"_prior_{d}_mean") for d in _RAPM_DIMS]
    career_agg = [pl.col(f"{career_raw_prefix}{d}").drop_nulls().mean().alias(f"_career_{d}_mean") for d in _RAPM_DIMS]
    count_agg = [
        pl.col(f"{prior_raw_prefix}{first_dim}").drop_nulls().len().alias(count_prior_col),
        pl.col(f"{career_raw_prefix}{first_dim}").drop_nulls().len().alias(count_career_col),
    ]

    grouped = on_ice.group_by("_row_idx").agg(prior_agg + career_agg + count_agg)

    prior_out = [
        pl.when(pl.col(count_prior_col) >= MIN_WITH_RAPM)
        .then(pl.col(f"_prior_{d}_mean"))
        .otherwise(None)
        .alias(f"{out_prior_prefix}{d}")
        for d in _RAPM_DIMS
    ]
    career_out = [
        pl.when(pl.col(count_career_col) >= MIN_WITH_RAPM)
        .then(pl.col(f"_career_{d}_mean"))
        .otherwise(None)
        .alias(f"{out_career_prefix}{d}")
        for d in _RAPM_DIMS
    ]
    output_cols = [c for d in _RAPM_DIMS for c in (f"{out_prior_prefix}{d}", f"{out_career_prefix}{d}")]

    return grouped.with_columns(prior_out + career_out).select(["_row_idx"] + output_cols)


def _compute_teammates_rapm(df: pl.DataFrame, per_season: pl.DataFrame, career: pl.DataFrame) -> pl.DataFrame:
    """Compute mean RAPM of on-ice teammates (shooter excluded) for all dims × prior + career.

    Adds teammates_rapm_{dim} and teammates_rapm_career_{dim} columns, plus
    shooter_vs_teammates_rapm_{metric}_off differentials.
    Events with fewer than MIN_WITH_RAPM valid entries receive null per column group.
    """
    df = df.with_row_index("_row_idx")

    on_ice = df.select(
        [
            "_row_idx",
            "player_1_api_id",
            "season",
            "_rapm_situation",
            pl.when(pl.col("is_home") == 1)
            .then(pl.col("home_on_api_id"))
            .otherwise(pl.col("away_on_api_id"))
            .alias("_on_ice_ids"),
        ]
    )
    on_ice = (
        on_ice.with_columns(pl.col("_on_ice_ids").str.split(", ").alias("_ids_list"))
        .explode("_ids_list")
        .filter(pl.col("_ids_list").is_not_null() & (pl.col("_ids_list") != ""))
        .with_columns(pl.col("_ids_list").cast(pl.Int64).alias("_teammate_id"))
        .drop(["_on_ice_ids", "_ids_list"])
        .filter(pl.col("_teammate_id") != pl.col("player_1_api_id"))
    )

    teammates_mean = _join_aggregate_rapm(
        on_ice,
        per_season,
        career,
        entity_col="_teammate_id",
        prior_raw_prefix="_tm_prior_",
        career_raw_prefix="_tm_career_",
        out_prior_prefix="teammates_rapm_",
        out_career_prefix="teammates_rapm_career_",
        count_prior_col="_tm_prior_count",
        count_career_col="_tm_career_count",
    )
    df = df.join(teammates_mean, on="_row_idx", how="left").drop("_row_idx")

    diff_exprs = []
    for metric in _OFF_METRICS:
        d = f"{metric}_off"
        diff_exprs.append(
            (pl.col(f"shooter_rapm_{d}") - pl.col(f"teammates_rapm_{d}")).alias(
                f"shooter_vs_teammates_rapm_{metric}_off"
            )
        )
        diff_exprs.append(
            (pl.col(f"shooter_rapm_career_{d}") - pl.col(f"teammates_rapm_career_{d}")).alias(
                f"shooter_vs_teammates_rapm_career_{metric}_off"
            )
        )
    return df.with_columns(diff_exprs)


def _compute_opponent_rapm(df: pl.DataFrame, per_season: pl.DataFrame, career: pl.DataFrame) -> pl.DataFrame:
    """Compute mean RAPM of opposing on-ice skaters for all dims × prior + career → 12 columns.

    Uses away_on_api_id when is_home == 1, home_on_api_id otherwise.
    Events with fewer than MIN_WITH_RAPM valid entries receive null per column group.
    """
    df = df.with_row_index("_row_idx")

    on_ice = df.select(
        [
            "_row_idx",
            "season",
            "_rapm_situation",
            pl.when(pl.col("is_home") == 1)
            .then(pl.col("away_on_api_id"))
            .otherwise(pl.col("home_on_api_id"))
            .alias("_on_ice_ids"),
        ]
    )
    on_ice = (
        on_ice.with_columns(pl.col("_on_ice_ids").str.split(", ").alias("_ids_list"))
        .explode("_ids_list")
        .filter(pl.col("_ids_list").is_not_null() & (pl.col("_ids_list") != ""))
        .with_columns(pl.col("_ids_list").cast(pl.Int64).alias("_opp_id"))
        .drop(["_on_ice_ids", "_ids_list"])
    )

    opp_mean = _join_aggregate_rapm(
        on_ice,
        per_season,
        career,
        entity_col="_opp_id",
        prior_raw_prefix="_opp_prior_",
        career_raw_prefix="_opp_career_",
        out_prior_prefix="opp_rapm_",
        out_career_prefix="opp_rapm_career_",
        count_prior_col="_opp_prior_count",
        count_career_col="_opp_career_count",
    )
    return df.join(opp_mean, on="_row_idx", how="left").drop("_row_idx")


def main() -> None:
    """Assemble pred_goal training and hold-out data."""
    data_dir = Path(__file__).parent.parent / "data"
    # Tier 2 (context_xg) scored parquets are the base_margin source for pred_goal.
    # They contain both base_xg (Tier 1) and context_xg (Tier 2). Drop Tier 1;
    # context_xg keeps its own name throughout the pred_goal pipeline.
    scored_dir = data_dir / "context_xg" / "scored"
    rapm_path = data_dir / "rapm" / "rapm_by_season.parquet"

    # 1. Load and combine scored parquets
    frames = []
    for strength, situation in track(STRENGTH_RAPM_SITUATION.items(), description="Loading scored parquets..."):
        raw = pl.read_parquet(scored_dir / f"{strength}.parquet")
        # context_xg parquets carry both base_xg (Tier 1) and context_xg (Tier 2).
        # Drop the Tier 1 column; context_xg stays as context_xg.
        if "base_xg" in raw.columns:
            raw = raw.drop("base_xg")
        df = raw.with_columns(pl.lit(situation).alias("_rapm_situation"))
        frames.append(df)

    combined = cast(
        pl.DataFrame, pl.concat(frames, how="diagonal_relaxed").sort(["season", "game_id", "period", "period_seconds"])
    )
    del frames

    # 2. Rolling GxG / GSAx
    with ChickenProgressIndeterminate(transient=True) as progress:
        task = progress.add_task("Computing rolling GxG / GSAx...", total=None)
        progress.start_task(task)
        combined = compute_rolling_stats(combined)
        progress.update(task, total=1, advance=1, description="Finished computing rolling stats", refresh=True)

    # 3. RAPM joins
    per_season, career = _load_rapm(rapm_path)
    combined = _join_shooter_rapm(combined, per_season, career)
    combined = _compute_teammates_rapm(combined, per_season, career)
    combined = _compute_opponent_rapm(combined, per_season, career)

    # Drop internal columns before saving
    combined = combined.drop(["_rapm_situation"])

    # 4. Split and save
    pred_goal_dir = data_dir / "pred_goal"
    strengths = list(STRENGTH_RAPM_SITUATION.keys())
    strength_state_map = {
        "even_strength": ["5v5", "4v4", "3v3"],
        "powerplay": ["5v4", "4v3", "5v3"],
        "shorthanded": ["4v5", "3v4", "3v5"],
        "empty_for": ["Ev5", "Ev4", "Ev3"],
        "empty_against": ["5vE", "4vE", "3vE"],
    }

    for strength in track(strengths, description="Saving pred_goal parquets..."):
        states = strength_state_map[strength]
        subset = combined.filter(pl.col("strength_state").is_in(states))
        # Drop all Tier 1 and Tier 2 features — both are encoded in base_margin
        # (logit(context_xg)), not pred_goal inputs.
        env_cols = [c for c in BASE_XG_FEATURE_COLUMNS + CONTEXT_XG_FEATURE_COLUMNS if c in subset.columns]
        if env_cols:
            subset = subset.drop(env_cols)
        write_train_holdout_split(subset, pred_goal_dir / "train", pred_goal_dir / "hold_out", strength)


if __name__ == "__main__":
    main()
