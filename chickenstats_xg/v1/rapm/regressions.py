
from typing import cast

import polars as pl
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from joblib import parallel_backend
from scipy.sparse import coo_matrix

from chickenstats.utilities import ChickenProgress

from pathlib import Path
from chickenstats_xg.v1.config import RAPM_TOI_LIMITS


def build_position_map(pbp_df: pl.LazyFrame):
    """Function to build a map of players and their positions."""
    position_map = (
        pl.concat(
            [
                pbp_df.select(
                    id=pl.col("home_on_api_id").str.split(", "), pos=pl.col("home_on_positions").str.split(", ")
                ).explode(["id", "pos"]),
                pbp_df.select(
                    id=pl.col("away_on_api_id").str.split(", "), pos=pl.col("away_on_positions").str.split(", ")
                ).explode(["id", "pos"]),
            ]
        )
        .unique("id")
        .collect()
    )

    return position_map


def build_matrix(stints_df: pl.DataFrame, metric="xg", situation="EV", min_toi=1):
    """Build a sparse matrix for RAPM ridge regression.

    Each stint appears twice (home and away perspective). Columns are binary indicators:
    offensive skaters (0..num_s-1), defensive skaters (num_s..2*num_s-1), optional goalies,
    strength state, OZS, home advantage, B2B, and score state.

    Returns X, Y, W (TOI weights), dates (game_date array or None), skater_list, player_metrics.
    """
    metric_column_map = {"context_xg": "xg", "corsi": "c", "goals": "g"}
    if metric not in metric_column_map:
        raise ValueError(f"Unknown metric: {metric!r}. Expected one of {list(metric_column_map)}")
    metric_column = metric_column_map[metric]

    # Filter dataframe for desired situation
    if situation == "EV":
        stints_df = stints_df.filter(pl.col("h_cnt") == pl.col("a_cnt"))
    elif situation == "PP" or situation == "SH":
        stints_df = stints_df.filter(pl.col("h_cnt") != pl.col("a_cnt"))

    # Extract Raw Metrics from RAW Stints (BEFORE Stacking)
    home_player_metrics = (
        stints_df.select(
            [
                pl.col("h_skaters").alias("players"),
                pl.col("h_team").alias("team"),
                pl.col("toi"),
                pl.col(f"h_{metric_column}f").cast(pl.Float64).alias("metric_for"),
                pl.col(f"a_{metric_column}f").cast(pl.Float64).alias("metric_against"),
            ]
        )
        .explode("players")
        .drop_nulls("players")
    )

    away_player_metrics = (
        stints_df.select(
            [
                pl.col("a_skaters").alias("players"),
                pl.col("a_team").alias("team"),
                pl.col("toi"),
                pl.col(f"a_{metric_column}f").cast(pl.Float64).alias("metric_for"),
                pl.col(f"h_{metric_column}f").cast(pl.Float64).alias("metric_against"),
            ]
        )
        .explode("players")
        .drop_nulls("players")
    )

    player_metrics = pl.concat([home_player_metrics, away_player_metrics]).with_columns(
        player_team=pl.col("players") + "_" + pl.col("team")
    )
    player_metrics = (
        player_metrics.group_by("player_team")
        .agg([pl.col("toi").sum(), pl.col("metric_for").sum(), pl.col("metric_against").sum()])
        .filter(pl.col("toi") >= (min_toi * 60))
    )

    # Resolve game_date — include if present, otherwise fill with null
    has_date = "game_date" in stints_df.columns
    if has_date and stints_df["game_date"].dtype == pl.Utf8:
        stints_df = stints_df.with_columns(pl.col("game_date").str.to_date(strict=False))
    date_col = [pl.col("game_date")] if has_date else [pl.lit(None).cast(pl.Date).alias("game_date")]

    # Perspective Stacking for Ridge Regression
    home_stints = stints_df.select(
        [
            pl.col("toi"),
            (pl.col(f"h_{metric_column}f") / (pl.col("toi") / 3600)).alias("Y"),
            pl.col("h_skaters").alias("offense"),
            pl.col("a_skaters").alias("defense"),
            pl.col("a_goalies").alias("goalie_def"),
            pl.col("h_team").alias("off_team"),
            pl.col("a_team").alias("def_team"),
            pl.col("strength"),
            pl.col("ozs"),
            pl.col("nzs"),
            pl.col("dzs"),
            pl.col("h_b2b").alias("b2b"),
            pl.col("h_s7" if metric == "corsi" else "h_s3").alias("score_state"),
            pl.lit(1).alias("home_adv"),
            pl.col("h_cnt").alias("off_cnt"),
            pl.col("a_cnt").alias("def_cnt"),
        ] + date_col
    )

    away_stints = stints_df.select(
        [
            pl.col("toi"),
            (pl.col(f"a_{metric_column}f") / (pl.col("toi") / 3600)).alias("Y"),
            pl.col("a_skaters").alias("offense"),
            pl.col("h_skaters").alias("defense"),
            pl.col("h_goalies").alias("goalie_def"),
            pl.col("a_team").alias("off_team"),
            pl.col("h_team").alias("def_team"),
            pl.col("strength"),
            pl.col("ozs"),
            pl.col("nzs"),
            pl.col("dzs"),
            pl.col("a_b2b").alias("b2b"),
            pl.col("a_s7" if metric == "corsi" else "a_s3").alias("score_state"),
            pl.lit(0).alias("home_adv"),
            pl.col("a_cnt").alias("off_cnt"),
            pl.col("h_cnt").alias("def_cnt"),
        ] + date_col
    )

    # Each stint is doubled: row_idx tracks which matrix row each perspective maps to
    stacked = pl.concat([home_stints, away_stints]).with_row_index("row_idx")

    # Same skater list used for both offense and defense columns; offset shifts the column range
    skater_list = sorted(player_metrics["player_team"].to_list())
    num_s = len(skater_list)
    offense_skaters_map = pl.DataFrame({"player_team": skater_list, "idx": range(num_s)})
    defense_skaters_map = pl.DataFrame({"player_team": skater_list, "idx": range(num_s, num_s * 2)})

    if metric == "goals":
        goalies = (
            stacked.select(pl.col("goalie_def").alias("goalie"), pl.col("def_team").alias("team"))
            .explode("goalie")
            .drop_nulls("goalie")
        )
        goalies = goalies.with_columns(goalie_team=pl.col("goalie") + "_" + pl.col("team"))
        goalie_list = sorted(goalies["goalie_team"].unique().to_list())
    else:
        goalie_list = []

    goalies_map = pl.DataFrame({"goalie_team": goalie_list, "idx": range(num_s * 2, (num_s * 2) + len(goalie_list))})

    # offset tracks the next available column index as we add each feature group
    offset = (num_s * 2) + len(goalie_list)
    strengths = sorted(stacked["strength"].drop_nulls().unique().to_list())
    str_dict = {s: offset + i for i, s in enumerate(strengths)}
    offset += len(strengths)

    # Five scalar binary features each get a single dedicated column
    idx_ozs, idx_nzs, idx_dzs, idx_h_adv, idx_b2b = offset, offset + 1, offset + 2, offset + 3, offset + 4
    offset += 5

    # Score state: 7-bucket for corsi (sensitive to small leads), 3-bucket for xG and goals
    score_lvls = [-3, -2, -1, 1, 2, 3] if metric == "corsi" else [-1, 1]
    score_dict = {lvl: offset + i for i, lvl in enumerate(score_lvls)}

    # Build (row, col) coordinate lists for the sparse matrix — all values will be 1
    all_rows, all_cols = [], []

    offense_df = stacked.select("row_idx", "offense", "off_team").explode("offense").drop_nulls("offense")
    offense_df = offense_df.with_columns(player_team=pl.col("offense") + "_" + pl.col("off_team"))
    offense_df = offense_df.join(offense_skaters_map, on="player_team", how="inner")
    all_rows.append(offense_df["row_idx"].to_numpy())
    all_cols.append(offense_df["idx"].to_numpy())

    defense_df = stacked.select("row_idx", "defense", "def_team").explode("defense").drop_nulls("defense")
    defense_df = defense_df.with_columns(player_team=pl.col("defense") + "_" + pl.col("def_team"))
    defense_df = defense_df.join(defense_skaters_map, on="player_team", how="inner")
    all_rows.append(defense_df["row_idx"].to_numpy())
    all_cols.append(defense_df["idx"].to_numpy())

    if metric == "goals":
        goalies_df = (
            stacked.select("row_idx", pl.col("goalie_def").alias("goalie"), "def_team")
            .explode("goalie")
            .drop_nulls("goalie")
        )
        goalies_df = goalies_df.with_columns(goalie_team=pl.col("goalie") + "_" + pl.col("def_team"))
        goalies_df = goalies_df.join(goalies_map, on="goalie_team", how="inner")
        all_rows.append(goalies_df["row_idx"].to_numpy())
        all_cols.append(goalies_df["idx"].to_numpy())

    str_mapped = (
        stacked.select("row_idx", "strength")
        .with_columns(idx=pl.col("strength").replace_strict(str_dict, default=None))
        .drop_nulls("idx")
    )
    all_rows.append(str_mapped["row_idx"].to_numpy())
    all_cols.append(str_mapped["idx"].to_numpy())

    score_mapped = (
        stacked.select("row_idx", "score_state")
        .with_columns(idx=pl.col("score_state").replace_strict(score_dict, default=None))
        .drop_nulls("idx")
    )
    all_rows.append(score_mapped["row_idx"].to_numpy())
    all_cols.append(score_mapped["idx"].to_numpy())

    ozs_df = stacked.filter(pl.col("ozs")).select("row_idx")
    all_rows.append(ozs_df["row_idx"].to_numpy())
    all_cols.append(np.full(len(ozs_df), idx_ozs, dtype=np.intp))

    nzs_df = stacked.filter(pl.col("nzs")).select("row_idx")
    all_rows.append(nzs_df["row_idx"].to_numpy())
    all_cols.append(np.full(len(nzs_df), idx_nzs, dtype=np.intp))

    dzs_df = stacked.filter(pl.col("dzs")).select("row_idx")
    all_rows.append(dzs_df["row_idx"].to_numpy())
    all_cols.append(np.full(len(dzs_df), idx_dzs, dtype=np.intp))

    hadv_df = stacked.filter(pl.col("home_adv") == 1).select("row_idx")
    all_rows.append(hadv_df["row_idx"].to_numpy())
    all_cols.append(np.full(len(hadv_df), idx_h_adv, dtype=np.intp))

    b2b_df = stacked.filter(pl.col("b2b")).select("row_idx")
    all_rows.append(b2b_df["row_idx"].to_numpy())
    all_cols.append(np.full(len(b2b_df), idx_b2b, dtype=np.intp))

    Y = stacked["Y"].fill_nan(0.0).to_numpy()
    W = stacked["toi"].to_numpy()
    dates = stacked["game_date"].to_numpy() if has_date else None

    rows = np.concatenate(all_rows)
    cols = np.concatenate(all_cols)
    data = np.ones(len(rows), dtype=np.float32)

    X = coo_matrix((data, (rows, cols)), shape=(len(Y), offset + len(score_lvls)), dtype=np.float32).tocsr()

    return X, Y, W, dates, skater_list, player_metrics


def _decay_weights(W: np.ndarray, dates: np.ndarray | None, decay_halflife: float) -> np.ndarray:
    """Multiply TOI weights by exponential recency decay.

    Stints closer to the last game in the dataset receive weight 1.0;
    earlier stints decay with half-life = decay_halflife days.
    """
    if dates is None or decay_halflife <= 0:
        return W
    valid_mask = dates != None  # noqa: E711
    if not valid_mask.any():
        return W
    max_date = np.max(dates[valid_mask])
    days_before_end = np.where(
        valid_mask,
        (max_date - dates).astype("timedelta64[D]").astype(float),
        0.0,
    )
    decay = np.exp(-np.log(2) / decay_halflife * days_before_end)
    return W * decay


def run_regression(
    stints_df: pl.DataFrame,
    season: int,
    session: str,
    situation: str,
    metric: str,
    toi_limit: int,
    decay_halflife: float = 60.0,
):
    """Run RAPM ridge regression on a single season. Returns results and best alpha.

    decay_halflife: half-life in days for recency weighting within the season.
    Stints played closer to the end of the season receive proportionally more
    weight in the solver. Set to 0 to disable.
    """
    X, Y, W, dates, players, player_toi = build_matrix(stints_df, metric, situation, toi_limit)
    W_decay = _decay_weights(W, dates, decay_halflife)

    with parallel_backend("threading"):
        grid = GridSearchCV(
            Ridge(fit_intercept=False, solver="sparse_cg"),
            param_grid={"alpha": np.logspace(3, 5.5, 15)},
            cv=5,
            n_jobs=-1,
        )
        grid.fit(X, Y - np.average(Y, weights=W_decay), sample_weight=W_decay)

    num_p = len(players)
    results = pl.DataFrame(
        {
            "season": season,
            "session": session,
            "player_team": players,
            "metric": metric,
            "situation": situation,
            "off_coeff": grid.best_estimator_.coef_[:num_p],
            "def_coeff": grid.best_estimator_.coef_[num_p : num_p * 2],
        }
    ).join(player_toi, on="player_team")

    alpha = grid.best_params_["alpha"]

    return results, alpha


def run_all_regressions(
    stints_directory: Path,
    position_map: pl.DataFrame,
    seasons: list | None = None,
    sessions: list | None = None,
    situations: list | None = None,
    metrics: list | None = None,
    toi_limits: dict | None = None,
    decay_halflife: float = 60.0,
) -> pl.DataFrame:
    """Run RAPM regressions for all seasons.

    decay_halflife: half-life in days for within-season recency weighting.
    Career RAPM (season=0 rows) is computed in process_regression_results() as a
    TOI-weighted mean of individual season coefficients — no separate regression needed.
    """
    if seasons is None:
        seasons = [int(f"{x}{x + 1}") for x in list(range(2010, 2026))]

    if metrics is None:
        metrics = ["context_xg", "corsi", "goals"]

    if sessions is None:
        sessions = ["R", "P"]

    if situations is None:
        situations = ["EV", "PP", "SH"]

    if toi_limits is None:
        toi_limits = {"ev_r": 10, "ev_p": 5, "other_r": 5, "other_p": 1}

    all_results = []

    with ChickenProgress() as outer_progress:
        outer_progress_task = outer_progress.add_task(
            f"Running regressions for {str(seasons[0])[:4]}-{str(seasons[0])[4:]} season...", total=len(seasons)
        )

        for idx, season in enumerate(seasons):
            with ChickenProgress(transient=True) as inner_progress:
                inner_progress_task = inner_progress.add_task(
                    f"Running regression with {metrics[0]} as target metric...", total=len(metrics)
                )

                # Load each session's stints once; reused across all metrics and situations
                session_stints = {
                    session: pl.read_parquet(stints_directory / f"stints_{str(season)[:4]}_{session.lower()}.parquet")
                    for session in sessions
                }

                for idx_, metric in enumerate(metrics):
                    for situation in situations:
                        for session in sessions:
                            toi_limit = (
                                toi_limits["ev_r" if session == "R" else "ev_p"]
                                if situation == "EV"
                                else toi_limits["other_r" if session == "R" else "other_p"]
                            )

                            results, alpha = run_regression(
                                stints_df=session_stints[session],
                                season=season,
                                session=session,
                                metric=metric,
                                situation=situation,
                                toi_limit=toi_limit,
                                decay_halflife=decay_halflife,
                            )

                            all_results.append(results)

                    if idx_ + 1 < len(metrics):
                        next_inner_message = f"Running regression with {metrics[idx_ + 1]} as target metric..."
                    else:
                        next_inner_message = f"Finished running RAPM regressions for {str(seasons[idx])[:4]}-{str(seasons[idx])[4:]} season"
                    inner_progress.update(inner_progress_task, description=next_inner_message, advance=1, refresh=True)

            if idx + 1 < len(seasons):
                next_message = (
                    f"Running regressions for {str(seasons[idx + 1])[:4]}-{str(seasons[idx + 1])[4:]} season..."
                )
            else:
                next_message = "Finished running RAPM regressions"

            outer_progress.update(outer_progress_task, description=next_message, advance=1, refresh=True)

    results = pl.concat(all_results).with_columns(
        player=pl.col("player_team").str.split("_").list.get(0), team=pl.col("player_team").str.split("_").list.get(1)
    )

    final_results = process_regression_results(results, position_map)

    return final_results


def process_regression_results(regression_results: pl.DataFrame, position_map: pl.DataFrame):
    """Process regression results into per-season and career RAPM tables.

    Career rows (season=0) are the TOI-weighted mean of per-season coefficients.
    This is preferable to a single pooled regression because each season's Ridge
    coefficients are internally consistent within their competitive context; the
    career estimate is a summary statistic over those stable per-season estimates.
    """
    per_season = regression_results.group_by(["season", "session", "player", "team", "situation", "metric"]).agg(
        [
            ((pl.col("off_coeff") * pl.col("toi")).sum() / pl.col("toi").sum()).alias("off_coeff"),
            ((pl.col("def_coeff") * pl.col("toi")).sum() / pl.col("toi").sum()).alias("def_coeff"),
            pl.col("metric_for").sum().alias("metric_for"),
            pl.col("metric_against").sum().alias("metric_against"),
            pl.col("toi").sum().alias("toi_seconds"),
        ]
    )

    per_season = per_season.with_columns(
        [
            (pl.col("toi_seconds") / 60).alias("toi_minutes"),
            (pl.col("metric_for") - pl.col("metric_against")).alias("metric_diff"),
            (pl.col("metric_for") / (pl.col("toi_seconds") / 3600)).alias("on_ice_for_60"),
            (pl.col("metric_against") / (pl.col("toi_seconds") / 3600)).alias("on_ice_against_60"),
        ]
    ).with_columns((pl.col("on_ice_for_60") - pl.col("on_ice_against_60")).alias("on_ice_diff_60"))

    per_season = per_season.join(position_map.select(["id", "pos"]), left_on="player", right_on="id", how="left")

    final_db = per_season.pivot(
        values=[
            "off_coeff",
            "def_coeff",
            "metric_for",
            "metric_against",
            "metric_diff",
            "on_ice_for_60",
            "on_ice_against_60",
            "on_ice_diff_60",
        ],
        index=["season", "session", "player", "team", "pos", "situation", "toi_minutes"],
        on="metric",
    )

    final_db = final_db.with_columns(
        [
            (pl.col("off_coeff_context_xg") - pl.col("def_coeff_context_xg")).alias("total_rapm_context_xg"),
            (pl.col("off_coeff_corsi") - pl.col("def_coeff_corsi")).alias("total_rapm_corsi"),
            (pl.col("off_coeff_goals") - pl.col("def_coeff_goals")).alias("total_rapm_goals"),
        ]
    )

    # Career RAPM: TOI-weighted mean of per-season coefficients, stored as season=0 rows.
    # Groups across all seasons and teams for a given (session, player, situation).
    # Uses toi_minutes as the weight so that seasons with more ice time carry more influence.
    coeff_cols = [c for c in final_db.columns if c.startswith(("off_coeff", "def_coeff", "total_rapm"))]

    career_rows = (
        final_db
        .group_by(["session", "player", "pos", "situation"])
        .agg(
            [
                pl.lit(0).alias("season"),
                pl.lit(None).cast(pl.Utf8).alias("team"),
                pl.col("toi_minutes").sum(),
            ] + [
                (
                    (pl.col(c) * pl.col("toi_minutes")).sum() /
                    (pl.col("toi_minutes") * pl.col(c).is_not_null().cast(pl.Float64)).sum()
                ).alias(c)
                for c in coeff_cols
            ]
        )
    )

    final_db = pl.concat([final_db, career_rows], how="diagonal_relaxed")

    skip = {"season", "session", "player", "team", "pos", "situation", "toi_minutes"}
    z_exprs = [
        (
            (pl.col(col) - pl.col(col).mean().over(["season", "session", "situation", "pos"]))
            / pl.col(col).std().over(["season", "session", "situation", "pos"])
        ).alias(f"{col}_z")
        for col in final_db.columns
        if any(m in col for m in ["context_xg", "corsi", "goals"]) and col not in skip
    ]
    final_db = final_db.with_columns(z_exprs)

    return final_db


def main():
    """Function to run and process all regressions."""
    base_dir = Path(__file__).parent.parent
    rapm_pbp_dir = base_dir / "data" / "rapm" / "pbp"

    pbp_files = sorted(rapm_pbp_dir.glob("pbp_*.parquet"))
    if not pbp_files:
        raise FileNotFoundError(
            f"No enriched PBP parquets found in {rapm_pbp_dir}. Run rapm/prep_pbp.py first."
        )

    all_pbp_lazy = pl.scan_parquet(pbp_files)
    position_map = build_position_map(all_pbp_lazy)
    toi_limits = RAPM_TOI_LIMITS

    seasons = sorted(cast(pl.DataFrame, all_pbp_lazy.select(pl.col("season")).unique().collect())["season"].to_list())

    metrics = ["context_xg", "corsi", "goals"]
    sessions = ["R", "P"]
    situations = ["EV", "PP", "SH"]

    stints_directory = base_dir / "data" / "rapm" / "stints"

    regression_results = run_all_regressions(
        stints_directory=stints_directory,
        position_map=position_map,
        seasons=seasons,
        sessions=sessions,
        situations=situations,
        metrics=metrics,
        toi_limits=toi_limits,
        decay_halflife=60.0,
    )

    final_results_path = base_dir / "data" / "rapm" / "rapm_by_season.parquet"
    regression_results.write_parquet(final_results_path, mkdir=True)


if __name__ == "__main__":
    main()