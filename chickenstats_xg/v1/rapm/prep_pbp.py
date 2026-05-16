"""Enrich per-year raw PBP with base_xg scores and write to the RAPM input directory.

Reads the frozen base_xg scored parquets (one per strength state) produced by
base_xg/score.py, joins them onto each per-year raw PBP file by (game_id, event_idx),
and writes the enriched file to:

    1_0_0/data/rapm/pbp/pbp_{year}.parquet

Fenwick events receive their base_xg probability; all other events get 0.0.
These enriched parquets are the canonical input to rapm/process_stints.py.

Run this after base_xg/score.py whenever the base_xg model is updated.

Usage:
    python rapm/prep_pbp.py
    python rapm/prep_pbp.py --years 2023 2024
"""

import argparse
from pathlib import Path

import polars as pl
from chickenstats.utilities import ChickenProgress
from rich.progress import TaskID


def load_scored_xg(scored_dir: Path) -> pl.DataFrame:
    """Combine all five strength-state scored parquets into a single (game_id, event_idx) → base_xg map."""
    strengths = ["even_strength", "powerplay", "shorthanded", "empty_for", "empty_against"]
    frames = [
        pl.scan_parquet(scored_dir / f"{s}.parquet").select(["game_id", "event_idx", "base_xg"])
        for s in strengths
        if (scored_dir / f"{s}.parquet").exists()
    ]
    if not frames:
        raise FileNotFoundError(
            f"No scored base_xg parquets found in {scored_dir}. Run base_xg/score.py first."
        )
    # event_idx is unique within a game, so (game_id, event_idx) should already be unique
    # across strength states. The group_by handles any theoretical overlap at no cost.
    return (
        pl.concat(frames)
        .group_by(["game_id", "event_idx"])
        .agg(pl.col("base_xg").sum())
        .collect()
    )


def enrich_year(
    year_file: Path,
    scored_xg: pl.DataFrame,
    out_dir: Path,
) -> None:
    """Join base_xg onto a single per-year raw PBP file and write the result."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich per-year raw PBP with base_xg scores.")
    parser.add_argument(
        "--years", "-y", type=int, nargs="+",
        help="NHL season start-years to process (e.g. 2023 for 2023-24). Defaults to all available.",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    pbp_dir = base_dir.parent / "raw_data" / "pbp"
    scored_dir = base_dir / "data" / "base_xg" / "scored"
    out_dir = base_dir / "data" / "rapm" / "pbp"

    pbp_files = sorted(pbp_dir.glob("pbp_*.parquet"))
    if args.years:
        pbp_files = [p for p in pbp_files if int(p.stem.split("_")[1]) in args.years]
    if not pbp_files:
        print("No raw PBP files found for the requested years.")
        return

    with ChickenProgress(speed_estimate_period=300) as progress:
        task: TaskID = progress.add_task("Loading scored base_xg...", total=len(pbp_files))
        scored_xg = load_scored_xg(scored_dir)
        for pbp_file in pbp_files:
            progress.update(task, description=f"Enriching {pbp_file.stem}...", refresh=True)
            enrich_year(pbp_file, scored_xg, out_dir)
            progress.update(task, advance=1, refresh=True)
        progress.update(task, description=f"Finished enriching {len(pbp_files)} PBP files", refresh=True)


if __name__ == "__main__":
    main()