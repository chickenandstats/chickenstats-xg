"""Shared RAPM enrichment helpers for base_xg and context_xg score.py."""

from pathlib import Path

import polars as pl
from chickenstats.utilities import ChickenProgress
from rich.progress import TaskID

from chickenstats_xg.v1.config import STRENGTHS


def load_scored_xg(scored_dir: Path, pred_col: str) -> pl.DataFrame:
    """Combine all five strength-state scored parquets into a (game_id, event_idx) → pred_col map."""
    frames = [
        pl.scan_parquet(scored_dir / f"{s}.parquet")
        .select(["game_id", "event_idx", pred_col])
        .with_columns(pl.col(pred_col).cast(pl.Float64))
        for s in STRENGTHS
        if (scored_dir / f"{s}.parquet").exists()
    ]
    return pl.concat(frames).group_by(["game_id", "event_idx"]).agg(pl.col(pred_col).sum()).collect()


def enrich_rapm_year(year_file: Path, scored_xg: pl.DataFrame, out_dir: Path, pred_col: str) -> None:
    """Join pred_col onto a single per-year PBP file and write the enriched parquet."""
    df = pl.read_parquet(year_file)
    if pred_col in df.columns:
        df = df.drop(pred_col)
    df = df.join(scored_xg, on=["game_id", "event_idx"], how="left").with_columns(pl.col(pred_col).fill_null(0.0))
    out_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_dir / year_file.name)


def prep_rapm(
    source_dir: Path,
    scored_dir: Path,
    pred_col: str,
    years: list[int] | None = None,
) -> None:
    """Build RAPM input parquets enriched with pred_col predictions.

    Args:
        source_dir: Directory containing per-year PBP parquets to enrich.
                    base_xg passes raw_dir; context_xg passes rapm_dir (in-place update).
        scored_dir: Directory with strength-split scored parquets (source of predictions).
        pred_col:   Prediction column name ("base_xg" or "context_xg").
        years:      Season end-years to include. None → all found in source_dir.
    """
    out_dir = scored_dir.parent.parent / "rapm" / "pbp"

    pbp_files = sorted(source_dir.glob("pbp_*.parquet"))
    if years:
        pbp_files = [p for p in pbp_files if int(p.stem.split("_")[1]) in years]
    if not pbp_files:
        print(f"  [rapm] No PBP files found in {source_dir} — skipping RAPM prep.")
        return

    with ChickenProgress(speed_estimate_period=300, transient=True) as progress:
        task: TaskID = progress.add_task(f"Loading scored {pred_col}...", total=len(pbp_files))
        scored_xg = load_scored_xg(scored_dir, pred_col)
        for pbp_file in pbp_files:
            progress.update(task, description=f"Enriching {pbp_file.stem}...", refresh=True)
            enrich_rapm_year(pbp_file, scored_xg, out_dir, pred_col)
            progress.update(task, advance=1, refresh=True)
        progress.update(task, description=f"Finished enriching RAPM PBP with {pred_col}", refresh=True)
