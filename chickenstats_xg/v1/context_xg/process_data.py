"""Split base_xg scored parquets into context_xg train / hold_out inputs.

base_xg/scored/{strength}.parquet contains all Fenwick shots with calibrated
base_xg predictions (OOF-corrected for training shots). This script splits by
season and writes context_xg/{train,hold_out}/{strength}.parquet — the input
that experiments.py (Optuna tuning) and context_xg/finalize.py both read.

Run after base_xg/finalize.py and base_xg/score.py have completed.

Usage:
    python context_xg/process_data.py
    python context_xg/process_data.py --strength even_strength
"""

import argparse
from pathlib import Path

import polars as pl
from chickenstats.utilities import ChickenProgress

from chickenstats_xg.v1.config import CONTEXT_XG_FEATURE_COLUMNS, PASSTHROUGH_COLS, STRENGTHS
from chickenstats_xg.v1.utils.data_splitting import write_train_holdout_split

_BM_EPS = 1e-7
# Cap logit_base_xg at ±4.0 (sigmoid(4) ≈ 0.982) to prevent numerical spikes.
# Without this, empty_against base_xg values near 1.0 (point-blank empty-net shots)
# produce logit ≈ 16, pinning XGBoost raw predictions at exactly 1.0 and making
# structural_flaw_penalty artificially large in screen_trials. Even_strength is
# unaffected (its logit_base_xg max is ~0.9).
_LOGIT_CAP = 4.0
KEEP_COLS = ["season", "goal", "base_xg"] + CONTEXT_XG_FEATURE_COLUMNS + PASSTHROUGH_COLS


def process_strength(strength: str, scored_dir: Path, out_dir: Path) -> None:
    scored_path = scored_dir / f"{strength}.parquet"
    if not scored_path.exists():
        print(f"  [{strength}] No scored parquet at {scored_path} — run base_xg/finalize.py first.")
        return

    df = pl.read_parquet(scored_path)

    # Compute logit_base_xg — T1 prior passed as a learnable feature in each
    # flag constraint group so gbtree can learn quality-conditional flag effects.
    p = pl.col("base_xg").clip(_BM_EPS, 1 - _BM_EPS)
    df = df.with_columns((p / (1.0 - p)).log().clip(-_LOGIT_CAP, _LOGIT_CAP).alias("logit_base_xg"))

    # Keep only the columns context_xg needs; tolerate missing optional cols
    keep = [c for c in KEEP_COLS if c in df.columns]
    df = df.select(keep)

    write_train_holdout_split(df, out_dir / "train", out_dir / "hold_out", strength)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build context_xg training data from base_xg scored parquets.")
    parser.add_argument(
        "--strength",
        "-s",
        type=str,
        choices=STRENGTHS,
        default=None,
        help="Single strength state. Defaults to all five.",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    scored_dir = base_dir / "data" / "base_xg" / "scored"
    out_dir = base_dir / "data" / "context_xg"

    targets = [args.strength] if args.strength else STRENGTHS
    with ChickenProgress() as progress:
        task = progress.add_task(f"Processing {targets[0]}...", total=len(targets))
        for strength in targets:
            progress.update(task, description=f"Processing {strength}...", refresh=True)
            process_strength(strength, scored_dir, out_dir)
            progress.update(task, advance=1, refresh=True)
        progress.update(task, description="Finished processing context_xg data", refresh=True)


if __name__ == "__main__":
    main()
