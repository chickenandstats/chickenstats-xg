
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

import numpy as np
import pandas as pd

from chickenstats_xg.v1.config import CONTEXT_XG_FEATURE_COLUMNS, HOLD_OUT_SEASON, PASSTHROUGH_COLS, STRENGTHS

_BM_EPS = 1e-7
KEEP_COLS = ["season", "goal", "base_xg"] + CONTEXT_XG_FEATURE_COLUMNS + PASSTHROUGH_COLS


def process_strength(strength: str, scored_dir: Path, out_dir: Path) -> None:
    scored_path = scored_dir / f"{strength}.parquet"
    if not scored_path.exists():
        print(f"  [{strength}] No scored parquet at {scored_path} — run base_xg/finalize.py first.")
        return

    df = pd.read_parquet(scored_path)

    # Compute logit_base_xg — T1 prior passed as a learnable feature in each
    # flag constraint group so gbtree can learn quality-conditional flag effects.
    p = df["base_xg"].clip(_BM_EPS, 1 - _BM_EPS)
    df = df.assign(logit_base_xg=np.log(p / (1 - p)))

    # Keep only the columns context_xg needs; tolerate missing optional cols
    keep = [c for c in KEEP_COLS if c in df.columns]
    df = df[keep]

    train = df[df["season"] < HOLD_OUT_SEASON].reset_index(drop=True)
    hold_out = df[df["season"] >= HOLD_OUT_SEASON].reset_index(drop=True)

    for split, name in [(train, "train"), (hold_out, "hold_out")]:
        dest = out_dir / name
        dest.mkdir(parents=True, exist_ok=True)
        split.to_parquet(dest / f"{strength}.parquet", index=False)
        print(f"  [{strength}] {name}: {len(split):,} shots → {dest / f'{strength}.parquet'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build context_xg training data from base_xg scored parquets.")
    parser.add_argument("--strength", "-s", type=str, choices=STRENGTHS, default=None,
                        help="Single strength state. Defaults to all five.")
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    scored_dir = base_dir / "data" / "base_xg" / "scored"
    out_dir = base_dir / "data" / "context_xg"

    targets = [args.strength] if args.strength else STRENGTHS
    print(f"Processing {len(targets)} strength state(s)...")
    for strength in targets:
        process_strength(strength, scored_dir, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
