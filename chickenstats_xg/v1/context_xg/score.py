"""Score all PBP with the frozen context_xg gbtree model.

Reads base_xg scored parquets (which contain calibrated base_xg + all sequence
features), computes logit_base_xg, passes it as both base_margin (residual
learning from T1 prior) and a feature (preserving interaction constraint groups),
applies Platt calibration, overrides training-era shots with OOF predictions,
and writes:

    data/context_xg/scored/{strength}.parquet

This is the primary input to pred_goal/process_data.py (as the base_margin
prior that replaces the old single-tier base_xg).

Usage:
    python context_xg/score.py --strength even_strength
    python context_xg/score.py --all
    python context_xg/score.py --all --years 2024 2025
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from chickenstats.utilities import ChickenProgress

from chickenstats_xg.v1.config import CONTEXT_XG_FEATURE_COLUMNS, STRENGTHS
from chickenstats_xg.v1.utils.artifacts import load_model_artifacts
from chickenstats_xg.v1.utils.transforms import apply_fixed_categoricals, logit
from chickenstats_xg.v1.utils.rapm import prep_rapm
from chickenstats_xg.v1.utils.scoring import apply_oof_predictions


def score_strength(
    strength: str,
    base_xg_scored_dir: Path,
    models_dir: Path,
    scored_dir: Path,
) -> bool:
    model_path = models_dir / strength / "model.ubj"
    if not model_path.exists():
        print(f"  [{strength}] No model at {model_path} — run context_xg/finalize.py first.")
        return False

    src_path = base_xg_scored_dir / f"{strength}.parquet"
    if not src_path.exists():
        print(f"  [{strength}] No base_xg scored parquet at {src_path} — run base_xg/score.py first.")
        return False

    df = pd.read_parquet(src_path)

    # Compute logit_base_xg — used as both base_margin (residual learning) and
    # a feature (preserving interaction constraint groups). Clipped to ±4.0 to
    # match process_data.py and prevent base_margin spikes for near-certain shots
    # (e.g. empty_against point-blank) that would pin raw predictions at 1.0.
    logit_bm = np.clip(logit(df["base_xg"].to_numpy()), -4.0, 4.0)
    df = df.assign(logit_base_xg=logit_bm)

    feat_cols = [c for c in CONTEXT_XG_FEATURE_COLUMNS if c in df.columns]
    X = df[feat_cols].copy()
    X = apply_fixed_categoricals(X, strength)

    model, calibrator = load_model_artifacts(models_dir, strength)
    if calibrator is None:
        print(f"  [{strength}] WARNING: no calibrator found — context_xg will be uncalibrated.")

    raw_probs = model.predict_proba(X, base_margin=logit_bm)[:, 1]
    context_xg = calibrator.predict_proba(raw_probs.reshape(-1, 1))[:, 1] if calibrator else raw_probs

    df = (
        df.assign(context_xg=context_xg)
        .sort_values(["season", "game_id", "period", "period_seconds"])
        .reset_index(drop=True)
    )

    # Override training-era shots with calibrated OOF predictions
    df = apply_oof_predictions(df, models_dir, strength, "context_xg")

    scored_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(scored_dir / f"{strength}.parquet", index=False)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Score PBP with the frozen context_xg model.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--strength", "-s", type=str, choices=STRENGTHS)
    group.add_argument("--all", "-a", action="store_true")
    parser.add_argument(
        "--no-rapm",
        action="store_true",
        help="Skip RAPM PBP enrichment after scoring (only applies with --all).",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    base_xg_scored_dir = base_dir / "data" / "base_xg" / "scored"
    models_dir = base_dir / "models" / "context_xg"
    scored_dir = base_dir / "data" / "context_xg" / "scored"

    targets = STRENGTHS if args.all else [args.strength]
    scored_count = 0
    with ChickenProgress() as progress:
        task = progress.add_task(f"Scoring {targets[0]}...", total=len(targets))
        for strength in targets:
            progress.update(task, description=f"Scoring {strength}...", refresh=True)
            scored_count += score_strength(strength, base_xg_scored_dir, models_dir, scored_dir)
            progress.update(task, advance=1, refresh=True)
        progress.update(task, description="Finished scoring all strength states", refresh=True)

    if args.all and not args.no_rapm:
        if scored_count == 0:
            print("  [rapm] No strengths scored — skipping RAPM PBP enrichment.")
        else:
            rapm_dir = base_dir / "data" / "rapm" / "pbp"
            prep_rapm(rapm_dir, scored_dir, "context_xg")


if __name__ == "__main__":
    main()
