"""Shared scoring helpers for base_xg and context_xg score.py."""

from pathlib import Path

import pandas as pd


def apply_oof_predictions(
    df: pd.DataFrame,
    models_dir: Path,
    strength: str,
    pred_col: str,
) -> pd.DataFrame:
    """Override in-sample predictions with calibrated OOF values where available.

    Training shots scored by the final model are biased (in-sample). This replaces
    them with the unbiased OOF predictions written by finalize.py. Earliest-fold
    shots not covered by any validation fold are left at the final-model score.
    """
    oof_path = models_dir / strength / "oof.parquet"
    if oof_path.exists():
        oof_df = pd.read_parquet(oof_path).dropna(subset=[pred_col])
        oof_map = oof_df.set_index(["game_id", "event_idx"])[pred_col]
        idx = df.set_index(["game_id", "event_idx"]).index
        in_oof = idx.isin(oof_map.index)
        df.loc[in_oof, pred_col] = oof_map.reindex(idx[in_oof]).values
        print(f"  [{strength}] {in_oof.sum():,} training shots replaced with OOF {pred_col} predictions.")
    return df
