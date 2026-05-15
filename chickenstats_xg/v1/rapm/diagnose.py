
"""RAPM regression quality diagnostic.

Run from repo root:
    uv run python 1_0_0/rapm/diagnose.py
    uv run python 1_0_0/rapm/diagnose.py --situation PP

Checks (run once after regressions.py):
  1. Coefficient range     — z-score outliers in per-season EV R off_coeff_context_xg
  2. Positional plausibility — mean F offense > mean D offense (EV R)
  3. YOY stability         — Pearson r of consecutive-season EV R off_coeff_context_xg
  4. Pred_goal coverage    — fraction of pred_goal train shots with non-null shooter RAPM

Informational (no pass/fail):
  5. Top / bottom leaderboard — career EV R total_rapm_context_xg, min 200 min TOI
  6. Per-season mean / std of coefficients — spots structural shifts across eras

What to watch for:
  - |z| > 4: player-season with extreme coefficient (data issue or tiny sample)
  - YOY r < 0.15: coefficients are essentially random (no talent signal)
  - YOY r > 0.70: suspiciously stable (under-regularised ridge)
  - Coverage < 60%: too many shots without RAPM features for pred_goal to exploit
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

_BASE = _Path(__file__).parent.parent
RAPM_PATH = _BASE / "data" / "rapm" / "rapm_by_season.parquet"
PRED_GOAL_TRAIN_DIR = _BASE / "data" / "pred_goal" / "train"

SITUATIONS = ["EV", "PP", "SH"]

# Position mapping: the parquet stores C / L / R / D, not F
FORWARD_POSITIONS = {"C", "L", "R", "LW", "RW", "W"}

# Outlier thresholds (applied on the parquet's pre-computed z-scores).
# A single elite player at z>6 is expected (McDavid, Tkachuk peak seasons).
# Only flag FAIL when multiple extreme outliers suggest a data / regularisation issue.
Z_WARN = 4.0
Z_FAIL = 6.0
Z_FAIL_COUNT = 5   # must have this many |z|>Z_FAIL to trigger FAIL
Z_WARN_PCT   = 5.0 # % of player-seasons with |z|>Z_WARN to trigger WARN

# YOY Pearson r thresholds (EV R consecutive seasons)
YOY_LOW_FAIL  = 0.05
YOY_LOW_WARN  = 0.15
YOY_HIGH_WARN = 0.70
YOY_HIGH_FAIL = 0.85

# Coverage: % of pred_goal train shots with non-null shooter_rapm_xg_off
COVERAGE_WARN = 60.0
COVERAGE_FAIL = 40.0

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


def _icon(s: str) -> str:
    return {"PASS": "✅", "WARN": "⚠️ ", "FAIL": "❌"}.get(s, "—")


# ── Check 1: coefficient range / outliers ─────────────────────────────────────

def check_coeff_range(df: pd.DataFrame, situation: str) -> str:
    sub = df[(df["session"] == "R") & (df["situation"] == situation) & (df["season"] != 0)]
    coeff_col, z_col = "off_coeff_context_xg", "off_coeff_context_xg_z"

    if coeff_col not in sub.columns:
        print(f"\n  Coefficient range: {coeff_col} missing — skip")
        return WARN

    vals = sub[coeff_col].dropna()
    zvals = sub[z_col].dropna() if z_col in sub.columns else pd.Series(dtype=float)

    n_warn = int((zvals.abs() > Z_WARN).sum()) if len(zvals) else 0
    n_fail = int((zvals.abs() > Z_FAIL).sum()) if len(zvals) else 0
    pct_warn = 100 * n_warn / len(vals) if len(vals) else 0.0

    status = FAIL if n_fail >= Z_FAIL_COUNT else (WARN if pct_warn > Z_WARN_PCT else PASS)

    print(f"\n  ── Coefficient range ({situation} R — off_coeff_context_xg) ────────────────")
    print(f"     player-seasons: {len(vals):,}   mean: {vals.mean():.4f}   std: {vals.std():.4f}")
    print(f"     min: {vals.min():.4f}   p1: {vals.quantile(0.01):.4f}   p99: {vals.quantile(0.99):.4f}   max: {vals.max():.4f}")
    if len(zvals):
        print(f"     |z| > {Z_WARN}: {n_warn:,}  |z| > {Z_FAIL}: {n_fail:,}")
    print(f"  {_icon(status)} Coefficient range [{status}]")
    return status


# ── Check 2: positional plausibility ──────────────────────────────────────────

def check_positional(df: pd.DataFrame) -> str:
    sub = df[(df["session"] == "R") & (df["situation"] == "EV") & (df["season"] != 0)]
    col = "off_coeff_context_xg"

    if col not in sub.columns or "pos" not in sub.columns:
        print(f"\n  Positional plausibility: required columns missing — skip")
        return WARN

    sub = sub[[col, "pos"]].dropna()
    sub["_pos_group"] = sub["pos"].apply(lambda p: "F" if p in FORWARD_POSITIONS else p)

    by_pos = (
        sub.groupby("_pos_group")[col]
        .agg(n="count", mean_off="mean", std_off="std")
        .reset_index()
        .sort_values("mean_off", ascending=False)
    )

    f_mean = sub.loc[sub["_pos_group"] == "F", col].mean()
    d_mean = sub.loc[sub["_pos_group"] == "D", col].mean()

    status = PASS if (not np.isnan(f_mean) and not np.isnan(d_mean) and f_mean > d_mean) else WARN

    print(f"\n  ── Positional plausibility (EV R off_coeff_context_xg) ────────────────────")
    print(f"  {'Pos':>4}  {'n':>6}  {'mean_off':>10}  {'std_off':>10}")
    for _, row in by_pos.iterrows():
        print(f"  {row['_pos_group']:>4}  {int(row['n']):>6,}  {row['mean_off']:>10.4f}  {row['std_off']:>10.4f}")
    print(f"     → F mean {f_mean:.4f} {'>' if f_mean > d_mean else '<='} D mean {d_mean:.4f}")
    print(f"  {_icon(status)} Positional plausibility [{status}]")
    return status


# ── Check 3: YOY stability ────────────────────────────────────────────────────

def check_yoy_stability(df: pd.DataFrame) -> str:
    sub = df[(df["session"] == "R") & (df["situation"] == "EV") & (df["season"] != 0)].copy()
    col = "off_coeff_context_xg"

    if col not in sub.columns:
        print(f"\n  YOY stability: {col} missing — skip")
        return WARN

    sub = sub[["player", "season", col]].dropna(subset=[col]).sort_values(["player", "season"])
    sub["_next_season"] = sub.groupby("player")["season"].shift(-1)
    sub["_next_coeff"] = sub.groupby("player")[col].shift(-1)

    pairs = sub.dropna(subset=["_next_coeff"])
    # Season format is YYYYYYYY (e.g. 20112012). Consecutive seasons differ by 10001.
    pairs = pairs[pairs["_next_season"] == pairs["season"] + 10001]

    if len(pairs) < 30:
        print(f"\n  YOY stability: too few consecutive pairs ({len(pairs)}) — skip")
        return WARN

    r, pval = pearsonr(pairs[col], pairs["_next_coeff"])

    if r < YOY_LOW_FAIL or r > YOY_HIGH_FAIL:
        status = FAIL
    elif r < YOY_LOW_WARN or r > YOY_HIGH_WARN:
        status = WARN
    else:
        status = PASS

    diagnosis = (
        "no talent signal (random)" if r < YOY_LOW_FAIL
        else "weak signal — high turnover" if r < YOY_LOW_WARN
        else "suspiciously stable (under-regularised?)" if r > YOY_HIGH_FAIL
        else "healthy talent persistence"
    )

    print(f"\n  ── YOY stability (EV R off_coeff_context_xg, season S → S+1) ─────────────")
    print(f"     consecutive player-season pairs: {len(pairs):,}")
    print(f"     Pearson r = {r:.3f}  (p = {pval:.2e})  → {diagnosis}")
    print(f"  {_icon(status)} YOY stability [{status}]")
    return status


# ── Check 4: pred_goal RAPM coverage ──────────────────────────────────────────

def check_coverage() -> str:
    strengths = [
        "even_strength", "powerplay", "shorthanded", "empty_for", "empty_against"
    ]
    rapm_col = "shooter_rapm_xg_off"

    print(f"\n  ── RAPM coverage in pred_goal train parquets ───────────────────────────")
    print(f"  {'Strength':<22}  {'n_shots':>9}  {'non_null':>9}  {'cover%':>8}  status")

    statuses = []
    any_found = False
    for strength in strengths:
        path = PRED_GOAL_TRAIN_DIR / f"{strength}.parquet"
        if not path.exists():
            print(f"  {strength:<22}  (missing — run pred_goal/process_data.py first)")
            continue
        any_found = True
        df = pd.read_parquet(path, columns=[rapm_col])
        n = len(df)
        nn = int(df[rapm_col].notna().sum())
        pct = 100 * nn / n if n else 0.0
        s = PASS if pct >= COVERAGE_WARN else (WARN if pct >= COVERAGE_FAIL else FAIL)
        statuses.append(s)
        print(f"  {strength:<22}  {n:>9,}  {nn:>9,}  {pct:>7.1f}%  {_icon(s)} {s}")

    if not any_found:
        print("     No train parquets found — run pred_goal/process_data.py first")
        return WARN

    overall = FAIL if FAIL in statuses else (WARN if WARN in statuses else PASS)
    print(f"  {_icon(overall)} RAPM coverage [{overall}]")
    return overall


# ── Informational: leaderboard ─────────────────────────────────────────────────

def show_leaderboard(df: pd.DataFrame, n: int = 10) -> None:
    career = df[
        (df["session"] == "R") & (df["situation"] == "EV")
        & (df["season"] == 0) & (df["toi_minutes"] >= 200)
    ].copy()
    col = "total_rapm_context_xg"
    if col not in career.columns:
        return

    career = career[["player", "pos", "toi_minutes", col]].dropna(subset=[col])
    career["_pos_group"] = career["pos"].apply(lambda p: "F" if p in FORWARD_POSITIONS else p)
    career = career.sort_values(col, ascending=False).reset_index(drop=True)

    print(f"\n  ── Career EV RAPM leaderboard (≥200 min, context_xg total_rapm) ──────────")
    print(f"  {'Rank':>4}  {'Player':>10}  {'Pos':>3}  {'TOI_min':>8}  {'total_rapm':>10}")

    for i, row in career.head(n).iterrows():
        print(f"  {i+1:>4}  {str(row['player']):>10}  {str(row['_pos_group']):>3}  "
              f"{int(row['toi_minutes']):>8,}  {row[col]:>10.4f}")
    print(f"  {'···':>4}")
    bottom = career.tail(n).iloc[::-1].reset_index(drop=True)
    for j, row in bottom.iterrows():
        rank = len(career) - n + j + 1
        print(f"  {rank:>4}  {str(row['player']):>10}  {str(row['_pos_group']):>3}  "
              f"{int(row['toi_minutes']):>8,}  {row[col]:>10.4f}")


# ── Informational: per-season distribution ────────────────────────────────────

def show_season_distribution(df: pd.DataFrame, situation: str) -> None:
    sub = df[(df["session"] == "R") & (df["situation"] == situation) & (df["season"] != 0)]
    col = "off_coeff_context_xg"
    if col not in sub.columns:
        return

    print(f"\n  ── Per-season distribution ({situation} R off_coeff_context_xg) ─────────────")
    print(f"  {'Season':>8}  {'n':>6}  {'mean':>8}  {'std':>8}  {'p1':>8}  {'p99':>8}")
    for season in sorted(sub["season"].unique()):
        s = sub[sub["season"] == season][col].dropna()
        if len(s) < 10:
            continue
        print(f"  {season:>8}  {len(s):>6,}  {s.mean():>8.4f}  {s.std():>8.4f}  "
              f"{s.quantile(0.01):>8.4f}  {s.quantile(0.99):>8.4f}")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="RAPM quality diagnostic")
    parser.add_argument("--situation", "-s", choices=SITUATIONS, default="EV",
                        help="Primary situation for coefficient range check (default: EV)")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")

    print("=" * 62)
    print("  RAPM quality diagnostic")
    print("=" * 62)

    if not RAPM_PATH.exists():
        print(f"\n  rapm_by_season.parquet not found at {RAPM_PATH}")
        print("  Run rapm/regressions.py first.")
        return

    df = pd.read_parquet(RAPM_PATH)

    statuses: dict[str, str] = {}
    statuses["coeff_range"]  = check_coeff_range(df, args.situation)
    statuses["positional"]   = check_positional(df)
    statuses["yoy_stability"] = check_yoy_stability(df)
    statuses["coverage"]     = check_coverage()

    show_leaderboard(df)
    show_season_distribution(df, args.situation)

    print(f"\n{'=' * 62}")
    print(f"  Summary")
    print(f"{'─' * 62}")
    checks  = ["coeff_range", "positional", "yoy_stability", "coverage"]
    headers = ["coeff_range", "positional", "yoy_stab",     "coverage"]
    print(f"  {'Check':<20}  " + "  ".join(f"{h:>11}" for h in headers))
    row = "  ".join(f"{_icon(statuses.get(c, '—'))+'':>11}" for c in checks)
    print(f"  {'RAPM':<20}  {row}")
    overall = FAIL if FAIL in statuses.values() else (WARN if WARN in statuses.values() else PASS)
    print(f"\n  {_icon(overall)} Overall RAPM: {overall}")
    print(f"{'=' * 62}\n")


if __name__ == "__main__":
    main()