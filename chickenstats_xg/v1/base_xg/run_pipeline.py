"""Orchestrate the base_xg finalize → score → context_xg data prep pipeline.

Runs three steps in sequence:
  1. base_xg/finalize.py       — retrain and freeze best base_xg models (all 5 strengths)
  2. base_xg/score.py          — score raw PBP + enrich RAPM PBP with base_xg
  3. context_xg/process_data.py — split base_xg scored parquets into context_xg train/hold_out

After this pipeline completes, run experiments.py for each context_xg strength state, then:
  - context_xg/run_pipeline.py --version 1.0.0 --no-log

Each step is a subprocess call so output streams live to the terminal and a
failed step exits with a non-zero code rather than silently continuing.

Usage:
    # Full pipeline
    uv run python 1_0_0/base_xg/run_pipeline.py --version 1.0.0

    # Skip MLflow / SHAP logging in finalize
    uv run python 1_0_0/base_xg/run_pipeline.py --version 1.0.0 --no-log

    # Limit scoring to specific seasons
    uv run python 1_0_0/base_xg/run_pipeline.py --version 1.0.0 --years 2023 2024 2025

    # Resume: finalize already ran, start from base_xg scoring
    uv run python 1_0_0/base_xg/run_pipeline.py --version 1.0.0 --skip-finalize

    # Resume: finalize and score done, only run context_xg data prep
    uv run python 1_0_0/base_xg/run_pipeline.py --version 1.0.0 --skip-finalize --skip-score
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).parent


def _run(cmd: list[str], label: str) -> None:
    """Run a subprocess step, streaming output. Exit if it fails."""
    print(f"\n{'=' * 62}")
    print(f"  {label}")
    print(f"  $ {' '.join(cmd)}")
    print(f"{'=' * 62}\n")
    t0 = time.monotonic()
    result = subprocess.run(cmd)
    elapsed = time.monotonic() - t0
    mins, secs = divmod(int(elapsed), 60)
    duration = f"{mins}m {secs}s" if mins else f"{secs}s"
    if result.returncode != 0:
        print(f"\n  ❌ {label} FAILED (exit {result.returncode}) after {duration}")
        sys.exit(result.returncode)
    print(f"\n  ✅ {label} done ({duration})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run base_xg finalize → score → context_xg data prep pipeline."
    )
    parser.add_argument("--version", "-v", required=True,
                        help="Model version string passed to finalize.py (e.g. 1.0.0)")
    parser.add_argument("--years", "-y", type=int, nargs="+",
                        help="Season end-years to score (e.g. 2023 2024 2025). Default: all.")
    parser.add_argument("--no-log", action="store_true",
                        help="Pass --no-log to finalize.py (skip MLflow / SHAP).")
    parser.add_argument("--top-n", "-n", type=int, default=15,
                        help="Number of top CV PR-AUC trials to screen by calibrated hold-out log loss (default: 15).")
    parser.add_argument("--skip-finalize", action="store_true",
                        help="Skip step 1 (base_xg finalize). Use if models already frozen.")
    parser.add_argument("--skip-score", action="store_true",
                        help="Skip step 2 (base_xg score + RAPM PBP). Use if scored parquets exist.")
    parser.add_argument("--skip-process", action="store_true",
                        help="Skip step 3 (context_xg/process_data.py). Use if context_xg train/hold_out already split.")
    args = parser.parse_args()

    py = sys.executable

    # ── Step 1: finalize base_xg ───────────────────────────────────────────────
    if not args.skip_finalize:
        cmd = [py, str(_HERE / "finalize.py"), "--all", "--version", args.version]
        if args.no_log:
            cmd.append("--no-log")
        cmd += ["--top-n", str(args.top_n)]
        _run(cmd, "Step 1 — base_xg finalize")
    else:
        print("\n  Skipping step 1 (base_xg finalize)")

    # ── Step 2: score raw PBP + base_xg RAPM PBP enrichment ───────────────────
    if not args.skip_score:
        cmd = [py, str(_HERE / "score.py"), "--all"]
        if args.years:
            cmd += ["--years"] + [str(y) for y in args.years]
        _run(cmd, "Step 2 — base_xg score + RAPM PBP enrichment")
    else:
        print("\n  Skipping step 2 (base_xg score + RAPM PBP)")

    # ── Step 3: build context_xg train/hold_out from base_xg scored parquets ──
    if not args.skip_process:
        _run([py, str(_HERE.parent / "context_xg" / "process_data.py")], "Step 3 — context_xg process data")
    else:
        print("\n  Skipping step 3 (context_xg process data)")

    print(f"\n{'=' * 62}")
    print("  Pipeline complete.")
    print("  Next: run experiments.py for each context_xg strength state,")
    print("        then: uv run python 1_0_0/context_xg/run_pipeline.py --version", args.version)
    print(f"{'=' * 62}\n")


if __name__ == "__main__":
    main()