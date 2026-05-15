"""Orchestrate the context_xg finalize → score → RAPM → pred_goal data prep pipeline.

Runs five steps in sequence:
  1. context_xg/finalize.py     — retrain and freeze best context_xg models (all 5 strengths)
  2. context_xg/score.py        — score all PBP + enrich RAPM PBP with context_xg
  3. rapm/process_stints.py     — build RAPM stints from context_xg-enriched PBP
  4. rapm/regressions.py        — run RAPM ridge regressions
  5. pred_goal/process_data.py  — split context_xg scored parquets into pred_goal train/hold_out

After this pipeline completes, run experiments.py for each pred_goal strength state, then:
  - pred_goal/finalize.py --all --version <version>
  - pred_goal/score.py --all

Each step is a subprocess call so output streams live to the terminal and a
failed step exits with a non-zero code rather than silently continuing.

Usage:
    # Full pipeline
    uv run python 1_0_0/context_xg/run_pipeline.py --version 1.0.0

    # Skip MLflow logging in finalize
    uv run python 1_0_0/context_xg/run_pipeline.py --version 1.0.0 --no-log

    # Limit scoring to specific seasons
    uv run python 1_0_0/context_xg/run_pipeline.py --version 1.0.0 --years 2023 2024 2025

    # Resume: finalize already ran, start from context_xg scoring
    uv run python 1_0_0/context_xg/run_pipeline.py --version 1.0.0 --skip-finalize

    # Resume: finalize + score done, start from RAPM stints
    uv run python 1_0_0/context_xg/run_pipeline.py --version 1.0.0 --skip-finalize --skip-score

    # Resume: RAPM done, only run pred_goal data prep
    uv run python 1_0_0/context_xg/run_pipeline.py --version 1.0.0 --skip-finalize --skip-score --skip-rapm
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
        description="Run context_xg finalize → score → RAPM → pred_goal data prep pipeline."
    )
    parser.add_argument("--version", "-v", required=True,
                        help="Model version string passed to finalize.py (e.g. 1.0.0)")
    parser.add_argument("--years", "-y", type=int, nargs="+",
                        help="Season end-years to score (e.g. 2023 2024 2025). Default: all.")
    parser.add_argument("--no-log", action="store_true",
                        help="Pass --no-log to finalize.py (skip MLflow logging).")
    parser.add_argument("--top-n", "-n", type=int, default=15,
                        help="Number of top CV PR-AUC trials to screen by calibrated hold-out log loss (default: 15). Use 150 for context_xg (see Issue 12).")
    parser.add_argument("--skip-finalize", action="store_true",
                        help="Skip step 1 (context_xg finalize). Use if models already frozen.")
    parser.add_argument("--skip-score", action="store_true",
                        help="Skip step 2 (context_xg score + RAPM PBP enrichment). Use if scored parquets exist.")
    parser.add_argument("--skip-rapm", action="store_true",
                        help="Skip steps 3–4 (RAPM stints + regressions). Use if RAPM outputs already built.")
    parser.add_argument("--skip-process", action="store_true",
                        help="Skip step 5 (pred_goal/process_data.py). Use if pred_goal train/hold_out already split.")
    args = parser.parse_args()

    py = sys.executable
    rapm_dir = _HERE.parent / "rapm"
    pred_goal_dir = _HERE.parent / "pred_goal"

    # ── Step 1: finalize context_xg ───────────────────────────────────────────
    if not args.skip_finalize:
        cmd = [py, str(_HERE / "finalize.py"), "--all", "--version", args.version]
        if args.no_log:
            cmd.append("--no-log")
        cmd += ["--top-n", str(args.top_n)]
        _run(cmd, "Step 1 — context_xg finalize")
    else:
        print("\n  Skipping step 1 (context_xg finalize)")

    # ── Step 2: score all PBP + enrich RAPM PBP with context_xg ──────────────
    if not args.skip_score:
        cmd = [py, str(_HERE / "score.py"), "--all"]
        if args.years:
            cmd += ["--years"] + [str(y) for y in args.years]
        _run(cmd, "Step 2 — context_xg score + RAPM PBP enrichment")
    else:
        print("\n  Skipping step 2 (context_xg score + RAPM PBP)")

    # ── Steps 3–4: RAPM stints + regressions ─────────────────────────────────
    if not args.skip_rapm:
        _run([py, str(rapm_dir / "process_stints.py")], "Step 3 — RAPM process stints")
        _run([py, str(rapm_dir / "regressions.py")], "Step 4 — RAPM regressions")
    else:
        print("\n  Skipping steps 3–4 (RAPM stints + regressions)")

    # ── Step 5: build pred_goal train/hold_out ────────────────────────────────
    if not args.skip_process:
        _run([py, str(pred_goal_dir / "process_data.py")], "Step 5 — pred_goal process data")
    else:
        print("\n  Skipping step 5 (pred_goal process data)")

    print(f"\n{'=' * 62}")
    print("  Pipeline complete.")
    print("  Next: run experiments.py for each pred_goal strength state,")
    print("        then: pred_goal/finalize.py --all --version", args.version)
    print("              pred_goal/score.py --all")
    print(f"{'=' * 62}\n")


if __name__ == "__main__":
    main()