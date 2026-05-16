# chickenstats-xg — v1.0.0 Cascade xG Model Reference

> Last updated: 2026-05-15

This document is the definitive reference for the `v1.0.0` cascade xG model: what it is, how every script fits together, the full feature pipeline, training instructions, infrastructure requirements, and implementation status.

---

## Table of Contents

1. [Model Overview](#1-model-overview)
2. [Repository Structure](#2-repository-structure)
3. [Data Sources](#3-data-sources)
4. [Feature Engineering Pipeline](#4-feature-engineering-pipeline)
5. [Model Architecture](#5-model-architecture)
6. [Training Infrastructure](#6-training-infrastructure)
7. [Script Reference](#7-script-reference)
8. [Training Execution Guide](#8-training-execution-guide)
9. [Hyperparameter Search Space](#9-hyperparameter-search-space)
10. [Data Directory Structure](#10-data-directory-structure)
11. [RAPM Pipeline Details](#11-rapm-pipeline-details)
12. [Rolling Stats Details](#12-rolling-stats-details)
13. [Leakage Prevention Rules](#13-leakage-prevention-rules)
14. [Live Inference Strategy](#14-live-inference-strategy)
15. [Implementation Status](#15-implementation-status)
16. [Deployment Checklist](#16-deployment-checklist)
17. [Known Gaps & Pending Work](#17-known-gaps--pending-work)

---

## 1. Model Overview

A three-tier cascade that progressively conditions on geometry, sequence context, and player talent:

```
Tier 1 — base_xg  (XGBoost gbtree)
  Shot geometry only (8 features — location, danger zones, shot type)
  → calibrated P(goal | where, shot type)   — no player identity, no shot sequence, no game state
  Output: base_xg ∈ (0, 1), OOF isotonic-calibrated

Tier 2 — context_xg  (XGBoost gbtree, depth=2, 9 flag/state isolation constraint groups)
  logit_base_xg (T1 prior) as BOTH base_margin AND learnable feature + binary flags + game-state
  modifiers + sequence features (20 features total)
  → calibrated P(goal | geometry + game state + sequence context)
  Output: context_xg ∈ (0, 1), pooled OOF + hold-out Platt-calibrated

Tier 3 — pred_goal  (XGBoost gbtree)
  logit(context_xg) as base_margin
  + rolling shooter GxG (goals above context_xg) — career / season / 10g / EWMA windows
  + rolling goalie GSAx (saves above context_xg) — career / season / 10g / EWMA windows
  + RAPM (context_xg xg_off / xg_def for shooter, teammates, opponents × prior season + career)
  → talent residual P(goal | who is shooting, given geometry + sequence context)
  Output: pred_goal ∈ (0, 1), pooled OOF + hold-out Platt-calibrated
```

**Why three tiers?**

The original two-tier architecture (base_xg with all 26 features → pred_goal) had a structural
GOAL event fingerprinting problem. Any gbtree model trained on [GOAL=1, SHOT=0] data can exploit
the fact that GOAL events have dramatically different shot sequence distributions than SHOT events
(rush_attempt is 12.5× higher for goals). By moving sequence features to a **gbtree depth-2 Tier 2
with per-flag interaction constraints**, we eliminate the fingerprint risk: each binary flag
(`is_rebound`, `is_scramble`, `rush_attempt`, `prior_face`) is paired with `logit_base_xg` in its
own constraint group. A depth-2 tree can only learn one quality-conditional flag effect per tree —
structurally impossible to combine two flags on one branch path. The multi-flag Rush+Rebound+<10ft
path that fingerprints goal events requires depth ≥ 3 and cross-group feature combinations, both
blocked by design. See `DECISIONS.md` Issues 7 and 9.

**Cascade mechanics:**
- Each tier's raw prediction is calibrated to true goal rates.
- T2 (context_xg): `logit_base_xg` serves **two complementary roles**:
  (1) as `base_margin` — shifts the gradient at each training example so trees learn the contextual
  residual from the T1 prior (`g = sigmoid(logit_base_xg + F(x)) − y`), collapsing the bimodal cliff
  that occurs when trees must predict the full sigmoid range from scratch; and (2) as a learnable
  feature in the 9 interaction constraint groups — each depth-2 tree can ask "given this shot
  quality, how much does THIS flag/state shift danger?" without logit_base_xg only having one
  split available per group. Both roles are required: base_margin alone (without the feature)
  degenerates each flag group to a single binary split, equivalent to gblinear.
- T3 (pred_goal): `logit(calibrated context_xg)` is passed as `base_margin`. The hard coefficient
  is appropriate here — pred_goal should treat the full geometry+sequence prior as its fixed starting
  point and learn only the talent residual on top.
- `pred_goal/process_data.py` reads from `context_xg/scored/` and renames `context_xg` → `base_xg`
  before writing train parquets, so `experiments.py`'s base_margin extraction is unchanged downstream.

**RAPM target:**
- RAPM regressions use `context_xg` (not `base_xg`) as the xGF regression target.
- This prevents rush/rebound talent effects from being double-counted in RAPM and in pred_goal's base_margin.

**Five variants per model (strength state):**

| Variant | Strength states covered |
|---|---|
| `even_strength` | 5v5, 4v4, 3v3 |
| `powerplay` | 5v4, 4v3, 5v3 |
| `shorthanded` | 4v5, 3v4, 3v5 |
| `empty_for` | Ev5, Ev4, Ev3 |
| `empty_against` | 5vE, 4vE, 3vE |

---

## 2. Repository Structure

```
chickenstats-xg/
│
├── raw_data/                          Shared raw data (all versions)
│   ├── pbp/
│   │   └── pbp_{YYYY}.parquet         One parquet per season end-year (e.g. pbp_2024)
│   └── scrape_raw_data.py             Scraper — reads NHL API, writes to raw_data/pbp/
│
├── chickenstats_xg/                   Python package (editable install: uv pip install -e .)
│   │
│   ├── utilities/                     Pure-function chart + style library (no yellowbrick dependency)
│   │   ├── __init__.py
│   │   ├── charts.py                  7 classifier charts (classification_report, roc_auc, etc.)
│   │   └── style.py                   Palette, colormap, set_style(), NHL team color utilities
│   │
│   ├── nuke_experiment.py             Delete MLflow experiment + Optuna study (version-agnostic)
│   │
│   └── v1/                            v1.0.0 version-specific code and data
│       ├── experiments.py             Optuna tuning for all three tiers
│       ├── config.py                  Feature column lists, strength states, constants
│       │
│       ├── planning/
│       │   ├── MODEL.md               This file
│       │   ├── DECISIONS.md           Architectural decision log (issue history + resolved issues)
│       │   └── DIAGNOSTICS.md         Consolidated diagnostic run results (all three tiers)
│       │
│       ├── utils/                     Shared helpers (extracted from experiments.py 2026-05-15)
│       │   ├── __init__.py
│       │   ├── shot_features.py       prep_data() stateless feature engineering
│       │   ├── artifacts.py           save/load model artifacts + meta.json metadata
│       │   ├── finalize_utils.py      OOF predictions, trial selection, ECE calculation
│       │   ├── calibration.py         IsotonicCalibrator wrapper
│       │   ├── transforms.py          apply_fixed_categoricals, logit
│       │   ├── scoring.py             apply_oof_predictions (OOF override for score.py)
│       │   ├── data_splitting.py      write_train_holdout_split
│       │   ├── rapm.py                RAPM PBP enrichment utilities
│       │   ├── diagnose_utils.py      Shared diagnostic functions + extract_model_hyperparams
│       │   └── log_model.py           CLI: upload local model artifacts to MLflow Model Registry
│       │
│       ├── base_xg/
│       │   ├── process_data.py        Stateless feature engineering → train/hold_out parquets
│       │   ├── finalize.py            Retrain best base_xg, OOF calibrate, freeze model
│       │   ├── score.py               Score arbitrary raw PBP years with frozen base_xg
│       │   ├── diagnose.py            Calibration, lift, and OOF gap diagnostics
│       │   └── run_pipeline.py        Orchestrates finalize → score → context_xg process_data
│       │
│       ├── context_xg/
│       │   ├── process_data.py        Split base_xg/scored/ → context_xg train/hold_out
│       │   ├── finalize.py            Retrain best context_xg gbtree depth-2, pooled OOF+hold-out Platt calibrate, freeze
│       │   ├── score.py               Score base_xg parquets through context_xg; enrich RAPM PBP
│       │   ├── diagnose.py            Calibration, lift, OOF gap, weight concentration diagnostics
│       │   └── run_pipeline.py        Orchestrates finalize → score → RAPM → pred_goal process_data
│       │
│       ├── pred_goal/
│       │   ├── process_data.py        Assemble pred_goal data (context_xg + RAPM + rolling)
│       │   ├── compute_rolling_stats.py  GxG / GSAx rolling windows (career/season/10g/1g/EWMA)
│       │   ├── finalize.py            Retrain best pred_goal → OOF calibrate → SHAP → freeze
│       │   └── diagnose.py            Calibration and lift diagnostics
│       │
│       ├── rapm/
│       │   ├── prep_pbp.py            Join scored base_xg onto raw PBP → data/rapm/pbp/
│       │   ├── process_stints.py      Aggregate enriched PBP → stints (TOI, xGF/A via context_xg, CF/A, GF/A)
│       │   ├── regressions.py         Ridge RAPM × 3 metrics × 3 situations × all seasons
│       │   └── diagnose.py            RAPM diagnostics
│       │
│       ├── models/                    Frozen model artifacts (gitignored)
│       │   ├── base_xg/
│       │   │   └── even_strength/
│       │   │       ├── model.ubj
│       │   │       ├── calibrator.joblib
│       │   │       └── meta.json
│       │   │       ... (5 subdirs, 3 files each)
│       │   ├── context_xg/
│       │   │   └── even_strength/
│       │   │       ├── model.ubj
│       │   │       ├── calibrator.joblib
│       │   │       ├── oof.parquet
│       │   │       └── meta.json
│       │   │       ... (5 subdirs, 4 files each)
│       │   └── pred_goal/
│       │       └── even_strength/
│       │           ├── model.ubj
│       │           ├── calibrator.joblib
│       │           └── meta.json
│       │           ... (5 subdirs, 3 files each)
│       │
│       └── data/                      Version-specific model data (gitignored, large files)
│           ├── base_xg/
│           │   ├── train/             5 parquets (one per strength)
│           │   ├── hold_out/          5 parquets (hold-out season = 2024-25)
│           │   └── scored/            5 parquets + base_xg + event_idx columns (full history)
│           ├── context_xg/
│           │   ├── train/             5 parquets (base_xg + sequence features)
│           │   ├── hold_out/          5 parquets
│           │   └── scored/            5 parquets + context_xg column (full history)
│           ├── pred_goal/
│           │   ├── train/             5 parquets (context_xg as base_xg + all talent features)
│           │   └── hold_out/          5 parquets
│           └── rapm/
│               ├── pbp/               pbp_{YYYY}.parquet — raw PBP enriched with base_xg + context_xg
│               ├── stints/            stints_{YYYY}_{r|p}.parquet (RAPM input)
│               └── rapm_by_season.parquet  All seasons; pivoted to one row per player per situation
│
├── .env.example                       Required env vars template
└── pyproject.toml                     Python dependencies + entry points (uv)
```

---

## 3. Data Sources

### 3.1 Raw Play-by-Play

**Source:** NHL Stats API via the `chickenstats` Python library (`chickenstats.chicken_nhl.Season`, `chickenstats.chicken_nhl.Scraper`).

**Script:** `raw_data/scrape_raw_data.py`

**Output:** `raw_data/pbp/pbp_{YYYY}.parquet` — one file per season. YYYY = season start year (e.g. 2024 = 2024-25 season).

**Coverage:** Seasons 2010 through 2024 (2010-11 through 2024-25). The scraper skips years where the parquet already exists.

**Key columns used downstream:**

| Column | Type | Used for |
|---|---|---|
| `event_idx` | int | Sequential per-game event index; join key between scored parquets and raw PBP |
| `event` | str | Filter to shot/miss/goal/faceoff/etc. |
| `strength_state` | str | Separate by model variant |
| `coords_x`, `coords_y` | float | Geometry features |
| `event_distance`, `event_angle` | float | Geometry features |
| `shot_type` | str | Categorical feature |
| `player_1_position` | str | Shooter position |
| `is_home` | int | Home advantage |
| `score_diff` | int | Score state |
| `zone` | str | Rush detection |
| `danger`, `high_danger` | int | Danger zone flags |
| `goal` | int | Binary target |
| `season`, `game_id`, `period`, `period_seconds`, `game_seconds` | int | Sorting + CV splits |
| `player_1_api_id` | int | Shooter identity (passthrough → rolling stats / RAPM join) |
| `opp_goalie_api_id` | int | Goalie identity (passthrough) |
| `home_on_api_id`, `away_on_api_id` | str | On-ice personnel (passthrough → teammates RAPM) |
| `session` | str | R (regular) / P (playoff) — categorical feature + RAPM filter |
| `base_xg` | float | Written by `rapm/prep_pbp.py` for RAPM stints; from `base_xg/scored/` |
| `event_length` | float | Stint duration for RAPM |
| `home_score_diff` | int | RAPM score state |
| `zone_start` | str | OZS/NZS/DZS for RAPM |
| `home_team`, `away_team` | str | RAPM team context |
| `game_date` | str | Back-to-back detection for RAPM |

### 3.2 Hold-Out Season

The hold-out year is **2024-25** (`HOLD_OUT_SEASON = 20242025`) — the most recent completed season. It is separated from training data before any model sees it. The tuning CV operates only on pre-hold-out seasons.

For `experiments.py` CV, an internal validation split is carved from the training parquets: **train on 2010-11 through 2022-23, test on 2023-24** (chronological, no shuffling).

---

## 4. Feature Engineering Pipeline

All feature engineering is centralised in `prep_data()` in `chickenstats_xg/v1/xg_utils.py`.

### 4.1 Base xG Features — Tier 1 (stateless geometry + game state)

Computed from raw PBP events by `prep_data()` in `xg_utils.py`. Three filters before feature construction:
- Event must be `SHOT`, `FAC`, `HIT`, `BLOCK`, `MISS`, `GIVE`, `TAKE`, or `GOAL`
- Strength state not `1v0` or `EvE`
- `coords_x` and `coords_y` must be non-null

`BASE_XG_FEATURE_COLUMNS` in `config.py` (8 features — pure geometry + shot type only):

| Feature | Description | Notes |
|---|---|---|
| `event_distance` | Distance from net | Monotone constraint: −1 (farther = less likely) |
| `event_angle` | Angle to net | Monotone constraint: −1 (wider = less likely) |
| `coords_x` | X coordinate on ice | Raw geometry |
| `coords_y` | Y coordinate on ice | Raw geometry |
| `abs_y_distance` | Absolute y offset from centre | Supplement to angle |
| `danger` | In danger zone (0/1) | |
| `high_danger` | In high-danger zone (0/1) | Highly predictive |
| `shot_type` | Shot type (wrist, slap, snap, etc.) | Fixed categorical; 11 values |

**Game-state features removed from base_xg (v1.0.0):** `score_diff`, `period`, `period_seconds`, `is_home`, `position`, `strength_state` were moved to `CONTEXT_XG_FEATURE_COLUMNS`. Keeping them in base_xg violated tier purity (base_xg should be a mathematically pristine spatial prior) and caused the empty_against OOF gap FAIL via `score_diff` era fingerprinting. See Issue 10 in `DECISIONS.md`.

**Passthrough columns** (kept in parquets for downstream joins, excluded from training matrix):
`event_idx`, `game_id`, `player_1_api_id`, `opp_goalie_api_id`, `session`, `home_on_api_id`, `away_on_api_id`

**Non-feature columns** excluded from feature matrix but kept for labelling:
`goal`, `season`

### 4.2 Context xG Features — Tier 2 (shot sequence)

Added to the base_xg scored parquets by `prep_data()` and consumed by `context_xg/process_data.py`.
Require the base_xg model to have scored the full history first.

`CONTEXT_XG_FEATURE_COLUMNS` in `config.py` (20 features):

| Feature | Description | Notes |
|---|---|---|
| `logit_base_xg` | `logit(base_xg)` — T1 geometry prior | Used as BOTH `base_margin` AND a learnable feature; paired with each binary flag/state in its own constraint group (Groups 0–7) and also in the continuous block (Group 8) |
| `is_rebound` | 1 if shot follows a rebound | Binary; constraint Group 0 with `logit_base_xg` |
| `is_scramble` | 1 if shot is part of a scramble | Binary; constraint Group 1 |
| `rush_attempt` | 1 if shot came off a rush | Binary; constraint Group 2 |
| `prior_face` | 1 if prior event was a faceoff | Binary; constraint Group 3 |
| `is_home` | 1 if shooting team is home | Game-state modifier; constraint Group 4. Moved from base_xg in Issue 10. |
| `position` | Shooter position (F/D/G) | Game-state modifier; constraint Group 5; `pd.Categorical` |
| `strength_state` | e.g. "5v5", "5v4" | Game-state modifier; constraint Group 6; `pd.Categorical` |
| `score_diff` | Score differential, clipped to ±4 | Game-state modifier; constraint Group 7. Moved from base_xg in Issue 10 to fix EA OOF gap. |
| `period` | 1–4 (OT = 4) | Temporal; in continuous block (Group 8) |
| `period_seconds` | Seconds into the period | Temporal; in continuous block (Group 8) |
| `play_speed` | `distance_from_last / seconds_since_last` | Continuous sequence; Group 8; null when ≤ 0 seconds |
| `seconds_since_last` | Seconds since prior event | Same game + period only; Group 8 |
| `distance_from_last` | Euclidean dist from prior event | Group 8 |
| `prior_event_angle` | Angle to net of the prior event | Nullable float; Group 8 |
| `prior_event_distance` | Distance from net of the prior event | Nullable float; Group 8 |
| `seconds_since_stoppage` | Seconds since last faceoff | Forward-filled over `game_id`; Group 8 |
| `prior_event_same` | Type of prior event by the *same* team | String categorical: SHOT/MISS/BLOCK/GIVE/TAKE/HIT; `pd.Categorical` via `_apply_fixed_categoricals()`; Group 8 |
| `prior_event_opp` | Type of prior event by the *opposing* team | Same categories; mutually exclusive with `prior_event_same`; Group 8 |

**9 interaction constraint groups:**
- Groups 0–3: `[logit_base_xg, <flag>]` (binary event flags)
- Groups 4–7: `[logit_base_xg, <state>]` (game-state modifiers)
- Group 8: continuous sequence + temporal features (including `logit_base_xg`)

`context_xg/process_data.py` drops these columns from pred_goal parquets after context_xg scoring.

### 4.3 pred_goal Features — Tier 3 (stateful talent signals)

Assembled by `pred_goal/process_data.py`. All require the frozen context_xg model to have scored
the full history first.

#### Environment prior (base_margin, not a feature)

| Feature | Description |
|---|---|
| `context_xg` | context_xg output probability; renamed to `base_xg` before writing train parquets; passed as `base_margin` (log-odds), not a feature column |

pred_goal does **not** receive any `BASE_XG_FEATURE_COLUMNS` or `CONTEXT_XG_FEATURE_COLUMNS` as
direct features. Both are stripped from pred_goal parquets in `pred_goal/process_data.py`.
The pred_goal feature matrix contains only the talent columns below.

#### Shooter GxG — Rolling Goals Above context_xg

Metric: `goal − context_xg` per shot. Computed via `pred_goal/compute_rolling_stats.py`.

| Feature | Window | Notes |
|---|---|---|
| `shooter_gax_career` | All prior shots | Null if < 20 shots |
| `shooter_gax_per_shot_career` | All prior shots | Bayesian shrinkage: `GaX / (shots + 100)` |
| `shooter_gax_season` | Current season + session | Null if < 20 shots |
| `shooter_gax_per_shot_season` | Current season + session | Same shrinkage |
| `shooter_gax_10g` | 10 most recent completed games | |
| `shooter_gax_per_shot_10g` | Same | |
| `shooter_gax_ewma` | EWMA, half-life = 50 shots | Captures hot/cold streaks |

All apply `.shift(1)` leakage guard over `player_1_api_id`.

#### Goalie GSAx — Rolling Saves Above context_xg

Metric: `context_xg − goal` per shot (inverted). Empty-net shots (`opp_goalie_api_id = null`) receive null for all goalie columns.

| Feature | Window |
|---|---|
| `goalie_gsax_career` | Career |
| `goalie_gsax_per_shot_career` | Career (Bayesian shrinkage) |
| `goalie_gsax_season` | Season + session |
| `goalie_gsax_per_shot_season` | Season (shrinkage) |
| `goalie_gsax_10g` | 10 most recent games |
| `goalie_gsax_per_shot_10g` | |
| `goalie_gsax_ewma` | EWMA, half-life = 50 shots |

All apply `.shift(1)` leakage guard over `opp_goalie_api_id`.

#### Multi-RAPM Features

Three separate ridge regressions per season/session/situation: `context_xg`, `corsi`, `goals`. Each yields offensive (`off_coeff`) and defensive (`def_coeff`) estimates. RAPM from season S is joined to season S+1 shots.

**Shooter RAPM (4 columns) — prior season + career, xg dims only (Issue 15 fix):**

| Feature | Description |
|---|---|
| `shooter_rapm_xg_off` | Shooter's context_xg offensive RAPM (prior season S−1) |
| `shooter_rapm_xg_def` | Shooter's context_xg defensive RAPM (prior season S−1) |
| `shooter_rapm_career_xg_off` | Shooter's career context_xg offensive RAPM (season=0 aggregate) |
| `shooter_rapm_career_xg_def` | Shooter's career context_xg defensive RAPM |

**Teammates RAPM (4 columns) — mean of on-ice teammates, shooter + goalie excluded:**

| Feature | Description |
|---|---|
| `teammates_rapm_xg_off` | Teammates avg context_xg offensive RAPM (prior season) |
| `teammates_rapm_xg_def` | Teammates avg context_xg defensive RAPM (prior season) |
| `teammates_rapm_career_xg_off` | Teammates avg career context_xg offensive RAPM |
| `teammates_rapm_career_xg_def` | Teammates avg career context_xg defensive RAPM |

**Opponent RAPM (4 columns) — mean of on-ice opponents:**

| Feature | Description |
|---|---|
| `opp_rapm_xg_off` | Opponents avg context_xg offensive RAPM (prior season) |
| `opp_rapm_xg_def` | Opponents avg context_xg defensive RAPM (prior season) |
| `opp_rapm_career_xg_off` | Opponents avg career context_xg offensive RAPM |
| `opp_rapm_career_xg_def` | Opponents avg career context_xg defensive RAPM |

**Shooter vs. Teammates differential (2 columns — xg offensive only):**

| Feature | Formula | Intuition |
|---|---|---|
| `shooter_vs_teammates_rapm_xg_off` | `shooter_xg_off − teammates_xg_off` (prior season) | Positive = shooter elevates the line |
| `shooter_vs_teammates_rapm_career_xg_off` | `shooter_career_xg_off − teammates_career_xg_off` | Career version of above |

No defensive differential — defensive RAPM is independent of the shooter's offensive role.

**Why corsi and goals dims dropped (Issue 15):** Corsi is a shot-volume signal already captured by context_xg's geometry and game-state features. Goals RAPM is extremely noisy (rare binary events). Restricting to xg dims reduces total RAPM features from 36 → 14 and gives the model a cleaner, more stable talent signal.

**RAPM situation matching:**

| Model variant | RAPM situation |
|---|---|
| `even_strength`, `empty_for`, `empty_against` | EV |
| `powerplay` | PP |
| `shorthanded` | SH |

Events with < 2 teammates having prior-season RAPM receive null — XGBoost handles this natively via missing value propagation.

---

## 5. Model Architecture

### 5.1 Base xG — XGBoost gbtree Classifier

- **Algorithm:** XGBoost `binary:logistic`, `booster=gbtree`
- **Feature set:** 8 `BASE_XG_FEATURE_COLUMNS` — pure location geometry + shot type. No game state, no player identity, no shot sequence features. (Game-state features were moved to context_xg in Issue 10.)
- **Categorical handling:** `enable_categorical=True`; fixed category lists via `pd.Categorical` (`shot_type`, `position`, `strength_state`)
- **Monotone constraints:** `event_distance: −1`, `event_angle: −1` (each filtered to columns present in `X_train`)
- **Calibration:** OOF isotonic regression (5-fold `TimeSeriesSplit`). Calibrator saved as `{strength}/calibrator.joblib`.
- **Output format:** Frozen booster saved as `{strength}/model.ubj` (binary UBJSON); provenance in `{strength}/meta.json`
- **Training data:** 2010-11 through 2023-24; hold-out is 2024-25

### 5.2 context_xg — XGBoost gbtree Classifier (depth=2, flag isolation)

- **Algorithm:** XGBoost `binary:logistic`, `booster=gbtree`, `max_depth=2` (fixed)
- **Why gbtree with depth=2?** Each binary flag (`is_rebound`, `is_scramble`, `rush_attempt`,
  `prior_face`) is placed in its own interaction constraint group paired with `logit_base_xg`.
  A depth-2 tree path is: `logit_base_xg > threshold → is_flag == 1 → leaf`. This learns a
  quality-conditional flag effect without allowing two flags to combine on one path. Prevents
  both the gblinear bimodal cliff (flags additive regardless of shot quality) and the GOAL
  fingerprinting risk (multi-flag Rush+Rebound path requires depth ≥ 3 + cross-group features).
- **Why not gblinear?** gblinear's additive structure adds fixed log-odds per flag regardless of
  shot quality. Multiple co-firing flags stack their shifts, creating a bimodal prediction cliff
  (~0.62–0.65) for high-flag shots. No calibrator can resolve this. Quality interaction features
  (`flag × base_xg`) are collinear with `logit_base_xg` in a linear model, causing weight collapse.
  See Issue 9 in `DECISIONS.md`.
- **Interaction constraint groups** (`CONTEXT_XG_INTERACTION_GROUPS` in `config.py`):
  - Groups 0–3: `[logit_base_xg, <flag>]` — is_rebound, is_scramble, rush_attempt, prior_face
  - Groups 4–7: `[logit_base_xg, <state>]` — is_home, position, strength_state, score_diff
  - Group 8: continuous sequence + temporal block (logit_base_xg included so depth-2 trees can ask "how does rush speed modify the spatial prior?")
- **No `colsample_*` parameters:** Column sampling selects features before constraints are applied;
  if a group's features aren't sampled for a given tree, that tree degenerates silently.
  All `colsample_*` fixed at 1.0 (omitted from param space).
- **Feature set:** 20 `CONTEXT_XG_FEATURE_COLUMNS` — `logit_base_xg` + 4 binary event flags + 4
  game-state modifiers (`is_home`, `position`, `strength_state`, `score_diff`) + continuous sequence
  + temporal features + `prior_event_same`/`prior_event_opp` (string categoricals).
- **Categorical handling:** `enable_categorical=True`; `prior_event_same`/`prior_event_opp`
  and `position`/`strength_state` converted to `pd.Categorical` by `_apply_fixed_categoricals()`.
- **`logit_base_xg` dual role — base_margin AND feature:** `logit_base_xg` is passed as
  `base_margin` (fixing `g = sigmoid(logit_base_xg + F(x)) − y`) AND included in the feature
  matrix. Two distinct roles: base_margin provides the gradient anchor (main effect, fixed coeff
  1.0); the feature enables quality-conditional flag/state adjustments (the tree's second split
  in each constraint group). Without base_margin, trees must output the full probability range from
  scratch, causing flag-group leaf weights to cluster all flag shots at a mid-range probability
  regardless of base quality (bimodal cliff). Without the feature, each group degenerates to a
  single binary split, equivalent to gblinear's additive coefficient. See Issues 8 and 11 in
  `DECISIONS.md`.
- **Critical finalization parameter:** `max_delta_step=1` is required to prevent bimodal collapse.
  Trials with `max_delta_step ≥ 2` rank well on CV PR-AUC (flat landscape) but produce bimodal
  prediction distributions (large leaf weights, calibrated log loss >> 2× null). The calibrated
  top-N trial screening in `finalize.py` (`--top-n 150`) reliably selects mds=1 trials. Using
  `--top-n 15` is unsafe — mds=1 candidates sit at rank 16+ and are excluded from screening.
- **Calibration:** Pooled OOF + hold-out Platt (LogisticRegression, C=1.0). OOF fold models are
  trained to `model.best_iteration` (no early stopping, same tree count as final model) to align
  the fold-model probability scale. OOF probs + hold-out probs are pooled before fitting the
  calibrator, giving 15-season temporal coverage. Platt's sigmoid ceiling (max < 1.0) is acceptable
  at context_xg's ~89% top-decile actual rate (Issue 5's ~99% ceiling concern applies to base_xg EA).
- **Output format:** Frozen booster as `{strength}/model.ubj`; calibrator as `{strength}/calibrator.joblib`;
  OOF calibrated predictions as `{strength}/oof.parquet`; provenance in `{strength}/meta.json`

### 5.3 pred_goal — Cascaded XGBoost gbtree + Isotonic Calibration

- **Algorithm:** XGBoost `binary:logistic` with `base_margin` = `logit(context_xg)`
- **Why base_margin?** Forces pred_goal to learn only the talent residual on top of the
  geometry + sequence prior. The model never needs to re-learn location or shot context.
- **Feature matrix:** All `BASE_XG_FEATURE_COLUMNS` and `CONTEXT_XG_FEATURE_COLUMNS` are excluded
  from pred_goal parquets entirely — dropped in `pred_goal/process_data.py` before writing.
  pred_goal sees only talent features: shooter GxG/GSAx rolling windows (career / season / 10g / EWMA;
  `_1g` window removed in Issue 15) and RAPM (xg_off / xg_def for shooter, teammates, opponents × prior + career;
  corsi and goals dims dropped in Issue 15 — total RAPM features 36 → 14).
- **Logit transform:** `log(clip(p, 1e-7, 1−1e-7) / (1 − clip(p, 1e-7, 1−1e-7)))` — avoids ±inf
- **Categorical handling:** `enable_categorical=True`
- **Calibration:** OOF isotonic regression (5-fold `TimeSeriesSplit`). Calibrator saved as `{strength}/calibrator.joblib`.
- **SHAP:** `shap.TreeExplainer` on a 2,000-row sample from hold-out; summary plot logged to MLflow
- **Inference sequence:**
  ```python
  bm = logit(context_xg)
  raw_prob = base_model.predict_proba(X, base_margin=bm)[:, 1]
  pred_goal = calibrator.predict(raw_prob)
  ```

---

## 6. Training Infrastructure

### 6.1 Optuna — Hyperparameter Tuning

- **Storage:** PostgreSQL RDB (`postgresql+psycopg2://...`) — same `mlflow-db` instance used by MLflow
- **Study naming:** `{strength}-{version}-{base|informed}` e.g. `even_strength-v1-base`
- **Sampler:** `TPESampler(multivariate=True, seed=615)` — models hyperparameter interactions jointly; converges in 300–500 trials vs. 1,000 for NSGA-II
- **Objective (single):** PR-AUC (maximise only). Log_loss is logged to every MLflow child run for monitoring but not returned to Optuna — calibration is handled post-hoc by the finalize scripts, and searching `scale_pos_weight` alongside raw log_loss distorts the Pareto front.
- **Best trial selection:**
  - *base_xg / pred_goal:* Best PR-AUC among completed trials with `max_depth ≤ MAX_DEPTH_CAP`
  - *context_xg:* Calibrated top-N screening — top-N trials by CV PR-AUC are retrained with quick
    hold-out Platt calibration; trials with calibrated log loss > 2× null_ll are rejected (bimodal
    detection); the passing trial with the highest calibrated hold-out PR-AUC is selected. Use
    `--top-n 150` (not the 15 default) because the CV landscape is flat and mds=1 non-bimodal
    candidates sit at rank 16+. See `_screen_trials()` in `context_xg/finalize.py`.
- **Why PR-AUC not ROC-AUC?** ~6–8% positive rate (goals) makes ROC-AUC misleading (TN abundance inflates it); PR-AUC penalises false positives in the rare class directly
- **Pruning:** Manual median pruner — fold PR-AUC stored as user attr `fold_{n}_prauc`; trial pruned if below median of completed trials at that fold after ≥ 5 completed baseline trials
- **Pruned runs:** Logged to MLflow with `trial_outcome=pruned`, `performance=pruned`, status `KILLED`

**Optuna ↔ MLflow bidirectional linking:**

| Link | Direction | Mechanism |
|---|---|---|
| Optuna trial → MLflow run | Forward | `trial.set_user_attr("mlflow_run_id", run_id)` |
| MLflow run → Optuna trial | Reverse | `mlflow.set_tag("optuna_trial_num", trial.number)` |

### 6.2 MLflow — Experiment Tracking

- **Server:** `mlflow-app` container at `https://{MLFLOW_DOMAIN}` (homelab)
- **Artifact store:** MinIO S3-compatible at `https://{MINIO_DOMAIN}`
- **Run structure:** Parent run (study summary) → nested child runs (one per trial)
- **Per-trial logging:** All XGBoost params + defaults, fold metrics immediately after each fold (so pruned runs retain partial metrics), test metrics, boosting metrics as timestep series, feature importance JSON, 7 visualization PNGs
- **Performance tag:** Assigned per trial based on final test metrics:

| Tag | Condition |
|---|---|
| `none` | `log_loss ≥ 0.33` OR `PR-AUC < 0.12` |
| `low` | All others |
| `medium` | `PR-AUC ≥ 0.22` AND `log_loss ≤ 0.27` |
| `high` | `PR-AUC ≥ 0.30` AND `log_loss ≤ 0.24` |
| `very high` | `PR-AUC ≥ 0.38` AND `log_loss ≤ 0.21` |

**Logged visualizations per trial:**
- Classification report (goal row on top, no goal on bottom; comma-formatted support)
- ROC-AUC (4 curves: goal, no goal, macro avg, micro avg; AUC values in legend)
- Class prediction error (comma-formatted bar labels)
- Precision-recall curve (binary PR curve + average precision horizontal line)
- Feature importance (top 10, absolute)
- Feature importance (top 10, relative)
- Confusion matrix

### 6.3 Environment Setup

Required env vars (see `.env.example`):

```bash
# MLflow tracking server
MLFLOW_TRACKING_URI=https://mlflow.yourdomain.com
MLFLOW_TRACKING_USERNAME=...
MLFLOW_TRACKING_PASSWORD=...

# MLflow PostgreSQL backend (for hard delete via nuke_experiment.py)
MLFLOW_BACKEND_STORE_URI=postgresql+psycopg2://user:pass@host:5432/dbname

# MinIO (MLflow artifact store)
MLFLOW_S3_ENDPOINT_URL=https://minio.yourdomain.com
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...

# Optuna PostgreSQL storage
DB_HOST=...
DB_PORT=5432
DB_USER=...
DB_PASSWORD=...
DB_NAME=...
```

---

## 7. Script Reference

| Script | Inputs | Outputs | Status |
|---|---|---|---|
| `raw_data/scrape_raw_data.py` | NHL API | `raw_data/pbp/pbp_{YYYY}.parquet` | ✅ Done |
| `chickenstats_xg/v1/utils/shot_features.py` | — | `prep_data()` feature engineering (stateless) | ✅ Done |
| `chickenstats_xg/v1/utils/artifacts.py` | — | `save/load_model_artifacts`, `save_model_metadata`, `params_from_run_name` | ✅ Done |
| `chickenstats_xg/v1/utils/finalize_utils.py` | — | `compute_oof_predictions`, `screen_trials`, `select_top_trials`, `calculate_ece` | ✅ Done |
| `chickenstats_xg/v1/utils/log_model.py` | Local model artifacts | Uploads frozen models to MLflow Model Registry | ✅ Done |
| `chickenstats_xg/v1/experiments.py` | `data/{model}/train/` parquets | MLflow runs, Optuna trials (base_xg, context_xg, pred_goal) | ✅ Done |
| `chickenstats_xg/v1/base_xg/process_data.py` | `raw_data/pbp/` | `chickenstats_xg/v1/data/base_xg/train/` + `hold_out/` | ✅ Done |
| `chickenstats_xg/v1/base_xg/finalize.py` | train + hold_out parquets, Optuna study | `models/base_xg/{strength}/model.ubj` + `calibrator.joblib` + `meta.json` | ✅ Done |
| `chickenstats_xg/v1/base_xg/score.py` | `raw_data/pbp/`, frozen model | `data/base_xg/scored/` (with `event_idx`); enriches `data/rapm/pbp/` with `base_xg` | ✅ Done |
| `chickenstats_xg/v1/context_xg/process_data.py` | `data/base_xg/scored/` | `data/context_xg/train/` + `hold_out/` (computes `logit_base_xg`) | ✅ Done |
| `chickenstats_xg/v1/context_xg/finalize.py` | train + hold_out parquets, Optuna study | `models/context_xg/{strength}/model.ubj` + `calibrator.joblib` + `oof.parquet` + `meta.json` (gbtree depth-2, pooled OOF + hold-out Platt) | ✅ Done |
| `chickenstats_xg/v1/context_xg/score.py` | `data/base_xg/scored/`, frozen model | `data/context_xg/scored/`; enriches `data/rapm/pbp/` with `context_xg` | ✅ Done |
| `chickenstats_xg/v1/context_xg/diagnose.py` | `data/context_xg/scored/`, `models/context_xg/` | Calibration, lift, OOF gap, weight concentration diagnostics | ✅ Done |
| `chickenstats_xg/v1/rapm/prep_pbp.py` | `raw_data/pbp/pbp_*.parquet`, `data/base_xg/scored/` | `data/rapm/pbp/pbp_{YYYY}.parquet` (initial enrichment with `base_xg`) | ✅ Done |
| `chickenstats_xg/v1/rapm/process_stints.py` | `data/rapm/pbp/pbp_{YYYY}.parquet` | `data/rapm/stints/` (h_xgf/a_xgf from `context_xg`) | ✅ Done |
| `chickenstats_xg/v1/rapm/regressions.py` | stints parquets | `data/rapm/rapm_by_season.parquet` (`off_coeff_context_xg`, etc.) | ✅ Done |
| `chickenstats_xg/v1/pred_goal/compute_rolling_stats.py` | scored parquets (as DataFrame) | GxG/GSAx columns appended (in-memory) | ✅ Done |
| `chickenstats_xg/v1/pred_goal/process_data.py` | `data/context_xg/scored/`, RAPM table | `data/pred_goal/train/` + `hold_out/` (both feature tiers stripped before write) | ✅ Done |
| `chickenstats_xg/v1/pred_goal/finalize.py` | train + hold_out parquets, Optuna study | `models/pred_goal/{strength}/model.ubj` + `calibrator.joblib` + `meta.json` | ✅ Done |
| `chickenstats_xg/v1/base_xg/run_pipeline.py` | — | Orchestrates base_xg finalize → score → context_xg process_data | ✅ Done |
| `chickenstats_xg/v1/context_xg/run_pipeline.py` | — | Orchestrates context_xg finalize → score → RAPM → pred_goal process_data | ✅ Done |
| `chickenstats_xg/nuke_experiment.py` | Study name, env | Deletes S3 + MLflow + Optuna records | ✅ Done |

---

## 8. Training Execution Guide

Training must happen **in this exact order** because each step depends on the previous.

### Step 1 — Scrape raw data

```bash
cd /path/to/chickenstats-xg
uv run python raw_data/scrape_raw_data.py
```

Skips seasons already present in `raw_data/pbp/`. Covers 2010–2024.

### Step 2 — Build base_xg training data

```bash
uv run python chickenstats_xg/v1/base_xg/process_data.py
```

Reads all `raw_data/pbp/pbp_{YYYY}.parquet` files, applies `prep_data()` for each year × strength combination, and writes:
- `chickenstats_xg/v1/data/base_xg/train/{strength}.parquet` (seasons except hold-out)
- `chickenstats_xg/v1/data/base_xg/hold_out/{strength}.parquet` (hold-out season = 2024-25)

**Note:** If adding new features to `xg_utils.py`, re-run this step before tuning.

### Step 3 — Tune base_xg model

Run all 5 strengths in parallel terminal tabs. Old studies have been nuked — clean slate.

```bash
uv run python chickenstats_xg/v1/experiments.py --model base_xg --strength even_strength --version 1.0.0 --trials 100
uv run python chickenstats_xg/v1/experiments.py --model base_xg --strength powerplay --version 1.0.0 --trials 100
uv run python chickenstats_xg/v1/experiments.py --model base_xg --strength shorthanded --version 1.0.0 --trials 100
uv run python chickenstats_xg/v1/experiments.py --model base_xg --strength empty_for --version 1.0.0 --trials 100
uv run python chickenstats_xg/v1/experiments.py --model base_xg --strength empty_against --version 1.0.0 --trials 100
```

Trials are logged to MLflow and stored in Optuna PostgreSQL. The study is created on first run and reused on subsequent runs (safe to resume).

**To delete a study and start over:**
```bash
uv run nuke-experiment --study even_strength-1.0.0-base --dry-run
uv run nuke-experiment --study even_strength-1.0.0-base --confirm
```

### Step 4 — Finalize base_xg + score all PBP + build context_xg data

```bash
uv run python chickenstats_xg/v1/base_xg/run_pipeline.py --version 1.0.0 --no-log
```

This runs:
1. `base_xg/finalize.py --all` — selects best trial by PR-AUC, retrains, OOF calibrates, writes `models/base_xg/{strength}/model.ubj` + `calibrator.joblib` + `meta.json`
2. `base_xg/score.py --all` — scores full PBP history, writes `data/base_xg/scored/` and enriches `data/rapm/pbp/` with `base_xg`
3. `context_xg/process_data.py` — splits base_xg scored parquets into context_xg train/hold_out (computes `logit_base_xg`)

### Step 5 — RAPM regressions (bootstrap — base_xg xGF)

```bash
uv run python chickenstats_xg/v1/rapm/regressions.py
```

Bootstrap run uses `base_xg` as xGF proxy (context_xg not yet available). Output:
`data/rapm/rapm_by_season.parquet` with `off_coeff_context_xg` (renamed from base_xg after step 8).

### Step 6 — Tune context_xg model (5 parallel tabs)

```bash
uv run python chickenstats_xg/v1/experiments.py --model context_xg --strength even_strength --version 1.0.0 --trials 100
uv run python chickenstats_xg/v1/experiments.py --model context_xg --strength powerplay --version 1.0.0 --trials 100
uv run python chickenstats_xg/v1/experiments.py --model context_xg --strength shorthanded --version 1.0.0 --trials 100
uv run python chickenstats_xg/v1/experiments.py --model context_xg --strength empty_for --version 1.0.0 --trials 100
uv run python chickenstats_xg/v1/experiments.py --model context_xg --strength empty_against --version 1.0.0 --trials 100
```

gbtree depth-2 tuning: `max_delta_step`, `min_child_weight`, `gamma`, `lambda`, `alpha`,
`subsample`, `learning_rate`. No `colsample_*` (column sampling disabled — subsets break
interaction constraint groups by potentially omitting one feature from a constrained pair).
`max_depth=2` is fixed, not tuned. `logit_base_xg` is BOTH a feature in the constraint groups
AND passed as `base_margin` to each `model.fit()` call.

### Step 7 — Finalize context_xg + score + rebuild RAPM + build pred_goal data

```bash
uv run python chickenstats_xg/v1/context_xg/run_pipeline.py --version 1.0.0 --no-log
```

This runs:
1. `context_xg/finalize.py --all --top-n 150` — retrains gbtree depth-2 using calibrated top-N trial screening (use `--top-n 150`, not the default 15 — the CV landscape is flat and mds=1 non-bimodal trials sit at rank 16+), pooled OOF + hold-out Platt calibrates, freezes model artifacts
2. `context_xg/score.py --all` — scores base_xg parquets; enriches `data/rapm/pbp/` with `context_xg`
3. `rapm/process_stints.py` — rebuilds stints using `context_xg` for h_xgf/a_xgf
4. `rapm/regressions.py` — RAPM with `context_xg` as xGF target; output: `off_coeff_context_xg`, etc.
5. `pred_goal/process_data.py` — assembles talent features; renames `context_xg` → `base_xg`; strips geometry + sequence columns

### Step 8 — Tune pred_goal model (5 parallel tabs)

```bash
uv run python chickenstats_xg/v1/experiments.py --model pred_goal --strength even_strength --version 1.0.0 --trials 100
uv run python chickenstats_xg/v1/experiments.py --model pred_goal --strength powerplay --version 1.0.0 --trials 100
uv run python chickenstats_xg/v1/experiments.py --model pred_goal --strength shorthanded --version 1.0.0 --trials 100
uv run python chickenstats_xg/v1/experiments.py --model pred_goal --strength empty_for --version 1.0.0 --trials 100
uv run python chickenstats_xg/v1/experiments.py --model pred_goal --strength empty_against --version 1.0.0 --trials 100
```

`context_xg` (renamed to `base_xg`) is converted to log-odds and passed as `base_margin`. The
pred_goal feature matrix contains only talent features (GxG/GSAx rolling windows, RAPM).

### Step 9 — Finalize pred_goal

```bash
uv run python chickenstats_xg/v1/pred_goal/finalize.py --all --version 1.0.0 --no-log
```

Writes:
- `chickenstats_xg/v1/models/pred_goal/{strength}/model.ubj`
- `chickenstats_xg/v1/models/pred_goal/{strength}/calibrator.joblib`
- `chickenstats_xg/v1/models/pred_goal/{strength}/meta.json`

Logs: hold-out metrics, SHAP summary (max 20 features, 2,000-row sample), 7 visualizations, calibrator sklearn model to MLflow.

### Step 10 — Validate

```bash
uv run python chickenstats_xg/v1/pred_goal/diagnose.py
uv run python chickenstats_xg/v1/rapm/diagnose.py
```

---

## 9. Hyperparameter Search Space

### 9.1 base_xg + pred_goal — XGBoost gbtree

`_params_pred_goal = _params_base_xg` — pred_goal uses the identical search space.

| Parameter | Type | Range | Notes |
|---|---|---|---|
| `max_depth` | int | 3–6 | Capped at 6 to prevent GOAL fingerprinting |
| `min_child_weight` | int (log) | 20–200 | Log scale |
| `max_delta_step` | int | 1–10 | Floor of 1 ensures the step cap is always active; tunable here (no base_margin, monotone constraints prevent bimodal cliff) |
| `scale_pos_weight` | float | 1.0–10.0 | Searched up to min(data_spw, 10.0); `empty_against` fixed at 1.0 (balanced) |
| `learning_rate` | float (log) | 1e-3–0.30 | No ceiling needed — mds tunable + no base_margin means high lr doesn't cause bimodal |
| `gamma` | float | 0.0–5.0 | Upper end acts as hard gate against noisy splits |
| `lambda` | float (log) | 0.1–10.0 | ES 1.0.0 selected 9.354 (near ceiling); raise to 100.0 if future ES runs need stronger regularization |
| `alpha` | float (log) | 1e-8–1.0 | L1 regularization |
| `subsample` | float (step 0.05) | 0.4–1.0 | Row sampling |
| `colsample_bytree` | float (step 0.05) | 0.6–1.0 | |
| `colsample_bylevel` | float (step 0.05) | 0.6–1.0 | Per-depth feature subsampling |
| `colsample_bynode` | float (step 0.05) | 0.6–1.0 | Per-split feature subsampling |

Fixed:
- `objective: binary:logistic`, `booster: gbtree`
- `n_estimators: 500` (with `early_stopping_rounds: 50`)
- `eval_metric: ["aucpr", "logloss"]` — early stopping on logloss (last metric)
- `enable_categorical: True`
- `random_state: 615`
- `monotone_constraints: {event_distance: -1, event_angle: -1}` (base_xg only; filtered to columns present)

### 9.2 context_xg — XGBoost gbtree (depth=2, flag+state isolation)

| Parameter | Type | Range | Notes |
|---|---|---|---|
| `max_delta_step` | **fixed: 1** | — | **Not tunable.** mds ≥ 2 causes bimodal cliff: large per-tree leaf updates accumulate into a high-probability cluster for flag shots. Fixed in the search space (not trial-suggested). |
| `min_child_weight` | int (log) | 100–500 | Floor raised from 20/50 — lower values caused low-decile overestimation for SH/EF (sparse flag groups) |
| `gamma` | float | 1.0–10.0 | Floor raised from 0.0 — prevents unconstrained splits; 1.0.0 trials span 1.4–5.7 |
| `lambda` | float (log) | 10.0–200.0 | Floor 10.0 critical — lambda < 10 produces bimodal even with mds=1; 1.0.0 EA selected 95.62 (near old 100.0 ceiling) |
| `alpha` | float (log) | 0.1–10.0 | Floor raised from 1e-6 |
| `subsample` | float (step 0.05) | 0.5–1.0 | Row sampling; higher floor (flag groups sparse) |
| `learning_rate` | float (log) | 0.01–**0.20** | **Ceiling 0.20 (not 0.30).** All passing 1.0.0 models had lr ≤ 0.21. Above 0.20, TPE converges to high-lr configs that look calibrated on the anomalous 2023-24 CV test year but accumulate enough log-odds over 50–130 trees to produce a bimodal distribution on the real 2024-25 hold-out. |
| `scale_pos_weight` | float | 1.0–3.0 | Cap 3.0 (vs 10.0 for base_xg); base_margin already anchors the base rate, so spw near 1.0 is expected |

Fixed:
- `objective: binary:logistic`, `booster: gbtree`
- `max_depth: 2` — fixed, not tuned. Depth-3 allows cross-group paths (defeating isolation).
- `max_delta_step: 1` — fixed, not tuned. See above.
- `interaction_constraints` — from `_build_context_interaction_constraints(list(X_train.columns))` (9 groups)
- `base_margin` — `logit_base_xg` passed as base_margin to model.fit() in addition to its role as a feature column
- `n_estimators: 500` (with `early_stopping_rounds: 50`)
- `eval_metric: ["aucpr", "logloss"]` — early stopping on logloss (last metric); aucpr-only stopping caused bimodal collapse (model stopped when ranking plateaued before calibration settled)
- `enable_categorical: True`
- `random_state: 615`
- No `colsample_*` — column sampling selects features before constraints are applied; omitting a group's features from a tree silently degenerates that constraint group

CV: **3-fold `TimeSeriesSplit`** inside Optuna (not `shuffle`) for all tiers. A full model (no early stopping) is also trained on `X_train` and evaluated on `X_test` for final test metrics.

---

## 10. Data Directory Structure

```
chickenstats_xg/v1/
│
├── models/
│   ├── base_xg/
│   │   └── even_strength/
│   │       ├── model.ubj
│   │       ├── calibrator.joblib
│   │       └── meta.json
│   │       ... (5 subdirs, 3 files each)
│   ├── context_xg/
│   │   └── even_strength/
│   │       ├── model.ubj
│   │       ├── calibrator.joblib
│   │       ├── oof.parquet
│   │       └── meta.json
│   │       ... (5 subdirs, 4 files each)
│   └── pred_goal/
│       └── even_strength/
│           ├── model.ubj
│           ├── calibrator.joblib
│           └── meta.json
│           ... (5 subdirs, 3 files each)
│
└── data/
    ├── base_xg/
    │   ├── train/
    │   │   ├── even_strength.parquet      Seasons 2010-11 through 2023-24
    │   │   └── ... (5 files total)
    │   ├── hold_out/
    │   │   └── {same 5 files}             Season 2024-25 only
    │   └── scored/
    │       └── {same 5 files}             Full history + base_xg + event_idx; input to context_xg
    │
    ├── context_xg/
    │   ├── train/
    │   │   └── {same 5 files}             base_xg + sequence features; seasons pre-hold-out
    │   ├── hold_out/
    │   │   └── {same 5 files}             Season 2024-25 only
    │   └── scored/
    │       └── {same 5 files}             Full history + context_xg; input to RAPM + pred_goal
    │
    ├── pred_goal/
    │   ├── train/
    │   │   └── {same 5 files}             context_xg (as base_xg) + all talent features; geometry + sequence stripped
    │   └── hold_out/
    │       └── {same 5 files}
    │
    └── rapm/
        ├── pbp/
        │   ├── pbp_2010.parquet           Raw PBP enriched with base_xg + context_xg (one per year)
        │   └── ...
        ├── stints/
        │   ├── stints_2010_r.parquet      One per season × session
        │   ├── stints_2010_p.parquet
        │   └── ...
        └── rapm_by_season.parquet         All seasons; off_coeff_context_xg/def_coeff_context_xg/etc.
```

---

## 11. RAPM Pipeline Details

### Enriched PBP (`rapm/prep_pbp.py`)

Bridges scored base_xg parquets to the raw per-year PBP. For each year:
1. Loads all 5 strength-state scored parquets; selects `(game_id, event_idx, base_xg)`
2. Groups by `(game_id, event_idx)` summing base_xg (handles any duplicate events across strength files)
3. Left-joins onto raw `pbp_{YYYY}.parquet` by `(game_id, event_idx)`
4. Fills null `base_xg` with 0.0 (non-Fenwick events have no shot probability)
5. Writes to `data/rapm/pbp/pbp_{YYYY}.parquet`

`context_xg/score.py --all` then enriches these files additively, appending the `context_xg`
column by joining on `(game_id, event_idx)` from the context_xg scored parquets.

### Stints (`rapm/process_stints.py`)

Reads from `data/rapm/pbp/pbp_{YYYY}.parquet` (already enriched). A stint is a contiguous period of unchanged on-ice personnel within a period. The stints DataFrame encodes per-stint:

- `toi` — time on ice in seconds
- `h_xgf` / `a_xgf` — xG for from `context_xg` column (home and away perspective)
- `h_cf` / `a_cf` — Corsi for (shots + misses + blocks + goals)
- `h_gf` / `a_gf` — goals for
- `h_skaters` / `a_skaters` — list of on-ice skater API IDs (goalies excluded)
- `h_goalies` / `a_goalies` — list of on-ice goalie API IDs
- `h_cnt` / `a_cnt` — skater counts (for EV/PP/SH filtering)
- `strength` — strength state
- `ozs`, `nzs`, `dzs` — zone start flags
- `h_b2b` / `a_b2b` — back-to-back game flag
- `h_s3` / `h_s7` — score state (3-bucket for xG/goals, 7-bucket for Corsi)

Stints with TOI = 0 or fewer than 3 skaters per side are excluded.

### Regressions (`rapm/regressions.py`)

**Sparse matrix structure:**
- Rows: doubled stints (home perspective + away perspective)
- Columns: offensive skaters (cols 0..N-1), defensive skaters (cols N..2N-1), optionally defensive goalies (goals metric only), then control variables (strength dummies, OZS/NZS/DZS, home advantage, B2B, score state)
- Values: binary (1 = player on ice in that role in that stint)
- Y: per-stint metric per 60 minutes (rate statistic, TOI-weighted)

**Three separate regression targets:**
- `context_xg` — xG per 60 (uses 3-bucket score state, no goalie term)
- `corsi` — Corsi events per 60 (uses 7-bucket score state, no goalie term)
- `goals` — goals per 60 (uses 3-bucket score state, includes goalie indicator column)

**Alpha selection:** 5-fold `GridSearchCV` over `np.logspace(3, 5.5, 15)` (ThreadingBackend, all cores).

**Output after `process_regression_results()`:** Pivoted to one row per `(season, session, player, team, pos, situation)` with columns `off_coeff_context_xg`, `def_coeff_context_xg`, `total_rapm_context_xg`, `off_coeff_corsi`, etc., plus z-scores for each.

**TOI minimums:** `ev_r=10` min (regular season EV), `ev_p=5` min (playoff EV), `other_r=5`, `other_p=1`.

---

## 12. Rolling Stats Details

**Source:** `pred_goal/compute_rolling_stats.py` — called inside `pred_goal/process_data.py` on the sorted combined scored parquets. Input must be sorted by `['season', 'game_id', 'period', 'period_seconds']`.

**Window types:**

| Window | Implementation | Leakage guard |
|---|---|---|
| Career | `cum_sum().shift(1).over(player_id)` | `.shift(1)` ensures current shot not included |
| Season | `cum_sum().shift(1).over([player_id, 'season', 'session'])` | Same |
| 10g | `rolling_sum(10).shift(1).over(player_id)` on game-level aggregate | `.shift(1)` on game-level |
| EWMA | `ewm_mean(half_life=50, ignore_nulls=True).shift(1).over(player_id)` | |

**Bayesian shrinkage on per-shot rates:**
`gax / (shots + 100)` — a player with 5 career shots sees their rate pulled toward 0 by 100 phantom average shots. Prevents small-sample extremes from misleading the model.

**Min shots floor (cumulative only):** Career/season cumulative `gax` values are `null` if < 20 shots. Per-shot rates use shrinkage instead of a floor.

**Goalie handling:** `goalie_rows = scored.filter(pl.col(GOALIE_ID_COL).is_not_null())` — empty-net events (null goalie ID) are separated, stats computed only on non-empty-net events, then joined back via `_row_idx`. Empty-net rows receive null for all goalie columns.

---

## 13. Leakage Prevention Rules

1. **Chronological CV only.** All cross-validation uses `TimeSeriesSplit`. Never `shuffle=True`. The outer hold-out (2024-25) is separated before any model fit.
2. **Rolling stat shifts.** All `gax`/`gsax` windows apply `.shift(1)` over the player ID so the current shot is never in its own rolling window.
3. **RAPM lag.** RAPM from season S is joined to season S+1. A player's 2023-24 RAPM is never used to score their 2024-25 shots. Season lag computed as `season − 10001` (e.g. `20242025 → 20232024` — consecutive NHL seasons differ by 10001, not 10000).
4. **Deduplication.** When a player appears for multiple teams in a season's RAPM, the entry with the highest TOI is kept. Only regular-season (`session='R'`) RAPM is used for lagged joins.
5. **No player identity in base_xg.** `player_1_api_id` and `opp_goalie_api_id` are passthrough columns only — they never enter the base_xg feature matrix.

---

## 14. Live Inference Strategy

For real-time `pred_goal` in the homelab API (`POST /inference/pred_goal/live`):

```
1. Generate stateless geometry + game state features from live event (14 BASE_XG_FEATURE_COLUMNS)
2. Score with frozen base_xg → calibrated base_xg probability
3. Compute logit_base_xg from base_xg; generate all 20 CONTEXT_XG_FEATURE_COLUMNS (including game-state modifiers: is_home, position, strength_state, score_diff); convert `prior_event_same`/`prior_event_opp` and `position`/`strength_state` to `pd.Categorical` (gbtree `enable_categorical=True`); binary flags as-is
4. Score with frozen context_xg gbtree depth-2: pass logit_base_xg as BOTH a feature column AND as `base_margin`; apply calibrator → calibrated context_xg probability
5. Look up shooter rolling GxG from rolling_stats_skater table (latest snapshot)
6. Look up goalie rolling GSAx from rolling_stats_goalie table
7. Look up shooter RAPM from rapm_scores table (current-season, situation-matched, off_coeff_context_xg / def_coeff_context_xg)
8. Explode on-ice IDs → look up teammate and opponent RAPM → compute means + differential
9. Convert context_xg to log-odds → pass as base_margin; exclude all BASE_XG_FEATURE_COLUMNS and CONTEXT_XG_FEATURE_COLUMNS from the pred_goal feature vector
10. Score with calibrated pred_goal → pred_goal probability
```

Rolling stats and RAPM are updated nightly by the scraper; RAPM is refreshed once per season at season start.

---

## 15. Implementation Status

### Scripts

| Script | Status |
|---|---|
| `raw_data/scrape_raw_data.py` | ✅ Complete |
| `chickenstats_xg/v1/utils/shot_features.py` | ✅ Complete (prep_data(); extracted from xg_utils.py) |
| `chickenstats_xg/v1/utils/artifacts.py` | ✅ Complete (save/load model artifacts, meta.json; extracted 2026-05-15) |
| `chickenstats_xg/v1/utils/finalize_utils.py` | ✅ Complete (OOF predictions, trial screening, ECE; extracted 2026-05-15) |
| `chickenstats_xg/v1/utils/log_model.py` | ✅ Complete (upload local artifacts to MLflow Model Registry) |
| `chickenstats_xg/v1/experiments.py` | ✅ Complete (base_xg, context_xg gbtree depth-2, pred_goal; fully refactored 2026-05-12; finalize utils extracted 2026-05-15) |
| `chickenstats_xg/v1/base_xg/process_data.py` | ✅ Complete |
| `chickenstats_xg/v1/base_xg/finalize.py` | ✅ Complete |
| `chickenstats_xg/v1/base_xg/score.py` | ✅ Complete |
| `chickenstats_xg/v1/context_xg/process_data.py` | ✅ Complete |
| `chickenstats_xg/v1/context_xg/finalize.py` | ✅ Complete |
| `chickenstats_xg/v1/context_xg/score.py` | ✅ Complete (scores base_xg parquets; enriches RAPM PBP) |
| `chickenstats_xg/v1/rapm/prep_pbp.py` | ✅ Complete (joins base_xg onto raw PBP via event_idx) |
| `chickenstats_xg/v1/rapm/process_stints.py` | ✅ Complete (h_xgf/a_xgf from context_xg) |
| `chickenstats_xg/v1/rapm/regressions.py` | ✅ Complete (3-metric RAPM; context_xg as xGF target) |
| `chickenstats_xg/v1/pred_goal/compute_rolling_stats.py` | ✅ Complete (career/season/10g/EWMA; `_1g` window removed Issue 15) |
| `chickenstats_xg/v1/pred_goal/process_data.py` | ✅ Complete (reads context_xg/scored/; multi-RAPM joins) |
| `chickenstats_xg/v1/pred_goal/finalize.py` | ✅ Complete |
| `chickenstats_xg/v1/base_xg/run_pipeline.py` | ✅ Complete (finalize → score → context_xg process_data) |
| `chickenstats_xg/v1/context_xg/run_pipeline.py` | ✅ Complete (finalize → score → RAPM → pred_goal process_data) |
| `chickenstats_xg/v1/context_xg/diagnose.py` | ✅ Complete (calibration, lift, OOF gap, weight concentration) |
| `nuke_experiment.py` | ✅ Complete |

### Pipeline Execution Progress (v1.0.0)

| Step | Action | Status |
|---|---|---|
| 1 | Scrape raw PBP (2010–2024) | ✅ Done |
| 2 | Build base_xg training data (8 pure geometry features) | ✅ Done |
| 3 | Tune base_xg (500+ trials ES/PP/SH; 150+ EF; 590+ EA) | ✅ Done (2026-05-13) |
| 4a | Finalize base_xg + diagnostics (ES/PP/SH/EF PASS, EA WARN-high-conf) | ✅ Done (2026-05-14) |
| 4b | Score all PBP + build context_xg data (`base_xg/run_pipeline.py`) | ✅ Done (2026-05-14) |
| 5 | RAPM regressions bootstrap | ✅ Done (2026-05-14) |
| 6 | Tune context_xg (750+ ES / 1000+ PP/SH/EF/EA; SH study nuked + restarted) | ✅ Done (2026-05-14) |
| 7 | Finalize context_xg `--top-n 150` + validate diagnostics (all PASS; EA WARN-structural) | ✅ Done (2026-05-14) |
| 8 | Score context_xg + rebuild RAPM stints/regressions + build pred_goal data | ✅ Done (2026-05-14) |
| 9 | Tune pred_goal (500 trials × 5 strengths) | ✅ Done (2026-05-14) |
| 10 | Finalize pred_goal (pooled OOF + hold-out calibration) | ✅ Done (2026-05-14) |
| 11 | Diagnose pred_goal — all FAIL; Issues 14+15 documented | ✅ Done (2026-05-14) |
| 12 | Issue 16 fix — context_xg score.py Booster.predict → XGBClassifier.predict_proba; RAPM recomputed | ✅ Done (2026-05-15) |
| 13 | Issue 15 fix — strip `_1g` features + RAPM subset to xg dims + process_data.py re-run | ✅ Done (2026-05-15) |
| 14 | Additional context_xg tuning (750→1500 ES; 1000→1500 PP/SH/EF/EA) + re-finalize `--top-n 150` + diagnostic | ✅ Done (2026-05-15) |
| 15 | Re-tune pred_goal (500+ trials × 5 strengths — studies stale after feature change) | ⏳ Current Step |
| 16 | Re-finalize + re-diagnose pred_goal; validate all tiers | ⬜ After step 15 |

### Data Status

| Directory | Status |
|---|---|
| `raw_data/pbp/pbp_*.parquet` | ✅ Present (2010–2024) |
| `chickenstats_xg/v1/data/base_xg/train/` | ✅ Present (8-feature pure geometry parquets) |
| `chickenstats_xg/v1/data/base_xg/hold_out/` | ✅ Present |
| `chickenstats_xg/v1/data/base_xg/scored/` | ✅ Present — rebuilt 2026-05-14 with v1.0.0 base_xg |
| `chickenstats_xg/v1/models/base_xg/` | ✅ Present — v1.0.0 finalized 2026-05-14 (ES/PP/SH/EF PASS, EA WARN-high-conf) |
| `chickenstats_xg/v1/data/context_xg/train/` | ✅ Present — rebuilt 2026-05-14 |
| `chickenstats_xg/v1/data/context_xg/hold_out/` | ✅ Present — rebuilt 2026-05-14 |
| `chickenstats_xg/v1/data/context_xg/scored/` | ✅ Present — re-scored 2026-05-15 with Issue 16 fix (XGBClassifier, dist_ratio 1.06–1.65×) |
| `chickenstats_xg/v1/models/context_xg/` | ✅ Present — v1.0.0 re-finalized 2026-05-15 `--top-n 150` (1500/1500 trials; ES/PP/SH WARN-high-conf; EF FAIL-OOF-gap; EA WARN-cal) |
| `chickenstats_xg/v1/data/rapm/pbp/` | ✅ Present — re-enriched 2026-05-15 with corrected context_xg |
| `chickenstats_xg/v1/data/rapm/stints/` | ✅ Present — rebuilt using context_xg for h_xgf/a_xgf |
| `chickenstats_xg/v1/data/rapm/rapm_by_season.parquet` | ✅ Present — re-regressed 2026-05-15 (YOY r=0.317 PASS; all 4 checks PASS) |
| `chickenstats_xg/v1/data/pred_goal/train/` | ✅ Present — rebuilt 2026-05-15 with Issue 15+16 feature set (_1g stripped; RAPM xg only) |
| `chickenstats_xg/v1/data/pred_goal/hold_out/` | ✅ Present — rebuilt 2026-05-15 |
| `chickenstats_xg/v1/models/pred_goal/` | ⏳ Stale — experiments in progress (2026-05-15); re-finalize after tuning completes |

---

## 16. Deployment Checklist

After all training steps complete, deploy to the homelab API:

```
[ ] Copy chickenstats_xg/v1/models/base_xg/{strength}/model.ubj (×5) → app/api/xg_models/base_xg/
[ ] Copy chickenstats_xg/v1/models/base_xg/{strength}/calibrator.joblib (×5) → app/api/xg_models/base_xg/
[ ] Copy chickenstats_xg/v1/models/context_xg/{strength}/model.ubj (×5) → app/api/xg_models/context_xg/
[ ] Copy chickenstats_xg/v1/models/context_xg/{strength}/calibrator.joblib (×5) → app/api/xg_models/context_xg/
[ ] Copy chickenstats_xg/v1/models/pred_goal/{strength}/model.ubj (×5) → app/api/xg_models/pred_goal/
[ ] Copy chickenstats_xg/v1/models/pred_goal/{strength}/calibrator.joblib (×5) → app/api/xg_models/pred_goal/
[ ] Rebuild backend Docker image: docker compose build backend
[ ] Restart: docker compose up -d backend
[ ] Smoke test: POST /inference/pred_goal/live with a test shot → verify non-null pred_goal
[ ] Run compute_pred_goal.py to backfill pbpcs column + sync to R2 Delta table
[ ] Verify: SELECT COUNT(*) FROM pbpcs WHERE pred_goal IS NOT NULL
[ ] Update BASE_XG_MODEL_VERSION in chickenstats/_game_core.py
[ ] Publish new chickenstats version to PyPI
```

---

## 17. Known Gaps & Pending Work

No critical blockers remain. All scripts are implemented. base_xg and context_xg are finalized and validated. Current step: `context_xg/run_pipeline.py` (Step 9 — score context_xg → rebuild RAPM → build pred_goal data).

### Non-blocking

- **`pred_goal/process_data.py` — `strength_state` filter relies on hardcoded string lists.** The `strength_state_map` in `main()` maps strength names to lists of strength state strings (e.g. `"even_strength" → ["5v5", "4v4", "3v3"]`). If `chickenstats` adds new strength state strings, this map needs updating.
- **Structural refactors** — `_objective_body()` monolith, OOF loop duplication, `sys.path.insert()` boilerplate, `model_name` variable naming. Documented in `DECISIONS.md` Proposed Refactors section. Not blocking training.

### Resolved

- **GOAL event fingerprinting** — three-tier cascade architecture (2026-05-12). See `DECISIONS.md` Issue 7.
- **base_xg calibration** — OOF isotonic calibration replaces Platt; `scale_pos_weight` no longer inflates outputs. See Issue 2.
- **pred_goal context leak** — interaction features (`gax_distance_interaction`, `gsax_danger_interaction`) removed. See Issue 3.
- **RAPM prior-season join arithmetic** — fixed `season − 10001` (was 10000). See Issue 4.
- **empty_against calibration ceiling** — isotonic calibrator reaches ~99% for empty-net-against. See Issue 5.
- **`experiments.py` `model_viz()` base_margin bug** — viz now accepts and forwards `base_margin`.
- **`regressions.py` `toi_limits`** — imported from `config.RAPM_TOI_LIMITS`. No longer hardcoded.
- **pred_goal feature stripping** — both `BASE_XG_FEATURE_COLUMNS` (14) and `CONTEXT_XG_FEATURE_COLUMNS` (13) are dropped in `pred_goal/process_data.py` before writing parquets.
- **context_xg base_margin saturation** — `logit(base_xg)` as fixed `base_margin` created a bimodal prediction distribution; all high-base_xg shots clustered at ~0.62–0.65 regardless of sequence context, destroying calibration and aggregate PR-AUC. Fixed by adding `logit_base_xg` as a learnable feature in the constraint groups (Issue 8). Subsequently, `logit_base_xg` was also added back as `base_margin` (in addition to the feature role) to prevent a different bimodal failure mode where flag-group leaf weights cluster non-goal shots at mid-range probabilities (Issue 11). Current architecture: logit_base_xg as BOTH base_margin AND feature.
- **`experiments.py` refactor** — `_objective_body()` monolith split into per-model param builders (`_params_base_xg`, `_params_context_xg`, `_params_pred_goal`) + `_run_cv_folds()` shared CV helper + thin coordinator. Dead code removed. `sys.path.insert` moved to top with both parent paths. (2026-05-12)

---

*This document reflects the state as of 2026-05-15. Three-tier cascade architecture: base_xg (8 pure geometry + shot_type features, OOF isotonic calibration) → context_xg (20 features: logit_base_xg as BOTH base_margin AND feature in 9 constraint groups + binary flags + game-state modifiers + sequence, gbtree depth-2, pooled OOF + hold-out Platt calibration; finalize with `--top-n 150`; Issue 16 scoring fix applied) → pred_goal (talent features only: GxG/GSAx career/season/10g/EWMA + xg RAPM dims; logit(context_xg) as base_margin; re-tuning in progress after Issue 15 feature redesign). RAPM diagnostics: all 4 checks PASS (YOY r=0.317). Utils modularization completed 2026-05-15: finalize_utils.py, artifacts.py, diagnose_utils.py. See `chickenstats_xg/v1/planning/DECISIONS.md` for the full issue history.*