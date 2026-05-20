# v1.0.0 Model Issues & Resolved Solutions

---

## Architectural Decision (2026-05-12)

**Final architecture — three-tier cascade:**

- **Tier 1 — base_xg**: Pure spatial prior (8 features: geometry + shot_type only). XGBoost gbtree. Outputs calibrated P(goal | location, shot type). `data/base_xg/scored/`
- **Tier 2 — context_xg**: `logit_base_xg` (T1 prior) + shot sequence features + game-state modifiers (21 features total). Uses **gbtree depth-2** — `max_depth=2` structurally limits each path to at most 2 features, preventing complex fingerprint paths without needing interaction constraints. `logit_base_xg` is used **both as `base_margin`** (shifting gradient `g = sigmoid(base_margin + F(x)) − y` so trees learn the contextual residual from the T1 prior) **and as a learnable feature** (enabling quality-conditional flag adjustments alongside all other features). Interaction constraints were removed in Issue 21 — they isolated binary flags into low-gain groups that could never compete with the continuous feature block, producing zero importance for `is_rebound`, `is_scramble`, etc. At `max_depth=2`, structural protection is sufficient. See Issues 11 and 21. `data/context_xg/scored/`
- **Tier 3 — pred_goal**: Shooter GxG, goalie GSAx, RAPM talent features layered on top of context_xg prior via `logit(context_xg)` as `base_margin`. XGBoost gbtree. `data/pred_goal/scored/`

**Why three tiers (not two):**

The original two-tier architecture (base_xg with all 26 features → pred_goal) had a structural GOAL event fingerprinting problem. Any model trained on [GOAL=1, SHOT=0] data can exploit the fact that GOAL events have dramatically different shot sequence distributions than SHOT events (e.g., rush_attempt is 12.5× higher for goals). By isolating sequence features in Tier 2 with gbtree `max_depth=2`, we eliminate the fingerprint risk: a depth-2 tree can access at most 2 features per path, making the Rush+Rebound+<10ft multi-feature combination structurally impossible. Interaction constraints were added initially (Issues 7 and 9) but later removed after they blocked binary flags from receiving any feature importance (Issue 21). `max_depth=2` alone provides the necessary structural protection. See Issues 7, 9, and 21 below.

**Cascade mechanics:**
- Each tier's raw prediction is calibrated to true goal rates.
- T2 uses `logit_base_xg` as both `base_margin` and a learnable feature. See Issues 8 and 11.
- T3 uses `logit(calibrated context_xg)` as `base_margin` — the hard 1.0 coefficient is correct here; pred_goal learns only the talent residual.
- `pred_goal/process_data.py` drops the Tier 1 `base_xg` column and carries `context_xg` through to pred_goal parquets. (R6 refactor complete 2026-05-16 — `process-pred-goal` re-run required before pred_goal tuning.)
- `pred_goal - context_xg` is a valid talent delta in probability units.

**RAPM consistency:**
- RAPM regressions use `context_xg` (not `base_xg`) as the xGF regression target.
- This prevents rush/rebound talent effects from being double-counted: once in RAPM coefficients and again in pred_goal's context_xg base_margin.
- `base_xg/score.py` enriches RAPM PBP with `base_xg`; `context_xg/score.py` then adds the `context_xg` column additively. `rapm/process_stints.py` reads `context_xg` for h_xgf/a_xgf.

**Feature column ownership (v1.0.0):**
- `BASE_XG_FEATURE_COLUMNS` (8): pure geometry + shot_type. Used by base_xg; dropped from context_xg and pred_goal parquets.
- `CONTEXT_XG_FEATURE_COLUMNS` (21): `logit_base_xg` (T1 prior) + 4 binary event flags + 4 game-state modifiers (`is_home`, `position`, `strength_state`, `score_diff`) + continuous sequence features (`play_speed`, `seconds_since_last`, `distance_from_last`, `prior_event_angle`, `prior_event_distance`, `seconds_since_stoppage`, `seconds_since_event_team_change`, `seconds_since_opp_team_change`) + `period`/`period_seconds` + `prior_event_same`/`prior_event_opp` (string categoricals via `_apply_fixed_categoricals()`). `position` and `strength_state` also pd.Categorical. No interaction constraints (removed Issue 21 — `max_depth=2` is sufficient protection). Used by context_xg; dropped from pred_goal parquets.
- pred_goal parquets contain only talent features + `context_xg` prior (R6 complete; `process-pred-goal` re-run required to regenerate parquets with updated column names).

---

## Issue 1: base_xg overfitting on GOAL event characteristics

### Problem

base_xg produces near-1.0 predictions for a subset of GOAL events. The model learns to
bifurcate event types rather than learning pure location + shot quality.

### Root Cause

NHL PBP data has a structural constraint: SHOT events always have `goal=0`,
GOAL events always have `goal=1`. Contextual/sequencing features have
dramatically different distributions between the two event types:

| Feature            | GOAL mean | SHOT mean | Ratio  |
|--------------------|-----------|-----------|--------|
| rush_attempt       | 0.178     | 0.014     | 12.5x  |
| is_rebound         | 0.160     | 0.062     | 2.6x   |
| is_scramble        | 0.084     | 0.055     | 1.5x   |
| play_speed         | 9.61      | 8.18      | 1.2x   |
| seconds_since_last | 13.0      | 15.4      | 0.84x  |

Any model trained on [GOAL=1, SHOT=0] data can exploit these differences.
With deep trees (max_depth > 5) and low regularisation, the model can memorise
rush+rebound GOAL events and produce near-1.0 predictions.

### Status: RESOLVED (run 3, 2026-05-11)

Resolved through three compounding regularisation layers:

1. **Parameter bounds** (`experiments.py`): `min_child_weight` 50–150, `max_delta_step` 1–10, `gamma` 0–5, `lambda` 0.1–10, `max_depth` 3–6, `colsample_bynode` 0.4–1.0, `scale_pos_weight` 1.0–10.0.
2. **Interaction constraints** (`config.py` `BASE_XG_INTERACTION_GROUPS`): rush_attempt isolated with temporal features; is_rebound isolated alone; is_scramble isolated alone; geometry features unlisted (free to pair with any group). Prevents rush+rebound+scramble co-occurring on one branch.
3. **Isotonic calibration** (`base_xg/finalize.py`): OOF isotonic calibrator fitted and applied — corrects `scale_pos_weight` inflation and maps model outputs to true goal rates.

**Note — label smoothing removed (2026-05-11):** `LABEL_SMOOTH_EPS` was added in an earlier iteration to cap outputs and was used via `XGBRegressor` to accept float targets. This created a tuning/finalize mismatch (tuning used `XGBClassifier` on 0/1 labels; finalize used `XGBRegressor` on 0.05/0.95 labels). With isotonic calibration handling the output scale, label smoothing is redundant. Both tuning and finalize now use `XGBClassifier` with integer labels — consistent end to end.

**Run 3 results (accepted):**

| Strength | Dist ratio | OOF gap | Calibration | Feature gain | Overall |
|---|---|---|---|---|---|
| even_strength | 6.00× WARN* | 0.035 WARN | PASS | PASS | WARN |
| powerplay | 4.91× WARN* | 0.046 WARN | PASS | PASS | WARN |
| shorthanded | 2.97× WARN | 0.061 WARN† | PASS | PASS | WARN |
| empty_for | 3.31× WARN | 0.047 WARN | WARN | PASS | WARN |
| empty_against | 1.42× PASS | 0.040 WARN | WARN | PASS | WARN |

*Distribution ratios driven by rush+high_danger physics (genuine ~75% goal rate), not memorisation.
†Shorthanded gap is temporal variance (2023-24 training season underperformed hold-out), not overfitting.

**Superseded by Issue 7 — three-tier architecture (2026-05-12):**
The run 3 regularisation approach was accepted at the time, but subsequent analysis revealed that the distribution ratios were driven by genuine physics (not overfitting). The root cause was confirmed as GOAL event fingerprinting (9.4% of 2024 hold-out GOAL events at 90%+ base_xg, 0% of SHOT events). The three-tier architecture moves sequence features to a gblinear Tier 2, making this issue structurally impossible in Tier 1.

---

## Issue 2: base_xg calibration (scale_pos_weight inflation)

### Problem

`scale_pos_weight` is tuned by Optuna (1.0–10.0) and included in `best_params`, which is passed to the final model. For even-strength with ~7% goal rate, `scale_pos_weight` can reach 10x, inflating base_xg outputs from ~7% to ~30%+. This means `logit(base_xg)` passed as base_margin to pred_goal was ~-0.85 instead of the correct ~-2.58, forcing pred_goal to spend capacity correcting for the wrong prior rather than learning talent signals.

A secondary inconsistency: tuning used `XGBClassifier` on 0/1 labels; finalize used `XGBRegressor` on smoothed labels. The optimal scale_pos_weight from tuning did not transfer exactly.

### Status: RESOLVED (2026-05-11)

**Fix applied to `base_xg/finalize.py`:**
- Switched from `XGBRegressor` (smoothed labels) to `XGBClassifier` (integer labels) — matches tuning exactly.
- OOF isotonic calibration: after the OOF loop, fit `IsotonicRegression` on `oof_prob[oof_mask]` vs `y_train[oof_mask]`. Apply calibrator to all train and hold-out predictions before writing to parquets.
- Save `models/base_xg/{strength}_calibrator.joblib` for use in `score.py`.

**Fix applied to `base_xg/score.py`:**
- Load calibrator, apply to `model.predict_proba(X)[:, 1]` before the OOF override. OOF parquet values are already calibrated (written by finalize.py), so no double-application.

**Result:** base_xg outputs now reflect true goal rates. `logit(calibrated_base_xg)` gives the correct prior (~-2.58 for EV). pred_goal can devote all tree capacity to talent signals.

---

## Issue 3: pred_goal context leak (interaction features)

### Problem

`pred_goal/process_data.py` computed two interaction features before dropping `BASE_XG_FEATURE_COLUMNS`:

```python
gax_distance_interaction  = shooter_gax_career × event_distance
gsax_danger_interaction   = goalie_gsax_career × high_danger
```

Both `event_distance` and `high_danger` are in `BASE_XG_FEATURE_COLUMNS`. By multiplying them with talent features, these columns survived the drop and carried shot environment information into pred_goal's feature matrix. The context-leak diagnostic in `diagnose.py` did not catch them (column names not in `BASE_XG_FEATURE_COLUMNS`).

### Status: RESOLVED (2026-05-11)

Both features removed from `pred_goal/process_data.py` and from `xg_utils.py` (schema and `select_columns`). The talent signals (shooter GxG, goalie GSAx) are already present as standalone features; the interaction terms added no information that the cascade architecture can't express through the base_margin mechanism.

---

## Issue 4: RAPM prior-season join arithmetic

### Problem

`pred_goal/process_data.py` used `pl.col("season") - 10000` to compute the prior-season key for RAPM joins. NHL seasons are formatted as YYYYYYYY (e.g., 20112012), and consecutive seasons differ by **10001** not 10000. Every prior-season lookup resolved to a non-existent season, making all per-season RAPM columns 100% null.

### Status: RESOLVED (2026-05-11)

Changed all three occurrences (`_join_shooter_rapm`, `_compute_teammates_rapm`, `_compute_opponent_rapm`) from `- 10000` to `- 10001`. Career RAPM columns (no season lag) were unaffected and were 0% null throughout.

---

## Issue 5: empty_against calibration ceiling

### Problem

Platt (logistic) calibration has a sigmoid ceiling — it cannot output probabilities near 1.0. For empty_against, the actual goal rate in the top prediction decile is ~99.25%. The Platt calibrator could not reach this, producing top-decile calibration error of 0.124.

### Status: RESOLVED (2026-05-11)

For empty_against only, `pred_goal/finalize.py` uses `IsotonicCalibrator` (a wrapper around `sklearn.isotonic.IsotonicRegression`) instead of Platt logistic regression. Isotonic regression fits a monotone step function and can reach 0.99 directly. Top-decile calibration error dropped from 0.124 to 0.039 after the fix. `IsotonicCalibrator` lives in `experiments.py` so all scripts that deserialize the joblib file can import it.

---

## Issue 6: Performance tag gate blocked by label smoothing

### Problem

`compute_performance_tag` in `config.py` gated artifact uploads on both PR-AUC AND log_loss. Label smoothing (eps=0.05) systematically inflates log_loss on binary targets, causing well-performing trials to be tagged "none" and miss artifact uploads. Optuna continued selecting these runs as best trials (PR-AUC was fine) but the tag was wrong.

### Status: RESOLVED (2026-05-11)

`compute_performance_tag` now gates on PR-AUC only. `log_loss_val` is accepted at the call site for compatibility but ignored. Since label smoothing has also been removed entirely, this issue is doubly resolved.

---

## Issue 7: GOAL event fingerprinting in base_xg (three-tier architecture)

### Problem

9.4% of 2024 hold-out GOAL events received base_xg ≥ 90%, while 0% of SHOT events did. The model itself (not the OOF override) was producing these predictions. XGBoost gbtree with enough depth can learn that GOAL events have structurally different rush_attempt / is_rebound / is_scramble distributions than SHOT events and exploit this at inference time.

This is distinct from memorisation: the 90%+ predictions appear in the 2024 hold-out year (not training data), confirming the model learned a generalising fingerprint based on genuine distributional differences between GOAL and SHOT events. No amount of regularisation fully removes this risk when sequence features and the training label are co-produced by the same event type.

### Status: RESOLVED (2026-05-12)

**Three-tier cascade architecture:**

- **Tier 1 — base_xg** uses only geometry + shot_type (8 features in v1.0.0; originally 14 with game-state — see Issue 10). These features have essentially identical distributions for GOAL and SHOT events at the same location. No fingerprint risk.
- **Tier 2 — context_xg** adds sequence features using XGBoost **gblinear** (linear booster). A linear model cannot build the non-linear Rush+Rebound+<10ft interaction that fingerprints goal events. It applies sequence adjustments multiplicatively via the `base_margin` mechanism.
- **Tier 3 — pred_goal** uses talent features only, with `logit(context_xg)` as base_margin.

**Code changes (2026-05-12):**

| File | Change |
|---|---|
| `config.py` | Split `BASE_XG_FEATURE_COLUMNS` to 14 geometry features; added `CONTEXT_XG_FEATURE_COLUMNS` (12 sequence features, incl. OHE'd `prior_event_same`/`prior_event_opp`); cleared `BASE_XG_INTERACTION_GROUPS`; relaxed `MONOTONE_CONSTRAINTS`; added `"context_xg"` to `MODELS`; added `HOLD_OUT_SEASON = 20242025` as single source of truth |
| `experiments.py` | Added context_xg param space (initially gblinear — see Issue 9 for subsequent changes); updated base_margin extraction for context_xg |
| `base_xg/finalize.py` | Relaxed anti-memorisation overrides (depth cap 6, no min_child_weight floor); Platt calibration for non-EA strengths (logistic regression); isotonic for empty_against |
| `context_xg/process_data.py` | New — splits base_xg scored parquets into train/hold_out for context_xg tuning |
| `context_xg/finalize.py` | New — retrain with OOF loop + calibration; writes frozen model + calibrator + OOF parquet (booster type evolved through Issue 9 iterations) |
| `context_xg/score.py` | New — scores base_xg parquets through context_xg gblinear model; enriches RAPM PBP with context_xg column |
| `pred_goal/process_data.py` | Reads from `context_xg/scored/`; renames `context_xg` → `base_xg`; drops both `BASE_XG_FEATURE_COLUMNS` and `CONTEXT_XG_FEATURE_COLUMNS`; `_RAPM_COEFF_COLS` updated to `off_coeff_context_xg`/`def_coeff_context_xg` |
| `rapm/process_stints.py` | `h_xgf`/`a_xgf` now use `context_xg` column (not `base_xg`) |
| `rapm/regressions.py` | `metric_column_map` key `"base_xg"` → `"context_xg"`; `metrics` lists updated; output columns `off_coeff_context_xg`/`def_coeff_context_xg`/`total_rapm_context_xg` |
| `rapm/diagnose.py` | All `off_coeff_base_xg`/`total_rapm_base_xg` references renamed to context_xg variants |
| `base_xg/run_pipeline.py` | Moved from `run_base_pipeline.py`; now 3 steps: finalize → score → context_xg/process_data.py |
| `context_xg/run_pipeline.py` | New — 5 steps: finalize → score → RAPM stints → regressions → pred_goal/process_data.py |
| `context_xg/diagnose.py` | New — calibration, lift over base_xg, OOF gap, weight concentration checks |

---

## Issue 8: context_xg base_margin saturation

### Problem

After the three-tier architecture was implemented (Issue 7), `context_xg/finalize.py` and `score.py`
passed `logit(base_xg)` as `base_margin` to the gblinear model. Diagnostics revealed:

- **Bimodal prediction distribution**: even_strength p75=0.148, p90=0.619 — a cliff with almost no
  predictions between 0.15 and 0.62. The base_margin dominated sigmoid output for high-base_xg shots,
  clustering them all at ~0.62–0.65 regardless of sequence features.
- **Calibration FAIL across all 5 strengths**: decile 8 had mean_pred=0.615, actual=0.024 for
  even_strength. The calibrator had no way to resolve this bimodal structure.
- **Negative aggregate lift**: context_xg aggregate PR-AUC (0.113) < base_xg PR-AUC (0.156) for
  even_strength, despite context_xg beating base_xg in 11 of 15 individual seasons. The high-
  confidence wrong predictions (0.62 predicted, 2.4% actual) destroyed aggregate precision.

### Root Cause

Using `base_margin=logit(base_xg)` fixes the T1 coefficient at **1.0**. For high-base_xg shots
(e.g., logit(0.62) ≈ +0.49), the base_margin saturates the sigmoid before gblinear's small sequence
weights can meaningfully adjust the output. All high-base_xg shots — goals and non-goals alike —
cluster at the same predicted probability. The Platt calibrator, being monotone, cannot undo this.

The fix is architectural, not a tuning problem. No amount of regularisation adjustment resolves it.

### Status: RESOLVED (2026-05-12)

**Architecture change:** Drop `base_margin` for context_xg. Add `logit_base_xg` as a regular
learnable feature in `CONTEXT_XG_FEATURE_COLUMNS`.

- **Why this works:** gblinear learns the coefficient on `logit_base_xg` from data — typically < 1.0
  — giving sequence features room to shift predictions across the full probability range. The cascade
  is preserved: T1's output still feeds T2, but the coupling strength is data-determined rather than
  hard-coded to 1.0.
- **Why T3 keeps base_margin:** pred_goal should treat the full geometry+sequence prior as a fixed
  starting point and learn only the talent residual. The saturation risk does not apply because
  context_xg's distribution, after fixing T2, will be well-spread across [0, 1].

**Code changes (2026-05-12):**

| File | Change |
|---|---|
| `config.py` | Added `"logit_base_xg"` as first entry in `CONTEXT_XG_FEATURE_COLUMNS` (12 → 13 features) |
| `context_xg/process_data.py` | Computes `logit_base_xg = log(base_xg / (1 - base_xg))` before column selection |
| `experiments.py` | context_xg `load_data`: removed `base_xg` sidecar; `logit_base_xg` now in feature matrix via `CONTEXT_XG_FEATURE_COLUMNS`. `use_base_margin` now pred_goal-only |
| `context_xg/finalize.py` | `_split_df()` returns `(X, y)` only — no base_margin; all `base_margin=` args removed |
| `context_xg/score.py` | Computes `logit_base_xg` from `base_xg` before feature selection; `DMatrix` built without `base_margin` |
| `context_xg/diagnose.py` | New — added "Lift over base_xg" check and "Weight concentration" check (replaces gain dominance for gblinear) |

---

## Issue 9: context_xg T2 feature design — iteration history

### Problem

After implementing Issue 8 (logit_base_xg as a learnable feature), context_xg calibration
still failed across all five strengths. Diagnostics revealed:

- **Bimodal prediction distribution**: even_strength showed p75=0.148, p90=0.619 — a cliff with
  almost no predictions between 0.15 and 0.62.
- **Calibration FAIL**: decile 8 had mean_pred=0.62, actual=0.024 for even_strength. No monotone
  calibrator can resolve a bimodal raw prediction distribution.

Three distinct T2 approaches have been attempted. The root cause in each case is that gblinear's
additive structure is the wrong inductive bias for shot sequence features.

---

### Attempt 1 — Binary flags in gblinear (FAILED)

**Approach:** Use `is_rebound`, `is_scramble`, `rush_attempt`, `prior_face` as binary features
in a gblinear model. gblinear adds a fixed log-odds shift per feature.

**Failure mode:** Multiple flags co-firing adds their fixed shifts additively. A poor-angle shot
(base_xg=0.03) with 3 flags receives the same total boost as a high-danger shot (base_xg=0.20)
with the same flags. This creates a "flag cluster" at a fixed predicted probability (~0.62–0.65)
whenever ≥2 flags are active simultaneously. No calibrator can resolve a bimodal raw distribution.

Switching to `IsotonicCalibrator` made things worse: it memorised the OOF distribution exactly
(abs_err=0.0000 for training deciles 0–6), masking the bimodal structure rather than correcting it.

**Root cause:** gblinear's additive structure — fixed log-odds per feature, regardless of shot
quality. The bimodal cliff is structural; no tuning or calibration approach can fix it.

---

### Attempt 2 — Quality interaction features (`flag × base_xg`) in gblinear (FAILED)

**Approach:** Replace binary flags with quality interaction features:

| Binary (removed) | Quality (added) | Computation |
|---|---|---|
| `is_rebound` | `rebound_quality` | `is_rebound × base_xg` |
| `is_scramble` | `scramble_quality` | `is_scramble × base_xg` |
| `rush_attempt` | `rush_quality` | `rush_attempt × base_xg` |
| `prior_face` | `face_quality` | `prior_face × base_xg` |

Rationale: each quality feature is bounded [0, 1], sparse (zero when flag not fired), and scales
the sequence boost continuously with shot quality. Also switched to in-sample Platt calibration
(fit on `model.predict_proba(X_train)[:, 1]`) to avoid the OOF/final-model distribution mismatch.
Lambda lower bound raised to 1.0.

**Failure mode:** `rebound_quality = is_rebound × base_xg` is nearly collinear with
`logit_base_xg` in a linear model — both are monotone transformations of `base_xg`. L2
regularisation (lambda) cannot distinguish their contributions and drives quality feature weights
toward zero. Diagnosis on the finalized model:

- **Powerplay**: `logit_base_xg` weight = 0.000% — the model degenerated to ~2 effective features
  with alpha/lambda fighting each other. Quality features invisible.
- **Even strength**: quality features had < 5% combined weight despite being the intended signal.
- **Overall**: context_xg PR-AUC matched base_xg PR-AUC for most strengths — the T2 layer learned
  nothing the T1 prior didn't already provide.

**Root cause:** Collinearity between `flag × base_xg` and `logit_base_xg` in a linear model.
L2 regularisation cannot distinguish the two and kills the quality features. This is structural
to gblinear regardless of lambda range or feature scaling.

---

### Attempt 3 — gbtree depth-2 with flag isolation constraints (IN PROGRESS)

**Approach:** Restore binary flags (`is_rebound`, `is_scramble`, `rush_attempt`, `prior_face`).
Switch from gblinear to gbtree with `max_depth=2` and per-flag interaction constraints:

```
Group 0: [logit_base_xg, is_rebound]    ← depth-2 tree learns quality-conditional rebound boost
Group 1: [logit_base_xg, is_scramble]   ← quality-conditional scramble boost
Group 2: [logit_base_xg, rush_attempt]  ← quality-conditional rush boost
Group 3: [logit_base_xg, prior_face]    ← quality-conditional faceoff boost
Group 4: [play_speed, seconds_since_last, distance_from_last, prior_event_angle,
          prior_event_distance, seconds_since_stoppage, prior_event_same, prior_event_opp]
```

**Why this solves both failure modes:**
- *Bimodal cliff*: no tree can combine `is_rebound` AND `is_scramble` (different groups) → multi-flag
  additive cliff eliminated. Max depth-2 path in any flag group is `logit_base_xg > threshold →
  is_flag == 1 → leaf` — structurally impossible to combine two flags on one path.
- *Collinearity*: tree splits are threshold comparisons, not linear coefficients. `logit_base_xg`
  dominating as the first split in each flag group is expected and correct — it represents "what is
  the shot quality before the flag is checked?" and the second split then learns the quality-
  conditional flag effect.
- *Fingerprinting*: depth=2 + one-flag-per-group prevents the Rush+Rebound+<10ft multi-feature
  path that memorises goal events.

**Calibration:** Pooled OOF + hold-out Platt (LogisticRegression, C=1.0) for all strengths.
Three calibration approaches were attempted:

- *Hold-out isotonic (FAILED):* Only ~1,000 EA hold-out shots → isotonic produces a coarse step
  function that extrapolates wildly (EA abs_err=0.2492). Switched to Platt.
- *Hold-out Platt (FAILED for PP/SH):* Temporal mismatch — calibrator fit on 2023-25 hold-out
  under-represents the full 15-season training range; deciles 7-8 over-predicted (PP abs_err=0.191).
- *Pooled OOF + hold-out Platt (CURRENT):* OOF fold models trained to `model.best_iteration`
  (no early stopping; same tree count as final model) align the fold-model probability scale.
  Pooling training OOF + hold-out gives calibrator coverage across all 15 seasons, eliminating
  the temporal mismatch. Platt's sigmoid ceiling (max < 1.0) is safe here; Issue 5's ~99% ceiling
  concern applies to base_xg EA, not context_xg EA (~89% top-decile actual rate).

**`logit_base_xg` as feature (not base_margin alone):** If passed as `base_margin` only and excluded from the feature matrix, each flag group has only one feature (binary) → depth-2 tree can only split once → equivalent to gblinear's additive coefficient. `logit_base_xg` must be IN the feature matrix, paired with each flag in its constraint group. After confirming the bimodal failure is structural (not tuning-solvable — see Issue 11), `logit_base_xg` was added as `base_margin` in addition to its role as a feature.

**No `colsample_*` params:** Column subsampling selects features before interaction constraints are
applied. If a group's features aren't sampled for a given tree, that tree degenerates silently. Fixed
at 1.0 by omitting all `colsample_*` from the param space.

**Status: RESOLVED (2026-05-12/13)** — gbtree depth-2 with interaction constraints is the confirmed architecture. Superseded by Issue 10 (feature enrichment) and Issue 11 (base_margin addition), neither of which changes the structural constraint design.

**Code changes (2026-05-12):**

| File | Change |
|---|---|
| `config.py` | `CONTEXT_XG_FEATURE_COLUMNS`: reverted to binary flags (removed quality features); added `CONTEXT_XG_INTERACTION_GROUPS` (5 groups, logit_base_xg paired in first 4) |
| `experiments.py` | context_xg param block: `gblinear` → `gbtree`, `max_depth=2` fixed, `interaction_constraints` from `_build_context_interaction_constraints()`; added `_build_context_interaction_constraints()` helper; removed `_apply_context_ohe()` call for context_xg (gbtree uses `_apply_fixed_categoricals()` with `enable_categorical=True`); removed `if data.model != "context_xg":` guard on feature importance (gbtree supports `get_fscore()`) |
| `context_xg/finalize.py` | Params: `gblinear` → `gbtree`, `max_depth=2`, `interaction_constraints`; calibration: pooled OOF + hold-out Platt (`LogisticRegression`); OOF loop runs to `model.best_iteration` (no early stopping) to match final model scale; `_split_df()` calls `_apply_fixed_categoricals()` |
| `context_xg/process_data.py` | Removed 4 quality feature assignments; only `logit_base_xg` computed |
| `context_xg/score.py` | Same quality features removed; `_apply_fixed_categoricals()` replaces `_apply_context_ohe()`; DMatrix `enable_categorical=True` |
| `context_xg/diagnose.py` | `check_weight_concentration()` (gblinear-specific, `importance_type='weight'`) replaced with `check_feature_gain_concentration()` (`importance_type='gain'`); thresholds PASS<60%, WARN<80% (logit_base_xg legitimately high in flag groups) |

---

## Issue 10: Feature architecture migration — tier purity + empty_against OOF gap

### Problem

base_xg v1.0.0 (14 features) had two compounding problems diagnosed from the 500-trial run:

**1. empty_against OOF gap FAIL (0.0868):** `score_diff` at 26.3% of feature gain encoded era-specific goalie-pull timing. Analytics adoption shifted the median pull from down-1 (2010-2016) to down-2 (2020-2025). The 2024-25 hold-out had 13.2% of events at sd≥3 vs. 8.7% in training (52% increase in the distribution tail). The model's score_diff-driven splits didn't generalise to this shifted distribution.

**2. Tier purity violation:** `position` (24.9% gain in PP, 11% in SH), `strength_state`, `is_home`, `period`, `period_seconds` are personnel/tactical/temporal features that should *condition on* shot quality, not define it. Keeping them in base_xg meant the "pure spatial prior" was contaminated by game-management signals — the wrong architectural layer.

### Status: RESOLVED (2026-05-13)

**Architecture change in `config.py`:**

`BASE_XG_FEATURE_COLUMNS` stripped from 14 → **8 features** (pure geometry + shot_type):

```
event_distance, event_angle, coords_x, coords_y, abs_y_distance, danger, high_danger, shot_type
```

The 6 removed features (`score_diff`, `period`, `period_seconds`, `is_home`, `position`, `strength_state`) are migrated to `CONTEXT_XG_FEATURE_COLUMNS` (13 → **20 features**), each with its own isolated interaction constraint group:

```
group 4: [logit_base_xg, is_home]
group 5: [logit_base_xg, position]
group 6: [logit_base_xg, strength_state]
group 7: [logit_base_xg, score_diff]
```

`logit_base_xg` is also added to the continuous sequence block (group 8), allowing depth-2 trees to ask "how does rush speed modify the spatial prior?"

**Why this resolves the empty_against OOF gap:** `score_diff` no longer appears in base_xg at all. In context_xg, it is constrained to pair only with `logit_base_xg` — the tree can ask "given this shot quality, does score state shift danger?" but cannot combine score_diff with period_seconds to build the era-fingerprint path `[score_diff < -1] AND [period_seconds > 3400]` that drove the OOF failure.

**Why this resolves tier purity:** base_xg is now a mathematically pristine spatial prior: it does not know if a 15-foot shot is 5v5 or 3v3, a forward or a defenseman, or whether the home team is desperate. context_xg applies all game-state conditioning under the structural guarantee of max_depth=2 + isolated constraints.

| File | Change |
|---|---|
| `config.py` | `BASE_XG_FEATURE_COLUMNS` 14→8; `CONTEXT_XG_FEATURE_COLUMNS` 13→20; `CONTEXT_XG_INTERACTION_GROUPS` 5→9 groups (added 4 state pairs, logit_base_xg in continuous block) |

**base_xg v1.0.0 retrain completed 2026-05-13:** 500+ trials for ES/PP/SH, 150+ for EF, 590+ for EA. All 5 strength states finalized. Diagnostics: ES/PP/SH/EF PASS, EA WARN (high-confidence check only — physically appropriate). Context_xg tuning ready to begin.

---

## Issue 11: context_xg bimodal calibration failure — base_margin addition

### Problem

After ~500 tuning trials across all 5 strength states (Issue 9 Attempt 3 + Issue 10 enrichment),
context_xg calibration FAILs for 4 of 5 states. The bimodal cliff is structural:

| State | ΔLog Loss % vs null | ΔBrier % vs null | Notes |
|---|---|---|---|
| even_strength | −221.8% | −376.2% | worsened vs 100-trial run (−195.5% / −323.4%) |
| powerplay | −135.7% | −218.5% | unchanged vs 100-trial run |
| shorthanded | −3.4% | −1.8% | improved (FAIL → WARN) |
| empty_for | −37.0% | −42.0% | worsened vs 100-trial run (−12.2% / −10.5%) |
| empty_against | +1.5% | +1.6% | passes; unchanged |

Crucially, ES and EF **worsened** with more trials. The optimizer found configurations with stronger
flag boosts, pushing the bimodal cluster higher. This is the definitive signal that the failure is
structural — not a tuning problem.

### Root Cause

Without `base_margin`, the trees must predict the full sigmoid output range from scratch. A depth-2
tree for the `[logit_base_xg, is_rebound]` group fires a large leaf value when `is_rebound=1`,
regardless of `logit_base_xg` magnitude — a poor-angle rebound (base_xg=0.02) and a slot rebound
(base_xg=0.18) receive the same boost. This clusters ~10% of events near 0.48–0.51 regardless of
whether they score, producing a bimodal distribution that Platt calibration cannot resolve.

### Decision: logit_base_xg as both base_margin and feature

`base_margin = logit_base_xg` shifts the gradient at each training example:

```
g = sigmoid(logit_base_xg + F(x)) − y
```

Trees now learn the **contextual residual** from the T1 spatial prior. A high-quality shot that
doesn't score has a strong negative gradient — the tree learns to output a small positive value for
the flag on that shot, rather than the same large value regardless of base quality. The bimodal
cliff collapses: the model no longer needs to blindly boost all flag events to the same probability.

**Why keep logit_base_xg in the feature matrix too:**

Two distinct roles that are architecturally complementary, not redundant:

1. **`base_margin`** — provides the unconditional linear shift (main effect, coefficient fixed at 1.0)
2. **Feature in `X`** — enables quality-conditional flag adjustments (non-linear interaction effects,
   captured by the tree's second split in each constraint group)

If logit_base_xg were dropped from the feature matrix, each flag group would have only one feature
(the binary flag), and a depth-2 tree could only make one split — equivalent to gblinear's additive
coefficient. All 9 interaction constraint groups would be functionally broken.

**Why this doesn't cause collinearity problems (unlike gblinear):**

With gblinear (Issue 8), the linear optimizer had two competing degrees of freedom for the same input:
the fixed 1.0 base_margin coefficient and the learnable feature weight. These fought each other and
produced saturation artifacts. With gbtree, trees make threshold splits, not linear weight estimates.
The base_margin provides the gradient offset; the feature enables conditional splits. No competition
exists. Regularisation (lambda, min_child_weight) and the monotone constraint on logit_base_xg prevent
the tree from learning anti-correlated splits that would partially cancel the base_margin effect.

**Verification:** After retuning, calibration should PASS or WARN (not FAIL). Log loss and Brier
should be positive vs null. The bimodal cliff (SHOT p90 ~0.51 currently) should collapse to <0.20.
PR AUC and OOF gap should remain comparable to the current ~500-trial run.

### Status: RESOLVED (2026-05-13)

**Code changes:**

| File | Change |
|---|---|
| `experiments.py` | `_objective_body()`: replaced `use_base_margin = ...` block with if/elif chain; context_xg branch sets `bm_train/bm_test = X_train["logit_base_xg"]` without dropping the column from the feature matrix |
| `context_xg/finalize.py` | `_split_df()`: now returns `(X, y, bm)` where `bm = df["logit_base_xg"]`; `_finalize_one()`: passes `base_margin=bm_train` and `base_margin_eval_set=[bm_hold_out]` to `model.fit()`; OOF loop slices and passes `bm_tr`/`bm_val` per fold; all three `predict_proba()` calls pass the appropriate `base_margin=` argument |
| `context_xg/score.py` | `score_strength()`: `DMatrix` now built with `base_margin=logit_bm` in addition to the feature column |

**Operational requirement:** Existing context_xg Optuna studies (~500 trials) trained without
`base_margin` have different gradient landscapes and their optimal hyperparameters do not transfer.
Nuke all 5 studies and restart tuning with 500+ trials before finalizing.

---

## Issue 12: Post-base_margin diagnostics — middle-band calibration failure and SH catastrophic regression

### Context

First context_xg diagnostic run after the base_margin addition (Issue 11) and full re-tuning (~500 trials per state in the new gradient landscape). Results confirm the base_margin architecture is correct but reveal two remaining failure modes requiring additional work.

### What improved (confirming Issue 11 fix works)

| State | Old SHOT p90 (no bm) | New SHOT p90 (bm) | Old ΔLL% | New ΔLL% | Change |
|---|---|---|---|---|---|
| even_strength | 0.513 | **0.217** | −221.8% | **−33.7%** | ✅ Major improvement |
| powerplay | ~0.51 | **0.197** | −135.7% | **−2.6%** | ✅ Major improvement |
| empty_for | ~0.25 | **0.355** | −37.0% | −77.8% | ⚠️ Mixed |
| empty_against | ~0.27 | 0.686 | +1.5% | +0.4% | ✅ Slight improvement |
| shorthanded | normal | **0.784** | **−3.4%** | **−462.6%** | ❌ Catastrophic regression |

OOF gaps: all PASS (0.001–0.026). EA OOF gap fixed from 0.030 (WARN) to 0.007 (PASS). The bimodal cliff structure collapsed for ES and PP, confirming the base_margin gradient anchor works as designed.

### Failure Mode 1: Middle-band overestimation (ES, PP, EF, EA)

Deciles 0–6 (predictions ≤ p70) calibrate well across all four states. Decile 9 (top 10%) calibrates reasonably. But deciles 7–8 (70th–90th percentile) are **2–4× overestimated** in ES, PP, and EF:

- ES dec 8: mean_pred=0.203, actual=0.057 (3.6× overestimate)
- PP dec 7: mean_pred=0.162, actual=0.085 (1.9× overestimate)
- EF dec 8: mean_pred=0.347, actual=0.054 (6.4× overestimate)
- EA dec 5: mean_pred=0.640, actual=0.488 (1.3× overestimate, shifted range)

This is not bimodal — SHOT p90 values are normal for ES/PP (0.217, 0.197). The pattern is a concentration of shots at mid-range predictions (~0.14–0.35 depending on state) where context features (rebound, scramble, seconds_since_stoppage, seconds_since_last) fire large leaf boosts for shots that don't actually convert at those rates.

**Root cause:** The depth-2 tree's leaf values in the flag constraint groups are too large for medium-quality shots. The Platt calibrator (monotone) cannot simultaneously compress deciles 7–8 and preserve decile 9 — this is a regularization tuning problem, not a calibration strategy problem. Higher lambda/gamma values shrink leaf weights, reducing the overcorrection. 500 trials did not sufficiently explore the high-regularization region in the new gradient landscape.

**Fix:** Add 500+ trials to each existing study (ES, PP, EF, EA). The calibrated top-N screening (Issue 12 tooling already in place) will ensure the selected trial has cal_ll < 2× null. Re-finalize after more coverage. Do NOT nuke studies — the 500 completed trials provide valid landscape information; extending them is the right approach.

### Failure Mode 2: SH catastrophic regression (bimodal at 0.78–0.82)

Shorthanded went from ⚠️ WARN (log loss −3.4% vs null, previous run without base_margin) to ❌ FAIL catastrophic (log loss **−462.6%** vs null). SHOT p90 jumped from normal to **0.784**. Decile 8 has mean_pred=0.775, actual=0.018.

This bimodal cliff at 0.78–0.82 is structurally different from the pre-base_margin ES/PP cliff (~0.48–0.51). For SH, the dominant feature `seconds_since_stoppage` (33.5% gain) fires a large leaf (+3.5–4.0 log-odds) for a specific SH play pattern that doesn't convert at the predicted rate. With base_margin = logit_base_xg already providing −2.5 to −3.0 log-odds for typical SH shots, the tree adds +4 log-odds to reach 0.78+ predictions.

**Why the calibrated top-N screening failed for SH:** The screening hit its fallback path — all 15 top SH candidates in the new gradient landscape had calibrated log loss >> 2× null (bimodal). The least-bad bimodal was selected. SH's small dataset (35,983 shots; 2,592 hold-out events) creates noisy optimization gradients, and 500 trials in a completely fresh landscape were insufficient to find non-bimodal configurations with the required high regularization.

**Fix:**
1. **Nuke the SH study and restart** — existing 500 trials are all in bimodal territory; adding to the same study risks getting stuck.
2. Run **1000+ trials** for SH. More coverage in the fresh landscape is required to find configurations where leaf weights are small enough to prevent the 0.78+ cliff.
3. After retuning, re-finalize with `--top-n 15`. The screening should now have at least some non-bimodal candidates.
4. SH is the **critical path** — its calibration failure makes context_xg counterproductive for SH shots in pred_goal's base_margin. Resolve before proceeding to pred_goal.

### Additional observation: EA logit_base_xg feature gain (0.9%)

EA's logit_base_xg has only 0.9% feature gain (vs 4.5–14.5% for other states). For EA, base_xg values span a narrow range (many shots are high-danger close-in situations, base_xg ≈ 0.4–0.7, logit_base_xg ≈ −0.4 to +0.85). The base_margin already places EA predictions at 50–70% before tree splits, leaving little residual for the feature version of logit_base_xg to contribute. The interaction constraint groups that pair logit_base_xg with each flag/state feature are effectively not learning quality-conditional adjustments for EA.

This may require architectural investigation after more tuning: specifically whether the isolated logit_base_xg constraint groups are the right design for EA. For now, more tuning is the first step (acceptable outcome: log loss < 10% worse than null).

### Status: RESOLVED (2026-05-14)

**Tuning completed:** ES 750 trials; PP/SH/EF/EA 1000 trials each (SH study was nuked and restarted per plan).

**Root cause of screening failure identified:** The CV PR-AUC landscape is nearly flat — the top-5 candidates across all states span ≤0.0005 PR-AUC. Bimodal models (max_delta_step ≥ 2) rank shots correctly and score comparably to non-bimodal models on PR-AUC, so they dominate the top-15 candidates. Non-bimodal (max_delta_step=1) trials consistently sit at rank 16–80. With `--top-n 15`, all 15 candidates fail the 2× null cal_ll threshold and the fallback selects the least-bad bimodal model — catastrophic miscalibration.

**Fix:** Re-finalized with `--top-n 150`, exposing mds=1 candidates. All 5 selected trials have max_delta_step=1 (confirmed via Optuna DB inspection + best_iteration matching). Hyperparameters for selected trials documented in `context_xg/diagnostics.md`.

**Diagnostic results after `--top-n 150` re-finalization:**

| State | ΔLL% vs null | ΔBrier% vs null | SHOT p90/base_rate | Overall |
|---|---|---|---|---|
| even_strength | +11.1% | +13.6% | 1.79× PASS | ✅ PASS |
| powerplay | +5.3% | +7.5% | 2.05× PASS | ✅ PASS |
| shorthanded | +6.7% | +8.1% | 1.91× PASS | ✅ PASS |
| empty_for | +8.3% | +10.2% | 1.88× PASS | ✅ PASS |
| empty_against | +3.1% | +4.2% | PASS | ⚠️ WARN (mid-range calibration) |

EA WARN is structural mid-range compression (decile 5-6, max abs error 0.059) driven by Platt calibration's inability to simultaneously compress mid-range and preserve extremes after the base_margin anchor at 50–70%. Not a bimodal failure. Acceptable for v1.0.0.

**Critical insight for future finalization:** Always use `--top-n 150` (or higher). The `--top-n 15` default is unsafe for context_xg due to the flat CV landscape. The `_screen_trials()` function correctly gates on calibrated log loss; the failure was that the top-15 window excluded all passing (mds=1) candidates.

---

## Issue 13: diagnose.py check logic errors — inverted distribution check and non-base-rate-aware high-confidence check

### Problem

Two diagnostic checks in `context_xg/diagnose.py` were producing incorrect pass/fail results:

**1. Distribution check (inverted):** `check_distribution()` computed `GOAL p90 / SHOT p90` and flagged high values as WARN/FAIL. This is backwards — a high ratio means goals score much higher than non-goals, which is *excellent discrimination*. Well-calibrated models with good discrimination were being flagged as FAIL (ratio ~7.46×) while bimodal models were passing (ratio ~1.16× — goals and non-goals had nearly identical distributions, both near the cliff). The check was measuring the opposite of what was intended.

**2. High-confidence check (non-base-rate-aware):** `check_high_confidence()` used fixed thresholds (GOAL_WARN=10%, SHOT_WARN=1%) for all strength states. For empty_against with a 56.7% base rate, 25.9% of goals naturally exceed 0.80 probability (well-calibrated behavior). The fixed threshold triggered a FAIL despite 91.3% precision at that threshold. Any state with a naturally elevated base rate was systematically penalized.

### Status: RESOLVED (2026-05-14)

**Fix 1 — Distribution check:** Changed metric from `GOAL p90 / SHOT p90` to `SHOT p90 / base_rate`. The new check detects bimodal cliffs: a non-goal shot reaching high predicted probability means the model is placing non-goal events in high-probability territory. Well-calibrated models have `shot_p90 ≈ 1–2× base_rate`. Bimodal models have `shot_p90 >> base_rate`. Thresholds: WARN = 2.5×, FAIL = 5.0×.

**Fix 2 — High-confidence check:** Thresholds now scale with `base_rate / 0.07` (reference base rate). `scale = max(1.0, base_rate / 0.07)`. Adjusted thresholds are capped at 50% (GOAL_WARN) and 10% (SHOT_WARN) to avoid absurd values at extreme base rates. EA's 56.7% base rate → scale=8.1×, adjusted thresholds accommodate naturally elevated predictions without penalizing calibration.

**Code changes (`context_xg/diagnose.py`):**

| Change | Detail |
|---|---|
| Threshold constants | `P90_RATIO_WARN/FAIL` → `SHOT_P90_BASE_RATIO_WARN=2.5, SHOT_P90_BASE_RATIO_FAIL=5.0`; `_BASE_RATE_REF=0.07` added |
| `check_distribution()` | Metric changed from `goals.quantile(0.90) / shots.quantile(0.90)` to `shots.quantile(0.90) / base_rate` |
| `check_high_confidence()` | `scale = max(1.0, base_rate / _BASE_RATE_REF)` applied to all thresholds; `adj_*` variables used for status evaluation |

---

## Issue 14: pred_goal OOF calibration temporal drift — catastrophic miscalibration

### Problem

First pred_goal diagnostic run (2026-05-14, 500 trials per state, OOF-only Platt calibration):

| State | Log Loss | Null LL | Ratio | Assessment |
|---|---|---|---|---|
| even_strength | 2.4735 | 0.2260 | **10.9× null** | Catastrophic |
| powerplay | 1.1272 | 0.3310 | **3.4× null** | Catastrophic |
| shorthanded | 1.4037 | 0.2592 | **5.4× null** | Catastrophic |
| empty_for | 0.9769 | 0.2722 | **3.6× null** | Catastrophic |
| empty_against | 0.6426 | 0.6842 | 0.94× null | PASS |

All four low-base-rate states had log loss 3–11× worse than a null model that simply predicts the base rate for every event. The model was actively destroying information.

The distribution check also revealed the cause: decile 8 (mean_pred=0.921 for ES) had only 2.3% actual goal rate — a 40:1 overestimate. The calibrated predictions were tri-modal (low cluster ≈ 0.04, medium ≈ 0.40, high ≈ 0.92–0.94) with the high cluster being catastrophically wrong.

### Root Cause

`pred_goal/finalize.py` fit the Platt calibrator (LogisticRegression) exclusively on OOF training predictions. In OOF data spanning 2010–2023, extreme talent matchup features — specifically `goalie_gsax_per_shot_1g` (45% of gain in ES) representing a goalie's per-shot performance in their last 1 game — identified shots in training where the goalie was having a disastrous stretch AND shot quality was high. In the training era, these combinations genuinely produced high goal rates (~30–90% in the specific training-era folds). The OOF calibrator learned: raw_prob ≈ 0.97 → calibrated ≈ 0.92.

In the hold-out season (2024-25), the model produces the same high raw probabilities for similar-looking feature combinations, but the actual goal rate for those events is only 2.3%. The OOF calibrator cannot correct for this because it was fit on training-era data with a different distribution.

**Why the `_screen_trials()` calibration did not catch this:** The screening applies a quick Platt calibration fitted *on hold-out data directly* (transductive calibration). With hold-out raw probs bimodal at ~0.03 and ~0.97, and hold-out actual rates of ~3% and ~10% for those clusters, the screening calibrator correctly maps 0.97 → 0.10. Screening cal_ll ≈ null or better → the trial PASSES the 2× threshold. But the production OOF calibrator fits on training-era data where the same raw 0.97 cluster had 40–90% actual goal rates → maps 0.97 → 0.92. The screening and production calibrations operate on different distributions; screening cannot detect temporal calibration drift.

### Status: RESOLVED (2026-05-14)

**Fix applied to `pred_goal/finalize.py`:** Switched from OOF-only calibration to pooled OOF + hold-out calibration, matching the approach used in `context_xg/finalize.py`.

```python
# Pool OOF training probs with hold-out raw probs for calibration.
hold_raw = base_model.predict_proba(X_hold_out, base_margin=bm_hold_out)[:, 1]
pool_probs = np.concatenate([oof_prob[oof_mask], hold_raw])
pool_labels = np.concatenate([y_train.to_numpy()[oof_mask], y_hold_out.to_numpy()])
calibrator = LogisticRegression(C=1.0, max_iter=1000).fit(pool_probs.reshape(-1, 1), pool_labels)
```

This anchors the calibrator to the hold-out (inference-time) distribution, preventing the temporal drift. The tradeoff is that hold-out log loss is slightly optimistic (calibrator saw hold-out labels), but the probabilities become usable.

**Results after fix:**

| State | Log Loss Before | Log Loss After | Null LL | Improvement |
|---|---|---|---|---|
| even_strength | 2.4735 (−994%) | **0.2660 (−17.7%)** | 0.2260 | 56× better |
| powerplay | 1.1272 (−241%) | **0.3930 (−18.8%)** | 0.3310 | 12× better |
| shorthanded | 1.4037 (−442%) | **0.3175 (−22.5%)** | 0.2592 | 16× better |
| empty_for | 0.9769 (−259%) | **0.3132 (−15.0%)** | 0.2722 | 9× better |
| empty_against | calibration unchanged | **0.6179 (+9.7%)** | 0.6842 | calibration PASS |

**Residual calibration issue:** A decile-8 non-monotone pattern persists for ES/PP/SH/EF. The decile with the second-highest mean predictions has a lower actual goal rate than the decile below it (e.g., ES: dec 8 pred=0.152 actual=0.023 vs dec 7 pred=0.091 actual=0.130). This is the residual of the bimodal structure: the pooled calibrator found a compromise between training-era statistics (where the high cluster had ~30–90% actual rate) and hold-out statistics (where it has ~2–4%), but even the compromise value (0.15–0.19) overestimates at hold-out time. The calibration FAIL threshold (max abs error > 0.10) is still being triggered; see Issue 15 for the broader interpretation and path forward.

---

## Issue 15: pred_goal negligible lift over context_xg and EA negative lift

### Problem

After the pooled calibration fix (Issue 14), pred_goal's discrimination metrics reveal a deeper issue: the model adds essentially no ranking improvement over context_xg.

**Lift over context_xg (hold-out PR AUC):**

| State | ctx_xg PR AUC | pred_goal PR AUC | Lift |
|---|---|---|---|
| even_strength | 0.3202 | 0.3204 | **+0.0001** |
| powerplay | 0.3415 | 0.3420 | **+0.0005** |
| shorthanded | 0.3330 | 0.3336 | **+0.0006** |
| empty_for | 0.3112 | 0.3121 | **+0.0009** |
| empty_against | 0.7820 | 0.7480 | **−0.0340** |

The seasonal PR-AUC table confirms: pred_goal and ctx_xg produce near-identical values in every single season across all 15 training years. The talent tier is adding no meaningful discriminative signal.

**EA negative lift** of −0.034 PR AUC is consistent across all 15 seasons — pred_goal degrades EA performance relative to context_xg in every single year in the dataset.

### Root Cause

**Non-EA states — feature design dominated by noisy short-horizon signals:**

Feature gain is dominated by rolling goalie performance features in every non-EA state:
- ES: `goalie_gsax_per_shot_1g` (45%), `goalie_gsax_ewma` (28%), `goalie_gsax_per_shot_10g` (21%) = **94% from goalie features**
- PP: goalie + shooter gax features split evenly but both are rolling/ewma metrics
- SH/EF: similar pattern

RAPM features (career-level stable talent signal) account for < 1% of gain in ES, ~1% in PP/SH, and ~5.5% in EF. The model has learned to ask "is the goalie having a bad recent game?" (short-horizon goalie form) rather than "is this shooter genuinely talented?" (stable career talent).

Two compounding reasons this produces negligible lift:
1. **Context_xg already captures much of the goalie quality signal** implicitly through score_diff, strength_state, is_home, and period interactions. The marginal contribution of explicit goalie rolling features is small.
2. **Short-horizon features are too noisy to generalize across seasons.** `goalie_gsax_per_shot_1g` gets 45% gain in training because single-game goalie performance genuinely predicts goals within the training era. But this pattern doesn't transfer reliably to the hold-out season because specific player-goalie matchup combinations from training (Ovechkin vs. a goalie on a cold streak) don't replicate.

**EA negative lift — structural:**

In empty-against situations, the goalie has been pulled. Goalie form/talent features are irrelevant by construction. Shooter talent features add noise because EA goals are determined by shot quality (already captured by context_xg geometry + score state), not by who is shooting. The pred_goal model's talent features are spending tree capacity on information that is non-informative for EA outcomes, which actively degrades discrimination by pulling predictions away from the context_xg prior.

**Current state:** Calibration is usable after Issue 14 fix (log loss 1.13–1.23× null for non-EA; 0.90× for EA). The model provides talent-adjusted probability estimates, but the adjustments are so small as to be practically identical to context_xg for non-EA, and actively worse for EA.

**Options under consideration:**

1. **Short-term: Accept current state for v1.0.0.** The probabilities are valid and the OOF gap is healthy. Pred_goal provides a documented talent delta (`pred_goal - context_xg`) even if it is small. The RAPM tier in the cascade is present and correctly implemented; the low gain reflects the difficulty of talent prediction rather than a coding error.

2. **Feature re-weighting:** Reduce or remove `goalie_gsax_per_shot_1g` and other very-short-horizon features. Force the model to rely more on career RAPM and longer EWMA signals. This would likely reduce overfitting to training-era patterns at the cost of possibly lower within-sample performance.

3. **EA-specific:** Remove pred_goal for empty_against entirely; use context_xg output directly. The negative lift of −0.034 is large enough to materially degrade EA probability quality.

4. **Architectural:** Investigate whether the negligible lift is an inherent property of the three-tier cascade design (context_xg already capturing most predictable variance) or a correctable feature-design problem.

### Planned Fix (2026-05-14)

**Decisions made:**
- **Option 2 (partial) — Strip `_1g` features:** Remove all 1-game horizon rolling stats from `compute_rolling_stats.py`. The `_1g` window (raw `_1g`, per-shot `_per_shot_1g`, and count `_shots_1g`) is too noisy to generalise across seasons. The `_10g` and `_ewma` windows remain.
- **RAPM subsetting:** Reduce `_RAPM_COEFF_COLS` in `process_data.py` from 6 dims (xg, corsi, goals × off/def) to 2 (xg_off, xg_def only). Drop corsi (shot-volume signal, not shot-quality signal — context_xg already captures shot difficulty) and goals (rare binary events → very noisy RAPM estimates). `_OFF_METRICS` changed from `["xg", "corsi", "goals"]` to `["xg"]`, reducing differentials from 6 to 2. Total RAPM features: 36 → 14.
- **Option 3 (EA pass-through) and max_depth=1:** Deferred. Address after re-tuning confirms whether lift improves with cleaner features.

**Code changes applied (`chickenstats_xg/v1/pred_goal/`):**

| File | Change |
|---|---|
| `compute_rolling_stats.py` | `_game_level_stats()`: removed `g1`/`s1` assignments; removed `_1g`, `_per_shot_1g`, `_shots_1g` from `with_columns()` and `select()`; updated `goalie_stat_cols` list; docstring updated (16 → 13 columns; "4 windows" → "3 windows") |
| `process_data.py` | `_RAPM_COEFF_COLS`: 6 entries → 2 (xg_off + xg_def only); `_OFF_METRICS`: `["xg", "corsi", "goals"]` → `["xg"]`. All entity join functions iterate over `_RAPM_DIMS` (derived), so they propagate automatically. |

**Operational sequence:**

```bash
# Step 15A — context_xg re-scored with Booster.predict fix (Issue 16):
# ✅ DONE (2026-05-15) — score.py: Booster.predict → XGBClassifier.predict_proba (respects best_iteration)
# All 5 states re-scored; RAPM PBP enriched; RAPM re-regressed (YOY r: 0.107 WARN → 0.317 PASS)

# Step 15B — Regenerate pred_goal train/hold_out parquets with corrected context_xg + new feature set:
# ✅ DONE (2026-05-15) — process_data.py re-run; _1g features stripped; RAPM subset to xg dims

# Step 15C — Re-tune (existing Optuna studies are stale — feature columns changed):
# ⏳ IN PROGRESS (2026-05-15)
uv run xg-experiments --model pred_goal --strength even_strength --version 1.0.0 --trials 500
# ... (repeat for all 5 strengths)

# Step 15D — Re-finalize:
uv run finalize-pred-xg --all --version 1.0.0 --no-log

# Step 15E — Re-diagnose:
uv run diagnose-pred-xg
```

**Expected improvements:**
- Distribution check: no trimodal cliff (`goalie_gsax_per_shot_1g` removed as dominant feature)
- Calibration: improved decile-8 monotonicity (residual bimodal structure eliminated)
- RAPM feature gain: higher (no longer overwhelmed by 1g noise features)
- Lift over context_xg: should improve from near-zero once RAPM has signal

### Status: IN PROGRESS (2026-05-15) — code changes applied (2026-05-14); `process_data.py` re-run 2026-05-15 with corrected context_xg (Issue 16 scoring fix) and new feature set; experiments re-tuning in progress

---

## Issue 16: context_xg/score.py Booster.predict() using all trees — bimodal scored predictions

### Problem

After finalizing context_xg models and validating them via `screen_trials()` (calibrated log loss ≤ 2× null, dist_ratio ≤ 3.0×), the per-season RAPM coefficients showed a catastrophic 10× scale discontinuity: boundary seasons (2010-12, 2024-25) had std ≈ 1.1 while middle seasons (2013-24) had std ≈ 0.07–0.11. The YOY stability check measured r=0.107 (WARN), driven by cross-era scale mismatch rather than genuine talent instability.

A diagnostic script confirmed the root cause: same params, same iteration count — fresh model retrain produced dist_ratio 2.37× (PASS); saved model scoring through `score.py` produced dist_ratio 8.98× (bimodal FAIL). The saved model itself was correct; the loading/scoring path was wrong.

### Root Cause

`context_xg/score.py` loaded the saved booster with `xgb.Booster()` and called `booster.predict(dmat)` without specifying `iteration_range`. With `EARLY_STOPPING_ROUNDS=50`, the booster contains `best_iteration + 50` trees (e.g., 126 trees for even_strength with `best_iteration=76`). Scoring all 126 trees — rather than the `best_iteration=76` used during training — accumulated 50 additional post-early-stopping tree outputs that inflated leaf weights and produced a bimodal raw probability distribution.

`base_xg/score.py` and `pred_goal/score.py` both used `XGBClassifier.predict_proba()`, which internally limits prediction to `best_iteration` trees. Only `context_xg/score.py` used raw `xgb.Booster`.

### Status: RESOLVED (2026-05-15)

**Fix applied to `context_xg/score.py`:**

Replaced the `xgb.Booster()` + DMatrix + `booster.predict(dmat)` block with `load_model_artifacts()` + `model.predict_proba(X, base_margin=logit_bm)[:, 1]` from `utils/artifacts.py`. `XGBClassifier` loaded via `load_model_artifacts()` automatically limits prediction to `best_iteration` trees.

**Impact:**

| State | Old dist_ratio (Booster.predict) | New dist_ratio (XGBClassifier) |
|---|---|---|
| even_strength | 8.98× | 1.65× ✅ |
| powerplay | 5.37× | 1.36× ✅ |
| shorthanded | 7.66× | 1.58× ✅ |
| empty_for | 7.06× | 1.24× ✅ |
| empty_against | 1.14× | 1.06× ✅ |

RAPM recomputed against correct context_xg: YOY stability improved from r=0.107 (WARN, results invalid) to r=0.317 (PASS). All per-season stds now consistent at 0.064–0.112.

**Key rule:** Always use `XGBClassifier.predict_proba()` when loading a frozen booster for inference. Never use `xgb.Booster.predict()` without `iteration_range` on an early-stopped model. See `utils/artifacts.py` `load_model_artifacts()` for the canonical loading pattern.

| File | Change |
|---|---|
| `context_xg/score.py` | Replaced `xgb.Booster()` + DMatrix + `booster.predict()` with `load_model_artifacts()` + `model.predict_proba(X, base_margin=logit_bm)[:, 1]`; removed `joblib` import |

---

## Issue 17: pred_goal/diagnose.py distribution check uses wrong metric

### Problem

After fixing the same bug in `context_xg/diagnose.py` (Issue 13), `pred_goal/diagnose.py` was never updated. It used `GOAL p90 / SHOT p90` as the distribution check ratio. This fired FAIL for ES/SH/EF and WARN for PP — all false positives. The correct metric is `SHOT p90 / base_rate`: fingerprinting pushes SHOT events up, not GOAL events.

### Status: RESOLVED (2026-05-16)

**Fix applied to `pred_goal/diagnose.py`:**

- Renamed threshold constants: `P90_RATIO_WARN=6.0` / `P90_RATIO_FAIL=10.0` → `SHOT_P90_BASE_RATIO_WARN=2.5` / `SHOT_P90_BASE_RATIO_FAIL=5.0`
- `check_distribution()` rewritten to compute `shot_p90 / base_rate` (was `goal_p90 / shot_p90`)
- All five strength states now correctly PASS: ES 0.96×, PP 1.01×, SH 1.11×, EF 1.02×, EA 1.11×

| File | Change |
|---|---|
| `pred_goal/diagnose.py` | `check_distribution()`: SHOT p90/base_rate metric; renamed threshold constants; updated print format |

---

## Issue 18: context_xg GOAL-side fingerprinting reasserting at the tail

### Problem

Confirmed via scored pbp_2024.parquet: 1,568 events have context_xg ≥ 0.90 in the 2024 season data. Of these, 99.6% are goals. 1,086 events have base_xg < 0.15 but context_xg ≥ 0.90 — all goals — with an average logit boost of +7.6 from contextual features alone (range +4.6 to +9.2). A shot from 16 feet at 30 degrees (base_xg ≈ 14.5%, logit ≈ −1.73) reaches context_xg = 99.7% from context features alone.

**SHOT p99 = 9.9%, but 44.8% of goals sit above that threshold.** The model has split the goal population into ~55% that look like ordinary shots (moderate predictions) and ~45% that land in probability territory shots almost never reach. This is the same fingerprinting problem from Issue 7 reasserting itself in the context layer.

The depth-2 constraint + `mds=1` slowed it but did not prevent it: with `lr ≈ 0.26` × 76 trees and 9 interaction constraint groups, each group can contribute smaller boosts that compound to +7.6 log-odds over the full forest. Rush flag shots that are goals tend to have `rush_attempt=1` AND high `play_speed` AND short `seconds_since_last` AND close `prior_event_distance` firing simultaneously across groups, stacking per-group boosts.

The existing distribution check (`SHOT p90 / base_rate ≈ 0.96×`) **passes because SHOT events are clean** — but that is exactly the fingerprint signature: SHOT events stay low, GOAL events develop a high-probability tail. The check is measuring the wrong population.

### Why the previously proposed tail calibration constraint (soft penalty) does not fix this

The proposed penalty (`max(0, mean_pred_top10 − actual_rate_top10)`) penalizes cases where predictions exceed actual goal rate in the top decile. For these events, **actual_rate ≈ 1.0 and predicted ≈ 0.99**, so `tail_overconf = max(0, 0.99 − 1.0) = 0`. The constraint sees no error — the model is correctly calibrated against a training set where those contextual features appear almost exclusively on goal events. Any post-hoc calibration penalty that uses actual labels cannot distinguish fingerprinting from genuine high-quality chances.

### Impact on RAPM

In the ridge regression, these 1,568 events each contribute ≈0.997 xGF to their stint. Since virtually all are goals, the RAPM essentially sees exact-goal targets for these events (same as a goals-based regression) rather than a smoothed xGF. The RAPM coefficients for players involved in these events are partly driven by "happened to be on ice for a tap-in" rather than sustained possession quality. Magnitude of impact unknown; likely small given 1,568 events in 1.1M total.

### Root cause options and tradeoffs

| Option | Mechanism | Tradeoff |
|---|---|---|
| **A. Hard cap on context_xg output** | Clip calibrated predictions at 0.60 or 0.70 in `score.py` | Destroys calibration for top-tier chances; changes RAPM targets; pred_goal base_margin wrong |
| **B. Lower lr ceiling in Optuna search** | Reduce `lr` upper bound from 0.30 → 0.10; force lower per-tree updates | Requires re-tuning all studies; may reduce PR-AUC |
| **C. Reduce best_iteration budget** | Early stopping on stricter convergence or lower n_estimators | May stop before model has fit contextual residual |
| **D. GOAL-side distribution check in screening** | Hard-reject trials where goal_p90 > threshold × base_rate (screens out forest-level fingerprinting at finalize time) | Need to calibrate threshold without rejecting legitimately strong models |
| **E. Accept as structural** | Contextual features genuinely correlate with goals; model is correctly expressing that correlation given labeled data | RAPM impact small; PR-AUC is real; problem is philosophical (true max shot probability ≈ 40–60%, not 99%) |

### Fingerprinting confirmed across all five strength states (2026-05-16)

Full analysis of pbp_2024.parquet by strength state confirmed the issue is universal, not isolated to even_strength:

| State (model) | cx≥0.90 events | All goals? | % goals above SHOT p99 | Caught by existing check? |
|---|---|---|---|---|
| 5v5 (even_strength) | 1,094 | 100% | 45.2% | No — SHOT p90=0.000, dist_ratio=0.00 |
| 5v4 (powerplay) | 270 | 100% | 27.7% | No — dist_ratio=2.93 (below 3.0 cap) |
| 4v5 (shorthanded) | 31 | 100% | 74.7% | No — SHOT p90=0.000, dist_ratio=0.00 |
| 5vE (empty_for) | 81 | 92.6% | 66.4% | No — max cx≈0.93, dist_ratio=0.00 |
| Ev5 (empty_against) | 49 | 100% | 41.5% | Yes — dist_ratio=3.47 > hard cap |

The existing SHOT-side check passes ES/PP/SH/EF because non-goal shots stay near zero. EA was already caught (dist_ratio=3.47 explaining its ❌ FAIL in diagnostics). The other four states had been slipping through.

### Status: ✅ RESOLVED (2026-05-16)

**Resolution — combined hyperparameter constraints + GOAL-side screening:**

**A. Tighter context_xg-specific budget (config.py):**
- Added `N_ESTIMATORS_CONTEXT_XG = 100` (was using shared `N_ESTIMATORS = 500`)
- Added `EARLY_STOPPING_ROUNDS_CONTEXT_XG = 20` (was using shared `EARLY_STOPPING_ROUNDS = 50`)
- With best_iter ≤ 80 and lr ≤ 0.10: max accumulated log-odds ≈ 0.10 × 80 × 0.385 = **3.08**
- This caps context_xg at ~0.58 from a 5% base_xg (fingerprinted events previously reached 0.997)
- Genuinely high-geometry chances (base_xg ≈ 30%) can still reach 0.90+ — the constraint only prevents low-geometry events from being fingerprinted

**B. Lowered lr ceiling (experiments.py `_params_context_xg`):**
- `suggest_float("learning_rate", 0.01, 0.10)` — was `0.01, 0.30`
- Necessary alongside n_estimators cap: early stopping compensates for lower lr by increasing best_iter, so capping n_estimators alone (without capping lr) leaves the compensation loop intact

**C. Raised lambda ceiling (experiments.py `_params_context_xg`):**
- `suggest_float("lambda", 10.0, 500.0)` — was `10.0, 200.0`
- EA top-50 had median lambda=153 and p90=191, pressing against the old 200 ceiling; SH p90=158 similarly. lambda directly shrinks leaf update magnitudes (leaf_weight ∝ 1/(H+lambda)), so raising the ceiling enables more aggressive L2 regularization.

**D. GOAL-side hard rejection in `screen_trials()` (utils/finalize_utils.py):**
- Added `goal_fraction_high = np.mean(prob_platt[y_np==1] > 0.85)`
- Hard reject if `goal_fraction_high > 0.05` (>5% of GOAL events post-Platt above 85%)
- Safety net: current fingerprinted ES has ~19.7% of goals above 90%, well above 5%. After the lr fix, this should never fire in normal operation.
- Printed as `❌ goal_fp` with ratio shown per trial in screening output.

**All five `{strength}-1.0.1-context` Optuna studies nuked and re-running with new constraints (2026-05-16).** Downstream pipeline (RAPM, pred_goal) will need to be re-run after new context_xg models are finalized.

---

## Proposed Refactors (Deferred — Post First Training Run)

Identified during the 2026-05-12 code quality audit. All are structural improvements; none block
training or correctness. Target: implement after first successful end-to-end training run.

### R1: `_objective_body()` is ~240 lines (`experiments.py`) — ✅ DONE (2026-05-12)

The Optuna objective function is a single monolithic function covering base_xg, context_xg, and
pred_goal with nested if/else branches controlling booster type, param space, and OHE application.

**Why deferred:** Code is functionally correct. Splitting requires carefully verifying all three
model-type branches still exercise the right code paths — best done with a working training run
as a regression baseline.

**Done when:** Three separate `_objective_base_xg()`, `_objective_context_xg()`,
`_objective_pred_goal()` functions, each ≤ 80 lines, delegating shared logic (CV loop, MLflow
logging, pruning) to a shared `_run_cv()` helper.

**Completed (2026-05-12):** `experiments.py` fully refactored: `_objective_body` split into
`_params_base_xg`, `_params_context_xg`, `_params_pred_goal` (per-model param builders) +
`_run_cv_folds` (shared CV helper) + thin `_objective_body` coordinator (~60 lines). Dead code
removed (`_apply_context_ohe`, `_build_model`, `cast` import). `sys.path.insert` moved to top
with both `parent` and `parent.parent` entries. `min_child_weight` now `log=True` for all three
models.

---

### R2: OOF loop copy-pasted across 3 finalize scripts — ✅ DONE (2026-05-15)

`base_xg/finalize.py`, `context_xg/finalize.py`, and `pred_goal/finalize.py` each implemented
an independent OOF `TimeSeriesSplit` loop. The structure is identical; the differences are the
calibrator type (isotonic vs. Platt) and whether `base_margin` is required.

**Completed (2026-05-15):** `compute_oof_predictions()` extracted to `utils/finalize_utils.py`.
All three finalize scripts now import and call the shared implementation. Per-tier calibrator
selection logic (LogisticRegression vs IsotonicCalibrator) correctly remains in each finalize script —
only the fold loop and prediction array assembly are shared.

---

### R3: `sys.path.insert()` boilerplate (all tier scripts) — ✅ DONE (2026-05-15)

Every script under `base_xg/`, `context_xg/`, `pred_goal/`, and `rapm/` adds:

```python
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).parent.parent))
```

to import `config.py` and `experiments.py` from the parent `1_0_0/` directory.

**Completed (2026-05-15):** `chickenstats_xg/` is an editable package installed via
`uv pip install -e .` (`pyproject.toml` entry points). All `sys.path.insert` lines removed.
Imports use `from chickenstats_xg.v1.config import ...` throughout.

---

### R4: `model_name` variable used where `strength` is meant (`experiments.py`) — ✅ DONE (2026-05-12)

All 13 occurrences of `model_name` (local variable holding the strength state string, e.g.
`"even_strength"`) renamed to `strength` throughout `experiments.py`. Affected functions:
`_apply_fixed_categoricals()`, `load_data()`, `tune_model()`, and `main()`. The `--model` CLI
argument was not renamed.

---

### R6: pred_goal pipeline renames `context_xg` → `base_xg` — carry column names through instead

`pred_goal/process_data.py` currently drops the Tier 1 `base_xg` column and renames
`context_xg` → `base_xg` so downstream code can uniformly look for `"base_xg"` as the
base_margin column. This caused confusion during development: `compute_rolling_stats.py`
uses `pl.col("base_xg")` but is actually computing GAx against the Tier 2 context_xg
prior — the column name doesn't reflect what it is.

**Fix:** Keep `context_xg` named `context_xg` throughout the pred_goal pipeline. Still
drop the Tier 1 `base_xg` (not needed in pred_goal parquets). Update all pred_goal-
specific code paths to look for `context_xg` instead of `base_xg`.

**Files to change (6):**

| File | Change |
|---|---|
| `pred_goal/process_data.py` | Remove rename; only drop Tier 1 `base_xg` |
| `pred_goal/compute_rolling_stats.py` | `pl.col("base_xg")` → `pl.col("context_xg")`; update docstrings |
| `pred_goal/finalize.py` | `_split_df()`: `"base_xg"` → `"context_xg"` for base_margin extraction |
| `pred_goal/score.py` | `_split_df()` + `_STALE_XG_COLS`: `"base_xg"` → `"context_xg"` |
| `pred_goal/diagnose.py` | `_NON_FEATURE_COLS`, `_predict()`, `check_lift()`, `check_holdout_metrics()`, `check_season_prauc()`, `run_strength()` |
| `experiments.py` | pred_goal base_margin branch (lines 415–419): `"base_xg"` → `"context_xg"` |

**After code changes:** Re-run `uv run process-pred-goal` — existing parquets have the
old renamed column and must be regenerated.

**Why deferred:** Implementing mid-tuning would require re-running `process-pred-goal`
and invalidating the in-progress Optuna studies. Zero correctness benefit — the rename
is purely cosmetic. Implement after the 500-trial pred_goal tuning run completes and
models are finalized.

---

### R5: Mixed `print()` / MLflow logging (all finalize + score scripts)

Progress messages go to `print()` in some scripts and to MLflow tags/metrics in others. There is
no single place to see a run's full progress — terminal for finalize, MLflow for experiments.

**Why deferred:** Style/observability improvement only. No correctness impact.

**Done when:** All `for strength in strengths_to_run:` loops in finalize scripts and the
`screen_trials()` candidate loop in `finalize_utils.py` are wrapped in `ChickenProgress` with
`progress.update(task, description=...)` replacing bare `print(f"Finalizing {strength}...")` calls.
Per-trial diagnostic output in `screen_trials()` uses `progress.console.print()` so it renders
cleanly alongside the progress bar. ✅ DONE (2026-05-16)

---

## Issue 19: Switch all three models to multi-objective Pareto front (PR-AUC + log loss)

### Problem

The single-objective composite score for context_xg — and raw PR-AUC for base_xg and pred_goal — optimises for discrimination at the expense of calibration. Models with high PR-AUC but overconfident predictions (e.g. a 10% geometric shot predicted at 40%) score well and get selected. The user explicitly prefers realistic probability estimates over maximising ranking.

### Decision

Switch all three models from single-objective to **multi-objective optimisation**: `(maximize cal_prauc, minimize cal_ll)`. Optuna uses the **NSGA-II sampler** and returns a Pareto front. At finalize time, `select_top_trials()` picks the knee of the front (or a user-chosen tradeoff point) rather than the top composite score.

**Why log loss and not ECE or Brier score:** Log loss directly penalises overconfident predictions — outputting 0.9 when truth is 0 costs much more than outputting 0.6. ECE and Brier score are useful diagnostics but log loss is the cleanest single calibration signal for the sampler.

**Why this subsumes the composite penalties:** In a two-objective Pareto setting, a trial with high PR-AUC but bimodal raw probabilities (structural flaw) will have poor cal_ll and sit on the dominated frontier — NSGA-II naturally deprioritises it without needing explicit penalty weights. ECE, structural flaw penalty, and distribution ratio penalty are all proxied by cal_ll.

**What stays:** The hard rejection gates in `screen_trials()` remain — `structural_flaw_penalty > struct_cap` (relative to null_ll) and `cal_ll > 2× null_ll`. `goal_fp` and `dist_ratio` are diagnostics only (not hard gates — see Issues 22 and 23).

**What gets removed:** The composite formula (`cal_prauc − penalties`) in `_objective_body()` for context_xg. The `_GOAL_FP_WEIGHT`/`_GOAL_FP_RAMP_START` soft penalty added for Issue 18 can be dropped from the objective once multi-objective is live (cal_ll captures it); the hard gate in `screen_trials()` covers the rest.

### Pipeline consequence

base_xg studies must be **nuked and re-run** under the new sampler. This invalidates the current context_xg studies (which were tuned against the old base_xg calibration). Full rebuild order:

1. Re-tune base_xg (multi-objective, NSGA-II) → finalize + score
2. context_xg process_data (rebuilds logit_base_xg from fresh base_xg)
3. Re-tune context_xg (multi-objective + Issue 18 hp constraints) → finalize + score
4. RAPM stints → regressions → pred_goal process_data
5. Re-tune pred_goal (multi-objective) → finalize + score

### Status

⏳ PENDING — implementation required before next tuning run.

**Files to change:**
- `chickenstats_xg/v1/experiments.py` — `_objective_body()`: return `(cal_prauc, cal_ll)` tuple for all three models; switch study to `directions=["maximize", "minimize"]` with NSGA-II sampler in `tune_model()`
- `chickenstats_xg/v1/utils/finalize_utils.py` — `select_top_trials()` / `screen_trials()`: select from Pareto front (e.g. max `cal_prauc - weight × cal_ll` at a chosen tradeoff weight, or knee detection)
- Remove `_GOAL_FP_WEIGHT` / `_GOAL_FP_RAMP_START` from objective once multi-objective is live (keep in finalize_utils for documentation but no longer used in experiments.py)

---

## Issue 20: eval_metric order causing best_iter=0 for context_xg and pred_goal

### Problem

context_xg and pred_goal finalize runs and Optuna trials used `eval_metric=["aucpr", "logloss"]`. XGBoost early stopping is controlled by the **last** metric in the list — so logloss was driving early stopping. 

For models that use `base_margin=logit_base_xg` (i.e., context_xg and pred_goal), the model starts from an already-calibrated prior. Logloss **increases** in the first ~17 rounds as tree 0 slightly overshoots the calibrated prior before recovering. With `early_stopping_rounds=20`, early stopping fired at `best_iteration=0` — logloss was lowest at round 0 before any trees were added, and the subsequent increase was never recovered within 20 rounds. Confirmed with per-round output: `[0] logloss:0.20791` (best), `[1] 0.20797`, ..., `[20] 0.20846`.

The result: context_xg learned nothing contextual — only `logit_base_xg` had nonzero feature importance (the base_margin bias term). All binary flags (`is_rebound`, `is_scramble`, `rush_attempt`, `prior_face`) and game-state modifiers had zero importance.

Note: base_xg does not use `base_margin` and is unaffected; logloss-based early stopping is correct there.

### Root Cause

`eval_metric=["aucpr", "logloss"]` → logloss is last → early stopping on logloss → logloss increases in early rounds for base_margin models → `best_iter=0` selected.

### Status: RESOLVED (2026-05-18)

**Fix:** Changed `eval_metric` to `["logloss", "aucpr"]` for context_xg and pred_goal so aucpr drives early stopping. aucpr is monotonically non-decreasing in the early rounds (starting from a calibrated prior doesn't cause the same initial rise), so early stopping fires at the correct best iteration. After fix: best_iter=96 for context_xg ES, `cal_PR=0.2014`, all binary flags appear in feature importance.

The original concern about aucpr-only stopping causing bimodal collapse predated `max_delta_step=1`. With that structural cap, goal_fp=0.023 (well below 0.05 threshold) and dist_ratio=2.95 (below 3.0 cap).

**Files changed:**
- `chickenstats_xg/v1/experiments.py` — `_objective_body()`: changed eval_metric to `["logloss", "aucpr"]` for context_xg and pred_goal branches (conditional on `data.model in ("context_xg", "pred_goal")`); base_xg branch keeps `["aucpr", "logloss"]`
- `chickenstats_xg/v1/context_xg/finalize.py` — `fixed_params`: changed `eval_metric` to `["logloss", "aucpr"]`
- `chickenstats_xg/v1/pred_goal/finalize.py` — `fixed_params`: changed `eval_metric` to `["logloss", "aucpr"]`

---

## Issue 21: context_xg interaction constraints blocking binary flags (zero feature importance)

### Problem

After the eval_metric fix (Issue 20) was applied and context_xg was re-run with working early stopping, feature importance inspection showed only the 6 continuous features from Group 8 (`logit_base_xg`, `play_speed`, `seconds_since_last`, `prior_event_angle`, `seconds_since_stoppage`, `distance_from_last`) with nonzero gain. All binary flags (`is_rebound`, `is_scramble`, `rush_attempt`, `prior_face`) and game-state modifiers (`is_home`, `strength_state`, `period`, `period_seconds`) had exactly zero gain.

### Root Cause

`interaction_constraints` isolated each binary flag into its own group paired only with `logit_base_xg` (e.g., Group 0: `[logit_base_xg, is_rebound]`). The continuous Group 8 had 10+ features and correspondingly higher per-split gain. XGBoost greedily allocates all 100 boosting rounds to the highest-gain group — Group 8. The flag groups had no path to generate comparable gain because each flag can only pair with `logit_base_xg` and binary splits on a low-base-rate feature (~16% for is_rebound, ~1.4% for rush_attempt) produce very low information gain compared to continuous feature splits.

At `max_depth=2`, this meant every tree went to Group 8 and no flag split was ever selected.

### Status: RESOLVED (2026-05-18)

**Fix:** Removed `interaction_constraints` entirely from `experiments.py` (`_params_context_xg`) and from `context_xg/finalize.py` (`fixed_params`). At `max_depth=2`, each tree path can access at most 2 features regardless. This is structurally equivalent to enforcing the fingerprint protection without the group competition problem.

Verified: goal_fp 0.055→0.054 (unchanged), dist_ratio 2.84→2.82 (unchanged). No increase in fingerprinting risk. After fix: `is_rebound` and `is_scramble` appear in feature importance as expected.

**Note on `CONTEXT_XG_INTERACTION_GROUPS` in `config.py`:** The constant is retained for reference only (labeled as such in comments); no longer passed to XGBoost.

**Files changed:**
- `chickenstats_xg/v1/experiments.py` — removed `_build_context_interaction_constraints()` function; removed `constraints = ...` and `"interaction_constraints": constraints` from `_params_context_xg()`; removed `CONTEXT_XG_INTERACTION_GROUPS` from imports; updated docstring
- `chickenstats_xg/v1/context_xg/finalize.py` — removed `"interaction_constraints"` from `fixed_params`; removed `_build_context_interaction_constraints` import; removed `CONTEXT_XG_INTERACTION_GROUPS` import

**Also fixed in this session:**
- `base_xg/finalize.py`: removed stale `load_model_artifacts` import (imported but never called)
- `base_xg/process_data.py`, `base_xg/score.py`, `pred_goal/score.py`, `rapm/prep_pbp.py`: all four had one too few `.parent` calls to reach the repo root `raw_data/pbp` directory (resolved to `chickenstats_xg/raw_data/pbp` instead of repo root `raw_data/pbp`)

---

## Issue 22: `goal_fp > 0.05` hard gate wrong for context_xg

### Problem

After removing the `dist_ratio` hard cap (prior session), all 15 top candidates still failed
`screen_trials()` for the even_strength context_xg model with `goal_fp=0.1304`.

The diagnostic run on trial 68 revealed:
- `structural_flaw_penalty = 0.0074` — smooth distribution, NOT bimodal
- `Platt cal_ll = 0.1868` — PASSES (< 0.4521 threshold)
- `goal_fp = 0.1304` — FAILS (> 0.05 gate)
- `raw predictions: max=0.99, p90=0.135, p99=0.532` — no compression to 1.0

The `goal_fp` gate rejected a well-calibrated model. The 13% figure is correct behavior:
~900 events (0.91% of 97,552 hold-out) have extreme contextual features (rebounds during
rushes, close range + high play_speed). These events score at ~85% rate in the hold-out.
Platt calibration correctly maps them to prob_platt > 0.85. 765/5,824 goals ≈ 13% of all
goals fall in this tier → goal_fp = 0.1304.

XGBoost trees predict from features only. Two events with identical input features receive
the same raw prediction regardless of their labels. If fingerprinting were occurring, the
raw distribution would be bimodal (shot cluster near base_rate, goal cluster near 1.0) and
`structural_flaw_penalty` would be large. At 0.0074, Isotonic barely outperforms Platt (0.4%
log-loss improvement) — the distribution is smooth and heavy-tailed.

The `goal_fp > 0.05` threshold was calibrated for base_xg, where no even-strength shot should
exceed 15–20% probability. context_xg with 21 features correctly identifies extreme-danger
situations with 85%+ actual scoring rates. Verified: even_strength scored parquet shows 6,274
events with context_xg ≥ 0.90, of which 6,039 are actual goals → 96.3% actual goal rate.

### Root Cause

`goal_fp > 0.05` was designed for base_xg (geometry only). In context_xg, legitimate
extreme-danger shots produce goal_fp ≈ 13% — this is correct calibration, not fingerprinting.

### Status: RESOLVED (2026-05-18)

**Fix:** Replaced `is_goal_fingerprinting` hard gate with `structural_flaw_penalty > struct_cap`.
`is_goal_fingerprinting` and `goal_fraction_high` remain computed and printed per trial for
diagnostics; they are no longer used for hard rejection.

Hard rejection logic after fix:
```python
is_bimodal = cal_ll > bimodal_threshold or structural_flaw_penalty > struct_cap
```

**Files changed:**
- `chickenstats_xg/v1/utils/finalize_utils.py` — `screen_trials()`: removed `is_goal_fingerprinting`
  from `is_bimodal` condition; `reject_reason = "goal_fp"` branch removed; `reject_reason = "struct"`
  added alongside existing `reject_reason = "cal_ll"`.

---

## Issue 23: Absolute `_STRUCT_PENALTY_CAP` too tight for small-dataset / high-base-rate states

### Problem

After the Issue 22 fix, all 15 empty_against candidates still failed `screen_trials()` with
`structural_flaw_penalty ≈ 0.025 > 0.020 = _STRUCT_PENALTY_CAP`.

Diagnostic analysis:
- EA hold-out: 1,002 events, base_rate=0.567, null_ll=0.684, healthy struct_penalty ≈ 0.025 = 3.7% of null_ll
- ES hold-out: 97,552 events, base_rate=0.060, null_ll=0.226, healthy struct_penalty ≈ 0.007 = 3.1% of null_ll
- Same relative magnitude — different absolute values

Root cause: Isotonic regression can use up to N steps (one per hold-out event) vs Platt's
2 parameters. With only 1,002 EA events, Isotonic has far more degrees of freedom relative
to the dataset size than with 97,552 ES events. Even for a smooth, healthy distribution, the
extra DOF inflate the absolute Isotonic–Platt log-loss gap. A fixed absolute cap cannot
account for this structural difference.

### Status: RESOLVED (2026-05-18)

**Fix:** Replaced `_STRUCT_PENALTY_CAP = 0.02` with `_STRUCT_PENALTY_REL_CAP = 0.088`
(8.8% of null_ll), computed as `struct_cap = _STRUCT_PENALTY_REL_CAP * null_ll`.

Calibration:
- ES: cap = 0.088 × 0.226 = 0.020 (unchanged from prior absolute value)
- EA: cap = 0.088 × 0.684 = 0.060 (now passes 0.025)
- Headroom: 8.8% / 3.1% = 2.8× over ES healthy ratio of 3.1%

EA finalized as trial 1372 with struct_cap=0.0602; struct_penalty ≈ 0.025 < 0.060 → passes.

**Files changed:**
- `chickenstats_xg/v1/utils/finalize_utils.py` — `_STRUCT_PENALTY_CAP` constant replaced with
  `_STRUCT_PENALTY_REL_CAP = 0.088`; `screen_trials()` computes `struct_cap = _STRUCT_PENALTY_REL_CAP * null_ll`
  after computing `null_ll`; diagnostic print updated to show computed `struct_cap`.

---

## Issue 24: `logit_base_xg` extreme values (±16) from `_BM_EPS = 1e-7`

### Problem

`context_xg/process_data.py` used `_BM_EPS = 1e-7` to clip `base_xg` before computing
`logit_base_xg`. `logit(1 − 1e-7) ≈ 16.1`. In empty_against hold-out, 10 events had
`logit_base_xg > 5` (all base_xg ≈ 1.0 — point-blank open-net shots). At base_margin=16,
XGBoost outputs raw ≈ 1.0 regardless of tree corrections (sigmoid(16) ≈ 1.0000000).

One of these 10 events was a shot (not a goal). Platt log-loss for a single event with
predicted probability ≈ 1.0 and true label = 0 is extremely large (−log(1−0.9999) ≈ 9.2).
This inflated `structural_flaw_penalty` artificially for the EA hold-out, compounding the
Issue 23 problem.

Additionally, inference-time scoring in `score.py` had no corresponding clip, creating
inconsistency between training-time and inference-time logit values.

### Status: RESOLVED (2026-05-18)

**Fix:** Added `_LOGIT_CAP = 4.0` in `process_data.py` and explicit `np.clip(..., -4.0, 4.0)`
in `score.py`. sigmoid(4) ≈ 0.982 — a near-certain shot remains near-certain but the gradient
is not numerically saturated. Even-strength is unaffected (its max base_xg ≈ 0.713).

Note: regenerating empty_against parquets with the logit cap confirmed `max(logit_base_xg)=4.0`.
However, the logit cap alone did NOT resolve the struct_penalty issue — the relative cap fix
(Issue 23) was also required. The cap prevents base_margin saturation; the Isotonic DOF
imbalance is independent of logit scale.

**Files changed:**
- `chickenstats_xg/v1/context_xg/process_data.py` — added `_LOGIT_CAP = 4.0` constant and
  `.clip(-_LOGIT_CAP, _LOGIT_CAP)` on the logit expression; empty_against parquets regenerated
  via `process-context-xg --strength empty_against`
- `chickenstats_xg/v1/context_xg/score.py` — added `logit_bm = np.clip(logit(...), -4.0, 4.0)`
  for inference-time consistency with training-time values

---

## Issue 25: pred_goal `max_delta_step` bimodal (100% shorthanded screening failures)

### Problem

After the context_xg bimodal fix (Issue 12 — `max_delta_step` fixed at 1 for context_xg), pred_goal was never given the same treatment. `_params_pred_goal = _params_base_xg` was a direct alias — pred_goal used the same Optuna param space as base_xg, where `max_delta_step` is tunable 1–10.

pred_goal uses `logit(context_xg)` as `base_margin` — structurally identical to how context_xg uses `logit_base_xg` as base_margin. The bimodal mechanism is the same: with base_margin providing a calibrated prior, tree-0 residuals are small; `max_delta_step ≥ 2` amplifies leaf weight updates disproportionately, producing a bimodal prediction distribution that fails the `structural_flaw_penalty > struct_cap` gate.

Symptom: 100% of shorthanded pred_goal screening candidates failed. For small-sample states (shorthanded: ~36K training shots), the bimodal cliff is most severe because the optimizer finds large-leaf solutions quickly. The four previously finalized states (ES/PP/EF/EA, 2026-05-16 run) happened to pass screening with mds≥2 trials — either by chance (larger datasets make the cliff less pronounced) or because those trials were selected before the structural_flaw_penalty gate replaced the cruder goal_fp gate.

### Root Cause

`_params_pred_goal = _params_base_xg` — pred_goal was never updated to reflect that it uses `base_margin`. The mds=1 fix was applied to context_xg's param space (Issue 12) but the pred_goal alias was never updated.

### Status: RESOLVED (2026-05-19)

**Fix applied in two places:**

1. **`chickenstats_xg/v1/utils/finalize_utils.py` — `screen_trials()`:** Added `screen_params["max_delta_step"] = 1` when `bm_train is not None` (after existing max_depth clamp).

2. **`chickenstats_xg/v1/pred_goal/finalize.py` — `_finalize_one()`:** Added `params["max_delta_step"] = 1` when `bm_train is not None` (after existing max_depth clamp).

3. **`chickenstats_xg/v1/experiments.py` — `_params_pred_goal()`:** New standalone function replacing the alias. `max_delta_step = 1` fixed (not trial-suggested). See Issue 27 for the full corrected param space.

**Note:** The clamps in `screen_trials` and `_finalize_one` remain permanently as defense-in-depth for any future runs against stale studies that may contain mds≥2 trials.

---

## Issue 26: pred_goal `lambda` floor — bimodal even with mds=1

### Problem

After applying the mds=1 clamp (Issue 25), pred_goal screening continued to reject candidates for shorthanded. `experiments.py` (line ~306, `_params_context_xg` docstring) explicitly documents: **"lambda < 10 produces bimodal output even with mds=1."**

The `_params_base_xg` param space had a `lambda` **ceiling** of 10.0. All pred_goal tuning trials had `lambda ≤ 10`. After the mds=1 clamp, `screen_trials` forced `lambda = max(lambda, 10.0)` — but that forced every candidate to exactly 10.0 (no diversity). All 15 screening candidates were identical in the lambda dimension, and many still produced bimodal or near-bimodal distributions.

### Root Cause

The base_xg lambda ceiling (10.0) equals the minimum lambda required for base_margin models to avoid bimodal output. Every pred_goal trial was tuned at or below this floor, and the finalize clamp collapsed all candidates to a single degenerate point in lambda space.

### Status: RESOLVED (2026-05-19)

**Fix applied:**

1. **`chickenstats_xg/v1/utils/finalize_utils.py` — `screen_trials()`:** Added `screen_params["lambda"] = max(screen_params.get("lambda", 10.0), 10.0)` when `bm_train is not None`.

2. **`chickenstats_xg/v1/pred_goal/finalize.py` — `_finalize_one()`:** Added `params["lambda"] = max(params.get("lambda", 10.0), 10.0)` when `bm_train is not None`.

3. **`chickenstats_xg/v1/experiments.py` — `_params_pred_goal()`:** `lambda` range `[10.0, 100.0]` in the new function. See Issue 27.

**Why clamps are insufficient for optimal results:** All existing pred_goal trials were tuned in the wrong regime (lambda 0.1–10, mds 1–10). After clamping, the finalized models use lambda=10 with no diversity explored above that floor. Studies must be nuked and re-run under the corrected param space to find genuinely optimal hyperparameters.

---

## Issue 27: pred_goal comprehensive param space gaps → new `_params_pred_goal` function

### Problem

After fixing mds (Issue 25) and lambda (Issue 26), a comprehensive audit revealed four additional param space gaps and one stale comment:

| Parameter | Old (`_params_base_xg`) | Correct for pred_goal | Reason |
|---|---|---|---|
| `alpha` | 1e-8–1.0 | 0.1–10.0 | All winning trials had alpha ≈ 0 (near the 1e-8 floor). pred_goal has many sparse RAPM career features — meaningful L1 sparsity requires a ≥ 0.1 floor |
| `learning_rate` | 1e-3–0.30 | 0.01–0.10 | Same Issue 18 rationale as context_xg: lr > 0.10 accumulates large log-odds over 500 trees with base_margin |
| `min_child_weight` | 20–200 | 50–300 | RAPM features are null for ~13% of shots; low mcw overfits to training-era player matchup patterns |
| `gamma` | 0.0–5.0 | 1.0–10.0 | Zero-floor allows unconstrained splits on sparse RAPM features |

**Impact on already-finalized models:**
- **lambda:** All winning trials tuned at lambda ≤ 10 (old ceiling = required floor); after clamp, all at exactly 10 — no diversity
- **alpha:** empty_against trial had alpha=0.001 (effectively zero L1); even_strength trial at alpha=0.0001
- **learning_rate:** empty_against winning trial had lr=0.2384 (well above new 0.10 ceiling)
- **min_child_weight:** empty_against winning trial had mcw=21 (below new 50 floor)

**Stale comment:** `_objective_body()` had a comment describing pred_goal as "logloss last → logloss drives early stopping" even though the code correctly branches pred_goal into the aucpr-last group (Issue 20 fix). Comment corrected.

### Status: RESOLVED (2026-05-19) — code changes applied; studies must be nuked and re-tuned

**Fix applied:**

`_params_pred_goal` replaced with a new standalone function (no longer an alias for `_params_base_xg`) with all corrected bounds. `_PARAM_BUILDERS` updated: `"pred_goal": _params_pred_goal`. See `MODEL.md` Section 9.3 for the full parameter table.

Also fixed: stale eval_metric comment in `_objective_body()`.

**Operational requirement:** All 5 pred_goal Optuna studies must be nuked before re-tuning — old trials have incompatible param ranges (mds=2–10, lambda < 10) that contaminate the Pareto front:

```
even_strength-1.0.0-pred_goal
powerplay-1.0.0-pred_goal
shorthanded-1.0.0-pred_goal
empty_for-1.0.0-pred_goal
empty_against-1.0.0-pred_goal
```

See the re-tune plan for the full nuke → tune → finalize sequence.

---

## Summary Table

| Issue | Description | Status |
|---|---|---|
| 1 | base_xg overfitting on GOAL events | ✅ RESOLVED (run 3; superseded by Issue 7) |
| 2 | base_xg uncalibrated (scale_pos_weight inflation) | ✅ RESOLVED |
| 3 | pred_goal context leak via interaction features | ✅ RESOLVED |
| 4 | RAPM prior-season join arithmetic (10000 vs 10001) | ✅ RESOLVED |
| 5 | empty_against Platt calibration ceiling | ✅ RESOLVED |
| 6 | Performance tag gate broken by label smoothing inflation | ✅ RESOLVED |
| 7 | GOAL event fingerprinting → three-tier cascade architecture | ✅ RESOLVED |
| 8 | context_xg base_margin saturation → bimodal predictions, calibration FAIL, negative lift | ✅ RESOLVED |
| 9 | context_xg T2 feature design — 3 attempts: binary gblinear (bimodal cliff) → quality features gblinear (collinearity collapse) → gbtree depth-2 flag isolation | ✅ RESOLVED |
| 10 | Feature architecture migration — base_xg tier purity (14→8 features) + empty_against OOF gap fix (score_diff moved to context_xg under isolated constraint) | ✅ RESOLVED (base_xg v1.0.0 finalized 2026-05-13) |
| 11 | context_xg bimodal calibration failure — more trials worsened ES/EF; confirmed structural root cause; logit_base_xg added as base_margin in addition to feature role | ✅ RESOLVED (2026-05-13) — requires study nuke + retune |
| 12 | Post-base_margin diagnostics — middle-band overestimation (ES/PP/EF/EA); SH catastrophic regression (bimodal at 0.78–0.82). Root cause: bimodal (mds≥2) trials dominate top-15 in flat CV landscape; mds=1 candidates at rank 16+. Fix: `--top-n 150` | ✅ RESOLVED (2026-05-14) — ES/PP/SH/EF PASS; EA WARN (structural mid-range, acceptable) |
| 13 | diagnose.py check logic errors — distribution check inverted (GOAL/SHOT ratio flagged good discrimination as FAIL); high-confidence check used fixed thresholds regardless of base rate (EA FAIL despite correct calibration) | ✅ RESOLVED (2026-05-14) — SHOT p90/base_rate check; base-rate-scaled high-conf thresholds |
| 14 | pred_goal OOF calibration temporal drift — OOF-only Platt calibrator learned training-era talent matchup statistics (goalie_gsax_per_shot_1g as dominant feature) that don't hold in hold-out; ES log loss 10.9× null, PP/SH/EF 3–5× null | ✅ RESOLVED (2026-05-14) — pooled OOF + hold-out calibration; log loss improved to 1.13–1.23× null; residual decile-8 miscalibration remains (FAIL, max err 0.13–0.19) |
| 15 | pred_goal negligible lift (+0.0001–+0.0009 PR AUC over context_xg for non-EA) and EA negative lift (−0.034 PR AUC); RAPM features < 1% gain; model dominated by noisy short-horizon goalie rolling features | ✅ RESOLVED (2026-05-16) — _1g features stripped; RAPM reduced to xg dims only; re-tuned 500 trials × 5 states; lift now real for non-EA (+0.0020 ES, +0.0014 PP, +0.0140 SH, +0.0042 EF); calibration resolved; EA structural negative lift −0.0078 remains (architectural ceiling, no goalie) |
| 16 | context_xg/score.py `Booster.predict()` using all trees — bimodal scored predictions from saved model (dist_ratio 8.98× vs 2.37× from fresh retrain); caused 10× RAPM coefficient scale discontinuity | ✅ RESOLVED (2026-05-15) — replaced `xgb.Booster.predict()` with `load_model_artifacts()` + `XGBClassifier.predict_proba()` which respects `best_iteration`; all states cleaned (dist_ratio 1.24–1.65×); RAPM YOY r improved from 0.107 WARN to 0.317 PASS |
| 17 | pred_goal/diagnose.py distribution check uses wrong metric — `GOAL p90 / SHOT p90` fires FAIL on well-discriminating models; same bug was fixed in context_xg/diagnose.py in Issue 13; pred_goal diagnostic was never updated | ✅ RESOLVED (2026-05-16) — SHOT p90/base_rate with WARN=2.5×/FAIL=5.0×; all five states correctly PASS (ES 0.96×, PP 1.01×, SH 1.11×, EF 1.02×, EA 1.11×) |
| 18 | context_xg GOAL-side fingerprinting at tail — all 5 strength states affected; 1,086+ events with base_xg < 15% boosted to 99%+ by context alone; avg logit boost +7.6; SHOT distribution clean so existing check passes; depth-2 + mds=1 insufficient to prevent compounding across 76 trees × 9 constraint groups | ✅ RESOLVED (2026-05-16) — lr ceiling 0.30→0.10; N_ESTIMATORS_CONTEXT_XG=100; EARLY_STOPPING_ROUNDS_CONTEXT_XG=20; lambda ceiling 200→500; GOAL-side hard rejection (>5% goals > 0.85 post-Platt) in screen_trials(); all 5 context_xg studies re-running |
| 19 | Single-objective optimisation favours discrimination over calibration — overconfident probabilities; prefer realistic percentages over higher PR-AUC | ⏳ PENDING — switch all 3 models to multi-objective (PR-AUC + log loss) with NSGA-II; full pipeline rebuild from base_xg required |
| 20 | eval_metric order caused best_iter=0 for context_xg and pred_goal — logloss increases in early rounds with calibrated base_margin prior; early stopping always selected 0 trees | ✅ RESOLVED (2026-05-18) — changed to `["logloss","aucpr"]` for context_xg and pred_goal (aucpr last → early stopping on aucpr); base_xg unchanged |
| 21 | interaction_constraints blocked binary flags (is_rebound, is_scramble, etc.) from context_xg — isolated low-gain flag groups could not compete with continuous Group 8; zero importance for all flags | ✅ RESOLVED (2026-05-18) — removed interaction_constraints entirely; max_depth=2 is sufficient structural protection; no fingerprinting increase observed |
| 22 | `goal_fp > 0.05` hard gate wrong for context_xg — 85%+ predictions legitimate for extreme-danger shots (struct_penalty=0.0074 confirms smooth distribution, not fingerprinting; 96.3% actual goal rate for context_xg ≥ 0.90 confirmed) | ✅ RESOLVED (2026-05-18) — replaced with `structural_flaw_penalty > struct_cap`; goal_fp still computed and printed for diagnostics |
| 23 | `_STRUCT_PENALTY_CAP = 0.02` absolute too tight — Isotonic DOF inflation on small holdouts (EA: 1,002 vs ES: 97,552 events); same 3.7% relative gap fails fixed absolute cap | ✅ RESOLVED (2026-05-18) — replaced with `_STRUCT_PENALTY_REL_CAP = 0.088` (8.8% of null_ll); ES cap=0.020 unchanged; EA cap=0.060 |
| 24 | `logit_base_xg` extreme values (±16) from `_BM_EPS = 1e-7` — base_margin saturation pins XGBoost outputs at 1.0; catastrophic Platt log-loss for 1 shot in EA hold-out with raw prob ≈ 1.0 | ✅ RESOLVED (2026-05-18) — added `_LOGIT_CAP = 4.0` clip in `process_data.py` and `score.py`; parquets regenerated |
| R5 | Mixed print/MLflow logging across finalize + score scripts | ✅ DONE (2026-05-16) — ChickenProgress wraps strength loops in all 3 finalize.py files; screen_trials() candidate loop uses progress bar + progress.console.print() for per-trial diagnostics |
| R6 | pred_goal pipeline renames `context_xg` → `base_xg` causing confusion (column name doesn't reflect content; `compute_rolling_stats.py` appears to use Tier 1 but actually uses Tier 2) | ✅ DONE (2026-05-16) — 6-file rename complete; `process-pred-goal` re-run required to regenerate parquets |
| 25 | pred_goal `max_delta_step` not fixed at 1 — `_params_pred_goal = _params_base_xg` alias meant pred_goal used tunable mds 1–10 despite using `base_margin` identically to context_xg; 100% shorthanded screening failures (structural_flaw_penalty >> struct_cap for all candidates) | ✅ RESOLVED (2026-05-19) — mds=1 cap added to `screen_trials` and `_finalize_one` when `bm_train is not None`; fixed in new `_params_pred_goal` (Issue 27); all 5 studies must be nuked and re-tuned |
| 26 | pred_goal `lambda` floor — lambda < 10 produces bimodal even with mds=1 (documented in `_params_context_xg` docstring); base_xg lambda ceiling was 10.0 = the required minimum floor; after mds clamp, all 15 screening candidates forced to exactly lambda=10 with zero lambda diversity | ✅ RESOLVED (2026-05-19) — lambda ≥ 10 clamp added to `screen_trials` and `_finalize_one`; lambda range [10, 100] in new `_params_pred_goal`; all 5 studies must be nuked and re-tuned |
| 27 | pred_goal comprehensive param space gaps — alpha floor too low (1e-8 → 0.1); lr ceiling too high (0.30 → 0.10); mcw floor too low (20 → 50); gamma floor too low (0.0 → 1.0); all winning trials showed alpha≈0, empty_against lr=0.24 and mcw=21 (above/below corrected bounds); stale eval_metric comment claiming pred_goal uses logloss-last early stopping (code correctly uses aucpr-last per Issue 20) | ✅ RESOLVED (2026-05-19) — new `_params_pred_goal` function with all corrected bounds (see MODEL.md Section 9.3); stale comment fixed; all 5 studies must be nuked and re-tuned |
| 28 | pred_goal `screen_trials` struct hard-gate false rejections for base_margin models — SH hold-out has only 187 positive examples; adversarial test (mds=5, lambda=1) produces identical struct ~11% of null_ll as healthy params (mds=1, lambda=17): gate has zero discrimination for pred_goal, only rejects all candidates. Root cause: Isotonic's extreme DOF relative to 187 positives creates a non-linear right-tail calibration effect that Platt cannot match — this is a data artifact, not bimodal signal. EF (230 positives) does NOT show this behaviour (struct=3.3%) because its prediction tail is less extreme | ✅ RESOLVED (2026-05-19) — `screen_trials`: `struct_hard_fail = (bm_train is None) and (structural_flaw_penalty > struct_cap)`; struct gate disabled when bm_train is not None (pred_goal); mds=1+lambda≥10 clamps prevent structural bimodality by construction; struct_penalty still used as soft composite-score penalty |

---

## Current Pipeline Re-run Order (v1.0.0)

```
# ✅ DONE (2026-05-13): base_xg tuned (500+ trials ES/PP/SH; 150+ EF; 590+ EA) and finalized.
# base_xg v1.0.0 diagnostics: ES/PP/SH/EF PASS, EA WARN (high-confidence check only).

# ✅ DONE (2026-05-13): Steps 1–3 orchestrated by base_xg/run_pipeline.py:
# base_xg/run_pipeline.py --version 1.0.0 --no-log
#   Step 1: base_xg/finalize.py --all
#   Step 2: base_xg/score.py --all              (base_xg → data/base_xg/scored/ + RAPM PBP)
#   Step 3: context_xg/process_data.py          (computes logit_base_xg; builds train/hold_out)

# ✅ DONE (2026-05-13): Step 4 — base_xg validated:
# base_xg/diagnose.py

# ✅ DONE (2026-05-13): First context_xg tuning round (~500 trials each, first run with base_margin).
# Results: ES/PP improved but still failing calibration; SH catastrophically regressed (bimodal
# at 0.78–0.82); EF/EA failing. See Issue 12 and context_xg/diagnostics.md (run 3 row).

# ✅ DONE (2026-05-14): Step 5 (round 2) — Issue 12 tuning:
# SH nuked and restarted; 750 trials for ES; 1000 trials for PP/SH/EF/EA.
experiments.py --model context_xg --strength even_strength  --version 1.0.0 --trials 750
experiments.py --model context_xg --strength powerplay      --version 1.0.0 --trials 1000
experiments.py --model context_xg --strength shorthanded    --version 1.0.0 --trials 1000   # NUKED — fresh start
experiments.py --model context_xg --strength empty_for      --version 1.0.0 --trials 1000
experiments.py --model context_xg --strength empty_against  --version 1.0.0 --trials 1000

# ✅ DONE (2026-05-14): Step 6 — Re-finalized with --top-n 150 (not 15; see Issue 12 resolution):
# context_xg/finalize.py --all --version 1.0.0 --no-log --top-n 150
# All 5 selected trials have max_delta_step=1. See context_xg/diagnostics.md for hyperparameter table.

# ✅ DONE (2026-05-14): Step 7 — Diagnostics validated:
# context_xg/diagnose.py   → ES/PP/SH/EF PASS; EA WARN (structural mid-range, acceptable).
# diagnose.py distribution and high-confidence checks were also fixed (Issue 13).

# ✅ DONE (2026-05-14): Steps 8–11 — context_xg scoring + RAPM + pred_goal data prep:
# context_xg/run_pipeline.py --version 1.0.0 --no-log
#   Step 8:  context_xg/score.py --all           (enriches RAPM PBP with context_xg column)
#   Step 9:  rapm/process_stints.py              (rebuilds stints using context_xg for h_xgf/a_xgf)
#   Step 10: rapm/regressions.py                 (RAPM now uses context_xg as xGF target)
#   Step 11: pred_goal/process_data.py           (assembles talent features; renames context_xg → base_xg)

# ✅ DONE (2026-05-14): Steps 12–13 — pred_goal tuning and finalization:
# experiments.py --model pred_goal --strength {all} --version 1.0.0 --trials 500
# pred_goal/finalize.py --all --version 1.0.0 --no-log
# Note: finalize.py was updated to use pooled OOF + hold-out calibration (Issue 14 fix) before
# the final run. --top-n 15 (default) used; no bimodal screening issue detected.

# ✅ DONE (2026-05-14): Step 14 — pred_goal diagnostic:
# pred_goal/diagnose.py
# Results: ES/PP/SH/EF FAIL (calibration — residual decile-8 pattern, max err 0.13–0.19; see Issue 14).
# EA FAIL (negative lift −0.034 PR AUC vs context_xg; see Issue 15).
# Lift negligible for non-EA states (+0.0001–+0.0009). Issues 14 and 15 documented.

# ✅ DONE (2026-05-14): Step 14 — pred_goal diagnostic + Issue 14/15 analysis.
# pred_goal/diagnose.py — calibration FAIL (residual decile-8); negligible lift.
# Issues 14 and 15 documented. Pooled calibration applied (Issue 14 resolved).

# ✅ DONE (2026-05-15): Issue 16 fix — context_xg re-scored (Booster.predict → XGBClassifier.predict_proba):
# score-context-xg --all (with RAPM prep): all 5 states cleaned (dist_ratio 1.06–1.65×)
# RAPM PBP enriched with corrected context_xg; stints rebuilt; regressions re-run
# RAPM diagnostics: YOY r=0.317 PASS; all 4 checks PASS

# ✅ DONE (2026-05-15): Step 15B — pred_goal data rebuilt (Issue 15 + Issue 16):
# process-pred-goal re-run: corrected context_xg as base_margin; _1g features stripped; RAPM → xg dims only

# ✅ DONE (2026-05-15): Additional context_xg tuning + re-finalize + re-diagnostic:
# ES extended 750→1500 trials; PP/SH/EF/EA extended 1000→1500 — all 5 studies at 1500/1500
# uv run finalize-context-xg --all --version 1.0.0 --no-log --top-n 150
# uv run diagnose-context-xg
# Results: ES/PP/SH ⚠️ WARN (high-conf); EF ❌ FAIL (OOF gap anomalous positive); EA ⚠️ WARN (cal)
# ES PR AUC 0.3213→0.3800; SH best OOF gap of all states (0.0032); EF best PR AUC (0.4318)

# ✅ DONE (2026-05-16): Step 15C — Re-tuned pred_goal (500 trials × 5 states, Issue 15+16 feature set):
uv run xg-experiments --model pred_goal --strength even_strength --version 1.0.0 --trials 500   # × 5 strengths

# ✅ DONE (2026-05-16): Step 16 — Re-finalized + diagnosed pred_goal:
# uv run finalize-pred-goal --all --version 1.0.0 --no-log
# uv run diagnose-pred-xg
# Results: ES ✅ PASS (corrected), PP ✅ PASS (corrected), SH ⚠️ WARN (cal small sample),
#          EF ❌ FAIL (OOF gap structural — early-season RAPM quality drag), EA ❌ FAIL (lift structural).
# Distribution FAIL/WARN are false positives — see Issue 17.
# Lift real for non-EA: +0.0020 ES, +0.0014 PP, +0.0140 SH, +0.0042 EF.

# ✅ DONE (2026-05-16): Issue 17 — Fixed pred_goal/diagnose.py distribution check (SHOT p90/base_rate)
# uv run diagnose-pred-xg  →  all five states correctly PASS distribution check

# ✅ DONE (2026-05-16): Issue 18 — context_xg hyperparameter constraints + GOAL-side screening
# Code changes applied:
#   config.py: N_ESTIMATORS_CONTEXT_XG=100, EARLY_STOPPING_ROUNDS_CONTEXT_XG=20
#   experiments.py: lr ceiling 0.30→0.10, lambda ceiling 200→500
#   utils/finalize_utils.py: GOAL-side hard rejection (>5% goals > 0.85 post-Platt)
# All 5 {strength}-1.0.1-context Optuna studies nuked; re-running with new constraints.

# ✅ DONE (2026-05-18): Issues 20+21 — eval_metric fix + interaction constraints removal
# Code changes applied:
#   experiments.py: eval_metric=["logloss","aucpr"] for context_xg and pred_goal branches
#   context_xg/finalize.py: same eval_metric fix; interaction_constraints removed from fixed_params
#   pred_goal/finalize.py: same eval_metric fix
#   base_xg/finalize.py: removed stale load_model_artifacts import
#   base_xg/process_data.py, base_xg/score.py, pred_goal/score.py, rapm/prep_pbp.py: filepath depth fix
# Context_xg studies (already re-running under Issue 18 constraints) will benefit from these fixes
# automatically at finalize time. No re-nuke needed — the eval_metric and constraints are in
# finalize.py fixed_params, not in the Optuna trial params.

# ✅ DONE (2026-05-18): Issues 22+23+24 — screen_trials() goal_fp gate + relative struct cap + logit clip
# Code changes applied:
#   utils/finalize_utils.py: goal_fp hard gate replaced by structural_flaw_penalty > struct_cap;
#     _STRUCT_PENALTY_CAP = 0.02 (absolute) replaced by _STRUCT_PENALTY_REL_CAP = 0.088 (fraction of null_ll)
#   context_xg/process_data.py: _LOGIT_CAP = 4.0 clip added to logit_base_xg computation
#   context_xg/score.py: matching np.clip(..., -4.0, 4.0) added at inference time
# context_xg data regenerated for empty_against: uv run process-context-xg --strength empty_against
# All 5 context_xg states finalized (run 1): uv run finalize-context-xg --all --version 1.0.1 --no-log --top-n 15
#   even_strength: confirmed well-calibrated (96.3% actual goal rate for context_xg ≥ 0.90)
#   empty_against: trial 1372 selected, struct_cap=0.0602, all 15 candidates passing

# ✅ DONE (2026-05-18): score-context-xg + RAPM rebuild + diagnose-rapm
# uv run score-context-xg --all (RAPM PBP enriched 19:46)
# uv run diagnose-context-xg (run 1: ES 0.3658, PP 0.4140, SH 0.3460, EF 0.3862, EA 0.7820; PASS/WARN/FAIL pattern confirmed)
# RAPM rebuilt (process_stints → regressions); diagnose-rapm: all 4 checks PASS (YOY r=0.333, max coeff 0.651)

# ✅ DONE (2026-05-18): context_xg re-finalized (refresh, run 2) + diagnose-context-xg (run 2)
# Same constraints as run 1; different Optuna trial selections:
#   PP: trial 571→1556 (lambda 44.8→13.1, lighter L2; mcw 106→141)
#   SH: trial 1017→1851 (alpha 4.72→5.01, spw 2.08→2.47; PR-AUC improved 0.3460→0.3476)
#   EF: trial 490→1772 (gamma 6.76→8.10, alpha 0.32→3.63)
#   ES trial 350 and EA trial 1803 unchanged
# Final metrics: ES 0.3658, PP 0.4139, SH 0.3476, EF 0.3859, EA 0.7820
# All PASS/FAIL/WARN outcomes unchanged from run 1
# NOTE: score-context-xg --all re-run required after refresh before process-pred-goal

# ✅ DONE (2026-05-18): RAPM re-finalized against context_xg run 2
# uv run score-context-xg --all (re-scored with run 2 models)
# uv run pipeline-context-xg --version 1.0.1 --no-log (process_stints + regressions rebuilt)
# uv run diagnose-rapm: all 4 checks PASS (YOY r=0.337, max coeff +0.6512 unchanged)
# Mid-era per-season shifts: 2014-15 std 0.068→0.087; 2016-17 std 0.061→0.078;
#   2017-18 std 0.083→0.064; 2019-20 bubble calmed std 0.099→0.077
# Leaderboard stable (ranks 6/7 swapped; 8466333 D 5,576 min enters bottom 10)

# ✅ DONE (2026-05-19): Issues 25+26+27 — pred_goal param space corrected; clamps applied; 4/5 states re-finalized
# Code changes applied:
#   experiments.py: new _params_pred_goal function (mds=1 fixed, lambda [10,100], lr [0.01,0.10],
#                   alpha [0.1,10], mcw [50,300], gamma [1.0,10.0], spw_cap=3.0); stale comment fixed
#   utils/finalize_utils.py: mds=1 and lambda≥10 clamps in screen_trials when bm_train is not None
#   pred_goal/finalize.py: same mds=1 and lambda≥10 clamps in _finalize_one when bm_train is not None
# Re-finalized 4/5 states (ES/PP/EF/EA) with clamps applied against old _params_base_xg studies:
#   ES trial 424: max_depth=5, lambda=10 (forced), alpha=0.0001, mcw=42, lr=0.0093, best_iter=86, PR AUC=0.3670 (+0.0013 lift)
#   PP trial 453: max_depth=6, lambda=10 (forced), alpha=0.0493, mcw=141, lr=0.0088, best_iter=80, PR AUC=0.4147 (+0.0008 lift)
#   EF trial 1834: max_depth=3, lambda=10 (forced), alpha=0.054, mcw=103, lr=0.051, best_iter=65, PR AUC=0.3891 (+0.0032 lift)
#   EA trial 1732: max_depth=4, lambda=10 (forced), alpha=0.001, mcw=21, lr=0.2384, best_iter=1, PR AUC=0.7681 (-0.0138 lift)
# SH not finalized — 100% bimodal from old studies even with clamps.
# NOTE: These 4 models have compromised params (lambda forced to 10, alpha≈0, EA lr/mcw out of range).
# All 5 studies must be nuked and re-tuned with corrected _params_pred_goal before pred_goal is final.

# ✅ DONE (2026-05-19): Issues 25+26+27 — nuked all 5 pred_goal 1.0.0 studies; 1.0.1 studies created with corrected param space
# ✅ DONE (2026-05-19): Issue 28 — struct hard-gate disabled for base_margin models
#   utils/finalize_utils.py: struct_hard_fail = (bm_train is None) and (structural_flaw_penalty > struct_cap)
#   SH adversarial test confirmed gate has zero discrimination for pred_goal; struct_penalty is still a soft penalty.

# ✅ DONE (2026-05-20): 2K tuning run complete + all 5 states finalized at version 1.0.1
# All 5 studies at version 1.0.1 with _params_pred_goal (mds=1 fixed, lambda [10,100], lr [0.01,0.10], alpha [0.1,10], mcw [50,300]):
#   even_strength-1.0.1-pred_goal:  2000 completed → finalized trial 1895 (lambda=29.8, alpha=1.71, best_iter=9)
#   powerplay-1.0.1-pred_goal:      2000 completed → finalized trial 1654 (lambda=69.8, alpha=0.17, best_iter=74)
#   shorthanded-1.0.1-pred_goal:    2000 completed → finalized trial 1549 (lambda=18.5, alpha=3.87, best_iter=94) ← first SH model
#   empty_for-1.0.1-pred_goal:      2000 completed → finalized trial 1547 (lambda=32.3, alpha=1.29, best_iter=419)
#   empty_against-1.0.1-pred_goal:  2000 completed → finalized trial 928  (lambda=13.2, alpha=1.03, best_iter=19)
# Lift: ES +0.0012 (14W/1L), PP +0.0007 (11W/4L), SH +0.0080 (9W/5L), EF +0.0076 (6W clear), EA −0.0142 (structural)
# All FAIL/WARN results are structural (inverse OOF gap from RAPM maturity; EA negative lift — no goalie). See DIAGNOSTICS.md.
uv run diagnose-pred-xg

# PENDING — Issue 19: implement multi-objective (PR-AUC + log loss, NSGA-II) before next tuning run.
# Required code changes:
#   experiments.py: _objective_body() → return (cal_prauc, cal_ll) tuple for all 3 models
#   experiments.py: tune_model() → directions=["maximize","minimize"], sampler=NSGAIISampler()
#   finalize_utils.py: select_top_trials() / screen_trials() → select from Pareto front
#   experiments.py: remove _GOAL_FP_WEIGHT / _GOAL_FP_RAMP_START soft penalty (cal_ll subsumes it)

# PENDING — full pipeline rebuild (base_xg → pred_goal) after Issue 19 code changes:
# Step 1: Nuke all base_xg, context_xg, and pred_goal studies
# Step 2: Re-tune base_xg (multi-objective, NSGA-II)
# uv run xg-experiments --model base_xg --strength {all} --version 1.0.0 --trials N
# Step 3: Finalize + score base_xg + rebuild context_xg process_data
# uv run pipeline-base-xg --version 1.0.0 --no-log
# uv run diagnose-base-xg
# Step 4: Re-tune context_xg (multi-objective + Issue 18 hp constraints)
# uv run xg-experiments --model context_xg --strength {all} --version 1.0.0 --trials N
# Step 5: Finalize + score context_xg + RAPM + pred_goal process_data
# uv run finalize-context-xg --all --version 1.0.0 --no-log
# uv run diagnose-context-xg
# uv run pipeline-context-xg --version 1.0.0 --no-log
# uv run diagnose-rapm
# Step 6: Re-tune pred_goal (multi-objective)
# uv run xg-experiments --model pred_goal --strength {all} --version 1.0.0 --trials 500
# Step 7: Finalize + diagnose pred_goal
# uv run finalize-pred-goal --all --version 1.0.0 --no-log
# uv run diagnose-pred-xg
```
