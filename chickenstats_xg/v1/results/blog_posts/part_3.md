# Part 3 Master Plan: The Systems Layer (Tier 2 Context)

## 1. Metadata, Aesthetics, and Technical Specs
* **Primary Publication:** `chickenandstats.com` (Ghost)
* **Code Documentation:** `chickenstats-blog` (nbdev/GitHub Pages)
* **Hero Image:** "Implied Chaos." A dark-mode rink overlaid with a web of glowing Magenta/Purple vectors. Dense vectors near the net represent rapid pre-shot sequences (rebounds, rushes); sparse vectors at the perimeter represent settled offense — contrasting visually with the static blue heatmap of Part 2.
* **Architecture Diagram:** The dual-role handshake. Tier 1 Probability → `logit(p)` → two simultaneous arrows: (1) into XGBoost `base_margin` parameter, and (2) back into the feature matrix as `logit_base_xg` across 9 interaction constraint groups. Label clearly: "One number. Two roles."
* **Color Standard:** Magenta/Purple dominates this post.

## 2. Narrative Structure (Ghost Post)

### Section 1: Two Types of Context — Explicit Signals and Implied Chaos

* **The Reality Check:** The NHL play-by-play data does not track passes. What it *does* track explicitly are event type flags: whether a shot followed a rebound, whether it came off a rush, whether it was a scramble situation. These are labeled directly in the event stream — not inferred.
* **The Genuinely Inferred Quantities:** What we *do* have to compute from physics: how fast the puck was moving (`play_speed = distance_from_last / seconds_since_last`), how long since the last event, how far the puck traveled between events. If a shot happens 2.5 seconds after a faceoff win from 120 feet away, the puck moved at roughly 48 feet per second — we can see the fingerprint of that breakout pass in the coordinates and timestamps, even without the pass itself recorded.
* **The Goal of Tier 2:** Tier 2 does not re-predict whether the shot goes in. It predicts *how much the pre-shot sequence changes the geometric baseline* established in Part 2. The logit of the Tier 1 probability is passed forward; Tier 2 learns only the residual — the contextual adjustment on top of the spatial prior.

### Section 2: Feature Engineering — Four Families of Context

The 20 context features are grouped into four distinct families:

**1. Binary Event Flags (explicit from NHL event data):**
* `is_rebound` — shot follows a save from the same possession
* `is_scramble` — shot occurs during a loose-puck scramble
* `rush_attempt` — shot came off a rush
* `prior_face` — the prior event was a faceoff

**2. Game-State Modifiers:** Structural conditions that change the expected conversion rate independent of shot sequence.
* `is_home` — home team shooting advantage
* `position` — shooter position (F/D/G)
* `strength_state` — e.g. "5v5", "5v4" (within-strength variation beyond the model split)
* `score_diff` — score differential clipped to ±4 (trailing teams take higher-danger chances; leading teams suppress)

**3. Sequence Physics (inferred from coordinates and timestamps):**
* `play_speed` — Euclidean distance from last event divided by elapsed seconds; the proxy for pass velocity
* `seconds_since_last` — time elapsed since the prior event in the same period
* `distance_from_last` — Euclidean distance traveled from the prior event's coordinates
* `prior_event_angle` / `prior_event_distance` — geometry of the prior event (where was the play coming from?)
* `seconds_since_stoppage` — time since the last faceoff; measures how settled or chaotic the play is

**4. Prior Event Type Categoricals:**
* `prior_event_same` — the shooting team's prior event type (SHOT, MISS, BLOCK, GIVE, TAKE, HIT)
* `prior_event_opp` — the opposing team's prior event type (same categories; mutually exclusive with `prior_event_same`)

### Section 3: The Dual Role of `logit_base_xg` — The Architecture's Core

*This is the most important technical detail in Tier 2.*

`logit_base_xg` serves two simultaneous roles — and both are required:

**Role 1 — `base_margin` (gradient anchor):** The logit of the Tier 1 probability is passed to XGBoost's `base_margin` parameter. Every tree starts its residual computation from this anchor point rather than from zero. Without it, trees must predict the full probability range from scratch, causing high-flag shots to cluster at intermediate probabilities regardless of underlying shot quality — a bimodal distribution that no calibrator can fully resolve.

**Role 2 — Learnable feature in constraint groups:** `logit_base_xg` is *also* included in the feature matrix. In each of the 9 interaction constraint groups, it is paired with one other feature. This lets a depth-2 tree ask: "Given that this shot has *this* geometry quality, how much does `is_rebound` adjust the danger?" A rebound from the slot is far more dangerous than a rebound from behind the goal line — the feature role lets the model learn this quality-conditional adjustment. Without it, each group degenerates to a single binary split (equivalent to a linear additive coefficient), losing the quality-conditional structure entirely.

**The architecture diagram for this post must show both arrows** from `logit_base_xg`: one to the `base_margin` input and one into the feature matrix.

### Section 4: The Interaction Constraint Architecture — Anti-Fingerprinting

*This is the most important structural decision in Tier 2.*

**The Problem with Unconstrained Deep Trees:** Given 20 sequence features and max_depth=6, XGBoost will discover that "is_rebound=1 AND rush_attempt=1 AND distance < 10ft AND seconds_since_last < 1.5" strongly predicts goals. This is true — but it is also a fingerprint of the specific combination of events leading to GOAL labels in training data, not a generalizable rule. Rush rebounds from in close are rare enough that the model memorizes the specific sequence signatures rather than learning transferable physics.

**The Solution — 9 Interaction Constraint Groups:** Each of the 20 features is assigned to exactly one constraint group. Trees are only allowed to split on features within a single group per tree. The 9 groups are:

* Groups 0–3: `[logit_base_xg, <binary flag>]` — one group each for `is_rebound`, `is_scramble`, `rush_attempt`, `prior_face`
* Groups 4–7: `[logit_base_xg, <game-state modifier>]` — one group each for `is_home`, `position`, `strength_state`, `score_diff`
* Group 8: All continuous sequence + temporal features — `play_speed`, `seconds_since_last`, `distance_from_last`, `prior_event_angle`, `prior_event_distance`, `seconds_since_stoppage`, `prior_event_same`, `prior_event_opp`, plus `logit_base_xg`

**What this prevents:** A tree in Group 0 can only ask "Given shot quality X, how dangerous is this rebound?" It cannot combine `is_rebound` with `rush_attempt` on a single path — that cross-group combination requires a tree to access features from two groups simultaneously, which the constraints structurally forbid. The specific multi-flag combination (rebound + rush + close range) that fingerprints GOAL events requires depth ≥ 3 and cross-group access. Both are blocked by design.

**What `max_depth=2` adds:** With depth=2, each tree has exactly one split per group: `logit_base_xg > threshold` → `is_flag == 1` → leaf. This enforces the quality-conditional structure — every tree *must* condition on shot quality before applying the flag adjustment. The learned rules are fully human-interpretable.

**Concrete example (one tree from Group 0):**
```
Is logit_base_xg > −2.1  (base_xg ≈ 11%)?
  YES + is_rebound == 1  →  +0.8 log-odds
  YES + is_rebound == 0  →  +0.0 log-odds
  NO  + is_rebound == 1  →  +0.3 log-odds
  NO  + is_rebound == 0  →  +0.0 log-odds
```
The model learned: *a rebound matters more when the underlying shot quality was already high.* That is hockey physics, not memorization.

### Section 5: What the Model Learned (SHAP)

**Managing expectations about the SHAP output:** `logit_base_xg` will dominate the SHAP summary plot. This is correct and expected — it is both the `base_margin` anchor and a feature in all 9 groups, making it structurally the highest-weight variable. This is not a flaw; it confirms that Tier 2 is correctly conditioning on the geometry prior rather than overriding it.

**The residual story:** After `logit_base_xg`, the binary flags (`is_rebound`, `rush_attempt`) and sequence metrics (`play_speed`, `seconds_since_stoppage`) show the next-largest contributions — the contextual adjustments Tier 2 applies on top of the spatial prior.

**The Context Delta Distribution:** A histogram of `(context_xg − base_xg)` — the uplift Tier 2 applies over Tier 1 — should be heavily concentrated near zero (most settled 5v5 shots get minimal adjustment), with a tail of large positive values (rebounds, high-speed rushes) and a smaller negative tail (poor game states suppressing the baseline). This is surgical adjustment, not wholesale re-prediction.

## 3. Required Analyses, Charts, and Visuals

**1. The Architecture Diagram (Dual Role)**
* *What it is:* Flowchart showing `logit_base_xg` → two simultaneous paths: (1) `base_margin` input to XGBoost, (2) feature column in 9 constraint groups. Label both paths clearly.

**2. SHAP Summary Plot (Tier 2)**
* *What it is:* Standard XGBoost SHAP beeswarm showing all 20 features.
* *Expectation and framing:* `logit_base_xg` appears first — explain in the caption that this confirms the cascade architecture. Secondary contributors (`is_rebound`, `rush_attempt`, `play_speed`) are the contextual signal.

**3. SHAP Dependence Plot: `play_speed` vs. Contextual SHAP Value**
* *What it is:* Scatter of `play_speed` on X-axis, its SHAP value on Y-axis.
* *Why:* Proves that faster puck movement → larger contextual boost, capturing the inferred chaos proxy.

**4. Single Tree Visualization (Group 0 — `is_rebound`)**
* *What it is:* One depth-2 decision tree from the Group 0 ensemble, plotted with `xgboost.plot_tree` or `dtreeviz`.
* *Why:* Makes the constraint architecture concrete. Shows exactly the quality-conditional rebound rule as a human-readable decision path.

**5. Context Delta Distribution**
* *What it is:* Histogram of `(context_xg − base_xg)` for all training shots.
* *Why:* Proves Tier 2 makes surgical adjustments. Most shots near zero; high-context chances in the positive tail.

**6. The Bimodal Cliff: Before vs. After (Dual Histogram) — Priority Visual**
* *What it is:* Two side-by-side panels. Each panel has two overlapping kernel density curves: SHOT predictions (blue) and GOAL predictions (red). Left panel = a bimodal trial (`max_delta_step=3`, SHOT p90=0.71 — all non-goal shots pushed to high probability). Right panel = the production model (`max_delta_step=1`, SHOT p90=0.08 — clean separation).
* *Why:* This is the most important original visualization in the entire series. No other public hockey xG post has shown the bimodal cliff failure mode. It makes the `mds=1` constraint feel inevitable rather than arbitrary, and tells the model selection story without requiring the reader to understand log-odds accumulation.
* *Data:* Rerun inference on a bimodal trial (pull stored params from Optuna DB for any `mds≥2` ES trial, retrain on training data, predict hold-out) vs. production scored parquet. Alternatively: use scored parquets from the 2026-05-14 top-n=15 bimodal run (changelog entry) if still on disk.
* *Color:* SHOT = Steel Blue, GOAL = Crimson. Left panel title: "mds=3 (Bimodal — Calibration FAIL)". Right panel title: "mds=1 (Production — Calibration PASS)". Vertical line at base rate.

**7. Optuna Landscape Scatter: The Model Selection Trap**
* *What it is:* Scatter plot where each point = one Optuna trial. x-axis = CV PR-AUC (what Optuna optimizes), y-axis = Platt-calibrated hold-out log loss (what actually matters for deployment), color = `max_delta_step` value (mds=1 green, mds=2 orange, mds≥3 red).
* *Why:* Shows two clusters — the red/orange cluster scores marginally higher on CV PR-AUC but dramatically worse on calibrated log loss. The green mds=1 cluster is just below in CV PR-AUC but far better on calibration. This is the proof that CV PR-AUC is an insufficient selection criterion for this model class, and why the `--top-n 150` screening window was required.
* *Data:* Optuna DB trial values (CV PR-AUC) + trial params (mds) for the `even_strength-1.0.0-context` study. Calibrated log loss = recomputed via `screen_trials()` for each stored trial, or estimated from finalize run logs.
* *Simpler fallback:* Box plot of CV PR-AUC grouped by mds value (1 vs 2 vs 3+). Less information but achievable without re-running inference.

**8. SHAP Beeswarm — Tier 2 Even Strength (upgrade from summary)**
* *What it is:* `shap.plots.beeswarm` for the context_xg ES model showing all 20 features. Place directly alongside the Tier 1 beeswarm from Part 2.
* *Why:* The contrast with Part 2's beeswarm is stark: geometry features (event_distance, high_danger) shrink to near-zero gain; timing/sequence features (seconds_since_stoppage 53.7%, prior_event_angle 16.7%) dominate. This is the visual proof that each tier learns genuinely different signal.
* *Data:* `shap.TreeExplainer` on the context_xg ES model with `base_margin=logit(base_xg)`.

**9. Context Delta Violin by Event Type**
* *What it is:* Violin plot. x-axis = event type (Normal Shot / Rebound / Rush / Rush Rebound). y-axis = `context_xg − base_xg`. Median line shown.
* *Why:* Supplements the full-distribution histogram (#5) with a breakdown by the most important contextual categories. Rush Rebound gets the largest positive adjustment; settled zone-entry shots cluster near zero. Proves the model is capturing real sequence information, not noise.
* *Data:* ES scored context_xg parquet, grouped by `is_rebound` + `rush_attempt` flags.

## 4. nbdev Companion Notebook Specs (`03_the_systems_layer.ipynb`)

* **Cell 1 (Explicit Flags):** Show how `is_rebound`, `rush_attempt`, `is_scramble` are pulled directly from the NHL event data via `chickenstats` — not computed from coordinates. This distinction matters.
* **Cell 2 (Sequence Physics):** Show the Polars code to compute `play_speed`, `seconds_since_last`, `distance_from_last` — the genuinely inferred quantities.
    ```python
    df = df.with_columns([
        (pl.col("distance_from_last") / pl.col("seconds_since_last").clip(lower_bound=0.1)).alias("play_speed")
    ])
    ```
* **Cell 3 (The Logit Bridge + DMatrix):**
    ```python
    import scipy.special as sc
    log_odds = sc.logit(df["base_xg"].to_numpy())
    dtrain = xgb.DMatrix(X_train, label=y_train, base_margin=log_odds_train, enable_categorical=True)
    ```
* **Cell 4 (Interaction Constraints):** Show the Python list defining the 9 constraint groups, matching `_build_context_interaction_constraints()` from `config.py`. Column indices must match the feature matrix column order exactly.
* **Cell 5 (Model Training):** Hyperparameter dictionary, explicitly highlighting `"max_depth": 2` and `"interaction_constraints": constraint_groups`.
* **Cell 6 (SHAP & Context Delta):** Extract Tier 2 SHAP values, generate the beeswarm summary, and compute the `context_xg − base_xg` distribution histogram.
* **Cell 7 (Tree Visualization):** `xgboost.plot_tree(model, num_trees=0, rankdir='LR')` for one depth-2 stump from Group 0.

---
### Summary of Result Reporting (Part 3)
**Part 3 (Methodology/Systems):** Establishes the interaction constraint architecture. Proves Tier 2 conditions on geometry rather than overriding it. The SHAP summary, context delta distribution, and single-tree visualization demonstrate surgical contextual adjustment. Full metric results are reserved for Part 5.