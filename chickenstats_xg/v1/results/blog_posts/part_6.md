# Part 6 Master Plan: The Scouting Layer (Talent Delta, SHAP Interpretability, and the Shooter Archetype Matrix)

## 1. Metadata, Aesthetics, and Technical Specs
* **Primary Publication:** `chickenandstats.com` (Ghost)
* **Code Documentation:** `chickenstats-blog` (nbdev/GitHub Pages)
* **Hero Image:** "The Archetype Matrix." A sleek, dark-mode 4-quadrant scatter plot glowing with neon data points across the full tier color spectrum. Recognizable player positions at the quadrant extremes — a finisher in Q1, a sniper in Q2, a system driver in Q3 — tease the scouting insights inside.
* **Architecture Diagram:** The "Scouting Engine Flow." Full cascade predictions → Player aggregation (ES filter, >500 min) → Talent Delta computation → Archetype classification.
* **Color Standard:** All tier colors active. Talent Delta uses a divergent Green (positive) / Red (negative) color map for the Archetype Matrix. SHAP Waterfall uses Cyan → Magenta → Gold progression.

## 2. Narrative Structure (Ghost Post)

### Section 1: Opening the Black Box (Introduction)

* **The Transition:** Part 5 proved the cascade is more accurate and better calibrated than the public benchmark across all five strength states and four holdout seasons. But accuracy is table stakes for a scouting tool. A coach or analyst won't trust a model that just outputs a number — they need to understand *why* the number is what it is.
* **The Goal of Part 6:** Translate the cascade's log-odds machinery into actionable hockey intelligence. Show who is finishing above system expectations. Show whose goal totals are a product of elite chance generation rather than individual skill. Show where every forward in the NHL sits on the two dimensions that matter most.

### Section 2: The Ultimate SHAP Waterfall (Anatomy of a Goal)

* **The Micro View:** Pick a memorable, highly complex 2024 goal — a broken play, a chaotic rebound, an elite cross-ice pass and instant finish. Walk the reader step-by-step through the single-shot SHAP waterfall.
* **The Three-Tier Breakdown:**
    * *Cyan layer (Tier 1):* How much danger did the location itself carry? A tap-in from the crease starts high; a one-timer from the perimeter starts low.
    * *Magenta layer (Tier 2):* How much did the pre-shot sequence add or subtract? A rush rebound from close range adds substantially; a settled zone-entry shot at 5v5 adds little.
    * *Gold layer (Tier 3):* How much did the specific shooter's talent and the specific goalie's talent adjust the final probability? A career 120% finishing-rate shooter against a 90% save-rate goalie moves the needle; two average players cancel out.
* **What to Show:** The SHAP waterfall chart from `shap.plots.waterfall`, with axis labels explicitly showing Tier 1 (base), Tier 2 (context adjustment), and Tier 3 (talent adjustment). Leaf values from each tier color-coded.

### Section 3: The Talent Delta

* **The Core Metric:** `talent_delta = pred_goal − context_xg`. For each shot, this is the probability-space contribution of the specific player combination net of the system chance.
* **Probability-Space Caveat:** This subtraction is meaningful only because both `pred_goal` and `context_xg` share the same probability space after sigmoid calibration. In log-odds space, the equivalent quantity is the raw Tier 3 XGBoost margin — interpretable as a multiplicative odds adjustment, but not directly comparable to probabilities. Aggregating `talent_delta` across shots gives the player's cumulative probability contribution above system, which is the correct basis for ranking.
* **True Finishers and System Beneficiaries:** Players with a sustained positive Talent Delta are finishing above what their system's chances would predict for an average shooter — these are the true elite finishers. Players with a sustained negative Talent Delta who still score 20+ goals are benefiting from elite chance creation by their linemates; their finishing is below expectation. Neither label is pejorative — it is a diagnostic.
* **Why This Matters for Team Building:** A team evaluating a free agent goal scorer needs to know which type they are acquiring. A True Finisher will maintain output on a weaker line. A System Beneficiary may not.

### Section 4: The Shooter Archetype Matrix

*The viral centerpiece of Part 6. Every NHL forward plotted on two dimensions derived directly from the cascade.*

* **The Filter:** Even-strength shots only, >500 ES minutes in the sample season. Power-play specialists inflate context_xg by operating in a strength state with structurally elevated chaos; restricting to even-strength isolates system quality from manpower advantages. The 500-minute minimum ensures a statistically meaningful sample.

* **The Axes:**
    * **X-Axis:** Even-strength `context_xg` generated per 60 minutes. How effective is this player (and their line) at creating chaotic, high-danger, sequence-elevated scoring chances?
    * **Y-Axis:** Per-shot Talent Delta, aggregated across the season. How much does this player finish above (or below) the system chances they receive?

* **The Four Quadrants:**
    * **Q1 — Complete Forwards (Top Right):** High context generation *and* positive Talent Delta. Elite at creating chances and at converting them. The franchise cornerstones.
    * **Q2 — Pure Finishers (Top Left):** Low context generation but high Talent Delta. Does not drive the play, but if the puck reaches them, they will convert at an elite rate. Classic "sniper" who needs a playmaking center.
    * **Q3 — Chance Creators (Bottom Right):** High context generation, neutral or negative Talent Delta. Creates massive chaos and high-danger sequences for their line but finishes below the system's expectation. The "volume driver" — great to have, but needs finishing talent around them.
    * **Q4 — Replacement Level (Bottom Left):** Does not drive play, does not finish above expectation.

* **Annotating Extremes:** Label the 10–15 most recognizable players at the quadrant extremes with their names. These are the shareable data points that make the chart a social media artifact.

### Section 5: The Launchpad (Setup for Part 7)

* **The Summary:** We went from raw NHL API coordinates → geometry → sequence context → individual talent → a two-dimensional map of every forward in the league. The cascade is not just accurate. It is interpretable down to a single shot.
* **The Bridge:** Every number in this matrix is queryable. Part 7 shows how — the live inference pipeline, the FastAPI endpoints, and the per-tier query structure that makes these results available programmatically to anyone building hockey analytics on top of the cascade.

## 3. Required Analyses, Charts, and Visuals

**1. The SHAP Waterfall (Single Goal)**
* *What it is:* `shap.plots.waterfall` for one memorable 2024 goal. Tier 1 / Tier 2 / Tier 3 contributions shown explicitly.
* *Color:* Cyan (base), Magenta (context uplift), Gold (talent uplift/discount). Red for negative contributions.

**2. True Finishers and System Beneficiaries Table**
* *What it is:* Two clean side-by-side tables.
    * Left: Top 5 players by positive Talent Delta (20+ goals filter) — the True Finishers.
    * Right: Top 5 players by negative Talent Delta (20+ goals filter) — System Beneficiaries.
* *Columns:* Player | Goals | `context_xg` Total | `pred_goal` Total | Talent Delta Total
* *Note:* 20+ goals filter focuses the analysis on players whose output is being evaluated at meaningful volume.

**3. The Shooter Archetype Matrix**
* *What it is:* 2D scatter of even-strength NHL forwards (>500 ES min). Crosshairs at league median for both axes.
* *Aesthetics:* Annotate the 10–15 extreme outliers by name. Quadrant labels in muted text. Four-color quadrant background shading (light, not dominant).
* *Why it works:* The chart is immediately readable — a fan recognizes Matthews in Q1, a defense-first grinder in Q4, and wonders where their favorite player falls.

**4. SHAP Beeswarm — Tier 3 pred_goal (Even Strength)**
* *What it is:* `shap.plots.beeswarm` for the pred_goal ES model, showing the talent features: `shooter_rapm_career_xg_off`, `shooter_gxg_off_60_rolling`, `goalie_gsax_60_rolling`, and goalie/shooter RAPM components. Use `logit(context_xg)` as base_margin when computing SHAP values.
* *Why:* Completes the three-tier beeswarm narrative that began in Part 2 (geometry) and continued in Part 3 (sequence/timing). The three beeswarms side-by-side form a single coherent story: geometry → sequence → talent. RAPM and rolling finishing rate are visible as the dominant Tier 3 drivers.
* *Data:* `shap.TreeExplainer` on the pred_goal ES model. Hold-out season shots.

**5. RAPM → pred_goal SHAP Dependence Plot**
* *What it is:* Scatter plot. x=`shooter_rapm_career_xg_off`, y=SHAP value for that feature in the pred_goal model. Color = shot quality tier (base_xg decile, 1–10).
* *Why:* Shows the near-linear relationship between player RAPM and the model's talent adjustment — the most important validation that Tier 3 is learning real skill signal and not noise or interaction artifacts. A curved or scattered relationship would indicate overfitting to a small set of players.
* *Data:* SHAP values from pred_goal model on the hold-out season. One dot per shot.

**6. Talent Delta Year-Over-Year Stability**
* *What it is:* Dot plot or small table. Rows = top 5 True Finishers (by 2024-25 talent_delta, ES, 20+ goals). Columns = `talent_delta` in 2022-23, 2023-24, and 2024-25. Dots sized by goals scored that season.
* *Why:* Makes the Archetype Matrix actionable for scouting. A player with persistent positive talent_delta across 3 seasons is demonstrably a True Finisher — the skill is stable and portable. A one-season outlier is not. This chart is the difference between a descriptive metric and a predictive one.
* *Data:* pred_goal scored parquets for ES, filtered to the top 5 identified True Finishers, across 3 seasons.

## 4. nbdev Companion Notebook Specs (`06_the_scouting_layer.ipynb`)

* **Cell 1 (Data & Model Load):** Load the fully trained cascade models and the full season Parquet dataset.
* **Cell 2 (Pipeline Inference):** Run batch inference to generate `base_xg`, `context_xg`, and `pred_goal` for every shot.
* **Cell 3 (Talent Delta Computation):**
    ```python
    df = df.with_columns([
        (pl.col("pred_goal") - pl.col("context_xg")).alias("talent_delta")
    ])
    ```
* **Cell 4 (Player Aggregation — ES Filter):** Apply the even-strength filter and 500-minute minimum before aggregating. Show the Polars `group_by` logic for per-player per-60 rates. The ES filter belongs in the aggregation step, not at inference time.
    ```python
    df_es = df.filter(pl.col("strength_state") == "even_strength")
    player_agg = (
        df_es
        .group_by("shooter_id", "shooter_name")
        .agg([
            pl.col("context_xg").sum().alias("context_xg_total"),
            pl.col("pred_goal").sum().alias("pred_goal_total"),
            pl.col("talent_delta").sum().alias("talent_delta_total"),
            pl.col("es_toi").sum().alias("toi_total"),
        ])
        .filter(pl.col("toi_total") >= 500)
        .with_columns([
            (pl.col("context_xg_total") / pl.col("toi_total") * 60).alias("context_xg_per60"),
        ])
    )
    ```
* **Cell 5 (SHAP Waterfall):** Generate the SHAP waterfall for the selected single goal. Use the `shap.TreeExplainer` on the Tier 3 model, showing the three-tier contribution breakdown explicitly.
* **Cell 6 (Archetype Matrix Plot):** Matplotlib scatter with crosshairs at league medians, quadrant labels, and annotated outlier names. Apply the four-quadrant background shading and ensure the plot is export-ready at high DPI.

---
### Summary of Result Reporting (Part 6)
**Part 6 (Scouting):** Translates cascade outputs into hockey intelligence. Introduces the Talent Delta metric with probability-space rigor, reframes finisher evaluation as "True Finishers vs. System Beneficiaries" (diagnostic, not pejorative), and delivers the Archetype Matrix as the series' viral centerpiece — every NHL forward mapped on even-strength context generation vs. finishing talent above expectation.
