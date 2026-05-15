# Part 4 Master Plan: The Talent Layer (Tier 3 Architecture)

## 1. Metadata, Aesthetics, and Technical Specs
* **Primary Publication:** `chickenandstats.com` (Ghost)
* **Code Documentation:** `chickenstats-blog` (nbdev/GitHub Pages)
* **Hero Image:** "The Gold Standard." A cinematic, dark-mode action shot of a player taking a one-timer. The player and the puck are glowing in a brilliant Gold/Orange, cutting through a faint haze of Cyan (Geometry) and Magenta (Context), symbolizing the isolation of pure talent.
* **Architecture Diagram:** The final Log-Odds Handshake: Tier 2 Probability → `logit(p)` → Tier 3 `base_margin` → Final `pred_goal` → `sigmoid(x)`.
* **Color Standard:** Gold/Orange dominates this post.

## 2. Narrative Structure (Ghost Post)

### Section 1: The Final Piece of the Puzzle (Introduction)
* **The Recap:** We mapped the ice (Tier 1) and we mapped the chaos (Tier 2). But a 20% system chance for a fourth-liner is not the same as a 20% system chance for Auston Matthews — and a 20% system chance against Igor Shesterkin is not the same as one against an AHL call-up.
* **The Concrete Stakes:** A `context_xg` of 20% against Shesterkin historically resolves to roughly 17% `pred_goal`; the same chance against a replacement-level goaltender becomes roughly 23%. That 6-point spread is entirely the talent signal doing its job. Without Tier 3, both shots wear the same number.
* **The Goal of Tier 3:** To inject historical, rolling player talent (shooters and goalies) into the final prediction, completely isolating *execution* from the *environment*. The environment is already captured in `context_xg`. Tier 3 only needs to ask: given this system chance, what do we know about who is pulling the trigger and who is standing between the puck and the net?

### Section 2: Feature Engineering — True Talent Priors

* **Rolling RAPM:** Regularized Adjusted Plus-Minus, computed via ridge regression from season stint data, gives a regularized estimate of each player's contribution net of teammates and opponents. Three RAPM metrics (offensive, defensive, overall) are computed across five strength situations and all available seasons.
* **Rolling GxG (Goals vs. `context_xg`):** For each shooter, we maintain a rolling count of goals scored minus cumulative `context_xg` over their prior shots. This is the shooter's observed finishing talent above (or below) what the system earned them.
* **Rolling GSAx (Saves Above `context_xg`):** For each goalie, we maintain the rolling saves above (or below) what `context_xg` predicted they would allow. This is the goalie's shot-stopping talent above what the system expected.

* **The Rule of Zero Leakage:** A shot taken in October 2024 uses RAPM computed through the end of the 2023–24 season. No in-season data from the current season bleeds into the prior. GxG and GSAx priors are updated with `.shift(1)` so that each shot sees only the player's history *before* that shot — never including the shot itself.

* **Cold-Start Handling:** Debut players and first-year skaters do not yet have RAPM or rolling finishing history. Rather than imputing or guessing, Tier 3 treats null priors as league-average (zero log-odds adjustment). The model degrades gracefully to `context_xg` for unknown players — the geometric and contextual information still applies, but no talent claim is made in either direction. This is the correct epistemic position when there is no evidence.

### Section 3: The Final Handshake

* **The Math:** Tier 2's calibrated `context_xg` output is passed through the `logit` function, producing a log-odds value that anchors Tier 3's `base_margin` parameter — the same logit bridge used between Tier 1 and Tier 2. Tier 3's trees learn only the residual: how much does this player combination move the log-odds up or down from the contextual baseline?

* **No Interaction Constraints:** Unlike Tier 2, Tier 3 does not use interaction constraints. Shooter talent and goalie talent interact directly — an elite shooter against an elite goalie produces a meaningfully different outcome than the same shooter against an average goalie — and the model should be free to learn this cross-term.

* **The Sigmoid Reveal:** After Tier 3 adds its adjustment, the final log-odds value is squashed through the `sigmoid` (inverse logit) function, returning a bounded probability between 0 and 1. This is `pred_goal` — the final cascade output, in standard probability space, ready for aggregation and display.

### Section 4: What's in Part 5

We now have all three tiers computing. Architecture is complete. Part 5 moves from design to evidence — a full formal benchmark across all five strength states with the four-column cascade progression (`base_xg` → EH → `context_xg` → `pred_goal`) and every metric Evolving-Hockey has never published about themselves.

## 3. Required Analyses, Charts, and Visuals

**1. The Architecture Diagram**
* *What it is:* Tier 2 probability → `logit` → Tier 3 `base_margin` → final `pred_goal` → `sigmoid`. Label the log-odds space explicitly.

**2. Talent Feature SHAP Summary**
* *What it is:* SHAP beeswarm for Tier 3 showing the rolling talent features (shooter GxG, goalie GSAx, RAPM components).
* *Why:* Confirms that goalie GSAx and shooter GxG are the dominant contributors, with RAPM as secondary regularizers.

**3. RAPM and GSAx Distribution Histograms**
* *What it is:* Two side-by-side bell curves: one for `shooter_rapm_career_xg_off` and one for `goalie_gsax_60_rolling` across all players in the training dataset. Tails labeled with recognizable player names (best/worst finishers by RAPM; best/worst goalies by GSAx).
* *Why:* Establishes that the talent signal is real and discriminative before the reader sees the model use it. The distributions should be approximately normal, centered near 0, with meaningful spread. If they're degenerate or noise-dominated, the SHAP beeswarm in chart #2 would tell a very different story.
* *Data:* pred_goal train parquet, `shooter_rapm_career_xg_off` and `goalie_gsax_60_rolling` columns.

**4. Probability Shift Scatter (Two-Panel)**
* *What it is:* Two scatter panels using a sample of ~20K hold-out shots. Panel 1: x=`base_xg`, y=`context_xg`, colored by `is_rebound` flag. Panel 2: x=`context_xg`, y=`pred_goal`, colored by shooter RAPM quartile (Q1–Q4).
* *Why:* Shows the cascade in action. Panel 1: context lifts rebounds (red dots above diagonal) regardless of shooter. Panel 2: pred_goal lifts high-RAPM shooters (warm colors above diagonal) regardless of shot type. The two panels together make the full three-tier story visual rather than verbal.
* *Data:* pred_goal hold-out scored parquet, which contains all three cascade columns.

## 4. nbdev Companion Notebook Specs (`04_the_talent_layer.ipynb`)

* **Cell 1 (Rolling GxG / GSAx with Leakage Guard):** Show the Polars code to build the rolling finishing metric with `.shift(1)` applied before the join, making the no-future-data guarantee explicit.
    ```python
    df = df.with_columns([
        pl.col("shooter_goals_cumulative").shift(1).over("shooter_id").alias("prior_shooter_gxg"),
        pl.col("goalie_saves_cumulative").shift(1).over("goalie_id").alias("prior_goalie_gsax"),
    ])
    ```
* **Cell 2 (RAPM Null Handling):** Show the conditional fill logic for cold-start players — null RAPM → 0.0 (league-average adjustment).
    ```python
    df = df.with_columns([
        pl.col("shooter_rapm_offense").fill_null(0.0),
        pl.col("goalie_rapm_defense").fill_null(0.0),
    ])
    ```
* **Cell 3 (The Final Bridge — Tier 2 → Tier 3 DMatrix):**
    ```python
    import scipy.special as sc
    log_odds = sc.logit(df["context_xg"].to_numpy())
    dtrain = xgb.DMatrix(X_train, label=y_train, base_margin=log_odds_train, enable_categorical=True)
    ```
* **Cell 4 (Model Training):** Hyperparameter dictionary for Tier 3. Highlight that `interaction_constraints` is absent (intentional — talent features are allowed to interact freely).
* **Cell 5 (The Sigmoid Reveal):**
    ```python
    raw_margins = model.predict(dtest, output_margin=True)
    pred_goal = sc.expit(raw_margins)   # sigmoid: log-odds → probability
    ```

---
### Summary of Result Reporting (Part 4)
**Part 4 (Architecture):** Establishes the talent prior construction (rolling GxG, GSAx, RAPM), the cold-start handling policy, and the final logit handoff from Tier 2. Includes the Shesterkin example to make the 6-point spread concrete. Full metric results are in Part 5.
