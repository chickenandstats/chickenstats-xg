# Series Overview: Deconstructing Expected Goals
The **Chickenstats 3-Tier Cascade** represents a paradigm shift in hockey analytics. Instead of treating Expected Goals (xG) as a monolithic, "black-box" probability, we deconstruct every shot into three strictly isolated, additive components: **Pure Geometry**, **Sequence Context**, and **True Talent**. By isolating these variables, we solve the "Credit Assignment Problem"—finally answering whether a goal was the result of a lucky bounce, a brilliant team system, or the elite finishing skill of the shooter.

### Table of Contents
* **Part 1: The Credit Assignment Problem** (Introduction, Literature Review, and the Architecture of the Cascade)
* **Part 2: The Foundation** (Tier 1: `base_xg` and Geometric Priors)
* **Part 3: The Systems Layer** (Tier 2: `context_xg` — Interaction Constraints, Explicit Flags, and Sequence Physics)
* **Part 4: The Talent Layer** (Tier 3: `pred_goal` — RAPM, Rolling Finishing Talent, and the Final Cascade)
* **Part 5: The Proof** (Formal Results — Full Benchmark Suite vs. Evolving-Hockey across all five strength states)
* **Part 6: The Scouting Layer** (Talent Delta, SHAP Interpretability, and the Shooter Archetype Matrix)
* **Part 7: The Chickenstats API** (Infrastructure, Live Inference Pipeline, and Product Deployment)

---

# Detailed Master Plan: Part 1

## 1. Metadata, Aesthetics, and Technical Specs
* **Primary Publication:** `chickenandstats.com` (Ghost)
* **Code Documentation:** `chickenstats-blog` (nbdev/GitHub Pages)
* **Hero Image:** "Shattering the Monolith." A cinematic, dark-themed render of a hockey puck impacting a crystalline block, which shatters into three distinct glowing streams: Cyan (Geometry), Magenta (Context), and Gold (Talent).
* **Architecture Diagram:** A high-level technical flowchart showing the data lineage:
    * `chickenstats` Python Scraper $\rightarrow$ Raw NHL JSON $\rightarrow$ Polars Pre-processing $\rightarrow$ Parquet $\rightarrow$ 3-Tier XGBoost Cascade $\rightarrow$ FastAPI Endpoint.
* **Color Standard:**
    * **Tier 1 (`base_xg`):** Cyan (Representing the cold, hard logic of geometry).
    * **Tier 2 (`context_xg`):** Magenta (Representing the dynamic chaos of the game).
    * **Tier 3 (`pred_goal`):** Gold (Representing the "Gold Standard" of player talent).
    * **Public Benchmarks (EH):** Gray (Dashed/Muted).

## 2. Narrative Structure (Ghost Post)

### Section 1: Introduction — Standing on the Shoulders of Giants (The Lit Review)
* **The Data Origin:** Introduce the `chickenstats` Python package as the source of truth. Acknowledge that while you are scraping official NHL RTSS data, you are working with a known handicap: **The NHL does not track passes.**
* **The Literature Review:**
    * **The Pioneers:** Explicitly cite **Brian Macdonald** (the father of NHL xG/RAPM concepts) and the foundational public work of **Evolving-Hockey (Josh and Luke Younggren)** and **MoneyPuck**.
    * **The Accomplishment:** These models revolutionized hockey by moving us past "Shots on Goal" into "Shot Quality." They proved that *where* you shoot from matters.
    * **The Identified Gap:** Explain that these models are "Monolithic." They throw location, game state, and player IDs into one giant XGBoost soup. 
* **The Credit Assignment Problem:** Define the core thesis. If a model says a shot is 15% xG, you don't know *why*. Is it 15% because it was from the slot? Or because it was a rush? Or because it was Auston Matthews? Monoliths conflate these, making it impossible for a coach to tell if their *system* is working or if they just have a *star player* masking their flaws.

### Section 2: Deconstructing the Monolith (The Architecture)
* **Philosophy of Isolation:** Explain why you used a cascade rather than a single model.
* **Tier 1: `base_xg` (Geometry):** The "Physics" layer. Uses only distance, angle, and shot type. It establishes the "Spatial Prior."
* **Tier 2: `context_xg` (Context/Systems):** The "Tactical" layer. Twenty features across four families: (1) explicit binary event flags from NHL data (`is_rebound`, `rush_attempt`, `is_scramble`, `prior_face`); (2) game-state modifiers (`is_home`, `position`, `strength_state`, `score_diff`); (3) sequence physics inferred from coordinates and timestamps (`play_speed`, `seconds_since_last`, `distance_from_last`, `prior_event_angle`, `prior_event_distance`, `seconds_since_stoppage`); (4) prior event categoricals (`prior_event_same`, `prior_event_opp`).
    * *Key Technical Note:* Nine interaction constraint groups (one per binary flag and game-state feature, one for all continuous physics) with `max_depth=2`. This is the "Anti-Fingerprinting" mechanism — it forces the model to ask "Given this shot quality, how much does this flag adjust the danger?" rather than memorizing specific multi-flag combinations.
* **Tier 3: `pred_goal` (Talent):** The "Execution" layer. It injects rolling Shooter and Goalie RAPM. This isolates the finishing skill from the chance itself.

### Section 3: The Video Room (Scouting Proof of Concept)
*This section provides the "Aha!" moment for the reader. You will use the `chickenstats` scraper to pull 4 distinct goals that prove the model's logic.*

1.  **Case 1: The "Grinder's Geometry" (High Base / Low Talent):** * *Video:* A fourth-liner taps in a rebound from the goal line. 
    * *Logic:* Base xG is massive ($0.45$). Context adds little. Talent actually *downgrades* the chance because the shooter is below average.
2.  **Case 2: The "Superstar's Snipe" (High Talent / Low Geometry):** * *Video:* A star snipes top-corner from the perimeter in a settled 5v5 state.
    * *Logic:* Base xG is tiny ($0.03$). Context is neutral. The *only* reason this is expected to go in is the Gold Talent layer ($+0.05$ delta).
3.  **Case 3: The "System's Strike" (High Context / Low Base):** * *Video:* A defenseman taps in a cross-ice royal road pass on a 2-on-1 rush.
    * *Logic:* The location is mediocre, but the `context_xg` skyrockets due to the "implied chaos" (rapid change in coordinates and speed of play).
4.  **Case 4: The Robbery vs. The Sieve (The Goalie Impact):**
    * *Video:* Side-by-side of an elite goalie stopping a 20% xG chance versus a poor goalie letting in an identical 20% chance.
    * *Logic:* Highlight how Tier 3 correctly identifies that the first was "saved" by talent, not "missed" by the shooter.

### Section 4: The Math — Infinite Ladders and Sigmoid Curves
* **The Probability Room:** Explain that percentages have a "ceiling" at $1.0$ ($100\%$) and a "floor" at $0$. You cannot just add $10\% + 10\%$ and get $20\%$ forever; eventually, you'd hit $110\%$.
* **The Infinite Ladder (Log-Odds):** This is the log-odds space ($-\infty, \infty$). There is no ceiling.
* **The Cascade Math:** 1.  **Logit Transform:** Convert Tier 1 probability into a "starting step" on the ladder.
    2.  **Additive Boosting:** Tier 2 and 3 trees add or subtract "steps" (weights) from that starting point.
    3.  **Sigmoid Link:** Squash the final position on the ladder back into a bounded $0-100\%$ probability for the human reader.

### Section 5: The Promise (Why the Metric Matters)

* **PR-AUC vs. ROC-AUC:** Goals are rare — roughly 6% of all shot events. ROC-AUC is easily "cheated" by predicting "No Goal" on every shot: you are correct 94% of the time and learn nothing. **Precision-Recall AUC** does not allow this escape. To score well on PR-AUC, a model must identify the actual goals precisely *and* find a high fraction of them — both matter simultaneously.
* **The Full Evaluation:** Discrimination (PR-AUC, ROC-AUC) is necessary but not sufficient. A model can rank shots correctly while being badly miscalibrated — predicting 30% on shots that go in only 12% of the time. Part 5 will show the complete four-column cascade progression (`base_xg` → EH → `context_xg` → `pred_goal`) across all five strength states, with every metric Evolving-Hockey has never published about themselves: Brier Score, Log-Loss, ECE, and calibration curves.
* **What to Expect:** The goal of the remaining posts is not just to show the final number. It is to make every step of the improvement legible — geometry, then context, then talent — so the reader can evaluate the architecture, not just the headline.

## 3. Required Analyses, Charts, and Visuals

**1. Five-Rink base_xg Heatmap (Opening Visual)**
* *What it is:* One hexbin or smoothed-contour rink per strength state, arranged side-by-side (5 panels). Each rink is colored by mean `base_xg` at each ice location.
* *Why here:* Part 1 is the intro — giving the reader one visual that immediately shows the geometry prior differs meaningfully across states anchors the whole series. ES has a tight high-danger cluster; PP is shifted toward the slot; SH concentrates on odd-man rush angles; EA is near-uniform across the offensive zone; EF has a distinct pattern driven by the pulled-goalie situation.
* *Data:* Scored base_xg parquets (one per state), binned by coords_x / coords_y into hexagons.
* *Color:* Dark-mode with Cyan → Gold gradient for low → high `base_xg`. Consistent scale across all 5 panels.

## 4. The Detailed Results Roadmap
This table maps each post to its primary analytical contribution.

| Post | Focus | Evidence Released |
| :--- | :--- | :--- |
| **Part 1** | Architecture Overview | Cascade design rationale; PR-AUC framing |
| **Part 2** | Geometric Prior | SHAP summary; 93% signal concentration in 3 features |
| **Part 3** | Context Architecture | 9 interaction constraint groups; context delta distribution |
| **Part 4** | Talent Architecture | RAPM cold-start handling; Shesterkin example; logit handoff |
| **Part 5** | Formal Results | Four-column benchmark: `base_xg` → EH → `context_xg` → `pred_goal` across 5 strength states |
| **Part 6** | Scouting Layer | Talent Delta; True Finishers table; Archetype Matrix (ES only) |
| **Part 7** | Product | API infrastructure; live inference pipeline |

## 5. nbdev Companion Notebook Specs (`01_deconstructing_xg.ipynb`)
* **Cell 1 (Scraper):** Demo code showing `chickenstats.fetch_game_videos(game_id)` to pull the videos used in the Ghost post.
* **Cell 2 (Filters):** The exact Polars queries used to find "The Grinder's Tap-in" vs. "The Superstar's Snipe" (Filtering by high/low Tier deltas).
* **Cell 3 (The Bridge):** A clean Python implementation of the Logit/Sigmoid bridge.
    ```python
    def prob_to_logit(p): return np.log(p / (1 - p))
    def logit_to_prob(l): return 1 / (1 + np.exp(-l))
    ```
* **Cell 4 (Evaluation):** Running the `sklearn.metrics.precision_recall_curve` on the holdout Parquet file.
* **Cell 5 (Viz):** Matplotlib/Seaborn code to generate the "Anatomy of a Goal" Waterfall chart.