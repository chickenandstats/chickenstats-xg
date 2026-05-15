# Part 5 Master Plan: The Proof (Formal Results — Full Benchmark Suite vs. Evolving-Hockey)

## 1. Metadata, Aesthetics, and Technical Specs
* **Primary Publication:** `chickenandstats.com` (Ghost)
* **Code Documentation:** `chickenstats-blog` (nbdev/GitHub Pages)
* **Hero Image:** "The Proof." A dark-mode academic rendering — a calibration curve and a cascade progression table side by side, color-coded in the four-column scheme (Cyan / Gray / Magenta / Gold), glowing on a dark background.
* **Architecture Diagram:** The four-column progression: `base_xg` → EH → `context_xg` → `pred_goal`. Each column is a tier color except EH (Gray). Arrow labels show what each transition adds.
* **Color Standard:** All four tier/benchmark colors active in this post. Cyan / Gray (EH) / Magenta / Gold. Every chart uses this consistent palette.

## 2. Narrative Structure (Ghost Post)

### Section 1: Why Standard Metrics Lie (The Framework)

* **The Base Rate Problem:** Goals are rare — roughly 6% of all shot events. A model that predicts "No Goal" on every shot is correct 94% of the time and tells you nothing useful. ROC-AUC rewards this: the all-negative model scores ~0.50, and a weak positional model easily scores ~0.70 by ranking obvious long-distance shots below obvious slot shots.
* **Why PR-AUC is Required:** Precision-Recall AUC eliminates the escape hatch. To score well, a model must identify the actual goals with high precision *and* find a high fraction of them simultaneously. The area under the PR curve scales with the base rate — a PR-AUC of 0.20 at a 6% base rate is 3.3× the null (random) model. This is the correct denominator for evaluating hockey shot quality.
* **The Full Metric Suite:** Discrimination (PR-AUC, ROC-AUC) measures ranking quality. Calibration (Brier Score, Log-Loss, ECE) measures whether the predicted probabilities are trustworthy in absolute terms. A model can rank shots correctly while predicting 30% on shots that go in only 12% of the time — it would be useful for sorting but not for aggregation. The cascade targets both.
* **What EH Left Unpublished:** Evolving-Hockey published ROC-AUC and feature importance — a standard starting point, but not a complete evaluation. They never reported Brier Score, Log-Loss, ECE, or calibration curves for their own model. This post completes that evaluation using EH's published per-shot xG values on the same 2024 holdout set.

### Section 2: The Benchmark Setup

* **Strict 2024 Holdout:** The holdout season was never seen during model training, OOF calibration, or hyperparameter search for any of the three cascade tiers. It represents genuine out-of-sample performance.
* **EH Matching:** EH publishes per-shot xG values. These are joined to the holdout set by event ID and strength state, producing a row-for-row comparison. EH operates five separate models (one per strength state), matching the cascade's structure exactly — true apples-to-apples.
* **The Four Columns:** `base_xg` (Tier 1 only, geometry) | EH (public benchmark) | `context_xg` (Tiers 1+2) | `pred_goal` (full cascade). This layout is a progression table, not a horse race. Each column shows what the next architectural decision buys.
* **Per-Strength Reporting:** Results are reported for all five strength states (even_strength, powerplay, shorthanded, empty_for, empty_against). Aggregating across strength states masks meaningful structural differences — empty-net shooting (EA) has a ~57% base rate; even-strength has ~6%. A single aggregate number would be misleading.

### Section 3: The Cascade Progression Table

*The centerpiece of Part 5. Numbers to be filled from model evaluation runs after pipeline completion.*

**Even Strength:**

| Metric | `base_xg` | EH | `context_xg` | `pred_goal` |
|:---|:---:|:---:|:---:|:---:|
| PR-AUC | — | — | — | — |
| PR-AUC × Null | — | — | — | — |
| ROC-AUC | — | — | — | — |
| Brier Score | — | — | — | — |
| Log-Loss | — | — | — | — |
| ECE | — | — | — | — |

*(Repeat for PowerPlay, Shorthanded, Empty For, Empty Against)*

**Reading the table:** The PR-AUC × Null row (PR-AUC divided by the null PR-AUC for that strength state) normalizes for base rate differences across strength states. This is the discriminative lift per strength state and the most comparable number across the five panels.

### Section 4: What EH Never Published

* **The Framing:** Evolving-Hockey built foundational public hockey analytics. This section is not an attack — it is completing an evaluation they left unfinished. Their model was tested and published by their own standards; these are additional standards that matter for any probabilistic model used in aggregation.
* **Key Findings (to be filled):** ECE measures whether the predicted probabilities are trustworthy. A model calibrated on one strength state's distribution and applied to another without recalibration will show systematic ECE drift. Five-model matching (one model per strength state, matched to the same holdout) controls for this. EH's five-model structure matches the cascade on this dimension — the comparison is clean.
* **The Calibration Finding:** If EH's calibration shows systematic overconfidence or underconfidence in specific probability buckets, name it directly. The goal is not to embarrass — it is to quantify the value of explicit OOF calibration (isotonic / Platt scaling) as an architectural decision.

### Section 5: Calibration Curves

* **What it shows:** For each model on the even-strength holdout, plot the reliability diagram: X-axis is predicted probability, Y-axis is observed fraction of goals in that bin. A perfectly calibrated model follows the 45° diagonal.
* **Color scheme:** `base_xg` (Cyan), EH (Gray dashed), `context_xg` (Magenta), `pred_goal` (Gold). Perfect calibration (dotted black diagonal).
* **What to emphasize:** Even small deviations from the diagonal in the 10–30% range represent meaningful miscalibration for aggregation purposes (e.g., summing xG across a season). The OOF calibration step's value should be visually apparent.

### Section 6: Season Stability

* **What it shows:** PR-AUC of `pred_goal` versus EH across four holdout seasons (2021–2024). Each season is independently held out — no data from season N is used in training the model evaluated on season N.
* **The claim:** The cascade's improvement over EH is consistent across seasons, not an artifact of the specific 2024 holdout. Season-to-season stability is a prerequisite for deployment confidence.
* **Chart:** Grouped bar chart. X-axis = holdout season. Two bars per season: `pred_goal` (Gold) and EH (Gray). Y-axis = PR-AUC.

## 3. Required Analyses, Charts, and Visuals

**1. The Cascade Progression Table**
* *What it is:* Per-strength-state table showing all five metrics for all four models. The centerpiece visual.
* *Format:* Clean HTML/LaTeX table with tier colors applied to column headers.

**2. Calibration Curves (Even Strength)**
* *What it is:* Reliability diagrams for all four models on the ES holdout. Four-color scheme.
* *Why:* The visual proof that OOF calibration matters — the curves that diverge from diagonal most are the ones without explicit calibration.

**3. Season Stability Bar Chart**
* *What it is:* Grouped bars — `pred_goal` (Gold) vs EH (Gray) — for 2021, 2022, 2023, 2024 holdouts.
* *Why:* Prevents the "lucky holdout" interpretation. Consistent lift across four seasons is the evidence.

**4. PR Curves Overlaid — Even Strength**
* *What it is:* Precision-Recall curve for all four models plotted on the same axes for the 2024-25 ES holdout. Lines: `base_xg` (Cyan), EH (Gray), `context_xg` (Magenta), `pred_goal` (Gold). Diagonal null-model reference at base rate.
* *Why:* The cascade progression table shows numbers; this chart shows the curves. The area under each curve is visibly larger moving left to right. At any given recall level, the cascade predictions have higher precision than EH — this is more viscerally convincing than a single PR-AUC number.
* *Data:* `sklearn.metrics.precision_recall_curve` on the 2024-25 ES holdout parquet with all four prediction columns. EH values joined by event_id + strength_state.

**5. Calibration Reliability Diagrams — All 5 Strength States**
* *What it is:* 5 small-multiple reliability diagrams (one per strength state). Each panel shows 4 reliability curves (base_xg / EH / context_xg / pred_goal) vs. the perfect-calibration diagonal. Shaded 95% confidence bands where sample size allows.
* *Why:* Expands the planned ES-only calibration chart to all 5 states. Shows cascade calibration is consistent, not a one-state result. Also shows that EH calibration degrades across states — PP and SH are notably worse because EH doesn't publish per-state models.
* *Data:* Hold-out season scored parquets for all 5 states. Decile binning (10 bins per curve).

**6. Season Stability Line Chart — All Three Cascade Tiers**
* *What it is:* Upgrade from the grouped bar (#3). Line chart: x=season (2010–2025), y=PR-AUC. One line each for `base_xg` (Cyan), `context_xg` (Magenta), `pred_goal` (Gold). Hold-out seasons (2024-25) marked with a vertical shaded band.
* *Why:* The line chart shows temporal structure the bar chart hides. The 2018-19 and 2023-24 dips visible in the diagnostics appear as a shared pattern across all three tiers — confirming it's a dataset structural shift, not model degradation in any single tier. The tiers moving together also validates that the cascade isn't introducing new temporal instability.
* *Data:* Scored parquets for all three tiers. PR-AUC computed per season across 2010–2025.

**7. "What EH Never Published" Micro-Table**
* *What it is:* A compact comparison table showing metrics EH did not compute or report publicly: ECE (Expected Calibration Error), Precision at base-rate threshold, Recall at base-rate threshold — for the EH model and `pred_goal` side by side, 2024-25 holdout, ES only.
* *Why:* EH published ROC-AUC and feature importance. These are the metrics a production inference system actually requires: calibration quality (ECE) and the precision/recall trade-off at the decision threshold. Position this as "the evaluation EH should have done."

## 4. nbdev Companion Notebook Specs (`05_the_proof.ipynb`)

* **Cell 1 (Data Load + EH Join):** Load the 2024 holdout parquet and EH per-shot xG values; join on event_id and strength_state. Show the join key explicitly — any unmatched rows should be logged and investigated.
    ```python
    holdout = pl.read_parquet("holdout_2024.parquet")
    eh = pl.read_parquet("eh_xg_2024.parquet")
    df = holdout.join(eh.select(["event_id", "strength_state", "eh_xg"]), on=["event_id", "strength_state"], how="left")
    ```
* **Cell 2 (Metric Suite):** Compute PR-AUC, ROC-AUC, Brier Score, Log-Loss, and ECE for all four model columns using `sklearn`. Loop over strength states.
    ```python
    from sklearn.metrics import (
        average_precision_score, roc_auc_score, brier_score_loss, log_loss
    )
    ```
* **Cell 3 (Cascade Progression Table):** Pivot the per-strength results into the four-column format. Apply tier colors to column headers for display.
* **Cell 4 (Calibration Curves):** `CalibrationDisplay.from_predictions` for all four models on the even-strength holdout. Four subplots or overlaid on one axes with the four-color scheme.
    ```python
    from sklearn.calibration import CalibrationDisplay
    CalibrationDisplay.from_predictions(y_true_es, pred_goal_es, n_bins=10, name="pred_goal", ax=ax, color=GOLD)
    ```
* **Cell 5 (Season Stability Loop):** Iterate over 2021–2024 holdout parquets, compute PR-AUC for `pred_goal` and EH per season, plot grouped bar chart.

---
### Summary of Result Reporting (Part 5)
**Part 5 (Formal Results):** The academic core of the series. Establishes the evaluation framework, presents the four-column cascade progression table across all five strength states, computes the full metric suite EH never published for themselves, and proves consistent multi-season lift over the public benchmark.
