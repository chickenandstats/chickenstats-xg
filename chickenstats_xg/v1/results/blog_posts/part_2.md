# Part 2 Master Plan: The Foundation (Tier 1 Geometry)

## 1. Metadata, Aesthetics, and Technical Specs
* **Primary Publication:** `chickenandstats.com` (Ghost)
* **Code Documentation:** `chickenstats-blog` (nbdev/GitHub Pages)
* **Hero Image:** "The Spatial Canvas." A sleek, dark-mode top-down view of a hockey offensive zone. The "home plate" area in front of the net glows intensely in Cyan, fading out to dark blue at the blue line, representing the pure geometric value of the ice.
* **Architecture Diagram:** A focused mini-flowchart showing just the Tier 1 extraction: `chickenstats` coordinates $\rightarrow$ Distance/Angle Math $\rightarrow$ `base_xg` XGBoost Model $\rightarrow$ Logit Handoff.
* **Color Standard:** * **Tier 1 (`base_xg`):** Cyan dominates this post. All charts, SHAP values, and heatmaps should utilize a Cyan-to-Dark-Blue gradient to visually reinforce that the reader is inside "Tier 1."

## 2. Narrative Structure (Ghost Post)

### Section 1: The Rink as a Canvas (Introduction)
* **The Concept:** Start by stripping the game of its chaos. No players, no passes, no clock. If a puck is instantly teleported to a specific spot on the ice and shot at the net, what are the chances it goes in? This is the pure "Physics" layer.
* **The Goal:** Explain that to accurately measure *how* a team generated a chance (Tier 2) or *who* finished it (Tier 3), we first need an undeniable, mathematically sound baseline of the shot's location. This baseline is our **Spatial Prior**.

### Section 2: Feature Engineering (The 8 Variables)
* **The Raw Data:** Show how the NHL API provides raw (X, Y) coordinates. 
* **The Math:** Briefly explain how you use the Pythagorean theorem to calculate **Distance** to the center of the goal line, and trigonometry to calculate the **Angle** (the "shooter's illusion" of the net).
* **Native Categoricals (Best Practice):** Highlight a key technical win here. Note that instead of using messy One-Hot Encoding for the `shot_type` (Wrist, Slap, Deflection), you leverage Polars categoricals and XGBoost's native categorical splitting (a direct nod to "Effective XGBoost" practices).

### Section 3: Training the Base Model (The Physics Layer)
* **Chronological Validation:** Explain *how* the model is trained. Emphasize that you never use random k-fold cross-validation in sports because it leaks the future into the past. You train on past seasons to predict future seasons.
* **Keep it Simple:** Point out that this specific model doesn't need intense interaction constraints. It is essentially a non-linear logistic regression mapping geometry to probability.

### Section 4: Visualizing the Spatial Prior (What the Model Learned)
* **The "Aha!" Visual:** Present a high-resolution 2D heatmap of the offensive zone colored by `base_xg`. 
* **Hockey Theory Validated:** Point out how the machine learning model naturally "discovered" classic hockey coaching concepts without being told about them. Show how the heatmap clearly defines the "Royal Road" and the high-danger "Home Plate" area. 
* **Feature Importance:** Show a SHAP Summary plot. Three features — `high_danger` (55.6%), `abs_y_distance` (23.9%), and `event_distance` (13.9%) — account for 93.4% of total SHAP gain. `shot_type` contributes only 3.3%, confirming that location dominates shot selection in the base model. Distance is king, angle is queen, and shot type is a modifier — but a weak one at this tier.

### Section 5: The Handoff (Setting up Tier 2)
* **The Conclusion:** We now have a robust `base_xg` for every shot. A 30-foot wrist shot might be a $0.06$ ($6\%$) chance. 
* **The Cliffhanger:** Remind the reader of the "Credit Assignment Problem." What happens if that 30-foot wrist shot was taken while the goalie was completely out of position after a cross-ice pass? The $6\%$ geometric baseline is wrong. 
* **The Calibration Step:** Before the 6% baseline can be handed to Tier 2, it must be calibrated. A raw XGBoost probability of 0.06 is only meaningful if shots the model scores at 0.06 actually go in 6% of the time. To enforce this, we apply Out-of-Fold (OOF) calibration using isotonic regression: each shot's calibrated probability is predicted by a fold that never saw that shot during training. The calibrated output is what gets logit-transformed and passed forward. Without this step, `logit(base_xg)` is an anchor built on a shifted scale — the entire residual-learning architecture of Tier 2 would be correcting Tier 1 miscalibration instead of learning sequence signal.
* **The Bridge:** Briefly reiterate that we take this calibrated $6\%$ (`base_xg`), convert it to log-odds using the `logit` function, and pass it forward as the `base_margin` for Tier 2, which we will build in Part 3.

## 3. Required Analyses, Charts, and Visuals

**1. The Spatial Prior Heatmap (Crucial)**
* *What it is:* A 2D hexbin or smoothed contour plot of the offensive zone.
* *Data:* Binned average `base_xg` probabilities across the ice.

**2. SHAP Summary Plot (Tier 1)**
* *What it is:* Standard XGBoost SHAP summary showing the impact of the 8 base features.
* *Focus:* Visually confirming that low distance = positive SHAP, high distance = negative SHAP.

**3. Feature Dependence Plot: Distance vs. `base_xg`**
* *What it is:* A scatter plot with `event_distance` on the X-axis and predicted `base_xg` on the Y-axis. 
* *Why:* Shows the non-linear decay of shot danger as a player moves further from the net — a logistic curve, not exponential. Flat plateau near the net (0–10 ft), steep falloff through the mid-range, asymptoting toward zero at distance.

**4. SHAP Beeswarm (upgrade from summary bar)**
* *What it is:* `shap.plots.beeswarm` for the base_xg even-strength model on ~50K sampled shots. Each dot is one shot, positioned on the x-axis by its SHAP value for that feature, colored by raw feature value (high = warm, low = cool).
* *Why:* The beeswarm shows the *distribution* of SHAP values, not just mean absolute importance. `high_danger` appears as a binary cluster (0 or a large positive value — the flag fires or doesn't). `event_distance` shows a continuous gradient. Significantly more informative than a summary bar and sets up the contrast with the Tier 2 beeswarm in Part 3.
* *Data:* `shap.TreeExplainer` on the base_xg ES `model.ubj`, 50K random sample from scored parquet.

**5. Feature Gain Stacked Bar — All 5 Strength States**
* *What it is:* One horizontal stacked bar per strength state. Each segment = one feature's share of total gbtree gain. States ordered ES / PP / SH / EF / EA.
* *Why:* Shows that the same 8 geometry features decompose differently across states. `high_danger` dominates SH (83.4%) and PP (62.7%), while `event_distance` contributes more for ES (13.9%) and EA (14.1%). The model did not simply learn one "xG formula" — it learned state-specific geometry.
* *Data:* `booster.get_score(importance_type="gain")` for all 5 base_xg models. Tier color palette for the feature segments.

**6. Optuna Hyperparameter Importance (base_xg Even Strength)**
* *What it is:* Bar chart of Optuna's FANOVA hyperparameter importance scores for the `even_strength-1.0.0-base` study. Shows which tuned parameters (lambda, scale_pos_weight, learning_rate, subsample) actually drove CV PR-AUC variance across trials.
* *Why:* Establishes that the base_xg model is stable to hyperparameter choice — geometry is a clean, low-noise signal. This directly contrasts with Part 3's context_xg, where the optimization landscape is nearly flat and `max_delta_step` dominates selection in a way that isn't visible in standard CV rankings.
* *Data:* PostgreSQL Optuna DB, `even_strength-1.0.0-base` study; `optuna.importance.get_param_importances()`.

## 4. nbdev Companion Notebook Specs (`02_the_foundation.ipynb`)

**File:** `02_the_foundation.ipynb`

* **Cell 1 (Data & Setup):** Import the `chickenstats` package. Pull a clean DataFrame of raw shot events.
* **Cell 2 (Polars Feature Engineering):** Note that `event_distance` is pre-computed by the `chickenstats` package from adjusted NHL coordinates — it measures the Euclidean distance from the shot location to the center of the goal line, not from the coordinate origin. The notebook should show how to confirm the feature is present in the pulled DataFrame and inspect the distribution, rather than re-deriving it from raw `x_coord`/`y_coord`.
* **Cell 3 (Data Types):** Show how to cast `shot_type` as `pl.Categorical` for native XGBoost handling.
* **Cell 4 (Model Training):** The actual `xgboost.train()` code block. Show the exact hyperparameter dictionary used for the `base_xg` model.
* **Cell 5 (SHAP Extraction):** Code to generate the SHAP values and plot the Summary chart using the `shap` library.
* **Cell 6 (Rink Plotting):** Provide the exact `matplotlib` code (ideally using a package like `hockey_rink` or a custom background image) to generate the Spatial Prior Heatmap.
* **Cell 7 (OOF Calibration):** Show the `TimeSeriesSplit` OOF loop — for each fold, fit the model on the training split, predict the held-out split, and collect OOF probabilities. Fit isotonic regression on the pooled OOF predictions. Show a before/after reliability diagram (predicted vs. actual goal rate by decile) confirming the calibrator pulls the curve to the diagonal. This is the step that makes `logit(base_xg)` a valid log-odds anchor for Tier 2.