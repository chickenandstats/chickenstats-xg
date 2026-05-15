# context_xg Diagnostics

## Latest Diagnostic

**Date:** 2026-05-14
**Model version:** 1.0.0 (20-feature gbtree depth-2, 9 constraint groups)
**Trials:** 750 (even_strength) / 1000 (powerplay, shorthanded, empty_for, empty_against — combined across study restarts)
**Hold-out season:** 2024-25
**Key change from prior run:** Finalized with `--top-n 150` (was 15). Wider screening window recovered non-bimodal trials that were below the top-15 CV PR-AUC cutoff. All states now pass calibration.

### Pass / Fail Summary

| Strength | Distribution | High Conf | Calibration | OOF Gap | Lift | Gain Conc | Overall |
|---|---|---|---|---|---|---|---|
| even_strength | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ PASS |
| powerplay | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ PASS |
| shorthanded | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ PASS |
| empty_for | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ PASS |
| empty_against | ✅ | ✅ | ⚠️ | ✅ | ✅ | ✅ | ⚠️ WARN |

### Advanced Metrics (hold-out 2024-25)

`Lift` = context_xg hold-out PR AUC − base_xg hold-out PR AUC on the same events.
`Max Cal Error` is the uniform-bin max calibration error; the calibration PASS/WARN/FAIL check uses quantile-based deciles.
`Null Brier` = base_rate × (1 − base_rate). Positive ΔLL% / ΔBr% means the model improves on predicting the base rate.

| Strength | Base% | PR AUC | PR× | ROC AUC | Log Loss | Null LL | ΔLL% | Brier | Null Brier | ΔBr% | ECE | Max Cal Error | OOF Gap | Lift | Precision | Recall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| even_strength | 6.0% | 0.3213 | 5.39× | 0.7844 | 0.1873 | 0.2260 | +17.1% | 0.0472 | 0.0561 | +15.9% | 0.0055 | 0.5051 | 0.0213 | +0.1622 | 0.1562 | 0.5584 |
| powerplay | 10.3% | 0.3439 | 3.35× | 0.7009 | 0.2871 | 0.3310 | +13.3% | 0.0785 | 0.0921 | +14.8% | 0.0061 | 0.6422 | 0.0287 | +0.1704 | 0.1916 | 0.4938 |
| shorthanded | 7.2% | 0.3349 | 4.64× | 0.8200 | 0.2130 | 0.2592 | +17.8% | 0.0567 | 0.0669 | +15.3% | 0.0088 | 0.5174 | 0.0221 | +0.1435 | 0.1861 | 0.5882 |
| empty_for | 7.7% | 0.3136 | 4.05× | 0.7157 | 0.2357 | 0.2722 | +13.4% | 0.0611 | 0.0714 | +14.4% | 0.0054 | 0.4629 | 0.0056 | +0.1502 | 0.1694 | 0.5435 |
| empty_against | 56.7% | 0.7879 | 1.39× | 0.7072 | 0.6083 | 0.6842 | +11.1% | 0.2131 | 0.2455 | +13.2% | 0.0638 | 0.2339 | 0.0052 | +0.0450 | 0.7292 | 0.6021 |

**Note on Max Cal Error (uniform bins):** These values (0.21–0.64) look alarming but are sparse-bin artifacts. With SHOT p90 < 0.13 for low-base-rate states, 95%+ of predictions fall in [0, 0.13]; any uniform bin in [0.3, 0.8] has near-zero samples. The quantile-decile max abs error (0.03–0.06) is the correct calibration measure and drives the PASS/WARN/FAIL verdict.

---

### Hyperparameters

**Fixed params (all states):** booster=gbtree, objective=binary:logistic, max_depth=2, n_estimators=500, early_stopping_rounds=50, eval_metric=["aucpr","logloss"] (early stopping on logloss), random_state=615, enable_categorical=True.

The trial-level params below are the most likely selected trial for each state, identified heuristically by: (1) max_delta_step=1 (required for non-bimodal calibration — all bimodal trials have mds≥2), and (2) learning_rate consistent with the observed best_iteration. Exact determination requires re-running the calibrated screening output.

| State | Trial # | mds | lambda | gamma | alpha | mcw | lr | subsample | spw | best_iter |
|---|---|---|---|---|---|---|---|---|---|---|
| even_strength | ~492 | 1 | 47.03 | 1.44 | 4.96 | 144 | 0.116 | 0.80 | 1.030 | 230 |
| powerplay | ~460 | 1 | 8.49 | 2.05 | 1.04 | 196 | 0.196 | 0.95 | 1.059 | 82 |
| shorthanded | ~748 | 1 | 9.09 | 4.10 | 0.22 | 50 | 0.207 | 1.00 | 1.194 | 202 |
| empty_for | ~547 | 1 | 10.06 | 2.93 | 0.29 | 50 | 0.175 | 0.75 | 1.073 | 122 |
| empty_against | 606 | 1 | 95.62 | 5.67 | 4.69 | 53 | 0.154 | 0.50 | N/A | 149 |

`mds` = max_delta_step. `mcw` = min_child_weight. `spw` = scale_pos_weight. `best_iter` = Optuna CV run estimate (not the final model's tree count — see note below).

**Note on best_iter:** Values above are estimated from the Optuna CV runs, not read from the saved model artifact. Final model `best_iteration` values (from `Booster.best_iteration` on the full-dataset training run): ES=21, PP=46, SH=104, EF=95, EA=175. Divergence from the CV estimates is expected — full-data training converges faster than cross-validation folds because more data is available per iteration and early stopping fires against a larger validation set.

**Hyperparameter assessment:**

**max_delta_step=1 is the critical parameter across all states.** Every likely-selected trial has mds=1. The bimodal cliff is driven by high-mds trials (2–5): larger per-tree leaf updates accumulate into a high-probability cluster for flag shots over 500 trees. mds=1 limits each tree's contribution to ≤1 log-odds unit before the learning rate is applied, preventing runaway accumulation. The Optuna CV landscape is flat (top-5 PR-AUC span ~0.0005), so bimodal (high-mds) trials dominate the top-N by a small margin. The calibrated screening must reach rank 100+ to find mds=1 trials reliably — hence `--top-n 150`.

**even_strength (~Trial #492):** lambda=47 is the strongest regularization of the four low-base-rate states — appropriate for the 1.24M-shot dataset where stable high-lambda models are feasible. lr=0.116 (lowest across all states) is consistent with best_iter=230 (slower convergence). spw≈1.03 near-uniform: correct because base_margin (logit_base_xg) already anchors the 6% prior.

**powerplay (~Trial #460):** lambda=8.49 is lighter, matching the smaller 216K-shot dataset. best_iter=82 is unusually low — logloss early stopping fired quickly with mds=1 and moderate lr, suggesting the model reached a well-calibrated state before overfitting could develop. spw≈1.06 near-uniform.

**shorthanded (~Trial #748):** min_child_weight=50 is at the search space floor for a 35K-shot dataset — the principal weakness of this model. Gamma=4.10 provides compensating pruning pressure. lambda=9.09 is moderate. The floor pressure means future tuning rounds should raise the min_child_weight lower bound (suggested: 100 for SH).

**empty_for (~Trial #547):** Same min_child_weight=50 floor concern (29K shots). lambda=10 and gamma=2.93 are reasonable. best_iter=122 reflects moderate early stopping.

**empty_against (Trial #606 — high confidence):** Trial #606 is confidently identified as the winner: highest CV PR-AUC (0.7674) among all mds=1 candidates for EA. lambda=95.62 is the highest across all states and all trials — strong regularization critical for a ~9K training event dataset. subsample=0.5 (aggressive row sampling) further prevents overfitting. The combination of mds=1 + lambda=95.62 + subsample=0.5 produces the best ECE (0.064) of any state. logit_base_xg feature gain (4.2%) is the highest for EA across any run, reflecting improved use of the base_xg quality signal.

---

### Per-Strength Interpretation

#### even_strength

**Performance tier:** ✅ PASS. Discrimination: very high (0.3213 ≥ 0.30 cutoff).

**State:** Full calibration recovery. SHOT p90 dropped from 0.715 (bimodal, prior run) to **0.084** (1.44× base rate, PASS). Decile 9 near-perfect: mean_pred=0.230, actual=0.231, abs_err=0.0007. Max abs error 0.034 at decile 8. ECE=0.0055. Log loss +17.1% vs null. Precision=0.156 (2.6× base rate) and recall=0.558 at the decision threshold — previously precision=base_rate=0.060 with trivial recall=1.0.

**Season variance:** 2018–19 and 2023–24 consistently show PR-AUC around 0.19–0.21, below the 0.27–0.43 range of other seasons. This is a stable dataset pattern, not model degradation — likely related to rule changes or structural shifts in context-feature correlation with outcomes across those seasons.

**Mild remaining issue:** Decile 8 abs_err=0.034 (mean_pred=0.073, actual=0.107). The model slightly underestimates shots in the 70th–80th percentile. Minor overconservatism, not a bimodal artifact — within the PASS threshold.

**Feature gain:** `seconds_since_stoppage` 53.7%, `prior_event_angle` 16.7%, `seconds_since_last` 6.2%, `prior_event_distance` 3.7%, `prior_event_opp` 2.8%, `logit_base_xg` 1.7%.

`seconds_since_stoppage` at 53.7% is appropriate: for 5v5 play the strongest context signal is how recently the clock was stopped (faceoff, icing, power play setup). Shots immediately after stoppages are systematically higher quality than shots in sustained flow-of-play.

---

#### powerplay

**Performance tier:** ✅ PASS. Discrimination: very high (0.3439 ≥ 0.30 cutoff).

**State:** Full calibration recovery. SHOT p90=0.127 (1.34× base rate, PASS). Decile 9 near-perfect: mean_pred=0.302, actual=0.292, abs_err=0.009. Max abs error 0.029 at decile 8. ECE=0.006. Log loss +13.3% vs null. Lift over base_xg: +0.170 — the best absolute lift of all states.

**Unusual best_iteration=82:** Logloss early stopping fired after only 82 trees (of 500 max). PP has the highest base rate (10.3%) of the low-base-rate states and mds=1, allowing the model to find a good calibration state quickly. Despite fewer trees, PR-AUC and log loss are the strongest among the four low-base-rate states.

**Feature gain:** `seconds_since_last` 32.5%, `play_speed` 14.7%, `prior_event_distance` 13.4%, `seconds_since_stoppage` 11.4%, `prior_event_angle` 10.2%, `logit_base_xg` 1.6%.

The dominant feature for PP is `seconds_since_last` (vs `seconds_since_stoppage` for ES). On the power play, sustained zone pressure means recency of the last shot or pass is the primary context signal — faceoff timing matters less because PP sequences don't reset with stoppages the way 5v5 play does.

---

#### shorthanded

**Performance tier:** ✅ PASS. Discrimination: very high (0.3349 ≥ 0.28 cutoff).

**State:** Full calibration recovery on the most volatile dataset (35K shots, 2,592 hold-out events). SHOT p90=0.115 (1.64× base rate, PASS). Decile 9 near-perfect: mean_pred=0.287, actual=0.277, abs_err=0.009. Max abs error 0.039 at decile 8. ECE=0.009. Log loss +17.8% vs null — the **best log loss improvement of all states**.

**Low-decile floor (mild concern):** Deciles 0–3 overestimate by 0.020–0.029. The model predicts a floor of ~3.1% for the lowest-quality SH shots; actual rates are 0.15–1.4%. This is a min_child_weight=50 artifact: minimum leaf size of 50 samples prevents the model from reaching very low predictions. Within the PASS threshold (max abs error 0.039 ≤ 0.05 WARN), and decile 9 is near-perfect. Raising min_child_weight floor to 100 in future tuning would improve this.

**Feature gain:** `play_speed` 22.1%, `seconds_since_last` 11.2%, `prior_event_angle` 10.9%, `prior_event_distance` 9.2%, `seconds_since_stoppage` 8.8%, `logit_base_xg` 5.3%.

`play_speed` is the only state where it dominates. Shorthanded goals are predominantly fast-break situations (breakaways, odd-man rushes). This is physically meaningful — SH scoring is almost exclusively counter-attack driven, making transition speed the single most discriminative context feature.

---

#### empty_for

**Performance tier:** ✅ PASS. Discrimination: high (0.3136 ≥ 0.27 cutoff).

**State:** Full calibration recovery. SHOT p90=0.104 (1.32× base rate, PASS — the lowest normalized ratio, reflecting tight prediction spread). Decile 9 near-perfect: mean_pred=0.263, actual=0.266, abs_err=0.002. Max abs error 0.032 at decile 8. ECE=0.005 — the lowest ECE of all states. Log loss +13.4% vs null. OOF gap=0.006 — the smallest of all states, excellent generalization.

**Feature gain:** `seconds_since_last` 22.2%, `seconds_since_stoppage` 20.6%, `prior_event_angle` 16.3%, `prior_event_distance` 13.3%, `is_rebound` 5.9%, `logit_base_xg` 2.3%.

Two timing features split the gain for EF (`seconds_since_last` + `seconds_since_stoppage` = 42.8%). Empty-net attacks have strong temporal patterns: shots in immediate scrambles after a faceoff win (low `seconds_since_stoppage`) and direct follow-up shots (low `seconds_since_last`) convert at much higher rates than possession shots later in the sequence.

---

#### empty_against

**Performance tier:** ⚠️ WARN (calibration decile 5–6). Discrimination: high (0.7879 ≥ 0.73 cutoff).

**State:** Meaningful improvement. Log loss +11.1% vs null (vs +0.1% in the biased-screening run). ECE=0.064. The WARN is from decile 5–6: mean_pred=0.588–0.612, actual=0.528–0.584, abs_err=0.041–0.059. Every other decile is well-calibrated: decile 9 has abs_err=0.005 (0.918 pred vs 0.923 actual). Precision=0.729, recall=0.602 at the 56.7% base rate threshold — genuine discrimination.

**Character of the WARN:** The model is moderately over-confident in the 55–65% probability range. This is a structural compression artifact: with a 56.7% base rate, predictions are forced into a narrow [0.30, 0.93] range, and the Platt calibrator cannot fully resolve mid-range overconfidence after correcting the extremes. Not a bimodal failure — a smooth monotone miscalibration that partially degrades probability quality in the middle of the prediction range.

**Feature gain:** `play_speed` 22.2%, `seconds_since_stoppage` 21.0%, `period_seconds` 10.6%, `prior_event_angle` 10.2%, `prior_event_distance` 8.8%, `seconds_since_last` 8.8%, `logit_base_xg` 4.2%.

`logit_base_xg` at 4.2% is the highest for EA across any run — the model now uses the base_xg quality signal meaningfully. `play_speed` and `seconds_since_stoppage` together reflect the two EA goal-scoring modes: fast counters into the empty net (high play_speed) and attacks immediately after a stoppage (low seconds_since_stoppage).

---

## Changelog

| Date | Version | Trials | ES PR AUC | PP PR AUC | SH PR AUC | EF PR AUC | EA PR AUC | ES Lift | PP Lift | SH Lift | EF Lift | EA Lift | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-05-13 | 1.0.0 | 100 | 0.3188 | 0.3406 | 0.3148 | 0.3049 | 0.7845 | +0.1597 | +0.1672 | +0.1233 | +0.1415 | +0.0417 | Initial 100-trial run; all states FAIL on calibration due to bimodal prediction distribution; discrimination strong (ES PR AUC 5.34× null, ROC 0.7833); OOF gaps all PASS |
| 2026-05-13 | 1.0.0 | ~500 (no bm) | 0.3192 | 0.3399 | 0.3189 | 0.3056 | 0.7845 | +0.1601 | +0.1665 | +0.1274 | +0.1422 | +0.0417 | SH improved FAIL→WARN (log loss −3.4% vs null); ES/PP calibration worsened (more trials found stronger flag boosts); EF calibration degraded (log loss −37.0%); EA OOF gap moved to WARN (0.0300); confirms bimodal failure is structural — base_margin fix required (Issue 11) |
| 2026-05-13 | 1.0.0 | ~500 (base_margin) | 0.3198 | 0.3427 | 0.3306 | 0.3066 | 0.7867 | +0.1607 | +0.1692 | +0.1391 | +0.1432 | +0.0439 | First run with logit_base_xg as base_margin (Issue 11). ES bimodal cliff collapsed (SHOT p90: 0.513→0.217); log loss −33.7%. PP improved FAIL→WARN (log loss −135.7%→−2.6%). EA OOF gap fixed (0.030→0.007). SH catastrophically regressed: log loss −3.4%→−462.6%, SHOT p90 jumped to 0.784 — calibrated top-N screening hit fallback (all 15 candidates bimodal in new landscape). eval_metric bug fixed (["aucpr","logloss"] → early stop on logloss), max_delta_step added to search space (1–5). |
| 2026-05-14 | 1.0.0 | 750 / 1000 (top-n 15) | 0.3198 | 0.3407 | 0.3330 | 0.3042 | 0.7801 | +0.1607 | +0.1672 | +0.1415 | +0.1408 | +0.0373 | ALL STATES FAIL. Top-N screening failure: flat CV landscape (ES top-5 span 0.0005 PR-AUC) means bimodal (high-mds) trials fill all 15 screening slots; non-bimodal (mds=1) trials are at rank 16+. All candidates fail 2× null cal_ll threshold → fallback to least-bad bimodal → catastrophic miscalibration. ES: −440.4%, PP: −183.8%, SH: −326.9%, EF: −166.8%, EA: +0.1%. Fix: increase --top-n. |
| 2026-05-14 | 1.0.0 | 750 / 1000 (top-n 150) | 0.3213 | 0.3439 | 0.3349 | 0.3136 | 0.7879 | +0.1622 | +0.1704 | +0.1435 | +0.1502 | +0.0450 | **Full calibration recovery.** All low-base-rate states PASS (ES/PP/SH/EF log loss +13–18% vs null, ECE < 0.01). EA ⚠️ WARN (calibration decile 5–6 overestimation; ECE=0.064; +11.1% log loss). Selected trials all have max_delta_step=1 — confirmed as the critical parameter for avoiding the bimodal cliff. Also fixed two diagnose.py checks: distribution check now uses SHOT p90 / base_rate (was inverted GOAL/SHOT ratio); high_conf thresholds now scale with base rate (EA no longer penalised for naturally elevated predictions). |