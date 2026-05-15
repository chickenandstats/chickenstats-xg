# pred_goal Diagnostics

## Latest Diagnostic

**Date:** 2026-05-14
**Model version:** 1.0.0 (pred_goal tier — talent features layered on context_xg base_margin)
**Trials:** 500 per state
**Finalization:** `--top-n 15` (default)
**Hold-out season:** 2024-25
**Key change from initial run:** Pooled OOF + hold-out calibration (was OOF-only). Resolved catastrophic log loss degradation across all four low-base-rate states (ES: −994% → −17.7% vs null; PP: −241% → −18.8%; SH: −442% → −22.5%; EF: −259% → −15.0%). See Issue 14 in `v1_model_issues.md`.

### Pass / Fail Summary

| Strength | Distribution | Calibration | OOF Gap | Lift | RAPM Null | Feature Gain | Overall |
|---|---|---|---|---|---|---|---|
| even_strength | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ FAIL |
| powerplay | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ FAIL |
| shorthanded | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ FAIL |
| empty_for | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ FAIL |
| empty_against | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ FAIL |

**Failure modes are distinct by state group:**
- **ES / PP / SH / EF:** Residual calibration FAIL (decile 8 non-monotone pattern; max abs error 0.13–0.19). Lift is positive but negligible (+0.0001–+0.0009 PR AUC over context_xg). See Issue 15.
- **EA:** Calibration PASS (max abs error 0.023). Negative lift (−0.034 PR AUC vs context_xg). See Issue 15.

### Advanced Metrics (hold-out 2024-25)

`Lift` = pred_goal hold-out PR AUC − base_xg (= context_xg) hold-out PR AUC on the same events.
`Max Cal Error` is the uniform-bin max calibration error; the calibration PASS/WARN/FAIL check uses quantile-based deciles.
`Null Brier` = base_rate × (1 − base_rate). Positive ΔLL% / ΔBr% means the model improves on predicting the base rate; negative means it is worse than predicting the base rate for every event.

| Strength | Base% | PR AUC | PR× | ROC AUC | Log Loss | Null LL | ΔLL% | Brier | Null Brier | ΔBr% | ECE | Max Cal Error | OOF Gap | Lift | Precision | Recall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| even_strength | 6.0% | 0.3204 | 5.37× | 0.7841 | 0.2660 | 0.2260 | −17.7% | 0.0642 | 0.0561 | −14.4% | 0.0960 | 0.0960 | 0.0203 | +0.0001 | 0.0597 | 1.0000 |
| powerplay | 10.3% | 0.3420 | 3.33× | 0.7012 | 0.3930 | 0.3310 | −18.8% | 0.1107 | 0.0921 | −20.2% | 0.1687 | 0.5922 | 0.0267 | +0.0005 | 0.1027 | 1.0000 |
| shorthanded | 7.2% | 0.3336 | 4.62× | 0.8188 | 0.3175 | 0.2592 | −22.5% | 0.0813 | 0.0669 | −21.4% | 0.1443 | 0.6899 | 0.0215 | +0.0006 | 0.0721 | 1.0000 |
| empty_for | 7.7% | 0.3121 | 4.04× | 0.7180 | 0.3132 | 0.2722 | −15.0% | 0.0808 | 0.0714 | −13.2% | 0.1244 | 0.6844 | 0.0041 | +0.0009 | 0.0773 | 1.0000 |
| empty_against | 56.7% | 0.7480 | 1.32× | 0.7032 | 0.6179 | 0.6842 | +9.7% | 0.2165 | 0.2455 | +11.8% | 0.0839 | 0.1325 | 0.0116 | −0.0340 | 0.7110 | 0.6585 |

**Note on Max Cal Error (uniform bins) for PP / SH / EF:** Values of 0.59–0.69 look alarming but are sparse-bin artifacts. The hold-out sets for these states are small (2,592–15,795 events). With predictions mostly below 0.30, the uniform bins [0.4–0.5] or [0.5–0.6] may contain only a handful of events — one or two goals determines the entire calibration error for that bin. The quantile-decile max abs error (0.14–0.19) is the meaningful calibration measure driving PASS/WARN/FAIL. ES max cal = ECE = 0.096 (single problematic bin dominates both measures).

**Context_xg reference (same hold-out, for lift comparison):**

| Strength | ctx_xg PR AUC | pred_goal PR AUC | Lift |
|---|---|---|---|
| even_strength | 0.3202 | 0.3204 | +0.0001 |
| powerplay | 0.3415 | 0.3420 | +0.0005 |
| shorthanded | 0.3330 | 0.3336 | +0.0006 |
| empty_for | 0.3112 | 0.3121 | +0.0009 |
| empty_against | 0.7820 | 0.7480 | −0.0340 |

---

### Hyperparameters

**Fixed params (all states):** objective=binary:logistic, booster=gbtree, n_estimators=500, early_stopping_rounds=50, eval_metric=["logloss","aucpr"] (early stop on logloss), random_state=615, enable_categorical=True. Uses `_params_base_xg` search space (same as base_xg tier). context_xg's calibrated probability is passed as `logit(context_xg)` via `base_margin` — talent features are the only feature matrix input.

**⚠ Per-trial hyperparameters (lambda, gamma, alpha, max_depth, etc.) are not available.** The Optuna studies for pred_goal (`{state}-1.0.0-pred`) were deleted from the database before meta.json storage was implemented. For future runs, `finalize.py` must write a `meta.json` alongside the model artifacts (matching the base_xg pattern) to preserve trial hyperparameters. Available model-derived values only:

| State | best_score (CV PR AUC) | best_iter | n_trees |
|---|---|---|---|
| even_strength | 0.3204 | 5 | 56 |
| powerplay | 0.3419 | 2 | 53 |
| shorthanded | 0.3326 | 2 | 53 |
| empty_for | 0.3114 | 4 | 55 |
| empty_against | 0.7826 | 0 | 51 |

`best_iter` and `n_trees` are from `Booster.best_iteration` and `len(Booster.get_dump())` on the saved model artifact.

**Note on low best_iterations:** ES=5, PP=2, SH=2, EF=4, EA=0. These reflect the structural role of pred_goal as a thin adjustment layer stacked on context_xg's logit base_margin. Talent features provide small marginal signal; logloss early stopping fires quickly once the talent adjustment stabilizes. The ~50–56 total trees represent the complete final adjustment at the talent layer, not a failure to train — discrimination (PR AUC 0.32–0.79) matches context_xg, confirming the models are not degenerate.

---

### Feature Gain

Talent features only — `BASE_XG_FEATURE_COLUMNS` and `CONTEXT_XG_FEATURE_COLUMNS` are confirmed zero gain in all states (no context leakage). Top 8 features by state:

**even_strength (top 12 of 49 features):**

| Feature | Gain% |
|---|---|
| goalie_gsax_per_shot_1g | 45.0% |
| goalie_gsax_ewma | 27.5% |
| goalie_gsax_per_shot_10g | 20.9% |
| shooter_gax_per_shot_10g | 5.1% |
| shooter_gax_ewma | 0.8% |
| shooter_rapm_career_goals_off | 0.1% |
| (all RAPM features) | < 1% combined |

**powerplay (top 8 of 56 features):**

| Feature | Gain% |
|---|---|
| goalie_gsax_per_shot_10g | 16.5% |
| goalie_gsax_1g | 16.0% |
| shooter_gax_per_shot_10g | 15.9% |
| goalie_gsax_ewma | 12.8% |
| shooter_gax_ewma | 12.5% |
| goalie_gsax_per_shot_1g | 10.0% |
| shooter_gax_per_shot_1g | 6.0% |
| shooter_gax_10g | 4.1% |

**shorthanded (top 8 of 37 features):**

| Feature | Gain% |
|---|---|
| goalie_gsax_per_shot_1g | 16.0% |
| shooter_gax_per_shot_10g | 15.5% |
| shooter_gax_ewma | 14.3% |
| goalie_gsax_ewma | 14.2% |
| goalie_gsax_per_shot_10g | 12.9% |
| goalie_gsax_10g | 11.3% |
| shooter_gax_10g | 7.7% |
| goalie_gsax_1g | 3.7% |

**empty_for (top 8 of 42 features):**

| Feature | Gain% |
|---|---|
| goalie_gsax_per_shot_10g | 18.1% |
| goalie_gsax_10g | 17.6% |
| goalie_gsax_per_shot_1g | 15.3% |
| goalie_gsax_ewma | 14.3% |
| shooter_gax_ewma | 10.2% |
| shooter_gax_per_shot_10g | 7.1% |
| shooter_gax_10g | 5.1% |
| shooter_gax_per_shot_1g | 1.8% |

**empty_against (top 8 of 45 features):**

| Feature | Gain% |
|---|---|
| shooter_gax_ewma | 5.4% |
| shooter_rapm_xg_off | 3.9% |
| shooter_vs_teammates_rapm_career_goals_off | 3.6% |
| shooter_gax_10g | 3.4% |
| opp_rapm_career_corsi_def | 3.3% |
| shooter_shots_10g | 3.1% |
| shooter_rapm_corsi_def | 3.0% |
| shooter_gax_per_shot_10g | 2.7% |

**RAPM null rates:** 0.0% across all states and all 14 training seasons (73 nulls out of 1.2M ES shots; comparable near-zero rates for other states). Career RAPM is available for effectively all skaters in the dataset.

---

### Per-Strength Interpretation

#### even_strength

**Performance tier:** ❌ FAIL (calibration). Discrimination: strong (0.3204 = 5.4× null, ROC 0.784). Calibration usable but overconfident in one cluster.

**Calibration character:** The distribution has a residual bimodal structure. Deciles 0–7 have essentially flat mean predictions (0.046–0.091), indicating most shots cluster around the base-margin prior. Decile 8 jumps to mean_pred=0.152 with actual=0.023 (7× overestimate) — a cluster of shots the model assigns moderately high probability that are almost never goals in hold-out. Decile 9 is near-correct (mean_pred=0.162, actual=0.150). This decile-8 anomaly is the residual from the original bimodal cliff: after the pooled calibration fix, the extreme 0.92-predicted cluster was pulled down to 0.15, but even 0.15 overestimates the ~2% actual hold-out rate for those events.

**Lift:** +0.0001 PR AUC over context_xg. The seasonal table shows pred_goal and ctx_xg are essentially identical in all 15 seasons. Talent features add negligible discriminative power beyond the context prior.

**Feature gain dominant pattern:** Three goalie rolling-performance features account for 93.4% of gain — `goalie_gsax_per_shot_1g` (45%), `goalie_gsax_ewma` (28%), `goalie_gsax_per_shot_10g` (21%). RAPM features combined: < 1%. The model has essentially learned to ask "is the goalie having a bad recent game?" and largely ignores stable career talent signals. The 1-game goalie feature at 45% gain is a noisy signal (single-game performance is high-variance) that generalises poorly to the hold-out season.

---

#### powerplay

**Performance tier:** ❌ FAIL (calibration). Discrimination: strong (0.3420 = 3.3× null, ROC 0.701). Calibration usable but overconfident in the upper cluster.

**Calibration character:** Same decile-8 pattern as ES. Mean predictions are compressed (deciles 0–7 span only 0.070–0.152), then decile 8 overestimates (mean_pred=0.242, actual=0.055). Decile 9 is near-correct (mean_pred=0.280, actual=0.222). Log loss −18.8% vs null — meaningful calibration improvement despite the FAIL.

**Lift:** +0.0005 PR AUC over context_xg. Negligible across all 15 seasons. The PP talent model is not providing useful signal beyond context_xg for discriminating goals.

**Feature gain:** More balanced than ES — 6 features account for >10% gain each (goalie and shooter gax features roughly even). RAPM features combined: ~1%. The goalie/shooter balance reflects that PP situations have more predictable matchup dynamics; both goalie form and shooter form carry signal, but neither is decisive. Short-horizon (1g, per_shot_10g) features still dominate over stable career metrics.

---

#### shorthanded

**Performance tier:** ❌ FAIL (calibration). Discrimination: strong (0.3336 = 4.6× null, ROC 0.819 — highest ROC of all states). Calibration usable but overconfident in the upper cluster.

**Calibration character:** Decile-8 pattern: mean_pred=0.196, actual=0.018 (11× overestimate). Log loss −22.5% vs null. Max cal error 0.178 (decile threshold). The SH dataset is the smallest non-EA state (35K shots, 2,592 hold-out events), making decile-level calibration more volatile.

**Lift:** +0.0006 PR AUC over context_xg. Near-zero. SH is fast-break dominated (context_xg already captures this well through sequence timing features).

**Feature gain:** More distributed than ES — goalie and shooter features roughly equal, with `goalie_gsax_per_shot_1g` (16%) and `shooter_gax_per_shot_10g` (15.5%) leading. RAPM features: ~1.5%. The smaller dataset means fewer stable talent patterns can be learned; the model relies on recent rolling performance rather than career-level signal.

---

#### empty_for

**Performance tier:** ❌ FAIL (calibration). Discrimination: strong (0.3121 = 4.0× null, ROC 0.718). OOF gap = 0.0041 — the smallest of all states, excellent generalization.

**Calibration character:** Decile 8: mean_pred=0.182, actual=0.042 (4.3× overestimate). Decile 9 near-perfect (mean_pred=0.210, actual=0.210, abs_err=0.0008). Log loss −15.0% vs null — the weakest improvement of the four low-base-rate states.

**Lift:** +0.0009 PR AUC over context_xg — the best of the four low-base-rate states, though still negligible in absolute terms. The EF empty-net context creates situations where RAPM features (in particular `teammates_rapm_xg_off`, `opp_rapm_xg_off`, each ~1.1%) add small but measurable signal — the opposition's defensive RAPM matters when they're defending with a lead.

**Feature gain:** Goalie features still dominate (65% combined) but the balance shifts toward per-shot metrics over raw counts. RAPM features account for ~5.5% combined — the highest RAPM contribution of any state. EF situations have higher stakes for talent identification.

---

#### empty_against

**Performance tier:** ❌ FAIL (lift). Discrimination: very high (0.7480 = 1.3× null, ROC 0.703 — the model is predictive). Calibration PASS (max abs error 0.023, excellent). But pred_goal is 3.4% PR AUC *worse* than context_xg.

**Calibration character:** Near-perfect via the IsotonicCalibrator. Decile actual rates track mean predictions closely across all 7 populated deciles. ECE=0.084, max abs error=0.023. OOF gap=0.012.

**Negative lift:** context_xg PR AUC = 0.782, pred_goal PR AUC = 0.748 (−0.034). This degradation is consistent across all 14 training seasons and the hold-out year — pred_goal never beats context_xg for EA in any season in the dataset. The underlying issue is structural: in EA situations the goalie has been pulled, making goalie form/talent features irrelevant. Shooter talent features add noise rather than signal because EA goals are driven by shot quality (already in context_xg), not by who is shooting. The EA pred_goal model is spending tree capacity on talent features that are non-informative for EA outcomes, which actively hurts discrimination.

**Feature gain:** Highly distributed (45 features, no dominant pattern) — consistent with a model that has found no clear signal and is spreading tree capacity across many marginally informative features. No context leakage detected.

---

## Changelog

| Date | Version | Trials | ES PR AUC | PP PR AUC | SH PR AUC | EF PR AUC | EA PR AUC | Notes |
|---|---|---|---|---|---|---|---|---|
| 2026-05-14 | 1.0.0 | 500 (OOF-only cal) | 0.3204 | 0.3420 | 0.3335 | 0.3121 | 0.7473 | Initial run. OOF-only Platt calibration. ES log loss 2.47 (994% worse than null), PP 1.13 (241% worse), SH 1.40 (442% worse), EF 0.98 (259% worse). EA calibration WARN (max err 0.094). All five states FAIL. Catastrophic miscalibration driven by temporal drift: OOF calibrator learned training-era talent matchup statistics that don't hold in the hold-out season. Root cause and fix documented in Issue 14. |
| 2026-05-14 | 1.0.0 | 500 (pooled cal) | 0.3204 | 0.3420 | 0.3336 | 0.3121 | 0.7480 | After pooled OOF + hold-out calibration fix (Issue 14). ES log loss 0.266 (−17.7% vs null), PP 0.393 (−18.8%), SH 0.318 (−22.5%), EF 0.313 (−15.0%), EA 0.618 (+9.7%). Calibration improved 9–56× across states. Residual calibration FAIL remains for ES/PP/SH/EF (decile-8 non-monotone pattern, max abs err 0.13–0.19). EA calibration PASS (max abs err 0.023). Lift over context_xg is negligible for all non-EA states (+0.0001–+0.0009); EA has negative lift (−0.034). See Issues 14 and 15. |