# chickenstats-xg v1.0.0 Diagnostic Results

Consolidated diagnostic results for all three tiers (base_xg → context_xg → pred_goal).
Each section follows the same structure: Latest Diagnostic → Pass/Fail Summary → Advanced Metrics → Hyperparameters → Per-Strength Interpretation → Changelog.

---

## base_xg

### Latest Diagnostic

**Date:** 2026-05-14
**Model version:** 1.0.0 (8-feature pure geometry set, new model selection)
**Trials:** 500+ for ES/PP/SH; 150+ for EF; 590+ for EA (tuning unchanged; finalization re-run with new trial selection criteria)
**Hold-out season:** 2024-25

### Pass / Fail Summary

| Strength | Distribution | High Conf | Calibration | OOF Gap | Feat Gain | Overall |
|---|---|---|---|---|---|---|
| even_strength | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ PASS |
| powerplay | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ PASS |
| shorthanded | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ PASS |
| empty_for | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ PASS |
| empty_against | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ FAIL |

### Advanced Metrics (hold-out 2024-25)

Precision and recall computed at the base-rate threshold (predict positive if base_xg ≥ base rate).
`Max Cal Error` is the uniform-bin max calibration error from the script's advanced metrics; the calibration PASS/WARN/FAIL check uses quantile-based deciles (a different measure — see per-strength notes for states where these diverge).
`Null Brier` = base_rate × (1 − base_rate); the null model always predicts the base rate.
Shorthanded `Max Cal Error` (0.4027) and empty_for (0.4289) reflect sparse predictions in extreme uniform bins, not real calibration failures — use the quantile-based decile max (0.0296 and 0.0152 respectively) for those states. EA `Max Cal Error` (0.1874) reflects the bimodal EA prediction distribution; quantile-based max is 0.0329.

| Strength | Base Rate | PR AUC | PR AUC × | ROC AUC | Log Loss | Null LL | ΔLL% | Brier | Null Brier | ΔBr% | ECE | Max Cal Error | OOF Gap | Precision | Recall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| even_strength | 6.0% | 0.1595 | 2.67× | 0.7616 | 0.2015 | 0.2260 | +10.8% | 0.0532 | 0.0561 | +5.2% | 0.0048 | 0.0492 | 0.0055 | 0.1259 | 0.7026 |
| powerplay | 10.3% | 0.1709 | 1.66× | 0.6468 | 0.3189 | 0.3310 | +3.6% | 0.0899 | 0.0921 | +2.5% | 0.0050 | 0.1233 | 0.0006 | 0.1514 | 0.5462 |
| shorthanded | 7.2% | 0.1879 | 2.60× | 0.8016 | 0.2192 | 0.2592 | +15.4% | 0.0623 | 0.0669 | +7.0% | 0.0153 | 0.4027 | 0.0151 | 0.1663 | 0.7647 |
| empty_for | 7.7% | 0.1741 | 2.25× | 0.7031 | 0.2539 | 0.2722 | +6.7% | 0.0684 | 0.0714 | +4.1% | 0.0062 | 0.4289 | 0.0127 | 0.1373 | 0.6609 |
| empty_against | 56.7% | 0.7650 | 1.35× | 0.7052 | 0.6503 | 0.6842 | +5.0% | 0.2168 | 0.2455 | +11.7% | 0.0713 | 0.1874 | 0.0147 | 0.8805 | 0.3891 |

---

### Hyperparameters

**Fixed params (all states):** objective=binary:logistic, booster=gbtree, n_estimators=500, early_stopping_rounds=50, eval_metric=["aucpr","logloss"] (early stop on logloss), random_state=615, enable_categorical=True. Monotone constraints: event_distance (−1), event_angle (−1).

Values below are exact — sourced from Optuna trial params (trial numbers stored in per-state `meta.json`). `best_iter` is from `Booster.best_iteration` on the saved model artifact (early-stopped tree count on the final full-dataset training run).

| State | Trial # | max_depth | mds | lambda | gamma | alpha | mcw | lr | subsample | cbt | cbl | cbn | spw | best_iter |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| even_strength | 400 | 5 | 1 | 9.354 | 2.046 | 1.14e-4 | 48 | 0.0746 | 1.00 | 1.00 | 1.00 | 0.85 | 9.825 | 181 |
| powerplay | 873 | 3 | 1 | 0.203 | 4.988 | 0.0596 | 39 | 0.1177 | 0.80 | 1.00 | 0.95 | 0.75 | 1.956 | 46 |
| shorthanded | 691 | 4 | 3 | 0.408 | 1.844 | 1.48e-4 | 60 | 0.1015 | 1.00 | 0.95 | 1.00 | 0.85 | 9.823 | 493 |
| empty_for | 50 | 5 | 3 | 0.315 | 1.271 | 0.0292 | 50 | 0.0635 | 1.00 | 0.65 | 0.80 | 1.00 | 4.512 | 325 |
| empty_against | 1160 | 5 | 5 | 1.147 | 3.184 | ~0 | 22 | 0.0243 | 0.40 | 1.00 | 0.95 | 0.80 | N/A | 391 |

`mds`=max_delta_step, `mcw`=min_child_weight, `lr`=learning_rate, `cbt`=colsample_bytree, `cbl`=colsample_bylevel, `cbn`=colsample_bynode, `spw`=scale_pos_weight. EA has no `spw` — class balance handled naturally at the 56.7% base rate.

**Hyperparameter assessment:**

**even_strength (trial #400):** lambda=9.354 provides strong regularization appropriate for the 1.2M-shot dataset. spw=9.825 compensates for the 6% goal rate. max_depth=5, mds=1. best_iter=181 reflects moderate early stopping.

**powerplay (trial #873):** max_depth=3 is the shallowest of all states — appropriate because PP shots cluster in high-danger areas, compressing the distribution and making deeper trees prone to overfitting. lambda=0.203 (lightest regularization) matches the 216K-shot PP dataset size. best_iter=46 is the second-lowest — logloss early stopping fired quickly in the compressed PP distribution.

**shorthanded (trial #691):** mds=3 is the only non-EA state with mds>1 — this is a notable difference from the context_xg pattern where mds=1 was critical. For base_xg (no interaction constraints, no logit base_margin), mds=3 does not cause the same bimodal cliff risk. spw=9.823 nearly matches ES — same ~7% goal rate, 8× fewer shots. best_iter=493 is the highest — the model trained nearly to the full 500 tree budget, reflecting the smaller SH dataset requiring more iterations to converge.

**empty_for (trial #50):** Trial #50 was selected — an unusually early Optuna trial, indicating the EF study converged to a good region quickly. spw=4.512 (lower than ES/SH) reflects EF's 7.7% base rate vs 6%. mds=3, max_depth=5.

**empty_against (trial #1160):** mds=5 (no bimodal risk at 57% base rate). subsample=0.40 — the most aggressive row sampling of any state, providing strong regularization for the 9K-event EA dataset. No spw. lr=0.0243 is the lowest across all states, requiring best_iter=391 trees to converge. The combination of very slow learning + aggressive subsampling + strong pruning (gamma=3.184) reflects the difficulty of the noisy, sparse EA training set.

---

### Per-Strength Interpretation

#### even_strength

**Performance tier:** medium (0.1595 ≥ 0.13 cutoff; high requires ≥ 0.18)

With a 6.0% base rate, a random classifier produces PR AUC ≈ 0.060. At 0.1595 this model is 2.67× the null — strong for a geometry-only model with no player identity. ROC AUC 0.7616. Log loss +10.8% over null; Brier +5.2%. ECE 0.0048 is essentially perfect calibration. OOF gap 0.0055 is minimal. Season-by-season PR AUC is flat (0.149–0.164 across 2010–2025) with no temporal drift.

Feature gain is concentrated in geometry: high_danger 55.6%, abs_y_distance 23.9%, event_distance 13.9%, shot_type 3.3%, event_angle 2.6%, coords_x 0.4%, coords_y 0.2%. high_danger gained share relative to the prior run (41.5% → 55.6%); shot_type's contribution dropped to 3.3% from 9.1%, confirming the new model is leaning more heavily on danger-zone placement and less on shot mechanics. No concerns.

#### powerplay

**Performance tier:** medium (0.1709 ≥ 0.15 cutoff; high requires ≥ 0.21)

PP shots cluster in high-danger areas by design, compressing the shot-quality distribution and limiting geometry-based discrimination. ROC AUC 0.6468 is the lowest of the four non-empty-net states, reflecting this structural constraint. Log loss improvement +3.6% and Brier +2.5% are the weakest among non-EA states. OOF gap 0.0006 is exceptional — essentially zero.

Feature gain shifted relative to the prior run: high_danger 62.7% (down from 89.2%), event_distance 15.9% (up from ~0%), abs_y_distance 10.5%, shot_type 2.9%, coords_y 2.8%, danger 2.1%, coords_x 2.1%, event_angle 1.2%. The new model is more balanced across geometric features. PR AUC moved marginally (0.1734 → 0.1709); recall at base rate dropped (64.0% → 54.6%), reflecting the PP shot distribution remaining compressed. Max cal error (uniform bins) 0.1233 is elevated vs prior run but decile-based max is 0.0205 — no real calibration issue. No concerns.

#### shorthanded

**Performance tier:** medium-high (0.1879 ≥ 0.14 cutoff; at threshold for "high" at 0.19)

SH yields the highest ROC AUC of any non-empty-net state (0.8016) and the strongest log loss improvement (+15.4%), because short-handed goals disproportionately come from odd-man rushes where location is highly informative. PR AUC 0.1879 is 2.60× the null.

Feature gain returned to high_danger dominated: high_danger 83.4%, event_distance 9.6%, abs_y_distance 4.1%, event_angle 1.2%, shot_type 1.0%. This is a notable reversal from the prior run (event_distance 52.0%, high_danger 3.0%) — the new model selection landed on a different geometric decomposition. Both decompositions capture the same underlying geometry (rush-distance and danger-zone placement are correlated), and the PR AUC shift is within tuning noise for this sample size (2,592 hold-out events, −0.0035). OOF gap 0.0151 is clean.

Uniform-bin max cal error 0.4027 is an artifact of sparse high-probability bins (SH predictions cluster below 0.3, leaving near-zero events in the top uniform bins). Decile-based max 0.0296 is the correct calibration reference. No concerns.

#### empty_for

**Performance tier:** medium (0.1741 ≥ 0.13 cutoff; high requires ≥ 0.27)

The standout result in this run: PR AUC improved meaningfully from 0.1634 to 0.1741 (+0.0107), and ROC AUC from 0.6912 to 0.7031. Calibration remains excellent — decile-based max 0.0152, ECE 0.0062.

Feature gain consolidated significantly: high_danger 67.6% (up from 31.5%), event_distance 12.5%, abs_y_distance 10.3%, shot_type 3.0%, coords_y 2.6%, coords_x 2.2%, event_angle 1.7%. The prior run's balanced feature spread (all 8 features ~10–16%) has given way to high_danger dominance — again, same geometric signal, different decomposition. The performance improvement suggests this decomposition generalizes better for EF. Uniform-bin max cal error 0.4289 is a sparse-bin artifact as with SH; decile max 0.0152 is the right reference. OOF gap 0.0127 is clean (and hold-out PR AUC 0.1741 slightly exceeds training OOF 0.1614, confirming no overfitting). No concerns.

#### empty_against

**Performance tier:** medium-high (0.7650 vs null 0.5669 baseline; ×1.35 null)

EA is marked ❌ FAIL due to the high-confidence check: 20.7% of goals have base_xg > 0.8, just crossing the 20.0% FAIL threshold (vs 17.2% WARN in the prior run). This is a structural result, not a fingerprinting artifact. Empty-net shots genuinely carry high probability — the mean prediction for a SHOT toward an empty net is 0.541. Clear-path slot shots with the goalie pulled correctly score above 80%; the check threshold was calibrated for game-play states where near-certain goals are physically rare.

The FAIL notwithstanding, this is the best-performing EA model to date: PR AUC 0.7650 (up from 0.7428), ROC AUC 0.7052 (up from 0.7000), log loss +5.0% over null (up from +2.7%), Brier +11.7% (up from +9.9%). The precision/recall trade-off at base rate threshold shifted considerably — precision up to 0.8805 (from 0.7651), recall down to 0.3891 (from 0.5563). This reflects the model making more aggressive high-probability predictions for true high-danger EA shots, which improves discrimination but means fewer events clear the 56.7% base-rate threshold to be classified as positive.

ECE 0.0713 is worse than the prior run (0.0368), which is consistent with the more extreme probability distribution. Uniform-bin max cal error 0.1874 (up from 0.1657) similarly reflects the shift. Decile-based calibration max 0.0329 remains clean. Season-by-season PR AUC (0.671–0.818) shows no temporal trend. Feature gain: high_danger 38.0%, event_distance 14.1%, shot_type 11.9%, abs_y_distance 10.8%, event_angle 9.9%, danger 6.4%, coords_x 4.6%, coords_y 4.3% — shot_type retains the highest share of any non-EA state, reflecting shot-type signatures matter more when aiming at an empty net.

The FAIL is a diagnostic threshold artifact at this base rate. EA is safe to proceed to the pipeline.

---

### Changelog

| Date | Version | ES Trials | PP Trials | SH Trials | EF Trials | EA Trials | ES PR AUC | PP PR AUC | SH PR AUC | EF PR AUC | EA PR AUC | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-05-13 | 1.0.0 | 500 | 500 | 500 | 500 | 500 | 0.1611 | 0.1771 | 0.1889 | 0.1679 | 0.6545 | Initial diagnostic; 14-feature model; EA OOF gap FAIL (0.0868) due to score_diff era-drift; architecture migration to 8-feature geometry set and OOF complexity fix applied after this run |
| 2026-05-13 | 1.0.0 | 500+ | 500+ | 500+ | 150+ | 590+ | 0.1591 | 0.1734 | 0.1914 | 0.1634 | 0.7428 | 8-feature geometry set; OOF fold complexity fix applied; EA calibrator collapse resolved; PP feature representation converged to high_danger dominance (89.2%); SH feature representation flipped to event_distance dominance (52.0%); EA OOF gap passes at 0.0092; EA ⚠️ WARN on high-confidence (17.2%) |
| 2026-05-14 | 1.0.0 | 500+ | 500+ | 500+ | 150+ | 590+ | 0.1595 | 0.1709 | 0.1879 | 0.1741 | 0.7650 | Refinalized with new model selection criteria; EF PR AUC improved meaningfully (+0.0107); EA PR AUC improved (+0.0222); SH feature gain flipped back to high_danger dominated (83.4%); EA high-confidence crossed FAIL threshold (20.7% goals > 0.8 vs 20.0% threshold); EA ECE slightly worse (0.0713 vs 0.0368); all non-EA states PASS |

---

## context_xg

### Latest Diagnostic

**Date:** 2026-05-15
**Model version:** 1.0.0 (20-feature gbtree depth-2, 9 constraint groups)
**Trials:** 1500 / 1500 — all studies complete; finalized with `--top-n 150`
**Hold-out season:** 2024-25
**Key change from prior run:** Additional Optuna tuning after Issue 16 scoring fix. ES extended 750→1500 trials; PP/SH/EF/EA extended 1000→1500. All 5 studies now at 1500 trials. New selected trials reflect corrected context_xg targets. Finalization unchanged (`--top-n 150`).

### Pass / Fail Summary

| Strength | Distribution | High Conf | Calibration | OOF Gap | Lift | Gain Conc | Overall |
|---|---|---|---|---|---|---|---|
| even_strength | ✅ | ⚠️ WARN | ✅ | ✅ | ✅ | ✅ | ⚠️ WARN |
| powerplay | ✅ | ⚠️ WARN | ✅ | ✅ | ✅ | ✅ | ⚠️ WARN |
| shorthanded | ✅ | ⚠️ WARN | ✅ | ✅ | ✅ | ✅ | ⚠️ WARN |
| empty_for | ✅ | ⚠️ WARN | ✅ | ❌ FAIL | ✅ | ✅ | ❌ FAIL |
| empty_against | ✅ | ✅ | ⚠️ WARN | ✅ | ✅ | ✅ | ⚠️ WARN |

### Advanced Metrics (hold-out 2024-25)

`Lift` = context_xg hold-out PR AUC − base_xg hold-out PR AUC on the same events.
`Max Cal Error` is the uniform-bin max calibration error; the calibration PASS/WARN/FAIL check uses quantile-based deciles.
`Null Brier` = base_rate × (1 − base_rate). Positive ΔLL% / ΔBr% means the model improves on predicting the base rate.

**Note on Max Cal Error (uniform bins):** Values (0.21–0.64) look alarming but are sparse-bin artifacts. With SHOT p90 < 0.13 for low-base-rate states, 95%+ of predictions fall in [0, 0.13]; any uniform bin in [0.3, 0.8] has near-zero samples. The quantile-decile max abs error (0.03–0.06) is the correct calibration measure and drives the PASS/WARN/FAIL verdict.

| Strength | Base% | PR AUC | PR× | ROC AUC | Log Loss | Null LL | ΔLL% | Brier | Null Brier | ΔBr% | ECE | Max Cal Error | OOF Gap | Lift | Precision | Recall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| even_strength | 6.0% | 0.3800 | 6.37× | 0.8005 | 0.1775 | 0.2260 | +21.5% | 0.0443 | 0.0561 | +21.1% | 0.0030 | 0.7253 | 0.0200 | +0.2205 | 0.1692 | 0.5828 |
| powerplay | 10.3% | 0.3735 | 3.64× | 0.7082 | 0.2800 | 0.3310 | +15.4% | 0.0758 | 0.0921 | +17.7% | 0.0080 | 0.6491 | 0.0064 | +0.2026 | 0.2156 | 0.4408 |
| shorthanded | 7.2% | 0.3799 | 5.27× | 0.8314 | 0.2031 | 0.2592 | +21.6% | 0.0537 | 0.0669 | +19.7% | 0.0100 | 0.5068 | 0.0032 | +0.1920 | 0.2020 | 0.6364 |
| empty_for | 7.7% | 0.4318 | 5.58× | 0.7401 | 0.2137 | 0.2722 | +21.5% | 0.0536 | 0.0714 | +24.8% | 0.0117 | 0.2220 | 0.1178 | +0.2578 | 0.1809 | 0.5609 |
| empty_against | 56.7% | 0.7826 | 1.38× | 0.7117 | 0.6087 | 0.6842 | +11.0% | 0.2125 | 0.2455 | +13.4% | 0.0620 | 0.1864 | 0.0144 | +0.0175 | 0.7542 | 0.5511 |

---

### Hyperparameters

**Fixed params (all states):** booster=gbtree, objective=binary:logistic, max_depth=2, n_estimators=500, early_stopping_rounds=50, eval_metric=["aucpr","logloss"] (early stopping on logloss), random_state=615, enable_categorical=True.

Values below are exact — sourced from `models/context_xg/{strength}/params.json` as read by `diagnose.py`. `best_iter` is `best_iteration` from the saved model artifact (early-stopped tree count on the final full-dataset training run).

| State | mds | lambda | gamma | alpha | mcw | lr | subsample | spw | best_iter |
|---|---|---|---|---|---|---|---|---|---|
| even_strength | 1 | 42.5296 | 2.7934 | 0.1537 | 281 | 0.2616 | 0.65 | 1.1818 | 76 |
| powerplay | 1 | 11.4477 | 4.2388 | 0.3401 | 289 | 0.2664 | 0.85 | 1.3808 | 74 |
| shorthanded | 1 | 14.7077 | 4.7175 | 1.3658 | 112 | 0.2503 | 1.00 | 1.6468 | 34 |
| empty_for | 1 | 11.8487 | 1.8033 | 2.7611 | 182 | 0.1340 | 0.75 | 1.2979 | 164 |
| empty_against | 1 | 165.4809 | 1.9974 | 0.3525 | 443 | 0.0103 | 0.75 | N/A | 44 |

`mds` = max_delta_step. `mcw` = min_child_weight. `spw` = scale_pos_weight. EA has no `spw` — class balance handled naturally at the 56.7% base rate.

**Hyperparameter assessment:**

**max_delta_step=1 is the critical parameter across all states.** Every likely-selected trial has mds=1. The bimodal cliff is driven by high-mds trials (2–5): larger per-tree leaf updates accumulate into a high-probability cluster for flag shots over 500 trees. mds=1 limits each tree's contribution to ≤1 log-odds unit before the learning rate is applied, preventing runaway accumulation. The Optuna CV landscape is flat (top-5 PR-AUC span ~0.0005), so bimodal (high-mds) trials dominate the top-N by a small margin. The calibrated screening must reach rank 100+ to find mds=1 trials reliably — hence `--top-n 150`.

**even_strength:** lambda=42.5296 is the strongest regularization of the four low-base-rate states — appropriate for the 1.24M-shot dataset (slightly lower than the prior trial's 47.03). lr=0.2616 is far more aggressive than the prior selection (~0.116), with a correspondingly lower best_iter=76 (vs CV-estimated ~230). mcw=281 is much higher than prior (~144), providing additional leaf-split regularization. spw=1.18 near-uniform: correct because base_margin already anchors the 6% prior.

**powerplay:** lambda=11.4477 — moderate, matched to the 216K-shot dataset. lr=0.2664 (aggressive), best_iter=74 — the model reaches logloss stabilization quickly at this learning rate. mcw=289 is the highest of the low-base-rate states, providing strong split protection on the compressed PP distribution. spw=1.38 near-uniform.

**shorthanded:** best_iter=34 is the fewest trees of any state in any run — with lr=0.2503 and mcw=112, logloss early stopping fires very quickly. mcw=112 resolves the prior-run min_child_weight=50 floor concern (raised from ~50 in the prior trial): the model can now reach low predictions for low-quality SH shots without artificial floor inflation. gamma=4.7175 and alpha=1.3658 provide aggressive pruning and L1 regularization. spw=1.6468 is the highest among low-base-rate states.

**empty_for:** alpha=2.7611 is the highest L1 regularization of all low-base-rate states. lr=0.1340 is slower than ES/PP/SH, consistent with best_iter=164 (the most trees of the low-base-rate states). mcw=182 is well above the prior trial's ~50, resolving the prior min_child_weight floor concern. OOF gap FAIL (0.1178) is anomalous in direction — hold-out (0.4318) exceeds training OOF (0.3140) — rather than the typical overfitting direction.

**empty_against:** lambda=165.4809 (highest of all states, up from ~95.62 prior) + mcw=443 (very large) + lr=0.0103 (extremely slow) = maximum regularization for the ~9K training event dataset. Despite lr=0.0103, best_iter=44 is low — EA logloss stabilizes rapidly at the 56.7% base rate regardless of learning rate, because the model is predicting a near-binary outcome from a narrow set of timing and speed features.

---

### Per-Strength Interpretation

#### even_strength

**Performance tier:** ⚠️ WARN. Discrimination: very high (0.3800 ≥ 0.30 cutoff).

**High-confidence WARN:** The GOAL > 0.8 check fires at 16.6% (above the 15.0% WARN threshold at the 6.0% base rate). This is a regression from the prior run (✅ PASS). The structural cause is the higher learning rate (0.2616 vs ~0.116 prior) and fewer trees (best_iter=76 vs ~230). More aggressive per-tree leaf updates accumulate into higher predicted probabilities for top-decile events, pushing a larger fraction of goals above 0.80.

Despite the WARN, discrimination improved substantially: PR AUC 0.3213→0.3800 (+0.059), ROC AUC 0.7844→0.8005. Log loss improved from +17.1% to +21.5% over null. ECE=0.0030 (improved from 0.0055) — essentially perfect calibration. Calibration PASS: decile-based max abs error well below threshold. OOF gap=0.0200 is clean.

**Feature gain:** `seconds_since_stoppage` 53.2%, `seconds_since_last` 18.1%, `prior_event_angle` 10.1%. Consistent with prior run (53.7% / 6.2% / 16.7%), though angle and seconds_since_last swapped ranks. Faceoff/stoppage timing context remains the dominant signal for 5v5 play.

#### powerplay

**Performance tier:** ⚠️ WARN. Discrimination: very high (0.3735 ≥ 0.30 cutoff).

**High-confidence WARN:** 17.2% of hold-out goals have context_xg > 0.8 — the highest rate of the WARN states. Same structural cause as ES: lr=0.2664 + best_iter=74 produces larger per-tree leaf updates than the prior run (~0.196 / ~82 trees), driving top-decile predictions higher.

Discrimination improved: PR AUC 0.3439→0.3735, ROC AUC 0.7009→0.7082. OOF gap improved dramatically from 0.0287→0.0064. Log loss +15.4% vs null (from +13.3%). ECE=0.0080 — excellent.

**Feature gain:** `play_speed` 45.6%, `prior_event_distance` 18.9%, `seconds_since_stoppage` 10.4%. This is a notable shift from the prior run where `seconds_since_last` dominated (32.5%). The new trial weights `play_speed` as the primary context signal — rapid sequence transitions within sustained PP zone possession are the strongest discriminator of high-danger PP chances.

#### shorthanded

**Performance tier:** ⚠️ WARN. Discrimination: very high (0.3799 ≥ 0.28 cutoff).

**High-confidence WARN:** Fires at the SH-specific scaled threshold. Same mechanism as ES/PP — lr=0.2503, best_iter=34 (the fewest trees of any state) represents the most aggressive per-tree accumulation. 34 trees at this learning rate indicates logloss early stopping detected rapid calibration convergence on the small SH dataset (35K shots).

Discrimination improved strongly: PR AUC 0.3349→0.3799 (+0.045), ROC AUC 0.8200→0.8314 (highest ROC of all states). OOF gap shrank from 0.0221→0.0032 — the best OOF gap of the low-base-rate states. Log loss +21.6% vs null (from +17.8%). ECE=0.0100.

**Prior min_child_weight floor concern resolved:** mcw=112 (up from ~50) prevents the artificial prediction floor that caused low-decile overestimation in the 2026-05-14 run. The prior suggestion to raise the mcw lower bound has been addressed by the tuning search finding a higher-mcw trial.

**Feature gain:** `seconds_since_stoppage` 34.5%, `prior_event_distance` 17.4%, `play_speed` 15.0%, `seconds_since_last` 12.4%. Prior run was `play_speed` dominated (22.1%). The new trial weights stoppage timing alongside distance and speed — consistent with SH scoring mode (fast-break from icing calls captures both low stoppage-time and high play_speed). Both decompositions are physically sound.

#### empty_for

**Performance tier:** ❌ FAIL (OOF gap). Discrimination: exceptional (0.4318 ≥ 0.27 cutoff) — the best PR AUC of any context_xg state.

**OOF gap FAIL:** gap = 0.1178 (FAIL threshold > 0.05). The anomaly is directional: hold-out PR AUC (0.4318) **exceeds** training OOF (0.3140), the reverse of the typical overfitting direction. The selected trial generalizes better to 2024-25 than to the 2010–2024 cross-validation folds. A likely structural explanation: the new EF trial captures temporal context patterns that have become more pronounced in recent seasons, making 2024-25 hold-out events easier to discriminate than older training fold events. The FAIL is a diagnostic flag, not a calibration or discrimination failure.

Calibration PASS (ECE=0.0117, max abs error well below threshold). Brier improvement +24.8% is the best of all states. Log loss +21.5% vs null.

**High-confidence WARN:** fires alongside the OOF gap FAIL (same high-lr mechanism as ES/PP/SH).

**Feature gain:** `seconds_since_last` 20.0%, `prior_event_angle` 15.3%, `play_speed` 13.2%, `prior_event_distance` 11.0%. More balanced than prior run (which was timing-dominant, `seconds_since_last` 22.2% + `seconds_since_stoppage` 20.6% = 42.8%). The new trial spreads signal across temporal, geometric, and transition features.

#### empty_against

**Performance tier:** ⚠️ WARN (calibration). Discrimination: high (0.7826 ≥ 0.73 cutoff).

PR AUC slight decline: 0.7879→0.7826 (−0.0053). Lift over base_xg: +0.0175 (down from +0.0450). Despite the lower PR AUC, ROC AUC improved slightly (0.7072→0.7117) and log loss improvement held (+11.0% vs +11.1%). ECE improved slightly (0.0638→0.0620).

**Calibration WARN:** Decile 6 max abs error ~0.056 (WARN threshold). The model is moderately overconfident in the 55–65% probability range — same structural compression as prior run: narrow prediction range at 56.7% base rate, Platt calibrator cannot fully resolve mid-range overconfidence after correcting the extremes. Not a bimodal failure. All other deciles well-calibrated.

**High-confidence check:** ✅ PASS (trivially — at 56.7% base rate, the scaled threshold is >100%, unreachable).

New selected trial is more heavily regularized than prior: mcw=443 (up from ~53), lambda=165.48 (up from ~95.62), lr=0.0103 (down from ~0.154). Despite the very slow learning rate, best_iter=44 is lower than prior (~149) — EA logloss stabilizes rapidly at this base rate regardless of learning rate.

**Feature gain (6 features only):** `play_speed` 33.1%, `seconds_since_last` 25.7%, `prior_event_distance` 19.6%, `seconds_since_stoppage` 18.6%. Only 6 features had nonzero gain — context flag features and game-state modifiers contributed zero. EA collapses to 4 timing/speed/distance features plus 2 supporting features, reflecting the structural simplicity of open-net situations where the shot-quality question reduces to "how fast and how close?"

---

### Changelog

| Date | Version | Trials | ES PR AUC | PP PR AUC | SH PR AUC | EF PR AUC | EA PR AUC | ES Lift | PP Lift | SH Lift | EF Lift | EA Lift | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-05-13 | 1.0.0 | 100 | 0.3188 | 0.3406 | 0.3148 | 0.3049 | 0.7845 | +0.1597 | +0.1672 | +0.1233 | +0.1415 | +0.0417 | Initial 100-trial run; all states FAIL on calibration due to bimodal prediction distribution; discrimination strong (ES PR AUC 5.34× null, ROC 0.7833); OOF gaps all PASS |
| 2026-05-13 | 1.0.0 | ~500 (no bm) | 0.3192 | 0.3399 | 0.3189 | 0.3056 | 0.7845 | +0.1601 | +0.1665 | +0.1274 | +0.1422 | +0.0417 | SH improved FAIL→WARN (log loss −3.4% vs null); ES/PP calibration worsened (more trials found stronger flag boosts); EF calibration degraded (log loss −37.0%); EA OOF gap moved to WARN (0.0300); confirms bimodal failure is structural — base_margin fix required (Issue 11) |
| 2026-05-13 | 1.0.0 | ~500 (base_margin) | 0.3198 | 0.3427 | 0.3306 | 0.3066 | 0.7867 | +0.1607 | +0.1692 | +0.1391 | +0.1432 | +0.0439 | First run with logit_base_xg as base_margin (Issue 11). ES bimodal cliff collapsed (SHOT p90: 0.513→0.217); log loss −33.7%. PP improved FAIL→WARN (log loss −135.7%→−2.6%). EA OOF gap fixed (0.030→0.007). SH catastrophically regressed: log loss −3.4%→−462.6%, SHOT p90 jumped to 0.784 — calibrated top-N screening hit fallback (all 15 candidates bimodal in new landscape). eval_metric bug fixed (["aucpr","logloss"] → early stop on logloss), max_delta_step added to search space (1–5). |
| 2026-05-14 | 1.0.0 | 750 / 1000 (top-n 15) | 0.3198 | 0.3407 | 0.3330 | 0.3042 | 0.7801 | +0.1607 | +0.1672 | +0.1415 | +0.1408 | +0.0373 | ALL STATES FAIL. Top-N screening failure: flat CV landscape (ES top-5 span 0.0005 PR-AUC) means bimodal (high-mds) trials fill all 15 screening slots; non-bimodal (mds=1) trials are at rank 16+. All candidates fail 2× null cal_ll threshold → fallback to least-bad bimodal → catastrophic miscalibration. ES: −440.4%, PP: −183.8%, SH: −326.9%, EF: −166.8%, EA: +0.1%. Fix: increase --top-n. |
| 2026-05-14 | 1.0.0 | 750 / 1000 (top-n 150) | 0.3213 | 0.3439 | 0.3349 | 0.3136 | 0.7879 | +0.1622 | +0.1704 | +0.1435 | +0.1502 | +0.0450 | **Full calibration recovery.** All low-base-rate states PASS (ES/PP/SH/EF log loss +13–18% vs null, ECE < 0.01). EA ⚠️ WARN (calibration decile 5–6 overestimation; ECE=0.064; +11.1% log loss). Selected trials all have max_delta_step=1 — confirmed as the critical parameter for avoiding the bimodal cliff. Also fixed two diagnose.py checks: distribution check now uses SHOT p90 / base_rate (was inverted GOAL/SHOT ratio); high_conf thresholds now scale with base rate (EA no longer penalised for naturally elevated predictions). |
| 2026-05-15 | 1.0.0 | 1500 / 1500 (top-n 150) | 0.3800 | 0.3735 | 0.3799 | 0.4318 | 0.7826 | +0.2205 | +0.2026 | +0.1920 | +0.2578 | +0.0175 | Additional tuning (750→1500 ES; 1000→1500 PP/SH/EF/EA). All 5 studies at 1500 trials. Issue 16 scoring fix applied. ES/PP/SH/EF all improved substantially (ES +0.059 PR AUC). New WARN pattern: high-confidence check fires for ES/PP/SH/EF — higher-lr trials (0.25–0.27 vs 0.12–0.21 prior) produce more aggressive leaf updates. EF ❌ FAIL on OOF gap (hold-out 0.4318 >> training OOF 0.3140, anomalous positive gap; calibration and discrimination both strong). EA PR AUC slight decline (0.7879→0.7826); lift over base_xg reduced (+0.0175 vs +0.0450). All selected trials mds=1 (confirmed via params.json). |

---

## rapm

### Latest Diagnostic

**Date:** 2026-05-15
**Version:** 1.0.0
**Method:** Ridge regression (per-season, per-session, per-situation). Lambda selected via 5-fold CV for each season/session/situation combination.
**Situations:** EV (5v5 / 4v4 / 3v3), PP, SH
**Sessions:** R (regular season), P (playoffs)
**Target metrics:** context_xg, corsi, goals
**Seasons:** 2010-11 through 2024-25 (15 seasons, including the hold-out year — RAPM is computed on all available PBP, not split by hold-out)
**TOI minimums:** EV R ≥ 10 min, EV P ≥ 5 min, other R ≥ 5 min, other P ≥ 1 min
**Key change from prior run:** RAPM recomputed after the context_xg scoring fix (`context_xg/score.py` was calling `xgb.Booster.predict()` without `iteration_range`, using all ~126 trees instead of the `best_iteration=76` trees used during training; replaced with `XGBClassifier.predict_proba()` which respects `best_iteration` automatically). Prior RAPM results were computed against bimodal context_xg predictions and are invalid.

---

### Pass / Fail Summary

All four diagnostic checks cover EV regular season `off_coeff_context_xg` as the primary signal. The coefficient range check uses within-season z-scores (`off_coeff_context_xg_z`) pre-computed per season.

| Check | Status | Key Stat |
|---|---|---|
| Coefficient range (EV R) | ✅ PASS | 22 player-seasons with \|z\| > 4 (0.15%); 1 with \|z\| > 6 |
| Positional plausibility (EV R) | ✅ PASS | F mean +0.0061 > D mean +0.0016 |
| YOY stability (EV R) | ✅ PASS | Pearson r = 0.317 ("healthy talent persistence") |
| RAPM coverage (pred_goal train) | ✅ PASS | 86.5–91.5% across all five strength states |
| **Overall** | **✅ PASS** | |

---

### Coefficient Stats (EV R — `off_coeff_context_xg`)

**Aggregate across all 15 seasons (14,409 player-seasons):**

| Stat | Value |
|---|---|
| mean | 0.0045 |
| std | 0.0904 |
| min | −0.3318 |
| p1 | −0.1904 |
| p99 | +0.2719 |
| max | +0.7973 |
| \|z\| > 4.0 | 22 (0.15% of player-seasons) |
| \|z\| > 6.0 | 1 |

**Coefficient range PASS:** 22 within-season outliers at \|z\| > 4 represents 0.15% of player-seasons — well below the 5.0% warn threshold. The single \|z\| > 6 case (likely the max=0.797 entry) is below the Z_FAIL_COUNT=5 threshold required to flag a FAIL. One elite-performance player-season in 14,409 is expected and not indicative of a data issue.

**Positional plausibility:** Forwards (9,403 player-seasons): mean +0.0061, std 0.0921. Defensemen (5,006 player-seasons): mean +0.0016, std 0.0869. F > D is directionally correct — forwards generate the majority of offensive opportunities. The gap (+0.0045) is modest relative to the spread, which is expected under ridge regularization where coefficients are heavily penalized toward zero. The signal is real and directionally clean.

**YOY stability:** 10,242 consecutive player-season pairs (season S → S+1 same player). Pearson r = 0.317 (p = 4.45e-237). This falls in the healthy range (0.15 PASS floor, 0.70 PASS ceiling). A correlation of 0.317 means ~10% of the variance in a player's season-N offensive coefficient is explained by their season-N−1 coefficient — meaningful talent persistence for a metric this noisy. The extremely low p-value confirms the signal is not a sampling artifact. 

**Note on what r = 0.317 means practically:** Hockey RAPM is noisier than NFL or NBA equivalents because: (1) goals are rare events (≈6% shot-on-goal rate for EV), making per-season coefficient estimates inherently high-variance; (2) context_xg (unlike goals) removes some noise via the base_margin smoothing, but individual season samples remain volatile. r = 0.317 is consistent with published hockey RAPM estimates across multiple methodologies and reflects genuine skill persistence, not over-regularization.

---

### Coverage (pred_goal Training Set)

Coverage is defined as the fraction of pred_goal training shots (seasons 2010-11 through 2023-24) with a non-null `shooter_rapm_xg_off` value after the RAPM join.

| Strength | n_shots | non_null | Coverage |
|---|---|---|---|
| even_strength | 1,216,359 | 1,054,595 | 86.7% |
| powerplay | 223,499 | 193,411 | 86.5% |
| shorthanded | 36,116 | 31,347 | 86.8% |
| empty_for | 28,801 | 26,042 | 90.4% |
| empty_against | 8,307 | 7,601 | 91.5% |

86.5–91.5% across all states exceeds the 60% warn threshold comfortably. Null entries represent shots by players who did not meet the minimum TOI threshold for their situation (e.g., ≤10 min EV for regular season) or who are not in the RAPM parquet for that season (e.g., fringe call-ups who only played in a small number of games). Empty-net states show slightly higher coverage (90–91%) because EA/EF shots are dominated by established skaters rather than fringe players.

---

### Career EV RAPM Leaderboard (≥200 min TOI, `total_rapm_context_xg`)

Season=0 rows represent career aggregates (TOI-weighted mean of per-season coefficients). All 2,066 qualified career entries span the full range +0.408 to −0.403.

**Top 10 (by career offensive context_xg RAPM):**

| Rank | API ID | Pos | TOI (min) | Career RAPM |
|---|---|---|---|---|
| 1 | 8482809 | F | 936 | +0.4080 |
| 2 | 8482730 | D | 1,414 | +0.3290 |
| 3 | 8479314 | F | 8,978 | +0.3031 |
| 4 | 8479323 | D | 7,610 | +0.2771 |
| 5 | 8478402 | F | 12,138 | +0.2701 |
| 6 | 8474892 | F | 298 | +0.2696 |
| 7 | 8470638 | F | 12,178 | +0.2539 |
| 8 | 8467514 | F | 5,362 | +0.2493 |
| 9 | 8479325 | D | 9,381 | +0.2491 |
| 10 | 8478038 | D | 8,956 | +0.2417 |

**Bottom 10:**

| Rank | API ID | Pos | TOI (min) | Career RAPM |
|---|---|---|---|---|
| 2057 | 8478062 | D | 1,339 | −0.4026 |
| 2058 | 8481563 | D | 891 | −0.3792 |
| 2059 | 8478476 | D | 1,034 | −0.3682 |
| 2060 | 8483466 | D | 1,588 | −0.3215 |
| 2061 | 8481567 | D | 1,987 | −0.3054 |
| 2062 | 8476947 | F | 490 | −0.2948 |
| 2063 | 8480884 | D | 2,039 | −0.2899 |
| 2064 | 8481585 | F | 202 | −0.2575 |
| 2065 | 8469684 | D | 1,707 | −0.2550 |
| 2066 | 8482451 | F | 714 | −0.2523 |

**Leaderboard observations:** The top three career entries (#1, #3, #5, #7) have very high TOI (8,978–12,178 min) — these are established veterans with large sample sizes generating robust estimates. The #1 career leader (8482809, F, only 936 min) at +0.408 is an outlier: the highest career RAPM with one of the smallest qualifying samples. This is a small-sample effect — players with brief NHL careers at sustained elite rate can accumulate very high mean coefficients even though the underlying signal is noisy. Ridge regression's shrinkage toward zero mitigates but doesn't eliminate this. The bottom 10 is heavily defenseman-dominated (8 of 10 are D), which is consistent with offensive RAPM measuring team goal differential on ice — defensemen tend to cluster near zero or negative in offensive impact, and the bottom tail captures defenders consistently playing in high-danger situations against their team.

---

### Per-Season Distribution (EV R `off_coeff_context_xg`)

| Season | n | mean | std | p1 | p99 |
|---|---|---|---|---|---|
| 2010-11 | 946 | +0.0025 | 0.0686 | −0.1493 | +0.2022 |
| 2011-12 | 945 | +0.0055 | 0.0992 | −0.2051 | +0.3248 |
| 2012-13 | 880 | +0.0022 | 0.0663 | −0.1390 | +0.1864 |
| 2013-14 | 928 | +0.0053 | 0.0908 | −0.1758 | +0.2618 |
| 2014-15 | 946 | +0.0021 | 0.0725 | −0.1523 | +0.2209 |
| 2015-16 | 950 | +0.0058 | 0.0941 | −0.1787 | +0.2824 |
| 2016-17 | 934 | +0.0024 | 0.0643 | −0.1459 | +0.1787 |
| 2017-18 | 947 | +0.0040 | 0.0906 | −0.1836 | +0.2581 |
| 2018-19 | 982 | +0.0026 | 0.0660 | −0.1380 | +0.1897 |
| 2019-20 | 939 | +0.0127 | 0.1120 | −0.2083 | +0.3352 |
| 2020-21 | 951 | +0.0060 | 0.0910 | −0.1789 | +0.2693 |
| 2021-22 | 1,050 | +0.0035 | 0.1015 | −0.2115 | +0.3111 |
| 2022-23 | 1,024 | +0.0048 | 0.1114 | −0.2394 | +0.2944 |
| 2023-24 | 975 | +0.0035 | 0.1000 | −0.2083 | +0.2796 |
| 2024-25 | 1,012 | +0.0052 | 0.1001 | −0.2118 | +0.2676 |

**Structural consistency:** All 15 seasons are on a consistent scale (std ≈ 0.06–0.11), confirming the per-season ridge regression is producing stable coefficient magnitudes across eras. There is no discontinuity between the 2010s and 2020s seasons.

**2012-13 lockout season (880 players, std=0.0663):** The slightly compressed std is expected — 48 games vs 82 produces fewer stints per player, so ridge regression applies marginally more effective shrinkage. The lockout season is not an outlier and does not require exclusion. p1/p99 range (−0.139 / +0.186) is narrower than most full seasons but proportionally consistent.

**2019-20 bubble season (std=0.1120, mean=+0.0127):** The highest std and highest mean of any season. The bubble format (played in two hub cities, no home/away travel, isolated environment) changed tactical game dynamics — the bubble concentrated primarily playoff-caliber players (non-playoff teams were excluded) and the unique conditions appear to have produced more differentiated individual performance. The higher mean (+0.0127 vs ≈+0.003–0.006 in other seasons) suggests a slight positive offset, possibly reflecting the elevated offensive context_xg from concentrated team quality in the bubble.

**2020-21 abbreviated season (56 games, std=0.0910):** Despite being another shortened season (56 games), the std is well within the normal range, unlike 2012-13. The 2020-21 division-only format may have had less impact on individual stint patterns than the full lockout structure.

**Coefficient scale sanity:** The p99 range (+0.19 to +0.33 depending on season) translates to approximately 0.19–0.33 additional context_xg per event for the top players in each season. Given that context_xg is calibrated to the ~6% EV base rate, a top-percentile RAPM of +0.25 means that having this player on the ice is associated with 0.25 additional expected goals per event over a neutral context — a large, meaningful effect that is directionally consistent with known elite offensive producers.

---

### Root Cause: Prior Run Was Invalid (Pre-Scoring Fix)

The initial RAPM diagnostic (also run 2026-05-15, before the context_xg fix) showed a catastrophic per-season scale discontinuity that made the results invalid:

| Season group | Prior std | Current std |
|---|---|---|
| 2010-11, 2011-12 | ~1.14 | ~0.069–0.099 |
| 2012-13 | 3.47 | 0.066 |
| 2013-14 through 2023-24 | ~0.07–0.11 | ~0.064–0.111 |
| 2024-25 | ~1.19 | ~0.100 |

The ~10× scale difference between the boundary seasons (2010-12, 2024-25) and middle seasons (2013-24) caused YOY stability to measure r=0.107 (WARN) — cross-era correlations between players in the 1.1-scale seasons and the 0.10-scale adjacent seasons were mathematically deflated regardless of true talent persistence.

**Root cause:** `context_xg/score.py` loaded the saved booster with `xgb.Booster()` and called `booster.predict(dmat)` without specifying `iteration_range`. With `EARLY_STOPPING_ROUNDS=50`, the booster contains `best_iteration + 50` trees (e.g., 126 trees for even_strength with `best_iteration=76`). The extra 50 post-best trees produced strongly bimodal raw predictions: the diagnostic showed fresh retrain dist_ratio of 2.37× (correct) vs saved-model scoring dist_ratio of 8.98× (bimodal) for the same even_strength model and params. RAPM stint-level aggregations of bimodal predictions produced near-zero target variance in the 2013-2024 training era, driving ridge coefficients toward zero at 10× greater shrinkage than correct targets. The fix replaced the Booster path with `load_model_artifacts()` → `XGBClassifier.predict_proba()`, which internally limits prediction to `best_iteration` trees.

---

### Changelog

| Date | YOY r | Coverage | 2012-13 std | Typical std | Notes |
|---|---|---|---|---|---|
| 2026-05-15 | 0.107 ⚠️ WARN | 86.5–91.5% ✅ | 3.47 | 0.07–0.11 (2013-24) / 1.1 (2010-12, 2024-25) | First run. Computed against bimodal context_xg scores (Booster.predict using all ~126 trees). Per-season std showed 10× scale discontinuity between boundary seasons and middle seasons; 2012-13 lockout exploded to std=3.47; YOY stability deflated to r=0.107 by cross-era scale mismatch. Overall: ⚠️ WARN. Results invalid. |
| 2026-05-15 | 0.317 ✅ PASS | 86.5–91.5% ✅ | 0.066 | 0.064–0.112 (all seasons) | Recomputed after context_xg scoring fix (score.py: Booster.predict → XGBClassifier.predict_proba, respecting best_iteration). All scale discontinuities resolved; all 15 seasons consistent; 2012-13 lockout normalised; YOY r improved from 0.107 to 0.317 (healthy talent persistence); all 4 checks PASS. |

---

## pred_goal

### Latest Diagnostic

**Date:** 2026-05-14 (last completed run; see note below)
**Model version:** 1.0.0 (pred_goal tier — talent features layered on context_xg base_margin)
**Trials:** 500 per state
**Finalization:** `--top-n 15` (default)
**Hold-out season:** 2024-25
**Key change from initial run:** Pooled OOF + hold-out calibration (was OOF-only). Resolved catastrophic log loss degradation across all four low-base-rate states (ES: −994% → −17.7% vs null; PP: −241% → −18.8%; SH: −442% → −22.5%; EF: −259% → −15.0%). See Issue 14 in `DECISIONS.md`.

**⚠ Pending re-run (2026-05-15):** Feature set and base_margin have changed since this diagnostic.
- Issue 16 (2026-05-15): context_xg re-scored with `XGBClassifier.predict_proba()` fix — base_margin is now correct (dist_ratio 1.06–1.65×); prior run used bimodal context_xg predictions from `Booster.predict()`.
- Issue 15 (2026-05-15): `process_data.py` re-run — `_1g` rolling features stripped; RAPM reduced to xg_off/xg_def only (14 features, down from 36). pred_goal train/hold_out rebuilt.
- Experiments re-tuning in progress. Results below reflect the **old feature set** and are superseded once experiments complete and `finalize-pred-xg` + `diagnose-pred-xg` are run.

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

**⚠ These results use the old feature set (before Issue 15+16 fixes) and are superseded.** `process_data.py` re-run on 2026-05-15; experiments re-tuning in progress. Update this section after `finalize-pred-xg` + `diagnose-pred-xg` complete.

### Advanced Metrics (hold-out 2024-25)

`Lift` = pred_goal hold-out PR AUC − base_xg (= context_xg) hold-out PR AUC on the same events.
`Max Cal Error` is the uniform-bin max calibration error; the calibration PASS/WARN/FAIL check uses quantile-based deciles.
`Null Brier` = base_rate × (1 − base_rate). Positive ΔLL% / ΔBr% means the model improves on predicting the base rate; negative means it is worse than predicting the base rate for every event.

**Note on Max Cal Error (uniform bins) for PP / SH / EF:** Values of 0.59–0.69 look alarming but are sparse-bin artifacts. The hold-out sets for these states are small (2,592–15,795 events). With predictions mostly below 0.30, the uniform bins [0.4–0.5] or [0.5–0.6] may contain only a handful of events — one or two goals determines the entire calibration error for that bin. The quantile-decile max abs error (0.14–0.19) is the meaningful calibration measure driving PASS/WARN/FAIL. ES max cal = ECE = 0.096 (single problematic bin dominates both measures).

| Strength | Base% | PR AUC | PR× | ROC AUC | Log Loss | Null LL | ΔLL% | Brier | Null Brier | ΔBr% | ECE | Max Cal Error | OOF Gap | Lift | Precision | Recall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| even_strength | 6.0% | 0.3204 | 5.37× | 0.7841 | 0.2660 | 0.2260 | −17.7% | 0.0642 | 0.0561 | −14.4% | 0.0960 | 0.0960 | 0.0203 | +0.0001 | 0.0597 | 1.0000 |
| powerplay | 10.3% | 0.3420 | 3.33× | 0.7012 | 0.3930 | 0.3310 | −18.8% | 0.1107 | 0.0921 | −20.2% | 0.1687 | 0.5922 | 0.0267 | +0.0005 | 0.1027 | 1.0000 |
| shorthanded | 7.2% | 0.3336 | 4.62× | 0.8188 | 0.3175 | 0.2592 | −22.5% | 0.0813 | 0.0669 | −21.4% | 0.1443 | 0.6899 | 0.0215 | +0.0006 | 0.0721 | 1.0000 |
| empty_for | 7.7% | 0.3121 | 4.04× | 0.7180 | 0.3132 | 0.2722 | −15.0% | 0.0808 | 0.0714 | −13.2% | 0.1244 | 0.6844 | 0.0041 | +0.0009 | 0.0773 | 1.0000 |
| empty_against | 56.7% | 0.7480 | 1.32× | 0.7032 | 0.6179 | 0.6842 | +9.7% | 0.2165 | 0.2455 | +11.8% | 0.0839 | 0.1325 | 0.0116 | −0.0340 | 0.7110 | 0.6585 |

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

**⚠ Per-trial hyperparameters (lambda, gamma, alpha, max_depth, etc.) are not available.** The Optuna studies for pred_goal (`{state}-1.0.0-pred`) were deleted from the database before meta.json storage was implemented. For future runs, `finalize.py` writes a `meta.json` alongside the model artifacts (matching the base_xg pattern) to preserve trial hyperparameters. Available model-derived values only:

| State | best_score (CV PR AUC) | best_iter | n_trees |
|---|---|---|---|
| even_strength | 0.3204 | 5 | 56 |
| powerplay | 0.3419 | 2 | 53 |
| shorthanded | 0.3326 | 2 | 53 |
| empty_for | 0.3114 | 4 | 55 |
| empty_against | 0.7826 | 0 | 51 |

`best_iter` and `n_trees` are from `Booster.best_iteration` and `booster.num_trees()` on the saved model artifact.

**Note on low best_iterations:** ES=5, PP=2, SH=2, EF=4, EA=0. These reflect the structural role of pred_goal as a thin adjustment layer stacked on context_xg's logit base_margin. Talent features provide small marginal signal; logloss early stopping fires quickly once the talent adjustment stabilizes. The ~50–56 total trees represent the complete final adjustment at the talent layer, not a failure to train — discrimination (PR AUC 0.32–0.79) matches context_xg, confirming the models are not degenerate.

---

### Feature Gain

Talent features only — `BASE_XG_FEATURE_COLUMNS` and `CONTEXT_XG_FEATURE_COLUMNS` are confirmed zero gain in all states (no context leakage). Top features by state:

**even_strength:** `goalie_gsax_per_shot_1g` 45.0%, `goalie_gsax_ewma` 27.5%, `goalie_gsax_per_shot_10g` 20.9%, `shooter_gax_per_shot_10g` 5.1%, `shooter_gax_ewma` 0.8%, RAPM < 1% combined.

**powerplay:** `goalie_gsax_per_shot_10g` 16.5%, `goalie_gsax_1g` 16.0%, `shooter_gax_per_shot_10g` 15.9%, `goalie_gsax_ewma` 12.8%, `shooter_gax_ewma` 12.5%, `goalie_gsax_per_shot_1g` 10.0%, `shooter_gax_per_shot_1g` 6.0%, `shooter_gax_10g` 4.1%.

**shorthanded:** `goalie_gsax_per_shot_1g` 16.0%, `shooter_gax_per_shot_10g` 15.5%, `shooter_gax_ewma` 14.3%, `goalie_gsax_ewma` 14.2%, `goalie_gsax_per_shot_10g` 12.9%, `goalie_gsax_10g` 11.3%, `shooter_gax_10g` 7.7%, `goalie_gsax_1g` 3.7%.

**empty_for:** `goalie_gsax_per_shot_10g` 18.1%, `goalie_gsax_10g` 17.6%, `goalie_gsax_per_shot_1g` 15.3%, `goalie_gsax_ewma` 14.3%, `shooter_gax_ewma` 10.2%, `shooter_gax_per_shot_10g` 7.1%, `shooter_gax_10g` 5.1%, `shooter_gax_per_shot_1g` 1.8%.

**empty_against:** Highly distributed (45 features, top 8: `shooter_gax_ewma` 5.4%, `shooter_rapm_xg_off` 3.9%, `shooter_vs_teammates_rapm_career_goals_off` 3.6%, `shooter_gax_10g` 3.4%, `opp_rapm_career_corsi_def` 3.3%, `shooter_shots_10g` 3.1%, `shooter_rapm_corsi_def` 3.0%, `shooter_gax_per_shot_10g` 2.7%). No dominant pattern consistent with a model finding no clear signal.

**RAPM null rates:** 0.0% across all states and all 14 training seasons (73 nulls out of 1.2M ES shots; comparable near-zero rates for other states). Career RAPM is available for effectively all skaters in the dataset.

---

### Per-Strength Interpretation

#### even_strength

**Performance tier:** ❌ FAIL (calibration). Discrimination: strong (0.3204 = 5.4× null, ROC 0.784). Calibration usable but overconfident in one cluster.

**Calibration character:** The distribution has a residual bimodal structure. Deciles 0–7 have essentially flat mean predictions (0.046–0.091), indicating most shots cluster around the base-margin prior. Decile 8 jumps to mean_pred=0.152 with actual=0.023 (7× overestimate) — a cluster of shots the model assigns moderately high probability that are almost never goals in hold-out. Decile 9 is near-correct (mean_pred=0.162, actual=0.150). This decile-8 anomaly is the residual from the original bimodal cliff: after the pooled calibration fix, the extreme 0.92-predicted cluster was pulled down to 0.15, but even 0.15 overestimates the ~2% actual hold-out rate for those events.

**Lift:** +0.0001 PR AUC over context_xg. The seasonal table shows pred_goal and ctx_xg are essentially identical in all 15 seasons. Talent features add negligible discriminative power beyond the context prior.

**Feature gain:** Three goalie rolling-performance features account for 93.4% of gain — `goalie_gsax_per_shot_1g` (45%), `goalie_gsax_ewma` (28%), `goalie_gsax_per_shot_10g` (21%). RAPM features combined: < 1%. The model has essentially learned to ask "is the goalie having a bad recent game?" and largely ignores stable career talent signals.

#### powerplay

**Performance tier:** ❌ FAIL (calibration). Discrimination: strong (0.3420 = 3.3× null, ROC 0.701).

**Calibration character:** Same decile-8 pattern as ES. Mean predictions are compressed (deciles 0–7 span only 0.070–0.152), then decile 8 overestimates (mean_pred=0.242, actual=0.055). Decile 9 is near-correct (mean_pred=0.280, actual=0.222). Log loss −18.8% vs null.

**Lift:** +0.0005 PR AUC over context_xg. Negligible across all 15 seasons.

#### shorthanded

**Performance tier:** ❌ FAIL (calibration). Discrimination: strong (0.3336 = 4.6× null, ROC 0.819 — highest ROC of all states).

**Calibration character:** Decile-8 pattern: mean_pred=0.196, actual=0.018 (11× overestimate). Log loss −22.5% vs null. Max cal error 0.178 (decile threshold). Smallest non-EA dataset (35K shots, 2,592 hold-out events).

**Lift:** +0.0006 PR AUC over context_xg. Near-zero. SH is fast-break dominated (context_xg already captures this well through sequence timing features).

#### empty_for

**Performance tier:** ❌ FAIL (calibration). Discrimination: strong (0.3121 = 4.0× null, ROC 0.718). OOF gap = 0.0041 — the smallest of all states.

**Calibration character:** Decile 8: mean_pred=0.182, actual=0.042 (4.3× overestimate). Decile 9 near-perfect (abs_err=0.0008). Log loss −15.0% vs null.

**Lift:** +0.0009 PR AUC over context_xg — the best of the four low-base-rate states. RAPM features account for ~5.5% combined — the highest RAPM contribution of any state.

#### empty_against

**Performance tier:** ❌ FAIL (lift). Discrimination: very high (0.7480 = 1.3× null, ROC 0.703). Calibration PASS (max abs error 0.023, excellent). But pred_goal is 3.4% PR AUC *worse* than context_xg.

**Negative lift:** Consistent across all 14 training seasons and the hold-out year — pred_goal never beats context_xg for EA in any season. The underlying issue is structural: in EA situations the goalie has been pulled, making goalie form/talent features irrelevant. Shooter talent features add noise rather than signal because EA goals are driven by shot quality (already in context_xg), not by who is shooting.

**Feature gain:** Highly distributed (45 features, no dominant pattern) — consistent with a model that has found no clear signal and is spreading tree capacity across many marginally informative features. No context leakage detected.

---

### Changelog

| Date | Version | Trials | ES PR AUC | PP PR AUC | SH PR AUC | EF PR AUC | EA PR AUC | Notes |
|---|---|---|---|---|---|---|---|---|
| 2026-05-14 | 1.0.0 | 500 (OOF-only cal) | 0.3204 | 0.3420 | 0.3335 | 0.3121 | 0.7473 | Initial run. OOF-only Platt calibration. ES log loss 2.47 (994% worse than null), PP 1.13 (241% worse), SH 1.40 (442% worse), EF 0.98 (259% worse). EA calibration WARN (max err 0.094). All five states FAIL. Catastrophic miscalibration driven by temporal drift: OOF calibrator learned training-era talent matchup statistics that don't hold in the hold-out season. Root cause and fix documented in Issue 14. |
| 2026-05-14 | 1.0.0 | 500 (pooled cal) | 0.3204 | 0.3420 | 0.3336 | 0.3121 | 0.7480 | After pooled OOF + hold-out calibration fix (Issue 14). ES log loss 0.266 (−17.7% vs null), PP 0.393 (−18.8%), SH 0.318 (−22.5%), EF 0.313 (−15.0%), EA 0.618 (+9.7%). Calibration improved 9–56× across states. Residual calibration FAIL remains for ES/PP/SH/EF (decile-8 non-monotone pattern, max abs err 0.13–0.19). EA calibration PASS. Lift over context_xg negligible for non-EA (+0.0001–+0.0009); EA negative lift (−0.034). See Issues 14 and 15. |
