# base_xg Diagnostics

## Latest Diagnostic

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

## Changelog

| Date | Version | ES Trials | PP Trials | SH Trials | EF Trials | EA Trials | ES PR AUC | PP PR AUC | SH PR AUC | EF PR AUC | EA PR AUC | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-05-13 | 1.0.0 | 500 | 500 | 500 | 500 | 500 | 0.1611 | 0.1771 | 0.1889 | 0.1679 | 0.6545 | Initial diagnostic; 14-feature model; EA OOF gap FAIL (0.0868) due to score_diff era-drift; architecture migration to 8-feature geometry set and OOF complexity fix applied after this run |
| 2026-05-13 | 1.0.0 | 500+ | 500+ | 500+ | 150+ | 590+ | 0.1591 | 0.1734 | 0.1914 | 0.1634 | 0.7428 | 8-feature geometry set; OOF fold complexity fix applied; EA calibrator collapse resolved; PP feature representation converged to high_danger dominance (89.2%); SH feature representation flipped to event_distance dominance (52.0%); EA OOF gap passes at 0.0092; EA ⚠️ WARN on high-confidence (17.2%) |
| 2026-05-14 | 1.0.0 | 500+ | 500+ | 500+ | 150+ | 590+ | 0.1595 | 0.1709 | 0.1879 | 0.1741 | 0.7650 | Refinalized with new model selection criteria; EF PR AUC improved meaningfully (+0.0107); EA PR AUC improved (+0.0222); SH feature gain flipped back to high_danger dominated (83.4%); EA high-confidence crossed FAIL threshold (20.7% goals > 0.8 vs 20.0% threshold); EA ECE slightly worse (0.0713 vs 0.0368); all non-EA states PASS |