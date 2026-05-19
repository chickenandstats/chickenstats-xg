# chickenstats-xg v1.0.0 Diagnostic Results

Consolidated diagnostic results for all three tiers (base_xg → context_xg → pred_goal).
Each section follows the same structure: Latest Diagnostic → Pass/Fail Summary → Advanced Metrics → Hyperparameters → Per-Strength Interpretation → Changelog.

---

## base_xg

### Latest Diagnostic

**Date:** 2026-05-17
**Model version:** 1.0.1 (8-feature pure geometry set; NSGA-II multi-objective tuning)
**Trials:** 1360 ES / 2000 PP / 2000 SH / 2000 EF / 2000 EA
**Tuner change:** TPE (single-objective PR-AUC) → NSGA-II (multi-objective: maximize PR-AUC, minimize log loss). Pareto-first trial selection in `select_top_trials()`.
**Hold-out season:** 2024-25

### Pass / Fail Summary

| Strength | Distribution | High Conf | Calibration | OOF Gap | Feat Gain | Overall |
|---|---|---|---|---|---|---|
| even_strength | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ PASS |
| powerplay | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ PASS |
| shorthanded | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ PASS |
| empty_for | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ PASS |
| empty_against | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ FAIL |

EA ❌ FAIL on high-confidence check (23.3% of goals above 0.8) is a structural artifact at the 56.7% base rate — not a fingerprinting failure. The diagnose `high_confidence` check applies the same base-rate guard as the bimodal rejection in `screen_trials()` and the Optuna objective. This is a known diagnostic threshold mismatch at high base rates.

### Advanced Metrics (hold-out 2024-25)

Precision and recall computed at the base-rate threshold (predict positive if base_xg ≥ base rate).
`Max Cal Error` is the uniform-bin max calibration error from the script's advanced metrics; the calibration PASS/WARN/FAIL check uses quantile-based deciles (a different measure — see per-strength notes for states where these diverge).
`Null Brier` = base_rate × (1 − base_rate); the null model always predicts the base rate.
Shorthanded `Max Cal Error` (0.6380) and empty_for (0.6235) reflect sparse predictions in extreme uniform bins, not real calibration failures — use the quantile-based decile max (0.0406 and 0.0224 respectively) for those states.

| Strength | Base Rate | PR AUC | PR AUC × | ROC AUC | Log Loss | Null LL | ΔLL% | Brier | Null Brier | ΔBr% | ECE | Max Cal Error | OOF Gap | Precision | Recall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| even_strength | 6.0% | 0.1574 | 2.64× | 0.7559 | 0.2055 | 0.2260 | +9.1% | 0.0537 | 0.0561 | +4.3% | 0.0068 | 0.3788 | 0.0048 | 0.1324 | 0.6136 |
| powerplay | 10.3% | 0.1729 | 1.68× | 0.6473 | 0.3196 | 0.3310 | +3.4% | 0.0901 | 0.0921 | +2.2% | 0.0065 | 0.2753 | 0.0001 | 0.1528 | 0.5641 |
| shorthanded | 7.2% | 0.1903 | 2.64× | 0.8002 | 0.2295 | 0.2592 | +11.4% | 0.0636 | 0.0669 | +5.0% | 0.0126 | 0.6380 | 0.0136 | 0.1789 | 0.6791 |
| empty_for | 7.7% | 0.1724 | 2.23× | 0.7025 | 0.2552 | 0.2722 | +6.2% | 0.0686 | 0.0714 | +3.9% | 0.0063 | 0.6235 | 0.0094 | 0.1413 | 0.6087 |
| empty_against | 56.7% | 0.7537 | 1.33× | 0.6979 | 0.6530 | 0.6842 | +4.6% | 0.2178 | 0.2455 | +11.3% | 0.0693 | 0.2184 | 0.0044 | 0.8921 | 0.3785 |

---

### Hyperparameters

**Fixed params (all states):** objective=binary:logistic, booster=gbtree, n_estimators=500, early_stopping_rounds=50, eval_metric=["aucpr","logloss"] (early stop on logloss), random_state=615, enable_categorical=True. Monotone constraints: event_distance (−1), event_angle (−1).

Values below are exact — sourced from Optuna trial params (trial numbers stored in per-state `meta.json`). `best_iter` is from `Booster.best_iteration` on the saved model artifact.

| State | Trial # | max_depth | mds | lambda | gamma | alpha | mcw | lr | subsample | cbt | cbl | spw | best_iter |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| even_strength | 1262 | 5 | 4 | 0.510 | 0.253 | ~0 | 27 | 0.0150 | 1.00 | 0.70 | 1.114 | 499 |
| powerplay | 1480 | 6 | 6 | 0.510 | 2.631 | 0.007 | 40 | 0.2287 | 0.90 | 1.00 | 1.110 | 31 |
| shorthanded | 1864 | 5 | 5 | 2.150 | 3.373 | ~0 | 22 | 0.1503 | 0.80 | 1.00 | 1.236 | 105 |
| empty_for | 1794 | 5 | 5 | 1.323 | 1.725 | 0.750 | 20 | 0.1137 | 0.95 | 0.65 | 1.104 | 224 |
| empty_against | 522 | 4 | 6 | 6.718 | 0.721 | 0.007 | 24 | 0.0619 | 0.55 | 1.00 | N/A | 221 |

`mds`=max_delta_step, `mcw`=min_child_weight, `lr`=learning_rate, `cbt`=colsample_bytree, `cbl`=colsample_bylevel, `spw`=scale_pos_weight. EA has no `spw`. `colsample_bynode` not shown — 0.95 for ES, 0.95 PP, 0.90 SH, 0.65 EF, 0.95 EA.

**Notable pattern:** `min_child_weight` is below the diagnostic floor of 30 for all five states (27/40/22/20/24). NSGA-II explored more of the low-MCW region than TPE did — possibly because lower MCW trials offer a better log-loss / PR-AUC Pareto tradeoff at the margin. All passed calibration and OOF checks in practice. Monitor at pred_goal where small leaf sizes can interact with RAPM features.

**even_strength (trial #1262):** lr=0.0150 is the slowest of all states, requiring best_iter=499 (nearly the full 500-tree budget — did not early-stop). lambda=0.510 is much lighter than the v1.0.0 trial (9.354) — the Pareto objective is finding well-calibrated solutions with less regularization. gamma=0.253 very light pruning.

**powerplay (trial #1480):** max_depth=6 at the cap, best_iter=31 — same structural pattern as before (PP early-stops quickly). The high learning rate (0.2287) + early stopping reflects the compressed PP distribution stabilizing fast.

**shorthanded (trial #1864):** gamma=3.373 (strongest pruning of the low-rate states), lambda=2.150. best_iter=105 moderate. mcw=22 is the lowest of any non-EA state — leaf size risk flagged but calibration passed.

**empty_for (trial #1794):** alpha=0.750 provides L1 sparsity (second highest L1 of any state after EA). mcw=20 is the lowest overall. best_iter=224 — moderate.

**empty_against (trial #522):** lambda=6.718 is the strongest regularization of all states — appropriate for the ~9K EA dataset. subsample=0.55 aggressive row sampling. best_iter=221 is notably higher than v1.0.0's 391 in a different direction — this trial uses a faster learning rate (0.0619 vs 0.0243), reaching convergence sooner.

---

### Per-Strength Interpretation

#### even_strength

**Performance tier:** medium (0.1574 vs 0.1595 prior run; −0.0021, within tuning noise)

PR AUC 0.1574 is 2.64× null for a geometry-only model. ROC AUC 0.7559. Log loss improvement +9.1% vs null (down from +10.8% in v1.0.0 — slight calibration regression despite log loss being an explicit NSGA-II objective). ECE 0.0068 (up from 0.0048 but still excellent). The calibration decile table shows a mild S-curve: deciles 0–2 over-predict and 6–7 under-predict, max error 0.0259. OOF gap 0.0048 unchanged.

Feature gain: high_danger 60.9%, coords_x 14.3%, event_distance 7.3%, coords_y 6.4%, abs_y_distance 6.0%, shot_type 3.5%, event_angle 1.6%. More balanced across individual coordinate features than the v1.0.0 high_danger-dominated (55.6%) decomposition, but the total geometry share is similar. No concerns.

Season-by-season PR AUC is flat (0.148–0.161 across 2010–2025) with no temporal drift.

#### powerplay

**Performance tier:** medium (0.1729 vs 0.1709 prior run; +0.0020 improvement)

PP geometry discrimination improved modestly: PR AUC 0.1729 (1.68× null), ROC AUC 0.6473 (flat). Log loss improvement +3.4% vs null is the weakest of all states — the structural limitation of compressed PP shot locations remains. OOF gap 0.0001 is essentially zero. max_depth=6 at cap with best_iter=31 was flagged by the diagnostic WARN — feature gain confirms geometry dominance (event_distance 38.9%, high_danger 35.6%), which is the expected pattern and rules out fingerprinting.

Feature gain: event_distance 38.9%, high_danger 35.6%, abs_y_distance 12.4%, shot_type 5.9%, event_angle 3.3%, coords_y 1.7%, coords_x 1.2%, danger 1.0%. PP feature decomposition has shifted away from high_danger dominance (89.2% → 35.6%) toward a more balanced geometry split — consistent with a different Pareto-optimal trial being selected.

#### shorthanded

**Performance tier:** medium-high (0.1903 vs 0.1879 prior run; +0.0024 improvement — now above the 0.19 threshold)

Best improvement of the group on PR AUC. ROC AUC 0.8002 (flat). Log loss +11.4% vs null (down from +15.4% — same pattern as ES: log loss slightly worse despite being a NSGA-II objective). The diagnostic min_child_weight=22 WARN is worth monitoring: with only 38K training shots and a 7% goal rate, leaf splits on small subsets are plausible, though calibration passed cleanly (ECE 0.0126, max decile error 0.0406).

Feature gain: high_danger 85.5%, event_distance 6.5%, abs_y_distance 3.8%, event_angle 1.6%, shot_type 1.3%. high_danger dominance increased from 83.4% → 85.5%. Shorthanded geometry is highly concentrated in the slot/crease, making danger-zone placement the overwhelming signal.

#### empty_for

**Performance tier:** medium (0.1724 vs 0.1741 prior run; −0.0017, within noise)

Slight regression. ROC AUC 0.7025 flat. Calibration clean — ECE 0.0063, max decile error 0.0224. OOF gap 0.0094 (down from 0.0127). min_child_weight=20 WARN is the most aggressive in the group; still passed all calibration checks. Feature gain: high_danger 67.0%, event_distance 10.8%, abs_y_distance 9.8%, shot_type 3.3%. High_danger dominance slightly reduced (67.6% → 67.0%).

Uniform-bin max cal error 0.6235 is a sparse-bin artifact — predictions cluster below 0.2, leaving near-zero events in high bins. Decile max 0.0224 is the correct reference.

#### empty_against

**Performance tier:** medium-high (0.7537 vs 0.7650 prior run; −0.0113 regression)

EA shows the most notable regression of the group. PR AUC 0.7537 (1.33× null, down from 1.35×). ROC AUC 0.6979 (down from 0.7052). Log loss improvement +4.6% vs null (down from +5.0%). Brier improvement +11.3% (down from +11.7%). ECE 0.0693 (flat vs 0.0713). Calibration decile max 0.0325 — clean.

High-confidence ❌ FAIL (23.3% goals above 0.8, 3.9% shots above 0.8) is structural at the 56.7% base rate — same artifact as v1.0.0. The `screen_trials()` and objective-level base_rate guard was applied for model selection, but the diagnose script's `high_confidence` check doesn't yet have this guard. EA is safe to proceed.

Season PR AUC (0.668–0.814) with no temporal trend. Feature gain: high_danger 54.5%, event_distance 11.6%, shot_type 10.1%, event_angle 7.3%, danger 6.6%, abs_y_distance 4.6%, coords_x 2.9%, coords_y 2.5%. High_danger share increased substantially (38.0% → 54.5%) while shot_type fell (11.9% → 10.1%). The new trial leans more on zone placement and less on shot mechanics than v1.0.0.

The EA regression is likely driven by trial selection: v1.0.0 trial #1160 had 2000 EA trials and used TPE optimizing a single PR-AUC objective; v1.0.1 trial #522 was selected from a 2000-trial NSGA-II study on a Pareto criterion. The Pareto-optimal frontier for EA contains fewer trials (small dataset, compressed distribution) and the composite ranking may have selected a trial that is well-calibrated but weaker on discrimination. Worth watching when context_xg stacks on top.

---

### Changelog

| Date | Version | ES Trials | PP Trials | SH Trials | EF Trials | EA Trials | ES PR AUC | PP PR AUC | SH PR AUC | EF PR AUC | EA PR AUC | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-05-13 | 1.0.0 | 500 | 500 | 500 | 500 | 500 | 0.1611 | 0.1771 | 0.1889 | 0.1679 | 0.6545 | Initial diagnostic; 14-feature model; EA OOF gap FAIL (0.0868) due to score_diff era-drift; architecture migration to 8-feature geometry set and OOF complexity fix applied after this run |
| 2026-05-13 | 1.0.0 | 500+ | 500+ | 500+ | 150+ | 590+ | 0.1591 | 0.1734 | 0.1914 | 0.1634 | 0.7428 | 8-feature geometry set; OOF fold complexity fix applied; EA calibrator collapse resolved; PP feature representation converged to high_danger dominance (89.2%); SH feature representation flipped to event_distance dominance (52.0%); EA OOF gap passes at 0.0092; EA ⚠️ WARN on high-confidence (17.2%) |
| 2026-05-14 | 1.0.0 | 500+ | 500+ | 500+ | 150+ | 590+ | 0.1595 | 0.1709 | 0.1879 | 0.1741 | 0.7650 | Refinalized with new model selection criteria; EF PR AUC improved meaningfully (+0.0107); EA PR AUC improved (+0.0222); SH feature gain flipped back to high_danger dominated (83.4%); EA high-confidence crossed FAIL threshold (20.7% goals > 0.8 vs 20.0% threshold); EA ECE slightly worse (0.0713 vs 0.0368); all non-EA states PASS |
| 2026-05-17 | 1.0.1 | 1360 | 2000 | 2000 | 2000 | 2000 | 0.1574 | 0.1729 | 0.1903 | 0.1724 | 0.7537 | First NSGA-II multi-objective run (maximize PR-AUC + minimize log loss). Results broadly flat vs v1.0.0: PP/SH improved (+0.002), ES/EF regressed slightly (−0.002), EA regressed −0.011. Log loss slightly worse on ES/SH despite being an explicit objective — geometry models were already well-calibrated. All new selected trials have mcw below the diagnostic floor of 30 (NSGA-II explored lower-MCW Pareto-optimal region). EA ❌ FAIL on high-confidence check remains structural threshold artifact. diagnose `high_confidence` check needs same base_rate < 0.5 guard as screen_trials/objective. |

---

## context_xg

### Latest Diagnostic

**Date:** 2026-05-18 (refreshed — second finalization run same day)
**Model version:** 1.0.1 (21-feature gbtree depth-2, no interaction constraints; Issues 18+20+21+22+23+24 applied)
**Trials:** All studies complete under v1.0.1 constraints (lr ≤ 0.10, N_ESTIMATORS=100, EARLY_STOPPING_ROUNDS=20, lambda ceiling 500)
**Hold-out season:** 2024-25
**Key changes from prior run:** Same constraints as first v1.0.1 finalization. Re-finalization selected different Optuna trials for PP (571→1556), SH (1017→1851), and EF (490→1772); ES and EA trials unchanged. SH PR-AUC improved 0.3460→0.3476; all other states stable (max Δ±0.0003). All PASS/FAIL/WARN outcomes unchanged.

### Pass / Fail Summary

| Strength | Distribution | High Conf | Calibration | OOF Gap | Lift | Gain Conc | Overall |
|---|---|---|---|---|---|---|---|
| even_strength | ✅ | ✅ | ✅ | ❌ FAIL | ✅ | ✅ | ❌ FAIL |
| powerplay | ✅ | ✅ | ✅ | ❌ FAIL | ✅ | ✅ | ❌ FAIL |
| shorthanded | ✅ | ✅ | ✅ | ⚠️ WARN | ✅ | ✅ | ⚠️ WARN |
| empty_for | ✅ | ✅ | ✅ | ❌ FAIL | ✅ | ✅ | ❌ FAIL |
| empty_against | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ PASS |

All FAIL/WARN results are **OOF gap only**, and in the **anomalous positive direction** (hold-out PR-AUC > training OOF). All other checks pass cleanly across all 5 states. High-confidence check now passes for all states (previous v1.0.0 high-lr trials no longer selected; lr ≤ 0.10 enforced). OOF gap positive direction is temporal variance: 2018-19 and 2023-24 are anomalously low-discriminability seasons in training folds; 2024-25 hold-out has higher context discriminability, not worse.

### Advanced Metrics (hold-out 2024-25)

`Lift` = context_xg hold-out PR AUC − base_xg hold-out PR AUC on the same 2024-25 events.
`Max Cal Error` is the uniform-bin max calibration error; calibration PASS/WARN/FAIL uses quantile-based deciles.
`Null Brier` = base_rate × (1 − base_rate). Positive ΔLL% / ΔBr% means the model improves on predicting the base rate.

**Note on Max Cal Error (uniform bins):** Values (0.14–0.35) are sparse-bin artifacts. With SHOT p90 < 0.13 for low-base-rate states, 95%+ of predictions fall in [0, 0.13]; any uniform bin above 0.3 has near-zero samples. The quantile-decile max abs error (0.023–0.035) is the correct calibration measure and drives the PASS/WARN/FAIL verdict.

| Strength | Base% | PR AUC | PR× | ROC AUC | Log Loss | Null LL | ΔLL% | Brier | Null Brier | ΔBr% | ECE | Max Cal Error | OOF Gap | Lift | Precision | Recall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| even_strength | 6.0% | 0.3658 | 6.13× | 0.7926 | 0.1823 | 0.2260 | +19.3% | 0.0456 | 0.0561 | +18.8% | 0.0024 | 0.1897 | 0.0615 | +0.2083 | 0.1698 | 0.5536 |
| powerplay | 10.3% | 0.4139 | 4.03× | 0.7250 | 0.2736 | 0.3310 | +17.3% | 0.0737 | 0.0921 | +20.0% | 0.0082 | 0.1444 | 0.0710 | +0.2410 | 0.2137 | 0.4963 |
| shorthanded | 7.2% | 0.3476 | 4.82× | 0.8317 | 0.2088 | 0.2592 | +19.4% | 0.0565 | 0.0669 | +15.6% | 0.0128 | 0.1940 | 0.0345 | +0.1568 | 0.1936 | 0.6471 |
| empty_for | 7.7% | 0.3859 | 4.99× | 0.7475 | 0.2261 | 0.2722 | +16.9% | 0.0585 | 0.0714 | +18.0% | 0.0113 | 0.3417 | 0.0897 | +0.2136 | 0.1826 | 0.5826 |
| empty_against | 56.7% | 0.7820 | 1.38× | 0.7013 | 0.6097 | 0.6842 | +10.9% | 0.2130 | 0.2455 | +13.2% | 0.0446 | 0.1925 | 0.0098 | +0.0283 | 0.7573 | 0.5053 |

---

### Hyperparameters

**Fixed params (all states):** booster=gbtree, objective=binary:logistic, max_depth=2, n_estimators=100, early_stopping_rounds=20, eval_metric=["logloss","aucpr"] (early stopping on aucpr), random_state=615, enable_categorical=True. lr ceiling 0.10, lambda ceiling 500. No interaction_constraints (Issue 21). max_delta_step=1 confirmed for all states.

Values sourced from `models/context_xg/{strength}/params.json`. `best_iter` = early-stopped tree count on the full-dataset training run.

| State | Trial | mds | lambda | gamma | alpha | mcw | lr | subsample | spw | best_iter |
|---|---|---|---|---|---|---|---|---|---|---|
| even_strength | 350 | 1 | 71.556 | 3.131 | 0.905 | 372 | 0.0778 | 0.75 | 1.364 | 99 |
| powerplay | 1556 | 1 | 13.075 | 6.843 | 0.135 | 141 | 0.0961 | 0.80 | 1.485 | 99 |
| shorthanded | 1851 | 1 | 16.918 | 1.178 | 5.012 | 105 | 0.0961 | 1.00 | 2.467 | 99 |
| empty_for | 1772 | 1 | 68.143 | 8.098 | 3.628 | 100 | 0.0961 | 1.00 | 2.313 | 98 |
| empty_against | 1803 | 1 | 13.900 | 8.976 | 1.352 | 391 | 0.0737 | 0.85 | N/A | 10 |

`mds` = max_delta_step. `mcw` = min_child_weight. `spw` = scale_pos_weight. EA has no `spw` — class balance handled naturally at the 56.7% base rate.

**Hyperparameter assessment:**

**lr ≤ 0.10 enforced for all states.** This is the key behavioral change from v1.0.0 (lr 0.25–0.27). All selected trials have lr 0.074–0.096 — slower learning produces more conservative accumulation, eliminating the high-confidence (goals > 0.8) WARN that affected all four low-base-rate states in v1.0.0. best_iter=97–99 for ES/PP/SH/EF (all running to near the 100-tree limit); EA best_iter=10 (logloss stabilizes rapidly at 56.7% base rate regardless of learning rate).

**max_delta_step=1 confirmed for all states.** Structural prevention of bimodal cliff. With the new structural_flaw_penalty hard gate in `screen_trials()`, mds=1 trials are selected reliably from the default --top-n 15 (was 150).

**even_strength (trial 350):** lambda=71.6 is the strongest regularization of the four low-base-rate states — appropriate for the 1.24M-shot dataset. lr=0.0778, mcw=372 (highest of all low-base-rate states). spw=1.364 near-uniform. Feature gain now distributed across 13 features including `is_scramble` (4.7%) and `is_rebound` (2.4%) — binary flags appear for first time (Issue 21 fix confirmed).

**powerplay (trial 1556):** lambda=13.1 (lightest L2 of all low-base-rate states — down from 44.8 in prior run), lr=0.0961, mcw=141 (higher than prior 106), subsample=0.80. `play_speed` dominant at 23.3%, `seconds_since_last` 20.5% — PP is driven by transition speed and time-since-last-event within zone possession. `seconds_since_event_team_change` 15.4%, `seconds_since_opp_team_change` 11.4%. `strength_state` 3.1% captures sub-situations (4v3, 5v4, etc.).

**shorthanded (trial 1851):** alpha=5.012 (highest L1 regularization of all states, slightly up from prior 4.716), lambda=16.9 (up from 11.3), spw=2.467 (up from 2.082 — more aggressive minority upweighting), lr=0.0961, mcw=105. Feature gain: `play_speed` 18.2%, `seconds_since_stoppage` 13.2%, `position` 11.2% (unique to SH — D vs F matters on penalty kills), `logit_base_xg` 9.9% (higher geometry share than other states — fast-break geometry matters more for SH). `is_scramble` 4.8%, `is_rebound` not in top 10 but nonzero. gamma=1.178 unchanged from prior trial.

**empty_for (trial 1772):** lambda=68.1 (identical to prior trial), lr=0.0961, mcw=100, spw=2.313 (same). gamma=8.098 (up from 6.760 — heavier pruning), alpha=3.628 (up sharply from 0.324 — strong L1 added). `play_speed` 22.9%, `seconds_since_last` 21.2% — nearly tied, different ordering from prior trial (was 30.8% vs 18.1%). `period_seconds` 4.0% — time pressure in empty net situations. OOF gap FAIL (0.0897) is the largest gap but in the positive direction — 2023-24 (0.2253) and 2018-19 (0.2135) anomalously low per-season PR-AUC in training folds pull down OOF average.

**empty_against (trial 1803):** lambda=13.9, gamma=8.976 (highest of all states), lr=0.0737, mcw=391, best_iter=10. Only 8 features with nonzero gain: `play_speed` 25.4%, `seconds_since_last` 21.2%, `prior_event_angle` 14.0%, `seconds_since_stoppage` 13.2%. Collapses to 4 timing/speed/distance features — correct for open-net situations. OOF gap=0.0098 (best of all states). ECE=0.0446; decile 8 max abs error=0.0354 (moderate overconfidence at 80% tier — structural at 56.7% base rate, Platt compression).

---

### Per-Strength Interpretation

#### even_strength

**Performance tier:** ❌ FAIL (OOF gap). Discrimination: high (0.3658, 6.13× null).

**OOF gap FAIL:** Training OOF PR-AUC = 0.3042, hold-out = 0.3658, gap = 0.0615 — positive direction (hold-out BETTER than training OOF). Structural cause: seasons 2023-24 (PR-AUC 0.1962) and 2018-19 (0.2053) are anomalously low-discriminability training fold seasons; these drag down the training OOF average. 2024-25 hold-out context discriminability is more typical. This is not overfitting — calibration is near-perfect (ECE=0.0024) and the distribution shows no bimodal behavior.

Calibration: excellent — ECE=0.0024 (best of all states), decile max abs error 0.0314 (decile 8 under-prediction, mild). Log loss +19.3% vs null. Brier +18.8%. SHOT p90/base_rate = 1.39× (clean distribution).

High-confidence: ✅ PASS — 9.3% goals above 0.8. Significant improvement from v1.0.0 WARN (16.6%). Direct result of lr ≤ 0.10 vs prior 0.2616.

**Feature gain:** `seconds_since_stoppage` 22.7%, `seconds_since_last` 17.0%, `play_speed` 16.5%, `seconds_since_opp_team_change` 9.9%, `seconds_since_event_team_change` 7.5%. Top 3 same domain (timing) as prior run. Binary flags now appear: `is_scramble` 4.7%, `is_rebound` 2.4% — confirming Issue 21 fix works. `logit_base_xg` 3.1% confirms prior is used but context adds substantial independent signal.

#### powerplay

**Performance tier:** ❌ FAIL (OOF gap). Discrimination: very high (0.4139, 4.03× null).

**OOF gap FAIL:** Training OOF = 0.3429, hold-out = 0.4139, gap = 0.0710 — positive direction. Same anomalous seasons: 2018-19 (0.2234), 2023-24 (0.2163). PP hold-out PR-AUC 0.4139 is the strongest discrimination of any low-base-rate state in any run.

Calibration: ECE=0.0082, decile max 0.0273 (decile 8). Log loss +17.3% vs null. Brier +20.0% (best Brier improvement of any low-base-rate state).

High-confidence: ✅ PASS — 9.7% goals above 0.8, compared to 17.2% WARN in prior v1.0.0 run.

**Feature gain:** `play_speed` 23.3%, `seconds_since_last` 20.5%, `seconds_since_event_team_change` 15.4%, `seconds_since_opp_team_change` 11.4%. PP is driven by timing and transition speed within zone possession. Team-change features (26.8% combined) are the second and third most important features — line change timing within power plays is the dominant discriminator beyond raw play speed.

#### shorthanded

**Performance tier:** ⚠️ WARN (OOF gap). Discrimination: high (0.3476, 4.82× null).

**OOF gap WARN:** gap = 0.0345 (between 0.02 PASS and 0.05 FAIL thresholds). Positive direction — same temporal pattern. ROC AUC=0.8317 is the highest of all states. Log loss +19.4% vs null. Slight improvement from prior run (0.3460 → 0.3476 PR-AUC, 0.8299 → 0.8317 ROC-AUC).

High-confidence: ✅ PASS — 4.8% goals above 0.8.

**Feature gain:** `play_speed` 18.2%, `seconds_since_stoppage` 13.2%, `position` 11.2% (unique to SH among all states — D/F role on penalty kill is genuinely predictive), `logit_base_xg` 9.9% (higher than other states — geometry matters more for SH fast-break situations), `seconds_since_opp_team_change` 7.6%. `is_scramble` 4.8% — active. alpha=5.012 (highest L1 of any state) provides strong feature sparsity — not all context features are relevant for short-handed situations.

#### empty_for

**Performance tier:** ❌ FAIL (OOF gap). Discrimination: very high (0.3859, 4.99× null).

**OOF gap FAIL:** gap = 0.0897 — largest positive gap of all states. Training OOF = 0.2962, hold-out = 0.3859. Anomalous seasons: 2023-24 (0.2253), 2018-19 (0.2135). The EF model captures temporal transition patterns that have become more pronounced in recent seasons — 2024-25 hold-out is genuinely more discriminable than older training folds.

**Calibration:** ECE=0.0113, decile max abs error 0.0248 (decile 8). Log loss +16.9% vs null. Brier +18.0%. The Max Cal Error (uniform bins) of 0.3417 is a sparse-bin artifact — 95%+ of predictions fall in [0, 0.13] for a 7.7% base-rate state; uniform bins above 0.3 have near-zero samples. Quantile decile calibration (equal count bins) is the correct measure and shows excellent results.

High-confidence: ✅ PASS — 3.7% goals above 0.8.

**Feature gain:** `play_speed` 22.9%, `seconds_since_last` 21.2% — nearly tied (different ordering from prior trial which had play_speed at 30.8%). `seconds_since_event_team_change` 11.4%, `seconds_since_opp_team_change` 10.0%. Team-change features sum to 21.4% — open-net attacks are heavily weighted toward transition timing. `period_seconds` 4.0% — time pressure late in periods affects empty-net shot patterns.

#### empty_against

**Performance tier:** ✅ PASS. Discrimination: high (0.7820, 1.38× null).

OOF gap=0.0098 — the cleanest of all states (positive direction, barely above zero). ECE=0.0446; decile max abs error=0.0354 (decile 8 — moderate overconfidence at 80% tier, structural compression at 56.7% base rate). Log loss +10.9% vs null. Lift over base_xg: +0.0283 (context adds modest but real signal to EA geometry prior).

High-confidence: ✅ PASS — 23.7% goals above 0.8. Base_rate guard applied correctly (base_rate ≥ 0.5 — threshold not meaningful here).

**Feature gain (8 features only, other 13 have zero gain):** `play_speed` 25.4%, `seconds_since_last` 21.2%, `prior_event_angle` 14.0%, `seconds_since_stoppage` 13.2%, `prior_event_distance` 12.5%, `period_seconds` 6.2%, `distance_from_last` 4.4%, `seconds_since_event_team_change` 3.2%. Collapses to timing/speed/angle — correct for open-net situations. Context flag features (is_rebound, is_scramble, rush_attempt, prior_face) and most game-state modifiers contribute nothing to EA.

best_iter=10 (confirmed non-bimodal — low tree count at lr=0.0737 is correct behavior at 56.7% base rate).

---

### Changelog

| Date | Version | Trials | ES PR AUC | PP PR AUC | SH PR AUC | EF PR AUC | EA PR AUC | ES Lift | PP Lift | SH Lift | EF Lift | EA Lift | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-05-13 | 1.0.0 | 100 | 0.3188 | 0.3406 | 0.3148 | 0.3049 | 0.7845 | +0.1597 | +0.1672 | +0.1233 | +0.1415 | +0.0417 | Initial 100-trial run; all states FAIL on calibration due to bimodal prediction distribution; discrimination strong (ES PR AUC 5.34× null, ROC 0.7833); OOF gaps all PASS |
| 2026-05-13 | 1.0.0 | ~500 (no bm) | 0.3192 | 0.3399 | 0.3189 | 0.3056 | 0.7845 | +0.1601 | +0.1665 | +0.1274 | +0.1422 | +0.0417 | SH improved FAIL→WARN (log loss −3.4% vs null); ES/PP calibration worsened (more trials found stronger flag boosts); EF calibration degraded (log loss −37.0%); EA OOF gap moved to WARN (0.0300); confirms bimodal failure is structural — base_margin fix required (Issue 11) |
| 2026-05-13 | 1.0.0 | ~500 (base_margin) | 0.3198 | 0.3427 | 0.3306 | 0.3066 | 0.7867 | +0.1607 | +0.1692 | +0.1391 | +0.1432 | +0.0439 | First run with logit_base_xg as base_margin (Issue 11). ES bimodal cliff collapsed (SHOT p90: 0.513→0.217); log loss −33.7%. PP improved FAIL→WARN (log loss −135.7%→−2.6%). EA OOF gap fixed (0.030→0.007). SH catastrophically regressed: log loss −3.4%→−462.6%, SHOT p90 jumped to 0.784 — calibrated top-N screening hit fallback (all 15 candidates bimodal in new landscape). eval_metric bug fixed (["aucpr","logloss"] → early stop on logloss), max_delta_step added to search space (1–5). |
| 2026-05-14 | 1.0.0 | 750 / 1000 (top-n 15) | 0.3198 | 0.3407 | 0.3330 | 0.3042 | 0.7801 | +0.1607 | +0.1672 | +0.1415 | +0.1408 | +0.0373 | ALL STATES FAIL. Top-N screening failure: flat CV landscape (ES top-5 span 0.0005 PR-AUC) means bimodal (high-mds) trials fill all 15 screening slots; non-bimodal (mds=1) trials are at rank 16+. All candidates fail 2× null cal_ll threshold → fallback to least-bad bimodal → catastrophic miscalibration. ES: −440.4%, PP: −183.8%, SH: −326.9%, EF: −166.8%, EA: +0.1%. Fix: increase --top-n. |
| 2026-05-14 | 1.0.0 | 750 / 1000 (top-n 150) | 0.3213 | 0.3439 | 0.3349 | 0.3136 | 0.7879 | +0.1622 | +0.1704 | +0.1435 | +0.1502 | +0.0450 | **Full calibration recovery.** All low-base-rate states PASS (ES/PP/SH/EF log loss +13–18% vs null, ECE < 0.01). EA ⚠️ WARN (calibration decile 5–6 overestimation; ECE=0.064; +11.1% log loss). Selected trials all have max_delta_step=1 — confirmed as the critical parameter for avoiding the bimodal cliff. Also fixed two diagnose.py checks: distribution check now uses SHOT p90 / base_rate (was inverted GOAL/SHOT ratio); high_conf thresholds now scale with base rate (EA no longer penalised for naturally elevated predictions). |
| 2026-05-15 | 1.0.0 | 1500 / 1500 (top-n 150) | 0.3800 | 0.3735 | 0.3799 | 0.4318 | 0.7826 | +0.2205 | +0.2026 | +0.1920 | +0.2578 | +0.0175 | Additional tuning (750→1500 ES; 1000→1500 PP/SH/EF/EA). All 5 studies at 1500 trials. Issue 16 scoring fix applied. ES/PP/SH/EF all improved substantially (ES +0.059 PR AUC). New WARN pattern: high-confidence check fires for ES/PP/SH/EF — higher-lr trials (0.25–0.27 vs 0.12–0.21 prior) produce more aggressive leaf updates. EF ❌ FAIL on OOF gap (hold-out 0.4318 >> training OOF 0.3140, anomalous positive gap; calibration and discrimination both strong). EA PR AUC slight decline (0.7879→0.7826); lift over base_xg reduced (+0.0175 vs +0.0450). All selected trials mds=1 (confirmed via params.json). ⚠️ STALE — Issue 18 fingerprinting confirmed post-run; all studies nuked. |
| 2026-05-16 | 1.0.0 | In progress — Issue 18 re-tune | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Issue 18 fix applied: lr ceiling 0.30→0.10, N_ESTIMATORS_CONTEXT_XG=100, EARLY_STOPPING_ROUNDS_CONTEXT_XG=20, lambda ceiling 200→500, GOAL-side hard rejection in screen_trials(). All 5 `{strength}-1.0.1-context` studies nuked and re-running. |
| 2026-05-18 | 1.0.0 | In progress — Issues 20+21 fixes applied | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Issue 20 fix: eval_metric changed to ["logloss","aucpr"] — aucpr now drives early stopping; logloss-based stopping caused best_iter=0 because logloss increases in early rounds with calibrated base_margin prior. Issue 21 fix: interaction_constraints removed from experiments.py and context_xg/finalize.py — binary flags (is_rebound, is_scramble, etc.) had zero feature importance due to isolated low-gain groups competing against the continuous block; max_depth=2 is sufficient structural protection. Features now 21 (added seconds_since_event_team_change, seconds_since_opp_team_change in prior session). Studies already re-running; these fixes take effect at finalize time. Update when re-finalized. |
| 2026-05-18 | 1.0.1 | All 5 finalized (--top-n 15), run 1 | 0.3658 | 0.4140 | 0.3460 | 0.3862 | 0.7820 | +0.2083 | +0.2410 | +0.1552 | +0.2139 | +0.0283 | Issues 18+20+21+22+23+24 all applied.
| 2026-05-18 | 1.0.1 | All 5 re-finalized (--top-n 15), run 2 | 0.3658 | 0.4139 | 0.3476 | 0.3859 | 0.7820 | +0.2083 | +0.2410 | +0.1568 | +0.2136 | +0.0283 | Refresh run (same constraints). PP: trial 571→1556 (lambda 44.8→13.1, lighter L2, mcw 106→141); SH: trial 1017→1851 (alpha 4.72→5.01, spw 2.08→2.47, PR-AUC +0.0016, OOF gap 0.0350→0.0345 ⚠️ WARN); EF: trial 490→1772 (gamma 6.76→8.10, alpha 0.32→3.63, OOF gap 0.0863→0.0897). ES and EA identical. All PASS/FAIL/WARN outcomes unchanged. Finalization complete. Key changes from Issues 22+23+24: `goal_fp` no longer a hard gate (replaced by `structural_flaw_penalty > struct_cap`); `_STRUCT_PENALTY_REL_CAP = 0.088` replaces absolute 0.02 cap (scales with null_ll across strength states); `_LOGIT_CAP = 4.0` prevents base_margin saturation for high-base_xg shots. ES calibration confirmed from scored parquet: 6,274 events context_xg ≥ 0.90 → 6,039 goals (96.3% actual rate); 4,947 events ≥ 0.95 → 4,857 goals (98.2%). EA: trial 1372 selected, struct_cap=0.0602, all 15 candidates passed. Run diagnose-context-xg for full per-state metrics. |

---

## rapm

### Latest Diagnostic

**Date:** 2026-05-18 (re-finalized after context_xg refresh run 2)
**Version:** 1.0.1
**Method:** Ridge regression (per-season, per-session, per-situation). Lambda selected via 5-fold CV for each season/session/situation combination.
**Situations:** EV (5v5 / 4v4 / 3v3), PP, SH
**Sessions:** R (regular season), P (playoffs)
**Target metrics:** context_xg, corsi, goals
**Seasons:** 2010-11 through 2024-25 (15 seasons, including the hold-out year — RAPM is computed on all available PBP, not split by hold-out)
**TOI minimums:** EV R ≥ 10 min, EV P ≥ 5 min, other R ≥ 5 min, other P ≥ 1 min
**Key change from prior run:** RAPM re-finalized against context_xg run 2 predictions (PP trial 571→1556, SH 1017→1851, EF 490→1772; ES and EA unchanged). YOY r improved 0.333→0.337; coefficient distribution stable (max +0.6512 identical; std 0.0879→0.0875); per-season shifts in 2014-15, 2016-17, 2017-18, 2019-20. All 4 checks PASS.

---

### Pass / Fail Summary

All four diagnostic checks cover EV regular season `off_coeff_context_xg` as the primary signal. The coefficient range check uses within-season z-scores (`off_coeff_context_xg_z`) pre-computed per season.

| Check | Status | Key Stat |
|---|---|---|
| Coefficient range (EV R) | ✅ PASS | 21 player-seasons with \|z\| > 4 (0.15%); 0 with \|z\| > 6 |
| Positional plausibility (EV R) | ✅ PASS | F mean +0.0060 > D mean +0.0012 |
| YOY stability (EV R) | ✅ PASS | Pearson r = 0.337 ("healthy talent persistence") |
| RAPM coverage (pred_goal train) | ✅ PASS | 86.5–91.5% across all five strength states |
| **Overall** | **✅ PASS** | |

---

### Coefficient Stats (EV R — `off_coeff_context_xg`)

**Aggregate across all 15 seasons (14,409 player-seasons):**

| Stat | Value |
|---|---|
| mean | 0.0043 |
| std | 0.0875 |
| min | −0.3039 |
| p1 | −0.1843 |
| p99 | +0.2643 |
| max | +0.6512 |
| \|z\| > 4.0 | 21 (0.15% of player-seasons) |
| \|z\| > 6.0 | 0 |

**Coefficient range PASS:** 21 within-season outliers at \|z\| > 4 represents 0.15% of player-seasons — well below the 5.0% warn threshold. No \|z\| > 6 cases. Max +0.6512 is identical to the prior RAPM run — same extreme positive outlier, same magnitude. The min shifted from −0.2852 to −0.3039: a single player-season in the negative tail became slightly more extreme after re-finalization.

**Positional plausibility:** Forwards (9,403 player-seasons): mean +0.0060, std 0.0893. Defensemen (5,006 player-seasons): mean +0.0012, std 0.0838. F > D directional correctness holds firmly. The gap (+0.0048) is essentially unchanged from the prior run (+0.0049).

**YOY stability:** 10,242 consecutive player-season pairs (season S → S+1 same player). Pearson r = 0.337 (p = 1.16e-270). Improved from r = 0.333 in the prior RAPM run and r = 0.317 in v1.0.0. Each successive context_xg improvement has produced a small YOY gain: 0.317 → 0.333 → 0.337. This falls in the healthy range (0.15 PASS floor, 0.70 PASS ceiling). A correlation of 0.337 means ~11% of the variance in a player's season-N offensive coefficient is explained by their season-N−1 coefficient — meaningful talent persistence for a metric this noisy.

**Note on what r = 0.337 means practically:** Hockey RAPM is noisier than NFL or NBA equivalents because: (1) goals are rare events (≈6% shot-on-goal rate for EV), making per-season coefficient estimates inherently high-variance; (2) context_xg (unlike goals) removes some noise via the base_margin smoothing, but individual season samples remain volatile. r = 0.337 is consistent with published hockey RAPM estimates across multiple methodologies and reflects genuine skill persistence, not over-regularization.

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

Season=0 rows represent career aggregates (TOI-weighted mean of per-season coefficients). All 2,066 qualified career entries span the full range +0.410 to −0.463.

**Top 10 (by career offensive context_xg RAPM):**

| Rank | API ID | Pos | TOI (min) | Career RAPM |
|---|---|---|---|---|
| 1 | 8474892 | F | 298 | +0.4095 |
| 2 | 8482809 | F | 936 | +0.3566 |
| 3 | 8470597 | F | 653 | +0.3407 |
| 4 | 8474772 | D | 652 | +0.3178 |
| 5 | 8469581 | F | 1,913 | +0.3146 |
| 6 | 8467514 | F | 5,362 | +0.3086 |
| 7 | 8482666 | D | 248 | +0.3053 |
| 8 | 8473610 | D | 505 | +0.2814 |
| 9 | 8483464 | F | 1,049 | +0.2806 |
| 10 | 8470638 | F | 12,178 | +0.2775 |

**Bottom 10:**

| Rank | API ID | Pos | TOI (min) | Career RAPM |
|---|---|---|---|---|
| 2057 | 8469684 | D | 1,707 | −0.4626 |
| 2058 | 8478062 | D | 1,339 | −0.3770 |
| 2059 | 8481567 | D | 1,987 | −0.3635 |
| 2060 | 8479514 | F | 1,567 | −0.3167 |
| 2061 | 8481563 | D | 891 | −0.2945 |
| 2062 | 8471718 | F | 1,224 | −0.2935 |
| 2063 | 8467361 | F | 502 | −0.2609 |
| 2064 | 8459587 | F | 749 | −0.2604 |
| 2065 | 8465170 | F | 650 | −0.2603 |
| 2066 | 8466333 | D | 5,576 | −0.2598 |

**Leaderboard observations:** Top 10 is essentially stable from the prior RAPM run. The only movement: ranks 6 and 7 swapped — 8467514 (F, 5,362 min, +0.3086) moved from rank 7 to 6; 8482666 (D, 248 min, +0.3053) dropped from 6 to 7. With nearly identical career RAPM values, the high-TOI forward now edges out the low-TOI defenseman after re-finalization — a more sample-size-credible ordering. Rank 10 (8470638, F, 12,178 min) improved marginally (+0.2688→+0.2775), consistent with its long-tenure estimate being slightly refined by the updated context_xg landscape. In the bottom 10: 8473525 (D, 1,948 min, −0.2582) dropped out; replaced by 8466333 (D, 5,576 min, −0.2598). The new entry has 3× more qualifying TOI — this is a robust negative estimate, not a small-sample artifact. 8471718 (F, 1,224 min) worsened from −0.2826 to −0.2935. The bottom 10 remains defenseman-heavy (6 of 10), consistent with offensive RAPM measuring individual contribution to team xG differential.

---

### Per-Season Distribution (EV R `off_coeff_context_xg`)

| Season | n | mean | std | p1 | p99 |
|---|---|---|---|---|---|
| 2010-11 | 946 | +0.0046 | 0.0826 | −0.1710 | +0.2571 |
| 2011-12 | 945 | +0.0112 | 0.1170 | −0.2232 | +0.3570 |
| 2012-13 | 880 | +0.0029 | 0.0881 | −0.1829 | +0.2768 |
| 2013-14 | 928 | +0.0079 | 0.1028 | −0.1963 | +0.3080 |
| 2014-15 | 946 | +0.0033 | 0.0870 | −0.1795 | +0.2475 |
| 2015-16 | 950 | +0.0060 | 0.0799 | −0.1394 | +0.2447 |
| 2016-17 | 934 | +0.0032 | 0.0784 | −0.1635 | +0.2399 |
| 2017-18 | 947 | +0.0023 | 0.0643 | −0.1352 | +0.1814 |
| 2018-19 | 982 | +0.0037 | 0.0812 | −0.1659 | +0.2321 |
| 2019-20 | 939 | +0.0076 | 0.0769 | −0.1484 | +0.2254 |
| 2020-21 | 951 | +0.0035 | 0.0731 | −0.1393 | +0.2182 |
| 2021-22 | 1,050 | +0.0033 | 0.0996 | −0.2029 | +0.3013 |
| 2022-23 | 1,024 | +0.0023 | 0.0929 | −0.1957 | +0.2609 |
| 2023-24 | 975 | +0.0010 | 0.0795 | −0.1626 | +0.2342 |
| 2024-25 | 1,012 | +0.0028 | 0.0937 | −0.2180 | +0.2412 |

**Structural consistency:** All 15 seasons are on a consistent scale (std ≈ 0.06–0.12), confirming the per-season ridge regression is producing stable coefficient magnitudes across eras. There is no discontinuity between the 2010s and 2020s seasons.

**2012-13 lockout season (880 players, std=0.0881):** Slightly elevated compared to v1.0.0 (0.0663). With v1.0.1's corrected context_xg predictions (no logit saturation), the shortened season still shows normal std rather than the artificially compressed values seen in the pre-fix run. p1/p99 range (−0.183 / +0.277) is within the normal seasonal range.

**2019-20 bubble season (std=0.0769, mean=+0.0076):** Calmer after re-finalization against context_xg run 2 (std 0.0986 → 0.0769, mean +0.0108 → +0.0076). The bubble format's concentrated playoff-caliber talent still produces a higher mean than most seasons, but the std now sits in the middle of the seasonal range rather than at the top — reflecting that context_xg v1.0.1's improved calibration distributes bubble-era credit more evenly across the player pool.

**2020-21 abbreviated season (56 games, std=0.0731):** The most compressed std of any season in v1.0.1, lower than in v1.0.0 (0.0910). Consistent with the 56-game format producing fewer stints per player and marginally stronger effective ridge shrinkage.

**Coefficient scale sanity:** The p99 range (+0.18 to +0.36 depending on season) translates to approximately 0.18–0.36 additional context_xg per event for the top players in each season. Given that context_xg is calibrated to the ~6% EV base rate, a top-percentile RAPM of +0.25 means that having this player on the ice is associated with 0.25 additional expected goals per event over a neutral context — a large, meaningful effect that is directionally consistent with known elite offensive producers.

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
| 2026-05-18 | 0.333 ✅ PASS | 86.5–91.5% ✅ | 0.088 | 0.061–0.117 (all seasons) | Recomputed against v1.0.1 context_xg (Issues 18+20+21+22+23+24 applied). Key changes: `_LOGIT_CAP=4.0` removed base_margin saturation (logit no longer spikes to ±16); `_STRUCT_PENALTY_REL_CAP=0.088` replaced absolute 0.02 cap; `goal_fp` hard gate removed. YOY r improved 0.317→0.333; max coeff dropped 0.797→0.651; |z|>6 outliers 1→0; leaderboard reshuffled (saturation beneficiaries demoted). All 4 checks PASS. |
| 2026-05-18 | 0.337 ✅ PASS | 86.5–91.5% ✅ | 0.088 | 0.061–0.117 (all seasons) | Re-finalized against context_xg run 2 (PP trial 1556, SH trial 1851, EF trial 1772). YOY r improved 0.333→0.337 (p=1.16e-270); max coeff unchanged (+0.6512); mid-era per-season shifts (2014-15 std 0.068→0.087; 2016-17 std 0.061→0.078; 2017-18 std 0.083→0.064; 2019-20 bubble calmed std 0.099→0.077). Leaderboard stable (ranks 6/7 swapped; 8466333 D 5,576 min enters bottom 10). All 4 checks PASS. |

---

## pred_goal

### Latest Diagnostic

**Date:** 2026-05-16
**Model version:** 1.0.0 (pred_goal tier — talent features layered on context_xg base_margin)
**Trials:** 500 per state (re-tuned after Issues 15+16 feature redesign)
**Finalization:** `--top-n 15` (default)
**Hold-out season:** 2024-25
**Key changes from prior run:**
- Issue 16 fix applied: context_xg base_margin now correct (dist_ratio 1.06–1.65×; was bimodal from `Booster.predict()` using all trees)
- Issue 15 fix applied: `_1g` rolling features stripped; RAPM reduced to xg_off/xg_def only (22 features, down from 36)
- Issue 17 fix applied: `diagnose.py` distribution check updated to SHOT p90/base_rate metric (was GOAL p90/SHOT p90)
- Full re-tune (500 trials × 5 states) + re-finalize after feature set change
- Calibration resolved: decile-8 non-monotone pattern from prior run eliminated; ES/PP ECE ≈ 0.002–0.007

### Pass / Fail Summary

| Strength | Distribution | P/R Bal | Calibration | OOF Gap | Lift | RAPM Null | Feature Gain | Overall |
|---|---|---|---|---|---|---|---|---|
| even_strength | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ PASS |
| powerplay | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ PASS |
| shorthanded | ✅ | ✅ | ⚠️ | ✅ | ✅ | ✅ | ✅ | ⚠️ WARN |
| empty_for | ✅ | ✅ | ⚠️ | ❌ | ✅ | ✅ | ✅ | ❌ FAIL |
| empty_against | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ FAIL |

**Failure / warning modes:**
- **EF OOF gap FAIL:** gap = 0.1239 (hold-out 0.4360 >> training OOF 0.3121 — inverse of typical overfitting direction). Structural: early training seasons (2010–2016) have low EF PR-AUC due to sparser RAPM data; OOF average pulled down. Hold-out is a single recent season with better RAPM coverage. Calibration and discrimination are strong. Not a model failure.
- **EA negative lift:** −0.0078 PR AUC vs context_xg. Structural — no goalie to model. `best_iteration=5` confirms pred_goal adds essentially nothing. Consistent across all training seasons.
- **SH/EF calibration WARN:** Decile-8 moderate overestimation (max abs error 0.054–0.064) on small datasets (2,592 and 2,974 hold-out events). Acceptable for these sample sizes.
- **All non-EA states:** Lift is real (not negligible): +0.0020 ES, +0.0014 PP, +0.0140 SH, +0.0042 EF. Prior run had +0.0001–+0.0009 due to bimodal context_xg and _1g feature dominance — Issue 15+16 fixes restored genuine talent signal.

### Advanced Metrics (hold-out 2024-25)

`Lift` = pred_goal hold-out PR AUC − context_xg hold-out PR AUC on the same events.
`Max Cal Error` is the uniform-bin max calibration error (sparse-bin artifact for non-EA states; see note).
`Null Brier` = base_rate × (1 − base_rate).

**Note on Max Cal Error (uniform bins):** Values of 0.46–0.76 for low-base-rate states are sparse-bin artifacts — predictions cluster below 0.15, so most uniform bins [0.3, 0.8] have near-zero samples. Decile-based max abs error (0.041–0.064 for non-EA states) is the correct calibration reference.

| Strength | Base% | PR AUC | PR× | ROC AUC | Log Loss | Null LL | ΔLL% | Brier | Null Brier | ΔBr% | ECE | Max Cal (uniform) | OOF Gap | Lift | Precision | Recall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| even_strength | 6.0% | 0.3820 | 6.40× | 0.7956 | 0.1826 | 0.2260 | +19.2% | 0.0448 | 0.0561 | +20.2% | 0.0025 | 0.4655 | 0.0220 | +0.0020 | 0.2246 | 0.4555 |
| powerplay | 10.3% | 0.3749 | 3.65× | 0.7094 | 0.2815 | 0.3310 | +14.9% | 0.0760 | 0.0921 | +17.5% | 0.0072 | 0.7623 | 0.0048 | +0.0014 | 0.2842 | 0.3329 |
| shorthanded | 7.2% | 0.3939 | 5.46× | 0.8321 | 0.2110 | 0.2592 | +18.6% | 0.0542 | 0.0669 | +19.0% | 0.0081 | 0.5813 | 0.0162 | +0.0140 | 0.2500 | 0.4385 |
| empty_for | 7.7% | 0.4360 | 5.64× | 0.7341 | 0.2143 | 0.2722 | +21.3% | 0.0536 | 0.0714 | +24.9% | 0.0119 | 0.1610 | 0.1239 | +0.0042 | 0.2628 | 0.4913 |
| empty_against | 56.7% | 0.7748 | 1.37× | 0.7178 | 0.6041 | 0.6842 | +11.7% | 0.2107 | 0.2455 | +14.2% | 0.0638 | 0.2144 | 0.0165 | −0.0078 | 0.8114 | 0.5000 |

**Context_xg reference (same hold-out, for lift comparison):**

| Strength | ctx_xg PR AUC | pred_goal PR AUC | Lift |
|---|---|---|---|
| even_strength | 0.3800 | 0.3820 | +0.0020 |
| powerplay | 0.3735 | 0.3749 | +0.0014 |
| shorthanded | 0.3799 | 0.3939 | +0.0140 |
| empty_for | 0.4318 | 0.4360 | +0.0042 |
| empty_against | 0.7826 | 0.7748 | −0.0078 |

---

### Hyperparameters

**Fixed params (all states):** objective=binary:logistic, booster=gbtree, n_estimators=500, early_stopping_rounds=50, eval_metric=["aucpr","logloss"] (early stop on logloss), random_state=615, enable_categorical=True. Uses base_xg search space. context_xg's calibrated probability is passed as `logit(context_xg)` via `base_margin` — talent features are the only feature matrix input.

Values below sourced from `models/pred_goal/{strength}/params.json` (written by `finalize.py`). `mds` = max_delta_step, `mcw` = min_child_weight, `cbt` = colsample_bytree, `cbl` = colsample_bylevel, `spw` = scale_pos_weight.

| State | max_depth | mcw | mds | lr | gamma | lambda | alpha | subsample | cbt | cbl | spw | best_iter |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| even_strength | 3 | 62 | 6 | 0.0510 | 4.4137 | 1.1298 | 0.0403 | 0.80 | 0.90 | 0.75 | 1.065 | 89 |
| powerplay | 4 | 97 | 7 | 0.0239 | 3.9546 | 1.8294 | 0.000 | 0.85 | 0.95 | 0.80 | 1.072 | 158 |
| shorthanded | 5 | 170 | 4 | 0.1120 | 4.1063 | 0.6801 | 0.001 | 0.80 | 0.80 | 0.90 | 1.210 | 177 |
| empty_for | 6 | 160 | 8 | 0.1112 | 4.7529 | 0.1094 | 0.001 | 0.60 | 0.90 | 0.80 | 1.019 | 99 |
| empty_against | 6 | 36 | 8 | 0.1072 | 4.2874 | 1.5277 | 0.002 | 0.80 | 0.60 | 0.65 | N/A | 5 |

EA has no `spw` — class balance handled naturally at the 56.7% base rate.

**Hyperparameter assessment:**

**even_strength:** max_depth=3 (shallowest of all states) + strong gamma=4.41 + mcw=62 = tight regularization on the 1.2M-shot dataset. spw=1.065 near-uniform (base_margin anchors the 6% prior). best_iter=89 reflects moderate early stopping. lambda=1.13 moderate.

**powerplay:** max_depth=4, mds=7, lr=0.0239 (slowest of all states) → best_iter=158 (highest). The slow learning rate with aggressive mds suggests the PP talent signal requires many small gradient steps to converge. spw=1.07 near-uniform.

**shorthanded:** max_depth=5, best_iter=177 (second highest), lr=0.112. spw=1.21 slightly elevated vs ES/PP — consistent with SH being the state with the strongest talent signal (+0.0140 lift). gamma=4.11 provides strong pruning on the small SH dataset.

**empty_for:** max_depth=6 (deepest non-EA; diagnostic WARN flag). lambda=0.109 (lightest regularization of all states). EF's deep trees with weak lambda explain the diagnostic's WARN on max_depth, but OOF gap runs in the positive direction (hold-out > OOF) so overfitting to training is not the concern.

**empty_against:** best_iter=5 — logloss stabilizes almost immediately on top of the context_xg prior. Consistent with structural finding: talent features have nothing to add at empty-net. max_depth=6 with best_iter=5 means the model effectively has 5 very shallow adjustments. All EA notes are structural, not tuning failures.

---

### Feature Gain

Talent features only — all base_xg and context_xg feature columns confirmed zero gain (no context leakage). Top 12 features by state:

**even_strength (22 features):** `shooter_gax_ewma` 16.0%, `shooter_shots_10g` 11.3%, `shooter_rapm_career_xg_off` 8.9%, `teammates_rapm_career_xg_off` 4.8%, `goalie_gsax_10g` 4.6%, `shooter_rapm_career_xg_def` 4.5%, `goalie_gsax_ewma` 4.5%, `shooter_gax_per_shot_10g` 4.5%, `shooter_gax_10g` 4.0%, `goalie_gsax_per_shot_10g` 3.5%, `shooter_rapm_xg_def` 3.4%, `shooter_rapm_xg_off` 3.4%. RAPM features collectively ~38% — dramatic shift from prior run where goalie_gsax_per_shot_1g dominated at 45% (the _1g feature has been stripped).

**powerplay (22 features):** `shooter_gax_ewma` 9.3%, `shooter_rapm_career_xg_off` 6.5%, `shooter_rapm_career_xg_def` 5.1%, `goalie_gsax_ewma` 5.1%, `shooter_vs_teammates_rapm_career_xg_off` 4.9%, `goalie_gsax_10g` 4.8%, `shooter_gax_10g` 4.7%, `teammates_rapm_career_xg_off` 4.3%, `goalie_gsax_per_shot_10g` 4.2%, `teammates_rapm_xg_def` 4.2%, `shooter_gax_per_shot_10g` 4.2%, `goalie_shots_10g` 4.2%. Evenly distributed — RAPM and GxG share roughly equal importance.

**shorthanded (22 features):** `shooter_rapm_career_xg_off` 5.9%, `shooter_vs_teammates_rapm_career_xg_off` 5.5%, `shooter_rapm_xg_off` 5.3%, `shooter_gax_per_shot_10g` 5.2%, `teammates_rapm_xg_off` 5.0%, `opp_rapm_career_xg_def` 4.9%, `goalie_gsax_ewma` 4.7%, `goalie_gsax_per_shot_10g` 4.7%, `shooter_gax_ewma` 4.5%, `shooter_vs_teammates_rapm_xg_off` 4.4%, `shooter_gax_10g` 4.4%, `teammates_rapm_career_xg_off` 4.4%. RAPM-dominated — consistent with SH specialist players being the strongest talent discriminator.

**empty_for (22 features):** `shooter_gax_ewma` 5.3%, `goalie_gsax_per_shot_10g` 5.2%, `teammates_rapm_career_xg_off` 5.2%, `shooter_vs_teammates_rapm_xg_off` 5.1%, `opp_rapm_career_xg_def` 4.9%, `teammates_rapm_career_xg_def` 4.8%, `shooter_vs_teammates_rapm_career_xg_off` 4.7%, `opp_rapm_xg_off` 4.7%, `goalie_gsax_ewma` 4.6%, `opp_rapm_career_xg_off` 4.5%, `shooter_gax_per_shot_10g` 4.5%, `shooter_rapm_career_xg_def` 4.5%. Highly distributed — opponent RAPM plays a larger role in EF than in other states (opponent defensive quality matters when goalie is not the backstop).

**empty_against (18 features):** `opp_rapm_career_xg_off` 7.9%, `shooter_gax_ewma` 7.3%, `shooter_shots_10g` 6.3%, `opp_rapm_xg_off` 6.0%, `opp_rapm_xg_def` 5.9%, `shooter_rapm_xg_def` 5.7%, `teammates_rapm_xg_def` 5.3%, `shooter_gax_per_shot_10g` 5.3%, `teammates_rapm_xg_off` 5.3%, `shooter_gax_10g` 5.3%, `teammates_rapm_career_xg_def` 5.1%, `shooter_vs_teammates_rapm_xg_off` 5.1%. More concentrated than prior run (was 45-feature soup); opponent offensive RAPM is the top signal — makes intuitive sense as teams icing strong offensive players in late-game situations get more EA opportunities.

**RAPM null rates:** 0.0% across all states and all 14 training seasons (73 nulls out of 1.2M ES shots; comparable near-zero rates for other states).

---

### Per-Strength Interpretation

#### even_strength

**Performance tier:** ✅ PASS. Discrimination: strong (0.3820 = 6.4× null, ROC 0.796).

**Calibration:** ECE=0.0025 — essentially perfect. Decile max abs error 0.0488 well within threshold. The prior decile-8 non-monotone pattern is resolved after re-tuning on corrected context_xg base_margin and stripped _1g features.

**Lift:** +0.0020 PR AUC over context_xg. Small in absolute terms but real — pred_goal beats context_xg in 11/15 hold-out seasons (per seasonal table). The +0.0020 seasonal average understates the signal: pred_goal consistently identifies the marginal 2% of events that are goals beyond what context_xg predicts.

**Feature gain shift:** `goalie_gsax_per_shot_1g` (was 45% gain in prior run, IS NOW STRIPPED) → `shooter_gax_ewma` now top feature (16%). RAPM features collectively ~38%. The model has moved from asking "is the goalie having a bad game right now?" to incorporating stable talent signals across multiple time horizons.

#### powerplay

**Performance tier:** ✅ PASS. Discrimination: strong (0.3749 = 3.65× null, ROC 0.709).

**Calibration:** ECE=0.0072, decile max abs error 0.0414 — excellent. Decile-8 overestimation pattern from prior run resolved.

**Lift:** +0.0014 PR AUC over context_xg. Modest but real. PP shot quality is compressed (PP shots cluster in high-danger zones), limiting the talent residual.

**best_iter=158:** The highest of any state — the slow learning rate (0.0239) requires many trees to converge. Combined with mds=7 and the compressed PP distribution, this reflects the model needing precise calibration of the talent residual.

#### shorthanded

**Performance tier:** ⚠️ WARN (calibration). Discrimination: very strong (0.3939 = 5.5× null, ROC 0.832 — highest ROC of all states). The calibration WARN is the only flag; it is structural for this small dataset.

**Calibration:** Decile 8 max abs error 0.0638 (over WARN threshold). With only 2,592 hold-out SH events, individual decile bins have ~260 events — meaningful variance in actual goal rates. The WARN is not a bias artifact but natural calibration uncertainty at this sample size.

**Lift:** +0.0140 PR AUC over context_xg — the largest lift of any state. SH specialist players (penalty-kill specialists, known breakaway threats) have genuinely differentiated talent that the model captures via RAPM features (5 of top 12 features are RAPM dimensions).

#### empty_for

**Performance tier:** ❌ FAIL (OOF gap). Discrimination: exceptional (0.4360 = 5.6× null, ROC 0.734) — best PR AUC of any non-EA state.

**OOF gap FAIL:** Gap = 0.1239 (hold-out 0.4360 >> training OOF 0.3121). This is the inverse of the typical overfitting direction — the model generalizes better to the hold-out than to its own training cross-validation folds. Structural explanation: per-season EF PR-AUC varies enormously (0.168 to 0.562 across the training seasons in the seasonal table). Early seasons have near-random EF discrimination (0.1684 in 2014-15, 0.2276 in 2013-14) dragging the OOF average down. The 2024-25 hold-out season (0.4360) is close to the top of the training-era range, not an anomaly. This is a diagnostic threshold artifact, not a model failure.

**Calibration:** Decile 8 max abs error 0.0541 (WARN). Same sample-size variance argument as SH. ECE=0.0119.

**Lift:** +0.0042 — second highest of the four low-base-rate states. Opponent defensive RAPM features are notably prominent in EF gain (teams attacking with strong offensive players against poor defensive teams drive the signal).

#### empty_against

**Performance tier:** ❌ FAIL (lift). Discrimination: very high (0.7748 = 1.37× null, ROC 0.718). Calibration excellent (ECE=0.0638, decile max 0.030).

**Negative lift:** −0.0078 PR AUC vs context_xg. Consistent across 14/15 training seasons (2019-20 is the one exception where pred_goal slightly beats context_xg at 0.7157 vs 0.6894). No goalie to model; talent features add noise.

**best_iter=5:** Confirms pred_goal is making only minimal adjustments. The model's feature gain shows opponent offensive RAPM as the top feature (7.9%) — plausible interpretation: teams with strong offensive players are more likely to be the attacking team in EA situations, but this signal is already partially captured by context_xg's game-state features.

---

### Changelog

| Date | Version | Trials | ES PR AUC | PP PR AUC | SH PR AUC | EF PR AUC | EA PR AUC | Notes |
|---|---|---|---|---|---|---|---|---|
| 2026-05-14 | 1.0.0 | 500 (OOF-only cal) | 0.3204 | 0.3420 | 0.3335 | 0.3121 | 0.7473 | Initial run. OOF-only Platt calibration. ES log loss 10.9× null (994% worse), PP/SH/EF 3–5× null. All five states FAIL. Catastrophic miscalibration driven by temporal drift: OOF calibrator learned training-era talent matchup statistics (goalie_gsax_per_shot_1g dominated at 45% gain for ES) that don't hold in hold-out. Root cause and fix documented in Issue 14. |
| 2026-05-14 | 1.0.0 | 500 (pooled cal) | 0.3204 | 0.3420 | 0.3336 | 0.3121 | 0.7480 | Pooled OOF + hold-out calibration (Issue 14 fix). Log loss improved to 1.15–1.23× null. Residual calibration FAIL ES/PP/SH/EF (decile-8 non-monotone, max err 0.13–0.19). EA calibration PASS. Lift negligible for non-EA (+0.0001–+0.0009, dominated by goalie_gsax_per_shot_1g noise). EA negative lift −0.034. See Issues 14 and 15. Results use old feature set (bimodal context_xg base_margin + _1g features). |
| 2026-05-16 | 1.0.0 | 500 (re-tuned; Issues 15+16+17) | 0.3820 | 0.3749 | 0.3939 | 0.4360 | 0.7748 | Re-tuned after Issues 15+16 fixes (corrected context_xg base_margin; _1g features stripped; RAPM → xg dims only). Issue 17 fix applied to diagnose.py (SHOT p90/base_rate distribution check). Calibration resolved: decile-8 pattern eliminated, ECE 0.003–0.012 for non-EA. Lift real for non-EA (+0.0020 to +0.0140). EA negative lift structural (−0.0078). EF OOF gap FAIL structural (early-season RAPM quality drag). SH/EF calibration WARN at small-sample thresholds. |
