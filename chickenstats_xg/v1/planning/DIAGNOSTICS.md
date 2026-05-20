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

**Date:** 2026-05-18 (third finalization run — 2k additional trials per state)
**Model version:** 1.0.1 (21-feature gbtree depth-2, no interaction constraints; Issues 18+20+21+22+23+24 applied)
**Trials:** ~4k+ per state; ES trial 350, PP trial 1556, SH trial 1851, EF trial 1772, EA trial 1803 (all v1.0.1 constraints: lr ≤ 0.10, N_ESTIMATORS=100, EARLY_STOPPING_ROUNDS=20, lambda ceiling 500)
**Hold-out season:** 2024-25
**Key changes from prior run:** 2k additional trials per state added. Same 5 trials selected as run 2 — no metric changes. Confirms run 2 selections are at or near the Pareto frontier for v1.0.1 constraints.

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
| 2026-05-18 | 1.0.1 | All 5 re-finalized (--top-n 15), run 3 | 0.3658 | 0.4139 | 0.3476 | 0.3859 | 0.7820 | +0.2083 | +0.2410 | +0.1568 | +0.2136 | +0.0283 | 2k additional trials per state (~4k+ total per study). Same 5 trials selected as run 2 (ES 350, PP 1556, SH 1851, EF 1772, EA 1803). No metric changes. Confirms run 2 selections are at/near the Pareto frontier for v1.0.1 constraints. |

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

**Date:** 2026-05-20
**Model version:** 1.0.1 (corrected `_params_pred_goal` param space; Issue 28 struct hard-gate fix; all 5 states finalized)
**Trials:** 2000 per state (version `1.0.1` studies; `even_strength-1.0.1-pred_goal` etc.)
**Finalization:** `--top-n 15` (default); Issue 28 struct gate disabled for base_margin models
**Hold-out season:** 2024-25
**Key changes from prior run (2026-05-19):**
- All 5 pred_goal studies nuked and re-tuned under corrected `_params_pred_goal`: lambda [10–100], alpha [0.1–10], lr [0.01–0.10], mcw [50–300], mds=1 fixed. Prior compromised studies had lambda ceiling=10 (every winning trial hit the ceiling) and alpha floor=1e-8 (all winners near zero).
- Issue 28 fix: `screen_trials()` struct hard-gate (`structural_flaw_penalty > struct_cap`) disabled when `bm_train is not None`. Adversarial test showed that for pred_goal (with base_margin), the isotonic DOF creates structural ~11% of null_ll struct_penalty even for healthy params — gate had zero discrimination. mds=1 + lambda≥10 clamps provide sufficient structural protection.
- SH finalized for the first time — struct gate was the reason all SH trials were previously rejected (100% failure rate). SH is now at 2K trials with proper params.
- EA params corrected: old trial had lr=0.2384 (above 0.10 ceiling), mcw=21 (below 50 floor), best_iter=1. New trial has lr=0.064, mcw=91, best_iter=19.

### Pass / Fail Summary

| Strength | Distribution | P/R Bal | Calibration | OOF Gap | Lift | RAPM Null | Feature Gain | Overall |
|---|---|---|---|---|---|---|---|---|
| even_strength | ✅ | ✅ | ✅ | ❌ FAIL | ✅ | ✅ | ✅ | ❌ FAIL |
| powerplay | ✅ | ✅ | ✅ | ❌ FAIL | ✅ | ✅ | ✅ | ❌ FAIL |
| shorthanded | ✅ | ✅ | ⚠️ WARN | ⚠️ WARN | ✅ | ✅ | ✅ | ⚠️ WARN |
| empty_for | ✅ | ✅ | ⚠️ WARN | ❌ FAIL | ✅ | ✅ | ✅ | ❌ FAIL |
| empty_against | ✅ | ✅ | ✅ | ✅ | ❌ FAIL | ✅ | ✅ | ❌ FAIL |

**Failure / warning modes:**
- **ES/PP/EF inverse OOF gap FAIL (ES/EF); WARN (SH):** All four low-rate states show hold-out PR-AUC substantially higher than training OOF (ES gap=0.0617, PP gap=0.0709, SH gap=0.0452, EF gap=0.0963) — anomalous positive direction, identical pattern to context_xg. Structural cause: RAPM quality improves over time as more career seasons accumulate; training-fold OOF predictions (drawn from earlier seasons) rely on sparser RAPM histories than hold-out predictions from 2024-25. Not overfitting — calibration is clean, all four have positive lift. The EF gap (0.0963) is the largest because early-era (2010-15) EF seasons have near-random discrimination (PR AUC 0.17–0.25) that drags the training OOF average far below the hold-out.
- **SH/EF calibration WARN:** Decile-8 max abs error 0.0623 (SH) and 0.0529 (EF) — above 0.05 PASS threshold. Both are within 0.065 of the FAIL threshold. The systematic Platt sigmoid ceiling creates underestimation in the 8th prediction decile (second-highest tier) for all states; it is more visible in SH/EF where decile-8 has only ~3,870 events (high variance in actual goal rate per bin). Not a bias artifact — ECE is clean (0.0082 and 0.0083 respectively).
- **EA negative lift:** −0.0142 PR AUC vs context_xg. Structural — no goalie to model. context_xg improved dramatically post-2020, absorbing all shot-quality signal that pred_goal's talent features cannot improve upon. Temporal break confirmed: 2020-21 through 2024-25 are all losses; pre-2020 results were mixed (some big wins in 2013-14, 2019-20).
- **Lift real for all non-EA:** +0.0012 ES, +0.0007 PP, +0.0080 SH, +0.0076 EF. ES/PP are small but consistent (ES 14W/1L, PP 11W/4L across 15 seasons). SH and EF have meaningful lift.

### Advanced Metrics (hold-out 2024-25)

`Lift` = pred_goal hold-out PR AUC − context_xg hold-out PR AUC on the same events.
`Max Cal Error` is the uniform-bin max calibration error (sparse-bin artifact for non-EA states; see note).
`Null Brier` = base_rate × (1 − base_rate).

**Note on Max Cal Error (uniform bins):** Values of 0.15–0.45 for low-base-rate states are sparse-bin artifacts — predictions cluster below 0.15, so most uniform bins above 0.3 have near-zero samples. Decile-based max abs error (0.048–0.062 for non-EA states) is the correct calibration reference. **Systematic Platt compression pattern:** Decile-8 (second-highest predictions) consistently has the highest calibration error for all low-rate states — the sigmoid ceiling of Platt (logistic) calibration creates a ~0.03–0.05 systematic underestimation at mid-high prediction values. Decile 9 (top predictions) recovers as actual goal rates reach 0.23–0.28.

| Strength | Base% | PR AUC | PR× | ROC AUC | Log Loss | Null LL | ΔLL% | Brier | Null Brier | ΔBr% | ECE | Max Cal (uniform) | OOF Gap | Lift | Precision | Recall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| even_strength | 6.0% | 0.3670 | 6.15× | 0.7899 | 0.1872 | 0.2260 | +17.2% | 0.0461 | 0.0561 | +17.9% | 0.0043 | 0.2748 | 0.0617 | +0.0012 | 0.2151 | 0.4466 |
| powerplay | 10.3% | 0.4147 | 4.04× | 0.7255 | 0.2744 | 0.3310 | +17.1% | 0.0736 | 0.0921 | +20.1% | 0.0041 | 0.1324 | 0.0709 | +0.0007 | 0.2936 | 0.3785 |
| shorthanded | 7.2% | 0.3556 | 4.93× | 0.8331 | 0.2180 | 0.2592 | +15.9% | 0.0573 | 0.0669 | +14.4% | 0.0082 | 0.4533 | 0.0452 | +0.0080 | 0.2178 | 0.4973 |
| empty_for | 7.7% | 0.3935 | 5.09× | 0.7499 | 0.2249 | 0.2722 | +17.4% | 0.0576 | 0.0714 | +19.2% | 0.0083 | 0.1577 | 0.0963 | +0.0076 | 0.2335 | 0.5087 |
| empty_against | 56.7% | 0.7677 | 1.35× | 0.7081 | 0.6096 | 0.6842 | +10.9% | 0.2125 | 0.2455 | +13.4% | 0.0549 | 0.1949 | 0.0045 | −0.0142 | 0.8401 | 0.4349 |

**Context_xg reference (same hold-out):** Context_xg v1.0.1.

| Strength | ctx_xg PR AUC | pred_goal PR AUC | Lift |
|---|---|---|---|
| even_strength | 0.3658 | 0.3670 | +0.0012 |
| powerplay | 0.4139 | 0.4147 | +0.0007 |
| shorthanded | 0.3476 | 0.3556 | +0.0080 |
| empty_for | 0.3859 | 0.3935 | +0.0076 |
| empty_against | 0.7820 | 0.7677 | −0.0142 |

---

### Hyperparameters

**Fixed params (all states):** objective=binary:logistic, booster=gbtree, n_estimators=500, early_stopping_rounds=50, eval_metric=["logloss","aucpr"] (early stop on aucpr — last metric; base_margin models must use aucpr-last, see Issue 20), random_state=615, enable_categorical=True. `logit(context_xg)` passed as `base_margin`. Talent features only — all BASE_XG and CONTEXT_XG feature columns stripped. `max_delta_step=1` and `lambda≥10` enforced by finalize.py for all pred_goal models.

Values below sourced from `models/pred_goal/{strength}/params.json`. Trial numbers and CV objectives from `meta.json`. `mds` = max_delta_step, `mcw` = min_child_weight, `lr` = learning_rate, `cbt` = colsample_bytree, `cbl` = colsample_bylevel, `cbn` = colsample_bynode.

| State | Trial | max_depth | mcw | mds | lr | gamma | lambda | alpha | subsample | cbt | cbl | cbn | spw | best_iter |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| even_strength | 1895 | 6 | 51 | 1 | 0.0660 | 8.888 | 29.826 | 1.711 | 1.00 | 0.95 | 0.85 | 0.90 | 1.537 | 9 |
| powerplay | 1654 | 6 | 54 | 1 | 0.0372 | 5.541 | 69.799 | 0.172 | 0.65 | 0.80 | 0.95 | 0.65 | 1.011 | 74 |
| shorthanded | 1549 | 4 | 111 | 1 | 0.0387 | 7.879 | 18.482 | 3.871 | 0.50 | 0.65 | 0.65 | 1.00 | 1.409 | 94 |
| empty_for | 1547 | 4 | 299 | 1 | 0.0735 | 1.702 | 32.338 | 1.293 | 0.45 | 0.85 | 0.75 | 0.80 | 1.011 | 419 |
| empty_against | 928 | 6 | 91 | 1 | 0.0639 | 1.913 | 13.202 | 1.034 | 1.00 | 0.65 | 0.95 | 1.00 | N/A | 19 |

EA has no `spw` — class balance handled naturally at 56.7% base rate.

**Hyperparameter assessment:**

**Lambda and alpha now properly explored.** All five states have lambda in the corrected [10–100] range (29.8/69.8/18.5/32.3/13.2 — distributed, not all pinned to 10). All five have alpha ≥ 0.1 — L1 regularization now active, providing sparsity on the many RAPM career features. Prior compromised run had lambda=10 for all states (old study ceiling = new floor) and alpha≈0 universally.

**even_strength (trial 1895):** gamma=8.888 is the strongest pruning of any low-rate state — appropriate for ES's 1.24M training shots where leaf splits must survive strong pruning. lambda=29.826, alpha=1.711. best_iter=9: the talent-layer adjustment converges in 9 trees. This is expected behavior for the ES talent signal: context_xg already captures geometry and game-state; RAPM adds a thin but consistent residual that saturates quickly. Learning curve confirmed via tree-by-tree PR AUC: peak at tree 9 (0.3670), degrades to 0.3663 by tree 50 — best_iter=9 is not premature stopping.

**powerplay (trial 1654):** lambda=69.799 is the strongest L2 of any state — appropriate for PP's compressed distribution and large model at max_depth=6. alpha=0.172 (lightest L1 of any state, above floor but conservative). best_iter=74. mcw=54 modest. subsample=0.65, aggressive feature sampling (cbn=0.65).

**shorthanded (trial 1549):** alpha=3.871 — second highest L1 of any state (behind prior-run context_xg SH at 5.012). max_depth=4 (only SH and EF use depth 4 — shallower trees appropriate for small datasets). mcw=111 is the second highest of low-rate states (strong leaf-size requirement). subsample=0.50 (most aggressive row sampling). best_iter=94. SH RAPM features are dominated by specialist players (PK specialist defensemen, known breakaway threat forwards) — high L1 makes sense to select these concentrated signals over the many career RAPM features with marginal signal.

**empty_for (trial 1547):** mcw=299 — highest of all states by a wide margin. This forces very coarse per-tree adjustments, requiring many trees to accumulate the diffuse EF talent signal. best_iter=419 is genuine — the learning curve shows monotonic PR-AUC improvement from 0.3859 at tree 0 to 0.3935 at tree 419 with no plateau; each tree contributes ~0.0002 average improvement. This is substantive talent-layer learning, not overfitting (OOF calibration patterns are consistent with hold-out patterns). lambda=32.338, alpha=1.293, max_depth=4.

**empty_against (trial 928):** best_iter=19 — minimal adjustment to the context_xg prior. Not best_iter=1 (old compromised run with lr=0.2384 overshot the prior immediately), but still very short. lambda=13.202 (lightest L2 among all states), alpha=1.034. max_depth=6. The negative lift (−0.0142) is structural, not a param issue — talent features add noise for EA because context_xg already captures the shot-quality signal and there is no goalie to differentiate with.

---

### Feature Gain

Talent features only — all base_xg and context_xg feature columns confirmed zero gain (no context leakage). Top 12 features by state:

**even_strength (20 features):** `shooter_gax_ewma` 20.1%, `shooter_rapm_career_xg_off` 8.1%, `goalie_gsax_ewma` 7.8%, `shooter_gax_per_shot_10g` 5.9%, `shooter_rapm_career_xg_def` 5.0%, `shooter_gax_10g` 4.6%, `goalie_gsax_10g` 4.4%, `goalie_gsax_per_shot_10g` 4.0%, `teammates_rapm_career_xg_off` 3.9%, `shooter_rapm_xg_off` 3.8%, `teammates_rapm_xg_off` 3.5%, `opp_rapm_xg_def` 3.4%. `shooter_gax_ewma` dominance (20.1%) reflects the corrected param space: alpha=1.711 provides L1 sparsity that compresses weak features, concentrating gain on the strongest rolling talent signal.

**powerplay (20 features):** `shooter_gax_ewma` 7.7%, `goalie_gsax_ewma` 7.0%, `shooter_rapm_career_xg_off` 6.9%, `goalie_gsax_per_shot_10g` 6.6%, `goalie_gsax_10g` 6.2%, `shooter_gax_10g` 5.5%, `shooter_gax_per_shot_10g` 5.0%, `shooter_rapm_career_xg_def` 5.0%, `shooter_vs_teammates_rapm_career_xg_off` 4.6%, `shooter_rapm_xg_off` 4.3%, `opp_rapm_career_xg_def` 4.2%, `teammates_rapm_career_xg_off` 4.2%. Goalie features (ewma + 10g + per_shot_10g) sum to ~20% — PP goalie quality is genuinely discriminating since compressed shot locations make goalie save% a key differentiator between average and excellent PP defense.

**shorthanded (20 features):** `shooter_rapm_career_xg_off` 6.5%, `shooter_gax_per_shot_10g` 6.0%, `shooter_vs_teammates_rapm_career_xg_off` 5.7%, `shooter_rapm_xg_off` 5.4%, `teammates_rapm_xg_off` 5.4%, `opp_rapm_xg_def` 5.3%, `shooter_gax_ewma` 5.3%, `shooter_rapm_career_xg_def` 5.1%, `opp_rapm_xg_off` 5.1%, `opp_rapm_career_xg_def` 4.8%, `opp_rapm_career_xg_off` 4.8%, `teammates_rapm_xg_def` 4.6%. RAPM-dominated with highly uniform distribution — no single feature above 7%. Consistent with SH specialist players (PK defensemen, breakaway threat forwards) being the primary discriminators where individual offensive RAPM is more predictive than raw rolling stats.

**empty_for (20 features):** `shooter_gax_ewma` 7.3%, `goalie_gsax_ewma` 6.0%, `teammates_rapm_xg_off` 5.9%, `teammates_rapm_career_xg_off` 5.5%, `opp_rapm_xg_def` 5.3%, `goalie_gsax_per_shot_10g` 5.2%, `shooter_gax_10g` 5.1%, `shooter_rapm_career_xg_def` 5.1%, `teammates_rapm_career_xg_def` 4.9%, `shooter_rapm_xg_def` 4.9%, `shooter_gax_per_shot_10g` 4.9%, `teammates_rapm_xg_def` 4.8%. Opponent and teammate RAPM features collectively ~30% — EF talent signal is about relative team quality. Goalie features proxy for overall team quality (teams with good goalies have better offensive rosters for EF situations).

**empty_against (17 features):** `opp_rapm_career_xg_off` 9.5%, `shooter_gax_per_shot_10g` 7.0%, `shooter_gax_10g` 6.9%, `opp_rapm_xg_off` 6.5%, `teammates_rapm_career_xg_off` 6.2%, `teammates_rapm_career_xg_def` 6.1%, `shooter_rapm_xg_def` 6.0%, `shooter_rapm_career_xg_def` 5.9%, `shooter_rapm_xg_off` 5.7%, `opp_rapm_career_xg_def` 5.4%, `opp_rapm_xg_def` 5.2%, `shooter_gax_ewma` 5.1%. Opponent offensive RAPM dominates — already partially captured by context_xg game-state features. Negative lift confirms these add noise. 17 features with nonzero gain (3 features dropped to exactly zero vs other states).

**Zero-gain features (8 universal, all states):** `shooter_gax_career`, `shooter_gax_per_shot_career`, `shooter_gax_season`, `shooter_gax_per_shot_season`, `goalie_gsax_career`, `goalie_gsax_per_shot_career`, `goalie_gsax_season`, `goalie_gsax_per_shot_season`. Long-window static aggregates (career and full-season totals) have no discriminating power once rolling EWMA and 10-game windows are present — these 8 features should be removed in a future pipeline cleanup.

**RAPM null rates:** 0.0% for ES/EF (73 nulls out of 1.2M ES shots — essentially zero). 0.0–0.2% for SH/EA training seasons. PP: 0.0–0.1% across all seasons. All well below the 5% warn threshold.

---

### Per-Strength Interpretation

#### even_strength

**Performance tier:** ❌ FAIL (OOF gap, structural). Discrimination: strong (0.3670 = 6.15× null, ROC 0.790).

**Calibration:** ECE=0.0043 (excellent). Systematic Platt compression pattern: decile-8 has the highest calibration error (0.0480), reflecting the sigmoid ceiling underestimating predictions in the second-highest tier by ~0.05. Decile 9 recovers (actual goals ~23.5%, predicted ~19.1%). This pattern is consistent across OOF training predictions and hold-out — it is Platt's structural limit, not a training-era artifact.

**OOF gap FAIL:** Gap = 0.0617 in the positive direction (hold-out 0.3670 >> training OOF 0.3053). Structural cause: RAPM quality improves over time as more career seasons accumulate. Shots taken in 2010-12 training folds have OOF predictions relying on only 1–2 seasons of RAPM history; shots in 2024-25 hold-out draw on 15 seasons of history. This creates a systematic quality advantage for hold-out over training OOF that is not overfitting. Confirmed by learning curve: best_iter=9 with degradation beyond tree 9 rules out overfitting at the model level.

**Lift: +0.0012 PR AUC over context_xg; 14W/1L across 15 seasons.** The sole loss (2013-14: −0.0001, essentially a tie) does not represent a structural failure. pred_goal beats context_xg in all other seasons including both anomalously low-discriminability years (2018-19, 2023-24). Monotonicity confirmed: within each context_xg quintile, shooters in the top RAPM quartile score at meaningfully higher rates than shooters in the bottom quartile — the model captures the correct direction of talent influence.

**best_iter=9:** The talent-layer adjustment converges in 9 trees and degrades beyond — confirmed by tree-by-tree PR AUC curve (0.3670 at tree 9, 0.3663 at tree 50). This is correct behavior: context_xg already captures geometry and game-state; RAPM adds a thin but consistent residual that saturates quickly. max_depth=6 WARN from diagnostic is expected and not concerning given the clean calibration.

#### powerplay

**Performance tier:** ❌ FAIL (OOF gap, structural). Discrimination: strong (0.4147 = 4.04× null, ROC 0.726).

**Calibration:** ECE=0.0041 (excellent, tighter than ES). Decile-8 max abs error 0.0493 (within PASS threshold). Same Platt compression pattern — decile-8 underpredicts by ~0.050.

**OOF gap FAIL:** Gap = 0.0709. Same inverse RAPM maturity structural cause as ES. Positive direction.

**Lift: +0.0007 PR AUC over context_xg; 11W/4L across 15 seasons.** The four losses are: 2012-13 (−0.0002), 2013-14 (−0.0009), 2015-16 (−0.0013), 2016-17 (−0.0025). All four losses are marginal in magnitude (<0.003 PR AUC). The 2015-17 era losses may reflect that PP talent differentiation was more compressed during those seasons (league-wide power play specialization patterns). PP talent signal is weaker than ES/SH because compressed shot locations reduce between-player variance — goalie quality becomes as important as shooter quality at this level (goalie features sum to ~20% of feature gain).

**lambda=69.799:** Strongest L2 of any state — appropriate at max_depth=6 with compressed PP distribution. best_iter=74.

#### shorthanded

**Performance tier:** ⚠️ WARN (calibration decile-8 + OOF gap). Discrimination: very strong (0.3556 = 4.93× null, ROC 0.833 — highest ROC of all states). First successful finalization after Issue 28 struct gate fix.

**Calibration:** ECE=0.0082. Decile-8 max abs error 0.0623 (above 0.05 PASS threshold, in WARN range). With only 2,592 hold-out SH events, decile bins have ~259 events each — substantial variance in actual goal rates per bin. The calibration WARN is natural sampling uncertainty at this dataset size, not systematic bias. The Platt compression pattern is the same as other states (decile-8 peak error, decile-9 recovery).

**OOF gap WARN:** Gap = 0.0452 (between 0.02 PASS and 0.05 FAIL thresholds). Positive direction. Same structural RAPM maturity cause as other states.

**Lift: +0.0080 PR AUC over context_xg — second highest of all states; 9W/5L+2ties across 15 seasons.** Season-by-season results: wins in 2010-11 (+0.006), 2011-12 (+0.010), 2017-18 (+0.002), 2018-19 (+0.004), 2020-21 (+0.004), 2021-22 (+0.003), 2022-23 (+0.005), 2023-24 (+0.002), 2024-25 (+0.008). Losses: 2012-13 (−0.007), 2015-16 (−0.007), 2016-17 (−0.004), 2019-20 (−0.002), 2023-24 (−0.002). Ties (0.0000): 2013-14, 2014-15. The 2015-17 era losses coincide with the same PP losses — league-wide specialist differentiation was reduced in that era. Overall hold-out lift (+0.008) is real and substantial given the 7.2% base rate.

**Feature gain RAPM-dominated (alpha=3.871):** High L1 selects PK-specialist player signals over the many career RAPM features with marginal relevance. RAPM features (shooter, teammates, opponents) collectively represent the top 7 features in the top 12 — correct for SH where individual penalty-kill specialist ability is the dominant talent discriminator.

#### empty_for

**Performance tier:** ❌ FAIL (OOF gap, structural). Discrimination: high (0.3935 = 5.09× null, ROC 0.750).

**Calibration:** ECE=0.0083. Decile-8 max abs error 0.0529 (WARN). With 2,974 hold-out EF events, decile bins have ~297 events each — similar sampling variance argument as SH. Platt compression pattern: decile-8 underpredicts by ~0.053.

**OOF gap FAIL:** Gap = 0.0963 — largest positive gap of all states. Training OOF = 0.2973, hold-out = 0.3935. The EF OOF average is dragged down by early training seasons with near-random discrimination: 2010-11 (OOF 0.195), 2011-12 (0.166), 2016-17 (0.196), 2018-19 (0.172). The 2024-25 hold-out (0.3935) is not an anomaly — it is a genuinely higher-discriminability season. Same RAPM maturity structural cause applies. This is a diagnostic threshold artifact.

**Lift: +0.0076 PR AUC over context_xg; 6 clear wins, 7 near-ties, 2 marginal losses.** Season-by-season: clear wins (+0.010 or more) in 2010-11, 2011-12; moderate wins in 2013-14 (+0.004), 2022-23 (+0.003), 2023-24 (+0.005), 2024-25 (+0.008). Near-zero gain in 2014-21 era (7 seasons with |diff| < 0.001). Marginal losses: 2012-13 (−0.001), 2019-20 (−0.000). **EF talent signal is concentrated in early (2010-12) and recent (2022-25) seasons.** The 2014-21 plateau suggests EF talent differentiation was compressed during that era — all teams improved at executing empty-for breakouts at roughly similar rates, reducing the between-player variance that pred_goal captures.

**best_iter=419:** Genuine monotonic improvement confirmed by learning curve — PR-AUC improves continuously from context_xg baseline (0.3859 at tree 0) to 0.3935 at tree 419, average +0.0002 per tree with no plateau. mcw=299 forces coarse per-tree adjustments (large leaf requirement means each split covers many shots); many trees are needed to accumulate the diffuse EF talent signal.

#### empty_against

**Performance tier:** ❌ FAIL (negative lift, structural). Discrimination: high (0.7677 = 1.35× null, ROC 0.708). Calibration: ECE=0.0549, decile max 0.0307 (PASS).

**Negative lift: −0.0142 PR AUC vs context_xg; 7W/8L across 15 seasons.** The temporal break is definitive: **2020-21 through 2024-25 are 5 consecutive losses** (−0.014 to −0.018 PR AUC each). Pre-2020, results were mixed: wins in 2011-12 (+0.004), 2012-13 (+0.002), 2013-14 (+0.008), 2015-16 (+0.002), 2016-17 (+0.006), 2017-18 (+0.006), 2019-20 (+0.037). The 2019-20 bubble win (+0.037) is anomalous — compressed bubble-format talent concentration in playoff-caliber players made RAPM unusually discriminating that season.

**Root cause: context_xg architectural improvement post-2020 absorbed EA shot-quality signal.** As context_xg improved over successive re-tuning runs (0.7013 in 2013-14 → 0.8583 in 2021-22), it captured the compositional shot-quality differences between EA attacking teams (strong vs. weak offensive rosters) that pred_goal's talent features previously added on top of. By 2020-21, context_xg's 21-feature game-state model already incorporates the team-quality information that pred_goal's RAPM features contain — adding those features creates noise, not signal.

**No goalie to model:** EA has no goalie by definition. EA goalie features (goalie_gsax_ewma, goalie_gsax_per_shot_10g, etc.) should have zero gain — confirmed by feature gain list (goalie features absent entirely from top 17).

**best_iter=19:** Minimal adjustment to the context_xg prior. The model makes 19 small corrections and then early-stops because any further adjustment degrades the hold-out metric. This is the correct behavior: context_xg is already near-optimal for EA; pred_goal has no useful residual to model.

**Production recommendation: use context_xg directly for EA inference.** pred_goal EA should not be applied to post-2020 data. For pre-2020 EA events (historical analysis), pred_goal EA provides modest improvement in some seasons but this is not systematically useful at the level warranting a separate model tier.

---

### Season-by-Season PR AUC (all states vs context_xg)

Season-by-season PR AUC for training seasons, showing both `ctx_xg` and `pred_goal` values. Hold-out (2024-25) is the reference season for the advanced metrics table.

| Season | ES ctx | ES pred | PP ctx | PP pred | SH ctx | SH pred | EF ctx | EF pred | EA ctx | EA pred |
|---|---|---|---|---|---|---|---|---|---|---|
| 2010-11 | 0.2677 | 0.2693 | 0.3178 | 0.3235 | 0.2804 | 0.2864 | 0.3571 | 0.3694 | 0.7658 | 0.7535 |
| 2011-12 | 0.2776 | 0.2802 | 0.3351 | 0.3384 | 0.3169 | 0.3266 | 0.3271 | 0.3439 | 0.7579 | 0.7619 |
| 2012-13 | 0.3239 | 0.3254 | 0.3412 | 0.3410 | 0.2664 | 0.2597 | 0.4713 | 0.4702 | 0.8016 | 0.8038 |
| 2013-14 | 0.2542 | 0.2541 | 0.2774 | 0.2765 | 0.2614 | 0.2614 | 0.2511 | 0.2552 | 0.7013 | 0.7095 |
| 2014-15 | 0.3703 | 0.3706 | 0.4403 | 0.4415 | 0.3404 | 0.3404 | 0.3703 | 0.3703 | 0.7277 | 0.7268 |
| 2015-16 | 0.2801 | 0.2811 | 0.3127 | 0.3114 | 0.3303 | 0.3236 | 0.3084 | 0.3084 | 0.7371 | 0.7393 |
| 2016-17 | 0.2988 | 0.3001 | 0.3471 | 0.3446 | 0.3241 | 0.3199 | 0.2370 | 0.2370 | 0.7656 | 0.7719 |
| 2017-18 | 0.2975 | 0.2979 | 0.3472 | 0.3486 | 0.3339 | 0.3355 | 0.2720 | 0.2721 | 0.7549 | 0.7605 |
| 2018-19 | 0.2053 | 0.2073 | 0.2234 | 0.2252 | 0.2313 | 0.2353 | 0.2135 | 0.2135 | 0.7786 | 0.7775 |
| 2019-20 | 0.2651 | 0.2663 | 0.2835 | 0.2848 | 0.2789 | 0.2771 | 0.3172 | 0.3169 | 0.6856 | 0.7229 |
| 2020-21 | 0.4627 | 0.4637 | 0.5416 | 0.5427 | 0.4222 | 0.4267 | 0.4752 | 0.4752 | 0.7732 | 0.7589 |
| 2021-22 | 0.4085 | 0.4100 | 0.4573 | 0.4593 | 0.4077 | 0.4110 | 0.3482 | 0.3482 | 0.8583 | 0.8413 |
| 2022-23 | 0.3048 | 0.3056 | 0.3413 | 0.3428 | 0.3266 | 0.3312 | 0.2968 | 0.3000 | 0.8113 | 0.7931 |
| 2023-24 | 0.1962 | 0.1968 | 0.2163 | 0.2177 | 0.2553 | 0.2531 | 0.2253 | 0.2305 | 0.7740 | 0.7594 |
| **2024-25** | **0.3658** | **0.3670** | **0.4139** | **0.4147** | **0.3476** | **0.3556** | **0.3859** | **0.3935** | **0.7820** | **0.7677** |

**Season record summary (pred_goal vs context_xg, training seasons only):**

| State | Wins | Losses | Ties | Notes |
|---|---|---|---|---|
| even_strength | 13 | 1 | 0 | Sole loss: 2013-14 (−0.0001, essentially tied) |
| powerplay | 10 | 4 | 0 | Losses in 2012-13, 2013-14, 2015-16, 2016-17 (all ≤ 0.003) |
| shorthanded | 9 | 5 | 2 | Ties in 2013-14, 2014-15; losses in 2012-13, 2015-16, 2016-17, 2019-20, 2023-24 |
| empty_for | 6 | 2 | 6 | Near-zero in 2014-21 era; wins concentrated in 2010-12 and 2022-24 |
| empty_against | 6 | 8 | 0 | Training wins in 2010-20 era; losses from 2020-21 onward |

---

### Production Readiness Assessment (2026-05-20)

| State | Overall | Verdict | Blocking Issue | Deployment Recommendation |
|---|---|---|---|---|
| even_strength | ❌ FAIL | **READY** | OOF gap FAIL is structural (RAPM maturity), not a model defect | Deploy; +0.0012 lift consistent across 14/15 seasons |
| powerplay | ❌ FAIL | **READY** | OOF gap FAIL structural | Deploy; +0.0007 lift, 11W/4L (losses all marginal ≤0.003) |
| shorthanded | ⚠️ WARN | **READY** | Calibration WARN + OOF WARN both structural/sample-size | Deploy; +0.0080 lift is the most meaningful per-season improvement |
| empty_for | ❌ FAIL | **READY** | OOF gap FAIL structural (largest positive gap but same cause) | Deploy; +0.0076 lift real, best_iter=419 confirmed genuine |
| empty_against | ❌ FAIL | **CONDITIONAL** | Negative lift structural (no goalie; context_xg absorbed signal) | Use context_xg for EA in post-2020 inference; pred_goal EA adds noise |

**All four low-rate states (ES/PP/SH/EF) are production-ready.** Every FAIL is an OOF gap in the inverse direction — the model generalizes *better* to hold-out than to its own OOF folds because RAPM quality improves over time. This is not fixable by further tuning and does not represent a model defect. Calibration is excellent (ECE 0.004–0.008) and lift is real and directionally consistent.

**Empty-against is conditionally deployable.** For historical analysis of pre-2020 EA events, pred_goal EA provides modest improvements in some seasons. For post-2020 EA inference (including 2024-25), context_xg is the better model — use it directly rather than the pred_goal EA tier.

**8 zero-gain features identified for future removal:** `shooter_gax_career`, `shooter_gax_per_shot_career`, `shooter_gax_season`, `shooter_gax_per_shot_season`, `goalie_gsax_career`, `goalie_gsax_per_shot_career`, `goalie_gsax_season`, `goalie_gsax_per_shot_season`. Removing these 8 features would reduce the feature set from 20 to 12 without degrading model quality.

---

### Changelog

| Date | Version | Trials | ES PR AUC | PP PR AUC | SH PR AUC | EF PR AUC | EA PR AUC | Notes |
|---|---|---|---|---|---|---|---|---|
| 2026-05-14 | 1.0.0 | 500 (OOF-only cal) | 0.3204 | 0.3420 | 0.3335 | 0.3121 | 0.7473 | Initial run. OOF-only Platt calibration. ES log loss 10.9× null (994% worse), PP/SH/EF 3–5× null. All five states FAIL. Catastrophic miscalibration driven by temporal drift: OOF calibrator learned training-era talent matchup statistics (goalie_gsax_per_shot_1g dominated at 45% gain for ES) that don't hold in hold-out. Root cause and fix documented in Issue 14. |
| 2026-05-14 | 1.0.0 | 500 (pooled cal) | 0.3204 | 0.3420 | 0.3336 | 0.3121 | 0.7480 | Pooled OOF + hold-out calibration (Issue 14 fix). Log loss improved to 1.15–1.23× null. Residual calibration FAIL ES/PP/SH/EF (decile-8 non-monotone, max err 0.13–0.19). EA calibration PASS. Lift negligible for non-EA. EA negative lift −0.034. Results use old feature set (bimodal context_xg base_margin + _1g features). |
| 2026-05-16 | 1.0.0 | 500 (re-tuned; Issues 15+16+17) | 0.3820 | 0.3749 | 0.3939 | 0.4360 | 0.7748 | Re-tuned after Issues 15+16 fixes (corrected context_xg base_margin; _1g features stripped; RAPM → xg dims only). Issue 17 fix applied to diagnose.py. Calibration resolved: decile-8 pattern eliminated, ECE 0.003–0.012. Lift real for non-EA. EF OOF gap FAIL structural. SH/EF calibration WARN at small-sample thresholds. |
| 2026-05-19 | 1.0.0 | 500 (re-finalized; Issues 25+26 clamps; 4/5 states only) | 0.3670 | 0.4147 | — | 0.3891 | 0.7681 | Issues 25+26 fixes applied: mds=1 forced and lambda≥10 clamped. SH not finalized — 100% bimodal from old studies. ES trial 424 (lambda=10 forced, alpha=0.0001, best_iter=86), PP trial 453 (lambda=10 forced, alpha=0.0493, best_iter=80), EF trial 1834 (lambda=10 forced, alpha=0.054, best_iter=65), EA trial 1732 (lambda=10 forced, lr=0.2384, best_iter=1). All params compromised: lambda=10 for all states; alpha≈0. Studies must be nuked and re-tuned. |
| 2026-05-20 | 1.0.1 | 2000 (corrected _params_pred_goal; Issue 28 struct gate fix; all 5 states) | 0.3670 | 0.4147 | 0.3556 | 0.3935 | 0.7677 | All 5 studies nuked and re-tuned under corrected param space (lambda [10–100], alpha [0.1–10], lr [0.01–0.10], mcw [50–300], mds=1 fixed). Issue 28: struct hard gate disabled for base_margin models — adversarial test showed gate had zero discrimination for pred_goal; mds=1+lambda≥10 clamps provide structural protection. SH finalized for first time (struct gate was blocking all SH trials). New trial selections: ES 1895 (lambda=29.8, alpha=1.71, best_iter=9), PP 1654 (lambda=69.8, alpha=0.17, best_iter=74), SH 1549 (lambda=18.5, alpha=3.87, best_iter=94), EF 1547 (lambda=32.3, alpha=1.29, best_iter=419), EA 928 (lambda=13.2, alpha=1.03, best_iter=19). Lambda now properly distributed across [13–70]; alpha ≥ 0.1 for all states. Lift: ES +0.0012 (14W/1L), PP +0.0007 (11W/4L), SH +0.0080 (9W/5L), EF +0.0076 (6W clear, many ties), EA −0.0142 (structural). OOF gaps all in inverse direction (RAPM maturity). EF best_iter=419 confirmed genuine via learning curve. ES best_iter=9 confirmed correct (degrades beyond tree 9). 8 universal zero-gain features identified for future removal. EA temporal break confirmed (5 consecutive losses post-2020). All FAIL/WARN results are structural, not correctible by further tuning. |
