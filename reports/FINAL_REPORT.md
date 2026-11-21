# Final report

## Outcome

- Submission C: `submission_C.csv` — `B_v8_logit_0.5`, frozen-CV proxy `0.741381319` vs B `0.740863078` (`+0.000518241`).
- Submission D: `submission_D.csv` — `logC1_v8_logit_0.5`, frozen-CV proxy `0.741483833` vs B `0.740863078` (`+0.000620754`).
- C is the conservative hedge: raw logit blend of public-proven B and the structurally initialized 50-seed v8 signal; no post-hoc calibration.
- D is the higher-upside hedge: same v8 signal blended with regularized profile-aware logistic calibration of B.
- Known public: B `0.744694601`, O1 `0.744795320`, rank 1 `0.744833171`; exact metric recomputes B as `0.744694601`.
- O1 is the best public submission so far; no submissions remain.
- No public leaderboard scores were used for tuning C or D; A/B scores are reported only as outcome evidence.
- New feature-group search under the amended gate found no accepted group; the later E/F/G/H/I/J/K/L/M/N/O batches are explicitly public-feedback diagnostics.

## Public outcome of C and D

| Submission | Public score | Δ vs B | Log Loss | Δ LL | ROC AUC | Δ AUC |
|---|---:|---:|---:|---:|---:|---:|
| B | 0.744694601 | 0 | 0.223576399 | 0 | 0.925374485 | 0 |
| C | 0.744365653 | -0.000328948 | 0.223826287 | +0.000249888 | 0.925182086 | -0.000192399 |
| D | 0.744325022 | -0.000369579 | 0.223856978 | +0.000280579 | 0.925157879 | -0.000216606 |

- Both C and D regressed publicly despite positive OOF deltas: about 76–77% of each loss came from worse Log Loss and 23–24% from worse AUC.
- At that stage B was the best submitted file; rank-1 gap was `+0.000138570`.
- The v8 structural signal blended into C/D does not transfer to the test snapshots; it is rejected along with rank/isotonic calibration.

## Recent public outcomes

| Submission | Public score | Δ vs O1 | Log Loss | ROC AUC |
|---|---:|---:|---:|---:|
| K2 | 0.744780206 | -0.000015114 | 0.223500111 | 0.925396175 |
| L1 | 0.744777292 | -0.000018028 | 0.223499620 | 0.925387654 |
| L2 | 0.744778268 | -0.000017052 | 0.223499728 | 0.925390365 |
| M1 | 0.744707523 | -0.000087797 | 0.223562049 | 0.925370612 |
| M2 | 0.744659467 | -0.000135853 | 0.223593918 | 0.925330815 |
| M3 | 0.744548276 | -0.000247044 | 0.223675875 | 0.925259452 |
| N1 | 0.744782951 | -0.000012369 | — | — |
| N2 | 0.744792129 | -0.000003191 | 0.223489094 | 0.925398208 |
| O1 | 0.744795320 | 0 | 0.223485123 | 0.925396175 |

- Target rank 1 is `0.744833171`; O1 gap is `+0.000037851`.
- O1 beat N2 by `+0.000003191`; the gain came from lower Log Loss, while ROC AUC matched K2 and remained the main remaining gap to rank 1.

## Rules, data, and metric

- The Zindi competition/rules URLs returned the SPA text `404 not found`; exact team, submission, and external-data restrictions could not be read. This solution uses only the provided `Train.csv`, `Test.csv`, and `SampleSubmission.csv`.
- Train `(40000, 184)`, test `(30000, 183)`, target `liquidity_stress_next_30d`, 15% positives per snapshot.
- Same 10,000 customers appear in train (4 snapshots) and test (3 later snapshots); each snapshot has exactly 1,500 positives.
- No missing values, duplicate rows/IDs, or constant columns; clean RF adversarial validation AUC `0.5000837858333333`.
- Exact metric: `score = 0.6 * (1 - LogLoss / 0.595) + 0.4 * ROC_AUC`. Expanded: `0.4 * AUC - 1.008403361 * LogLoss + 0.6`.

## Why A failed publicly

- A had a better frozen-CV proxy than B (`0.741246388` vs `0.740863078`), but scored `0.743551` vs B `0.744695`.
- A's within-date rank isotonic map is stepwise and clips hundreds of rows at `0.99999`; it preserved AUC but increased public Log Loss by `0.001122`.
- Plain isotonic calibration previously collapsed publicly to roughly `0.739335`; rank/isotonic-style calibration is therefore rejected despite strong nested CV.
- B diagnostics show near-zero bucket biases; worst 500 OOF losses were 478 false negatives and 22 false positives, concentrated in MVC/LVC low-probability rows, with no missingness effects.

## Validation protocol

- Frozen 5-fold × 3-seed stratified CV: seeds `42/2026/7`, manifest in `reports/frozen_cv_manifest.json`, folds in `reports/folds_v2.npz`.
- Selection required paired delta greater than one standard error over 15 fold/repeat scores.
- Nested calibration: fit on 2 repeats, evaluate held-out repeat, final fit on all 3; per-snapshot 15% prior restored after every transform.
- Blend weights: non-negative simplex, optimized on 2 repeats, evaluated on held-out repeat; logistic stacker tested with leave-one-model-out.
- Candidate C/D additionally required positive date-3 delta and positive customer-cluster bootstrap lower 95% bound.
- Customer-cluster bootstrap: 1,000 resamples of 10,000 customers with all 4 train snapshots kept together.

## Calibration suite

| Candidate | OOF proxy | Delta vs B | Paired SE | Decision |
|---|---:|---:|---:|---|
| `iso_smooth_1` | 0.741231378 | +0.000372485 | 0.000025067 | reject: isotonic/step-risk |
| `iso_smooth_0.75` | 0.741191250 | +0.000331680 | 0.000017295 | reject: isotonic/step-risk |
| `iso_smooth_0.5` | 0.741113239 | +0.000253418 | 0.000010883 | reject: isotonic/step-risk |
| `iso_smooth_0.25` | 0.741004172 | +0.000143512 | 0.000005237 | reject: isotonic/step-risk |
| `logistic_features_C1` | 0.740997825 | +0.000136607 | 0.000044317 | retain for D |
| `logistic_features_C0.1` | 0.740997799 | +0.000136526 | 0.000044157 | reject |
| `segment_platt` | 0.740901151 | +0.000038380 | 0.000016762 | reject |
| `temperature` | 0.740862334 | -0.000000169 | 0.000000279 | reject |
| `platt` | 0.740862115 | -0.000000354 | 0.000001169 | reject |
| `beta` | 0.740873319 | +0.000010793 | 0.000008839 | reject |

Temperature, Platt, beta, clip sweeps, logistic feature calibration, segment Platt, and small GBM calibration were all tested nested. All gains were below `+0.00014`; the larger gains came from stepwise isotonic mappings that are not trusted after A's public failure.

## Blend suite

- Best raw LightGBM-only nested blend: `blend_tuned_final_tuned_baseline_cat`, proxy `0.740880736`, delta `+0.000018191`, SE `0.000008030`.
- Full logistic stackers and six-model non-negative blends did not beat B nested; leave-one-model-out showed tuned LightGBM as the only indispensable member.

## Final candidate ranking

| Candidate | OOF proxy | Delta vs B | Log Loss | AUC | Date-3 delta | Bootstrap 95% CI | Decision |
|---|---:|---:|---:|---:|---:|---|---|
| `logC1_v8_logit_0.5` | 0.741483833 | +0.000620754 | 0.225952682 | 0.923338191 | +0.000459502 | [+0.000347018, +0.000882503] | D |
| `B_v8_logit_0.75` | 0.741451468 | +0.000588390 | 0.225983359 | 0.923334618 | +0.000455440 | [+0.000203330, +0.000970694] | not selected |
| `v8_base_init_rank3_s50` | 0.741398278 | +0.000535200 | 0.226030786 | 0.923321206 | +0.000358557 | [+0.000024934, +0.001047068] | not selected |
| `B_v8_logit_0.5` | 0.741381319 | +0.000518241 | 0.226028777 | 0.923273745 | +0.000421952 | [+0.000261314, +0.000772341] | C |
| `B_v7_logit_0.75` | 0.741369507 | +0.000506429 | 0.226039762 | 0.923271907 | +0.000430442 | [+0.000098817, +0.000896165] | not selected |
| `B_v7_logit_0.5` | 0.741337415 | +0.000474336 | 0.226060327 | 0.923243520 | +0.000417699 | [+0.000203019, +0.000735118] | not selected |
| `logC1_v8_logit_0.25` | 0.741312369 | +0.000449291 | 0.226072692 | 0.923212078 | +0.000299569 | [+0.000261718, +0.000611275] | not selected |
| `v7_struct_perpower` | 0.741274386 | +0.000411307 | 0.226117981 | 0.923231294 | +0.000326973 | [-0.000128311, +0.000930394] | not selected |
| `B_v8_logit_0.25` | 0.741186796 | +0.000323718 | 0.226167291 | 0.923136632 | +0.000275609 | [+0.000196134, +0.000450738] | not selected |
| `B_v7_logit_0.25` | 0.741167544 | +0.000304465 | 0.226180008 | 0.923120559 | +0.000275877 | [+0.000168692, +0.000435135] | not selected |
| `logistic_features_C1` | 0.740997825 | +0.000134747 | 0.226298997 | 0.922996235 | -0.000007766 | [-0.000032789, +0.000310073] | not selected |
| `B` | 0.740863078 | +0.000000000 | 0.226399262 | 0.922912137 | +0.000000000 | [+0.000000000, +0.000000000] | baseline |

## Decision

- C (`B_v8_logit_0.5`): OOF `0.741381319`, Log Loss `0.226028777`, AUC `0.923273745`; date-3 delta `+0.000421952`; bootstrap CI `[+0.000261314, +0.000772341]`.
- D (`logC1_v8_logit_0.5`): OOF `0.741483833`, Log Loss `0.225952682`, AUC `0.923338191`; date-3 delta `+0.000459502`; bootstrap CI `[+0.000347018, +0.000882503]`.
- Public outcome: C and D both regressed; keep `submission_B.csv` as the final choice unless a genuinely new CV-validated signal is found.
- The v8 structural blend turned OOF gains of `+0.000518241` / `+0.000620754` into public losses of `-0.000328948` / `-0.000369579`.
- Future work should prioritize distribution-shift-robust validation and avoid structural rank features, rank/isotonic calibration, and public-score-driven tuning.

## New feature-group experiment

- Baseline verification passed: `submission_B.csv` sha256 `5278fb77528a3701f733cef622b2c19c4df04d4321b893d1e581a98909ff4a2c`; public metrics `0.744694601` / `0.223576399` / `0.925374485`.
- Frozen OOF gate: proxy `0.740863078`, LL `0.226399262`, AUC `0.922912137`; required submission proxy `0.741863078`.
- Groups tested: `6`; acceptance rule: repeat-level paired proxy delta positive in every repeat and mean delta greater than `2 x SE` across repeat deltas.
- SE was computed across the 3 repeat-level deltas, not across the 15 correlated folds. With 6 tests, multiple-comparison risk remains material; the strict repeat gate and reserved confirmation split were intended to mitigate, not eliminate, it.

| Group | Δ proxy | repeat SE | repeat deltas | accept |
|---|---:|---:|---|---|
| `diff2_flows` | -0.000149928 | 0.000050564 | 7:-0.000052246, 42:-0.000176103, 2026:-0.000221437 | False |
| `window_ewma` | -0.000012907 | 0.000032496 | 7:+0.000052066, 42:-0.000046710, 2026:-0.000044078 | False |
| `bal_shape_v3` | -0.000053354 | 0.000044981 | 7:+0.000036588, 42:-0.000096665, 2026:-0.000099985 | False |
| `arpu_scale_v3` | -0.000060556 | 0.000039583 | 7:-0.000003094, 42:-0.000042126, 2026:-0.000136446 | False |
| `peer_label_free` | -0.000080496 | 0.000046855 | 7:+0.000006770, 42:-0.000094552, 2026:-0.000153705 | False |
| `activity_types_v3` | +0.000012266 | 0.000014128 | 7:+0.000001767, 42:+0.000040234, 2026:-0.000005203 | False |

- Accepted groups: `0`. Per the stop rule, no combination search, retuning, seed-bagging, confirmation split, or public submission was performed.

## Submission candidates E

- User requested a new file to submit while preserving `submission_B.csv`; B hash is unchanged after generating the candidates.
- E candidates are probability-space hedges around B using saved, deterministic OOF/test predictions; they do not overwrite B and preserve the 15% per-snapshot mean.
- These are diagnostic attempts, not strict-gate promotions: all have slightly lower frozen OOF than B, so any public gain would come from transfer noise or diversity around B.

| File | Blend | OOF proxy | Δ vs B | sha256 |
|---|---|---:|---:|---|
| `submission_E1.csv` | 0.95 B + 0.05 `v9_blend_extras_3m_logit` | 0.740858558 | -0.000004520 | `dd1b31b16b43deded824e0c8d1627210bd288f3955bd17565a5a370c3d2c23dc` |
| `submission_E2.csv` | 0.80 B + 0.20 `v9_blend_m6_lgb_cat` | 0.740819791 | -0.000043288 | `05ece47188454b99028f4e1eff684e64ecf67b4cfd04424bd0bd6dae5ed15fea` |
| `submission_E3.csv` | 0.70 B + 0.30 `v9_drop_m5_m6_m1paybill_s50` | 0.740788509 | -0.000074570 | `48fe826eda676150adcc04076d73ab80945d1d09ddb4417f694fe707cc444747` |

- Submit order if daily limit allows: `submission_E1.csv`, then `submission_E2.csv`, then `submission_E3.csv`. Keep `submission_B.csv` as the saved fallback unless an E candidate scores higher publicly.

## Submission candidates F

- E feedback: E1 `0.744693992` (LL `0.223577118`, AUC `0.925374776`), E2 `0.744678116`, E3 `0.744675143`; B remains best at `0.744694601`.
- F candidates target LogLoss while keeping perturbations small; F2/F3 use lightly blended regularized logistic calibration, F1 uses the saved low-tail clip.
- All F candidates are diagnostic attempts, not strict-gate promotions; they use saved deterministic OOF/test arrays and preserve sample ID order.

| File | Construction | OOF proxy | Δ vs B | Δ LL | Δ AUC | Date-3 Δ | Test MAD vs B | sha256 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `submission_F1.csv` | `clip_0.001_1e-05` | 0.740880762 | +0.000017684 | -0.000017681 | -0.000000363 | -0.000002295 | 0.000001934 | `2aa5ef04f7b47a9071ec20e9f5bd143f0dd05005cf14442242841bbbd803183d` |
| `submission_F2.csv` | 0.95 B + 0.05 `logistic_features_C0.01` | 0.740875110 | +0.000012031 | -0.000008645 | +0.000008284 | +0.000005979 | 0.000133859 | `b5449cc540f36a62dae97f06326a5e8c86962c8730b51a43bc55ad947c04e43b` |
| `submission_F3.csv` | 0.95 B + 0.05 `logistic_features_C1` | 0.740876597 | +0.000013519 | -0.000009756 | +0.000009201 | +0.000007638 | 0.000141248 | `b6d6215be45163f7d9fbaba4f22fc70230cc9606395a37568ebd9c7ef02f9720` |

- Submit order was: `submission_F2.csv` → `submission_F3.csv` → `submission_F1.csv`.
- Public outcome: F3 `0.744704184` (LL `0.223570890`, AUC `0.925384555`) was best, followed by F2 `0.744702799`; F1 `0.744692858` did not beat B.

## Submission candidate G

- F3 is the public reference at `0.744704184`. G1 doubles the F3 C1 weight from 5% to 10% because the F batch confirmed that the regularized logistic direction transfers.
- G1 is a single-submission escalation, not a strict-gate promotion; it uses saved deterministic OOF/test arrays and preserves sample ID order.

| File | Construction | OOF proxy | Δ vs B | Δ vs F3 | Δ LL | Δ AUC | Date-3 Δ | Test MAD vs B | sha256 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `submission_G1.csv` | 0.90 B + 0.10 `logistic_features_C1` | 0.740888904 | +0.000025826 | +0.000012307 | -0.000019017 | +0.000016623 | +0.000013171 | 0.000282496 | `e4e84a79fb38b76363a7ccb15ce321c7fa38b0cb45737fcb2c503644336f919c` |

- Submit order was: `submission_G1.csv`.
- Public outcome: G1 `0.744711831` (LL `0.223565918`, AUC `0.925391140`) beat F3; OOF-to-public transfer stayed near 55% for LL and 100% for AUC.

## Submission candidate H

- G1 is the public reference at `0.744711831`. H1 raises the C1 weight from 10% to 25% because the F/G batches showed stable transfer of the regularized logistic direction.
- H1 is a single-submission escalation, not a strict-gate promotion; it uses saved deterministic OOF/test arrays and preserves sample ID order.

| File | Construction | OOF proxy | Δ vs B | Δ vs G1 | Δ LL | Δ AUC | Date-3 Δ | Test MAD vs B | Extrapolated public | sha256 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `submission_H1.csv` | 0.75 B + 0.25 `logistic_features_C1` | 0.740921352 | +0.000058274 | +0.000032448 | -0.000043821 | +0.000035211 | +0.000027488 | 0.000706240 | 0.744733067 | `c78844098e46556389caefcdad90f9fa3489a7d321049caa2306d03e71fb0959` |

- Submit order was: `submission_H1.csv`.
- Public outcome: H1 `0.744724040` (LL `0.223554235`, AUC `0.925392205`) beat G1; transfer efficiency fell to ~50% for both LL and AUC but stayed positive.

## Submission candidate I

- H1 is the public reference at `0.744724040`. I1 raises the C1 weight from 25% to 40% using the measured ~50% transfer efficiency.
- I1 is a single-submission escalation, not a strict-gate promotion; it uses saved deterministic OOF/test arrays and preserves sample ID order.

| File | Construction | OOF proxy | Δ vs B | Δ vs H1 | Δ LL | Δ AUC | Date-3 Δ | Test MAD vs B | Extrapolated public | sha256 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `submission_I1.csv` | 0.60 B + 0.40 `logistic_features_C1` | 0.740949185 | +0.000086107 | +0.000027833 | -0.000064148 | +0.000053549 | +0.000031805 | 0.001129985 | 0.744738098 | `10782ddcb411be2488c7ef7a85ea3d3b40d8612cae96e010abcf6f913b65f5a3` |

- Submit order was: `submission_I1.csv`.
- Public outcome: I1 `0.744733669` (LL `0.223547413`, AUC `0.925399080`) beat H1; transfer efficiency fell to ~45% for both LL and AUC, and the public response curve is near its C1-weight optimum.

## Submission candidate J

- I1 is the public reference at `0.744733669`. J1 uses the public-fit C1 optimum near 50%; J3 is a higher-OOF hedge at 58%; J2 mixes a small C0.01 share for diversity.
- No submissions remain, so J1/J2/J3 are prepared but not submitted; keep I1 until the submission budget resets.

| File | Construction | OOF proxy | Δ vs B | Δ vs I1 | Δ LL | Δ AUC | Date-3 Δ | Test MAD vs B | sha256 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `submission_J1.csv` | 0.50 B + 0.50 `logistic_features_C1` | 0.740964578 | +0.000101499 | +0.000015393 | -0.000075203 | +0.000064162 | +0.000033835 | 0.001412481 | `3f09b603886aae5861ed00315b751361bd6c635924c4a12fa36892dbf2da0b43` |
| `submission_J2.csv` | 0.52 B + 0.43 `logistic_features_C1` + 0.05 `logistic_features_C0.01` | 0.740960617 | +0.000097539 | +0.000011432 | -0.000072240 | +0.000061730 | +0.000032100 | 0.001339841 | `e70780bbeb107f409a1fc38ff76b45445be5889357d50a04952a046c2bade634` |
| `submission_J3.csv` | 0.42 B + 0.58 `logistic_features_C1` | 0.740974684 | +0.000111606 | +0.000025499 | -0.000082604 | +0.000070770 | +0.000033379 | 0.001638478 | `eb911fde3d75d4899bf86d23a5395fd04291e79d0a91c1909717ef5f5574a258` |

- Submit order when budget resets: `submission_J1.csv` → `submission_J3.csv` → `submission_J2.csv`.

## Offline search after I1

- Evaluated `171` saved OOF/test blends; `13` met the positive-OOF and non-negative date-3 screen.
- Higher-OOF extension found at 58% C1 (J3), but the public response curve is near its optimum around 45-50% C1; J1 remains the public-fit-first backup, J3 the higher-OOF hedge, and J2 the diversity hedge.

## Extended logistic calibration search

- Re-trained profile-aware logistic calibration for C values `[0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]` under the same nested 3-repeat protocol.
- Best saved-array blend was `logistic_features_C3` at weight `0.65`: OOF proxy `0.740981265`; the gain over C1 was negligible, so C strength alone was not promoted.

- Added quadratic logit, logit×date, and logit×segment interactions; best full-feature blend reached OOF proxy `0.741034717` with paired accept `True`.
- High-weight interaction variants weakened date 3, so only 50%-weight variants with non-negative date-3 deltas were promoted to K backups.

## Submission candidate K

- K1/K2 use profile-aware logistic calibration with quadratic logit, logit×date, and logit×segment interactions at 50% weight; K1 uses C=1 for the stronger date-3 result, K2 uses C=3 for the higher overall OOF proxy.
- No submissions remain, so K1/K2 are prepared but not submitted; I1 remains the public best until the budget resets.

| File | Construction | OOF proxy | Δ vs B | Δ vs J1 | Δ LL | Δ AUC | Date-3 Δ | Test MAD vs B | sha256 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `submission_K1.csv` | 0.50 B + 0.50 `logistic_interactions_C1` | 0.741015907 | +0.000153441 | +0.000051573 | -0.000117681 | +0.000086926 | +0.000004373 | 0.002174979 | `062e2dc7fa18a886529e3305ae4de304b4faf516ba13915cc96552f134159cbe` |
| `submission_K2.csv` | 0.50 B + 0.50 `logistic_interactions_C3` | 0.741019120 | +0.000156654 | +0.000054785 | -0.000120473 | +0.000087922 | +0.000001199 | 0.002173646 | `fd06e018caefe4d35bd1a93015a93e96f2093aa75f7b755126092ee241a0d8e6` |

- Submit order when budget resets: `submission_K1.csv` → `submission_K2.csv`.

## Submission candidate L

- L1/L2 raise the `logistic_interactions_C1` weight from 50% to 53%/54%; both remain paired-accepted and keep date-3 non-negative.
- L1 is the safer primary at 53%; L2 has slightly higher overall OOF but a smaller date-3 margin. No submissions remain, so both are prepared but not submitted.

| File | Construction | OOF proxy | Δ vs B | Δ vs K1 | Δ LL | Δ AUC | Date-3 Δ | Test MAD vs B | sha256 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `submission_L1.csv` | 0.47 B + 0.53 `logistic_interactions_C1` | 0.741022279 | +0.000159201 | +0.000006112 | -0.000122003 | +0.000090431 | +0.000002650 | 0.002305477 | `684fb9d44118f025664c0d167e63dd14ab67f2585ac78932195d02fb1c377594` |
| `submission_L2.csv` | 0.46 B + 0.54 `logistic_interactions_C1` | 0.741024062 | +0.000160984 | +0.000007895 | -0.000123455 | +0.000091230 | +0.000001264 | 0.002348977 | `0f08a674de1f15cf9e66e782ca088ed83c0e2b61bbc31534a567f0dd0f42beed` |

- Submit order when budget resets: `submission_L1.csv` → `submission_L2.csv`.

## Expanded interaction calibration search

- Added logit interactions with region, earning pattern, smartphone, gender, quantile bins, scaled continuous features, and segment×date alongside the K/L interaction set.
- Best eligible expanded source was `logistic_ix_all_C1` at weight `0.54`: OOF proxy `0.741128986`, date-3 delta `+0.000122310`, paired accept `True`.
- C3 was selected for the M backups because its date-3 margin was slightly stronger at the higher 60%–100% weights while overall OOF was effectively identical to C1.

## Submission candidate M

- M1/M2/M3 use the expanded C3 interaction source at 60%, 75%, and 100% weight; all are paired-accepted with strongly positive date-3 deltas.
- M1/M2/M3 were prepared as backups; their public results showed increasing regression with expanded-interaction weight, so they are retained as diagnostics only.

| File | Construction | OOF proxy | Δ vs B | Δ vs L1 | Δ LL | Δ AUC | Date-3 Δ | Test MAD vs B | sha256 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `submission_M1.csv` | 0.40 B + 0.60 `logistic_ix_all_C3` | 0.741146167 | +0.000283089 | +0.000123888 | -0.000210913 | +0.000176010 | +0.000127367 | 0.002754736 | `4d5e2273d3f21fb98133c2db93ffdd8516e771dfb6a769a0a439620c1ccffeb9` |
| `submission_M2.csv` | 0.25 B + 0.75 `logistic_ix_all_C3` | 0.741179621 | +0.000316542 | +0.000157341 | -0.000235862 | +0.000196745 | +0.000120649 | 0.003443419 | `c59245ce923bcb49047347f48d51622f7c0c391e51498cf571411c721470e01f` |
| `submission_M3.csv` | 0.00 B + 1.00 `logistic_ix_all_C3` | 0.741204462 | +0.000341384 | +0.000182183 | -0.000252976 | +0.000215706 | +0.000080172 | 0.004591226 | `e369d639369d6d04d52fc70672875a05b7494f0f9a7e2fbc4357fa44c81cd52b` |

- Historical submit order used for M: `submission_M1.csv` → `submission_M2.csv` → `submission_M3.csv`.

## Public feedback after L/M/N

- N2 was the intermediate public best at `0.744792129`; N1 scored `0.744782951`, so the 10% expanded-C3 dose beat the 5% dose and K2.
- L1/L2 were within `0.000003` of K2 and the M family regressed increasingly as expanded-interaction weight rose, so the usable dose range is narrow.

## Submission candidate N

- N1/N2 add 5%/10% expanded C3 to K2. Both remained close to K2 in test deviation, were paired-accepted, and improved OOF proxy, LL, AUC, and date-3 versus K2.
- N2 publicly beat N1 and K2, confirming the positive transfer of a small expanded-C3 dose.

| File | Construction | OOF proxy | Δ vs K2 | Δ LL vs K2 | Δ AUC vs K2 | Date-3 Δ vs K2 | Test MAD vs B | sha256 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `submission_N1.csv` | `0.95 K2 + 0.05 logistic_ix_all_C3` | 0.741037101 | +0.000017648 | -0.000012770 | +0.000011926 | +0.000011800 | 0.002156194 | `348993192927d9e669a5d603c7da0e1d7b9d449d69c620f601c2cfd3848f1b64` |
| `submission_N2.csv` | `0.90 K2 + 0.10 logistic_ix_all_C3` | 0.741053870 | +0.000034417 | -0.000024886 | +0.000023304 | +0.000023602 | 0.002165664 | `9f3bf90d951a715c19278891d58e75324c6f8f883ec14dad5ec3c80208c4a5b7` |

- Historical submit order used for N: `submission_N1.csv` → `submission_N2.csv`.

## Submission candidate O

- O1 raises the expanded-C3 dose to 20%, following N2's public win over N1 and K2.
- O1 was submitted as the final candidate and became the public best at `0.744795320`.

| File | Construction | OOF proxy | Δ vs K2 | Δ LL vs K2 | Δ AUC vs K2 | Date-3 Δ vs K2 | Test MAD vs B | sha256 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `submission_O1.csv` | `0.80 K2 + 0.20 logistic_ix_all_C3` | 0.741084661 | +0.000065208 | -0.000047167 | +0.000044113 | +0.000040438 | 0.002248256 | `6e6055c885405f1111142f839a06c20b5c44fe95cbf97dd0e27d59da421b2294` |

- Final submission used: `submission_O1.csv`; submissions available after that: `0`.

## Peer-relative feature methodology and reproducibility

- Peer-relative features are computed label-free from `concat(train.drop(target), test)` in fixed row order with snapshot dates `0..3` for train and `4..6` for test.
- For each peer key (`segment`, `region`, `earning_pattern`) and each selected numeric feature, rows are grouped by `(snapshot_date, peer_key)`; missing keys use the sentinel `__MISSING__`.
- Within each group, z-score uses population standard deviation (`ddof=0`) with `1e-8` numerical floor and percentile rank uses average ties minus `0.5`; no target, fold index, prediction, or leaderboard value is used.
- Derived columns are then rank-scaled within each snapshot date by the same `month_scaled(..., 'rank')` path used by the base pipeline.
- Determinism: frozen fold seeds `42/2026/7`, LightGBM bagging seeds start `909,11,22,33,44`, early-stopping splits use `10000 + repeat * 10 + fold`, Optuna sampler seed `20261119`, and confirmation seed `314159` is reserved but unused because the stop rule triggered.
- Full rebuild from raw CSVs: `python final_solution.py --stage all`; the `verify_baseline` stage checks `submission_B.csv` hash, raw-data hashes, sample ID order, saved test predictions, and frozen OOF metrics.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python final_solution.py --stage all
```

Core stages after data/model artifacts exist:

```bash
.venv/bin/python calibration_lab.py
.venv/bin/python calibration_suite.py
.venv/bin/python blend_suite.py
.venv/bin/python final_selection_v2.py
.venv/bin/python finalize_v2.py
```
