# v7 Summary: Structural Extra Factors + Rank Transforms

## Metric

`score = 0.6 * (1 - LogLoss / 0.595) + 0.4 * ROC_AUC`

All validation below is leave-one-month-out (LOMO) OOF on the 40,000 training rows.
The structural baseline is the exact 2nd-place generator reconstruction:

- Base LOMO: `0.740474849`
- Base LogLoss: `0.226701877`
- Base AUC: `0.922704461`
- Reference public score for the structural submission: `0.744516108`

## Main Result

Adding forward-selected extra liquidity factors directly into the structural probit,
with a global rank-cubed transform and fixed intercepts, gives a large LOMO gain:

| Candidate | LOMO | Gain vs base | LogLoss | AUC | Per-month gains |
|---|---:|---:|---:|---:|---|
| `rank3_10` | `0.741210425` | `+0.000735576` | `0.226170392` | `0.923203520` | `[+0.000884, +0.000648, +0.000552, +0.000884]` |
| `perpower` | `0.741274386` | `+0.000799537` | `0.226117981` | `0.923231294` | `[+0.000902, +0.000610, +0.000714, +0.000987]` |
| `rank3_10_nom5` | `0.740985753` | `+0.000510904` | `0.226340251` | `0.923070059` | `[+0.000616, +0.000324, +0.000512, +0.000616]` |
| `blend_struct_v6res` | `0.741283035` | `+0.000808186` | `0.226111447` | `0.923236446` | n/a |
| `extra4 global_rank` | `0.740876177` | `+0.000401327` | `0.226409662` | `0.922971103` | positive |

`rank3_10` features:

1. `m5_deposit_agents`
2. `m2_merchantpay_merchants`
3. `m4_transfer_from_bank_volume`
4. `m5_withdraw_highest_amount`
5. `m4_paybill_volume`
6. `m4_transfer_from_bank_banks`
7. `m6_mm_send_total_value`
8. `m6_transfer_from_bank_banks`
9. `m3_withdraw_agents`
10. `m5_merchantpay_highest_amount`

`perpower` uses per-feature global rank powers:

- `m5_deposit_agents`: 1.5
- `m2_merchantpay_merchants`: 1.5
- `m4_transfer_from_bank_volume`: 2.5
- `m5_withdraw_highest_amount`: 2.5
- `m4_paybill_volume`: 4.0
- `m4_transfer_from_bank_banks`: 4.0
- `m6_mm_send_total_value`: 3.0
- `m6_transfer_from_bank_banks`: 4.0
- `m3_withdraw_agents`: 4.0
- `m5_merchantpay_highest_amount`: 4.0

## Robustness

- Paired customer bootstrap for `rank3_10` vs base (2,000 replicates):
  - Composite gain: mean `+0.000746`, sd `0.000256`, 95% CI `[+0.000240, +0.001248]`, 99.75% positive.
  - LogLoss improvement: mean `+0.000539`, 95% CI `[+0.000163, +0.000908]`.
  - AUC gain: mean `+0.000506`, 95% CI `[+0.000190, +0.000840]`.
- Per-month OOF gains for `rank3_10` are all positive, so the gain is not driven by a single snapshot.
- The rank-power curve is a broad plateau: `rank^2` `0.741182`, `rank^2.5` `0.741202`, `rank^3` `0.741210`, `rank^4` `0.741201`. The `rank^3` choice is not a sharp spike.
- The collinear pair `m4_transfer_from_bank_volume` / `m4_transfer_from_bank_banks` has Spearman correlation `0.968`. A single contrast feature `rank^3(banks) - rank^3(volume)` scores `0.741179` with one fewer parameter, while the pair/contrast-sum scores `0.741210`. The pair adds only `3.2e-5`, so it is a small, potentially noisy refinement.
- Greedy forward selection using only 3 training months did not reliably generalize to the held-out month. Treat the forward-selection path as exploratory; the fixed `rank3_10` set still has positive LOMO per-month gains, but the feature-selection process itself is unstable.
- A 5-seed LightGBM residual on the `rank3_10` structural init scored `0.741195`, i.e. `-1.6e-5` vs the structural model. The residual is redundant and was not scaled to 50 seeds.
- The no-`m5_deposit` fallback remains important because `m5_deposit_agents` is the column the 2nd-place solution found did not hold in test.
- Test predictions are highly correlated: `rank3_10` vs `perpower` Spearman `0.99988`; `rank3_10` vs `nom5` Spearman `0.99907`.

## Submission Files

All files have 30,000 rows, exact sample ID order, finite probabilities in `[0, 1]`,
overall mean `0.15`, and per-test-snapshot means `[0.15, 0.15, 0.15]`.

| File | LOMO | Role |
|---|---:|---|
| `submission_v7_struct_rank3_10.csv` | `0.741210425` | Recommended primary: best simple/robust structural model |
| `submission_v7_struct_perpower.csv` | `0.741274386` | Higher LOMO, slightly more tuned; compare public |
| `submission_v7_struct_rank3_10_nom5.csv` | `0.740985753` | Hedge against `m5_deposit` test shift |
| `submission_v7_blend_struct_v6res.csv` | `0.741283035` | 92.5% perpower + 7.5% v6 drop-m5 residual; marginal gain |
| `submission_v7_struct_contrast_r3.csv` | `0.741210426` | Equivalent to `rank3_10`; redundant |
| `submission_v7_struct_perpower_nom5.csv` | `0.740981466` | No-m5 perpower; slightly below `rank3_10_nom5` |

## Recommendation

Submit in this order and compare public scores:

1. `submission_v7_struct_rank3_10.csv` (primary)
2. `submission_v7_struct_perpower.csv` (upside candidate)
3. `submission_v7_struct_rank3_10_nom5.csv` (insurance)

If `perpower` beats `rank3_10` publicly, keep it. If it underperforms, fall back to
`rank3_10`. If the `m5_deposit` columns shift on the hidden test, the no-m5 fallback
should be more reliable. The blend is optional and only adds `+8.7e-6` OOF, so it is
not a priority unless the public leaderboard confirms the perpower direction.
