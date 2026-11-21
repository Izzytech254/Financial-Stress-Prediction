# v7 Structural Extra

- Variant: `rank3_10`
- Transform: `rank3`
- Fit mode: `fixed`
- Extra columns: `['m5_deposit_agents', 'm2_merchantpay_merchants', 'm4_transfer_from_bank_volume', 'm5_withdraw_highest_amount', 'm4_paybill_volume', 'm4_transfer_from_bank_banks', 'm6_mm_send_total_value', 'm6_transfer_from_bank_banks', 'm3_withdraw_agents', 'm5_merchantpay_highest_amount']`
- Base LOMO: `0.740474849`
- Extra LOMO: `0.741210425`
- Gain: `+0.000735576`
- Extra LogLoss: `0.226170392`
- Extra AUC: `0.923203520`
- Full-train gamma: `[0.13136820189561194, -0.10358283703416774, -0.2768439935275431, 0.12712246330505186, -0.11004977514071478, 0.2087085137827665, 0.0759737020297579, -0.0791968652083556, -0.062017123538268756, 0.053986810703678716]`

| Snapshot | Base | Extra | Gain |
|---:|---:|---:|---:|
| 0 | 0.737187499 | 0.738071595 | +0.000884097 |
| 1 | 0.740178118 | 0.740825688 | +0.000647569 |
| 2 | 0.744951626 | 0.745503639 | +0.000552013 |
| 3 | 0.739443534 | 0.740327286 | +0.000883752 |

- Test mean: `0.150000000000`
- Test snapshot means: `[0.15, 0.15, 0.15]`
- Submission: `submission_v7_struct_rank3_10.csv`
