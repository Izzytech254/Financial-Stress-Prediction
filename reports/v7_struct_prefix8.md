# v7 Structural Extra

- Variant: `prefix8`
- Transform: `rank3`
- Fit mode: `fixed`
- Extra columns: `['m5_deposit_agents', 'm2_merchantpay_merchants', 'm5_withdraw_highest_amount', 'm4_paybill_volume', 'm6_mm_send_total_value', 'm4_transfer_from_bank_volume', 'm4_transfer_from_bank_banks', 'm3_withdraw_agents']`
- Base LOMO: `0.740474849`
- Extra LOMO: `0.741149387`
- Gain: `+0.000674538`
- Extra LogLoss: `0.226209917`
- Extra AUC: `0.923150569`
- Full-train gamma: `[0.12607853531970184, -0.10661673548339105, 0.12869992978829517, -0.10907928325078556, 0.08003769246490026, -0.2709695795266623, 0.20034964711100242, -0.06754021111959692]`

| Snapshot | Base | Extra | Gain |
|---:|---:|---:|---:|
| 0 | 0.737187499 | 0.738111248 | +0.000923750 |
| 1 | 0.740178118 | 0.740671981 | +0.000493863 |
| 2 | 0.744951626 | 0.745531810 | +0.000580184 |
| 3 | 0.739443534 | 0.740168376 | +0.000724841 |

- Test mean: `0.150000000000`
- Test snapshot means: `[0.15, 0.15, 0.15]`
- Submission: `submission_v7_struct_prefix8.csv`
