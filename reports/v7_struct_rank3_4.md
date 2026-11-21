# v7 Structural Extra

- Variant: `rank3_4`
- Transform: `rank3`
- Fit mode: `fixed`
- Extra columns: `['m5_deposit_agents', 'm2_merchantpay_merchants', 'm4_transfer_from_bank_volume', 'm5_withdraw_highest_amount']`
- Base LOMO: `0.740474849`
- Extra LOMO: `0.740906160`
- Gain: `+0.000431311`
- Extra LogLoss: `0.226397913`
- Extra AUC: `0.923016441`
- Full-train gamma: `[0.11360930024235429, -0.11867329996869794, -0.09617192623934591, 0.09948252863137831]`

| Snapshot | Base | Extra | Gain |
|---:|---:|---:|---:|
| 0 | 0.737187499 | 0.737865414 | +0.000677915 |
| 1 | 0.740178118 | 0.740164251 | -0.000013867 |
| 2 | 0.744951626 | 0.745453005 | +0.000501379 |
| 3 | 0.739443534 | 0.740032372 | +0.000588838 |

- Test mean: `0.150000000000`
- Test snapshot means: `[0.15, 0.15, 0.15]`
- Submission: `submission_v7_struct_rank3_4.csv`
