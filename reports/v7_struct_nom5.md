# v7 Structural Extra

- Variant: `nom5`
- Transform: `sqrt`
- Fit mode: `fixed`
- Extra columns: `['m2_merchantpay_merchants', 'm5_withdraw_highest_amount', 'm4_mm_send_recipients']`
- Base LOMO: `0.740474849`
- Extra LOMO: `0.740664810`
- Gain: `+0.000189961`
- Extra LogLoss: `0.226572354`
- Extra AUC: `0.922852833`
- Full-train gamma: `[-0.0191777916212169, 0.0004458888656135107, -0.009567455622770612]`

| Snapshot | Base | Extra | Gain |
|---:|---:|---:|---:|
| 0 | 0.737187499 | 0.737602184 | +0.000414686 |
| 1 | 0.740178118 | 0.740141686 | -0.000036432 |
| 2 | 0.744951626 | 0.745173065 | +0.000221439 |
| 3 | 0.739443534 | 0.739607137 | +0.000163603 |

- Test mean: `0.150000000000`
- Test snapshot means: `[0.15, 0.15, 0.15]`
- Submission: `submission_v7_struct_nom5.csv`
