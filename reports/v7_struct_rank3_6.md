# v7 Structural Extra

- Variant: `rank3_6`
- Transform: `rank3`
- Fit mode: `fixed`
- Extra columns: `['m5_deposit_agents', 'm2_merchantpay_merchants', 'm4_transfer_from_bank_volume', 'm5_withdraw_highest_amount', 'm4_paybill_volume', 'm4_transfer_from_bank_banks']`
- Base LOMO: `0.740474849`
- Extra LOMO: `0.741054052`
- Gain: `+0.000579203`
- Extra LogLoss: `0.226285398`
- Extra AUC: `0.923102520`
- Full-train gamma: `[0.11924689485107767, -0.10751099281362096, -0.24654425344776487, 0.14926541981021432, -0.09773224191336402, 0.17908950414900476]`

| Snapshot | Base | Extra | Gain |
|---:|---:|---:|---:|
| 0 | 0.737187499 | 0.737999419 | +0.000811920 |
| 1 | 0.740178118 | 0.740503902 | +0.000325784 |
| 2 | 0.744951626 | 0.745509070 | +0.000557444 |
| 3 | 0.739443534 | 0.740109056 | +0.000665522 |

- Test mean: `0.150000000000`
- Test snapshot means: `[0.15, 0.15, 0.15]`
- Submission: `submission_v7_struct_rank3_6.csv`
