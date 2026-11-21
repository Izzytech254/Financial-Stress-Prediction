# v7 Structural Extra

- Variant: `rank3_8`
- Transform: `rank3`
- Fit mode: `fixed`
- Extra columns: `['m5_deposit_agents', 'm2_merchantpay_merchants', 'm4_transfer_from_bank_volume', 'm5_withdraw_highest_amount', 'm4_paybill_volume', 'm4_transfer_from_bank_banks', 'm6_mm_send_total_value', 'm6_transfer_from_bank_banks']`
- Base LOMO: `0.740474849`
- Extra LOMO: `0.741152198`
- Gain: `+0.000677349`
- Extra LogLoss: `0.226215313`
- Extra AUC: `0.923171201`
- Full-train gamma: `[0.12357154245657258, -0.10770458652615005, -0.2638672793704784, 0.1335513101524575, -0.10622766336684202, 0.19486591369330655, 0.08404836888777006, -0.08220444023846912]`

| Snapshot | Base | Extra | Gain |
|---:|---:|---:|---:|
| 0 | 0.737187499 | 0.737933482 | +0.000745984 |
| 1 | 0.740178118 | 0.740745393 | +0.000567275 |
| 2 | 0.744951626 | 0.745479096 | +0.000527470 |
| 3 | 0.739443534 | 0.740345652 | +0.000902118 |

- Test mean: `0.150000000000`
- Test snapshot means: `[0.15, 0.15, 0.15]`
- Submission: `submission_v7_struct_rank3_8.csv`
