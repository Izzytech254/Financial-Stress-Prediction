# v7 Structural Extra

- Variant: `rank3_9`
- Transform: `rank3`
- Fit mode: `fixed`
- Extra columns: `['m5_deposit_agents', 'm2_merchantpay_merchants', 'm4_transfer_from_bank_volume', 'm5_withdraw_highest_amount', 'm4_paybill_volume', 'm4_transfer_from_bank_banks', 'm6_mm_send_total_value', 'm6_transfer_from_bank_banks', 'm3_withdraw_agents']`
- Base LOMO: `0.740474849`
- Extra LOMO: `0.741184072`
- Gain: `+0.000709223`
- Extra LogLoss: `0.226186415`
- Extra AUC: `0.923178034`
- Full-train gamma: `[0.13382427412342865, -0.10004138281564388, -0.2677100778254527, 0.1399675111297478, -0.10134659683427819, 0.20427353774904644, 0.08848130321381129, -0.07525728022081016, -0.05917587675692383]`

| Snapshot | Base | Extra | Gain |
|---:|---:|---:|---:|
| 0 | 0.737187499 | 0.738032054 | +0.000844556 |
| 1 | 0.740178118 | 0.740823504 | +0.000645386 |
| 2 | 0.744951626 | 0.745435111 | +0.000483485 |
| 3 | 0.739443534 | 0.740336451 | +0.000892917 |

- Test mean: `0.150000000000`
- Test snapshot means: `[0.15, 0.15, 0.15]`
- Submission: `submission_v7_struct_rank3_9.csv`
