# v7 Structural Extra

- Variant: `rank3_7`
- Transform: `rank3`
- Fit mode: `fixed`
- Extra columns: `['m5_deposit_agents', 'm2_merchantpay_merchants', 'm4_transfer_from_bank_volume', 'm5_withdraw_highest_amount', 'm4_paybill_volume', 'm4_transfer_from_bank_banks', 'm6_mm_send_total_value']`
- Base LOMO: `0.740474849`
- Extra LOMO: `0.741109149`
- Gain: `+0.000634300`
- Extra LogLoss: `0.226247127`
- Extra AUC: `0.923143779`
- Full-train gamma: `[0.11340326853505218, -0.116176978190057, -0.26686154453776717, 0.12008868515951261, -0.11555417448393464, 0.18903961051794493, 0.07401793102923576]`

| Snapshot | Base | Extra | Gain |
|---:|---:|---:|---:|
| 0 | 0.737187499 | 0.737993425 | +0.000805926 |
| 1 | 0.740178118 | 0.740566294 | +0.000388175 |
| 2 | 0.744951626 | 0.745607154 | +0.000655528 |
| 3 | 0.739443534 | 0.740159330 | +0.000715796 |

- Test mean: `0.150000000000`
- Test snapshot means: `[0.15, 0.15, 0.15]`
- Submission: `submission_v7_struct_rank3_7.csv`
