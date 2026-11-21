# v7 Structural Extra

- Variant: `rank3_10_nom5`
- Transform: `rank3`
- Fit mode: `fixed`
- Extra columns: `['m2_merchantpay_merchants', 'm4_transfer_from_bank_volume', 'm5_withdraw_highest_amount', 'm4_paybill_volume', 'm4_transfer_from_bank_banks', 'm6_mm_send_total_value', 'm6_transfer_from_bank_banks', 'm3_withdraw_agents', 'm5_merchantpay_highest_amount']`
- Base LOMO: `0.740474849`
- Extra LOMO: `0.740985753`
- Gain: `+0.000510904`
- Extra LogLoss: `0.226340251`
- Extra AUC: `0.923070059`
- Full-train gamma: `[-0.08566727927226135, -0.29034921767239086, 0.13792777425458838, -0.09881082139829202, 0.23368607825487622, 0.08570218578223433, -0.0649149277530382, -0.03878372781662881, 0.06260634031929727]`

| Snapshot | Base | Extra | Gain |
|---:|---:|---:|---:|
| 0 | 0.737187499 | 0.737803930 | +0.000616431 |
| 1 | 0.740178118 | 0.740502532 | +0.000324414 |
| 2 | 0.744951626 | 0.745463854 | +0.000512228 |
| 3 | 0.739443534 | 0.740059097 | +0.000615562 |

- Test mean: `0.150000000000`
- Test snapshot means: `[0.15, 0.15, 0.15]`
- Submission: `submission_v7_struct_rank3_10_nom5.csv`
