# v7 Structural Extra

- Variant: `prefix10`
- Transform: `rank3`
- Fit mode: `fixed`
- Extra columns: `['m5_deposit_agents', 'm2_merchantpay_merchants', 'm5_withdraw_highest_amount', 'm4_paybill_volume', 'm6_mm_send_total_value', 'm4_transfer_from_bank_volume', 'm4_transfer_from_bank_banks', 'm3_withdraw_agents', 'm1_paybill_total_value', 'm6_transfer_from_bank_banks']`
- Base LOMO: `0.740474849`
- Extra LOMO: `0.741186120`
- Gain: `+0.000711271`
- Extra LogLoss: `0.226186278`
- Extra AUC: `0.923182809`
- Full-train gamma: `[0.13266394290959888, -0.10172088422496041, 0.13387747433059657, -0.10580692010902747, 0.08288243492412137, -0.2723835892735196, 0.20640485744151274, -0.060377740786515095, 0.02494641629378333, -0.07705521402304424]`

| Snapshot | Base | Extra | Gain |
|---:|---:|---:|---:|
| 0 | 0.737187499 | 0.738021404 | +0.000833906 |
| 1 | 0.740178118 | 0.740816557 | +0.000638439 |
| 2 | 0.744951626 | 0.745452906 | +0.000501280 |
| 3 | 0.739443534 | 0.740345746 | +0.000902212 |

- Test mean: `0.150000000000`
- Test snapshot means: `[0.15, 0.15, 0.15]`
- Submission: `submission_v7_struct_prefix10.csv`
