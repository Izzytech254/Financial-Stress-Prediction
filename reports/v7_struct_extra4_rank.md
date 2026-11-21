# v7 Structural Extra

- Variant: `extra4`
- Transform: `rank`
- Fit mode: `fixed`
- Extra columns: `['m5_deposit_agents', 'm2_merchantpay_merchants', 'm5_withdraw_highest_amount', 'm4_mm_send_recipients']`
- Base LOMO: `0.740474849`
- Extra LOMO: `0.740874680`
- Gain: `+0.000399831`
- Extra LogLoss: `0.226410746`
- Extra AUC: `0.922970093`
- Full-train gamma: `[0.12832950739349347, -0.1258534330511885, 0.08624273787984146, -0.0898557019481932]`

| Snapshot | Base | Extra | Gain |
|---:|---:|---:|---:|
| 0 | 0.737187499 | 0.737874410 | +0.000686912 |
| 1 | 0.740178118 | 0.740232314 | +0.000054196 |
| 2 | 0.744951626 | 0.745557421 | +0.000605795 |
| 3 | 0.739443534 | 0.739703132 | +0.000259598 |

- Test mean: `0.150000000000`
- Test snapshot means: `[0.15, 0.15, 0.15]`
- Submission: `submission_v7_struct_extra4_rank.csv`
