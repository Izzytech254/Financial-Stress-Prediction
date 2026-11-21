# v7 Structural Extra

- Variant: `extra4`
- Transform: `sqrt`
- Fit mode: `fixed`
- Extra columns: `['m5_deposit_agents', 'm2_merchantpay_merchants', 'm5_withdraw_highest_amount', 'm4_mm_send_recipients']`
- Base LOMO: `0.740474849`
- Extra LOMO: `0.740867269`
- Gain: `+0.000392420`
- Extra LogLoss: `0.226420094`
- Extra AUC: `0.922975132`
- Full-train gamma: `[0.03662739290904021, -0.02504374040423112, 0.0003079396891438439, -0.016001065859368887]`

| Snapshot | Base | Extra | Gain |
|---:|---:|---:|---:|
| 0 | 0.737187499 | 0.737848183 | +0.000660684 |
| 1 | 0.740178118 | 0.740320306 | +0.000142188 |
| 2 | 0.744951626 | 0.745499318 | +0.000547692 |
| 3 | 0.739443534 | 0.739667820 | +0.000224286 |

- Test mean: `0.150000000000`
- Test snapshot means: `[0.15, 0.15, 0.15]`
- Submission: `submission_v7_struct_extra4_sqrt.csv`
