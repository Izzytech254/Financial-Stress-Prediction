# v6 LightGBM Residual

- Seeds: `10`
- Month scaling: `rank`
- Round scale: `1.0`
- Drop m5 deposit columns: `False`
- Features: `201`
- Structural LOMO: `0.740474849`
- Residual LOMO: `0.740556180`
- Gain: `+0.000081331`
- Residual LogLoss: `0.226645380`
- Residual AUC: `0.922765358`

| Snapshot | Residual | Structural | Gain |
|---:|---:|---:|---:|
| 0 | 0.737191015 | 0.737187499 | +0.000003516 |
| 1 | 0.740248169 | 0.740178118 | +0.000070051 |
| 2 | 0.745046008 | 0.744951626 | +0.000094382 |
| 3 | 0.739606563 | 0.739443534 | +0.000163029 |

- Final rounds: `[26, 26, 18, 55, 14]`
- Test mean: `0.150000000000`
- Test snapshot means: `[0.15, 0.15, 0.15]`
- Elapsed seconds: `214.2`
- Submission: `submission_v6_residual_s10.csv`
