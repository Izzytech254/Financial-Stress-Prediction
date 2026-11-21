# v6 LightGBM Residual

- Seeds: `50`
- Month scaling: `rank`
- Round scale: `2.0`
- Drop m5 deposit columns: `True`
- Features: `197`
- Structural LOMO: `0.740474849`
- Residual LOMO: `0.740493622`
- Gain: `+0.000018773`
- Residual LogLoss: `0.226692007`
- Residual AUC: `0.922726510`

| Snapshot | Residual | Structural | Gain |
|---:|---:|---:|---:|
| 0 | 0.737193676 | 0.737187499 | +0.000006177 |
| 1 | 0.740211797 | 0.740178118 | +0.000033679 |
| 2 | 0.744930586 | 0.744951626 | -0.000021040 |
| 3 | 0.739493080 | 0.739443534 | +0.000049546 |

- Final rounds: `[34, 32, 34, 38, 14]`
- Test mean: `0.150000000000`
- Test snapshot means: `[0.15, 0.15, 0.15]`
- Elapsed seconds: `1113.6`
- Submission: `submission_v6_residual_dropm5_t200.csv`
