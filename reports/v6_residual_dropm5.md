# v6 LightGBM Residual

- Seeds: `50`
- Month scaling: `rank`
- Round scale: `1.0`
- Drop m5 deposit columns: `True`
- Features: `197`
- Structural LOMO: `0.740474849`
- Residual LOMO: `0.740497298`
- Gain: `+0.000022449`
- Residual LogLoss: `0.226687864`
- Residual AUC: `0.922725255`

| Snapshot | Residual | Structural | Gain |
|---:|---:|---:|---:|
| 0 | 0.737207780 | 0.737187499 | +0.000020281 |
| 1 | 0.740196222 | 0.740178118 | +0.000018104 |
| 2 | 0.744946856 | 0.744951626 | -0.000004771 |
| 3 | 0.739497504 | 0.739443534 | +0.000053970 |

- Final rounds: `[17, 16, 17, 19, 7]`
- Test mean: `0.150000000000`
- Test snapshot means: `[0.15, 0.15, 0.15]`
- Elapsed seconds: `1277.7`
- Submission: `submission_v6_residual_dropm5.csv`
