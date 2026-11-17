# v6 LightGBM Residual

- Seeds: `200`
- Month scaling: `rank`
- Round scale: `1.0`
- Drop m5 deposit columns: `True`
- Features: `197`
- Structural LOMO: `0.740474849`
- Residual LOMO: `0.740496366`
- Gain: `+0.000021517`
- Residual LogLoss: `0.226688141`
- Residual AUC: `0.922723623`

| Snapshot | Residual | Structural | Gain |
|---:|---:|---:|---:|
| 0 | 0.737203403 | 0.737187499 | +0.000015904 |
| 1 | 0.740197397 | 0.740178118 | +0.000019278 |
| 2 | 0.744941271 | 0.744951626 | -0.000010355 |
| 3 | 0.739504859 | 0.739443534 | +0.000061325 |

- Final rounds: `[17, 16, 17, 19, 7]`
- Test mean: `0.150000000000`
- Test snapshot means: `[0.15, 0.15, 0.15]`
- Elapsed seconds: `1354.9`
- Submission: `submission_v9_residual_dropm5_s200.csv`
