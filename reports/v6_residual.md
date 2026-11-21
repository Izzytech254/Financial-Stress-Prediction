# v6 LightGBM Residual

- Seeds: `50`
- Month scaling: `rank`
- Round scale: `1.0`
- Drop m5 deposit columns: `False`
- Features: `201`
- Structural LOMO: `0.740474849`
- Residual LOMO: `0.740562939`
- Gain: `+0.000088090`
- Residual LogLoss: `0.226641470`
- Residual AUC: `0.922772397`

| Snapshot | Residual | Structural | Gain |
|---:|---:|---:|---:|
| 0 | 0.737190402 | 0.737187499 | +0.000002904 |
| 1 | 0.740262745 | 0.740178118 | +0.000084627 |
| 2 | 0.745048391 | 0.744951626 | +0.000096765 |
| 3 | 0.739616123 | 0.739443534 | +0.000172589 |

- Final rounds: `[26, 26, 18, 55, 14]`
- Test mean: `0.150000000000`
- Test snapshot means: `[0.15, 0.15, 0.15]`
- Elapsed seconds: `1347.5`
- Submission: `submission_v6_residual.csv`
