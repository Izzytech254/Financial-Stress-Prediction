# v7 LightGBM Residual on Extra Structural Init

- Variant: `rank3_10`
- Transform: `rank3`
- Fit mode: `fixed`
- Seeds: `5`
- Month scaling: `rank`
- Round scale: `1.0`
- Drop m5 deposit columns: `False`
- Features: `201`
- Structural base LOMO: `0.740474849`
- Structural extra LOMO: `0.741210425`
- Residual LOMO: `0.741194541`
- Gain vs extra: `-0.000015884`
- Gain vs base: `+0.000719692`
- Residual LogLoss: `0.226184706`
- Residual AUC: `0.923199897`

| Snapshot | Residual | Extra | Base | Gain vs Extra |
|---:|---:|---:|---:|---:|
| 0 | 0.738072880 | 0.738071595 | 0.737187499 | +0.000001285 |
| 1 | 0.740839540 | 0.740825688 | 0.740178118 | +0.000013852 |
| 2 | 0.745503163 | 0.745503639 | 0.744951626 | -0.000000476 |
| 3 | 0.740250150 | 0.740327286 | 0.739443534 | -0.000077137 |

- Final rounds: `[1, 7, 1, 1, 2]`
- Test mean: `0.150000000000`
- Test snapshot means: `[0.15, 0.15, 0.15]`
- Elapsed seconds: `335.3`
- Submission: `submission_v7_residual_rank3_10.csv`
