# v7 LightGBM Residual on Extra Structural Init

- Variant: `extra4`
- Transform: `global_rank`
- Fit mode: `fixed`
- Seeds: `5`
- Month scaling: `rank`
- Round scale: `1.0`
- Drop m5 deposit columns: `False`
- Features: `201`
- Structural base LOMO: `0.740474849`
- Structural extra LOMO: `0.740876177`
- Residual LOMO: `0.740876025`
- Gain vs extra: `-0.000000151`
- Gain vs base: `+0.000401176`
- Residual LogLoss: `0.226410975`
- Residual AUC: `0.922974034`

| Snapshot | Residual | Extra | Base | Gain vs Extra |
|---:|---:|---:|---:|---:|
| 0 | 0.737873159 | 0.737873685 | 0.737187499 | -0.000000526 |
| 1 | 0.740271035 | 0.740240745 | 0.740178118 | +0.000030290 |
| 2 | 0.745528862 | 0.745556195 | 0.744951626 | -0.000027333 |
| 3 | 0.739698725 | 0.739702780 | 0.739443534 | -0.000004055 |

- Final rounds: `[4, 8, 8, 14, 2]`
- Test mean: `0.150000000000`
- Test snapshot means: `[0.15, 0.15, 0.15]`
- Elapsed seconds: `163.9`
- Submission: `submission_v7_residual_extra4_s5.csv`
