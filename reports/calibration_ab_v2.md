# Calibration A/B v2

- Raw OOF AUC: `0.912623`
- Raw OOF LogLoss: `0.241631`
- Control Platt slope: `1.098401`, intercept: `0.135362`
- Control file matches current `submission.csv` within `5.00e-09`

| variant | file | test mean | OOF AUC | OOF LL in-sample | OOF LL CV | dLL CV vs control | max diff vs control |
|---|---|---:|---:|---:|---:|---:|---:|
| control | `submission_cal_control.csv` | 0.153486 | 0.912623 | 0.240875 | 0.240912 | +0.000000 | 0.000000 |
| s100 | `submission_cal_s100.csv` | 0.153396 | 0.912623 | 0.241615 | 0.241647 | +0.000735 | 0.039476 |
| s105 | `submission_cal_s105.csv` | 0.153443 | 0.912623 | 0.241048 | 0.241083 | +0.000171 | 0.018845 |
| prior150 | `submission_cal_prior150.csv` | 0.150000 | 0.912623 | 0.240958 | n/a | n/a | 0.012129 |

## Public A/B

| submission | Public Score | ROC AUC | LogLoss |
|---|---:|---:|---:|
| previous best (baseline) | 0.714583597 | 0.908407068 | 0.246706069 |
| prior150 (adopted) | 0.725863792 | 0.914406197 | 0.237899530 |
| delta | +0.011280195 | +0.005999129 | -0.008806539 |

`prior150` was the highest-scoring submitted variant. It is monotonic, so the AUC gain comes from v2, not calibration; the LL gain comes from the intercept shift to test mean `0.150`.

Adopted: `submission.csv` now matches `submission_cal_prior150.csv`.
