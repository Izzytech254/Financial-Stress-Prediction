# Frozen-CV experiment log

Harness: `reports/folds_v2.npz`, stratified 5-fold repeats `42/2026/7`, paired delta vs the 5-seed baseline, selection metric `0.6*(1-log_loss/0.595)+0.4*auc`.

| Hypothesis | Change | OOF LogLoss | OOF AUC | Proxy | Paired delta ± SE | Decision |
|---|---|---:|---:|---:|---:|---|
| Baseline | Current best LGB residual, `drop_m5_all`, 5 seeds | 0.226549106 | 0.922806593 | 0.740669757 | — | reference |
| Liquidity ratios help | Add runway/coverage/net features | 0.226570230 | 0.922787613 | 0.740640864 | -0.00002972 ± 0.00001238 | reject |
| Trend signals help | Add log slopes, CV, recent/older ratios | 0.226471761 | 0.922870564 | 0.740773340 | +0.00010337 ± 0.00002033 | keep |
| Stress signals help | Add drawdowns, declines, near-zero months | 0.226550312 | 0.922803618 | 0.740667351 | -0.00000245 ± 0.00000619 | reject |
| Concentration helps | Add HHI/entropy and interactions | 0.226538440 | 0.922813054 | 0.740683097 | +0.00001297 ± 0.00000756 | keep (weak) |
| 50-seed baseline | Same baseline with 50 seeds | 0.226546251 | 0.922806941 | 0.740672775 | +0.00000302 vs 5-seed | variance reduction |
| Combine winners | Add trend + concentration features | 0.226444309 | 0.922891775 | 0.740809508 | +0.00013984 ± 0.00002139 | keep |
| Trend on M1-M4 only | Drop m5/m6 from trend features | 0.226544277 | 0.922810922 | 0.740676359 | +0.00000478 ± 0.00001506 | reject |
| LGB tuning | 50 Optuna trials, 10 seeds, short rounds | 0.226379500 | 0.922932745 | 0.740891250 | +0.00008015 ± 0.00003780 vs raw trend+conc | keep |
| Two-way blend | Baseline + trend/conc, iso calibration | 0.226094578 | 0.922827387 | 0.741136423 | +0.00033024 ± 0.00002843 vs raw trend+conc | keep |
| Three-way blend | Baseline + trend/conc + tuned, iso calibration | 0.225979758 | 0.922809047 | 0.741244871 | +0.00010829 ± 0.00003586 vs two-way | CV-selected but public-rejected |
| Three-way raw | Same blend, no calibration | 0.226392745 | 0.922924652 | 0.740874655 | -0.00037022 vs iso (in-sample) | public-safe fallback |
| Hedge 25/75 | 25% current best + 75% tuned | — | — | 0.740850112 | +0.00017579 ± 0.00002694 vs baseline | hedge candidate |

| Rank calibration (tuned) | Iso map on within-date rank | — | — | 0.741156608 | +0.00027051 ± 0.00003925 vs tuned | candidate |
| Rank calibration (baseline) | Iso map on within-date rank | — | — | 0.741095147 | +0.00042512 ± 0.00005843 vs baseline | candidate |
| Rank calibration (best test) | Baseline-derived map on current-best test ranks | — | — | — | — | candidate |

Public feedback (2026-11-18): the isotonic-calibrated blend scored `0.739335283` with
LogLoss `0.228750672`, despite its strong nested-CV proxy. This indicates the OOF-to-test
calibration mapping does not transfer; isotonic/temperature/post-hoc calibration is now
rejected. The two non-calibrated models scored `0.744661667` and `0.744643099`, both
slightly below the existing best `0.744671204`; ID-to-file mapping still needs confirmation.
Two further raw/hedged submissions scored `0.744668297` and `0.744671047`, the latter
effectively tied with the existing best. These confirm a plateau for raw probability blends.
Rank-based isotonic calibration is the next untested shape correction; unlike plain
probability isotonic it keeps a mild `~0.001` lower tail and passes nested CV.
