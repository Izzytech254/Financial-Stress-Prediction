# v2 Executive Summary

- Baseline OOF AUC: **0.90659**; LogLoss: **0.24899**.
- Final OOF AUC: **0.91256**; cross-validated LogLoss: **0.24095**; accepted: **True**.
- Paired AUC delta vs baseline: +0.00599 +/- 0.00066; LogLoss delta: -0.00804 +/- 0.00066.
- Final blend: leak-free LightGBM v2 + reduced-budget CatBoost v2 only (mean-logit, Platt-calibrated).
- Best screening feature set: `combined` (base + customer features using Train+Test rows + history features).
- GroupKFold AUC: 0.91026.
- CatBoost OOF AUC: 0.91141.

## Interpretation

Customer-level features were evaluated with leave-one-row-out statistics from features only. The `tt` variant uses unlabeled test rows; the `tr` variant uses Train rows only. Random-row CV is the test proxy because every test customer appears in Train. GroupKFold is a customer-memorization sensitivity check, not the primary selection criterion.

## What worked

- The 20 targeted history features were the main driver: +0.00473 AUC and -0.00609 LogLoss in the 5-fold screen.
- Three-repeat LightGBM confirmation reached OOF AUC **0.91202**, and adding CatBoost improved the final blend to **0.91256**.
- GroupKFold AUC **0.91026** was essentially identical to the random-row screen **0.91030**, so there is no evidence of a customer-memorization gap.
- Adversarial validation AUC **0.50068** showed no Train/Test drift.

## What did not work

- Customer-level leave-one-row-out aggregates alone were rejected: `cust_tr` delta **-0.00043**, `cust_tt` delta **-0.00043**.
- Using unlabeled test rows in customer aggregates did not outperform Train-only aggregates.
- The combined customer+history set was only **+0.00007** AUC above history alone, which is within noise; the history features are the real source of the gain.
- Legacy baseline XGBoost/HistGradientBoosting models added no value to the final blend and were removed, keeping the final blend leak-free.

## Residual risks

- The 30-minute budget precludes nested Optuna; both models use fixed hyperparameters.
- CatBoost uses one 5-fold repeat, so its contribution is estimated with fewer folds.
- The baseline LogLoss is in-sample calibrated; new LogLoss is cross-validated.
- Small gains may remain within fold noise; the acceptance gate is deliberately conservative.
