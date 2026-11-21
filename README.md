# Financial Stress Prediction

Zindi financial-stress competition: predict whether a customer will be in financial
stress in each monthly snapshot.

## Data

- `40,000` training rows: `10,000` customers x `4` snapshots.
- `30,000` test rows: the same `10,000` customers x `3` future snapshots.
- Each snapshot has exactly `1,500` positives (`15%`).
- Metric:

```text
score = 0.6 * (1 - LogLoss / 0.595) + 0.4 * ROC_AUC
```

## Key insight

The labels are generated as the **top 15% of a latent per-snapshot risk score**:

```text
latent = S + N(0, 0.35^2)
S = weighted sum of seven capped liquidity risk factors, z-scored within snapshot
```

The seven factors are:

1. Balance decline (recent 3 months vs older 3 months)
2. Income decline
3. Withdrawal-to-income ratio
4. Spend-to-income ratio
5. Outflow-to-income ratio
6. Balance volatility
7. Inactivity (`1 - x_90_d_activity_rate`)

Hand weights `[0.25, 0.15, 0.15, 0.10, 0.20, 0.10, 0.05]` match the fitted probit
weights almost exactly. The structural probit is therefore the data-generating model,
not just a feature-engineered approximation.

## Models

### 1. Structural probit

`v6_structural.py` computes the seven risk factors, caps each at the 99th percentile,
z-scores within snapshot, applies the hand weights, and calibrates a probit intercept
so each snapshot has mean `0.15`.

- LOMO composite: `0.740474849`
- LOMO LogLoss: `0.226701877`
- LOMO AUC: `0.922704461`
- Public score: `0.744516108`

### 2. LightGBM residual

`v6_residual.py` uses the structural probit as an initial score and fits a small,
strongly regularized LightGBM residual on 197 features. The best public variant drops
all `m5_deposit_*` columns, which were found not to hold on the hidden test set.

- LOMO composite: `0.740497298`
- LOMO LogLoss: `0.226687864`
- LOMO AUC: `0.922725255`
- Public score: `0.744632428`

A 200-seed variance-reduced run (`submission_v9_residual_dropm5_s200.csv`) scored
LOMO `0.740496366`, LogLoss `0.226688141`, AUC `0.922723623`. Its test predictions
correlate `0.9999999` with the 50-seed model, so the public score is effectively
unchanged. The 50-seed model is already converged.

### 3. Structural variants

`v7_common.py` and `v8_screen.py` implement extra liquidity factors, rank transforms,
month scaling, feature selection, and drop-set variants. These improve LOMO but did not
improve the public leaderboard, which indicates a train/test distribution shift.

The `drop_m5_all` 50-seed hedge (`submission_v9_drop_m5_all_s50.csv`) scores LOMO
`0.740465127`, LogLoss `0.226717150`, AUC `0.922718657`, and public `0.744671189`.
Dropping both `m5_*` and `m6_*` improves both LOMO and the month-3 holdout:
`drop_m5_m6` 50-seed (`submission_v9_drop_m5_m6_s50.csv`) scores LOMO
`0.740498576`, LogLoss `0.226688815`, AUC `0.922730848`, month-3 `0.739527356`.
The `drop_m5_m6_m1paybill` variant, which also drops the top adversarial
`m1_paybill_*` block, scores LOMO `0.740495869` and month-3 `0.739534977`.

A LightGBM + XGBoost + CatBoost logit blend on `drop_m5_m6` scores LOMO
`0.740512887` and month-3 `0.739502099`; the two-model LightGBM + CatBoost blend
scores LOMO `0.740511625` and month-3 `0.739514120`. These are small,
within-noise gains. Transductive hard pseudo-labelling was negative (month-0
`-0.000574`) and was discarded.

## Validation

All model selection uses leave-one-month-out (LOMO) out-of-fold predictions on the
40,000 training rows. Test predictions are calibrated so each of the three test
snapshots has mean `0.15`.

## Frozen repeated-CV workflow

Leaderboard differences of `~0.0002` are within noise, so final selection uses a frozen
5-fold x 3-seed split in `reports/folds_v2.npz` (`42/2026/7`). The selection score is
`0.6*(1-LogLoss/0.595)+0.4*AUC`. Every accepted change must have a paired proxy delta
greater than one standard error over the 15 fold/repeat estimates. The manifest and
hashes are written to `reports/frozen_cv_manifest.json`.

`pipeline.py` runs the canonical stages:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python pipeline.py --stage manifest
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python pipeline.py --stage baseline
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python pipeline.py --stage features
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python pipeline.py --stage features_test
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python pipeline.py --stage features_preds
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python pipeline.py --stage ensemble2
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python pipeline.py --stage tune
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python pipeline.py --stage ensemble
```

Current frozen-CV results are in `reports/experiment_log_cv.md`. The trend and
concentration feature groups were accepted; liquidity and stress were rejected. The
selected raw model is `submission_feat_trend_conc_s50.csv`. A 50-trial Optuna sweep produces
`submission_tuned_trend_conc_s10.csv` (OOF proxy `0.740891250`). Nested isotonic calibration
of the three-way blend reaches OOF proxy `0.741244871`, but its public score collapsed to
`0.739335283`; post-hoc isotonic/temperature calibration is therefore rejected. The
public-safe candidates are `submission_tuned_trend_conc_s10.csv`,
`submission_feat_trend_conc_s50.csv`, and `submission_ensemble_three_way_raw.csv`.
Within-date rank isotonic calibration (`rank_calibrate_cv.py`) improves nested proxy to
`0.741156608` without extreme clipping; `submission_rankcal_best.csv`,
`submission_rankcal_baseline_cv_drop_m5_all_s50.csv`, and `submission_rankcal_tuned.csv`
are the next public tests.

## Leaderboard context

| Rank | Public score |
|---:|---:|
| 1 | `0.744833171` |
| 2 | `0.744763066` |
| 3 (this repo) | `0.744671204` |

The top three are within `0.0002`. We tested the obvious leak paths:

- next-snapshot features -> current target: AUC `~0.52`
- target history / persistence: lag correlation `~0.01`
- ID hash / row order: AUC `~0.50`
- exact and near-duplicate rows: none
- RNG seed recovery: no match across 400k seeds
- fixed-size sampling correction: no material gain
- segment-specific probit interactions: worse LOMO
- adversarial train/test AUC: `0.500376`; no train/test drift was detected
- time weighting / snapshot ordering: no reliable gain at 10 seeds

The data is at its Bayes ceiling. A large leaderboard lead is not achievable; the
realistic target is first place by a small margin.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python v6_structural.py
python v6_residual.py --seeds 50 --drop-m5-deposit
python v8_screen.py --variant drop_m5_all --seeds 50 --threads 8 --test \
  --tag drop_m5_all_s50 \
  --out submission_v9_drop_m5_all_s50.csv \
  --report reports/v9_drop_m5_all_s50.json \
  --preds reports/v9_drop_m5_all_s50_predictions.npz
python v8_screen.py --variant drop_m5_m6 --seeds 50 --threads 8 --test \
  --tag drop_m5_m6_s50 \
  --out submission_v9_drop_m5_m6_s50.csv \
  --report reports/v9_drop_m5_m6_s50.json \
  --preds reports/v9_drop_m5_m6_s50_predictions.npz
python v8_screen.py --variant drop_m5_m6_m1paybill --seeds 50 --threads 8 --test \
  --tag drop_m5_m6_m1paybill_s50 \
  --out submission_v9_drop_m5_m6_m1paybill_s50.csv \
  --report reports/v9_drop_m5_m6_m1paybill_s50.json \
  --preds reports/v9_drop_m5_m6_m1paybill_s50_predictions.npz
python v8_xgb.py --variant drop_m5_m6 --seeds 50 --threads 8 --test \
  --tag xgb_drop_m5_m6_s50 \
  --report reports/v9_xgb_drop_m5_m6_s50.json \
  --preds reports/v9_xgb_drop_m5_m6_s50_predictions.npz
python v8_cat.py --variant drop_m5_m6 --seeds 50 --threads 8 --test \
  --tag cat_drop_m5_m6_s50 \
  --report reports/v9_cat_drop_m5_m6_s50.json \
  --preds reports/v9_cat_drop_m5_m6_s50_predictions.npz
python v9_blend.py \
  --preds reports/v9_drop_m5_m6_s50_predictions.npz \
  --preds reports/v9_xgb_drop_m5_m6_s50_predictions.npz \
  --preds reports/v9_cat_drop_m5_m6_s50_predictions.npz \
  --names lgb_m6,xgb_m6,cat_m6 --mode logit \
  --out submission_v9_blend_m6_lgb_xgb_cat.csv \
  --report reports/v9_blend_m6_lgb_xgb_cat.json \
  --blend-preds reports/v9_blend_m6_lgb_xgb_cat_predictions.npz
python v9_time_residual.py --seeds 50 --time-weight linear --slope 0.5
```

The 200-seed run:

```bash
python v6_residual.py \
  --seeds 200 \
  --threads 8 \
  --drop-m5-deposit \
  --out submission_v9_residual_dropm5_s200.csv \
  --report reports/v9_residual_dropm5_s200.json \
  --preds reports/v9_residual_dropm5_s200_predictions.npz
```

## Repository layout

```text
v6_common.py          feature engineering, structural probit, metric, calibration
v6_structural.py      pure structural model and submission
v6_residual.py        structural-init LightGBM residual
v7_common.py          extra factors and rank transforms
v8_screen.py          variant screening and submission generator
v8_importance.py      feature importance
v8_xgb.py             XGBoost diversity check
v8_cat.py             CatBoost diversity check
v9_time_residual.py   time-weighted residual experiment
reports/              JSON/Markdown experiment summaries
submission_*.csv      selected submissions
```
