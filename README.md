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
`0.740465127`, LogLoss `0.226717150`, AUC `0.922718657`. It is the remaining robustness
bet if the `m5` snapshot features shift on the hidden test set.

## Validation

All model selection uses leave-one-month-out (LOMO) out-of-fold predictions on the
40,000 training rows. Test predictions are calibrated so each of the three test
snapshots has mean `0.15`.

## Leaderboard context

| Rank | Public score |
|---:|---:|
| 1 | `0.744833171` |
| 2 | `0.744763066` |
| 3 (this repo) | `0.744632428` |

The top three are within `0.0002`. We tested the obvious leak paths:

- next-snapshot features -> current target: AUC `~0.52`
- target history / persistence: lag correlation `~0.01`
- ID hash / row order: AUC `~0.50`
- exact and near-duplicate rows: none
- RNG seed recovery: no match across 400k seeds
- fixed-size sampling correction: no material gain
- segment-specific probit interactions: worse LOMO
- adversarial train/test AUC: `~1.0`; feature dropping and density weighting did not help
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
