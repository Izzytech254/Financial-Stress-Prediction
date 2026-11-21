import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

ROOT = Path('/home/boss/Desktop/zindi/financial-stress')
DATA = ROOT / 'data'

def load():
    train = pd.read_csv(DATA / 'Train.csv')
    test = pd.read_csv(DATA / 'Test.csv')
    sample = pd.read_csv(DATA / 'SampleSubmission.csv')
    return train, test, sample

def proxy(y, p):
    p = np.clip(p, 1e-15, 1-1e-15)
    ll = log_loss(y, p)
    auc = roc_auc_score(y, p)
    return 0.6*(1-ll/0.595) + 0.4*auc, ll, auc

print('=== STEP 1: DATA UNDERSTANDING ===')
train, test, sample = load()
print('\n1) Shapes/dtypes/target')
print('Train shape:', train.shape)
print('Test shape:', test.shape)
print('SampleSubmission shape:', sample.shape)
print('Train dtypes:\n', train.dtypes.value_counts())
print('Target column:', [c for c in train.columns if c not in test.columns])
y = train['liquidity_stress_next_30d']
print('Target balance counts:', y.value_counts().to_dict(), 'rate:', y.mean())
print('Missing train:', int(train.isna().sum().sum()), 'Missing test:', int(test.isna().sum().sum()))
print('Duplicate rows train:', int(train.duplicated().sum()), 'test:', int(test.duplicated().sum()))
print('Duplicate rows excl ID train:', int(train.drop(columns=['ID']).duplicated().sum()), 'test:', int(test.drop(columns=['ID']).duplicated().sum()))
print('Duplicate IDs train:', int(train['ID'].duplicated().sum()), 'test:', int(test['ID'].duplicated().sum()))
print('Train IDs in test:', int(train['ID'].isin(test['ID']).sum()))
print('Constant cols:', [c for c in train.columns if train[c].nunique(dropna=False)<=1])

print('\nCardinalities (selected):')
nu = train.nunique()
cats = ['gender','region','smartphone','segment','earning_pattern']
for c in cats:
    print(f'  {c}: train={train[c].nunique()} test={test[c].nunique()} vals={sorted(train[c].unique().tolist())}')
print('  age min/max:', train['age'].min(), train['age'].max())
print('  x_90_d_activity_rate nunique train:', train['x_90_d_activity_rate'].nunique())
low = nu.sort_values().head(12)
high = nu.sort_values().tail(8)
print('  Lowest nunique:', low.to_dict())
print('  Highest nunique:', high.to_dict())

print('\n2) Submission format')
print('Sample columns:', list(sample.columns))
print('Sample dtypes:', sample.dtypes.to_dict())
print('ID order match test:', bool((sample['ID'].values==test['ID'].values).all()))
print('Sample Target unique/sample:', sample['Target'].unique()[:5].tolist(), 'min/max', sample['Target'].min(), sample['Target'].max())

print('\n3) Metric')
print('README metric: score = 0.6 * (1 - LogLoss / 0.595) + 0.4 * ROC_AUC')
print('Competition page (SPA) not readable via fetch; must confirm live. Using README/proxy formula.')

print('\n4) Entities / repeated structure')
profile = ['arpu','age','gender','region','smartphone','segment','earning_pattern','x_90_d_activity_rate']
d_tr = np.tile(np.arange(4), 10000)
d_te = np.tile(np.arange(3), 10000) + 4
# verify block structure
block_ok = all((train[profile].iloc[i*4:(i+1)*4].nunique() <= 1).all() for i in range(10000))
print('Each consecutive 4 train rows same profile (10k customers):', block_ok)
both = pd.concat([train[profile], test[profile]], ignore_index=True)
cust = both.groupby(profile, sort=False).ngroup().to_numpy()
print('Unique customer profiles:', int(np.unique(cust).size))
print('Train cust range:', cust[:len(train)].min(), cust[:len(train)].max(), 'Test cust range:', cust[len(train):].min(), cust[len(train):].max())
print('Same customers in train/test?', set(np.unique(cust[:len(train)])) == set(np.unique(cust[len(train):])))
print('Per-snapshot positives train:', pd.Series(y.values).groupby(d_tr).sum().to_dict(), 'rates:', pd.Series(y.values).groupby(d_tr).mean().to_dict())
print('=> Repeated entity: customer (static profile). Temporal: 4 train snapshots (0-3) then 3 test snapshots (4-6) per same customer. Leakage risk if rows from same customer/snapshot ordering mixed naively; time-aware/LOMO validation recommended. Frozen harness uses stratified 5x3 row folds (repeated entity leakage possible across rows of same customer) - must interpret alongside LOMO.')

print('\n5) Adversarial validation (train vs test)')
# Approach A: use all feature columns as-is (no ID/target), treat each row independent; encode cats.
# Note: same static profiles appear in both sets but snapshot-month features (m1-m6 windows) correspond to different time points - some shift expected.
feat_cols = [c for c in test.columns if c != 'ID']
adv_tr = train[feat_cols].copy()
adv_te = test[feat_cols].copy()
adv_tr['__is_test'] = 0
adv_te['__is_test'] = 1
adv = pd.concat([adv_tr, adv_te], ignore_index=True)
for c in cats:
    adv[c] = adv[c].astype('category').cat.codes
# drop columns constant across combined
nun = adv.nunique()
drop_const = nun[nun <= 1].index.tolist()
X_adv = adv.drop(columns=['__is_test'] + drop_const)
y_adv = adv['__is_test'].values
print('Adv features:', X_adv.shape[1], 'dropped const:', drop_const)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(y_adv))
importances = np.zeros(X_adv.shape[1])
for tr_i, va_i in skf.split(X_adv, y_adv):
    m = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=20, n_jobs=-1, random_state=42)
    m.fit(X_adv.iloc[tr_i], y_adv[tr_i])
    oof[va_i] = m.predict_proba(X_adv.iloc[va_i])[:,1]
    importances += m.feature_importances_
auc_adv = roc_auc_score(y_adv, oof)
print('Adversarial AUC (RF, row-stratified):', auc_adv)
imp = pd.Series(importances/5, index=X_adv.columns).sort_values(ascending=False)
print('Top 15 adv features:', imp.head(15).to_dict())

# Approach B: customer-level / time-aware caution - average AUC by snapshot? 
print('Adversarial pred mean train rows:', oof[y_adv==0].mean(), 'test rows:', oof[y_adv==1].mean())

# Approach C: compare distributions of key columns
def ks_like(a,b):
    # simple standardized mean diff
    ma, mb = a.mean(), b.mean()
    sa, sb = a.std()+1e-12, b.std()+1e-12
    return abs(ma-mb) / ((sa+sb)/2)
num_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(train[c])]
drifts = {c: ks_like(train[c].values.astype(float), test[c].values.astype(float)) for c in num_cols}
drifts_s = pd.Series(drifts).sort_values(ascending=False)
print('Top 15 standardized mean diffs (train vs test):', drifts_s.head(15).to_dict())
print('Interpretation: test rows are future snapshots (4-6) of same customers; temporal shift in monthly behavior is structural, not sample drift to "drop" naively. High adv signal would need snapshot-label leakage or genuine covariate shift; investigate top features.')

print('\n=== STEP 1 DONE ===')
