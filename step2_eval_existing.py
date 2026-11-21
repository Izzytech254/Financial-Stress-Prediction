import json, glob
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import log_loss, roc_auc_score

ROOT = Path('/home/boss/Desktop/zindi/financial-stress')

def proxy(y,p):
    p=np.clip(p,1e-15,1-1e-15)
    ll=log_loss(y,p); auc=roc_auc_score(y,p)
    return 0.6*(1-ll/0.595)+0.4*auc, ll, auc

y = pd.read_csv(ROOT/'data/Train.csv')['liquidity_stress_next_30d'].values
d_tr = np.tile(np.arange(4), 10000)

folds_z = np.load(ROOT/'reports/folds_v2.npz')
folds = []
for rep in [42,2026,7]:
    for f in range(5):
        folds.append((rep,f,folds_z[f'tr_{rep}_{f}'], folds_z[f'va_{rep}_{f}']))

def eval_oof(name, oof, repeats=None):
    oof=np.clip(oof,1e-15,1-1e-15)
    overall = proxy(y,oof)
    rows=[]
    for rep,f,tr,va in folds:
        rows.append(proxy(y[va], oof[va])[0])
    rows=np.array(rows)
    # per repeat
    rep_scores=[]
    if repeats is not None and repeats.ndim==2:
        for r in range(repeats.shape[0]):
            rep_scores.append(proxy(y, np.clip(repeats[r],1e-15,1-1e-15))[0])
    print(f'{name}: proxy={overall[0]:.9f} ll={overall[1]:.9f} auc={overall[2]:.9f} fold_std={rows.std(ddof=1):.9f} fold_se={rows.std(ddof=1)/np.sqrt(len(rows)):.9f}')
    if rep_scores:
        print(f'  per-repeat proxy: {[f"{s:.9f}" for s in rep_scores]} mean={np.mean(rep_scores):.9f} std={np.std(rep_scores,ddof=1):.9f}')
    # LOMO-style by snapshot (proxy per held-out snapshot month)
    lomo=[]
    for m in range(4):
        mask=d_tr==m
        lomo.append(proxy(y[mask], oof[mask])[0])
    print(f'  per-snapshot proxy (0-3): {[f"{s:.9f}" for s in lomo]} std={np.std(lomo,ddof=1):.9f}')

print('=== Existing frozen-CV models (from saved predictions) ===')
candidates = [
    ('baseline_cv_drop_m5_all_s50', 'reports/baseline_cv_drop_m5_all_s50_predictions.npz'),
    ('feat_trend_conc_s5', 'reports/feat_trend_conc_s5_predictions.npz'),
    ('tuned_trend_conc_s10', 'reports/tuned_trend_conc_s10_predictions.npz'),
    ('feat_trend_conc_s50_predsfile', 'reports/feat_trend_conc_s50_predictions.npz'),
    ('ensemble_three_way_raw_preds', 'reports/ensemble_three_way_raw_predictions.npz'),
    ('ensemble_baseline_trendconc_preds', 'reports/ensemble_baseline_trendconc_predictions.npz'),
]
for name, path in candidates:
    p = ROOT/path
    if not p.exists():
        print(name, 'MISSING'); continue
    z=np.load(p)
    oof=z['oof']
    reps = z['oof_repeats'] if 'oof_repeats' in z.files else None
    eval_oof(name, oof, reps)

# also evaluate submissions? we can compare test distributions and do pairwise correlations
print('\n=== Key submission stats ===')
samp=pd.read_csv(ROOT/'data/SampleSubmission.csv')
subs = [
 'submission_baseline_cv_drop_m5_all_s50.csv',
 'submission_feat_trend_conc_s50.csv',
 'submission_tuned_trend_conc_s10.csv',
 'submission_ensemble_three_way_raw.csv',
 'submission_ensemble_baseline_trendconc.csv',
 'submission_rankcal_best.csv',
 'submission_rankcal_tuned.csv',
 'submission_hedge_best_raw3_25.csv',
 'submission_hedge_best_tune_50.csv',
 'submission_v9_drop_m5_m6_s50.csv',
 'submission_v9_blend_m6_lgb_xgb_cat.csv',
 'submission_v6_residual.csv',
 'submission_v6_structural.csv',
]
loaded={}
for s in subs:
    p=ROOT/s
    if not p.exists(): continue
    d=pd.read_csv(p)
    loaded[s]=d['Target'].values
    ok=(d['ID'].values==samp['ID'].values).all()
    v=d['Target'].values
    print(f'{s}: id_ok={ok} mean={v.mean():.6f} min={v.min():.6g} max={v.max():.6g} std={v.std():.6f}')

print('\nPairwise corr (selected):')
keys=list(loaded.keys())
base='submission_baseline_cv_drop_m5_all_s50.csv'
for k in keys:
    if k==base: continue
    c=np.corrcoef(loaded[base], loaded[k])[0,1]
    print(f'corr {base} vs {k}: {c:.9f}')
