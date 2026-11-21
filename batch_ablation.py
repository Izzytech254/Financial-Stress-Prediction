from __future__ import annotations
import time, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.special import expit
from features_cv_v2 import build_extra_v2
from frozen_cv import load_folds, paired_delta, evaluate_folds
from v6_common import DATA_DIR, MZ_SLOPE, TARGET, init_of, load_data, month_z_score, per_date, probit_eta, risk_factors, snapshot_dates, write_json
from v6_residual import bag, early_stopping_rounds, make_params, scaled_rounds
from v8_screen import build_variant
from run_ablation_fast import inner_split

GROUPS = {
 'base': [],
 'tc': ['trend','concentration'],
 'tc_cr': ['trend','concentration','cross_ratio'],
 'tc_vol': ['trend','concentration','volatility'],
 'tc_ms': ['trend','concentration','month_shape'],
 'tc_pn': ['trend','concentration','profile_num'],
 'tc_cr_vol': ['trend','concentration','cross_ratio','volatility'],
 'tc_all': ['trend','concentration','cross_ratio','volatility','month_shape','profile_num'],
}
CONFIGS=[{}, {'num_leaves':15,'max_depth':4}]
SEEDS=(909,11,22,33,44)

def pred_one(X,y,d_tr,d_all,lin,fold):
    rep,f,tr,va=fold
    fit_rel,val_rel=inner_split(y[tr],d_tr[tr],10000+rep*10+f)
    fit=tr[fit_rel]; val=tr[val_rel]
    init_fit=init_of(lin,d_all,fit)[0]; init_val=init_of(lin,d_all,val)[0]
    init_a=init_of(lin,d_all,tr)[0]; init_b=init_of(lin,d_all,va)[0]
    raws=[]
    for cfg in CONFIGS:
        p=make_params(cfg,8)
        best=early_stopping_rounds(X[fit],y[fit],init_fit,X[val],y[val],init_val,p,250,30)
        raws.append(bag(X[tr],y[tr],init_a,X[va],max(1,best),p,SEEDS))
    return per_date(probit_eta(expit(init_b+np.mean(raws,axis=0))),d_tr[va]), [best]

def main():
    t0=time.time()
    folds=[f for f in load_folds() if f[0]==42]
    train,test,_=load_data(DATA_DIR)
    d_tr,d_te=snapshot_dates(train,test); y=train[TARGET].to_numpy(); n=len(train)
    all_rows=pd.concat([train.drop(columns=TARGET),test],ignore_index=True); d_all=np.r_[d_tr,d_te]
    R=risk_factors(all_rows); lin=MZ_SLOPE*month_z_score(R,d_all)
    Xbase,names0,drop_cols,month_scale,extra=build_variant('drop_m5_all',all_rows,d_all,R,None)
    oofs={}
    for tag,groups in GROUPS.items():
        t=time.time()
        X=Xbase.copy(); names=list(names0)
        if groups:
            Xe,ne=build_extra_v2(all_rows,d_all,groups); X=np.c_[X,Xe].astype(np.float32); names+=ne
        oof=np.empty(n,dtype=np.float64)
        rounds=[]
        for fold in folds:
            oof[fold[3]],r=pred_one(X,y,d_tr,d_all,lin,fold); rounds.append(r[0])
        m=evaluate_folds(y,oof,folds)
        oofs[tag]=oof
        print(f'{tag} feats={X.shape[1]} proxy={m["overall"]["proxy"]:.9f} ll={m["overall"]["log_loss"]:.9f} auc={m["overall"]["auc"]:.9f} time={time.time()-t:.1f}s',flush=True)
        np.savez_compressed(f'reports/batch_ab_{tag}_s5_one_repeat.npz', y=y, oof=oof, d_tr=d_tr, d_te=d_te, feature_names=np.asarray(names))
    base=oofs['tc']
    rows=[]
    for tag in GROUPS:
        if tag=='tc': continue
        d=paired_delta(base,oofs[tag],y,folds)
        rows.append({'tag':tag,'delta_vs_tc':d['delta'],'se':d['se'],'accept':d['accept']})
    df=pd.DataFrame(rows)
    print(df.to_string(index=False),flush=True)
    Path('reports').mkdir(exist_ok=True)
    df.to_csv('reports/batch_ablation_s5_one_repeat.csv',index=False)

if __name__=='__main__': main()
