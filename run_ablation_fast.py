from __future__ import annotations
import argparse, time, json
import numpy as np, pandas as pd
from pathlib import Path
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold

from features_cv_v2 import build_extra_v2
from frozen_cv import evaluate_folds, load_folds, paired_delta
from v6_common import DATA_DIR, MZ_SLOPE, TARGET, init_of, load_data, month_z_score, per_date, probit_eta, risk_factors, snapshot_dates, write_json
from v6_residual import bag, early_stopping_rounds, make_params, scaled_rounds, seeds_for
from v8_screen import build_variant

def inner_split(y, d, seed):
    strata = y*4 + d
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
    fit, val = next(skf.split(np.arange(len(y)), strata))
    return fit, val

def predict_folds(X, y, d_tr, d_all, lin, folds, params, seeds, max_rounds=250, patience=30, configs=2):
    repeat_ids=sorted({f[0] for f in folds}); rep_pos={s:j for j,s in enumerate(repeat_ids)}
    oof_reps=np.empty((len(repeat_ids),len(y)),dtype=np.float64)
    rounds=np.zeros((configs,len(folds)),dtype=np.int32)
    for i,(rep,fold,a,b) in enumerate(folds):
        fit_rel,val_rel=inner_split(y[a],d_tr[a],10000+rep*10+fold)
        fit=a[fit_rel]; val=a[val_rel]
        init_fit=init_of(lin,d_all,fit)[0]; init_val=init_of(lin,d_all,val)[0]
        init_a=init_of(lin,d_all,a)[0]; init_b=init_of(lin,d_all,b)[0]
        raws=[]
        for c in range(configs):
            cfg = [{} , {"num_leaves":15,"max_depth":4}][c]
            p=make_params(cfg,8)
            best=early_stopping_rounds(X[fit],y[fit],init_fit,X[val],y[val],init_val,p,max_rounds,patience)
            rounds[c,i]=best
            raws.append(bag(X[a],y[a],init_a,X[b],max(1,best),p,seeds))
        oof_reps[rep_pos[rep],b]=per_date(probit_eta(expit(init_b+np.mean(raws,axis=0))),d_tr[b])
    return oof_reps, rounds

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--groups', default='trend,concentration')
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--tag', default='fast')
    ap.add_argument('--variant', default='drop_m5_all')
    ap.add_argument('--folds', default='all')
    args=ap.parse_args()

    start=time.time()
    folds=load_folds()
    if args.folds=='one':
        folds=[f for f in folds if f[0]==42]
    train,test,_=load_data(DATA_DIR)
    d_tr,d_te=snapshot_dates(train,test); y=train[TARGET].to_numpy(); n=len(train)
    all_rows=pd.concat([train.drop(columns=TARGET),test],ignore_index=True); d_all=np.r_[d_tr,d_te]
    R=risk_factors(all_rows); lin=MZ_SLOPE*month_z_score(R,d_all)
    X,names,drop_cols,month_scale,extra_cols=build_variant(args.variant,all_rows,d_all,R,None)
    groups=[g for g in args.groups.split(',') if g]
    if groups:
        Xe,ne=build_extra_v2(all_rows,d_all,groups)
        X=np.c_[X,Xe].astype(np.float32); names=names+ne
    all_seeds=seeds_for(max(5,args.seeds)); seeds=all_seeds[:args.seeds]
    oof_reps,rounds=predict_folds(X,y,d_tr,d_all,lin,folds,None,seeds)
    oof=np.mean(oof_reps,axis=0)
    m=evaluate_folds(y,oof,folds)
    print(f'{args.tag} groups={args.groups} feats={X.shape[1]} proxy={m["overall"]["proxy"]:.9f} ll={m["overall"]["log_loss"]:.9f} auc={m["overall"]["auc"]:.9f} time={time.time()-start:.1f}s',flush=True)
    Path('reports').mkdir(exist_ok=True)
    np.savez_compressed(f'reports/{args.tag}_predictions.npz', y=y, oof=oof, oof_repeats=oof_reps, d_tr=d_tr, d_te=d_te, feature_names=np.asarray(names), rounds=rounds)
    write_json(f'reports/{args.tag}.json', {'groups':groups,'variant':args.variant,'features':X.shape[1],'seeds':list(seeds),'validation':m,'time':time.time()-start})

if __name__=='__main__': main()
