from __future__ import annotations
import numpy as np, pandas as pd, json, time
from pathlib import Path
from scipy.optimize import minimize
from scipy.special import ndtr
from sklearn.metrics import roc_auc_score, log_loss
from v6_common import DATA_DIR, TARGET, EPS, load_data, snapshot_dates, risk_factors, month_z_score, solve_intercept, composite
from frozen_cv import load_folds, paired_delta

def p_cal(S,d,slope=1.0):
    p=np.empty(len(S))
    for k in np.unique(d):
        m=d==k
        p[m]=ndtr(solve_intercept(S[m],0.15)+S[m])
    return np.clip(p,EPS,1-EPS)

def main():
    train,test,_=load_data(DATA_DIR); d_tr,d_te=snapshot_dates(train,test)
    y=train[TARGET].to_numpy(); n=len(train)
    all_rows=pd.concat([train.drop(columns=TARGET),test],ignore_index=True); d_all=np.r_[d_tr,d_te]
    R=risk_factors(all_rows)
    capped=np.minimum(R,np.quantile(R,0.99,axis=0))
    Z=np.empty_like(capped)
    for k in np.unique(d_all):
        m=d_all==k
        Z[m]=(capped[m]-capped[m].mean(0))/capped[m].std(0)
    Ztr=Z[:n]
    hand=np.array([5,3,3,2,4,2,1])/20
    folds=load_folds()
    # differentiable pairwise logistic surrogate: sample pairs within date with y1=1,y0=0
    rng=np.random.default_rng(42)
    pairs=[]
    for k in range(4):
        idx=np.flatnonzero(d_tr==k); pos=idx[y[idx]==1]; neg=idx[y[idx]==0]
        a=rng.choice(pos,3000,replace=True); b=rng.choice(neg,3000,replace=True)
        pairs.append((a,b))
    A=np.concatenate([p[0] for p in pairs]); B=np.concatenate([p[1] for p in pairs])
    D=Ztr[A]-Ztr[B]
    def obj(w):
        z=D@w
        # softplus(-z)
        return np.logaddexp(0,-z).mean()
    res=minimize(obj,hand,method='L-BFGS-B',bounds=[(0,1)]*7,options={'maxiter':5000,'ftol':1e-12})
    w=res.x
    print('hand',hand,'learned',w)
    oof_h=np.empty(n); oof_w=np.empty(n)
    for rep,f,tr,va in folds:
        S_h=Ztr@hand; S_w=Ztr@w
        oof_h[va]=p_cal(S_h[va]/0.35,d_tr[va]); oof_w[va]=p_cal(S_w[va]/0.35,d_tr[va])
    for name,p in [('hand',oof_h),('aucw',oof_w)]:
        print(name, composite(y,p))
    print('paired',paired_delta(oof_h,oof_w,y,folds))
    json.dump({'hand':hand.tolist(),'learned':w.tolist()},open('reports/auc_weight_search.json','w'),indent=2)
    np.savez_compressed('reports/auc_weight_predictions.npz',y=y,oof_h=oof_h,oof_w=oof_w,d_tr=d_tr)
if __name__=='__main__': main()
