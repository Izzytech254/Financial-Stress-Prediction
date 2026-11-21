from __future__ import annotations
import numpy as np, pandas as pd, json
from pathlib import Path
from scipy.optimize import minimize_scalar
from scipy.special import ndtr, log_ndtr, ndtri
from sklearn.metrics import log_loss, roc_auc_score
from v6_common import DATA_DIR, TARGET, EPS, load_data, snapshot_dates, risk_factors, month_z_score, solve_intercept, composite
from frozen_cv import load_folds

def p_from_sigma(S, d, sigma):
    p=np.empty(len(S))
    for k in np.unique(d):
        m=d==k
        if m.sum()==0: continue
        c=solve_intercept(S[m]/sigma,0.15)
        p[m]=ndtr(c+S[m]/sigma)
    return np.clip(p,EPS,1-EPS)

def fit_sigma(S,y,d,fit_idx,bounds=(0.1,1.0)):
    def nll(sig):
        p=p_from_sigma(S[fit_idx]/sig,d[fit_idx],1.0)  # sigma scaling handled externally
        # careful: p_from_sigma divides again; use direct
        p=np.empty(len(fit_idx)); dd=d[fit_idx]; ss=S[fit_idx]/sig
        for k in np.unique(dd):
            m=dd==k
            c=solve_intercept(ss[m],0.15)
            p[m]=ndtr(c+ss[m])
        return log_loss(y[fit_idx],np.clip(p,EPS,1-EPS))
    r=minimize_scalar(nll,bounds=bounds,method='bounded',options={'xatol':1e-5})
    return r.x

def main():
    train,test,_=load_data(DATA_DIR)
    d_tr,d_te=snapshot_dates(train,test); y=train[TARGET].to_numpy(); n=len(train)
    all_rows=pd.concat([train.drop(columns=TARGET),test],ignore_index=True); d_all=np.r_[d_tr,d_te]
    R=risk_factors(all_rows); S=month_z_score(R,d_all)[:n]
    folds=load_folds()
    print('sigma fixed grid (full OOF, prior-cal per snapshot):')
    for sig in [0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.6]:
        p=p_from_sigma(S,d_tr,sig)
        print(f'  sigma={sig:.2f} proxy={composite(y,p)[0]:.9f} ll={composite(y,p)[1]:.9f} auc={composite(y,p)[2]:.9f}')
    print('fold-fitted sigma:')
    oof=np.empty(n)
    sigmas=[]
    for rep,f,tr,va in folds:
        sig=fit_sigma(S,y,d_tr,tr)
        sigmas.append(sig)
        oof[va]=p_from_sigma(S[va]/sig,d_tr[va],1.0)
    print(f'  mean sigma={np.mean(sigmas):.5f} std={np.std(sigmas):.5f} min={np.min(sigmas):.4f} max={np.max(sigmas):.4f}')
    print('  OOF', composite(y,oof))
    np.savez_compressed('reports/sigma_search_predictions.npz',y=y,oof=oof,d_tr=d_tr,d_te=d_te)
    json.dump({'sigmas':list(map(float,sigmas))},open('reports/sigma_search.json','w'),indent=2)
if __name__=='__main__': main()
