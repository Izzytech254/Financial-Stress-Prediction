from __future__ import annotations
import numpy as np, pandas as pd, json, time
from pathlib import Path
from scipy.optimize import minimize
from scipy.special import ndtr, log_ndtr, ndtri
from sklearn.metrics import log_loss, roc_auc_score

from v6_common import DATA_DIR, TARGET, load_data, snapshot_dates, risk_factors, EPS, composite, solve_intercept
from frozen_cv import load_folds


def proxy(y,p):
    return composite(y,p)[0]


def fit_probit(Z, y, w0=None, ridge=1e-4, maxiter=2000):
    """Probit with per-date intercepts. Z: (n,p) already normalized factors."""
    dates = np.unique(Z[:,0])
    # Z includes date columns as first block? We'll pass design matrix X."""
    X = Z
    p = X.shape[1]
    def nll(w):
        eta = X @ w
        lp = log_ndtr(eta); lq = log_ndtr(-eta)
        phi = np.exp(-0.5*eta**2 - 0.5*np.log(2*np.pi))
        grad_eta = np.where(y==1, -phi/np.exp(lp), phi/np.exp(lq))
        val = -(y*lp + (1-y)*lq).sum() + 0.5*ridge*np.sum(w[3:]**2) if p>3 else 0
        g = X.T @ grad_eta
        if p>3: g[3:] += ridge*w[3:]
        return val, g
    init = np.zeros(p)
    init[:3] = -1.56
    init[3:] = 0.0
    res = minimize(nll, init, jac=True, method='L-BFGS-B', options={'ftol':1e-12,'gtol':1e-8,'maxiter':maxiter,'maxls':50})
    return res.x


def design(Z, d, n_dates=4):
    D = (d[:,None] == np.arange(n_dates)[None,:]).astype(float)
    return np.c_[D, Z]


def prior_cal(S, d, n_dates=4):
    p = np.empty(len(S))
    for k in range(n_dates):
        m = d==k
        if m.sum()==0: continue
        p[m] = ndtr(solve_intercept(S[m],0.15)+S[m])
    return np.clip(p, EPS, 1-EPS)


def oof_probit(X, y, d, folds, ridge=1e-4, scale=None):
    """X: design matrix with date dummies first 4 cols then features."""
    z = X[:,4:]
    oof = np.empty(len(y))
    for rep, fold, tr, va in folds:
        w = fit_probit(X[tr], y[tr], ridge=ridge)
        if scale == 'fit':
            w[4:] *= 1/0.35
        oof[va] = prior_cal(X[va] @ w, d[va])
    return oof


def main():
    train, test, _ = load_data(DATA_DIR)
    d_tr,d_te = snapshot_dates(train,test)
    y=train[TARGET].to_numpy(); n=len(train)
    all_rows=pd.concat([train.drop(columns=TARGET), test], ignore_index=True)
    d_all=np.r_[d_tr,d_te]
    R=risk_factors(all_rows)
    folds=load_folds()

    results={}

    # Baseline hand-weight structural
    from v6_common import month_z_score, MZ_SLOPE, structural_oof_mz
    s_mz=month_z_score(R,d_all)
    p_hand=structural_oof_mz(s_mz,d_tr)
    results['hand'] = composite(y,p_hand)
    print('hand', results['hand'])

    # scale variants: cap quantiles and z-scope
    for capq in [0.95,0.97,0.99,0.995,1.0]:
        for zscope in ['month','global']:
            capped=np.minimum(R, np.quantile(R,capq,axis=0))
            if zscope=='month':
                Z=np.empty_like(capped)
                for k in np.unique(d_all):
                    m=d_all==k
                    Z[m]=(capped[m]-capped[m].mean(0))/capped[m].std(0)
            else:
                Z=(capped-capped.mean(0))/capped.std(0)
            dw = np.array([5,3,3,2,4,2,1])/20
            S=Z@dw
            p=prior_cal(S[:n]/MZ_SLOPE,d_tr)
            name=f'hand_cap{capq}_{zscope}'
            results[name]=composite(y,p)

    # fitted global weights probit (train-fit full sample; for OOF need fold fit)
    capped=np.minimum(R,np.quantile(R,0.99,axis=0))
    Z=np.empty_like(capped)
    for k in np.unique(d_all):
        m=d_all==k
        Z[m]=(capped[m]-capped[m].mean(0))/capped[m].std(0)
    X=design(Z[:n],d_tr)
    for ridge in [1e-6, 1e-4, 1e-2, 0.1, 1.0]:
        p=oof_probit(X,y,d_tr,folds,ridge=ridge)
        name=f'fitprobit_r{ridge}'
        results[name]=composite(y,p)
        print(name, results[name], 'w=', fit_probit(X,y,ridge=ridge)[4:])

    # rank transform factors
    Rr=np.empty_like(R)
    for j in range(R.shape[1]):
        Rr[:,j]=pd.Series(R[:,j]).rank(pct=True).to_numpy()
    Zrank=np.empty_like(Rr)
    for k in np.unique(d_all):
        m=d_all==k
        Zrank[m]=(Rr[m]-Rr[m].mean(0))/Rr[m].std(0)
    Xr=design(Zrank[:n],d_tr)
    for ridge in [1e-4,1e-2,0.1]:
        p=oof_probit(Xr,y,d_tr,folds,ridge=ridge)
        results[f'fitprobit_rank_r{ridge}']=composite(y,p)

    # ensemble hand + fitted
    phand=p_hand
    for ridge in [1e-4,1e-2,0.1,1.0]:
        pf=oof_probit(X,y,d_tr,folds,ridge=ridge)
        for a in [0.25,0.5,0.75]:
            p=a*phand+(1-a)*pf
            results[f'ens_hand_fit_r{ridge}_a{a}']=composite(y,p)

    # print sorted
    df=pd.DataFrame([{'name':k,'proxy':v[0],'ll':v[1],'auc':v[2]} for k,v in results.items()]).sort_values('proxy',ascending=False)
    print(df.to_string(index=False))
    Path('reports').mkdir(exist_ok=True)
    df.to_csv('reports/probit_search.csv',index=False)
    json.dump({k:list(map(float,v)) for k,v in results.items()}, open('reports/probit_search.json','w'), indent=2)

if __name__=='__main__': main()
