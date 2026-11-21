from __future__ import annotations
import argparse, json, itertools
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, roc_auc_score
from frozen_cv import load_folds, evaluate_folds, paired_delta
from v6_common import DATA_DIR, EPS, load_data, write_json

def proxy(y,p):
    p=np.clip(p,EPS,1-EPS)
    ll=log_loss(y,p); auc=roc_auc_score(y,p)
    return 0.6*(1-ll/0.595)+0.4*auc, ll, auc

def load_pred(path):
    z=np.load(path)
    out={'oof':np.clip(z['oof'],EPS,1-EPS),'y':z['y'],'d_tr':z['d_tr'],'d_te':z['d_te']}
    out['repeats']=z['oof_repeats'] if 'oof_repeats' in z.files else np.repeat(out['oof'][None,:],3,axis=0)
    out['test']=np.clip(z['test'],EPS,1-EPS) if 'test' in z.files else None
    return out

def date_rank(p,d):
    out=np.empty(len(p),dtype=np.float64)
    for k in np.unique(d):
        m=d==k
        order=np.flatnonzero(m)[np.argsort(p[m],kind='mergesort')]
        out[order]=np.arange(m.sum(),dtype=np.float64)/max(1,m.sum()-1)
    return np.clip(out,EPS,1-EPS)

def blend_prob(preds, w, kind, d=None):
    P=np.column_stack([np.clip(p,EPS,1-EPS) for p in preds])
    if kind=='prob': return np.clip(P@w,EPS,1-EPS)
    if kind=='logit': return np.clip(expit(logit(P)@w),EPS,1-EPS)
    if kind=='rank':
        if d is None: raise ValueError('rank needs d')
        R=np.column_stack([date_rank(p,d) for p in preds])
        return np.clip(R@w,EPS,1-EPS)
    raise ValueError(kind)

def opt_weights(y,preds,kind='logit',d=None):
    n=len(preds)
    if kind=='rank':
        Z=np.column_stack([date_rank(p,d) for p in preds])
        def f(w): return -proxy(y,np.clip(Z@w,EPS,1-EPS))[0]
    else:
        Z=np.column_stack([logit(np.clip(p,EPS,1-EPS)) for p in preds])
        def f(w): return -proxy(y,np.clip(expit(Z@w),EPS,1-EPS))[0]
    res=minimize(f,np.full(n,1/n),method='SLSQP',bounds=[(0,1)]*n,constraints=[{'type':'eq','fun':lambda w:w.sum()-1}],options={'maxiter':300,'ftol':1e-12})
    w=np.clip(res.x,0,1); return w/w.sum()

def fit_cal(y,p,kind):
    if kind=='temp':
        z=np.clip(logit(np.clip(p,EPS,1-EPS)),-12,12)
        def f(t): return log_loss(y,np.clip(expit(t*z),EPS,1-EPS))
        from scipy.optimize import minimize_scalar
        return float(minimize_scalar(f,bounds=(0.5,1.5),method='bounded').x)
    if kind=='platt':
        m=LogisticRegression(C=1e6,max_iter=1000).fit(np.clip(logit(np.clip(p,EPS,1-EPS)),-12,12)[:,None],y)
        return m
    if kind=='iso':
        m=IsotonicRegression(out_of_bounds='clip',y_min=EPS,y_max=1-EPS).fit(p,y)
        return m

def apply_cal(m,p,kind):
    if kind=='temp':
        return np.clip(expit(m*np.clip(logit(np.clip(p,EPS,1-EPS)),-12,12)),EPS,1-EPS)
    if kind=='platt':
        return m.predict_proba(np.clip(logit(np.clip(p,EPS,1-EPS)),-12,12)[:,None])[:,1]
    return np.clip(m.predict(p),EPS,1-EPS)

def nested_cal(y,repeats,kind):
    out=np.empty_like(repeats)
    for j in range(len(repeats)):
        idx=[i for i in range(len(repeats)) if i!=j]
        m=fit_cal(np.concatenate([y]*len(idx)),np.concatenate([repeats[i] for i in idx]),kind)
        out[j]=apply_cal(m,repeats[j],kind)
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--preds',action='append',required=True)
    ap.add_argument('--names',required=True)
    ap.add_argument('--out-prefix',default='submission_final')
    args=ap.parse_args()
    names=args.names.split(',')
    assert len(names)==len(args.preds)
    models=[load_pred(p) for p in args.preds]
    y=models[0]['y']
    for m in models[1:]: assert np.array_equal(m['y'],y)
    folds=load_folds()
    results={}; oof_cands={}; test_cands={}
    for kind in ('prob','logit','rank'):
        w=opt_weights(y,[m['oof'] for m in models],kind,models[0]['d_tr'])
        P=[m['oof'] for m in models]
        T=[m['test'] for m in models]
        o=blend_prob(P,w,kind,models[0]['d_tr']); t=blend_prob(T,w,kind,models[0]['d_te'])
        reps=np.mean([blend_prob([m['repeats'][j] for m in models],w,kind,models[0]['d_tr']) for j in range(3)],axis=0)
        oof_cands[kind]=o; test_cands[kind]=t
        results[kind]={'weights':w.tolist(),'proxy':proxy(y,o)[0],'ll':proxy(y,o)[1],'auc':proxy(y,o)[2],'repeat_oof':reps}
    best=max(results,key=lambda k:results[k]['proxy'])
    print('best blend',best, {k:results[k]['proxy'] for k in results})
    chosen=oof_cands[best]; chosen_test=test_cands[best]
    chosen_reps=np.asarray([blend_prob([m['repeats'][j] for m in models],opt_weights(y,[m['oof'] for m in models],best,models[0]['d_tr']),best,models[0]['d_tr']) for j in range(3)])
    cal_results={}; cal_oofs={}
    for kind in ('temp','platt','iso'):
        cal_oof=nested_cal(y,chosen_reps,kind).mean(axis=0); cal_oofs[kind]=cal_oof
        d=paired_delta(chosen_reps.mean(axis=0),cal_oof,y,folds)
        cal_results[kind]={'nested_proxy':proxy(y,cal_oof)[0],'nested_ll':proxy(y,cal_oof)[1],'delta':d}
        print('cal',kind,cal_results[kind])
    stack_oof=np.empty(len(y))
    for j in range(3):
        idx=[i for i in range(3) if i!=j]
        Xtr=np.column_stack([np.concatenate([logit(np.clip(m['repeats'][r],EPS,1-EPS)) for r in idx]) for m in models])
        ytr=np.concatenate([y]*2)
        st=LogisticRegression(C=0.1,max_iter=2000).fit(Xtr,ytr)
        Xv=np.column_stack([logit(np.clip(m['repeats'][j],EPS,1-EPS)) for m in models])
        stack_oof=st.predict_proba(Xv)[:,1]
    stack_full=LogisticRegression(C=0.1,max_iter=2000).fit(np.column_stack([logit(np.clip(m['oof'],EPS,1-EPS)) for m in models]),y)
    stack_test=stack_full.predict_proba(np.column_stack([logit(np.clip(m['test'],EPS,1-EPS)) for m in models]))[:,1]
    print('stacker proxy',proxy(y,stack_oof), 'delta',paired_delta(chosen_reps.mean(axis=0),stack_oof,y,folds))
    final={best:(chosen,chosen_test)}
    for kind in ('temp','platt','iso'):
        if cal_results[kind]['delta']['accept']:
            m=fit_cal(y,chosen_reps.mean(axis=0),kind)
            final[f'{best}+{kind}']=(cal_oofs[kind],apply_cal(m,chosen_test,kind))
    final['stacker']=(stack_oof,stack_test)
    _,sample=load_data(DATA_DIR)[1:]
    for name,(o,t) in final.items():
        t=np.clip(t,1e-5,1-1e-5)
        out=sample.copy(); out['Target']=t
        fn=f'{args.out_prefix}_{name}.csv'
        out.to_csv(fn,index=False,float_format='%.17g')
        print(fn,'proxy',proxy(y,o),flush=True)
    report={'names':names,'blends':{k:{kk:vv for kk,vv in v.items() if kk!='repeat_oof'} for k,v in results.items()},'calibration':cal_results,'stacker':proxy(y,stack_oof),'folds':len(folds)}
    write_json('reports/ensemble_final.json',report)

if __name__=='__main__': main()
