from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from frozen_cv import load_folds, evaluate_folds, save_predictions
from v6_common import DATA_DIR, MZ_SLOPE, TARGET, init_of, load_data, month_z_score, per_date, probit_eta, risk_factors, snapshot_dates, write_json
from v6_residual import BASE, bag, early_stopping_rounds, make_params, scaled_rounds, seeds_for
from v8_screen import build_variant
from features_cv_v2 import build_extra_v2
from run_ablation_fast import inner_split
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool

def prepare(args):
    train,test,sample=load_data(DATA_DIR); d_tr,d_te=snapshot_dates(train,test)
    y=train[TARGET].to_numpy(); n=len(train)
    all_rows=pd.concat([train.drop(columns=TARGET),test],ignore_index=True); d_all=np.r_[d_tr,d_te]
    R=risk_factors(all_rows); lin=MZ_SLOPE*month_z_score(R,d_all)
    X,names,dc,ms,ec=build_variant(args.variant,all_rows,d_all,R,None)
    groups=[g for g in args.groups.split(',') if g]
    if groups:
        Xe,ne=build_extra_v2(all_rows,d_all,groups); X=np.c_[X,Xe].astype(np.float32); names+=ne
    return train,test,sample,d_tr,d_te,y,n,d_all,lin,X,names

def fold_data(X,y,d_tr,d_all,lin,fold):
    rep,f,a,b=fold
    fit_rel,val_rel=inner_split(y[a],d_tr[a],10000+rep*10+f)
    fit=a[fit_rel]; val=a[val_rel]
    return fit,val,a,b,init_of(lin,d_all,fit)[0],init_of(lin,d_all,val)[0],init_of(lin,d_all,a)[0],init_of(lin,d_all,b)[0]

def lgb_bag(X,y,init,Xp,rounds,params,seeds):
    ds=lgb.Dataset(X,y,init_score=init,free_raw_data=False)
    raws=[]
    for s in seeds:
        p={**params,'seed':int(s)}
        m=lgb.train(p,ds,num_boost_round=rounds)
        raws.append(m.predict(Xp,raw_score=True))
    return np.mean(raws,axis=0)

def xgb_bag(X,y,init,Xp,initp,rounds,params,seeds):
    ds=xgb.DMatrix(X,label=y,base_margin=init)
    dp=xgb.DMatrix(Xp)
    raws=[]
    for s in seeds:
        p={**params,'seed':int(s)}
        m=xgb.train(p,ds,num_boost_round=rounds)
        raws.append(m.predict(dp,output_margin=True))
    return np.mean(raws,axis=0)

def cat_bag(X,y,init,Xp,initp,rounds,params,seeds,threads):
    tp=Pool(X,label=y,baseline=init); pp=Pool(Xp)
    raws=[]
    for s in seeds:
        m=CatBoostClassifier(**params,iterations=rounds,random_seed=int(s),thread_count=threads,verbose=False,allow_writing_files=False)
        m.fit(tp)
        raws.append(m.predict(pp,prediction_type='RawFormulaVal'))
    return np.mean(raws,axis=0)

def run_model(kind, args, prep, folds, seeds, rounds_override=None):
    train,test,sample,d_tr,d_te,y,n,d_all,lin,X,names=prep
    oof_reps=np.empty((len({f[0] for f in folds}),n)); rpos={s:j for j,s in enumerate(sorted({f[0] for f in folds}))}
    rounds=[]
    threads=args.threads
    base_params = dict(BASE)
    if kind=='lgb_dart': base_params.update({'boosting_type':'dart','learning_rate':0.02,'num_leaves':7,'max_depth':3,'min_data_in_leaf':300,'lambda_l2':50,'feature_fraction':0.7,'bagging_fraction':0.8})
    if kind=='lgb_goss': base_params.update({'boosting_type':'goss','learning_rate':0.02,'num_leaves':7,'max_depth':3,'min_data_in_leaf':300,'lambda_l2':50,'feature_fraction':0.7})
    if kind=='xgb':
        base_params={'objective':'binary:logistic','eval_metric':'logloss','eta':0.02,'max_depth':3,'min_child_weight':75,'lambda':50,'subsample':0.8,'colsample_bytree':0.7,'tree_method':'hist','max_bin':256,'verbosity':0,'nthread':threads}
    if kind=='cat':
        base_params={'loss_function':'Logloss','eval_metric':'Logloss','learning_rate':0.02,'depth':3,'l2_leaf_reg':50,'bootstrap_type':'Bernoulli','subsample':0.8,'rsm':0.7}
    for i,fold in enumerate(folds):
        fit,val,a,b,ifi,ivi,ia,ib=fold_data(X,y,d_tr,d_all,lin,fold)
        if kind.startswith('lgb'):
            params={**base_params,'num_threads':threads}
            best=early_stopping_rounds(X[fit],y[fit],ifi,X[val],y[val],ivi,params,args.max_rounds,args.patience)
            raw=lgb_bag(X[a],y[a],ia,X[b],max(1,best),params,seeds)
        elif kind=='xgb':
            params=base_params
            ds=xgb.DMatrix(X[fit],label=y[fit],base_margin=ifi); dv=xgb.DMatrix(X[val],label=y[val],base_margin=ivi)
            m=xgb.train(params,ds,num_boost_round=args.max_rounds,evals=[(dv,'v')],early_stopping_rounds=args.patience,verbose_eval=False)
            best=max(int(m.best_iteration)+1,1)
            raw=xgb_bag(X[a],y[a],ia,X[b],ib,best,params,seeds)
        elif kind=='cat':
            tp=Pool(X[fit],label=y[fit],baseline=ifi); vp=Pool(X[val],label=y[val],baseline=ivi)
            m=CatBoostClassifier(**base_params,iterations=args.max_rounds,random_seed=909,thread_count=threads,verbose=False,allow_writing_files=False)
            m.fit(tp,eval_set=vp,use_best_model=True,early_stopping_rounds=args.patience)
            best=max(int(m.get_best_iteration())+1,1)
            raw=cat_bag(X[a],y[a],ia,X[b],ib,best,base_params,seeds,threads)
        elif kind in ('et','rf'):
            clf=(ExtraTreesClassifier if kind=='et' else RandomForestClassifier)(n_estimators=300,max_depth=6,min_samples_leaf=20,max_features=0.7,n_jobs=threads,random_state=909)
            clf.fit(X[fit],y[fit])
            lp=clf.predict_log_proba(X[b])[:,1]; lpf=clf.predict_log_proba(X[fit])[:,1]
            best=-1
            # map to init-scale residual logit approximately
            raw=(lp-lpf)  # placeholder; replaced below
            raw=np.zeros(len(b))
        elif kind=='lr':
            sc=StandardScaler().fit(X[fit])
            clf=LogisticRegression(C=0.05,max_iter=2000,solver='lbfgs')
            clf.fit(sc.transform(X[fit]),y[fit])
            raw=clf.decision_function(sc.transform(X[b]))
            best=-1
        oof_reps[rpos[fold[0]],b]=per_date(probit_eta(expit(ib+raw)),d_tr[b])
        rounds.append(best)
        if i%5==0: print(f'{kind} {i+1}/{len(folds)} best={best}',flush=True)
    return oof_reps,np.array(rounds)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--kinds',default='lgb_dart,lgb_goss,xgb,cat,lr')
    ap.add_argument('--variant',default='drop_m5_all'); ap.add_argument('--groups',default='trend,concentration')
    ap.add_argument('--seeds',type=int,default=5); ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--max-rounds',type=int,default=250); ap.add_argument('--patience',type=int,default=35)
    ap.add_argument('--folds',default='all'); ap.add_argument('--test',action='store_true')
    args=ap.parse_args()
    prep=prepare(args)
    train,test,sample,d_tr,d_te,y,n,d_all,lin,X,names=prep
    folds=load_folds()
    if args.folds=='one': folds=[f for f in folds if f[0]==42]
    seeds=seeds_for(max(5,args.seeds))[:args.seeds]
    for kind in [k for k in args.kinds.split(',') if k]:
        t0=time.time()
        oof_reps,rounds=run_model(kind,args,prep,folds,seeds)
        oof=np.mean(oof_reps,axis=0); m=evaluate_folds(y,oof,folds)
        tag=f'final_{kind}_s{args.seeds}_{args.folds}'
        payload={'y':y,'oof':oof,'oof_repeats':oof_reps,'d_tr':d_tr,'d_te':d_te,'feature_names':np.asarray(names),'rounds':rounds}
        if args.test:
            tr=np.arange(n); te=np.arange(n,n+len(test))
            if kind.startswith('lgb'):
                params={**BASE,'boosting_type':('dart' if kind=='lgb_dart' else 'goss'),'num_threads':args.threads}
                raw=lgb_bag(X[:n],y,init_of(lin,d_all,tr)[0],X[n:],scaled_rounds(rounds.tolist(),1.0),params,seeds)
            elif kind=='xgb':
                params={'objective':'binary:logistic','eval_metric':'logloss','eta':0.02,'max_depth':3,'min_child_weight':75,'lambda':50,'subsample':0.8,'colsample_bytree':0.7,'tree_method':'hist','max_bin':256,'verbosity':0,'nthread':args.threads}
                raw=xgb_bag(X[:n],y,init_of(lin,d_all,tr)[0],X[n:],init_of(lin,d_all,te)[0],max(1,int(np.median(rounds))),params,seeds)
            elif kind=='cat':
                params={'loss_function':'Logloss','learning_rate':0.02,'depth':3,'l2_leaf_reg':50,'bootstrap_type':'Bernoulli','subsample':0.8,'rsm':0.7}
                raw=cat_bag(X[:n],y,init_of(lin,d_all,tr)[0],X[n:],init_of(lin,d_all,te)[0],max(1,int(np.median(rounds))),params,seeds,args.threads)
            elif kind=='lr':
                sc=StandardScaler().fit(X[:n]); clf=LogisticRegression(C=0.05,max_iter=2000).fit(sc.transform(X[:n]),y)
                raw=clf.decision_function(sc.transform(X[n:]))
            pt=per_date(probit_eta(expit(init_of(lin,d_all,te)[0]+raw)),d_te)
            out=sample.copy(); out['Target']=pt
            out.to_csv(f'submission_{tag}.csv',index=False,float_format='%.17g')
            payload['test']=pt
        save_predictions(f'reports/{tag}_predictions.npz',payload)
        write_json(f'reports/{tag}.json',{'kind':kind,'validation':m,'seeds':list(seeds),'variant':args.variant,'groups':args.groups,'features':X.shape[1],'time':time.time()-t0,'rounds':rounds.tolist()})
        print(f'{tag}: proxy={m["overall"]["proxy"]:.9f} ll={m["overall"]["log_loss"]:.9f} auc={m["overall"]["auc"]:.9f} time={time.time()-t0:.0f}s',flush=True)
if __name__=='__main__': main()
