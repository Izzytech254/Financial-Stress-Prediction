from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd, optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from scipy.special import expit
from sklearn.metrics import log_loss, roc_auc_score
from frozen_cv import load_folds, evaluate_folds, save_predictions
from v6_common import DATA_DIR, MZ_SLOPE, TARGET, init_of, load_data, month_z_score, per_date, probit_eta, risk_factors, snapshot_dates, write_json
from v6_residual import BASE, bag, early_stopping_rounds, make_params, scaled_rounds, seeds_for
from v8_screen import build_variant
from features_cv_v2 import build_extra_v2
from run_ablation_fast import inner_split

def predict_folds(X,y,d_tr,d_all,lin,folds,params,max_rounds,patience,seeds):
    n=len(y); reps={}; rounds=[]
    for rep, fold, a, b in folds:
        fit_rel,val_rel=inner_split(y[a],d_tr[a],10000+rep*10+fold)
        fit=a[fit_rel]; val=a[val_rel]
        init_fit=init_of(lin,d_all,fit)[0]; init_val=init_of(lin,d_all,val)[0]
        init_a=init_of(lin,d_all,a)[0]; init_b=init_of(lin,d_all,b)[0]
        best=early_stopping_rounds(X[fit],y[fit],init_fit,X[val],y[val],init_val,params,max_rounds,patience)
        rounds.append(best)
        raw=bag(X[a],y[a],init_a,X[b],max(1,best),params,seeds)
        reps.setdefault(rep,np.empty(n)); reps[rep][b]=per_date(probit_eta(expit(init_b+raw)),d_tr[b])
    return reps, rounds

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--variant',default='drop_m5_all')
    ap.add_argument('--groups',default='trend,concentration')
    ap.add_argument('--trials',type=int,default=100)
    ap.add_argument('--search-seeds',type=int,default=3)
    ap.add_argument('--final-seeds',type=int,default=10)
    ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--max-rounds',type=int,default=250)
    ap.add_argument('--patience',type=int,default=35)
    ap.add_argument('--tag',default='final_tuned_lgb')
    args=ap.parse_args()
    start=time.time()
    folds=load_folds()
    train,test,sample=load_data(DATA_DIR); d_tr,d_te=snapshot_dates(train,test)
    y=train[TARGET].to_numpy(); n=len(train)
    all_rows=pd.concat([train.drop(columns=TARGET),test],ignore_index=True); d_all=np.r_[d_tr,d_te]
    R=risk_factors(all_rows); lin=MZ_SLOPE*month_z_score(R,d_all)
    X,names,dc,ms,ec=build_variant(args.variant,all_rows,d_all,R,None)
    groups=[g for g in args.groups.split(',') if g]
    if groups:
        Xe,ne=build_extra_v2(all_rows,d_all,groups); X=np.c_[X,Xe].astype(np.float32); names+=ne
    sf=[f for f in folds if f[0]==42]
    search_seeds=seeds_for(max(5,args.search_seeds))[:args.search_seeds]
    def objective(trial):
        params={**BASE,
            'learning_rate':trial.suggest_float('learning_rate',0.005,0.04,log=True),
            'num_leaves':trial.suggest_int('num_leaves',5,31),
            'max_depth':trial.suggest_int('max_depth',2,7),
            'min_data_in_leaf':trial.suggest_int('min_data_in_leaf',50,500),
            'lambda_l2':trial.suggest_float('lambda_l2',1.0,100.0,log=True),
            'feature_fraction':trial.suggest_float('feature_fraction',0.5,1.0),
            'bagging_fraction':trial.suggest_float('bagging_fraction',0.6,1.0),
            'bagging_freq':1,'num_threads':args.threads}
        reps,_=predict_folds(X,y,d_tr,d_all,lin,sf,params,args.max_rounds,args.patience,search_seeds)
        p=next(iter(reps.values()))
        ll=log_loss(y,p); auc=roc_auc_score(y,p)
        return 0.6*(1-ll/0.595)+0.4*auc
    study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=20261119))
    def cb(study,trial):
        if (trial.number+1)%10==0:
            print(f'trial={trial.number+1}/{args.trials} best={study.best_value:.9f} elapsed={time.time()-start:.0f}s',flush=True)
    study.optimize(objective,n_trials=args.trials,callbacks=[cb])
    best={**BASE,**study.best_params,'num_threads':args.threads}
    print('best',json.dumps(study.best_params,sort_keys=True),flush=True)
    seeds=seeds_for(max(5,args.final_seeds))[:args.final_seeds]
    reps,rounds=predict_folds(X,y,d_tr,d_all,lin,folds,best,args.max_rounds,args.patience,seeds)
    rep_arrays=[reps[k] for k in sorted(reps)]
    oof=np.mean(rep_arrays,axis=0)
    m=evaluate_folds(y,oof,folds)
    # test
    tr=np.arange(n); te=np.arange(n,n+len(test))
    init_tr=init_of(lin,d_all,tr)[0]; init_te=init_of(lin,d_all,te)[0]
    final_rounds=scaled_rounds(rounds,1.0)
    raw=bag(X[:n],y,init_tr,X[n:],final_rounds,best,seeds)
    p_test=per_date(probit_eta(expit(init_te+raw)),d_te)
    out=sample.copy(); out['Target']=p_test
    out.to_csv(f'submission_{args.tag}.csv',index=False,float_format='%.17g')
    save_predictions(f'reports/{args.tag}_predictions.npz', {'y':y,'oof':oof,'oof_repeats':np.asarray(rep_arrays),'d_tr':d_tr,'d_te':d_te,'test':p_test,'feature_names':np.asarray(names),'s_mz':lin})
    write_json(f'reports/{args.tag}.json',{'tag':args.tag,'trials':args.trials,'best_params':study.best_params,'best_search':study.best_value,'validation':m,'seeds':list(seeds),'groups':groups,'variant':args.variant,'final_rounds':int(final_rounds),'features':X.shape[1],'elapsed':time.time()-start})
    print(f'FINAL proxy={m["overall"]["proxy"]:.9f} ll={m["overall"]["log_loss"]:.9f} auc={m["overall"]["auc"]:.9f} elapsed={time.time()-start:.0f}s',flush=True)
if __name__=='__main__': main()
