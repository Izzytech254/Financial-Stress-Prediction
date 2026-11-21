from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from features_cv_v2 import build_extra_v2
from frozen_cv import evaluate_folds, load_folds
from run_ablation_v2 import inner_split
from v6_common import DATA_DIR, MZ_SLOPE, TARGET, composite, init_of, load_data, month_scaled, month_z_score, per_date, probit_eta, risk_factors, snapshot_dates, write_json
from v6_residual import bag, early_stopping_rounds, make_params
from v8_screen import build_variant

SEEDS = (909, 11, 22, 33, 44)
CONFIGS = {
    "resid_factors_lgb": {"num_leaves": 5, "min_data_in_leaf": 200, "learning_rate": 0.02, "lambda_l2": 50.0},
    "resid_factors_mono": {"num_leaves": 5, "min_data_in_leaf": 200, "learning_rate": 0.02, "lambda_l2": 50.0},
    "resid_full_base": {},
    "resid_full_reg": {"num_leaves": 5, "min_data_in_leaf": 500, "learning_rate": 0.01, "lambda_l2": 100.0},
    "resid_full_trend_reg": {"num_leaves": 5, "min_data_in_leaf": 500, "learning_rate": 0.01, "lambda_l2": 100.0},
}


def fit_predict(X, y, init, d, tr, va, cfg, monotone=False):
    params = make_params(cfg, 4)
    if monotone:
        params["monotone_constraints"] = [1] * X.shape[1]
    fit_rel, val_rel = inner_split(y[tr], d[tr], 90000 + int(d[tr].max()))
    fit = tr[fit_rel]
    val = tr[val_rel]
    best = early_stopping_rounds(X[fit], y[fit], init[fit], X[val], y[val], init[val], params, 300, 30)
    raw = bag(X[tr], y[tr], init[tr], X[va], max(1, best), params, SEEDS)
    return per_date(probit_eta(expit(init[va] + raw)), d[va]), best


def main():
    start = time.time()
    train, test, _ = load_data(DATA_DIR)
    d_tr, d_te = snapshot_dates(train, test)
    y = train[TARGET].to_numpy()
    n = len(train)
    all_rows = pd.concat([train.drop(columns=TARGET), test], ignore_index=True)
    d_all = np.r_[d_tr, d_te]
    R = risk_factors(all_rows)
    lin = MZ_SLOPE * month_z_score(R, d_all)
    init_all = init_of(lin, d_all, np.arange(len(all_rows)))[0]
    X_full, _, _, _, _ = build_variant("drop_m5_all", all_rows, d_all, R, None)
    X_trend, _ = build_extra_v2(all_rows, d_all, ["trend", "concentration"])
    X_full_trend = np.c_[X_full, X_trend].astype(np.float32)
    R_cap = np.minimum(R, np.quantile(R, 0.99, axis=0))
    X_factors = month_scaled(R_cap, d_all, "rank").astype(np.float32)
    matrices = {
        "resid_factors_lgb": X_factors,
        "resid_factors_mono": X_factors,
        "resid_full_base": X_full,
        "resid_full_reg": X_full,
        "resid_full_trend_reg": X_full_trend,
    }
    b = np.load("reports/submission_B_predictions.npz")["oof"]
    b_by_date = {}
    for t in range(4):
        m = d_tr == t
        b_by_date[t] = composite(y[m], b[m])[0]
    results = {}
    for t in (1, 2, 3):
        tr = np.where(d_tr < t)[0]
        va = np.where(d_tr == t)[0]
        preds = {"structural_hand": per_date(probit_eta(expit(init_all[va])), d_tr[va])}
        rounds = {"structural_hand": 0}
        for name, X in matrices.items():
            pred, best = fit_predict(X, y, init_all, d_tr, tr, va, CONFIGS[name], monotone=(name == "resid_factors_mono"))
            preds[name] = pred
            rounds[name] = int(best)
        scores = {}
        for name, pred in preds.items():
            score, ll, auc = composite(y[va], pred)
            scores[name] = {"proxy": score, "log_loss": ll, "auc": auc, "delta_vs_struct": score - composite(y[va], preds["structural_hand"])[0], "delta_vs_B_oof": score - b_by_date[t], "rounds": rounds[name]}
        results[str(t)] = scores
        print(f"holdout={t}", flush=True)
        for name, s in sorted(scores.items(), key=lambda kv: kv[1]["proxy"], reverse=True):
            print(f"  {name:24s} proxy={s['proxy']:.9f} ll={s['log_loss']:.9f} auc={s['auc']:.9f} dB={s['delta_vs_B_oof']:+.9f} rounds={s['rounds']}", flush=True)
    summary = {}
    for name in preds:
        vals = [results[str(t)][name]["proxy"] for t in (1, 2, 3)]
        db = [results[str(t)][name]["delta_vs_B_oof"] for t in (1, 2, 3)]
        summary[name] = {"mean_proxy": float(np.mean(vals)), "min_proxy": float(np.min(vals)), "mean_delta_vs_B": float(np.mean(db)), "min_delta_vs_B": float(np.min(db))}
    out = {"results": results, "summary": summary, "elapsed": time.time() - start, "seeds": list(SEEDS), "b_by_date": b_by_date}
    write_json("reports/forward_lab.json", out)
    print("summary", flush=True)
    for name, s in sorted(summary.items(), key=lambda kv: kv[1]["mean_proxy"], reverse=True):
        print(f"  {name:24s} mean={s['mean_proxy']:.9f} min={s['min_proxy']:.9f} mean_dB={s['mean_delta_vs_B']:+.9f} min_dB={s['min_delta_vs_B']:+.9f}", flush=True)


if __name__ == "__main__":
    main()
