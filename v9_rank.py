from __future__ import annotations

import argparse
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from v6_common import (
    DATA_DIR,
    REPORT_DIR,
    TARGET,
    calibrated,
    composite,
    fit_slope,
    load_data,
    risk_factors,
    snapshot_dates,
    validate_submission,
    write_json,
)
from v6_residual import scaled_rounds, seeds_for
from v8_screen import build_variant


def make_dataset(X, y, d):
    _, counts = np.unique(d, return_counts=True)
    return lgb.Dataset(
        X,
        label=y.astype(np.int32),
        group=counts,
        free_raw_data=False,
    )


def train_early(X_fit, y_fit, d_fit, X_val, y_val, d_val, params, max_rounds, patience):
    ds = make_dataset(X_fit, y_fit, d_fit)
    dv = make_dataset(X_val, y_val, d_val)
    booster = lgb.train(
        params,
        ds,
        num_boost_round=max_rounds,
        valid_sets=[dv],
        callbacks=[lgb.early_stopping(patience, verbose=False)],
    )
    return max(int(booster.best_iteration), 1)


def bag_rank(X_fit, y_fit, d_fit, X_pred, X_cal, rounds, params, seeds):
    ds = make_dataset(X_fit, y_fit, d_fit)
    raws_pred = []
    raws_cal = []
    for seed in seeds:
        p = {**params, "seed": int(seed)}
        booster = lgb.train(p, ds, num_boost_round=rounds)
        raws_pred.append(booster.predict(X_pred, raw_score=True))
        raws_cal.append(booster.predict(X_cal, raw_score=True))
    return np.mean(raws_pred, axis=0), np.mean(raws_cal, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--variant", default="drop_m5_all")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--max-rounds", type=int, default=400)
    ap.add_argument("--patience", type=int, default=60)
    ap.add_argument("--tag", default="rank_drop_m5_all")
    ap.add_argument("--report", default=None)
    ap.add_argument("--preds", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()

    start = time.time()
    seeds = seeds_for(args.seeds)
    train, test, sample = load_data(args.data_dir)
    d_tr, d_te = snapshot_dates(train, test)
    y = train[TARGET].to_numpy()
    n = len(train)
    all_rows = pd.concat([train.drop(columns=TARGET), test], ignore_index=True)
    d_all = np.r_[d_tr, d_te]

    R = risk_factors(all_rows)
    X, names, drop_cols, month_scale, extra_cols = build_variant(
        args.variant, all_rows, d_all, R, None
    )

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [1500],
        "learning_rate": 0.02,
        "num_leaves": 7,
        "max_depth": 3,
        "min_data_in_leaf": 300,
        "lambda_l2": 50.0,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "label_gain": [0, 1],
        "verbosity": -1,
        "num_threads": args.threads,
        "seed": 909,
    }

    tag = args.tag
    report_path = Path(args.report or REPORT_DIR / f"v9_{tag}.json")
    preds_path = Path(args.preds) if args.preds else None
    out_path = Path(args.out) if args.out else None

    print(
        f"rank variant={args.variant} tag={tag} features={X.shape[1]} "
        f"seeds={len(seeds)} scale={month_scale}",
        flush=True,
    )

    rounds: list[int] = []
    raw_oof = np.empty(n, dtype=np.float64)
    p_res = np.empty(n, dtype=np.float64)
    slopes = []
    for k in range(4):
        a = np.flatnonzero(d_tr != k)
        b = np.flatnonzero(d_tr == k)
        ik = [j for j in range(4) if j != k][-1]
        fit = a[d_tr[a] != ik]
        val = a[d_tr[a] == ik]

        best = train_early(
            X[fit], y[fit], d_tr[fit],
            X[val], y[val], d_tr[val],
            params, args.max_rounds, args.patience,
        )
        rounds.append(best)
        raw_b, raw_val = bag_rank(
            X[fit], y[fit], d_tr[fit],
            X[b], X[val], max(1, int(best)), params, seeds,
        )
        slope = fit_slope(raw_val, y[val], d_tr[val])
        slopes.append(slope)
        raw_oof[b] = raw_b
        p_res[b] = calibrated(slope * raw_b, d_tr[b])
        score_k, ll_k, auc_k = composite(y[b], p_res[b])
        print(
            f"month {k} composite={score_k:.9f} ll={ll_k:.9f} "
            f"auc={auc_k:.9f} rounds={best} slope={slope:.4f}",
            flush=True,
        )

    score_res, ll_res, auc_res = composite(y, p_res)
    slope_oof = fit_slope(raw_oof, y, d_tr)
    per_month = []
    for k in range(4):
        m = d_tr == k
        s_k, ll_k, auc_k = composite(y[m], p_res[m])
        per_month.append({"snapshot": k, "composite": s_k, "log_loss": ll_k, "auc": auc_k})

    p_test = None
    stats = None
    if args.test:
        init_rounds = scaled_rounds(rounds, 1.0)
        raw_test, _ = bag_rank(
            X[:n], y, d_tr, X[n:], X[:n], init_rounds, params, seeds
        )
        p_test = calibrated(slope_oof * raw_test, d_te)
        if out_path is None:
            out_path = Path(f"submission_v9_{tag}.csv")
        out = sample.copy()
        out["Target"] = p_test
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False, float_format="%.17g")
        stats = validate_submission(out_path, sample)

    if preds_path is not None:
        preds_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "y": y,
            "oof": p_res,
            "d_tr": d_tr,
            "d_te": d_te,
            "rounds": np.asarray(rounds),
            "slopes": np.asarray(slopes),
            "slope_oof": slope_oof,
            "raw_oof": raw_oof,
        }
        if p_test is not None:
            payload["test"] = p_test
            payload["raw_test"] = raw_test
        np.savez_compressed(preds_path, **payload)

    elapsed = time.time() - start
    payload = {
        "model": "v9_rank",
        "variant": args.variant,
        "tag": tag,
        "feature_count": int(X.shape[1]),
        "seeds": list(seeds),
        "n_seeds": len(seeds),
        "rounds_per_fold": rounds,
        "final_rounds": scaled_rounds(rounds, 1.0),
        "slopes_per_fold": slopes,
        "slope_oof": slope_oof,
        "validation": {"composite": score_res, "log_loss": ll_res, "auc": auc_res},
        "per_month": per_month,
        "elapsed_seconds": elapsed,
        "preds": str(preds_path.resolve()) if preds_path else None,
        "submission": str(out_path.resolve()) if out_path else None,
    }
    write_json(report_path, payload)
    print(
        f"rank LOMO composite={score_res:.9f} ll={ll_res:.9f} "
        f"auc={auc_res:.9f} slope_oof={slope_oof:.4f} elapsed={elapsed:.1f}s",
        flush=True,
    )
    print(f"report={report_path}", flush=True)


if __name__ == "__main__":
    main()
