from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from scipy.special import expit

from v6_common import (
    DATA_DIR,
    MZ_SLOPE,
    REPORT_DIR,
    TARGET,
    composite,
    init_of,
    load_data,
    month_z_score,
    per_date,
    probit_eta,
    risk_factors,
    snapshot_dates,
    validate_submission,
    write_json,
)
from v6_residual import scaled_rounds, seeds_for
from v8_screen import VARIANTS, build_variant, load_importance

BASE = {
    "loss_function": "Logloss",
    "eval_metric": "Logloss",
    "learning_rate": 0.02,
    "depth": 3,
    "l2_leaf_reg": 50.0,
    "bootstrap_type": "Bernoulli",
    "subsample": 0.8,
    "rsm": 0.7,
    "allow_writing_files": False,
}


def make_model(params: dict, iterations: int, seed: int, threads: int):
    return CatBoostClassifier(
        **params,
        iterations=iterations,
        random_seed=seed,
        thread_count=threads,
        verbose=False,
    )


def train_early(
    X: np.ndarray,
    y: np.ndarray,
    init: np.ndarray,
    Xv: np.ndarray,
    yv: np.ndarray,
    iv: np.ndarray,
    params: dict,
    max_rounds: int,
    patience: int,
    seed: int,
    threads: int,
) -> int:
    train_pool = Pool(X, label=y, baseline=init)
    valid_pool = Pool(Xv, label=yv, baseline=iv)
    model = make_model(params, max_rounds, seed, threads)
    model.fit(
        train_pool,
        eval_set=valid_pool,
        use_best_model=True,
        early_stopping_rounds=patience,
    )
    return max(int(model.get_best_iteration()) + 1, 1)


def bag(
    X: np.ndarray,
    y: np.ndarray,
    init: np.ndarray,
    Xp: np.ndarray,
    initp: np.ndarray,
    rounds: int,
    params: dict,
    seeds: tuple[int, ...],
    threads: int,
) -> np.ndarray:
    train_pool = Pool(X, label=y, baseline=init)
    pred_pool = Pool(Xp, baseline=initp)
    raws = []
    for seed in seeds:
        model = make_model(params, max(1, rounds), seed, threads)
        model.fit(train_pool)
        raws.append(model.predict(pred_pool, prediction_type="RawFormulaVal"))
    return np.mean(raws, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--variant", default="add_extras", choices=VARIANTS)
    ap.add_argument("--config-json", default=None)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--round-scale", type=float, default=1.0)
    ap.add_argument("--max-rounds", type=int, default=600)
    ap.add_argument("--patience", type=int, default=60)
    ap.add_argument("--importance", default=None)
    ap.add_argument("--tag", default="cat")
    ap.add_argument("--report", default=None)
    ap.add_argument("--preds", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()

    start = time.time()
    seeds = seeds_for(args.seeds)
    params = dict(BASE)
    if args.config_json:
        cfg = json.loads(args.config_json)
        if not isinstance(cfg, dict):
            raise ValueError("config must be an object")
        params.update(cfg)
    importance = load_importance(args.importance) if args.importance else None

    train, test, sample = load_data(args.data_dir)
    d_tr, d_te = snapshot_dates(train, test)
    y = train[TARGET].to_numpy()
    n = len(train)
    all_rows = pd.concat([train.drop(columns=TARGET), test], ignore_index=True)
    d_all = np.r_[d_tr, d_te]

    R = risk_factors(all_rows)
    s_mz = month_z_score(R, d_all)
    lin = MZ_SLOPE * s_mz
    X, names, drop_cols, month_scale, extra_cols = build_variant(
        args.variant, all_rows, d_all, R, importance
    )

    report_path = Path(args.report or REPORT_DIR / f"v8_cat_{args.tag}.json")
    preds_path = Path(args.preds) if args.preds else None
    out_path = Path(args.out) if args.out else None

    print(
        f"cat variant={args.variant} tag={args.tag} features={X.shape[1]} "
        f"seeds={len(seeds)} scale={month_scale}",
        flush=True,
    )

    rounds: list[int] = []
    p_res = np.empty(n, dtype=np.float64)
    for k in range(4):
        a = np.flatnonzero(d_tr != k)
        b = np.flatnonzero(d_tr == k)
        ik = [j for j in range(4) if j != k][-1]
        fit = a[d_tr[a] != ik]
        val = a[d_tr[a] == ik]

        init_fit = init_of(lin, d_all, fit)[0]
        init_val = init_of(lin, d_all, val)[0]
        init_a = init_of(lin, d_all, a)[0]
        init_b = init_of(lin, d_all, b)[0]

        best = train_early(
            X[fit],
            y[fit],
            init_fit,
            X[val],
            y[val],
            init_val,
            params,
            args.max_rounds,
            args.patience,
            seeds[0],
            args.threads,
        )
        rounds.append(best)
        raw = bag(
            X[a],
            y[a],
            init_a,
            X[b],
            init_b,
            max(1, int(round(args.round_scale * best))),
            params,
            seeds,
            args.threads,
        )
        p_res[b] = per_date(probit_eta(expit(raw)), d_tr[b])
        score_k, ll_k, auc_k = composite(y[b], p_res[b])
        print(
            f"month {k} composite={score_k:.9f} ll={ll_k:.9f} "
            f"auc={auc_k:.9f} rounds={best}",
            flush=True,
        )

    score_res, ll_res, auc_res = composite(y, p_res)
    per_month = []
    for k in range(4):
        m = d_tr == k
        s_res, ll_k, auc_k = composite(y[m], p_res[m])
        per_month.append(
            {
                "snapshot": k,
                "composite": s_res,
                "log_loss": ll_k,
                "auc": auc_k,
            }
        )

    final_rounds = [scaled_rounds(rounds, args.round_scale)]
    p_test = None
    stats = None
    if args.test:
        tr = np.arange(n)
        te = np.arange(n, n + len(test))
        init_tr = init_of(lin, d_all, tr)[0]
        init_te = init_of(lin, d_all, te)[0]
        raw = bag(
            X[:n],
            y,
            init_tr,
            X[n:],
            init_te,
            final_rounds[0],
            params,
            seeds,
            args.threads,
        )
        p_test = per_date(probit_eta(expit(raw)), d_te)
        if out_path is None:
            out_path = Path(f"submission_v8_{args.tag}.csv")
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
            "s_mz": s_mz,
        }
        if p_test is not None:
            payload["test"] = p_test
        np.savez_compressed(preds_path, **payload)

    elapsed = time.time() - start
    payload = {
        "model": "v8_cat",
        "variant": args.variant,
        "tag": args.tag,
        "month_scale": month_scale,
        "drop_cols": drop_cols,
        "extra_cols": extra_cols,
        "feature_count": int(X.shape[1]),
        "params": params,
        "seeds": list(seeds),
        "n_seeds": len(seeds),
        "threads": args.threads,
        "round_scale": args.round_scale,
        "rounds": rounds,
        "final_rounds": final_rounds,
        "validation": {
            "composite": score_res,
            "log_loss": ll_res,
            "auc": auc_res,
        },
        "per_month": per_month,
        "elapsed_seconds": elapsed,
        "report": str(report_path.resolve()),
        "preds": str(preds_path.resolve()) if preds_path else None,
        "submission": str(out_path.resolve()) if out_path else None,
    }
    if stats is not None and p_test is not None:
        payload["test"] = {
            **stats,
            "snapshot_means": [
                float(p_test[d_te == k].mean()) for k in (4, 5, 6)
            ],
        }
    write_json(report_path, payload)
    print(
        f"cat LOMO composite={score_res:.9f} ll={ll_res:.9f} "
        f"auc={auc_res:.9f} elapsed={elapsed:.1f}s",
        flush=True,
    )
    print(f"report={report_path}", flush=True)


if __name__ == "__main__":
    main()
