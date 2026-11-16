from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit

from v6_common import (
    DATA_DIR,
    MZ_SLOPE,
    REPORT_DIR,
    TARGET,
    composite,
    features,
    init_of,
    load_data,
    month_scaled,
    month_z_score,
    per_date,
    probit_eta,
    risk_factors,
    snapshot_dates,
    structural_oof_mz,
    validate_submission,
    write_json,
)
from v6_residual import BASE, CONFIGS, make_params, scaled_rounds, seeds_for


def time_weights(d: np.ndarray, kind: str, slope: float) -> np.ndarray:
    if kind == "none":
        return np.ones(len(d), dtype=np.float64)
    if kind == "linear":
        return 1.0 + slope * d.astype(np.float64)
    if kind == "sqrt":
        return np.sqrt(1.0 + slope * d.astype(np.float64))
    if kind == "exp":
        return np.exp(slope * d.astype(np.float64))
    raise ValueError(kind)


def early_stop(
    X: np.ndarray,
    y: np.ndarray,
    init: np.ndarray,
    w: np.ndarray,
    Xv: np.ndarray,
    yv: np.ndarray,
    iv: np.ndarray,
    params: dict,
    max_rounds: int,
    patience: int,
) -> int:
    ds = lgb.Dataset(X, y, init_score=init, weight=w, free_raw_data=False)
    dv = lgb.Dataset(Xv, yv, init_score=iv, reference=ds)
    booster = lgb.train(
        params,
        ds,
        num_boost_round=max_rounds,
        valid_sets=[dv],
        callbacks=[lgb.early_stopping(patience, verbose=False)],
    )
    return max(int(booster.best_iteration), 1)


def bag_weighted(
    X: np.ndarray,
    y: np.ndarray,
    init: np.ndarray,
    w: np.ndarray,
    Xp: np.ndarray,
    rounds: int,
    params: dict,
    seeds: tuple[int, ...],
) -> np.ndarray:
    ds = lgb.Dataset(X, y, init_score=init, weight=w, free_raw_data=False)
    raws = []
    for seed in seeds:
        p = {**params, "seed": seed}
        booster = lgb.train(p, ds, num_boost_round=rounds)
        raws.append(booster.predict(Xp, raw_score=True))
    return np.mean(raws, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--out", default="submission_v9_time_residual.csv")
    ap.add_argument("--report", default=str(REPORT_DIR / "v9_time_residual.json"))
    ap.add_argument("--preds", default=str(REPORT_DIR / "v9_time_residual_predictions.npz"))
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--time-weight", default="linear", choices=["none", "linear", "sqrt", "exp"])
    ap.add_argument("--slope", type=float, default=0.5)
    ap.add_argument("--round-scale", type=float, default=1.0)
    ap.add_argument("--max-rounds", type=int, default=600)
    ap.add_argument("--patience", type=int, default=60)
    ap.add_argument("--drop-m5-deposit", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--base-only", action="store_true")
    args = ap.parse_args()

    start = time.time()
    seeds = seeds_for(args.seeds)
    configs = [{}] if args.base_only else CONFIGS
    train, test, sample = load_data(args.data_dir)
    d_tr, d_te = snapshot_dates(train, test)
    y = train[TARGET].to_numpy()
    n = len(train)
    all_rows = pd.concat([train.drop(columns=TARGET), test], ignore_index=True)
    d_all = np.r_[d_tr, d_te]

    R = risk_factors(all_rows)
    s_mz = month_z_score(R, d_all)
    lin = MZ_SLOPE * s_mz
    drop_cols = (
        [c for c in all_rows.columns if c.startswith("m5_deposit")]
        if args.drop_m5_deposit
        else []
    )
    X = features(all_rows, R, drop_cols)
    X = month_scaled(X, d_all, "rank")

    p_struct = structural_oof_mz(s_mz, d_tr)
    score_struct, ll_struct, auc_struct = composite(y, p_struct)
    w_tr = time_weights(d_tr, args.time_weight, args.slope)
    print(
        f"structural={score_struct:.9f} features={X.shape[1]} seeds={len(seeds)} "
        f"weights={args.time_weight} slope={args.slope}",
        flush=True,
    )
    print(f"month weights={[float(w_tr[d_tr == k].mean()) for k in range(4)]}", flush=True)

    rounds: list[list[int]] = [[] for _ in configs]
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

        raws = []
        for c, cfg in enumerate(configs):
            params = make_params(cfg, args.threads)
            best = early_stop(
                X[fit],
                y[fit],
                init_fit,
                w_tr[fit],
                X[val],
                y[val],
                init_val,
                params,
                args.max_rounds,
                args.patience,
            )
            rounds[c].append(best)
            raws.append(
                bag_weighted(
                    X[a],
                    y[a],
                    init_a,
                    w_tr[a],
                    X[b],
                    max(1, int(round(args.round_scale * best))),
                    params,
                    seeds,
                )
            )
        combined = expit(init_b + np.mean(raws, axis=0))
        p_res[b] = per_date(probit_eta(combined), d_tr[b])
        score_k, ll_k, auc_k = composite(y[b], p_res[b])
        print(
            f"month {k} score={score_k:.9f} ll={ll_k:.9f} auc={auc_k:.9f} "
            f"rounds={[r[-1] for r in rounds]}",
            flush=True,
        )

    score_res, ll_res, auc_res = composite(y, p_res)
    print(
        f"residual LOMO={score_res:.9f} ll={ll_res:.9f} auc={auc_res:.9f} "
        f"gain={score_res - score_struct:+.9f}",
        flush=True,
    )

    tr_idx = np.arange(n)
    te_idx = np.arange(n, n + len(test))
    init_tr = init_of(lin, d_all, tr_idx)[0]
    init_te = init_of(lin, d_all, te_idx)[0]
    final_rounds = [scaled_rounds(r, args.round_scale) for r in rounds]
    final_raws = []
    for c, cfg in enumerate(configs):
        params = make_params(cfg, args.threads)
        final_raws.append(
            bag_weighted(
                X[:n],
                y,
                init_tr,
                w_tr,
                X[n:],
                final_rounds[c],
                params,
                seeds,
            )
        )
    p_test = per_date(probit_eta(expit(init_te + np.mean(final_raws, axis=0))), d_te)

    out = sample.copy()
    out["Target"] = p_test
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, float_format="%.17g")
    stats = validate_submission(args.out, sample)

    Path(args.preds).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.preds,
        y=y,
        oof=p_res,
        test=p_test,
        structural_oof=p_struct,
        d_tr=d_tr,
        d_te=d_te,
        rounds=np.asarray(rounds),
        s_mz=s_mz,
        weights=w_tr,
    )

    payload = {
        "model": "v9_time_residual",
        "seeds": list(seeds),
        "n_seeds": len(seeds),
        "threads": args.threads,
        "time_weight": args.time_weight,
        "slope": args.slope,
        "drop_m5_deposit": bool(args.drop_m5_deposit),
        "feature_count": int(X.shape[1]),
        "configs": configs,
        "rounds": rounds,
        "final_rounds": final_rounds,
        "validation": {
            "residual": {"composite": score_res, "log_loss": ll_res, "auc": auc_res},
            "structural": {
                "composite": score_struct,
                "log_loss": ll_struct,
                "auc": auc_struct,
            },
            "gain": score_res - score_struct,
        },
        "test": {
            **stats,
            "snapshot_means": [float(p_test[d_te == k].mean()) for k in (4, 5, 6)],
        },
        "elapsed_seconds": time.time() - start,
        "submission": str(Path(args.out).resolve()),
    }
    write_json(args.report, payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
