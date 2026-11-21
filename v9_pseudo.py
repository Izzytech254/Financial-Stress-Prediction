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
from v6_residual import (
    CONFIGS,
    early_stopping_rounds,
    make_params,
    seeds_for,
)
from v8_screen import build_variant


def bag_weighted(
    X: np.ndarray,
    y: np.ndarray,
    init: np.ndarray,
    weight: np.ndarray,
    Xp: np.ndarray,
    rounds: int,
    params: dict,
    seeds: tuple[int, ...],
) -> np.ndarray:
    ds = lgb.Dataset(
        X, y, init_score=init, weight=weight, free_raw_data=False
    )
    raws = []
    for seed in seeds:
        p = {**params, "seed": seed}
        booster = lgb.train(p, ds, num_boost_round=max(1, rounds))
        raws.append(booster.predict(Xp, raw_score=True))
    return np.mean(raws, axis=0)


def base_fold(
    X: np.ndarray,
    y: np.ndarray,
    d: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    init_a: np.ndarray,
    init_b: np.ndarray,
    seeds: tuple[int, ...],
    threads: int,
    max_rounds: int,
    patience: int,
    round_scale: float,
):
    ik = [j for j in range(4) if j != int(d[b][0])][-1]
    fit = a[d[a] != ik]
    val = a[d[a] == ik]
    init_fit = init_of_from(init_a, a, fit)
    init_val = init_of_from(init_a, a, val)
    raws = []
    rounds = []
    for cfg in CONFIGS:
        params = make_params(cfg, threads)
        best = early_stopping_rounds(
            X[fit],
            y[fit],
            init_fit,
            X[val],
            y[val],
            init_val,
            params,
            max_rounds,
            patience,
        )
        rounds.append(best)
        raws.append(
            bag_weighted(
                X[a],
                y[a],
                init_a,
                np.ones(len(a), dtype=np.float32),
                X[b],
                max(1, int(round(round_scale * best))),
                params,
                seeds,
            )
        )
    combined = expit(init_b + np.mean(raws, axis=0))
    return per_date(probit_eta(combined), d[b]), rounds


def init_of_from(init_full: np.ndarray, source_idx: np.ndarray, target_idx: np.ndarray):
    pos = {int(v): i for i, v in enumerate(source_idx)}
    return init_full[np.array([pos[int(v)] for v in target_idx], dtype=int)]


def pseudo_labels(p: np.ndarray, frac: float) -> np.ndarray:
    n = len(p)
    k = max(1, int(round(frac * n)))
    order = np.argsort(-p)
    labels = np.full(n, -1, dtype=np.int8)
    labels[order[:k]] = 1
    labels[order[-k:]] = 0
    return labels


def retrain_fold(
    X: np.ndarray,
    y: np.ndarray,
    d: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    pseudo: np.ndarray,
    init_a: np.ndarray,
    init_b: np.ndarray,
    rounds: list[int],
    pseudo_weight: float,
    seeds: tuple[int, ...],
    threads: int,
):
    keep = pseudo >= 0
    b_keep = b[keep]
    y_keep = pseudo[keep].astype(np.float64)
    idx = np.r_[a, b_keep]
    y_all = np.r_[y[a], y_keep]
    init_all = np.r_[init_a, init_b[keep]]
    weight = np.r_[np.ones(len(a)), np.full(len(b_keep), pseudo_weight)]
    raws = []
    for c, cfg in enumerate(CONFIGS):
        params = make_params(cfg, threads)
        raws.append(
            bag_weighted(
                X[idx],
                y_all,
                init_all,
                weight,
                X[b],
                rounds[c],
                params,
                seeds,
            )
        )
    combined = expit(init_b + np.mean(raws, axis=0))
    return per_date(probit_eta(combined), d[b])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--out", default="submission_v9_pseudo.csv")
    ap.add_argument("--report", default=str(REPORT_DIR / "v9_pseudo.json"))
    ap.add_argument("--preds", default=str(REPORT_DIR / "v9_pseudo_predictions.npz"))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--pseudo-weight", type=float, default=0.2)
    ap.add_argument("--pseudo-frac", type=float, default=0.15)
    ap.add_argument("--round-scale", type=float, default=1.0)
    ap.add_argument("--max-rounds", type=int, default=600)
    ap.add_argument("--patience", type=int, default=60)
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
    s_mz = month_z_score(R, d_all)
    lin = MZ_SLOPE * s_mz
    X, names, drop_cols, month_scale, extra_cols = build_variant(
        "drop_m5_all", all_rows, d_all, R, None
    )

    print(
        f"pseudo drop_m5_all features={X.shape[1]} seeds={len(seeds)} "
        f"weight={args.pseudo_weight} frac={args.pseudo_frac}",
        flush=True,
    )

    p_base = np.empty(n, dtype=np.float64)
    p_pseudo = np.empty(n, dtype=np.float64)
    fold_rounds = []
    for k in range(4):
        a = np.flatnonzero(d_tr != k)
        b = np.flatnonzero(d_tr == k)
        init_a = init_of(lin, d_all, a)[0]
        init_b = init_of(lin, d_all, b)[0]

        pb, rounds = base_fold(
            X,
            y,
            d_tr,
            a,
            b,
            init_a,
            init_b,
            seeds,
            args.threads,
            args.max_rounds,
            args.patience,
            args.round_scale,
        )
        p_base[b] = pb
        labels = pseudo_labels(pb, args.pseudo_frac)
        pp = retrain_fold(
            X,
            y,
            d_tr,
            a,
            b,
            labels,
            init_a,
            init_b,
            rounds,
            args.pseudo_weight,
            seeds,
            args.threads,
        )
        p_pseudo[b] = pp
        fold_rounds.append(rounds)
        sb, lb, ab = composite(y[b], pb)
        sp, lp, ap = composite(y[b], pp)
        print(
            f"month {k} base={sb:.9f} pseudo={sp:.9f} "
            f"gain={sp - sb:+.9f} rounds={rounds}",
            flush=True,
        )

    score_base, ll_base, auc_base = composite(y, p_base)
    score_pseudo, ll_pseudo, auc_pseudo = composite(y, p_pseudo)
    print(
        f"LOMO base={score_base:.9f} pseudo={score_pseudo:.9f} "
        f"gain={score_pseudo - score_base:+.9f}",
        flush=True,
    )

    p_test_base = None
    p_test_pseudo = None
    test_stats = None
    if args.test:
        tr = np.arange(n)
        te = np.arange(n, n + len(test))
        init_tr = init_of(lin, d_all, tr)[0]
        init_te = init_of(lin, d_all, te)[0]
        pt, rounds = base_fold(
            X,
            y,
            d_tr,
            tr,
            te,
            init_tr,
            init_te,
            seeds,
            args.threads,
            args.max_rounds,
            args.patience,
            args.round_scale,
        )
        p_test_base = pt
        labels = np.full(len(te), -1, dtype=np.int8)
        for k in (4, 5, 6):
            m = d_te == k
            idx = np.flatnonzero(m)
            local = pseudo_labels(pt[m], args.pseudo_frac)
            labels[idx] = local
        p_test_pseudo = retrain_fold(
            X,
            y,
            d_all,
            tr,
            te,
            labels,
            init_tr,
            init_te,
            rounds,
            args.pseudo_weight,
            seeds,
            args.threads,
        )
        out = sample.copy()
        out["Target"] = p_test_pseudo
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.out, index=False, float_format="%.17g")
        test_stats = validate_submission(args.out, sample)

    if args.preds:
        Path(args.preds).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "y": y,
            "base_oof": p_base,
            "pseudo_oof": p_pseudo,
            "d_tr": d_tr,
            "d_te": d_te,
            "rounds": np.asarray(fold_rounds),
        }
        if p_test_base is not None:
            payload["base_test"] = p_test_base
            payload["pseudo_test"] = p_test_pseudo
        np.savez_compressed(args.preds, **payload)

    payload = {
        "model": "v9_pseudo",
        "variant": "drop_m5_all",
        "pseudo_weight": args.pseudo_weight,
        "pseudo_frac": args.pseudo_frac,
        "seeds": list(seeds),
        "n_seeds": len(seeds),
        "threads": args.threads,
        "feature_count": int(X.shape[1]),
        "drop_cols": drop_cols,
        "month_scale": month_scale,
        "fold_rounds": fold_rounds,
        "base": {
            "composite": score_base,
            "log_loss": ll_base,
            "auc": auc_base,
        },
        "pseudo": {
            "composite": score_pseudo,
            "log_loss": ll_pseudo,
            "auc": auc_pseudo,
        },
        "test": test_stats,
        "elapsed_seconds": time.time() - start,
    }
    write_json(Path(args.report), payload)
    print(f"report={args.report}", flush=True)


if __name__ == "__main__":
    main()
