from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import log_ndtr, ndtri, softmax

from v6_common import (
    DATA_DIR,
    MZ_SLOPE,
    REPORT_DIR,
    TARGET,
    WEIGHTS,
    calibrated,
    composite,
    load_data,
    month_z_score,
    risk_factors,
    snapshot_dates,
    write_json,
)


def z_factors(R: np.ndarray, d_all: np.ndarray) -> np.ndarray:
    capped = np.minimum(R, np.quantile(R, 0.99, axis=0))
    Z = np.empty_like(capped)
    for k in np.unique(d_all):
        m = d_all == k
        sd = capped[m].std(0)
        Z[m] = (capped[m] - capped[m].mean(0)) / np.where(sd > 0, sd, 1)
    return Z


def fit_probit(Z, y, d, w0, s0=2.857, reg=1.0, n_starts=8, seed=42):
    levels = np.unique(d)
    idx = np.searchsorted(levels, d)
    K = len(levels)
    rng = np.random.default_rng(seed)

    def unpack(x):
        v = x[: Z.shape[1]]
        log_s = x[Z.shape[1]]
        b = x[Z.shape[1] + 1 :]
        w = softmax(v)
        s = np.exp(log_s)
        return w, s, b

    def nll(x):
        w, s, b = unpack(x)
        eta = b[idx] + s * (Z @ w)
        ll = y * log_ndtr(eta) + (1 - y) * log_ndtr(-eta)
        return -ll.sum() + reg * np.sum((w - w0) ** 2)

    bounds = (
        [(-10.0, 10.0)] * Z.shape[1]
        + [(np.log(0.1), np.log(30.0))]
        + [(-10.0, 10.0)] * K
    )
    base_b = np.full(K, ndtri(0.15))
    starts = [
        np.r_[np.log(w0), np.log(s0), base_b],
    ]
    for _ in range(n_starts - 1):
        w = rng.dirichlet(np.ones(Z.shape[1]))
        starts.append(
            np.r_[
                np.log(w),
                np.log(s0 * rng.uniform(0.7, 1.3)),
                base_b + rng.normal(0, 0.2, K),
            ]
        )
    best = None
    for x0 in starts:
        res = minimize(
            nll,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 3000, "ftol": 1e-13, "gtol": 1e-9},
        )
        if best is None or res.fun < best.fun:
            best = res
    w, s, b = unpack(best.x)
    return w, float(s), b


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--reg", type=float, default=1.0)
    ap.add_argument("--n-starts", type=int, default=8)
    ap.add_argument("--report", default=str(REPORT_DIR / "v9_struct_fit.json"))
    ap.add_argument("--preds", default=str(REPORT_DIR / "v9_struct_fit_predictions.npz"))
    args = ap.parse_args()

    start = time.time()
    train, test, sample = load_data(args.data_dir)
    d_tr, d_te = snapshot_dates(train, test)
    y = train[TARGET].to_numpy()
    all_rows = pd.concat([train.drop(columns=TARGET), test], ignore_index=True)
    d_all = np.r_[d_tr, d_te]
    R = risk_factors(all_rows)
    Z = z_factors(R, d_all)
    Z_tr = Z[: len(y)]
    Z_te = Z[len(y) :]
    lin_cur = MZ_SLOPE * month_z_score(R, d_all)

    p_cur = np.empty(len(y), dtype=np.float64)
    for k in range(4):
        m = d_tr == k
        p_cur[m] = calibrated(lin_cur[: len(y)][m], d_tr[m])
    score_cur, ll_cur, auc_cur = composite(y, p_cur)
    print(
        f"current structural LOMO={score_cur:.9f} ll={ll_cur:.9f} auc={auc_cur:.9f}",
        flush=True,
    )

    lin_fit = np.empty_like(lin_cur)
    p_fit = np.empty(len(y), dtype=np.float64)
    folds = []
    for k in range(4):
        fit = d_tr != k
        hold = d_tr == k
        w, s, b = fit_probit(
            Z_tr[fit], y[fit], d_tr[fit], WEIGHTS, reg=args.reg, n_starts=args.n_starts
        )
        score = s * (Z_tr[hold] @ w)
        lin_fit[np.flatnonzero(hold)] = score
        p_fit[hold] = calibrated(score, d_tr[hold])
        sk, llk, au = composite(y[hold], p_fit[hold])
        folds.append(
            {
                "snapshot": k,
                "weights": w.tolist(),
                "slope": s,
                "intercepts": b.tolist(),
                "composite": sk,
                "log_loss": llk,
                "auc": au,
            }
        )
        print(
            f"fold {k} composite={sk:.9f} ll={llk:.9f} auc={au:.9f} slope={s:.4f} "
            f"w={np.round(w, 5).tolist()}",
            flush=True,
        )

    w_full, s_full, b_full = fit_probit(
        Z_tr, y, d_tr, WEIGHTS, reg=args.reg, n_starts=args.n_starts
    )
    te = d_all >= 4
    lin_fit[te] = s_full * (Z_te @ w_full)
    p_test_struct = calibrated(lin_fit[te], d_te)
    score_fit, ll_fit, auc_fit = composite(y, p_fit)
    print(
        f"fitted structural LOMO={score_fit:.9f} ll={ll_fit:.9f} auc={auc_fit:.9f} "
        f"full slope={s_full:.4f}",
        flush=True,
    )
    print(
        f"full weights={np.round(w_full, 6).tolist()} current={np.round(WEIGHTS, 6).tolist()}",
        flush=True,
    )

    preds_path = Path(args.preds)
    preds_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        preds_path,
        y=y,
        d_tr=d_tr,
        d_te=d_te,
        lin_all=lin_fit,
        lin_current=lin_cur,
        oof=p_fit,
        test=p_test_struct,
        weights_folds=np.asarray([f["weights"] for f in folds]),
        slopes_folds=np.asarray([f["slope"] for f in folds]),
        weights_full=w_full,
        slope_full=s_full,
    )
    for a in (0.1, 0.25, 0.5, 0.75, 1.0):
        lin_a = (1.0 - a) * lin_cur + a * lin_fit
        np.savez_compressed(
            REPORT_DIR / f"v9_struct_lin_a{int(round(a * 100)):03d}.npz",
            lin_all=lin_a,
        )

    payload = {
        "model": "v9_struct_fit",
        "reg": args.reg,
        "n_starts": args.n_starts,
        "current": {
            "weights": WEIGHTS.tolist(),
            "composite": score_cur,
            "log_loss": ll_cur,
            "auc": auc_cur,
        },
        "fitted": {
            "weights_full": w_full.tolist(),
            "slope_full": s_full,
            "composite": score_fit,
            "log_loss": ll_fit,
            "auc": auc_fit,
        },
        "folds": folds,
        "elapsed_seconds": time.time() - start,
        "preds": str(preds_path.resolve()),
    }
    write_json(Path(args.report), payload)
    print(f"report={args.report}", flush=True)


if __name__ == "__main__":
    main()
