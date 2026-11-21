from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from v6_common import (
    DATA_DIR,
    MZ_SLOPE,
    REPORT_DIR,
    TARGET,
    calibrated,
    composite,
    load_data,
    month_z_score,
    probit_eta,
    risk_factors,
    snapshot_dates,
    validate_submission,
    write_json,
)
from v8_screen import build_variant


def correlation_matrix(X: np.ndarray) -> np.ndarray:
    C = np.corrcoef(X, rowvar=False)
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(C, 1.0)
    return 0.5 * (C + C.T)


def month_alphas(y, d_tr, struct_lin, best_lin):
    out = {}
    grid = np.linspace(-0.5, 1.5, 201)
    for k in np.unique(d_tr):
        m = d_tr == k
        best = None
        for a in grid:
            p = calibrated(struct_lin[m] + a * (best_lin[m] - struct_lin[m]), d_tr[m])
            sc = composite(y[m], p)[0]
            if best is None or sc > best[0]:
                best = (float(sc), float(a))
        out[int(k)] = best
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--variant", default="drop_m5_all")
    ap.add_argument("--best", default=str(REPORT_DIR / "v9_drop_m5_all_s50_predictions.npz"))
    ap.add_argument("--struct", default=str(REPORT_DIR / "v9_struct_fit_predictions.npz"))
    ap.add_argument("--out", default="submission_v9_snapshot_alpha.csv")
    ap.add_argument("--report", default=str(REPORT_DIR / "v9_snapshot_alpha.json"))
    args = ap.parse_args()

    train, test, sample = load_data(args.data_dir)
    d_tr, d_te = snapshot_dates(train, test)
    y = train[TARGET].to_numpy()
    n = len(train)
    all_rows = pd.concat([train.drop(columns=TARGET), test], ignore_index=True)
    d_all = np.r_[d_tr, d_te]

    R = risk_factors(all_rows)
    s_mz = month_z_score(R, d_all)
    X, names, drop_cols, month_scale, extra_cols = build_variant(
        args.variant, all_rows, d_all, R, None
    )

    best = np.load(args.best)
    best_oof = best["oof"]
    best_test = best["test"]
    struct = np.load(args.struct)
    if not np.array_equal(best["y"], y):
        raise ValueError("best prediction y mismatch")
    if not np.array_equal(struct["y"], y):
        raise ValueError("structural prediction y mismatch")
    struct_lin_tr = MZ_SLOPE * s_mz[:n]
    struct_lin_te = MZ_SLOPE * s_mz[n:]
    best_lin_oof = probit_eta(best_oof)
    best_lin_test = probit_eta(best_test)

    alphas = month_alphas(y, d_tr, struct_lin_tr, best_lin_oof)
    print("month alphas:", alphas, flush=True)

    corrs = {}
    for k in np.unique(d_all):
        m = d_all == k
        corrs[int(k)] = correlation_matrix(X[m])

    mapping = {}
    distances = {}
    for j in np.unique(d_te):
        c = corrs[int(j)]
        dists = {k: float(np.linalg.norm(c - corrs[k])) for k in range(4)}
        mapping[int(j)] = int(min(dists, key=dists.get))
        distances[int(j)] = dists
    print("snapshot mapping:", mapping, flush=True)
    print("distances:", distances, flush=True)

    p_test = np.empty(len(test), dtype=np.float64)
    test_alphas = {}
    for j in np.unique(d_te):
        m = d_te == j
        k = mapping[int(j)]
        a = 0.5 * alphas[k][1] + 0.5 * 0.25
        a = float(np.clip(a, -0.3, 1.5))
        test_alphas[int(j)] = a
        eta = struct_lin_te[m] + a * (best_lin_test[m] - struct_lin_te[m])
        p_test[m] = calibrated(eta, d_te[m])

    out = sample.copy()
    out["Target"] = p_test
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, float_format="%.17g")
    stats = validate_submission(args.out, sample)

    payload = {
        "model": "v9_snapshot_alpha",
        "variant": args.variant,
        "feature_count": int(X.shape[1]),
        "best_preds": str(Path(args.best).resolve()),
        "struct_preds": str(Path(args.struct).resolve()),
        "month_alphas": alphas,
        "snapshot_mapping": mapping,
        "snapshot_distances": distances,
        "test_alphas": test_alphas,
        "test": stats,
        "submission": str(Path(args.out).resolve()),
    }
    write_json(Path(args.report), payload)
    print(f"test alphas: {test_alphas}", flush=True)
    print(f"submission={args.out}", flush=True)
    print(f"report={args.report}", flush=True)


if __name__ == "__main__":
    main()
