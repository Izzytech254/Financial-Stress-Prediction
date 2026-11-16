from __future__ import annotations

import argparse
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from v6_common import (
    DATA_DIR,
    MZ_SLOPE,
    REPORT_DIR,
    TARGET,
    features,
    init_of,
    load_data,
    month_scaled,
    month_z_score,
    risk_factors,
    snapshot_dates,
    write_json,
)
from v6_residual import early_stopping_rounds, make_params
from v7_common import feature_names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--out", default=str(REPORT_DIR / "v8_feature_importance.json"))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--max-rounds", type=int, default=600)
    ap.add_argument("--patience", type=int, default=60)
    args = ap.parse_args()

    start = time.time()
    train, test, sample = load_data(args.data_dir)
    d_tr, d_te = snapshot_dates(train, test)
    y = train[TARGET].to_numpy()
    all_rows = pd.concat([train.drop(columns=TARGET), test], ignore_index=True)
    d_all = np.r_[d_tr, d_te]

    R = risk_factors(all_rows)
    s_mz = month_z_score(R, d_all)
    lin = MZ_SLOPE * s_mz
    drop_cols = [c for c in all_rows.columns if c.startswith("m5_deposit")]
    X = features(all_rows, R, drop_cols)
    X = month_scaled(X, d_all, "rank")
    names = feature_names(all_rows, drop_cols)
    assert len(names) == X.shape[1], (len(names), X.shape[1])

    gains = np.zeros(X.shape[1], dtype=np.float64)
    rounds = []
    for k in range(4):
        a = np.flatnonzero(d_tr != k)
        b = np.flatnonzero(d_tr == k)
        ik = [j for j in range(4) if j != k][-1]
        fit = a[d_tr[a] != ik]
        val = a[d_tr[a] == ik]
        init_fit = init_of(lin, d_all, fit)[0]
        init_val = init_of(lin, d_all, val)[0]
        init_a = init_of(lin, d_all, a)[0]
        params = make_params({}, args.threads)
        best = early_stopping_rounds(
            X[fit],
            y[fit],
            init_fit,
            X[val],
            y[val],
            init_val,
            params,
            args.max_rounds,
            args.patience,
        )
        rounds.append(best)
        ds = lgb.Dataset(X[a], y[a], init_score=init_a, free_raw_data=False)
        booster = lgb.train(params, ds, num_boost_round=max(1, best))
        gains += booster.feature_importance(importance_type="gain")
    gains /= 4.0

    order = np.argsort(-gains)
    ranked = [
        {"rank": int(i + 1), "feature": names[j], "gain": float(gains[j])}
        for i, j in enumerate(order)
    ]
    payload = {
        "model": "v8_feature_importance",
        "drop_cols": drop_cols,
        "feature_count": int(X.shape[1]),
        "rounds": rounds,
        "threads": args.threads,
        "features": ranked,
        "top20": [row["feature"] for row in ranked[:20]],
        "elapsed_seconds": time.time() - start,
    }
    write_json(args.out, payload)
    print("top features:")
    for row in ranked[:30]:
        print(f"{row['rank']:3d} {row['gain']:.1f} {row['feature']}")
    print(f"saved {args.out} in {payload['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
