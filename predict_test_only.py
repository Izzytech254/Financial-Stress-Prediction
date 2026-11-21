from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from features_cv import build_extra
from v6_common import (
    DATA_DIR,
    MZ_SLOPE,
    REPORT_DIR,
    TARGET,
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
from v6_residual import bag, make_params, scaled_rounds, seeds_for
from v8_screen import build_variant


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--from-report", required=True)
    ap.add_argument("--from-preds", required=True)
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--round-scale", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    start = time.time()
    src = json.loads(Path(args.from_report).read_text())
    z = np.load(args.from_preds)
    rounds = z["rounds"]
    if rounds.ndim != 2:
        raise ValueError(f"expected rounds shape (configs, folds), got {rounds.shape}")
    configs = src["configs"]
    if len(configs) != rounds.shape[0]:
        raise ValueError(f"config count {len(configs)} != rounds rows {rounds.shape[0]}")

    train, test, sample = load_data(args.data_dir)
    d_tr, d_te = snapshot_dates(train, test)
    y = train[TARGET].to_numpy()
    n = len(train)
    all_rows = pd.concat([train.drop(columns=TARGET), test], ignore_index=True)
    d_all = np.r_[d_tr, d_te]
    R = risk_factors(all_rows)
    lin = MZ_SLOPE * month_z_score(R, d_all)

    X, names, drop_cols, month_scale, extra_cols = build_variant(
        src["variant"], all_rows, d_all, R, None
    )
    groups = src.get("feature_groups", [])
    months = src.get("feature_months", [1, 2, 3, 4, 5, 6])
    if groups:
        X_extra, names_extra = build_extra(all_rows, d_all, groups, months)
        X = np.c_[X, X_extra].astype(np.float32)
        names = names + names_extra
    if X.shape[1] != len(names):
        raise ValueError(f"feature mismatch {X.shape[1]} != {len(names)}")

    tr = np.arange(n)
    te = np.arange(n, n + len(test))
    init_tr = init_of(lin, d_all, tr)[0]
    init_te = init_of(lin, d_all, te)[0]
    seeds = seeds_for(args.seeds)
    final_rounds = [
        scaled_rounds(rounds[c].tolist(), args.round_scale) for c in range(len(configs))
    ]
    print(f"final_rounds={final_rounds} seeds={len(seeds)} features={X.shape[1]}", flush=True)
    raws = []
    for c, cfg in enumerate(configs):
        raws.append(
            bag(
                X[:n], y, init_tr, X[n:], final_rounds[c],
                make_params(cfg, args.threads), seeds,
            )
        )
    p_test = per_date(
        probit_eta(expit(init_te + np.mean(raws, axis=0))), d_te
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = sample.copy()
    out["Target"] = p_test
    out.to_csv(out_path, index=False, float_format="%.17g")
    stats = validate_submission(out_path, sample)

    report_path = Path(args.report or str(REPORT_DIR / f"{out_path.stem}_test_only.json"))
    report = {
        "model": "predict_test_only",
        "source_report": str(args.from_report),
        "source_preds": str(args.from_preds),
        "variant": src["variant"],
        "feature_groups": groups,
        "feature_months": months,
        "feature_count": X.shape[1],
        "feature_names": names,
        "configs": configs,
        "final_rounds": final_rounds,
        "n_seeds": len(seeds),
        "seeds": list(seeds),
        "test": stats,
        "submission": str(out_path),
        "elapsed_seconds": time.time() - start,
    }
    write_json(report_path, report)
    print(f"wrote {out_path} mean={p_test.mean():.12f} min={p_test.min():.12g} max={p_test.max():.12g}", flush=True)


if __name__ == "__main__":
    main()
