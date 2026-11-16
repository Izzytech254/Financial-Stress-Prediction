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

BASE = {
    "objective": "binary",
    "learning_rate": 0.02,
    "num_leaves": 7,
    "max_depth": 3,
    "min_data_in_leaf": 300,
    "lambda_l2": 50.0,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "seed": 909,
    "deterministic": True,
    "force_row_wise": True,
    "verbose": -1,
}
CONFIGS = [
    {},
    {"num_leaves": 15, "max_depth": 4},
    {"min_data_in_leaf": 100},
    {"learning_rate": 0.01},
    {"learning_rate": 0.05},
]


def seeds_for(n: int) -> tuple[int, ...]:
    assert n >= 5, n
    return (909, 11, 22, 33, 44) + tuple(range(1001, 1001 + n - 5))


def make_params(cfg: dict, threads: int) -> dict:
    return {**BASE, **cfg, "num_threads": threads}


def early_stopping_rounds(
    X: np.ndarray,
    y: np.ndarray,
    init: np.ndarray,
    Xv: np.ndarray,
    yv: np.ndarray,
    iv: np.ndarray,
    params: dict,
    max_rounds: int,
    patience: int,
) -> int:
    ds = lgb.Dataset(X, y, init_score=init, free_raw_data=False)
    dv = lgb.Dataset(Xv, yv, init_score=iv, reference=ds)
    booster = lgb.train(
        params,
        ds,
        num_boost_round=max_rounds,
        valid_sets=[dv],
        callbacks=[lgb.early_stopping(patience, verbose=False)],
    )
    return max(int(booster.best_iteration), 1)


def bag(
    X: np.ndarray,
    y: np.ndarray,
    init: np.ndarray,
    Xp: np.ndarray,
    rounds: int,
    params: dict,
    seeds: tuple[int, ...],
) -> np.ndarray:
    ds = lgb.Dataset(X, y, init_score=init, free_raw_data=False)
    raws = []
    for seed in seeds:
        p = {**params, "seed": seed}
        booster = lgb.train(p, ds, num_boost_round=rounds)
        raws.append(booster.predict(Xp, raw_score=True))
    return np.mean(raws, axis=0)


def scaled_rounds(rounds: list[int], scale: float) -> int:
    return max(1, int(round(scale * float(np.median(rounds)))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--out", default="submission_v6_residual.csv")
    ap.add_argument("--report", default=str(REPORT_DIR / "v6_residual.json"))
    ap.add_argument("--preds", default=str(REPORT_DIR / "v6_residual_predictions.npz"))
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--month-scale", default="rank", choices=["none", "rank", "z"])
    ap.add_argument("--round-scale", type=float, default=1.0)
    ap.add_argument("--max-rounds", type=int, default=600)
    ap.add_argument("--patience", type=int, default=60)
    ap.add_argument("--drop-m5-deposit", action="store_true")
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
    drop_cols = (
        [c for c in all_rows.columns if c.startswith("m5_deposit")]
        if args.drop_m5_deposit
        else []
    )
    X = features(all_rows, R, drop_cols)
    if args.month_scale != "none":
        X = month_scaled(X, d_all, args.month_scale)

    p_struct = structural_oof_mz(s_mz, d_tr)
    score_struct, ll_struct, auc_struct = composite(y, p_struct)
    print(
        f"structural LOMO composite={score_struct:.9f} ll={ll_struct:.9f} auc={auc_struct:.9f}",
        flush=True,
    )
    print(f"features={X.shape[1]} drop_cols={drop_cols} seeds={len(seeds)}", flush=True)

    rounds: list[list[int]] = [[] for _ in CONFIGS]
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
        for c, cfg in enumerate(CONFIGS):
            params = make_params(cfg, args.threads)
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
            rounds[c].append(best)
            raws.append(
                bag(
                    X[a],
                    y[a],
                    init_a,
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
            f"month {k} composite={score_k:.9f} ll={ll_k:.9f} auc={auc_k:.9f} rounds={[r[-1] for r in rounds]}",
            flush=True,
        )

    score_res, ll_res, auc_res = composite(y, p_res)
    gain = score_res - score_struct
    per_month = []
    for k in range(4):
        m = d_tr == k
        s_res, ll_k, auc_k = composite(y[m], p_res[m])
        s_str, ll_s, auc_s = composite(y[m], p_struct[m])
        per_month.append(
            {
                "snapshot": k,
                "residual_composite": s_res,
                "structural_composite": s_str,
                "gain": s_res - s_str,
                "residual_log_loss": ll_k,
                "structural_log_loss": ll_s,
                "residual_auc": auc_k,
                "structural_auc": auc_s,
            }
        )
    print(
        f"residual LOMO composite={score_res:.9f} ll={ll_res:.9f} auc={auc_res:.9f} gain={gain:+.9f}",
        flush=True,
    )

    tr = np.arange(n)
    te = np.arange(n, n + len(test))
    init_tr = init_of(lin, d_all, tr)[0]
    init_te = init_of(lin, d_all, te)[0]
    final_rounds = [scaled_rounds(r, args.round_scale) for r in rounds]
    final_raws = []
    for c, cfg in enumerate(CONFIGS):
        params = make_params(cfg, args.threads)
        final_raws.append(
            bag(X[:n], y, init_tr, X[n:], final_rounds[c], params, seeds)
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
    )

    elapsed = time.time() - start
    payload = {
        "model": "v6_residual",
        "seeds": list(seeds),
        "n_seeds": len(seeds),
        "threads": args.threads,
        "month_scale": args.month_scale,
        "round_scale": args.round_scale,
        "drop_m5_deposit": bool(args.drop_m5_deposit),
        "drop_cols": drop_cols,
        "feature_count": int(X.shape[1]),
        "configs": CONFIGS,
        "rounds": rounds,
        "final_rounds": final_rounds,
        "validation": {
            "residual": {
                "composite": score_res,
                "log_loss": ll_res,
                "auc": auc_res,
            },
            "structural": {
                "composite": score_struct,
                "log_loss": ll_struct,
                "auc": auc_struct,
            },
            "gain": gain,
        },
        "per_month": per_month,
        "test": {
            **stats,
            "snapshot_means": [
                float(p_test[d_te == k].mean()) for k in (4, 5, 6)
            ],
        },
        "elapsed_seconds": elapsed,
        "submission": str(Path(args.out).resolve()),
    }
    write_json(args.report, payload)

    lines = [
        "# v6 LightGBM Residual",
        "",
        f"- Seeds: `{len(seeds)}`",
        f"- Month scaling: `{args.month_scale}`",
        f"- Round scale: `{args.round_scale}`",
        f"- Drop m5 deposit columns: `{bool(args.drop_m5_deposit)}`",
        f"- Features: `{X.shape[1]}`",
        f"- Structural LOMO: `{score_struct:.9f}`",
        f"- Residual LOMO: `{score_res:.9f}`",
        f"- Gain: `{gain:+.9f}`",
        f"- Residual LogLoss: `{ll_res:.9f}`",
        f"- Residual AUC: `{auc_res:.9f}`",
        "",
        "| Snapshot | Residual | Structural | Gain |",
        "|---:|---:|---:|---:|",
    ]
    for row in per_month:
        lines.append(
            f"| {row['snapshot']} | {row['residual_composite']:.9f} | "
            f"{row['structural_composite']:.9f} | {row['gain']:+.9f} |"
        )
    lines.extend(
        [
            "",
            f"- Final rounds: `{final_rounds}`",
            f"- Test mean: `{stats['mean']:.12f}`",
            f"- Test snapshot means: `{[round(float(p_test[d_te == k].mean()), 12) for k in (4, 5, 6)]}`",
            f"- Elapsed seconds: `{elapsed:.1f}`",
            f"- Submission: `{args.out}`",
        ]
    )
    Path(args.report).with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
