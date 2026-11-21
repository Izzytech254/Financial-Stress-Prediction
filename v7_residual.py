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
    REPORT_DIR,
    TARGET,
    composite,
    features,
    init_of,
    load_data,
    month_scaled,
    per_date,
    probit_eta,
    snapshot_dates,
    structural_oof_mz,
    validate_submission,
    write_json,
)
from v7_common import (
    EXTRA_SETS,
    build_extra_features,
    extra_lin,
    fit_extra_gamma,
    structural_oof_extra,
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
    ap.add_argument("--variant", default="extra4", choices=sorted(EXTRA_SETS))
    ap.add_argument(
        "--transform",
        default="auto",
        choices=[
            "auto",
            "global_rank",
            "rank",
            "rank1.5",
            "rank2",
            "rank2.5",
            "rank3",
            "rank4",
            "rank_sq",
            "sqrt",
            "raw",
            "log1p",
            "z_cap",
        ],
    )
    ap.add_argument("--fit-mode", default="fixed", choices=["fixed", "free"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--report", default=None)
    ap.add_argument("--preds", default=None)
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--month-scale", default="rank", choices=["none", "rank", "z"])
    ap.add_argument("--round-scale", type=float, default=1.0)
    ap.add_argument("--max-rounds", type=int, default=600)
    ap.add_argument("--patience", type=int, default=60)
    ap.add_argument("--drop-m5-deposit", action="store_true")
    args = ap.parse_args()

    transform = args.transform
    if transform == "auto":
        if args.variant.startswith("rank3"):
            transform = "rank3"
        elif args.variant == "nom5":
            transform = "sqrt"
        else:
            transform = "global_rank"
    out_path = Path(args.out or f"submission_v7_residual_{args.variant}.csv")
    report_path = Path(args.report or REPORT_DIR / f"v7_residual_{args.variant}.json")
    preds_path = Path(
        args.preds or REPORT_DIR / f"v7_residual_{args.variant}_predictions.npz"
    )

    start = time.time()
    seeds = seeds_for(args.seeds)
    train, test, sample = load_data(args.data_dir)
    d_tr, d_te = snapshot_dates(train, test)
    y = train[TARGET].to_numpy()
    n = len(train)
    all_rows, d_all, R, s_mz, Z = build_extra_features(
        train, test, d_tr, d_te, args.variant, transform
    )
    columns = EXTRA_SETS[args.variant]

    drop_cols = (
        [c for c in all_rows.columns if c.startswith("m5_deposit")]
        if args.drop_m5_deposit
        else []
    )
    X = features(all_rows, R, drop_cols)
    if args.month_scale != "none":
        X = month_scaled(X, d_all, args.month_scale)

    p_struct_extra, _, gammas_oof, _ = structural_oof_extra(
        s_mz, Z, y, d_tr, fit_mode=args.fit_mode
    )
    score_struct_extra, ll_struct_extra, auc_struct_extra = composite(y, p_struct_extra)
    p_struct_base = structural_oof_mz(s_mz, d_tr)
    score_struct_base, ll_struct_base, auc_struct_base = composite(y, p_struct_base)
    print(
        f"structural base LOMO={score_struct_base:.9f} ll={ll_struct_base:.9f} auc={auc_struct_base:.9f}",
        flush=True,
    )
    print(
        f"structural extra LOMO={score_struct_extra:.9f} ll={ll_struct_extra:.9f} auc={auc_struct_extra:.9f}",
        flush=True,
    )
    print(
        f"features={X.shape[1]} drop_cols={drop_cols} seeds={len(seeds)} "
        f"variant={args.variant} transform={transform} fit_mode={args.fit_mode}",
        flush=True,
    )

    rounds: list[list[int]] = [[] for _ in CONFIGS]
    fold_gammas: list[np.ndarray] = []
    p_res = np.empty(n, dtype=np.float64)
    for k in range(4):
        a = np.flatnonzero(d_tr != k)
        b = np.flatnonzero(d_tr == k)
        ik = [j for j in range(4) if j != k][-1]
        fit = a[d_tr[a] != ik]
        val = a[d_tr[a] == ik]
        fit_mask = np.isin(np.arange(n), fit)
        a_mask = np.isin(np.arange(n), a)

        gamma_fit, _, _ = fit_extra_gamma(
            s_mz[:n], Z[:n], y, d_tr, fit_mask, fit_mode=args.fit_mode
        )
        lin_fit = extra_lin(s_mz, Z, gamma_fit)
        init_fit = init_of(lin_fit, d_all, fit)[0]
        init_val = init_of(lin_fit, d_all, val)[0]

        gamma_a, _, _ = fit_extra_gamma(
            s_mz[:n], Z[:n], y, d_tr, a_mask, fit_mode=args.fit_mode
        )
        lin_a = extra_lin(s_mz, Z, gamma_a)
        init_a = init_of(lin_a, d_all, a)[0]
        init_b = init_of(lin_a, d_all, b)[0]
        fold_gammas.append(gamma_a)

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
        score_sk, ll_sk, auc_sk = composite(y[b], p_struct_extra[b])
        print(
            f"month {k} residual={score_k:.9f} extra={score_sk:.9f} "
            f"gain={score_k - score_sk:+.9f} ll={ll_k:.9f} auc={auc_k:.9f} "
            f"rounds={[r[-1] for r in rounds]}",
            flush=True,
        )

    score_res, ll_res, auc_res = composite(y, p_res)
    gain_extra = score_res - score_struct_extra
    gain_base = score_res - score_struct_base
    per_month = []
    for k in range(4):
        m = d_tr == k
        s_res, ll_k, auc_k = composite(y[m], p_res[m])
        s_ext, ll_e, auc_e = composite(y[m], p_struct_extra[m])
        s_bas, ll_b, auc_b = composite(y[m], p_struct_base[m])
        per_month.append(
            {
                "snapshot": k,
                "residual_composite": s_res,
                "extra_composite": s_ext,
                "base_composite": s_bas,
                "gain_vs_extra": s_res - s_ext,
                "gain_vs_base": s_res - s_bas,
                "residual_log_loss": ll_k,
                "extra_log_loss": ll_e,
                "base_log_loss": ll_b,
                "residual_auc": auc_k,
                "extra_auc": auc_e,
                "base_auc": auc_b,
            }
        )
    print(
        f"residual LOMO={score_res:.9f} ll={ll_res:.9f} auc={auc_res:.9f} "
        f"gain_vs_extra={gain_extra:+.9f} gain_vs_base={gain_base:+.9f}",
        flush=True,
    )

    tr = np.arange(n)
    te = np.arange(n, n + len(test))
    gamma_full, _, _ = fit_extra_gamma(
        s_mz[:n], Z[:n], y, d_tr, np.ones(n, dtype=bool), fit_mode=args.fit_mode
    )
    lin_full = extra_lin(s_mz, Z, gamma_full)
    init_tr = init_of(lin_full, d_all, tr)[0]
    init_te = init_of(lin_full, d_all, te)[0]
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, float_format="%.17g")
    stats = validate_submission(out_path, sample)

    preds_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        preds_path,
        y=y,
        oof=p_res,
        test=p_test,
        structural_extra_oof=p_struct_extra,
        structural_base_oof=p_struct_base,
        d_tr=d_tr,
        d_te=d_te,
        rounds=np.asarray(rounds),
        s_mz=s_mz,
        Z=Z,
        gammas_oof=gammas_oof,
        fold_gammas=np.asarray(fold_gammas),
        gamma_full=gamma_full,
        extra_columns=np.asarray(columns, dtype=object),
    )

    elapsed = time.time() - start
    payload = {
        "model": "v7_residual_extra",
        "variant": args.variant,
        "transform": transform,
        "fit_mode": args.fit_mode,
        "extra_columns": columns,
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
            "structural_extra": {
                "composite": score_struct_extra,
                "log_loss": ll_struct_extra,
                "auc": auc_struct_extra,
            },
            "structural_base": {
                "composite": score_struct_base,
                "log_loss": ll_struct_base,
                "auc": auc_struct_base,
            },
            "gain_vs_extra": gain_extra,
            "gain_vs_base": gain_base,
        },
        "per_month": per_month,
        "test": {
            **stats,
            "snapshot_means": [
                float(p_test[d_te == k].mean()) for k in (4, 5, 6)
            ],
        },
        "elapsed_seconds": elapsed,
        "submission": str(out_path.resolve()),
    }
    write_json(report_path, payload)

    lines = [
        "# v7 LightGBM Residual on Extra Structural Init",
        "",
        f"- Variant: `{args.variant}`",
        f"- Transform: `{transform}`",
        f"- Fit mode: `{args.fit_mode}`",
        f"- Seeds: `{len(seeds)}`",
        f"- Month scaling: `{args.month_scale}`",
        f"- Round scale: `{args.round_scale}`",
        f"- Drop m5 deposit columns: `{bool(args.drop_m5_deposit)}`",
        f"- Features: `{X.shape[1]}`",
        f"- Structural base LOMO: `{score_struct_base:.9f}`",
        f"- Structural extra LOMO: `{score_struct_extra:.9f}`",
        f"- Residual LOMO: `{score_res:.9f}`",
        f"- Gain vs extra: `{gain_extra:+.9f}`",
        f"- Gain vs base: `{gain_base:+.9f}`",
        f"- Residual LogLoss: `{ll_res:.9f}`",
        f"- Residual AUC: `{auc_res:.9f}`",
        "",
        "| Snapshot | Residual | Extra | Base | Gain vs Extra |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in per_month:
        lines.append(
            f"| {row['snapshot']} | {row['residual_composite']:.9f} | "
            f"{row['extra_composite']:.9f} | {row['base_composite']:.9f} | "
            f"{row['gain_vs_extra']:+.9f} |"
        )
    lines.extend(
        [
            "",
            f"- Final rounds: `{final_rounds}`",
            f"- Test mean: `{stats['mean']:.12f}`",
            f"- Test snapshot means: `{[round(float(p_test[d_te == k].mean()), 12) for k in (4, 5, 6)]}`",
            f"- Elapsed seconds: `{elapsed:.1f}`",
            f"- Submission: `{out_path}`",
        ]
    )
    report_path.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
