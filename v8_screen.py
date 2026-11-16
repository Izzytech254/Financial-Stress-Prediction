from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

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
    validate_submission,
    write_json,
)
from v6_residual import (
    CONFIGS,
    bag,
    early_stopping_rounds,
    make_params,
    scaled_rounds,
    seeds_for,
)
from v7_common import RANK3_10, extra_transform, feature_names

VARIANTS = [
    "base",
    "month_z",
    "month_none",
    "drop_m5_all",
    "drop_target_m5",
    "drop_collinear",
    "add_extras",
    "add_extras_nom5",
    "add_extras_rank2",
    "add_extras_rank4",
    "add_extras_global",
    "add_extras_drop_target",
    "add_extras_drop_m5_all",
    "add_extras_rank3_8",
    "add_extras_rank3_6",
    "top100",
    "top150",
]


def parse_configs(raw: list[str], include_base: bool) -> list[dict]:
    configs = [dict(c) for c in CONFIGS] if include_base else []
    for item in raw:
        cfg = json.loads(item)
        if not isinstance(cfg, dict):
            raise ValueError(f"config must be an object: {item}")
        configs.append(cfg)
    if not configs:
        raise ValueError("at least one config is required")
    return configs


def load_importance(path: str | Path) -> dict[str, float]:
    data = json.loads(Path(path).read_text())
    return {row["feature"]: float(row["gain"]) for row in data["features"]}


def build_variant(
    variant: str,
    all_rows: pd.DataFrame,
    d_all: np.ndarray,
    R: np.ndarray,
    importance: dict[str, float] | None,
):
    base_drop = [c for c in all_rows.columns if c.startswith("m5_deposit")]
    month_scale = "rank"
    drop_cols = list(base_drop)
    extra_cols: list[str] = []
    extra_kind = "rank3"
    top_k: int | None = None

    if variant == "base":
        pass
    elif variant == "month_z":
        month_scale = "z"
    elif variant == "month_none":
        month_scale = "none"
    elif variant == "drop_m5_all":
        drop_cols = [c for c in all_rows.columns if c.startswith("m5_")]
    elif variant == "drop_target_m5":
        drop_cols = base_drop + [
            "m5_withdraw_highest_amount",
            "m5_merchantpay_highest_amount",
        ]
    elif variant == "drop_collinear":
        drop_cols = base_drop + [
            "m4_transfer_from_bank_volume",
            "m4_transfer_from_bank_banks",
        ]
    elif variant == "add_extras":
        extra_cols = list(RANK3_10)
    elif variant == "add_extras_nom5":
        extra_cols = [c for c in RANK3_10 if c != "m5_deposit_agents"]
    elif variant == "add_extras_rank2":
        extra_cols = list(RANK3_10)
        extra_kind = "rank2"
    elif variant == "add_extras_rank4":
        extra_cols = list(RANK3_10)
        extra_kind = "rank4"
    elif variant == "add_extras_global":
        extra_cols = list(RANK3_10)
        extra_kind = "global_rank"
    elif variant == "add_extras_drop_target":
        drop_cols = base_drop + [
            "m5_withdraw_highest_amount",
            "m5_merchantpay_highest_amount",
        ]
        extra_cols = list(RANK3_10)
    elif variant == "add_extras_drop_m5_all":
        drop_cols = [c for c in all_rows.columns if c.startswith("m5_")]
        extra_cols = list(RANK3_10)
    elif variant == "add_extras_rank3_8":
        extra_cols = list(RANK3_10[:8])
    elif variant == "add_extras_rank3_6":
        extra_cols = list(RANK3_10[:6])
    elif variant.startswith("top"):
        top_k = int(variant[3:])
        if importance is None:
            raise ValueError(f"{variant} requires --importance")
    else:
        raise ValueError(f"unknown variant: {variant}")

    X = features(all_rows, R, drop_cols)
    if month_scale != "none":
        X = month_scaled(X, d_all, month_scale)
    names = feature_names(all_rows, drop_cols)

    if top_k is not None:
        ranked = sorted(
            (importance.get(n, 0.0), i, n) for i, n in enumerate(names)
        )
        keep = {i for _, i, _ in ranked[-top_k:]}
        idx = [i for i in range(len(names)) if i in keep]
        X = X[:, idx]
        names = [names[i] for i in idx]

    if extra_cols:
        Z = extra_transform(all_rows, extra_cols, d_all, extra_kind)
        X = np.c_[X, Z].astype(np.float32)
        names = names + [f"E{i}" for i in range(len(extra_cols))]

    return X, names, drop_cols, month_scale, extra_cols


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--variant", default="base", choices=VARIANTS)
    ap.add_argument("--config-json", action="append", default=[])
    ap.add_argument(
        "--include-base", action=argparse.BooleanOptionalAction, default=True
    )
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--round-scale", type=float, default=1.0)
    ap.add_argument("--max-rounds", type=int, default=600)
    ap.add_argument("--patience", type=int, default=60)
    ap.add_argument("--importance", default=None)
    ap.add_argument("--init-preds", default=None)
    ap.add_argument("--stack-preds", action="append", default=[])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--report", default=None)
    ap.add_argument("--preds", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()

    start = time.time()
    seeds = seeds_for(args.seeds)
    configs = parse_configs(args.config_json, args.include_base)
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
    if args.init_preds:
        z = np.load(args.init_preds)
        lin = z["lin_all"]
        assert len(lin) == len(all_rows), (len(lin), len(all_rows))
    X, names, drop_cols, month_scale, extra_cols = build_variant(
        args.variant, all_rows, d_all, R, importance
    )
    if args.stack_preds:
        stack_cols = []
        for i, path in enumerate(args.stack_preds):
            z = np.load(path)
            stack_cols.append(z["lin_all"])
            names.append(f"S{i}")
        X = np.c_[X, np.column_stack(stack_cols)].astype(np.float32)

    tag = args.tag or args.variant
    report_path = Path(args.report or REPORT_DIR / f"v8_screen_{tag}.json")
    preds_path = Path(args.preds) if args.preds else None
    out_path = Path(args.out) if args.out else None

    print(
        f"variant={args.variant} tag={tag} features={X.shape[1]} "
        f"configs={len(configs)} seeds={len(seeds)} scale={month_scale}",
        flush=True,
    )

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
            f"month {k} composite={score_k:.9f} ll={ll_k:.9f} "
            f"auc={auc_k:.9f} rounds={[r[-1] for r in rounds]}",
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

    final_rounds = [scaled_rounds(r, args.round_scale) for r in rounds]
    p_test = None
    stats = None
    if args.test:
        tr = np.arange(n)
        te = np.arange(n, n + len(test))
        init_tr = init_of(lin, d_all, tr)[0]
        init_te = init_of(lin, d_all, te)[0]
        final_raws = []
        for c, cfg in enumerate(configs):
            params = make_params(cfg, args.threads)
            final_raws.append(
                bag(X[:n], y, init_tr, X[n:], final_rounds[c], params, seeds)
            )
        p_test = per_date(
            probit_eta(expit(init_te + np.mean(final_raws, axis=0))), d_te
        )
        if out_path is None:
            out_path = Path(f"submission_v8_{tag}.csv")
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
        "model": "v8_screen",
        "variant": args.variant,
        "tag": tag,
        "init_preds": args.init_preds,
        "stack_preds": args.stack_preds,
        "month_scale": month_scale,
        "drop_cols": drop_cols,
        "extra_cols": extra_cols,
        "feature_count": int(X.shape[1]),
        "configs": configs,
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
        f"residual LOMO composite={score_res:.9f} ll={ll_res:.9f} "
        f"auc={auc_res:.9f} elapsed={elapsed:.1f}s",
        flush=True,
    )
    print(f"report={report_path}", flush=True)


if __name__ == "__main__":
    main()
