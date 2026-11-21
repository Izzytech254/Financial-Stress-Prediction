from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold

from features_cv import build_extra
from frozen_cv import evaluate_folds, load_folds, save_predictions, write_manifest
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
from v6_residual import bag, early_stopping_rounds, make_params, scaled_rounds, seeds_for
from v8_screen import build_variant, parse_configs


def inner_split(y: np.ndarray, d: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    strata = y * 4 + d
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
    fit, val = next(skf.split(np.arange(len(y)), strata))
    return fit, val


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--variant", default="drop_m5_all")
    ap.add_argument("--config-json", action="append", default=[])
    ap.add_argument("--include-base", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--max-rounds", type=int, default=600)
    ap.add_argument("--patience", type=int, default=60)
    ap.add_argument("--round-scale", type=float, default=1.0)
    ap.add_argument("--init-preds", default=None)
    ap.add_argument("--feature-groups", default="")
    ap.add_argument("--feature-months", default="1,2,3,4,5,6")
    ap.add_argument("--folds", default=str(REPORT_DIR / "folds_v2.npz"))
    ap.add_argument("--tag", default=None)
    ap.add_argument("--report", default=None)
    ap.add_argument("--preds", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--write-manifest", action="store_true")
    args = ap.parse_args()

    start = time.time()
    if args.write_manifest:
        write_manifest(args.folds)
    folds = load_folds(args.folds)
    seeds = seeds_for(args.seeds)
    configs = parse_configs(args.config_json, args.include_base)
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
        if len(lin) != len(all_rows):
            raise ValueError(f"init preds length {len(lin)} != {len(all_rows)}")

    X, names, drop_cols, month_scale, extra_cols = build_variant(
        args.variant, all_rows, d_all, R, None
    )
    feature_groups = [g for g in args.feature_groups.split(",") if g]
    feature_months = [int(m) for m in args.feature_months.split(",") if m]
    if feature_groups:
        X_extra, names_extra = build_extra(all_rows, d_all, feature_groups, feature_months)
        X = np.c_[X, X_extra].astype(np.float32)
        names = names + names_extra
    tag = args.tag or f"baseline_cv_{args.variant}_s{args.seeds}"
    report_path = Path(args.report or REPORT_DIR / f"{tag}.json")
    preds_path = Path(args.preds or REPORT_DIR / f"{tag}_predictions.npz")
    out_path = Path(args.out) if args.out else None

    print(
        f"model=baseline_cv variant={args.variant} tag={tag} "
        f"features={X.shape[1]} configs={len(configs)} seeds={len(seeds)} "
        f"folds={len(folds)} scale={month_scale}",
        flush=True,
    )

    repeat_ids = sorted({f[0] for f in folds})
    repeat_pos = {seed: j for j, seed in enumerate(repeat_ids)}
    oof_repeats = np.empty((len(repeat_ids), n), dtype=np.float64)
    rounds = np.empty((len(configs), len(folds)), dtype=np.int32)
    for i, (repeat, fold, a, b) in enumerate(folds):
        fit_rel, val_rel = inner_split(y[a], d_tr[a], 10000 + repeat * 10 + fold)
        fit = a[fit_rel]
        val = a[val_rel]
        init_fit = init_of(lin, d_all, fit)[0]
        init_val = init_of(lin, d_all, val)[0]
        init_a = init_of(lin, d_all, a)[0]
        init_b = init_of(lin, d_all, b)[0]

        raws = []
        for c, cfg in enumerate(configs):
            params = make_params(cfg, args.threads)
            best = early_stopping_rounds(
                X[fit], y[fit], init_fit, X[val], y[val], init_val,
                params, args.max_rounds, args.patience,
            )
            rounds[c, i] = best
            raws.append(
                bag(
                    X[a], y[a], init_a, X[b],
                    max(1, int(round(args.round_scale * best))),
                    params, seeds,
                )
            )
        combined = expit(init_b + np.mean(raws, axis=0))
        p_fold = per_date(probit_eta(combined), d_tr[b])
        if not np.isfinite(p_fold).all() or p_fold.min() < 0 or p_fold.max() > 1:
            raise ValueError(
                f"invalid fold preds repeat={repeat} fold={fold} "
                f"min={p_fold.min()} max={p_fold.max()} n_invalid={int((~np.isfinite(p_fold)).sum())}"
            )
        oof_repeats[repeat_pos[repeat], b] = p_fold
        if fold == 0:
            print(
                f"repeat={repeat} fold={fold} round={i + 1}/{len(folds)} "
                f"rounds={[int(rounds[c, i]) for c in range(len(configs))]}",
                flush=True,
            )

    oof = np.mean(oof_repeats, axis=0)
    if not np.isfinite(oof).all() or oof.min() < 0 or oof.max() > 1:
        bad = np.flatnonzero(~np.isfinite(oof) | (oof < 0) | (oof > 1))
        raise ValueError(f"invalid oof rows={bad[:10].tolist()} n={len(bad)}")
    metrics = evaluate_folds(y, oof, folds)
    repeat_metrics = []
    for j, seed in enumerate(repeat_ids):
        sf = [f for f in folds if f[0] == seed]
        repeat_metrics.append(evaluate_folds(y, oof_repeats[j], sf)["overall"])

    p_test = None
    stats = None
    if args.test:
        tr = np.arange(n)
        te = np.arange(n, n + len(test))
        init_tr = init_of(lin, d_all, tr)[0]
        init_te = init_of(lin, d_all, te)[0]
        final_rounds = [scaled_rounds(rounds[c].tolist(), args.round_scale) for c in range(len(configs))]
        final_raws = []
        for c, cfg in enumerate(configs):
            final_raws.append(
                bag(X[:n], y, init_tr, X[n:], final_rounds[c],
                    make_params(cfg, args.threads), seeds)
            )
        p_test = per_date(
            probit_eta(expit(init_te + np.mean(final_raws, axis=0))), d_te
        )
        if out_path is None:
            out_path = Path(f"submission_{tag}.csv")
        out = sample.copy()
        out["Target"] = p_test
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False, float_format="%.17g")
        stats = validate_submission(out_path, sample)

    payload = {
        "y": y,
        "oof": oof,
        "oof_repeats": oof_repeats,
        "d_tr": d_tr,
        "d_te": d_te,
        "rounds": rounds,
        "s_mz": s_mz,
        "feature_names": np.asarray(names),
    }
    if p_test is not None:
        payload["test"] = p_test
    save_predictions(preds_path, payload)

    report = {
        "model": "baseline_cv",
        "variant": args.variant,
        "tag": tag,
        "n_seeds": len(seeds),
        "seeds": list(seeds),
        "configs": configs,
        "feature_count": X.shape[1],
        "feature_names": names,
        "drop_cols": list(drop_cols),
        "extra_cols": list(extra_cols),
        "feature_groups": feature_groups,
        "feature_months": feature_months,
        "month_scale": month_scale,
        "folds": "reports/folds_v2.npz seeds 42/2026/7",
        "validation": metrics,
        "repeat_metrics": repeat_metrics,
        "final_rounds": final_rounds if args.test else None,
        "test": stats,
        "submission": str(out_path) if out_path else None,
        "preds": str(preds_path),
        "elapsed_seconds": time.time() - start,
    }
    write_json(report_path, report)
    print(
        f"oof proxy={metrics['overall']['proxy']:.9f} "
        f"ll={metrics['overall']['log_loss']:.9f} "
        f"auc={metrics['overall']['auc']:.9f} report={report_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
