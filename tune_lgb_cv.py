from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from scipy.special import expit

from baseline_cv import inner_split
from features_cv import build_extra
from frozen_cv import evaluate_folds, load_folds, save_predictions
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
from v6_residual import BASE, bag, early_stopping_rounds, make_params, scaled_rounds, seeds_for
from v8_screen import build_variant

optuna.logging.set_verbosity(optuna.logging.WARNING)


def predict_folds(X, y, d_tr, d_all, lin, folds, params, max_rounds, patience, seeds):
    n = len(y)
    oof_repeats = {}
    rounds = []
    for repeat, fold, a, b in folds:
        fit_rel, val_rel = inner_split(y[a], d_tr[a], 10000 + repeat * 10 + fold)
        fit, val = a[fit_rel], a[val_rel]
        init_fit = init_of(lin, d_all, fit)[0]
        init_val = init_of(lin, d_all, val)[0]
        init_a = init_of(lin, d_all, a)[0]
        init_b = init_of(lin, d_all, b)[0]
        best = early_stopping_rounds(
            X[fit], y[fit], init_fit, X[val], y[val], init_val,
            params, max_rounds, patience,
        )
        rounds.append(best)
        raw = bag(X[a], y[a], init_a, X[b], max(1, best), params, seeds)
        p = per_date(probit_eta(expit(init_b + raw)), d_tr[b])
        oof_repeats.setdefault(repeat, np.empty(n, dtype=np.float64))
        oof_repeats[repeat][b] = p
    return oof_repeats, rounds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--variant", default="drop_m5_all")
    ap.add_argument("--feature-groups", default="")
    ap.add_argument("--feature-months", default="1,2,3,4,5,6")
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--search-seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--max-rounds", type=int, default=600)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--tag", default="tuned_lgb_cv")
    ap.add_argument("--report", default=None)
    ap.add_argument("--preds", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()

    start = time.time()
    folds = load_folds()
    train, test, sample = load_data(args.data_dir)
    d_tr, d_te = snapshot_dates(train, test)
    y = train[TARGET].to_numpy()
    n = len(train)
    all_rows = pd.concat([train.drop(columns=TARGET), test], ignore_index=True)
    d_all = np.r_[d_tr, d_te]
    R = risk_factors(all_rows)
    lin = MZ_SLOPE * month_z_score(R, d_all)
    X, names, drop_cols, month_scale, extra_cols = build_variant(
        args.variant, all_rows, d_all, R, None
    )
    groups = [g for g in args.feature_groups.split(",") if g]
    months = [int(m) for m in args.feature_months.split(",") if m]
    if groups:
        Xx, nx = build_extra(all_rows, d_all, groups, months)
        X = np.c_[X, Xx].astype(np.float32)
        names = names + nx

    search_folds = [f for f in folds if f[0] == args.search_seed]

    def objective(trial: optuna.Trial) -> float:
        params = {
            **BASE,
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.04, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 5, 31),
            "max_depth": trial.suggest_int("max_depth", 2, 7),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 50, 500),
            "lambda_l2": trial.suggest_float("lambda_l2", 1.0, 100.0, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": 1,
            "num_threads": args.threads,
        }
        oof_repeats, _ = predict_folds(
            X, y, d_tr, d_all, lin, search_folds, params,
            args.max_rounds, args.patience, seeds_for(5),
        )
        p = next(iter(oof_repeats.values()))
        from sklearn.metrics import log_loss, roc_auc_score

        ll = log_loss(y, p)
        auc = roc_auc_score(y, p)
        return 0.6 * (1 - ll / 0.595) + 0.4 * auc

    def progress(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if (trial.number + 1) % 10 == 0:
            print(
                f"trial={trial.number + 1}/{args.trials} best={study.best_value:.9f}",
                flush=True,
            )

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=20261118),
    )
    study.optimize(objective, n_trials=args.trials, callbacks=[progress])
    best_params = {**BASE, **study.best_params, "num_threads": args.threads}
    print("best", json.dumps(study.best_params, sort_keys=True), flush=True)

    seeds = seeds_for(args.seeds)
    oof_repeats, round_list = predict_folds(
        X, y, d_tr, d_all, lin, folds, best_params,
        args.max_rounds, args.patience, seeds,
    )
    repeat_arrays = [oof_repeats[k] for k in sorted(oof_repeats)]
    oof = np.mean(repeat_arrays, axis=0)
    metrics = {}
    for j, seed in enumerate(sorted(oof_repeats)):
        sf = [f for f in folds if f[0] == seed]
        from frozen_cv import evaluate_folds

        metrics[str(seed)] = evaluate_folds(y, repeat_arrays[j], sf)["overall"]
    from frozen_cv import evaluate_folds

    overall = evaluate_folds(y, oof, folds)

    p_test = None
    if args.test:
        tr = np.arange(n)
        te = np.arange(n, n + len(test))
        init_tr = init_of(lin, d_all, tr)[0]
        init_te = init_of(lin, d_all, te)[0]
        rounds = scaled_rounds(round_list, 1.0)
        raw = bag(X[:n], y, init_tr, X[n:], rounds, best_params, seeds)
        p_test = per_date(probit_eta(expit(init_te + raw)), d_te)
        out = sample.copy()
        out["Target"] = p_test
        out_path = Path(args.out or f"submission_{args.tag}.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False, float_format="%.17g")
        stats = validate_submission(out_path, sample)
    else:
        stats = None
        out_path = None

    preds_path = Path(args.preds or REPORT_DIR / f"{args.tag}_predictions.npz")
    payload = {
        "y": y, "oof": oof,
        "oof_repeats": np.asarray(repeat_arrays),
        "d_tr": d_tr, "d_te": d_te,
        "s_mz": lin,
        "feature_names": np.asarray(names),
    }
    if p_test is not None:
        payload["test"] = p_test
    save_predictions(preds_path, payload)
    report_path = Path(args.report or REPORT_DIR / f"{args.tag}.json")
    report = {
        "model": "tuned_lgb_cv",
        "tag": args.tag,
        "trial_count": args.trials,
        "search_seed": args.search_seed,
        "best_params": study.best_params,
        "best_search_proxy": study.best_value,
        "seeds": list(seeds),
        "feature_groups": groups,
        "feature_months": months,
        "feature_count": X.shape[1],
        "feature_names": names,
        "validation": overall,
        "repeat_metrics": metrics,
        "test": stats,
        "submission": str(out_path) if out_path else None,
        "preds": str(preds_path),
        "elapsed_seconds": time.time() - start,
    }
    write_json(report_path, report)
    print(f"oof proxy={overall['overall']['proxy']:.9f} report={report_path}", flush=True)


if __name__ == "__main__":
    main()
