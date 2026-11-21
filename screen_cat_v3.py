"""Screen CatBoost tuned vs default params on the v3 feature set.

Trains each config with seed 42, frozen folds, N_SELECT=500, 700 iterations.
Saves OOF/test predictions and a compact comparison report.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from features_v2 import build_v2_features
from features_v3 import build_history_features_v3
from improve_v3 import CAT_TUNED, _last_row_idx, _metrics, log, run_cat_cv
from run_v2 import CAT_PARAMS, platt_cv_ll, stack
from solution import CAT_COLS, DATA_DIR, REPORT_DIR, TARGET

N_SELECT = 500
N_FOLDS = 5
SEED = 42
ITERATIONS = 700
T0 = time.time()

CONFIGS = {
    "CAT_TUNED_700": CAT_TUNED,
    "CAT_PARAMS_700": CAT_PARAMS,
}


def main() -> None:
    train = pd.read_csv(os.path.join(DATA_DIR, "Train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "Test.csv"))
    last = _last_row_idx(train)

    feats = build_v2_features(train, test)
    y = feats["y"]
    hist3 = build_history_features_v3(train, test)
    X, Xte, names = stack(feats["base"], [feats["cust_tt"], hist3])
    log(f"X={X.shape}")

    zfold = np.load(os.path.join(REPORT_DIR, "folds_v2.npz"))
    folds = [(zfold[f"tr_{SEED}_{i}"], zfold[f"va_{SEED}_{i}"]) for i in range(N_FOLDS)]
    cat_train = train[CAT_COLS].astype(str).values
    cat_test = test[CAT_COLS].astype(str).values

    report = {
        "n_select": N_SELECT,
        "iterations": ITERATIONS,
        "seed": SEED,
        "configs": {},
    }
    for name, params in CONFIGS.items():
        log(f"Training {name} ...")
        p, t, aucs, lls = run_cat_cv(
            X,
            y,
            Xte,
            folds,
            names,
            cat_train,
            cat_test,
            N_SELECT,
            SEED,
            params,
            ITERATIONS,
        )
        metrics = _metrics(y, p, last)
        cv_ll, _ = platt_cv_ll(p, y)
        proxy = 0.6 * cv_ll + 0.4 * (1.0 - metrics["auc"])
        np.savez_compressed(
            os.path.join(REPORT_DIR, f"cat_screen_{name}.npz"),
            y=y,
            last=last,
            oof=p,
            test=t,
            fold_aucs=np.asarray(aucs),
            fold_loglosses=np.asarray(lls),
        )
        report["configs"][name] = {
            **metrics,
            "platt_cv_ll": float(cv_ll),
            "proxy": float(proxy),
            "fold_aucs": [float(v) for v in aucs],
            "fold_loglosses": [float(v) for v in lls],
            "params": params,
            "predictions": f"reports/cat_screen_{name}.npz",
        }
        log(
            f"{name}: AUC={metrics['auc']:.6f} LL={metrics['logloss']:.6f} "
            f"lastAUC={metrics['last_auc']:.6f} lastLL={metrics['last_logloss']:.6f} "
            f"plattCV={cv_ll:.6f} proxy={proxy:.6f}"
        )

    with open(os.path.join(REPORT_DIR, "cat_screen.json"), "w") as f:
        json.dump(report, f, indent=2)

    lines = [
        "# CatBoost v3 screen",
        "",
        f"- Frozen seed `{SEED}`, `N_SELECT={N_SELECT}`, `iterations={ITERATIONS}`",
        "",
        "| config | AUC | LogLoss | last AUC | last LL | Platt CV LL | 60/40 proxy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, r in report["configs"].items():
        lines.append(
            f"| {name} | {r['auc']:.6f} | {r['logloss']:.6f} | "
            f"{r['last_auc']:.6f} | {r['last_logloss']:.6f} | "
            f"{r['platt_cv_ll']:.6f} | {r['proxy']:.6f} |"
        )
    lines += ["", f"Runtime: `{time.time() - T0:.1f}s`.", ""]
    with open(os.path.join(REPORT_DIR, "cat_screen.md"), "w") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))


if __name__ == "__main__":
    main()
