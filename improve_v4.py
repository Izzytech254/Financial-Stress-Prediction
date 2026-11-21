"""v4: test additional history features and LightGBM hyperparameters."""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from features_v2 import build_v2_features
from features_v3 import build_history_features_v3
from improve_v3 import (
    LGB_TUNED,
    _last_row_idx,
    _metrics,
    log,
    run_lgb_cv,
)
from run_v2 import stack, paired, summarize
from solution import DATA_DIR, RANDOM_STATE, REPORT_DIR, ROOT, make_lgb, select_features

N_FOLDS = 5
N_SELECT = 320
SEED = 42
T0 = time.time()

LGB_DEEP = {
    **LGB_TUNED,
    "learning_rate": 0.02,
    "num_leaves": 127,
    "max_depth": 7,
    "min_child_samples": 50,
    "colsample_bytree": 0.6,
    "reg_lambda": 1.0,
}
LGB_DEEPER = {
    **LGB_TUNED,
    "learning_rate": 0.015,
    "num_leaves": 255,
    "max_depth": 9,
    "min_child_samples": 30,
    "colsample_bytree": 0.5,
    "reg_lambda": 2.0,
}
LGB_LOWLR = {
    **LGB_TUNED,
    "learning_rate": 0.01,
}


def elapsed() -> float:
    return time.time() - T0


def run_lgb(
    X: np.ndarray,
    y: np.ndarray,
    Xte: np.ndarray,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    n_select: int = N_SELECT,
    seed: int = SEED,
    params: dict | None = None,
    n_estimators: int = 900,
) -> Tuple[np.ndarray, np.ndarray, List[float], List[float]]:
    params = LGB_TUNED if params is None else params
    oof = np.zeros(len(y), dtype=np.float64)
    test_pred = np.zeros(len(Xte), dtype=np.float64)
    aucs, lls = [], []
    for tr, va in folds:
        idx, _ = select_features(X[tr], y[tr], n_select, seed=RANDOM_STATE)
        model = make_lgb(params, seed, n_estimators=n_estimators)
        model.fit(
            X[tr][:, idx],
            y[tr],
            eval_set=[(X[va][:, idx], y[va])],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        p = model.predict_proba(X[va][:, idx])[:, 1]
        oof[va] = p
        test_pred += model.predict_proba(Xte[:, idx])[:, 1] / len(folds)
        aucs.append(float(roc_auc_score(y[va], p)))
        lls.append(float(log_loss(y[va], p)))
    return oof, test_pred, aucs, lls


def main() -> None:
    train = pd.read_csv(os.path.join(DATA_DIR, "Train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "Test.csv"))
    last = _last_row_idx(train)
    log(f"Loaded Train={train.shape} Test={test.shape}")

    feats = build_v2_features(train, test)
    y = feats["y"]
    hist3 = build_history_features_v3(train, test)
    X2, Xte2, _ = stack(feats["base"], [feats["cust_tt"], feats["history"]])
    X3, Xte3, _ = stack(feats["base"], [feats["cust_tt"], hist3])
    log(f"X2={X2.shape} X3={X3.shape} hist2={feats['history'][0].shape[1]} hist3={hist3[0].shape[1]}")

    zfold = np.load(os.path.join(REPORT_DIR, "folds_v2.npz"))
    folds = [(zfold[f"tr_{SEED}_{i}"], zfold[f"va_{SEED}_{i}"]) for i in range(N_FOLDS)]

    preds: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    results: Dict[str, dict] = {}

    def save() -> None:
        np.savez_compressed(
            os.path.join(REPORT_DIR, "v4_predictions.npz"),
            y=y,
            last=last,
            **{f"oof_{k}": v[0] for k, v in preds.items()},
            **{f"test_{k}": v[1] for k, v in preds.items()},
        )
        log("Saved v4 predictions")

    def run(name: str, X, Xte, n_select=320, params=None, n_estimators=900) -> dict:
        log(f"Run {name} ...")
        p, t, a, l = run_lgb(X, y, Xte, folds, n_select, SEED, params, n_estimators)
        preds[name] = (p, t)
        results[name] = _metrics(y, p, last)
        log(f"{name} {results[name]}")
        save()
        return results[name]

    ref = run("ref_lgb", X2, Xte2)
    v3 = run("v3_lgb", X3, Xte3)
    run("v3_lgb_400", X3, Xte3, n_select=400)
    run("v2_deep", X2, Xte2, params=LGB_DEEP)
    run("v2_deeper", X2, Xte2, params=LGB_DEEPER)
    run("v2_lowlr", X2, Xte2, params=LGB_LOWLR, n_estimators=2000)

    # Apply the best parameter set to the v3 features.
    best_params_name = max(
        ["ref_lgb", "v2_deep", "v2_deeper", "v2_lowlr"],
        key=lambda k: results[k]["auc"],
    )
    best_params = {
        "ref_lgb": LGB_TUNED,
        "v2_deep": LGB_DEEP,
        "v2_deeper": LGB_DEEPER,
        "v2_lowlr": LGB_LOWLR,
    }[best_params_name]
    best_n = 2000 if best_params_name == "v2_lowlr" else 900
    run("v3_bestparams", X3, Xte3, params=best_params, n_estimators=best_n)

    # Compare each to the reference on the same folds.
    report = {"reference": "ref_lgb", "results": {}, "deltas": {}}
    for k, r in results.items():
        report["results"][k] = r
        if k != "ref_lgb":
            da, dase, dl, dlse = paired(y, preds[k][0], preds["ref_lgb"][0], folds)
            report["deltas"][k] = {
                "delta_auc": float(da),
                "delta_auc_se": float(dase),
                "delta_ll": float(dl),
                "delta_ll_se": float(dlse),
            }
            log(
                f"Delta {k} vs ref: dAUC={da:+.6f}+/-{dase:.6f} "
                f"dLL={dl:+.6f}+/-{dlse:.6f}"
            )

    with open(os.path.join(REPORT_DIR, "improve_v4.json"), "w") as f:
        json.dump(report, f, indent=2)

    lines = [
        "# v4 feature / parameter screen",
        "",
        "| run | AUC | LogLoss | last AUC | last LogLoss | dAUC vs ref | dLL vs ref |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for k, r in results.items():
        d = report["deltas"].get(k)
        da = "n/a" if d is None else f"{d['delta_auc']:+.6f}"
        dl = "n/a" if d is None else f"{d['delta_ll']:+.6f}"
        lines.append(
            f"| {k} | {r['auc']:.6f} | {r['logloss']:.6f} | {r['last_auc']:.6f} | "
            f"{r['last_logloss']:.6f} | {da} | {dl} |"
        )
    lines += ["", f"Best params on v2 features: `{best_params_name}`.", ""]
    with open(os.path.join(REPORT_DIR, "improve_v4.md"), "w") as f:
        f.write("\n".join(lines))

    log(f"Done. Total runtime {elapsed():.1f}s")


if __name__ == "__main__":
    main()
