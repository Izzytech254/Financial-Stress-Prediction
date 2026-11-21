"""v3: add model diversity to the v2 blend.

Screens XGBoost, HistGradientBoosting, tuned CatBoost, and LightGBM variants on
the v2 feature set, then evaluates honest blends against the current v2 final.
Writes a candidate submission to submission_v3.csv (does not overwrite
submission.csv).
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from features_v2 import build_v2_features, _customer_key
from solution import (
    CAT_COLS,
    DATA_DIR,
    ID,
    RANDOM_STATE,
    REPORT_DIR,
    ROOT,
    TARGET,
    cross_val_weight_auc,
    make_cat,
    make_hist,
    make_lgb,
    make_xgb,
    select_features,
)
from run_v2 import stack, summarize, paired, platt_cv_ll, _logit, _is_onehot

SEEDS = [42, 2026, 7]
N_FOLDS = 5
N_SELECT = 320
T0 = time.time()

XGB_TUNED = {
    "learning_rate": 0.021795741277819067,
    "max_depth": 9,
    "min_child_weight": 18.043311207136256,
    "subsample": 0.9425192044349383,
    "colsample_bytree": 0.5523068845866854,
    "reg_alpha": 0.002458603276328005,
    "reg_lambda": 1.8141523033666782,
    "gamma": 2.2007624686980067,
}
LGB_TUNED = {
    "learning_rate": 0.023686687710753017,
    "num_leaves": 70,
    "max_depth": 5,
    "min_child_samples": 93,
    "subsample": 0.777633994480005,
    "colsample_bytree": 0.731261142176991,
    "reg_alpha": 0.017654048052495083,
    "reg_lambda": 0.5209069329868288,
}
CAT_TUNED = {
    "learning_rate": 0.04331752496177168,
    "depth": 9,
    "l2_leaf_reg": 8.960785365368121,
    "subsample": 0.8795975452591109,
    "rsm": 0.5624074561769746,
    "random_strength": 0.3119890406724053,
}
HIST_TUNED = {
    "learning_rate": 0.045826461745188736,
    "max_leaf_nodes": 47,
    "min_samples_leaf": 69,
    "l2_regularization": 0.003980771770257045,
    "max_bins": 255,
}


def elapsed() -> float:
    return time.time() - T0


def log(msg: str) -> None:
    print(f"[{elapsed():7.1f}s] {msg}", flush=True)


def _last_row_idx(train: pd.DataFrame) -> np.ndarray:
    key = _customer_key(train)
    return pd.Series(np.arange(len(train))).groupby(key).max().values


def _metrics(y: np.ndarray, p: np.ndarray, last: np.ndarray) -> dict:
    return {
        "auc": float(roc_auc_score(y, p)),
        "logloss": float(log_loss(y, p)),
        "last_auc": float(roc_auc_score(y[last], p[last])),
        "last_logloss": float(log_loss(y[last], p[last])),
    }


def run_xgb_cv(
    X: np.ndarray,
    y: np.ndarray,
    Xte: np.ndarray,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    n_select: int = N_SELECT,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[float], List[float]]:
    oof = np.zeros(len(y), dtype=np.float64)
    test_pred = np.zeros(len(Xte), dtype=np.float64)
    aucs, lls = [], []
    for tr, va in folds:
        idx, _ = select_features(X[tr], y[tr], n_select, seed=RANDOM_STATE)
        model = make_xgb(XGB_TUNED, seed)
        model.fit(
            X[tr][:, idx],
            y[tr],
            eval_set=[(X[va][:, idx], y[va])],
            verbose=False,
        )
        p = model.predict_proba(X[va][:, idx])[:, 1]
        oof[va] = p
        test_pred += model.predict_proba(Xte[:, idx])[:, 1] / len(folds)
        aucs.append(float(roc_auc_score(y[va], p)))
        lls.append(float(log_loss(y[va], p)))
    return oof, test_pred, aucs, lls


def run_hist_cv(
    X: np.ndarray,
    y: np.ndarray,
    Xte: np.ndarray,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    n_select: int = N_SELECT,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[float], List[float]]:
    oof = np.zeros(len(y), dtype=np.float64)
    test_pred = np.zeros(len(Xte), dtype=np.float64)
    aucs, lls = [], []
    for tr, va in folds:
        idx, _ = select_features(X[tr], y[tr], n_select, seed=RANDOM_STATE)
        model = make_hist(HIST_TUNED, seed)
        model.fit(X[tr][:, idx], y[tr])
        p = model.predict_proba(X[va][:, idx])[:, 1]
        oof[va] = p
        test_pred += model.predict_proba(Xte[:, idx])[:, 1] / len(folds)
        aucs.append(float(roc_auc_score(y[va], p)))
        lls.append(float(log_loss(y[va], p)))
    return oof, test_pred, aucs, lls


def run_lgb_cv(
    X: np.ndarray,
    y: np.ndarray,
    Xte: np.ndarray,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    n_select: int = N_SELECT,
    seed: int = 42,
    params: dict | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[float], List[float]]:
    params = LGB_TUNED if params is None else params
    oof = np.zeros(len(y), dtype=np.float64)
    test_pred = np.zeros(len(Xte), dtype=np.float64)
    aucs, lls = [], []
    for tr, va in folds:
        idx, _ = select_features(X[tr], y[tr], n_select, seed=RANDOM_STATE)
        model = make_lgb(params, seed)
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


def run_cat_cv(
    X: np.ndarray,
    y: np.ndarray,
    Xte: np.ndarray,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    names: List[str],
    cat_train: np.ndarray,
    cat_test: np.ndarray,
    n_select: int = N_SELECT,
    seed: int = 42,
    params: dict | None = None,
    iterations: int = 700,
) -> Tuple[np.ndarray, np.ndarray, List[float], List[float]]:
    params = CAT_TUNED if params is None else params
    oof = np.zeros(len(y), dtype=np.float64)
    test_pred = np.zeros(len(Xte), dtype=np.float64)
    aucs, lls = [], []
    for tr, va in folds:
        idx, _ = select_features(X[tr], y[tr], n_select, seed=RANDOM_STATE)
        num_idx = [i for i in idx if not _is_onehot(names[i])]
        if len(num_idx) < 20:
            num_idx = [i for i in range(X.shape[1]) if not _is_onehot(names[i])][:n_select]
        cols = [names[i] for i in num_idx]
        Xtr_cb = pd.DataFrame(X[tr][:, num_idx], columns=cols)
        Xva_cb = pd.DataFrame(X[va][:, num_idx], columns=cols)
        Xte_cb = pd.DataFrame(Xte[:, num_idx], columns=cols)
        for j, c in enumerate(CAT_COLS):
            Xtr_cb[c] = cat_train[tr, j]
            Xva_cb[c] = cat_train[va, j]
            Xte_cb[c] = cat_test[:, j]
        cat_idx = list(range(len(num_idx), len(num_idx) + len(CAT_COLS)))
        model = make_cat(params, seed, iterations=iterations)
        model.fit(
            Xtr_cb,
            y[tr],
            eval_set=(Xva_cb, y[va]),
            use_best_model=True,
            cat_features=cat_idx,
        )
        p = model.predict_proba(Xva_cb)[:, 1]
        oof[va] = p
        test_pred += model.predict_proba(Xte_cb)[:, 1] / len(folds)
        aucs.append(float(roc_auc_score(y[va], p)))
        lls.append(float(log_loss(y[va], p)))
    return oof, test_pred, aucs, lls


def _fit_intercept(values: np.ndarray, y: np.ndarray, slope: float) -> float:
    def objective(b: float) -> float:
        return float(log_loss(y, np.clip(expit(slope * values + b), 1e-9, 1 - 1e-9)))

    result = minimize_scalar(objective, bounds=(-5.0, 5.0), method="bounded")
    return float(result.x)


def _cv_logloss(values: np.ndarray, y: np.ndarray, slope: float) -> float:
    skf = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    pred = np.zeros(len(y))
    for tr, va in skf.split(values, y):
        b = _fit_intercept(values[tr], y[tr], slope)
        pred[va] = expit(slope * values[va] + b)
    return float(log_loss(y, np.clip(pred, 1e-9, 1 - 1e-9)))


def _prior_shift(values_test: np.ndarray, slope: float, intercept: float, target_mean: float) -> float:
    return float(
        brentq(
            lambda d: float(expit(slope * values_test + intercept + d).mean()) - target_mean,
            -8.0,
            8.0,
        )
    )


def _blend_report(
    name: str,
    p_oof: np.ndarray,
    p_test: np.ndarray,
    y: np.ndarray,
    last: np.ndarray,
    base_oof: np.ndarray,
    folds: List[Tuple[np.ndarray, np.ndarray]],
) -> dict:
    p_oof = np.clip(p_oof, 1e-6, 1 - 1e-6)
    p_test = np.clip(p_test, 1e-6, 1 - 1e-6)
    m = _metrics(y, p_oof, last)
    da, dase, dl, dlse = paired(y, p_oof, base_oof, folds)
    l = _logit(p_oof)
    lt = _logit(p_test)
    lr = LogisticRegression(C=1e6, max_iter=1000, solver="lbfgs")
    lr.fit(l.reshape(-1, 1), y)
    slope = float(lr.coef_[0, 0])
    intercept = float(lr.intercept_[0])
    cv_ll = _cv_logloss(l, y, slope)
    delta = _prior_shift(lt, slope, intercept, 0.150)
    p_shift_oof = expit(slope * l + intercept + delta)
    p_shift_test = expit(slope * lt + intercept + delta)
    shift_ll = float(log_loss(y, np.clip(p_shift_oof, 1e-9, 1 - 1e-9)))
    return {
        "name": name,
        **m,
        "delta_auc_vs_v2": float(da),
        "delta_auc_se": float(dase),
        "delta_ll_vs_v2": float(dl),
        "delta_ll_se": float(dlse),
        "platt_slope": slope,
        "platt_intercept": intercept,
        "platt_cv_ll": cv_ll,
        "prior_shift": delta,
        "shift_ll_insample": shift_ll,
        "shift_auc": float(roc_auc_score(y, p_shift_oof)),
        "test_mean_raw": float(p_test.mean()),
        "test_mean_shift": float(p_shift_test.mean()),
        "_p_shift_test": p_shift_test,
        "_p_shift_oof": p_shift_oof,
    }


def main() -> None:
    os.makedirs(REPORT_DIR, exist_ok=True)
    train = pd.read_csv(os.path.join(DATA_DIR, "Train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "Test.csv"))
    sample = pd.read_csv(os.path.join(DATA_DIR, "SampleSubmission.csv"))
    last = _last_row_idx(train)
    log(f"Loaded Train={train.shape} Test={test.shape} last_rows={len(last)}")

    feats = build_v2_features(train, test)
    y = feats["y"]
    base = feats["base"]
    hist = feats["history"]
    cust_tr = feats["cust_tr"]
    cust_tt = feats["cust_tt"]

    sets: Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]] = {
        "combined": stack(base, [cust_tt, hist]),
        "base_history": stack(base, [hist]),
        "all": stack(base, [cust_tr, cust_tt, hist]),
    }
    log(
        f"Features: combined={sets['combined'][0].shape[1]} "
        f"base_history={sets['base_history'][0].shape[1]} "
        f"all={sets['all'][0].shape[1]}"
    )

    zfold = np.load(os.path.join(REPORT_DIR, "folds_v2.npz"))
    folds_by_seed = {
        s: [(zfold[f"tr_{s}_{i}"], zfold[f"va_{s}_{i}"]) for i in range(N_FOLDS)]
        for s in SEEDS
    }
    folds = folds_by_seed[42]

    z2 = np.load(os.path.join(REPORT_DIR, "v2_predictions.npz"))
    zclean = np.load(os.path.join(REPORT_DIR, "v2_clean_predictions.npz"))
    zbase = np.load(os.path.join(REPORT_DIR, "preds_2seed.npz"))
    p_v2 = zclean["oof_final"]
    t_v2 = zclean["test_final"]
    p_lgb3 = z2["oof_lgb_v2"]
    t_lgb3 = z2["test_lgb_v2"]
    p_cat = z2["oof_cat_v2"]
    t_cat = z2["test_cat_v2"]
    v2m = _metrics(y, p_v2, last)
    log(f"v2 final OOF {v2m}")

    preds: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
        "lgb_v2_3seed": (p_lgb3, t_lgb3),
        "cat_v2": (p_cat, t_cat),
        "xgb_base": (zbase["oof_xgb"], zbase["test_xgb"]),
        "hist_base": (zbase["oof_hist"], zbase["test_hist"]),
        "lgb_base": (zbase["oof_lgb"], zbase["test_lgb"]),
    }
    screening: Dict[str, dict] = {}

    def save_preds() -> None:
        np.savez_compressed(
            os.path.join(REPORT_DIR, "v3_screen_predictions.npz"),
            y=y,
            last=last,
            **{f"oof_{k}": v[0] for k, v in preds.items()},
            **{f"test_{k}": v[1] for k, v in preds.items()},
        )
        log("Saved screening predictions")

    X, Xte, names = sets["combined"]
    cat_train = train[CAT_COLS].astype(str).values
    cat_test = test[CAT_COLS].astype(str).values

    log("Screen XGB v2 ...")
    p, t, a, l = run_xgb_cv(X, y, Xte, folds, N_SELECT, 42)
    preds["xgb_v2"] = (p, t)
    screening["xgb_v2"] = _metrics(y, p, last)
    log(f"xgb_v2 {screening['xgb_v2']}")
    save_preds()

    log("Screen Hist v2 ...")
    p, t, a, l = run_hist_cv(X, y, Xte, folds, N_SELECT, 42)
    preds["hist_v2"] = (p, t)
    screening["hist_v2"] = _metrics(y, p, last)
    log(f"hist_v2 {screening['hist_v2']}")
    save_preds()

    log("Screen CatBoost tuned v2 ...")
    p, t, a, l = run_cat_cv(X, y, Xte, folds, names, cat_train, cat_test, N_SELECT, 42, CAT_TUNED, 700)
    preds["cat_v2_tuned"] = (p, t)
    screening["cat_v2_tuned"] = _metrics(y, p, last)
    log(f"cat_v2_tuned {screening['cat_v2_tuned']}")
    save_preds()

    log("Screen LGB v2 N_SELECT=450 ...")
    p, t, a, l = run_lgb_cv(X, y, Xte, folds, 450, 42)
    preds["lgb_v2_450"] = (p, t)
    screening["lgb_v2_450"] = _metrics(y, p, last)
    log(f"lgb_v2_450 {screening['lgb_v2_450']}")
    save_preds()

    Xbh, Xtebh, _ = sets["base_history"]
    log("Screen LGB base+history ...")
    p, t, a, l = run_lgb_cv(Xbh, y, Xtebh, folds, N_SELECT, 42)
    preds["lgb_base_history"] = (p, t)
    screening["lgb_base_history"] = _metrics(y, p, last)
    log(f"lgb_base_history {screening['lgb_base_history']}")
    save_preds()

    Xall, Xteall, names_all = sets["all"]
    log("Screen LGB all (both customer sets + history) ...")
    p, t, a, l = run_lgb_cv(Xall, y, Xteall, folds, N_SELECT, 42)
    preds["lgb_all"] = (p, t)
    screening["lgb_all"] = _metrics(y, p, last)
    log(f"lgb_all {screening['lgb_all']}")
    save_preds()

    # Candidate blends.
    def mean_logit(keys: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        P = np.clip(np.column_stack([preds[k][0] for k in keys]), 1e-6, 1 - 1e-6)
        T = np.clip(np.column_stack([preds[k][1] for k in keys]), 1e-6, 1 - 1e-6)
        return expit(_logit(P).mean(axis=1)), expit(_logit(T).mean(axis=1))

    candidates: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    candidates["lgb3_cat"] = mean_logit(["lgb_v2_3seed", "cat_v2"])
    candidates["lgb3_cat_xgb"] = mean_logit(["lgb_v2_3seed", "cat_v2", "xgb_v2"])
    candidates["lgb3_cat_xgb_hist"] = mean_logit(
        ["lgb_v2_3seed", "cat_v2", "xgb_v2", "hist_v2"]
    )
    candidates["lgb3_cat_xgb_hist_cattuned"] = mean_logit(
        ["lgb_v2_3seed", "cat_v2", "xgb_v2", "hist_v2", "cat_v2_tuned"]
    )
    candidates["all_new"] = mean_logit(
        [
            "lgb_v2_3seed",
            "cat_v2",
            "xgb_v2",
            "hist_v2",
            "cat_v2_tuned",
            "lgb_v2_450",
            "lgb_base_history",
            "lgb_all",
        ]
    )
    candidates["all_plus_baselines"] = mean_logit(
        [
            "lgb_v2_3seed",
            "cat_v2",
            "xgb_v2",
            "hist_v2",
            "cat_v2_tuned",
            "lgb_v2_450",
            "lgb_base_history",
            "lgb_all",
            "xgb_base",
            "hist_base",
            "lgb_base",
        ]
    )

    # Weighted blend over the strongest diverse set.
    keys = ["lgb_v2_3seed", "cat_v2", "xgb_v2", "hist_v2", "cat_v2_tuned", "lgb_v2_450", "lgb_base_history", "lgb_all"]
    P = np.clip(np.column_stack([preds[k][0] for k in keys]), 1e-6, 1 - 1e-6)
    T = np.clip(np.column_stack([preds[k][1] for k in keys]), 1e-6, 1 - 1e-6)
    w_cv_auc, w = cross_val_weight_auc(P, y, seed=RANDOM_STATE)
    candidates["weighted_all"] = (
        np.clip(P @ w, 1e-6, 1 - 1e-6),
        np.clip(T @ w, 1e-6, 1 - 1e-6),
    )
    log(f"weighted_all CV AUC estimate {w_cv_auc:.6f} weights {dict(zip(keys, w.round(4).tolist()))}")

    # Logistic stack on the same set, honest CV.
    L = _logit(P)
    LT = _logit(T)
    skf = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    stack_oof = np.zeros(len(y))
    for tr, va in skf.split(L, y):
        meta = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
        meta.fit(L[tr], y[tr])
        stack_oof[va] = meta.predict_proba(L[va])[:, 1]
    meta = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    meta.fit(L, y)
    stack_test = meta.predict_proba(LT)[:, 1]
    candidates["logistic_stack_all"] = (stack_oof, stack_test)

    reports = {}
    for name, (p, t) in candidates.items():
        reports[name] = _blend_report(name, p, t, y, last, p_v2, folds)
        r = reports[name]
        log(
            f"Blend {name}: AUC={r['auc']:.6f} LL={r['logloss']:.6f} "
            f"lastAUC={r['last_auc']:.6f} lastLL={r['last_logloss']:.6f} "
            f"dAUC={r['delta_auc_vs_v2']:+.6f}+/-{r['delta_auc_se']:.6f} "
            f"dLL={r['delta_ll_vs_v2']:+.6f}+/-{r['delta_ll_se']:.6f} "
            f"plattCV={r['platt_cv_ll']:.6f}"
        )

    # Rank by a 60/40 metric proxy using the honest Platt CV LL and require no AUC regression.
    def proxy(r: dict) -> float:
        return 0.6 * r["platt_cv_ll"] + 0.4 * (1.0 - r["auc"])

    best_name = min(
        (k for k in reports if reports[k]["auc"] >= v2m["auc"] - 1e-6),
        key=lambda k: proxy(reports[k]),
    )
    best = reports[best_name]
    log(f"Best blend by 60/40 proxy: {best_name} proxy={proxy(best):.6f}")

    # Write candidate submission with prior shift.
    p_test = np.clip(best["_p_shift_test"], 1e-6, 1 - 1e-6)
    sub = pd.DataFrame({ID: test[ID].values, "Target": p_test})
    sub = sample[[ID]].merge(sub, on=ID, how="left")
    assert sub["Target"].notna().all()
    assert (sub[ID].values == sample[ID].values).all()
    out_path = os.path.join(ROOT, "submission_v3.csv")
    sub.to_csv(out_path, index=False, float_format="%.8f")
    log(f"Wrote {out_path} mean={p_test.mean():.6f} min={p_test.min():.6f} max={p_test.max():.6f}")

    # Clean reports (drop private arrays).
    clean_reports = {}
    for k, r in reports.items():
        clean_reports[k] = {kk: vv for kk, vv in r.items() if not kk.startswith("_")}

    report = {
        "v2_final": _metrics(y, p_v2, last),
        "screening": screening,
        "blends": clean_reports,
        "best_blend": best_name,
        "best_proxy": proxy(best),
        "best_shift": {
            "platt_slope": best["platt_slope"],
            "platt_intercept": best["platt_intercept"],
            "prior_shift": best["prior_shift"],
            "test_mean_raw": best["test_mean_raw"],
            "test_mean_shift": best["test_mean_shift"],
            "shift_ll_insample": best["shift_ll_insample"],
            "shift_auc": best["shift_auc"],
        },
        "submission_file": "submission_v3.csv",
        "runtime_seconds": float(elapsed()),
    }
    with open(os.path.join(REPORT_DIR, "improve_v3.json"), "w") as f:
        json.dump(report, f, indent=2)

    lines = [
        "# v3 model diversity",
        "",
        f"- v2 final OOF: AUC `{report['v2_final']['auc']:.6f}`, LL `{report['v2_final']['logloss']:.6f}` (calibrated)",
        f"- v2 final Platt CV LL reference: `0.240912`",
        f"- v2 last-row: AUC `{report['v2_final']['last_auc']:.6f}`, LL `{report['v2_final']['last_logloss']:.6f}`",
        "",
        "## Screening",
        "",
        "| model | AUC | LogLoss | last AUC | last LogLoss |",
        "|---|---:|---:|---:|---:|",
    ]
    for k, r in screening.items():
        lines.append(
            f"| {k} | {r['auc']:.6f} | {r['logloss']:.6f} | {r['last_auc']:.6f} | {r['last_logloss']:.6f} |"
        )
    lines += [
        "",
        "## Blends",
        "",
        "| blend | AUC | LogLoss | last AUC | last LogLoss | dAUC vs v2 | dLL vs v2 | Platt CV LL |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for k, r in clean_reports.items():
        lines.append(
            f"| {k} | {r['auc']:.6f} | {r['logloss']:.6f} | {r['last_auc']:.6f} | {r['last_logloss']:.6f} | "
            f"{r['delta_auc_vs_v2']:+.6f} | {r['delta_ll_vs_v2']:+.6f} | {r['platt_cv_ll']:.6f} |"
        )
    lines += [
        "",
        f"Best by 60/40 proxy: `{best_name}` (proxy `{proxy(best):.6f}`).",
        f"Candidate submission: `submission_v3.csv` (prior-shifted test mean `{best['test_mean_shift']:.6f}`).",
        "",
    ]
    with open(os.path.join(REPORT_DIR, "improve_v3.md"), "w") as f:
        f.write("\n".join(lines))

    log(f"Done. Total runtime {elapsed():.1f}s")


if __name__ == "__main__":
    main()
