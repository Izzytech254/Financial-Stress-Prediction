from __future__ import annotations

import json

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression

from calibration_lab import exact_proxy, load_prediction, recalibrate_prior
from frozen_cv import evaluate_folds, load_folds, paired_delta
from v6_common import DATA_DIR, EPS, load_data, write_json


def weights_to_p(z: np.ndarray, w: np.ndarray, d: np.ndarray) -> np.ndarray:
    return recalibrate_prior(expit(z @ w), d)


def fit_weights(z: np.ndarray, y: np.ndarray, d: np.ndarray, starts: list[np.ndarray]) -> np.ndarray:
    def objective(w):
        return -exact_proxy(y, weights_to_p(z, w, d))[0]

    bounds = [(0.0, 1.0)] * z.shape[1]
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    best = None
    for start in starts:
        res = minimize(objective, start, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 300, "ftol": 1e-9})
        if res.success and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        raise RuntimeError("weight optimization failed")
    return best.x / best.x.sum()


def nested_blend(z_rep, z_test, y, d_tr, d_te, names, starts):
    z_rep = np.asarray(z_rep).transpose(0, 2, 1)
    nested = np.empty_like(z_rep[:, :, 0])
    for j in range(z_rep.shape[0]):
        idx = [i for i in range(z_rep.shape[0]) if i != j]
        z = z_rep[idx].reshape(-1, z_rep.shape[2])
        yy = np.concatenate([y] * len(idx))
        dd = np.concatenate([d_tr] * len(idx))
        w = fit_weights(z, yy, dd, starts)
        nested[j] = weights_to_p(z_rep[j], w, d_tr)
    w = fit_weights(z_rep.reshape(-1, z_rep.shape[2]), np.concatenate([y] * z_rep.shape[0]), np.concatenate([d_tr] * z_rep.shape[0]), starts)
    return nested.mean(axis=0), weights_to_p(z_test, w, d_te), w


def nested_stacker(z_rep, z_test, y, d_tr, d_te, C):
    z_rep = np.asarray(z_rep).transpose(0, 2, 1)
    nested = np.empty_like(z_rep[:, :, 0])
    for j in range(z_rep.shape[0]):
        idx = [i for i in range(z_rep.shape[0]) if i != j]
        z = z_rep[idx].reshape(-1, z_rep.shape[2])
        yy = np.concatenate([y] * len(idx))
        model = LogisticRegression(C=C, solver="lbfgs", max_iter=2000)
        model.fit(z, yy)
        nested[j] = recalibrate_prior(model.predict_proba(z_rep[j])[:, 1], d_tr)
    model = LogisticRegression(C=C, solver="lbfgs", max_iter=2000)
    model.fit(z_rep.reshape(-1, z_rep.shape[2]), np.concatenate([y] * z_rep.shape[0]))
    return nested.mean(axis=0), recalibrate_prior(model.predict_proba(z_test)[:, 1], d_te), model


def main():
    paths = {
        "baseline": "reports/baseline_cv_drop_m5_all_s50_predictions.npz",
        "trend_conc": "reports/feat_trend_conc_s50_predictions.npz",
        "tuned": "reports/tuned_trend_conc_s10_predictions.npz",
        "final_tuned": "reports/final_tuned_lgb100_predictions.npz",
        "cat": "reports/final_cat_s5_all_predictions.npz",
        "xgb": "reports/final_xgb_s5_all_predictions.npz",
    }
    preds = {k: load_prediction(v) for k, v in paths.items()}
    y = preds["tuned"]["y"]
    d_tr = preds["tuned"]["d_tr"]
    d_te = preds["tuned"]["d_te"]
    names = list(paths)
    oof_rep = np.stack([preds[n]["oof_repeats"][j] for n in names for j in range(len(preds[n]["oof_repeats"]))]).reshape(len(names), 3, -1).transpose(1, 0, 2)
    test_p = np.stack([preds[n]["test"] for n in names], axis=1)
    z_rep = logit(np.clip(oof_rep, EPS, 1 - EPS))
    z_test = logit(np.clip(test_p, EPS, 1 - EPS))
    train, test_df, sample = load_data(DATA_DIR)
    n = len(y)
    starts = []
    starts.append(np.array([0, 0, 0.5, 0.5, 0, 0], dtype=float))
    starts.append(np.array([0, 0.172521, 0.604154, 0.223325, 0, 0], dtype=float))
    starts.append(np.full(len(names), 1 / len(names), dtype=float))
    folds = load_folds()
    base_rep = expit(0.5 * logit(np.clip(oof_rep[:, 2, :], EPS, 1 - EPS)) + 0.5 * logit(np.clip(oof_rep[:, 3, :], EPS, 1 - EPS)))
    base_oof = base_rep.mean(axis=0)
    base_test = recalibrate_prior(expit(0.5 * z_test[:, 2] + 0.5 * z_test[:, 3]), d_te)
    base_eval = evaluate_folds(y, base_oof, folds)["overall"]

    results = {}
    predictions = {"y": y, "d_tr": d_tr, "d_te": d_te, "B_oof": base_oof, "B_test": base_test}

    def add(label, oof, test, fitting, params):
        ev = evaluate_folds(y, oof, folds)["overall"]
        delta = paired_delta(base_oof, oof, y, folds)
        by_date = []
        for d in np.unique(d_tr):
            m = d_tr == d
            p0, l0, a0 = exact_proxy(y[m], base_oof[m])
            p1, l1, a1 = exact_proxy(y[m], oof[m])
            by_date.append({"date": int(d), "delta_proxy": p1 - p0, "delta_ll": l1 - l0, "delta_auc": a1 - a0})
        results[label] = {
            "proxy": ev["proxy"], "log_loss": ev["log_loss"], "auc": ev["auc"],
            "fold_std": evaluate_folds(y, oof, folds)["std_proxy"],
            "delta_proxy": ev["proxy"] - base_eval["proxy"],
            "delta_ll": ev["log_loss"] - base_eval["log_loss"],
            "delta_auc": ev["auc"] - base_eval["auc"],
            "paired_proxy_delta": delta["delta"], "paired_proxy_se": delta["se"], "accept": delta["accept"],
            "by_date": by_date, "fitting": fitting, "params": params,
        }
        predictions[label + "_oof"] = oof
        predictions[label + "_test"] = test

    add("B", base_oof, base_test, "identity", {})

    for subset in (names, [n for n in names if n not in ("xgb",)], [n for n in names if n not in ("cat", "xgb")], ["tuned", "final_tuned", "baseline", "cat"]):
        cols = [names.index(n) for n in subset]
        ss = []
        for full in starts:
            v = full[cols]
            if v.sum() == 0:
                v = np.ones(len(cols)) / len(cols)
            ss.append(v / v.sum())
        label = "blend_" + "_".join(subset)
        oof, test, w = nested_blend(z_rep[:, cols, :], z_test[:, cols], y, d_tr, d_te, subset, ss)
        add(label, oof, test, "nonnegative_logit_blend", {"models": subset, "weights": {k: float(v) for k, v in zip(subset, w)}})

    for C in (0.01, 0.1, 1.0):
        label = f"stack_C{C:g}"
        oof, test, model = nested_stacker(z_rep, z_test, y, d_tr, d_te, C)
        add(label, oof, test, "logistic_stacker", {"C": C, "coefficients": {k: float(v) for k, v in zip(names, model.coef_[0])}, "intercept": float(model.intercept_[0])})

    full_oof, full_test, full_w = nested_blend(z_rep, z_test, y, d_tr, d_te, names, starts)
    loo = {}
    for drop in names:
        subset = [n for n in names if n != drop]
        cols = [names.index(n) for n in subset]
        ss = []
        for full in starts:
            v = full[cols]
            v = v / v.sum() if v.sum() else np.ones(len(cols)) / len(cols)
            ss.append(v)
        oof, test, w = nested_blend(z_rep[:, cols, :], z_test[:, cols], y, d_tr, d_te, subset, ss)
        ev = evaluate_folds(y, oof, folds)["overall"]
        loo[drop] = {"proxy": ev["proxy"], "delta_vs_full": ev["proxy"] - evaluate_folds(y, full_oof, folds)["overall"]["proxy"], "weights": {k: float(v) for k, v in zip(subset, w)}}

    write_json("reports/blend_suite.json", {"base": base_eval, "models": names, "results": results, "leave_one_out": loo})
    np.savez_compressed("reports/blend_predictions.npz", **predictions)
    ordered = sorted(results.items(), key=lambda kv: kv[1]["proxy"], reverse=True)
    print(json.dumps({"base": base_eval, "ranked": [(k, round(v["proxy"], 9), round(v["delta_proxy"], 9), round(v["paired_proxy_se"], 9), v["accept"], v["params"]) for k, v in ordered], "loo": loo}, indent=2))


if __name__ == "__main__":
    main()
