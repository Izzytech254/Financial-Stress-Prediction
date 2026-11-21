from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import beta as beta_pdf, betainc, expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

from calibration_lab import exact_proxy, load_prediction, recalibrate_prior
from frozen_cv import evaluate_folds, load_folds, paired_delta
from v6_common import DATA_DIR, EPS, TARGET, load_data, write_json


def nested_predict(repeats, test, y, d_tr, d_te, fit_apply):
    nested = np.empty_like(repeats)
    for j in range(len(repeats)):
        idx = [i for i in range(len(repeats)) if i != j]
        rp = np.concatenate([repeats[i] for i in idx])
        ry = np.concatenate([y] * len(idx))
        rd = np.concatenate([d_tr] * len(idx))
        nested[j] = recalibrate_prior(fit_apply(rp, ry, rd, repeats[j]), d_tr)
    final = recalibrate_prior(
        fit_apply(np.concatenate(repeats), np.concatenate([y] * len(repeats)), np.concatenate([d_tr] * len(repeats)), test),
        d_te,
    )
    return nested.mean(axis=0), final


def temperature_fit(rp, ry, rd, rq):
    z = logit(np.clip(rp, EPS, 1 - EPS))

    def loss(t):
        return log_loss(ry, recalibrate_prior(expit(t * z), rd), labels=[0, 1])

    t = minimize_scalar(loss, bounds=(0.5, 1.5), method="bounded", options={"xatol": 1e-6}).x
    return expit(float(t) * logit(np.clip(rq, EPS, 1 - EPS)))


def platt_fit(rp, ry, rd, rq):
    z = logit(np.clip(rp, EPS, 1 - EPS))[:, None]
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    model.fit(z, ry)
    return model.predict_proba(logit(np.clip(rq, EPS, 1 - EPS))[:, None])[:, 1]


def segment_platt_fit(rp, ry, rd, rq, segment):
    out = np.empty(len(rq), dtype=np.float64)
    z = logit(np.clip(rp, EPS, 1 - EPS))[:, None]
    zq = logit(np.clip(rq, EPS, 1 - EPS))[:, None]
    for value in np.unique(segment):
        tr = segment == value
        te = rq >= 0
        model = LogisticRegression(C=1e3, solver="lbfgs", max_iter=1000)
        model.fit(z[tr], ry[tr])
        out[te] = model.predict_proba(zq)[:, 1]
    return out


def beta_fit(rp, ry, rd, rq):
    p = np.clip(rp, EPS, 1 - EPS)

    def objective(w):
        a, b, c = w
        if a <= 0 or b <= 0 or c < 0 or c > 1:
            return 1e6
        transformed = c * betainc(a, b, p) + (1 - c) * p
        return log_loss(ry, np.clip(transformed, EPS, 1 - EPS), labels=[0, 1])

    res = minimize(objective, [1.0, 1.0, 0.5], method="Nelder-Mead", bounds=[(0.2, 5), (0.2, 5), (0, 1)], options={"maxiter": 300, "xatol": 1e-5, "fatol": 1e-7})
    a, b, c = res.x
    return c * betainc(a, b, np.clip(rq, EPS, 1 - EPS)) + (1 - c) * np.clip(rq, EPS, 1 - EPS)


def isotonic_fit(rp, ry, rd, rq, weight):
    model = IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1 - EPS)
    model.fit(rp, ry)
    return (1 - weight) * np.clip(rq, EPS, 1 - EPS) + weight * model.predict(rq)


def clip_fit(rp, ry, rd, rq, low, high):
    return np.clip(rq, low, high)


def feature_matrix(train: pd.DataFrame, test: pd.DataFrame, d_tr: np.ndarray, d_te: np.ndarray, p_tr: np.ndarray, p_te: np.ndarray):
    cols = ["gender", "region", "smartphone", "segment", "earning_pattern"]
    out_tr = pd.DataFrame(index=np.arange(len(train)))
    out_te = pd.DataFrame(index=np.arange(len(test)))
    out_tr["blend_logit"] = logit(np.clip(p_tr, EPS, 1 - EPS))
    out_te["blend_logit"] = logit(np.clip(p_te, EPS, 1 - EPS))
    out_tr["date"] = d_tr
    out_te["date"] = d_te
    for col in cols:
        cats = pd.get_dummies(pd.concat([train[col], test[col]], ignore_index=True), prefix=col, dtype=float)
        out_tr = pd.concat([out_tr, cats.iloc[: len(train)].reset_index(drop=True)], axis=1)
        out_te = pd.concat([out_te, cats.iloc[len(train) :].reset_index(drop=True)], axis=1)
    for col, bins in (("arpu", 10), ("age", 5), ("x_90_d_activity_rate", 10)):
        values = train[col].to_numpy(np.float64)
        edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)))
        edges[0] = -np.inf
        edges[-1] = np.inf
        out_tr[col + "_q"] = np.digitize(values, edges[1:-1])
        out_te[col + "_q"] = np.digitize(test[col].to_numpy(np.float64), edges[1:-1])
    return out_tr.to_numpy(np.float64), out_te.to_numpy(np.float64)


def logistic_feature_fit(x_all_tr, x_te_all, tr_idx, q_idx, y, C):
    model = LogisticRegression(C=C, solver="lbfgs", max_iter=2000)
    model.fit(x_all_tr[tr_idx], y)
    return model.predict_proba(x_all_tr[q_idx])[:, 1], model.predict_proba(x_te_all)[:, 1]


def main():
    tune = load_prediction("reports/tuned_trend_conc_s10_predictions.npz")
    final = load_prediction("reports/final_tuned_lgb100_predictions.npz")
    y, d_tr, d_te = tune["y"], tune["d_tr"], tune["d_te"]
    train, test_df, _ = load_data(DATA_DIR)
    repeats = np.asarray([blend_repeat(tune["oof_repeats"][j], final["oof_repeats"][j]) for j in range(len(tune["oof_repeats"]))])
    test_p = blend_repeat(tune["test"], final["test"])
    base_oof = repeats.mean(axis=0)
    folds = load_folds()
    base_eval = evaluate_folds(y, base_oof, folds)["overall"]

    results = {}
    predictions = {"y": y, "d_tr": d_tr, "d_te": d_te, "base_oof": base_oof, "base_test": test_p}

    def add(name, oof, test, fitting, params):
        ev = evaluate_folds(y, oof, folds)["overall"]
        delta = paired_delta(base_oof, oof, y, folds)
        results[name] = {
            "proxy": ev["proxy"], "log_loss": ev["log_loss"], "auc": ev["auc"],
            "fold_std": evaluate_folds(y, oof, folds)["std_proxy"],
            "delta_proxy": ev["proxy"] - base_eval["proxy"],
            "delta_log_loss": ev["log_loss"] - base_eval["log_loss"],
            "delta_auc": ev["auc"] - base_eval["auc"],
            "paired_proxy_delta": delta["delta"], "paired_proxy_se": delta["se"], "accept": delta["accept"],
            "fitting": fitting, "params": params,
        }
        predictions[name + "_oof"] = oof
        predictions[name + "_test"] = test

    add("B", base_oof, test_p, "identity", {})
    for low in (1e-5, 3e-5, 1e-4, 3e-4, 1e-3):
        for high in (1 - 1e-5, 1 - 3e-5, 1 - 1e-4, 1 - 3e-4, 1 - 1e-3):
            name = f"clip_{low:g}_{1-high:g}"
            oof, test = nested_predict(repeats, test_p, y, d_tr, d_te, lambda rp, ry, rd, rq, lo=low, hi=high: clip_fit(rp, ry, rd, rq, lo, hi))
            add(name, oof, test, "clip", {"low": low, "high": high})

    for name, fn, fitting in (
        ("temperature", temperature_fit, "temperature"),
        ("platt", platt_fit, "platt"),
        ("beta", beta_fit, "beta"),
    ):
        oof, test = nested_predict(repeats, test_p, y, d_tr, d_te, fn)
        add(name, oof, test, fitting, {})

    for weight in (0.25, 0.5, 0.75, 1.0):
        name = f"iso_smooth_{weight:g}"
        oof, test = nested_predict(repeats, test_p, y, d_tr, d_te, lambda rp, ry, rd, rq, w=weight: isotonic_fit(rp, ry, rd, rq, w))
        add(name, oof, test, "isotonic_smoothed", {"weight": weight})

    segment_train_all = np.concatenate([np.asarray(train["segment"])] * len(repeats))
    segment_test = np.asarray(test_df["segment"])
    rep_idx = []
    for j in range(len(repeats)):
        rep_idx.append(np.arange(j * len(y), (j + 1) * len(y)))
    z_all = logit(np.clip(np.concatenate(repeats), EPS, 1 - EPS))[:, None]
    y_all = np.concatenate([y] * len(repeats))
    nested_segment = np.empty_like(repeats)
    for j in range(len(repeats)):
        idx = [i for i in range(len(repeats)) if i != j]
        tr_rows = np.concatenate([rep_idx[i] for i in idx])
        q_rows = rep_idx[j]
        out = np.empty(len(y), dtype=np.float64)
        for value in np.unique(segment_train_all):
            tr = tr_rows[segment_train_all[tr_rows] == value]
            q = q_rows[segment_train_all[q_rows] == value]
            if len(q) == 0:
                continue
            model = LogisticRegression(C=1e3, solver="lbfgs", max_iter=1000)
            model.fit(z_all[tr], y_all[tr])
            out[q % len(y)] = model.predict_proba(z_all[q])[:, 1]
        nested_segment[j] = recalibrate_prior(out, d_tr)
    z_train = logit(np.clip(np.concatenate(repeats), EPS, 1 - EPS))[:, None]
    z_test = logit(np.clip(test_p, EPS, 1 - EPS))[:, None]
    test_segment = np.empty(len(test_df), dtype=np.float64)
    for value in np.unique(segment_train_all):
        tr = segment_train_all == value
        q = segment_test == value
        if not q.any():
            continue
        model = LogisticRegression(C=1e3, solver="lbfgs", max_iter=1000)
        model.fit(z_train[tr], np.concatenate([y] * len(repeats))[tr])
        test_segment[q] = model.predict_proba(z_test[q])[:, 1]
    add("segment_platt", nested_segment.mean(axis=0), recalibrate_prior(test_segment, d_te), "segment_platt", {"segment": "segment"})

    feat_tr, feat_te = feature_matrix(train, test_df, d_tr, d_te, base_oof, test_p)
    feat_tr_all = np.concatenate([feat_tr] * len(repeats))
    for C in (0.01, 0.1, 1.0):
        def logistic_wrapper(rp, ry, rd, rq, c=C):
            return logistic_feature_fit(feat_tr, feat_te, np.arange(len(feat_tr)), np.arange(len(feat_tr)), ry, c)[0]
        def logistic_apply(rp, ry, rd, rq, c=C):
            model = LogisticRegression(C=c, solver="lbfgs", max_iter=2000)
            model.fit(feat_tr, ry)
            return model.predict_proba(feat_te)[:, 1]
        nested = np.empty_like(repeats)
        for j in range(len(repeats)):
            idx = [i for i in range(len(repeats)) if i != j]
            tr_rows = np.concatenate([rep_idx[i] for i in idx])
            ytr = np.concatenate([y] * len(idx))
            model = LogisticRegression(C=C, solver="lbfgs", max_iter=2000)
            model.fit(feat_tr_all[tr_rows], ytr)
            nested[j] = recalibrate_prior(model.predict_proba(feat_tr)[:, 1], d_tr)
        model = LogisticRegression(C=C, solver="lbfgs", max_iter=2000)
        model.fit(feat_tr_all, np.concatenate([y] * len(repeats)))
        test_cal = recalibrate_prior(model.predict_proba(feat_te)[:, 1], d_te)
        add(f"logistic_features_C{C:g}", nested.mean(axis=0), test_cal, "logistic_features", {"C": C})

    for leaves in (2, 4):
        def gbm_apply(leaves=leaves):
            def fn(rp, ry, rd, rq):
                model = HistGradientBoostingClassifier(max_iter=60, learning_rate=0.03, max_leaf_nodes=leaves, min_samples_leaf=500, l2_regularization=10, random_state=42)
                model.fit(feat_tr, ry)
                return model.predict_proba(feat_te)[:, 1]
            return fn
        nested = np.empty_like(repeats)
        for j in range(len(repeats)):
            idx = [i for i in range(len(repeats)) if i != j]
            tr_rows = np.concatenate([rep_idx[i] for i in idx])
            ytr = np.concatenate([y] * len(idx))
            model = HistGradientBoostingClassifier(max_iter=60, learning_rate=0.03, max_leaf_nodes=leaves, min_samples_leaf=500, l2_regularization=10, random_state=42)
            model.fit(feat_tr_all[tr_rows], ytr)
            nested[j] = recalibrate_prior(model.predict_proba(feat_tr)[:, 1], d_tr)
        model = HistGradientBoostingClassifier(max_iter=60, learning_rate=0.03, max_leaf_nodes=leaves, min_samples_leaf=500, l2_regularization=10, random_state=42)
        model.fit(feat_tr_all, np.concatenate([y] * len(repeats)))
        test_cal = recalibrate_prior(model.predict_proba(feat_te)[:, 1], d_te)
        add(f"gbm_features_leaves{leaves}", nested.mean(axis=0), test_cal, "gbm_features", {"leaves": leaves})

    ordered = sorted(results.items(), key=lambda kv: kv[1]["proxy"], reverse=True)
    write_json("reports/calibration_suite.json", {"base": base_eval, "results": results, "ranking": [k for k, _ in ordered]})
    np.savez_compressed("reports/calibration_predictions.npz", **predictions)
    print(json.dumps({"base": base_eval, "top": [(k, v["proxy"], v["log_loss"], v["auc"], v["paired_proxy_delta"], v["paired_proxy_se"], v["accept"]) for k, v in ordered[:15]]}, indent=2))


def blend_repeat(a, b):
    from scipy.special import expit, logit
    from v6_common import EPS
    return expit(0.5 * logit(np.clip(a, EPS, 1 - EPS)) + 0.5 * logit(np.clip(b, EPS, 1 - EPS)))


if __name__ == "__main__":
    main()
