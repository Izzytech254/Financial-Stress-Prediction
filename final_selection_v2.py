from __future__ import annotations

import json

import numpy as np
from scipy.special import expit, logit

from calibration_lab import exact_proxy, load_prediction, recalibrate_prior
from frozen_cv import evaluate_folds, load_folds, paired_delta
from v6_common import DATA_DIR, EPS, load_data, write_json


def blended(a, b, wa, d_tr=None, d_te=None, test_a=None, test_b=None):
    oof = expit((1 - wa) * logit(np.clip(a, EPS, 1 - EPS)) + wa * logit(np.clip(b, EPS, 1 - EPS)))
    if test_a is None:
        return oof, None
    test = expit((1 - wa) * logit(np.clip(test_a, EPS, 1 - EPS)) + wa * logit(np.clip(test_b, EPS, 1 - EPS)))
    return recalibrate_prior(oof, d_tr), recalibrate_prior(test, d_te)


def bootstrap_delta(y, base, cand, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    groups = np.arange(len(y)) // 4
    rows = np.arange(len(y))
    out = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        pick = rng.integers(0, 10000, 10000)
        idx = (pick[:, None] * 4 + np.arange(4)[None, :]).ravel()
        out[i] = exact_proxy(y[idx], cand[idx])[0] - exact_proxy(y[idx], base[idx])[0]
    return {
        "mean": float(out.mean()),
        "std": float(out.std(ddof=1)),
        "p025": float(np.quantile(out, 0.025)),
        "p975": float(np.quantile(out, 0.975)),
        "prob_positive": float((out > 0).mean()),
    }


def main():
    b = load_prediction("reports/submission_B_predictions.npz")
    y, d_tr, d_te = b["y"], b["d_tr"], b["d_te"]
    cal = np.load("reports/calibration_predictions.npz")
    v8 = load_prediction("reports/v8_base_init_rank3_s50_predictions.npz")
    z7 = np.load("reports/v7_struct_perpower_predictions.npz", allow_pickle=True)
    v7 = {k: z7[k] for k in ("y", "oof", "d_tr", "d_te", "test")}
    base_oof, base_test = b["oof"], b["test"]
    folds = load_folds()
    base_eval = evaluate_folds(y, base_oof, folds)["overall"]

    candidates = {
        "B": (base_oof, base_test),
        "logistic_features_C1": (cal["logistic_features_C1_oof"], cal["logistic_features_C1_test"]),
        "v8_base_init_rank3_s50": (v8["oof"], v8["test"]),
        "v7_struct_perpower": (v7["oof"], v7["test"]),
    }
    for w in (0.25, 0.5, 0.75):
        oof, test = blended(base_oof, v8["oof"], w, d_tr, d_te, base_test, v8["test"])
        candidates[f"B_v8_logit_{w:g}"] = (oof, test)
    for w in (0.25, 0.5, 0.75):
        oof, test = blended(base_oof, v7["oof"], w, d_tr, d_te, base_test, v7["test"])
        candidates[f"B_v7_logit_{w:g}"] = (oof, test)
    for w in (0.25, 0.5):
        oof, test = blended(cal["logistic_features_C1_oof"], v8["oof"], w, d_tr, d_te, cal["logistic_features_C1_test"], v8["test"])
        candidates[f"logC1_v8_logit_{w:g}"] = (oof, test)

    results = {}
    for name, (oof, test) in candidates.items():
        ev = evaluate_folds(y, oof, folds)["overall"]
        delta = paired_delta(base_oof, oof, y, folds)
        by_date = []
        for d in np.unique(d_tr):
            m = d_tr == d
            p0, l0, a0 = exact_proxy(y[m], base_oof[m])
            p1, l1, a1 = exact_proxy(y[m], oof[m])
            by_date.append({"date": int(d), "base": p0, "cand": p1, "delta": p1 - p0, "delta_ll": l1 - l0, "delta_auc": a1 - a0})
        boot = bootstrap_delta(y, base_oof, oof)
        results[name] = {
            "proxy": ev["proxy"], "log_loss": ev["log_loss"], "auc": ev["auc"],
            "delta_proxy": ev["proxy"] - base_eval["proxy"], "delta_ll": ev["log_loss"] - base_eval["log_loss"], "delta_auc": ev["auc"] - base_eval["auc"],
            "paired_delta": delta["delta"], "paired_se": delta["se"], "paired_accept": delta["accept"],
            "bootstrap": boot, "by_date": by_date,
            "test": {"min": float(test.min()), "max": float(test.max()), "mean": float(test.mean()), "unique": int(len(np.unique(test)))},
        }

    write_json("reports/final_selection_v2.json", {"base": base_eval, "results": results})
    np.savez_compressed("reports/final_candidates_v2_predictions.npz", y=y, d_tr=d_tr, d_te=d_te, **{f"{k}_oof": v[0] for k, v in candidates.items()}, **{f"{k}_test": v[1] for k, v in candidates.items()})
    ranked = sorted(results.items(), key=lambda kv: kv[1]["by_date"][-1]["delta"], reverse=True)
    print(json.dumps({"base": base_eval, "ranked_by_date3": [(k, v["proxy"], v["delta_proxy"], v["by_date"][-1]["delta"], v["bootstrap"]["p025"], v["bootstrap"]["p975"]) for k, v in ranked]}, indent=2))


if __name__ == "__main__":
    main()
