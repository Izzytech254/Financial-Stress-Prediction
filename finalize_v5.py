"""Finalize v5: choose the LGB/Cat blend weight and write the candidate submission."""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import log_loss, roc_auc_score

from improve_v3 import _blend_report, _logit
from solution import DATA_DIR, ID, REPORT_DIR, ROOT

SUB_TARGET = "Target"
WEIGHTS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]


def proxy(auc: float, ll: float) -> float:
    return 0.6 * ll + 0.4 * (1.0 - auc)


def main() -> None:
    z5l = np.load(os.path.join(REPORT_DIR, "v5_lgb3.npz"))
    z5c = np.load(os.path.join(REPORT_DIR, "v5_cat3.npz"))
    zclean = np.load(os.path.join(REPORT_DIR, "v2_clean_predictions.npz"))
    y = z5l["y"]
    last = z5l["last"]
    zfold = np.load(os.path.join(REPORT_DIR, "folds_v2.npz"))
    folds = [(zfold[f"tr_42_{i}"], zfold[f"va_42_{i}"]) for i in range(5)]
    base = zclean["oof_final"]
    lgb = (z5l["oof_lgb3"], z5l["test_lgb3"])
    cat = (z5c["oof_cat3"], z5c["test_cat3"])

    v2 = {
        "auc": float(roc_auc_score(y, base)),
        "logloss": float(log_loss(y, base)),
        "last_auc": float(roc_auc_score(y[last], base[last])),
        "last_logloss": float(log_loss(y[last], base[last])),
        "platt_cv_ll": 0.2409121687667058,
    }
    v2["proxy"] = proxy(v2["auc"], v2["platt_cv_ll"])

    rows = []
    for w in WEIGHTS:
        p = expit(w * _logit(lgb[0]) + (1 - w) * _logit(cat[0]))
        t = expit(w * _logit(lgb[1]) + (1 - w) * _logit(cat[1]))
        r = _blend_report(f"w{w:.2f}", p, t, y, last, base, folds)
        ps = r["_p_shift_oof"]
        rows.append(
            {
                "w_lgb": w,
                "auc": r["auc"],
                "logloss": r["logloss"],
                "last_auc": r["last_auc"],
                "last_logloss": r["last_logloss"],
                "cal_logloss": float(log_loss(y, ps)),
                "last_cal_logloss": float(log_loss(y[last], ps[last])),
                "platt_cv_ll": r["platt_cv_ll"],
                "proxy": proxy(r["auc"], r["platt_cv_ll"]),
                "delta_auc_vs_v2": r["delta_auc_vs_v2"],
                "delta_ll_vs_v2": r["delta_ll_vs_v2"],
            }
        )

    best = min(rows, key=lambda r: r["proxy"])
    w = best["w_lgb"]
    p = expit(w * _logit(lgb[0]) + (1 - w) * _logit(cat[0]))
    t = expit(w * _logit(lgb[1]) + (1 - w) * _logit(cat[1]))
    r = _blend_report("best", p, t, y, last, base, folds)

    test = pd.read_csv(os.path.join(DATA_DIR, "Test.csv"))
    sample = pd.read_csv(os.path.join(DATA_DIR, "SampleSubmission.csv"))
    pt = np.clip(r["_p_shift_test"], 1e-6, 1 - 1e-6)
    sub = pd.DataFrame({ID: test[ID].values, SUB_TARGET: pt})
    sub = sample[[ID]].merge(sub, on=ID, how="left")
    assert list(sub.columns) == [ID, SUB_TARGET]
    assert sub[SUB_TARGET].notna().all()
    assert (sub[ID].values == sample[ID].values).all()
    best_path = os.path.join(ROOT, "submission_v5_best.csv")
    sub.to_csv(best_path, index=False, float_format="%.8f")

    report = {
        "v2_final": v2,
        "lgb3_v3_500": {
            "auc": float(roc_auc_score(y, lgb[0])),
            "logloss": float(log_loss(y, lgb[0])),
            "last_auc": float(roc_auc_score(y[last], lgb[0][last])),
            "last_logloss": float(log_loss(y[last], lgb[0][last])),
        },
        "cat3_v3_500": {
            "auc": float(roc_auc_score(y, cat[0])),
            "logloss": float(log_loss(y, cat[0])),
            "last_auc": float(roc_auc_score(y[last], cat[0][last])),
            "last_logloss": float(log_loss(y[last], cat[0][last])),
        },
        "weight_sweep": rows,
        "best": best,
        "best_submission": "submission_v5_best.csv",
        "best_test_mean": float(pt.mean()),
        "best_platt_slope": r["platt_slope"],
        "best_platt_intercept": r["platt_intercept"],
        "best_prior_shift": r["prior_shift"],
    }
    with open(os.path.join(REPORT_DIR, "v5_final.json"), "w") as f:
        json.dump(report, f, indent=2)

    lines = [
        "# v5 final: v3 features + 3-seed LGB/Cat blend",
        "",
        f"- v2 final: AUC `{v2['auc']:.6f}`, LL `{v2['logloss']:.6f}`, "
        f"last AUC `{v2['last_auc']:.6f}`, last LL `{v2['last_logloss']:.6f}`, "
        f"Platt CV LL `{v2['platt_cv_ll']:.6f}`",
        f"- v5 LGB 3-seed: AUC `{report['lgb3_v3_500']['auc']:.6f}`, "
        f"LL `{report['lgb3_v3_500']['logloss']:.6f}`",
        f"- v5 Cat 3-seed: AUC `{report['cat3_v3_500']['auc']:.6f}`, "
        f"LL `{report['cat3_v3_500']['logloss']:.6f}`",
        "",
        "## LGB/Cat blend weight sweep (mean logit)",
        "",
        "| w_LGB | AUC | LogLoss | last AUC | last LL | cal LL | last cal LL | Platt CV LL | 60/40 proxy |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['w_lgb']:.2f} | {row['auc']:.6f} | {row['logloss']:.6f} | "
            f"{row['last_auc']:.6f} | {row['last_logloss']:.6f} | {row['cal_logloss']:.6f} | "
            f"{row['last_cal_logloss']:.6f} | {row['platt_cv_ll']:.6f} | {row['proxy']:.6f} |"
        )
    lines += [
        "",
        f"Best by 60/40 proxy: `w_LGB={best['w_lgb']:.2f}` "
        f"(proxy `{best['proxy']:.6f}` vs v2 `{v2['proxy']:.6f}`).",
        f"Recommended submission: `submission_v5_best.csv` "
        f"(prior-shifted test mean `{pt.mean():.6f}`).",
        "",
        "The v5 features add 38 history volatility/trend features to the v2 set "
        "(952 total, top 500 per fold). CatBoost 3-seed is the dominant component; "
        "the blend uses a small LGB weight because the models are highly correlated.",
        "",
    ]
    with open(os.path.join(REPORT_DIR, "v5_final.md"), "w") as f:
        f.write("\n".join(lines))

    print(f"Best w_LGB={w:.2f} proxy={best['proxy']:.6f} -> {best_path}")
    print(
        f"AUC={best['auc']:.6f} LL={best['logloss']:.6f} "
        f"lastAUC={best['last_auc']:.6f} lastLL={best['last_logloss']:.6f} "
        f"PlattCV={best['platt_cv_ll']:.6f}"
    )


if __name__ == "__main__":
    main()
