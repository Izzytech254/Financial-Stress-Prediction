"""Finalize v5 with polynomial calibration and prior shift.

Loads the existing v5 LGB/Cat predictions, rebuilds the selected w=0.25 blend,
compares Platt against regularized polynomial calibration on the blend logit,
selects by cross-validated 60/40 proxy, and writes submission_v5_poly.csv.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from solution import DATA_DIR, ID, REPORT_DIR, ROOT

SUB_TARGET = "Target"
W_LGB = 0.25
TARGET_MEAN = 0.150
N_FOLDS = 5
RANDOM_STATE = 42


def _logit(p: np.ndarray) -> np.ndarray:
    return logit(np.clip(p, 1e-6, 1 - 1e-6))


def proxy(auc: float, ll: float) -> float:
    return 0.6 * ll + 0.4 * (1.0 - auc)


def make_model(degree: int, c: float, scale: bool):
    steps = [PolynomialFeatures(degree=degree, include_bias=False)]
    if scale:
        steps.append(StandardScaler())
    steps.append(LogisticRegression(C=c, max_iter=2000, solver="lbfgs"))
    return make_pipeline(*steps)


def cv_predict(
    values: np.ndarray, y: np.ndarray, degree: int, c: float, scale: bool
) -> np.ndarray:
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    pred = np.zeros(len(y), dtype=np.float64)
    for tr, va in skf.split(values, y):
        model = make_model(degree, c, scale)
        model.fit(values[tr].reshape(-1, 1), y[tr])
        pred[va] = model.predict_proba(values[va].reshape(-1, 1))[:, 1]
    return pred


def fit_full(
    values: np.ndarray, y: np.ndarray, degree: int, c: float, scale: bool
):
    model = make_model(degree, c, scale)
    model.fit(values.reshape(-1, 1), y)
    return model


def prior_shift(test_logit: np.ndarray, target_mean: float) -> float:
    return float(
        brentq(
            lambda d: float(expit(test_logit + d).mean()) - target_mean,
            -8.0,
            8.0,
        )
    )


def validate_submission(path: str, sample: pd.DataFrame) -> dict:
    sub = pd.read_csv(path)
    assert list(sub.columns) == [ID, SUB_TARGET], f"{path}: columns={list(sub.columns)}"
    assert len(sub) == len(sample) == 30000, f"{path}: rows={len(sub)}"
    assert sub[ID].is_unique, f"{path}: duplicate IDs"
    assert (sub[ID].values == sample[ID].values).all(), f"{path}: ID order mismatch"
    assert sub[SUB_TARGET].notna().all(), f"{path}: NaN predictions"
    assert np.isfinite(sub[SUB_TARGET].values).all(), f"{path}: non-finite predictions"
    assert sub[SUB_TARGET].between(0, 1).all(), f"{path}: predictions outside [0,1]"
    values = sub[SUB_TARGET].values
    return {
        "rows": int(len(sub)),
        "mean": float(values.mean()),
        "min": float(values.min()),
        "max": float(values.max()),
        "q05": float(np.quantile(values, 0.05)),
        "q50": float(np.quantile(values, 0.50)),
        "q95": float(np.quantile(values, 0.95)),
    }


def main() -> None:
    z5l = np.load(os.path.join(REPORT_DIR, "v5_lgb3.npz"))
    z5c = np.load(os.path.join(REPORT_DIR, "v5_cat3.npz"))
    y = z5l["y"]
    lgb = (z5l["oof_lgb3"], z5l["test_lgb3"])
    cat = (z5c["oof_cat3"], z5c["test_cat3"])

    p_raw = expit(W_LGB * _logit(lgb[0]) + (1.0 - W_LGB) * _logit(cat[0]))
    t_raw = expit(W_LGB * _logit(lgb[1]) + (1.0 - W_LGB) * _logit(cat[1]))
    l_oof = _logit(p_raw)
    l_test = _logit(t_raw)

    raw_auc = float(roc_auc_score(y, p_raw))
    raw_ll = float(log_loss(y, p_raw))

    rows: List[dict] = []
    variants: Dict[str, dict] = {}

    # Platt control.
    platt = LogisticRegression(C=1e6, max_iter=2000, solver="lbfgs")
    platt.fit(l_oof.reshape(-1, 1), y)
    p_platt = platt.predict_proba(l_oof.reshape(-1, 1))[:, 1]
    t_platt = platt.predict_proba(l_test.reshape(-1, 1))[:, 1]
    platt_cv = cv_predict(l_oof, y, 1, 1e6, False)
    variants["platt"] = {
        "kind": "platt",
        "degree": 1,
        "C": 1e6,
        "scale": False,
        "oof": p_platt,
        "test": t_platt,
        "cv_pred": platt_cv,
    }

    for degree in (2, 3, 4):
        for c in (0.1, 0.3, 1.0, 3.0):
            for scale in (False, True):
                name = f"poly_d{degree}_C{c:g}_{'scale' if scale else 'raw'}"
                model = fit_full(l_oof, y, degree, c, scale)
                p_oof = model.predict_proba(l_oof.reshape(-1, 1))[:, 1]
                l_test_clip = np.clip(l_test, l_oof.min(), l_oof.max())
                t_pred = model.predict_proba(l_test_clip.reshape(-1, 1))[:, 1]
                cv_pred = cv_predict(l_oof, y, degree, c, scale)
                variants[name] = {
                    "kind": "poly",
                    "degree": degree,
                    "C": c,
                    "scale": scale,
                    "oof": p_oof,
                    "test": t_pred,
                    "cv_pred": cv_pred,
                }

    for name, info in variants.items():
        cv_auc = float(roc_auc_score(y, info["cv_pred"]))
        cv_ll = float(log_loss(y, info["cv_pred"]))
        oof_auc = float(roc_auc_score(y, info["oof"]))
        oof_ll = float(log_loss(y, info["oof"]))
        rows.append(
            {
                "name": name,
                "kind": info["kind"],
                "degree": info["degree"],
                "C": info["C"],
                "scale": info["scale"],
                "cv_auc": cv_auc,
                "cv_ll": cv_ll,
                "cv_proxy": proxy(cv_auc, cv_ll),
                "oof_auc": oof_auc,
                "oof_ll": oof_ll,
                "oof_proxy": proxy(oof_auc, oof_ll),
            }
        )

    best_row = min(rows, key=lambda r: r["cv_proxy"])
    best = variants[best_row["name"]]

    p_test_cal = np.clip(best["test"], 1e-6, 1 - 1e-6)
    delta = prior_shift(_logit(p_test_cal), TARGET_MEAN)
    p_test_shift = expit(_logit(p_test_cal) + delta)
    p_oof_shift = expit(_logit(np.clip(best["oof"], 1e-6, 1 - 1e-6)) + delta)

    test = pd.read_csv(os.path.join(DATA_DIR, "Test.csv"))
    sample = pd.read_csv(os.path.join(DATA_DIR, "SampleSubmission.csv"))
    sub = pd.DataFrame({ID: test[ID].values, SUB_TARGET: p_test_shift})
    sub = sample[[ID]].merge(sub, on=ID, how="left")
    assert list(sub.columns) == [ID, SUB_TARGET]
    assert sub[SUB_TARGET].notna().all()
    assert (sub[ID].values == sample[ID].values).all()
    path = os.path.join(ROOT, "submission_v5_poly.csv")
    sub.to_csv(path, index=False, float_format="%.8f")
    stats = validate_submission(path, sample)

    report = {
        "blend": {"w_lgb": W_LGB},
        "raw": {
            "auc": raw_auc,
            "logloss": raw_ll,
            "proxy": proxy(raw_auc, raw_ll),
        },
        "selected": best_row,
        "prior_shift": delta,
        "selected_oof_shift": {
            "auc": float(roc_auc_score(y, p_oof_shift)),
            "logloss": float(log_loss(y, p_oof_shift)),
            "proxy": proxy(
                float(roc_auc_score(y, p_oof_shift)),
                float(log_loss(y, p_oof_shift)),
            ),
        },
        "test": stats,
        "variants": rows,
        "submission_file": os.path.basename(path),
    }
    with open(os.path.join(REPORT_DIR, "v5_poly_final.json"), "w") as f:
        json.dump(report, f, indent=2)

    lines = [
        "# v5 polynomial calibration",
        "",
        f"- Blend: `w_LGB={W_LGB:.2f}`",
        f"- Raw OOF: AUC `{raw_auc:.6f}`, LL `{raw_ll:.6f}`, proxy `{proxy(raw_auc, raw_ll):.6f}`",
        f"- Selected: `{best_row['name']}`",
        f"- CV: AUC `{best_row['cv_auc']:.6f}`, LL `{best_row['cv_ll']:.6f}`, proxy `{best_row['cv_proxy']:.6f}`",
        f"- Prior shift: `{delta:+.6f}`",
        f"- Test mean: `{stats['mean']:.6f}`, min `{stats['min']:.6f}`, max `{stats['max']:.6f}`",
        "",
        "| variant | CV AUC | CV LL | CV proxy | OOF AUC | OOF LL |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(rows, key=lambda x: x["cv_proxy"]):
        lines.append(
            f"| {r['name']} | {r['cv_auc']:.6f} | {r['cv_ll']:.6f} | "
            f"{r['cv_proxy']:.6f} | {r['oof_auc']:.6f} | {r['oof_ll']:.6f} |"
        )
    lines += [
        "",
        f"Candidate submission: `{os.path.basename(path)}`.",
        "",
    ]
    with open(os.path.join(REPORT_DIR, "v5_poly_final.md"), "w") as f:
        f.write("\n".join(lines))

    print(
        f"Selected {best_row['name']} cv_proxy={best_row['cv_proxy']:.6f} "
        f"cv_ll={best_row['cv_ll']:.6f} cv_auc={best_row['cv_auc']:.6f}"
    )
    print(f"Wrote {path} mean={stats['mean']:.6f} min={stats['min']:.6f} max={stats['max']:.6f}")


if __name__ == "__main__":
    main()
