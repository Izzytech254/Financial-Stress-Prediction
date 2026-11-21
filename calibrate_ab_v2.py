"""Generate monotonic calibration A/B variants for the v2 blend."""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.join(ROOT, "reports")
ID = "ID"
TARGET = "Target"


def _logit(p: np.ndarray) -> np.ndarray:
    return logit(np.clip(p, 1e-6, 1 - 1e-6))


def _fit_intercept(values: np.ndarray, y: np.ndarray, slope: float) -> float:
    def objective(b: float) -> float:
        return float(log_loss(y, np.clip(expit(slope * values + b), 1e-9, 1 - 1e-9)))

    result = minimize_scalar(objective, bounds=(-5.0, 5.0), method="bounded")
    return float(result.x)


def _cv_logloss(values: np.ndarray, y: np.ndarray, slope: float) -> float:
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    pred = np.zeros(len(y))
    for tr, va in skf.split(values, y):
        b = _fit_intercept(values[tr], y[tr], slope)
        pred[va] = expit(slope * values[va] + b)
    return float(log_loss(y, np.clip(pred, 1e-9, 1 - 1e-9)))


def _write_submission(path: str, test_ids: np.ndarray, pred: np.ndarray) -> None:
    sub = pd.DataFrame({ID: test_ids, TARGET: np.clip(pred, 1e-6, 1 - 1e-6)})
    sub.to_csv(path, index=False, float_format="%.8f")


def _validate(path: str, sample: pd.DataFrame) -> dict:
    sub = pd.read_csv(path)
    assert list(sub.columns) == [ID, TARGET], f"{path}: columns={list(sub.columns)}"
    assert len(sub) == len(sample) == 30000, f"{path}: rows={len(sub)}"
    assert sub[ID].is_unique, f"{path}: duplicate IDs"
    assert (sub[ID].values == sample[ID].values).all(), f"{path}: ID order mismatch"
    assert sub[TARGET].notna().all(), f"{path}: NaN predictions"
    assert np.isfinite(sub[TARGET].values).all(), f"{path}: non-finite predictions"
    assert sub[TARGET].between(0, 1).all(), f"{path}: predictions outside [0,1]"
    values = sub[TARGET].values
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
    z = np.load(os.path.join(REPORT_DIR, "v2_predictions.npz"))
    y = z["y"]
    P = np.clip(np.column_stack([z["oof_lgb_v2"], z["oof_cat_v2"]]), 1e-6, 1 - 1e-6)
    T = np.clip(np.column_stack([z["test_lgb_v2"], z["test_cat_v2"]]), 1e-6, 1 - 1e-6)
    p_raw = expit(_logit(P).mean(axis=1))
    t_raw = expit(_logit(T).mean(axis=1))
    l = _logit(p_raw)
    lt = _logit(t_raw)

    lr = LogisticRegression(C=1e6, max_iter=1000, solver="lbfgs")
    lr.fit(l.reshape(-1, 1), y)
    slope = float(lr.coef_[0, 0])
    intercept = float(lr.intercept_[0])
    t_control = lr.predict_proba(lt.reshape(-1, 1))[:, 1]

    test = pd.read_csv(os.path.join(ROOT, "data", "Test.csv"))
    sample = pd.read_csv(os.path.join(ROOT, "data", "SampleSubmission.csv"))
    current = pd.read_csv(os.path.join(ROOT, "submission.csv"))
    control_gap = float(np.max(np.abs(current[TARGET].values - t_control)))
    assert control_gap < 1e-6, f"control mismatch: {control_gap}"

    variants = {
        "control": {
            "slope": slope,
            "intercept": intercept,
            "test": t_control,
            "oof": expit(slope * l + intercept),
            "cv_ll": _cv_logloss(l, y, slope),
            "note": "current submission",
        }
    }

    for name, s in [("s100", 1.00), ("s105", 1.05)]:
        b = _fit_intercept(l, y, s)
        variants[name] = {
            "slope": s,
            "intercept": b,
            "test": expit(s * lt + b),
            "oof": expit(s * l + b),
            "cv_ll": _cv_logloss(l, y, s),
            "note": f"fixed slope {s:.2f}",
        }

    target_mean = 0.150
    delta = brentq(
        lambda d: float(expit(slope * lt + intercept + d).mean()) - target_mean,
        -5.0,
        5.0,
    )
    variants["prior150"] = {
        "slope": slope,
        "intercept": intercept + float(delta),
        "test": expit(slope * lt + intercept + delta),
        "oof": expit(slope * l + intercept + delta),
        "cv_ll": None,
        "note": f"current Platt with intercept shifted {delta:+.5f} to test mean {target_mean:.3f}",
    }

    rows = []
    for name, info in variants.items():
        path = os.path.join(ROOT, f"submission_cal_{name}.csv")
        _write_submission(path, test[ID].values, info["test"])
        metrics = _validate(path, sample)
        oof_ll = float(log_loss(y, np.clip(info["oof"], 1e-9, 1 - 1e-9)))
        oof_auc = float(roc_auc_score(y, info["oof"]))
        diff = float(np.max(np.abs(info["test"] - t_control)))
        rows.append(
            {
                "variant": name,
                "file": os.path.basename(path),
                "slope": info["slope"],
                "intercept": info["intercept"],
                "note": info["note"],
                "test_mean": metrics["mean"],
                "test_min": metrics["min"],
                "test_max": metrics["max"],
                "test_q05": metrics["q05"],
                "test_q50": metrics["q50"],
                "test_q95": metrics["q95"],
                "oof_auc": oof_auc,
                "oof_ll_insample": oof_ll,
                "oof_ll_cv": info["cv_ll"],
                "delta_ll_cv_vs_control": (
                    None if info["cv_ll"] is None else float(info["cv_ll"] - variants["control"]["cv_ll"])
                ),
                "max_abs_diff_vs_control": diff,
            }
        )

    report = {
        "y_rate": float(y.mean()),
        "raw_oof_auc": float(roc_auc_score(y, p_raw)),
        "raw_oof_logloss": float(log_loss(y, p_raw)),
        "control_slope": slope,
        "control_intercept": intercept,
        "control_submission_gap": control_gap,
        "prior_shift_delta": float(delta),
        "variants": rows,
    }
    with open(os.path.join(REPORT_DIR, "calibration_ab_v2.json"), "w") as f:
        json.dump(report, f, indent=2)

    lines = [
        "# Calibration A/B v2",
        "",
        f"- Raw OOF AUC: `{report['raw_oof_auc']:.6f}`",
        f"- Raw OOF LogLoss: `{report['raw_oof_logloss']:.6f}`",
        f"- Control Platt slope: `{slope:.6f}`, intercept: `{intercept:.6f}`",
        f"- Control file matches current `submission.csv` within `{control_gap:.2e}`",
        "",
        "| variant | file | test mean | OOF AUC | OOF LL in-sample | OOF LL CV | dLL CV vs control | max diff vs control |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        cv = "n/a" if r["oof_ll_cv"] is None else f"{r['oof_ll_cv']:.6f}"
        d = "n/a" if r["delta_ll_cv_vs_control"] is None else f"{r['delta_ll_cv_vs_control']:+.6f}"
        lines.append(
            f"| {r['variant']} | `{r['file']}` | {r['test_mean']:.6f} | {r['oof_auc']:.6f} | "
            f"{r['oof_ll_insample']:.6f} | {cv} | {d} | {r['max_abs_diff_vs_control']:.6f} |"
        )
    lines += [
        "",
        "Submit `submission_cal_s100.csv` and `submission_cal_s105.csv` against the current control.",
        "AUC should be unchanged; compare public LogLoss only.",
    ]
    with open(os.path.join(REPORT_DIR, "calibration_ab_v2.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
