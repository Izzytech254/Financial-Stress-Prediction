from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from frozen_cv import evaluate_folds, load_folds, paired_delta
from v6_common import DATA_DIR, load_data, validate_submission, write_json

EPS = 1e-12


def proxy(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import log_loss, roc_auc_score

    return 0.6 * (1.0 - log_loss(y, p) / 0.595) + 0.4 * roc_auc_score(y, p)


def load_pred(path: str) -> dict:
    z = np.load(path)
    oof = np.clip(z["oof"], EPS, 1 - EPS)
    repeats = z["oof_repeats"] if "oof_repeats" in z.files else np.repeat(oof[None, :], 3, axis=0)
    test = np.clip(z["test"], EPS, 1 - EPS) if "test" in z.files else None
    return {"oof": oof, "repeats": repeats, "test": test, "y": z["y"]}


def fit_temp(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import log_loss

    z = np.clip(logit(np.clip(p, EPS, 1 - EPS)), -10.0, 10.0)

    def nll(t: float) -> float:
        return log_loss(y, np.clip(expit(t * z), EPS, 1 - EPS))

    return float(minimize_scalar(nll, bounds=(0.5, 1.5), method="bounded").x)


def apply_temp(t: float, p: np.ndarray) -> np.ndarray:
    z = np.clip(logit(np.clip(p, EPS, 1 - EPS)), -10.0, 10.0)
    return np.clip(expit(t * z), EPS, 1 - EPS)


def fit_platt(y: np.ndarray, p: np.ndarray):
    m = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    m.fit(logit(np.clip(p, EPS, 1 - EPS))[:, None], y)
    return m


def apply_platt(model, p: np.ndarray) -> np.ndarray:
    return model.predict_proba(logit(np.clip(p, EPS, 1 - EPS))[:, None])[:, 1]


def fit_iso(y: np.ndarray, p: np.ndarray):
    m = IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1 - EPS)
    m.fit(p, y)
    return m


def apply_iso(model, p: np.ndarray) -> np.ndarray:
    return np.clip(model.predict(p), EPS, 1 - EPS)


def rank_mean(preds: list[np.ndarray]) -> np.ndarray:
    ranked = [np.argsort(np.argsort(p)) / (len(p) - 1) for p in preds]
    return np.clip(np.mean(ranked, axis=0), EPS, 1 - EPS)


def optimized_weights(y: np.ndarray, preds: list[np.ndarray]) -> np.ndarray:
    n = len(preds)
    mat = np.column_stack([logit(np.clip(p, EPS, 1 - EPS)) for p in preds])

    def neg(w: np.ndarray) -> float:
        return -proxy(y, expit(mat @ w))

    res = minimize(
        neg,
        np.full(n, 1.0 / n),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"maxiter": 500, "ftol": 1e-12},
    )
    w = np.clip(res.x, 0.0, 1.0)
    return w / w.sum()


def blend_prob(preds: list[np.ndarray], weights: np.ndarray, kind: str) -> np.ndarray:
    if kind == "rank":
        return rank_mean(preds)
    if kind == "prob":
        return np.clip(np.average(np.column_stack(preds), axis=1, weights=weights), EPS, 1 - EPS)
    if kind == "logit":
        return expit(sum(w * logit(np.clip(p, EPS, 1 - EPS)) for w, p in zip(weights, preds)))
    raise ValueError(kind)


def nested_calibrate(
    y: np.ndarray,
    repeat_preds: list[np.ndarray],
    fitter,
    applier,
) -> np.ndarray:
    calibrated = np.empty((len(repeat_preds), len(y)), dtype=np.float64)
    for j in range(len(repeat_preds)):
        idx = [i for i in range(len(repeat_preds)) if i != j]
        y_fit = np.concatenate([y] * len(idx))
        p_fit = np.concatenate([repeat_preds[i] for i in idx])
        model = fitter(y_fit, p_fit)
        calibrated[j] = applier(model, repeat_preds[j])
    return calibrated.mean(axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", action="append", required=True)
    ap.add_argument("--names", default=None)
    ap.add_argument("--out", default="submission_cv_ensemble.csv")
    ap.add_argument("--report", default="reports/cv_ensemble.json")
    ap.add_argument("--blend-preds", default="reports/cv_ensemble_predictions.npz")
    ap.add_argument("--folds", default="reports/folds_v2.npz")
    ap.add_argument("--no-calibration", action="store_true")
    ap.add_argument("--calibrations", default="temp,platt,iso")
    args = ap.parse_args()

    preds = [load_pred(p) for p in args.preds]
    y = preds[0]["y"]
    names = args.names.split(",") if args.names else [Path(p).stem for p in args.preds]
    for p in preds[1:]:
        if not np.array_equal(p["y"], y):
            raise ValueError("target mismatch")

    results = {}
    candidates = {}
    repeat_candidates = {}
    tests = {}
    for kind in ("prob", "logit", "rank"):
        w = (
            np.full(len(preds), 1.0 / len(preds))
            if kind == "rank"
            else optimized_weights(y, [p["oof"] for p in preds])
        )
        poof = blend_prob([p["oof"] for p in preds], w, kind)
        prepeat = np.mean(
            [
                blend_prob([p["repeats"][j] for p in preds], w, kind)
                for j in range(preds[0]["repeats"].shape[0])
            ],
            axis=0,
        )
        results[kind] = {"weights": w.tolist(), "proxy": proxy(y, poof)}
        candidates[kind] = poof
        repeat_candidates[kind] = prepeat
        tests[kind] = blend_prob([p["test"] for p in preds], w, kind)

    best_kind = max(results, key=lambda k: results[k]["proxy"])
    best_oof = candidates[best_kind]
    best_repeat = repeat_candidates[best_kind]
    best_test = tests[best_kind]

    repeat_preds = [
        blend_prob([p["repeats"][j] for p in preds], np.asarray(results[best_kind]["weights"]), best_kind)
        for j in range(preds[0]["repeats"].shape[0])
    ]

    cal = {}
    if not args.no_calibration:
        folds = load_folds()
        calibrators = {
            "temp": (fit_temp, apply_temp),
            "platt": (fit_platt, apply_platt),
            "iso": (fit_iso, apply_iso),
        }
        for name in [x for x in args.calibrations.split(",") if x]:
            fitter, applier = calibrators[name]
            p_cal = nested_calibrate(y, repeat_preds, fitter, applier)
            delta = paired_delta(best_repeat, p_cal, y, folds)
            cal[name] = {"nested_proxy": proxy(y, p_cal), "delta_vs_uncalibrated": delta}
            if delta["accept"] and delta["delta"] > 0:
                model = fitter(y, best_repeat)
                if name == "temp":
                    best_test = apply_temp(model, best_test)
                else:
                    best_test = applier(model, best_test)
                best_oof = p_cal
                best_kind = f"{best_kind}+{name}"

    report = {
        "names": names,
        "blends": results,
        "calibration": cal,
        "selected": best_kind,
        "selected_proxy": proxy(y, best_oof),
    }
    np.savez_compressed(args.blend_preds, y=y, oof=best_oof, test=best_test)
    _, _, sample = load_data(DATA_DIR)
    out = sample.copy()
    out["Target"] = best_test
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, float_format="%.17g")
    report["validation"] = validate_submission(args.out, sample)
    write_json(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
