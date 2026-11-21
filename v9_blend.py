from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

from v6_common import (
    DATA_DIR,
    ID,
    REPORT_DIR,
    composite,
    load_data,
    per_date,
    probit_eta,
    validate_submission,
    write_json,
)


def calibrate_logit(eta: np.ndarray, d: np.ndarray) -> np.ndarray:
    out = np.empty(len(eta), dtype=np.float64)
    for k in np.unique(d):
        m = d == k
        # solve mean(sigmoid(c + eta)) = 0.15
        from scipy.optimize import brentq

        c = brentq(
            lambda c: expit(c + eta[m]).mean() - 0.15,
            -30.0,
            30.0,
            xtol=1e-14,
        )
        out[m] = expit(c + eta[m])
    return np.clip(out, 1e-15, 1 - 1e-15)


def load_model(path: Path):
    z = np.load(path)
    if "y" not in z.files or "oof" not in z.files:
        raise ValueError(f"{path} missing y/oof")
    return z


def blend_score(y, oofs, weights, d_tr, mode):
    if mode == "logit":
        z = sum(w * logit(np.clip(p, 1e-12, 1 - 1e-12)) for w, p in zip(weights, oofs))
        p = calibrate_logit(z, d_tr)
    elif mode == "probit":
        z = sum(w * probit_eta(p) for w, p in zip(weights, oofs))
        p = per_date(z, d_tr)
    else:
        p = sum(w * p for w, p in zip(weights, oofs))
        p = np.clip(p, 1e-15, 1 - 1e-15)
    return composite(y, p), p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", action="append", required=True)
    ap.add_argument("--names", default=None)
    ap.add_argument("--out", default="submission_v9_blend.csv")
    ap.add_argument("--report", default=str(REPORT_DIR / "v9_blend.json"))
    ap.add_argument("--blend-preds", default=str(REPORT_DIR / "v9_blend_predictions.npz"))
    ap.add_argument("--mode", default="logit", choices=["logit", "probit", "prob"])
    ap.add_argument("--equal", action="store_true")
    ap.add_argument("--weights", default=None, help="comma-separated fixed weights")
    args = ap.parse_args()

    paths = [Path(p) for p in args.preds]
    names = args.names.split(",") if args.names else [p.stem for p in paths]
    if len(names) != len(paths):
        raise ValueError("--names count must match --preds count")
    models = [load_model(p) for p in paths]
    y = models[0]["y"]
    d_tr = models[0]["d_tr"]
    d_te = models[0]["d_te"]
    oofs = [m["oof"] for m in models]
    tests = [m["test"] for m in models]
    for m in models[1:]:
        if not np.array_equal(m["y"], y):
            raise ValueError("y mismatch")
        if not np.array_equal(m["d_tr"], d_tr):
            raise ValueError("d_tr mismatch")

    n = len(oofs)
    if args.weights:
        weights = np.array([float(x) for x in args.weights.split(",")])
        if len(weights) != n:
            raise ValueError("--weights count must match --preds count")
        weights = weights / weights.sum()
    elif args.equal:
        weights = np.full(n, 1.0 / n)
    elif n == 2:
        best = None
        for w0 in np.linspace(0.0, 1.0, 101):
            w = np.array([w0, 1.0 - w0])
            (sc, _, _), _ = blend_score(y, oofs, w, d_tr, args.mode)
            if best is None or sc > best[0]:
                best = (sc, w)
        weights = best[1]
    elif n == 3:
        best = None
        for w0 in np.linspace(0.0, 1.0, 21):
            for w1 in np.linspace(0.0, 1.0 - w0, 21):
                w = np.array([w0, w1, 1.0 - w0 - w1])
                (sc, _, _), _ = blend_score(y, oofs, w, d_tr, args.mode)
                if best is None or sc > best[0]:
                    best = (sc, w)
        weights = best[1]
    else:
        rng = np.random.default_rng(42)
        candidates = [np.full(n, 1.0 / n)]
        candidates += [rng.dirichlet(np.ones(n)) for _ in range(400)]
        best = None
        for w in candidates:
            (sc, _, _), _ = blend_score(y, oofs, w, d_tr, args.mode)
            if best is None or sc > best[0]:
                best = (sc, w)
        weights = best[1]

    (score, ll, auc), p_oof = blend_score(y, oofs, weights, d_tr, args.mode)
    print(
        "blend mode=%s score=%.9f ll=%.9f auc=%.9f weights=%s"
        % (args.mode, score, ll, auc, np.round(weights, 6).tolist()),
        flush=True,
    )

    if args.mode == "logit":
        zt = sum(
            w * logit(np.clip(t, 1e-12, 1 - 1e-12))
            for w, t in zip(weights, tests)
        )
        p_test = calibrate_logit(zt, d_te)
    elif args.mode == "probit":
        zt = sum(w * probit_eta(t) for w, t in zip(weights, tests))
        p_test = per_date(zt, d_te)
    else:
        p_test = sum(w * t for w, t in zip(weights, tests))
        p_test = np.clip(p_test, 1e-15, 1 - 1e-15)

    _, test, sample = load_data(DATA_DIR)
    out = sample.copy()
    out["Target"] = p_test
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, float_format="%.17g")
    stats = validate_submission(args.out, sample)

    Path(args.blend_preds).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.blend_preds,
        y=y,
        oof=p_oof,
        test=p_test,
        d_tr=d_tr,
        d_te=d_te,
    )
    payload = {
        "model": "v9_blend",
        "mode": args.mode,
        "names": names,
        "preds": [str(p.resolve()) for p in paths],
        "weights": weights.tolist(),
        "validation": {"composite": score, "log_loss": ll, "auc": auc},
        "test": stats,
        "submission": str(Path(args.out).resolve()),
    }
    write_json(Path(args.report), payload)
    print(f"report={args.report}", flush=True)


if __name__ == "__main__":
    main()
