from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

from frozen_cv import load_folds, paired_delta
from v6_common import EPS, load_data, validate_submission, write_json


def date_rank(p: np.ndarray, d: np.ndarray) -> np.ndarray:
    ranks = np.empty(len(p), dtype=np.float64)
    for date in np.unique(d):
        idx = np.flatnonzero(d == date)
        order = idx[np.argsort(p[idx], kind="mergesort")]
        ranks[order] = np.arange(len(idx), dtype=np.float64) / max(1, len(idx) - 1)
    return np.clip(ranks, EPS, 1 - EPS)


def proxy(y: np.ndarray, p: np.ndarray) -> float:
    return 0.6 * (1 - log_loss(y, np.clip(p, EPS, 1 - EPS)) / 0.595) + 0.4 * roc_auc_score(y, p)


def fit_map(rank: np.ndarray, y: np.ndarray, kind: str):
    if kind == "iso":
        m = IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1 - EPS)
        m.fit(rank, y)
        return m
    m = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    m.fit(np.clip(logit(rank), -12, 12)[:, None], y)
    return m


def apply_map(model, rank: np.ndarray, kind: str) -> np.ndarray:
    if kind == "iso":
        return np.clip(model.predict(rank), EPS, 1 - EPS)
    return model.predict_proba(np.clip(logit(rank), -12, 12)[:, None])[:, 1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--kind", choices=["iso", "logit"], default="iso")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--blend-preds", required=True)
    args = ap.parse_args()

    z = np.load(args.preds)
    y, d_tr, d_te = z["y"], z["d_tr"], z["d_te"]
    repeats = z["oof_repeats"]
    test = z["test"]
    rank_tr = [date_rank(p, d_tr) for p in repeats]
    rank_te = date_rank(test, d_te)

    folds = load_folds()
    nested = np.empty((len(repeats), len(y)), dtype=np.float64)
    for j in range(len(repeats)):
        idx = [i for i in range(len(repeats)) if i != j]
        model = fit_map(np.concatenate([rank_tr[i] for i in idx]), np.concatenate([y] * len(idx)), args.kind)
        nested[j] = apply_map(model, rank_tr[j], args.kind)
    nested_oof = nested.mean(axis=0)
    delta = paired_delta(repeats.mean(axis=0), nested_oof, y, folds)

    model = fit_map(np.concatenate(rank_tr), np.concatenate([y] * len(repeats)), args.kind)
    p_test = apply_map(model, rank_te, args.kind)
    _, _, sample = load_data("data")
    out = sample.copy()
    out["Target"] = p_test
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, float_format="%.17g")
    stats = validate_submission(args.out, sample)
    np.savez_compressed(args.blend_preds, y=y, oof=nested_oof, test=p_test)

    report = {
        "source": args.preds,
        "kind": args.kind,
        "raw_proxy": proxy(y, repeats.mean(axis=0)),
        "nested_proxy": proxy(y, nested_oof),
        "delta_vs_raw": delta,
        "test": stats,
        "submission": args.out,
    }
    write_json(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
