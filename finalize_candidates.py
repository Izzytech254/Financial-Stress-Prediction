from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, roc_auc_score

from frozen_cv import evaluate_folds, load_folds, paired_delta
from v6_common import DATA_DIR, EPS, TARGET, load_data, validate_submission, write_json


def composite(y: np.ndarray, p: np.ndarray) -> tuple[float, float, float]:
    p = np.clip(p, EPS, 1 - EPS)
    ll = float(log_loss(y, p))
    auc = float(roc_auc_score(y, p))
    return 0.6 * (1 - ll / 0.595) + 0.4 * auc, ll, auc


def date_rank(p: np.ndarray, d: np.ndarray) -> np.ndarray:
    out = np.empty(len(p), dtype=np.float64)
    for k in np.unique(d):
        idx = np.flatnonzero(d == k)
        order = idx[np.argsort(p[idx], kind="mergesort")]
        out[order] = np.arange(len(idx), dtype=np.float64) / max(1, len(idx) - 1)
    return np.clip(out, EPS, 1 - EPS)


def load_pred(path: str):
    z = np.load(path)
    required = {"y", "oof", "d_tr", "d_te", "test"}
    if not required.issubset(z.files):
        raise ValueError(f"missing keys in {path}: {required - set(z.files)}")
    reps = z["oof_repeats"] if "oof_repeats" in z.files else np.asarray([z["oof"]])
    if reps.ndim != 2 or reps.shape[1] != len(z["y"]):
        raise ValueError(f"bad repeats shape in {path}: {reps.shape}")
    return {
        "path": path,
        "y": z["y"],
        "oof": np.clip(z["oof"], EPS, 1 - EPS),
        "repeats": np.clip(reps, EPS, 1 - EPS),
        "d_tr": z["d_tr"],
        "d_te": z["d_te"],
        "test": np.clip(z["test"], EPS, 1 - EPS),
    }


def blend(probs: list[np.ndarray], weights: np.ndarray, mode: str) -> np.ndarray:
    p = np.column_stack([np.clip(x, EPS, 1 - EPS) for x in probs])
    if mode == "prob":
        return np.clip(p @ weights, EPS, 1 - EPS)
    return np.clip(expit(logit(p) @ weights), EPS, 1 - EPS)


def opt_weights(y: np.ndarray, preds: list[np.ndarray], mode: str = "logit") -> np.ndarray:
    if len(preds) == 1:
        return np.array([1.0])
    z = np.column_stack([logit(np.clip(p, EPS, 1 - EPS)) for p in preds])
    if mode == "prob":
        z = np.column_stack([np.clip(p, EPS, 1 - EPS) for p in preds])

    def f(w):
        p = np.clip(z @ w, EPS, 1 - EPS)
        if mode == "logit":
            p = expit(p)
        return -composite(y, p)[0]

    res = minimize(
        f,
        np.full(len(preds), 1 / len(preds)),
        method="SLSQP",
        bounds=[(0, 1)] * len(preds),
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        options={"maxiter": 500, "ftol": 1e-12},
    )
    w = np.clip(res.x, 0, 1)
    return w / w.sum()


def rank_calibrate(
    repeats: np.ndarray, test: np.ndarray, y: np.ndarray, d_tr: np.ndarray, d_te: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    ranks = np.asarray([date_rank(p, d_tr) for p in repeats])
    rank_te = date_rank(test, d_te)
    nested = np.empty_like(repeats)
    for j in range(len(repeats)):
        idx = [i for i in range(len(repeats)) if i != j]
        model = IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1 - EPS)
        model.fit(
            np.concatenate([ranks[i] for i in idx]),
            np.concatenate([y] * len(idx)),
        )
        nested[j] = model.predict(ranks[j])
    model = IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1 - EPS)
    model.fit(np.concatenate(ranks), np.concatenate([y] * len(repeats)))
    calibrated_test = model.predict(rank_te)
    return np.clip(nested.mean(axis=0), EPS, 1 - EPS), np.clip(calibrated_test, EPS, 1 - EPS)


def make_candidates(models: dict[str, dict], y: np.ndarray):
    out: dict[str, dict] = {}
    for name, m in models.items():
        out[name] = {
            "family": "raw_single",
            "oof": m["oof"],
            "test": m["test"],
            "weights": None,
            "members": [name],
            "mode": None,
        }
        oof, test = rank_calibrate(m["repeats"], m["test"], y, m["d_tr"], m["d_te"])
        out[name + "_rankcal"] = {
            "family": "rankcal_single",
            "oof": oof,
            "test": test,
            "weights": None,
            "members": [name],
            "mode": "rank_iso",
        }

    sets = {
        "lgb_equal": ["tuned", "final_tuned"],
        "lgb_lgb": ["tuned", "final_tuned"],
        "lgb_all": ["baseline", "trend_conc", "tuned", "final_tuned"],
        "lgb_three": ["baseline", "trend_conc", "tuned"],
        "all_models": ["baseline", "trend_conc", "tuned", "final_tuned", "cat", "xgb"],
    }
    for set_name, members in sets.items():
        present = [x for x in members if x in models]
        if len(present) < 2:
            continue
        for mode in ("prob", "logit"):
            w = opt_weights(y, [models[x]["oof"] for x in present], mode) if set_name != "lgb_equal" else np.full(len(present), 1 / len(present))
            oof = blend([models[x]["oof"] for x in present], w, mode)
            test = blend([models[x]["test"] for x in present], w, mode)
            name = f"{set_name}_{mode}"
            out[name] = {
                "family": "raw_blend",
                "oof": oof,
                "test": test,
                "weights": w,
                "members": present,
                "mode": mode,
            }
            if len(present) >= 2:
                rep_oof = np.asarray(
                    [
                        blend([models[x]["repeats"][j] for x in present], w, mode)
                        for j in range(len(models[present[0]]["repeats"]))
                    ]
                )
                oof_rank, test_rank = rank_calibrate(
                    rep_oof, test, y, models[present[0]]["d_tr"], models[present[0]]["d_te"]
                )
                out[name + "_rankcal"] = {
                    "family": "rankcal_blend",
                    "oof": oof_rank,
                    "test": test_rank,
                    "weights": w,
                    "members": present,
                    "mode": mode + "+rank_iso",
                }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", action="append", required=True, help="name=path")
    ap.add_argument("--baseline", default="baseline")
    ap.add_argument("--b-candidate", default=None)
    ap.add_argument("--report", default="reports/final_candidates.json")
    ap.add_argument("--out-prefix", default="submission")
    ap.add_argument("--skip-submissions", action="store_true")
    args = ap.parse_args()

    specs = dict(x.split("=", 1) for x in args.pred)
    models = {name: load_pred(path) for name, path in specs.items()}
    y = next(iter(models.values()))["y"]
    for m in models.values():
        if not np.array_equal(m["y"], y):
            raise ValueError("target mismatch")
        if not np.array_equal(m["d_tr"], next(iter(models.values()))["d_tr"]):
            raise ValueError("date mismatch")

    folds = load_folds()
    baseline = models[args.baseline]["oof"]
    candidates = make_candidates(models, y)
    rows = []
    for name, c in candidates.items():
        m = evaluate_folds(y, c["oof"], folds)
        d = paired_delta(baseline, c["oof"], y, folds)
        rows.append(
            {
                "name": name,
                "family": c["family"],
                "members": c["members"],
                "mode": c["mode"],
                "weights": None if c["weights"] is None else c["weights"].round(6).tolist(),
                "proxy": m["overall"]["proxy"],
                "log_loss": m["overall"]["log_loss"],
                "auc": m["overall"]["auc"],
                "fold_std": m["std_proxy"],
                "delta_vs_baseline": d["delta"],
                "delta_se": d["se"],
                "delta_accept": d["accept"],
            }
        )
    rows.sort(key=lambda r: r["proxy"], reverse=True)

    raw_best = next(r for r in rows if r["family"].startswith("raw"))
    rank_best = next((r for r in rows if r["family"].startswith("rankcal")), raw_best)
    if args.b_candidate:
        matches = [r for r in rows if r["name"] == args.b_candidate]
        if not matches:
            raise ValueError(f"unknown B candidate: {args.b_candidate}")
        diverse = matches[0]
    else:
        diverse = max(
            (r for r in rows if r["family"].startswith("raw") and r["name"] != raw_best["name"]),
            key=lambda r: r["proxy"],
            default=raw_best,
        )

    _, sample = load_data(DATA_DIR)[:3][::2]
    if not args.skip_submissions:
        for suffix, row in (("A", rank_best), ("B", diverse)):
            p = np.clip(candidates[row["name"]]["test"], 1e-5, 1 - 1e-5)
            out = sample.copy()
            out["Target"] = p
            path = Path(f"{args.out_prefix}_{suffix}.csv")
            out.to_csv(path, index=False, float_format="%.17g")
            stats = validate_submission(path, sample)
            row["submission"] = str(path)
            row["submission_stats"] = stats
            np.savez_compressed(
                f"reports/{args.out_prefix}_{suffix}_predictions.npz",
                y=y,
                oof=candidates[row["name"]]["oof"],
                test=p,
                d_tr=candidates[row["name"]]["oof"] * 0 + next(iter(models.values()))["d_tr"],
                d_te=next(iter(models.values()))["d_te"],
            )

    report = {
        "submission_A": rank_best["name"],
        "submission_B": diverse["name"],
        "baseline": args.baseline,
        "rows": rows,
    }
    write_json(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
