from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit, ndtri

from frozen_cv import evaluate_folds, load_folds, paired_delta
from v6_common import DATA_DIR, EPS, TARGET, load_data, per_date, write_json


def exact_proxy(y: np.ndarray, p: np.ndarray) -> tuple[float, float, float]:
    p = np.clip(p, EPS, 1 - EPS)
    from sklearn.metrics import log_loss, roc_auc_score
    ll = float(log_loss(y, p))
    auc = float(roc_auc_score(y, p))
    return 0.6 * (1 - ll / 0.595) + 0.4 * auc, ll, auc


def load_prediction(path: str) -> dict:
    z = np.load(path)
    out = {k: z[k] for k in z.files}
    if "oof_repeats" not in out:
        out["oof_repeats"] = out["oof"][None, :]
    return out


def recalibrate_prior(p: np.ndarray, d: np.ndarray) -> np.ndarray:
    return per_date(ndtri(np.clip(p, EPS, 1 - EPS)), d)


def reliability_table(y: np.ndarray, p: np.ndarray, groups: dict[str, np.ndarray], n_bins: int = 10) -> dict:
    z = pd.DataFrame({"y": y, "p": p})
    z["bucket"] = pd.qcut(z["p"], n_bins, labels=False, duplicates="drop")
    rows = []
    for b, part in z.groupby("bucket", observed=True):
        rows.append({
            "bucket": int(b),
            "n": len(part),
            "mean_pred": float(part["p"].mean()),
            "observed": float(part["y"].mean()),
            "bias_pred_minus_obs": float(part["p"].mean() - part["y"].mean()),
            "log_loss": float(-np.mean(part["y"] * np.log(np.clip(part["p"], EPS, 1)) + (1 - part["y"]) * np.log(np.clip(1 - part["p"], EPS, 1)))),
        })
    out = {"overall": rows, "segments": {}}
    for name, g in groups.items():
        seg = []
        for value, idx in pd.Series(g).groupby(g, sort=True).groups.items():
            idx = np.asarray(list(idx), dtype=int)
            if len(idx) < 100:
                continue
            proxy, ll, auc = exact_proxy(y[idx], p[idx])
            seg.append({
                "value": str(value), "n": len(idx), "mean_pred": float(p[idx].mean()),
                "observed": float(y[idx].mean()), "bias": float(p[idx].mean() - y[idx].mean()),
                "log_loss": ll, "auc": auc, "proxy": proxy,
            })
        out["segments"][name] = seg
    return out


def diagnostics() -> None:
    train, test, _ = load_data(DATA_DIR)
    b = load_prediction("reports/submission_B_predictions.npz")
    y, p, d = b["y"], b["oof"], b["d_tr"]
    z = train.copy()
    groups = {
        "date": d,
        "gender": z["gender"].to_numpy(),
        "region": z["region"].to_numpy(),
        "smartphone": z["smartphone"].to_numpy(),
        "segment": z["segment"].to_numpy(),
        "earning_pattern": z["earning_pattern"].to_numpy(),
        "arpu_quintile": pd.qcut(z["arpu"], 5, labels=False, duplicates="drop").to_numpy(),
        "age_quintile": pd.qcut(z["age"], 5, labels=False, duplicates="drop").to_numpy(),
        "activity_quintile": pd.qcut(z["x_90_d_activity_rate"], 5, labels=False, duplicates="drop").to_numpy(),
        "confidence": np.asarray(pd.cut(p, [0, 0.01, 0.05, 0.15, 0.5, 1], labels=False, include_lowest=True)),
        "missing_count": z.isna().sum(axis=1).to_numpy(),
    }
    rel = reliability_table(y, p, groups)
    logp = np.clip(p, EPS, 1 - EPS)
    loss = -(y * np.log(logp) + (1 - y) * np.log(1 - logp))
    order = np.argsort(-loss)[:500]
    wrong = []
    for i in order[:100]:
        wrong.append({
            "row": int(i), "date": int(d[i]), "y": int(y[i]), "p": float(p[i]),
            "loss": float(loss[i]), "gender": str(z.iloc[i]["gender"]),
            "region": str(z.iloc[i]["region"]), "segment": str(z.iloc[i]["segment"]),
            "earning_pattern": str(z.iloc[i]["earning_pattern"]),
            "arpu": float(z.iloc[i]["arpu"]), "age": float(z.iloc[i]["age"]),
            "activity": float(z.iloc[i]["x_90_d_activity_rate"]),
        })
    wrong_summary = {
        "false_positive_count": int(np.sum(y[order] == 0)),
        "false_negative_count": int(np.sum(y[order] == 1)),
        "date_counts": {str(k): int(v) for k, v in pd.Series(d[order]).value_counts().sort_index().items()},
        "region_counts": {str(k): int(v) for k, v in z.iloc[order]["region"].value_counts().items()},
        "segment_counts": {str(k): int(v) for k, v in z.iloc[order]["segment"].value_counts().items()},
        "mean_pred": float(p[order].mean()), "mean_loss": float(loss[order].mean()),
    }
    folds = load_folds()
    report = {
        "oof": evaluate_folds(y, p, folds)["overall"],
        "public_b": {"score": 0.744695, "log_loss": 0.223576, "auc": 0.925374},
        "reliability": rel,
        "top500_wrong_summary": wrong_summary,
        "top100_wrong": wrong,
        "missing_values_total": int(z.isna().sum().sum()),
    }
    write_json("reports/calibration_diagnostics.json", report)
    lines = ["# Calibration diagnostics for submission B", "", "## OOF vs public", ""]
    lines.append(f"OOF: proxy `{report['oof']['proxy']:.9f}`, LL `{report['oof']['log_loss']:.9f}`, AUC `{report['oof']['auc']:.9f}`.")
    lines.append("Public B: score `0.744695`, LL `0.223576`, AUC `0.925374`.")
    lines += ["", "## Reliability by prediction bucket", "", "| Bucket | N | Mean pred | Observed | Bias | LL |", "|---:|---:|---:|---:|---:|---:|"]
    for r in rel["overall"]:
        lines.append(f"| {r['bucket']} | {r['n']} | {r['mean_pred']:.6f} | {r['observed']:.6f} | {r['bias_pred_minus_obs']:+.6f} | {r['log_loss']:.6f} |")
    lines += ["", "## Worst-confidence buckets by segment", ""]
    for name in ("date", "segment", "confidence"):
        lines.append(f"### {name}")
        lines.append("| Value | N | Mean pred | Observed | Bias | LL | AUC |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for r in rel["segments"][name]:
            lines.append(f"| {r['value']} | {r['n']} | {r['mean_pred']:.6f} | {r['observed']:.6f} | {r['bias']:+.6f} | {r['log_loss']:.6f} | {r['auc']:.6f} |")
        lines.append("")
    lines += ["## Top-500 high-loss rows", "", f"- False positives: {wrong_summary['false_positive_count']}; false negatives: {wrong_summary['false_negative_count']}."]
    lines.append(f"- Mean predicted probability: `{wrong_summary['mean_pred']:.6f}`; mean loss: `{wrong_summary['mean_loss']:.6f}`.")
    lines.append(f"- Dates: `{wrong_summary['date_counts']}`; segments: `{wrong_summary['segment_counts']}`.")
    lines.append(f"- Missing values in training data: `{report['missing_values_total']}`.")
    Path("reports/calibration_diagnostics.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"oof": report["oof"], "top500": wrong_summary}, indent=2))


if __name__ == "__main__":
    diagnostics()
