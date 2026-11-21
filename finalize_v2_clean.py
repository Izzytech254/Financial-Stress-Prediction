"""Finalize v2 with only the leak-free new models (LightGBM + CatBoost)."""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.join(ROOT, "reports")
ID = "ID"

z = np.load(os.path.join(REPORT_DIR, "v2_predictions.npz"))
base = np.load(os.path.join(REPORT_DIR, "preds_2seed.npz"))
y = z["y"]
P = np.clip(np.column_stack([z["oof_lgb_v2"], z["oof_cat_v2"]]), 1e-6, 1 - 1e-6)
T = np.clip(np.column_stack([z["test_lgb_v2"], z["test_cat_v2"]]), 1e-6, 1 - 1e-6)


def lgt(p):
    return logit(np.clip(p, 1e-6, 1 - 1e-6))


p_raw = expit(lgt(P).mean(axis=1))
t_raw = expit(lgt(T).mean(axis=1))

skf = StratifiedKFold(5, shuffle=True, random_state=42)
p_cal_cv = np.zeros(len(y))
for tr, va in skf.split(p_raw, y):
    lr = LogisticRegression(C=1e6, max_iter=1000, solver="lbfgs")
    lr.fit(lgt(p_raw[tr]).reshape(-1, 1), y[tr])
    p_cal_cv[va] = lr.predict_proba(lgt(p_raw[va]).reshape(-1, 1))[:, 1]

lr = LogisticRegression(C=1e6, max_iter=1000, solver="lbfgs")
lr.fit(lgt(p_raw).reshape(-1, 1), y)
t_final = lr.predict_proba(lgt(t_raw).reshape(-1, 1))[:, 1]

ll_before = float(log_loss(y, p_raw))
ll_after = float(log_loss(y, p_cal_cv))
apply_cal = ll_after <= ll_before + 1e-6
p_final = p_cal_cv if apply_cal else p_raw
t_final = t_final if apply_cal else t_raw

final_auc = float(roc_auc_score(y, p_final))
final_ll = float(ll_after if apply_cal else ll_before)
p_base = base["oof_final"]
baseline_auc = float(roc_auc_score(y, p_base))

folds = np.load(os.path.join(REPORT_DIR, "folds_v2.npz"))
seed42 = [(folds[f"tr_42_{i}"], folds[f"va_42_{i}"]) for i in range(5)]
da, dl = [], []
for _, va in seed42:
    da.append(float(roc_auc_score(y[va], p_final[va]) - roc_auc_score(y[va], p_base[va])))
    dl.append(float(log_loss(y[va], p_final[va]) - log_loss(y[va], p_base[va])))
da_mean, da_se = float(np.mean(da)), float(np.std(da, ddof=1) / np.sqrt(len(da)))
dl_mean, dl_se = float(np.mean(dl)), float(np.std(dl, ddof=1) / np.sqrt(len(dl)))
accept = (da_mean > da_se) and (dl_mean <= max(dl_se, 1e-4)) and (final_auc > baseline_auc)

print(f"AUC={final_auc:.5f} LL_cv={final_ll:.5f} dAUC={da_mean:+.5f}+/-{da_se:.5f} "
      f"dLL={dl_mean:+.5f}+/-{dl_se:.5f} accept={accept}")

if accept:
    test = pd.read_csv(os.path.join(ROOT, "data", "Test.csv"))
    sample = pd.read_csv(os.path.join(ROOT, "data", "SampleSubmission.csv"))
    sub = pd.DataFrame({ID: test[ID].values, "Target": np.clip(t_final, 1e-6, 1 - 1e-6)})
    sub = sample[[ID]].merge(sub, on=ID, how="left")
    assert sub["Target"].notna().all()
    assert (sub[ID].values == sample[ID].values).all()
    sub.to_csv(os.path.join(ROOT, "submission.csv"), index=False, float_format="%.8f")
    print("Wrote submission.csv")

np.savez_compressed(
    os.path.join(REPORT_DIR, "v2_clean_predictions.npz"),
    y=y, oof_final=p_final, test_final=t_final,
)

report_path = os.path.join(REPORT_DIR, "cv_results_v2.json")
report = json.load(open(report_path))
report["final"] = {
    "feature_set": "combined",
    "models": ["lgb_v2", "cat_v2"],
    "blend_mode": "mean_logit",
    "blend_weights": None,
    "logloss_cv_before_calibration": ll_before,
    "logloss_cv_after_calibration": ll_after,
    "calibration_applied": bool(apply_cal),
    "final_oof_auc": final_auc,
    "final_oof_logloss_cv": final_ll,
    "delta_auc_vs_baseline": da_mean,
    "delta_auc_se": da_se,
    "delta_ll_vs_baseline": dl_mean,
    "delta_ll_se": dl_se,
    "accepted": bool(accept),
    "legacy_baseline_models_in_blend": False,
}
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)
print("Updated cv_results_v2.json")
