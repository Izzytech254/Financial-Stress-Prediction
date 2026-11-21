from __future__ import annotations

import argparse
import subprocess
import sys

PY = ".venv/bin/python"

STAGES = {
    "manifest": [
        PY, "-c",
        "from frozen_cv import write_manifest; write_manifest('reports/folds_v2.npz')",
    ],
    "baseline": [
        PY, "baseline_cv.py", "--variant", "drop_m5_all", "--seeds", "50",
        "--threads", "8", "--test", "--tag", "baseline_cv_drop_m5_all_s50",
        "--report", "reports/baseline_cv_drop_m5_all_s50.json",
        "--preds", "reports/baseline_cv_drop_m5_all_s50_predictions.npz",
        "--out", "submission_baseline_cv_drop_m5_all_s50.csv",
    ],
    "features": [
        PY, "baseline_cv.py", "--variant", "drop_m5_all", "--seeds", "5",
        "--threads", "8", "--feature-groups", "trend,concentration",
        "--tag", "feat_trend_conc_s5",
        "--report", "reports/feat_trend_conc_s5.json",
        "--preds", "reports/feat_trend_conc_s5_predictions.npz",
    ],
    "features_test": [
        PY, "predict_test_only.py",
        "--from-report", "reports/feat_trend_conc_s5.json",
        "--from-preds", "reports/feat_trend_conc_s5_predictions.npz",
        "--seeds", "50", "--threads", "8",
        "--out", "submission_feat_trend_conc_s50.csv",
        "--report", "reports/feat_trend_conc_s50_test_only.json",
    ],
    "features_preds": [
        PY, "-c",
        (
            "import numpy as np,pandas as pd;"
            "z=np.load('reports/feat_trend_conc_s5_predictions.npz');"
            "t=pd.read_csv('submission_feat_trend_conc_s50.csv')['Target'].to_numpy();"
            "d={k:z[k] for k in z.files};d['test']=t;"
            "np.savez_compressed('reports/feat_trend_conc_s50_predictions.npz',**d)"
        ),
    ],
    "ensemble2": [
        PY, "ensemble_cv.py",
        "--preds", "reports/baseline_cv_drop_m5_all_s50_predictions.npz",
        "--preds", "reports/feat_trend_conc_s50_predictions.npz",
        "--names", "baseline,trend_conc",
        "--out", "submission_ensemble_baseline_trendconc.csv",
        "--report", "reports/ensemble_baseline_trendconc.json",
        "--blend-preds", "reports/ensemble_baseline_trendconc_predictions.npz",
    ],
    "ensemble": [
        PY, "ensemble_cv.py",
        "--preds", "reports/baseline_cv_drop_m5_all_s50_predictions.npz",
        "--preds", "reports/feat_trend_conc_s50_predictions.npz",
        "--preds", "reports/tuned_trend_conc_s10_predictions.npz",
        "--names", "baseline,trend_conc,tuned",
        "--out", "submission_ensemble_three_way.csv",
        "--report", "reports/ensemble_three_way.json",
        "--blend-preds", "reports/ensemble_three_way_predictions.npz",
    ],
    "tune": [
        PY, "tune_lgb_cv.py", "--feature-groups", "trend,concentration",
        "--trials", "50", "--seeds", "10", "--threads", "8",
        "--max-rounds", "150", "--patience", "25",
        "--tag", "tuned_trend_conc_s10",
        "--report", "reports/tuned_trend_conc_s10.json",
        "--preds", "reports/tuned_trend_conc_s10_predictions.npz",
        "--out", "submission_tuned_trend_conc_s10.csv", "--test",
    ],
}

ORDER = [
    "manifest", "baseline", "features", "features_test",
    "features_preds", "ensemble2", "tune", "ensemble",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=[*ORDER, "all"], default="all")
    args = ap.parse_args()
    stages = ORDER if args.stage == "all" else [args.stage]
    for stage in stages:
        cmd = STAGES[stage]
        print(f"+ {stage}: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    sys.exit(main())
