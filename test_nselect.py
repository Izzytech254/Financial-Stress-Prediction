"""Sweep feature-selection size on the v3 feature set (seed 42)."""

from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from features_v2 import build_v2_features
from features_v3 import build_history_features_v3
from improve_v3 import LGB_TUNED, _last_row_idx, _metrics, log
from improve_v4 import run_lgb
from run_v2 import stack, paired
from solution import DATA_DIR, REPORT_DIR

T0 = time.time()


def main() -> None:
    train = pd.read_csv(os.path.join(DATA_DIR, "Train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "Test.csv"))
    last = _last_row_idx(train)
    feats = build_v2_features(train, test)
    y = feats["y"]
    hist3 = build_history_features_v3(train, test)
    X3, Xte3, _ = stack(feats["base"], [feats["cust_tt"], hist3])
    log(f"X3={X3.shape}")

    zfold = np.load(os.path.join(REPORT_DIR, "folds_v2.npz"))
    folds = [(zfold[f"tr_42_{i}"], zfold[f"va_42_{i}"]) for i in range(5)]

    z4 = np.load(os.path.join(REPORT_DIR, "v4_predictions.npz"))
    ref = z4["oof_v3_lgb_400"]
    ref_m = _metrics(y, ref, last)
    log(f"reference v3_lgb_400 {ref_m}")

    preds = {"ref_400": (ref, z4["test_v3_lgb_400"])}
    results = {"ref_400": ref_m}
    deltas = {}

    for n in [500, 600, 700]:
        name = f"v3_lgb_{n}"
        log(f"Run {name} ...")
        p, t, _, _ = run_lgb(X3, y, Xte3, folds, n_select=n, seed=42, params=LGB_TUNED)
        preds[name] = (p, t)
        results[name] = _metrics(y, p, last)
        da, dase, dl, dlse = paired(y, p, ref, folds)
        deltas[name] = {
            "delta_auc": float(da),
            "delta_auc_se": float(dase),
            "delta_ll": float(dl),
            "delta_ll_se": float(dlse),
        }
        log(
            f"{name} {results[name]} dAUC={da:+.6f}+/-{dase:.6f} "
            f"dLL={dl:+.6f}+/-{dlse:.6f}"
        )
        np.savez_compressed(
            os.path.join(REPORT_DIR, "v4_nselect_predictions.npz"),
            y=y,
            last=last,
            **{f"oof_{k}": v[0] for k, v in preds.items()},
            **{f"test_{k}": v[1] for k, v in preds.items()},
        )

    report = {"reference": "v3_lgb_400", "results": results, "deltas": deltas}
    with open(os.path.join(REPORT_DIR, "v4_nselect.json"), "w") as f:
        json.dump(report, f, indent=2)
    log(f"Done. Total runtime {time.time() - T0:.1f}s")


if __name__ == "__main__":
    main()
