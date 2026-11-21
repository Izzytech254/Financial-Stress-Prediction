"""Add a second seed to the finalized ensemble and update submission if better."""

from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from finalize import HIST_PARAMS, LGB_PARAMS, XGB_PARAMS
from solution import (
    DATA_DIR,
    ID,
    N_SELECTED_FEATURES,
    REPORT_DIR,
    ROOT,
    TARGET,
    ModelResult,
    blend_models,
    build_features,
    calibrate,
    log,
    run_cv,
    select_features,
)


def main() -> None:
    t0 = time.time()
    train = pd.read_csv(os.path.join(DATA_DIR, "Train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "Test.csv"))
    sample_sub = pd.read_csv(os.path.join(DATA_DIR, "SampleSubmission.csv"))
    X_train_df, X_test_df, y, feature_names = build_features(train, test)
    X_full = X_train_df.values.astype(np.float32)
    X_test_full = X_test_df.values.astype(np.float32)
    selected_idx, _ = select_features(X_full, y, N_SELECTED_FEATURES, seed=42)
    X = X_full[:, selected_idx]
    X_test = X_test_full[:, selected_idx]
    log(f"Selected {len(selected_idx)} features")

    old = np.load(os.path.join(REPORT_DIR, "preds.npz"))
    old_report = json.load(open(os.path.join(REPORT_DIR, "cv_results.json")))
    old_auc = float(old_report["final_oof_auc"])
    log(f"Current final OOF AUC={old_auc:.5f}")

    results = []
    for name, params in [("xgb", XGB_PARAMS), ("lgb", LGB_PARAMS), ("hist", HIST_PARAMS)]:
        new = run_cv(name, params, X, y, X_test, seeds=[2026])
        oof = (old[f"oof_{name}"] + new.oof) / 2.0
        test_pred = (old[f"test_{name}"] + new.test) / 2.0
        auc = float(roc_auc_score(y, oof))
        ll = float(log_loss(y, oof))
        log(f"{name}: 2-seed OOF AUC={auc:.5f} LogLoss={ll:.5f}")
        results.append(
            ModelResult(
                name=name,
                oof=oof,
                test=test_pred,
                auc=auc,
                logloss=ll,
                fold_aucs=new.fold_aucs,
                fold_loglosses=new.fold_loglosses,
                params=params,
            )
        )

    oof_blend, test_blend, comparison, blend_info = blend_models(results, y)
    oof_final, test_final, calibration_info = calibrate(oof_blend, test_blend, y)
    test_final = np.clip(test_final, 1e-6, 1 - 1e-6)
    final_auc = float(roc_auc_score(y, oof_final))
    final_ll = float(log_loss(y, oof_final))
    log(f"2-seed blend OOF AUC={final_auc:.5f} LogLoss={final_ll:.5f}")

    np.savez_compressed(
        os.path.join(REPORT_DIR, "preds_2seed.npz"),
        y=y,
        **{f"oof_{r.name}": r.oof for r in results},
        **{f"test_{r.name}": r.test for r in results},
        oof_blend=oof_blend,
        test_blend=test_blend,
        oof_final=oof_final,
        test_final=test_final,
    )

    report = {
        "models": {
            r.name: {"oof_auc": r.auc, "oof_logloss": r.logloss} for r in results
        },
        "blend_comparison": comparison,
        "blend_selected": blend_info["selected"],
        "blend_weights": dict(zip(blend_info["names"], blend_info["weights"])),
        "calibration": calibration_info,
        "final_oof_auc": final_auc,
        "final_oof_logloss": final_ll,
        "test_prediction_stats": {
            "mean": float(test_final.mean()),
            "std": float(test_final.std()),
            "min": float(test_final.min()),
            "max": float(test_final.max()),
        },
        "runtime_seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(REPORT_DIR, "cv_results_2seed.json"), "w") as f:
        json.dump(report, f, indent=2)

    if final_auc > old_auc:
        submission = pd.DataFrame({ID: test[ID].values, "Target": test_final})
        submission = sample_sub[[ID]].merge(submission, on=ID, how="left")
        assert submission["Target"].notna().all()
        assert (submission[ID].values == sample_sub[ID].values).all()
        submission.to_csv(
            os.path.join(ROOT, "submission.csv"), index=False, float_format="%.8f"
        )
        with open(os.path.join(REPORT_DIR, "cv_results.json"), "w") as f:
            json.dump(report, f, indent=2)
        log(f"Improved OOF AUC by {final_auc - old_auc:+.5f}; submission updated.")
    else:
        log(
            f"No OOF improvement ({final_auc - old_auc:+.5f}); kept existing submission."
        )
    log(f"Runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
