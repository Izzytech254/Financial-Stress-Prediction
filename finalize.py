"""Fast finalization: train the tuned ensemble and write submission.csv.

CatBoost is intentionally skipped here because it dominated runtime; the
XGBoost + LightGBM + HistGradientBoosting blend is the primary ensemble.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from solution import (
    CATBOOST_SEEDS,
    CV_SEEDS,
    DATA_DIR,
    ID,
    N_SELECTED_FEATURES,
    REPORT_DIR,
    ROOT,
    TARGET,
    blend_models,
    build_features,
    calibrate,
    log,
    run_cv,
    select_features,
)

os.makedirs(REPORT_DIR, exist_ok=True)

XGB_PARAMS = {
    "learning_rate": 0.021795741277819067,
    "max_depth": 9,
    "min_child_weight": 18.043311207136256,
    "subsample": 0.9425192044349383,
    "colsample_bytree": 0.5523068845866854,
    "reg_alpha": 0.002458603276328005,
    "reg_lambda": 1.8141523033666782,
    "gamma": 2.2007624686980067,
}
LGB_PARAMS = {
    "learning_rate": 0.023686687710753017,
    "num_leaves": 70,
    "max_depth": 5,
    "min_child_samples": 93,
    "subsample": 0.777633994480005,
    "colsample_bytree": 0.731261142176991,
    "reg_alpha": 0.017654048052495083,
    "reg_lambda": 0.5209069329868288,
}
HIST_PARAMS = {
    "learning_rate": 0.045826461745188736,
    "max_leaf_nodes": 47,
    "min_samples_leaf": 69,
    "l2_regularization": 0.003980771770257045,
    "max_bins": 255,
}


def main() -> None:
    t0 = time.time()
    train = pd.read_csv(os.path.join(DATA_DIR, "Train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "Test.csv"))
    sample_sub = pd.read_csv(os.path.join(DATA_DIR, "SampleSubmission.csv"))
    log(f"Train={train.shape} Test={test.shape} target_rate={train[TARGET].mean():.4f}")

    X_train_df, X_test_df, y, feature_names = build_features(train, test)
    log(f"Engineered features: {X_train_df.shape[1]}")
    X_full = X_train_df.values.astype(np.float32)
    X_test_full = X_test_df.values.astype(np.float32)

    selected_idx, importance = select_features(X_full, y, N_SELECTED_FEATURES, seed=42)
    selected_names = [feature_names[i] for i in selected_idx]
    X = X_full[:, selected_idx]
    X_test = X_test_full[:, selected_idx]
    log(f"Selected {len(selected_idx)} features")

    pd.DataFrame({"feature": feature_names, "importance": importance}).sort_values(
        "importance", ascending=False
    ).to_csv(os.path.join(REPORT_DIR, "feature_importance.csv"), index=False)

    results = []
    results.append(run_cv("xgb", XGB_PARAMS, X, y, X_test, seeds=[42]))
    log("Saved XGB progress; continuing with LightGBM.")
    results.append(run_cv("lgb", LGB_PARAMS, X, y, X_test, seeds=[42]))
    log("Saved LightGBM progress; continuing with HistGradientBoosting.")
    results.append(run_cv("hist", HIST_PARAMS, X, y, X_test, seeds=[42]))

    oof_blend, test_blend, comparison, blend_info = blend_models(results, y)
    oof_final, test_final, calibration_info = calibrate(oof_blend, test_blend, y)
    test_final = np.clip(test_final, 1e-6, 1 - 1e-6)

    final_auc = float(roc_auc_score(y, oof_final))
    final_ll = float(log_loss(y, oof_final))
    log(f"FINAL OOF AUC={final_auc:.5f} LogLoss={final_ll:.5f}")
    log(f"Test mean={test_final.mean():.4f} min={test_final.min():.6f} max={test_final.max():.6f}")

    submission = pd.DataFrame({ID: test[ID].values, "Target": test_final})
    submission = sample_sub[[ID]].merge(submission, on=ID, how="left")
    assert submission["Target"].notna().all()
    assert (submission[ID].values == sample_sub[ID].values).all()
    out_path = os.path.join(ROOT, "submission.csv")
    submission.to_csv(out_path, index=False, float_format="%.8f")
    log(f"Saved submission to {out_path}")

    # Save predictions for later reuse / further blending.
    np.savez_compressed(
        os.path.join(REPORT_DIR, "preds.npz"),
        y=y,
        **{f"oof_{r.name}": r.oof for r in results},
        **{f"test_{r.name}": r.test for r in results},
        oof_blend=oof_blend,
        test_blend=test_blend,
        oof_final=oof_final,
        test_final=test_final,
    )

    report = {
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "target_rate": float(train[TARGET].mean()),
        "n_engineered_features": int(X_train_df.shape[1]),
        "n_selected_features": int(len(selected_idx)),
        "models": {
            r.name: {
                "oof_auc": r.auc,
                "oof_logloss": r.logloss,
                "fold_aucs": r.fold_aucs,
                "fold_loglosses": r.fold_loglosses,
                "params": r.params,
            }
            for r in results
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
    with open(os.path.join(REPORT_DIR, "cv_results.json"), "w") as f:
        json.dump(report, f, indent=2)
    log(f"Saved report to {os.path.join(REPORT_DIR, 'cv_results.json')}")
    log(f"Runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
