"""v5: 3-seed LGB/CatBoost on the v3 feature set with N_SELECT=500."""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import log_loss, roc_auc_score

from features_v2 import build_v2_features
from features_v3 import build_history_features_v3
from improve_v3 import (
    CAT_TUNED,
    LGB_TUNED,
    _blend_report,
    _last_row_idx,
    _logit,
    _metrics,
    log,
    run_cat_cv,
)
from improve_v4 import run_lgb
from run_v2 import CAT_PARAMS, stack, paired
from solution import CAT_COLS, DATA_DIR, REPORT_DIR, ROOT, ID, TARGET

SEEDS = [42, 2026, 7]
N_SELECT = 500
N_FOLDS = 5
T0 = time.time()


def elapsed() -> float:
    return time.time() - T0


def mean_logit(preds: List[np.ndarray]) -> np.ndarray:
    P = np.clip(np.column_stack(preds), 1e-6, 1 - 1e-6)
    return expit(_logit(P).mean(axis=1))


def main() -> None:
    train = pd.read_csv(os.path.join(DATA_DIR, "Train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "Test.csv"))
    sample = pd.read_csv(os.path.join(DATA_DIR, "SampleSubmission.csv"))
    last = _last_row_idx(train)

    feats = build_v2_features(train, test)
    y = feats["y"]
    hist3 = build_history_features_v3(train, test)
    X3, Xte3, names = stack(feats["base"], [feats["cust_tt"], hist3])
    log(f"X3={X3.shape}")

    zfold = np.load(os.path.join(REPORT_DIR, "folds_v2.npz"))
    folds_by_seed = {
        s: [(zfold[f"tr_{s}_{i}"], zfold[f"va_{s}_{i}"]) for i in range(N_FOLDS)]
        for s in SEEDS
    }

    # Seed 42 LGB is already available from the N_SELECT sweep.
    zn = np.load(os.path.join(REPORT_DIR, "v4_nselect_predictions.npz"))
    p_lgb_seed42 = zn["oof_v3_lgb_500"]
    t_lgb_seed42 = zn["test_v3_lgb_500"]
    log(f"LGB seed42 N500 {_metrics(y, p_lgb_seed42, last)}")

    p_lgb_parts = [p_lgb_seed42]
    t_lgb_parts = [t_lgb_seed42]
    for seed in [2026, 7]:
        log(f"Train LGB seed {seed} ...")
        p, t, _, _ = run_lgb(
            X3, y, Xte3, folds_by_seed[seed], n_select=N_SELECT, seed=seed, params=LGB_TUNED
        )
        p_lgb_parts.append(p)
        t_lgb_parts.append(t)
        log(f"LGB seed {seed} {_metrics(y, p, last)}")
        np.savez_compressed(
            os.path.join(REPORT_DIR, "v5_lgb_parts.npz"),
            y=y,
            last=last,
            **{f"oof_lgb_s{s}": pp for s, pp in zip(SEEDS[: len(p_lgb_parts)], p_lgb_parts)},
            **{f"test_lgb_s{s}": tt for s, tt in zip(SEEDS[: len(t_lgb_parts)], t_lgb_parts)},
        )

    p_lgb3 = np.mean(p_lgb_parts, axis=0)
    t_lgb3 = np.mean(t_lgb_parts, axis=0)
    log(f"LGB 3-seed N500 {_metrics(y, p_lgb3, last)}")
    np.savez_compressed(
        os.path.join(REPORT_DIR, "v5_lgb3.npz"),
        y=y,
        last=last,
        oof_lgb3=p_lgb3,
        test_lgb3=t_lgb3,
        **{f"oof_lgb_s{s}": pp for s, pp in zip(SEEDS, p_lgb_parts)},
        **{f"test_lgb_s{s}": tt for s, tt in zip(SEEDS, t_lgb_parts)},
    )

    cat_train = train[CAT_COLS].astype(str).values
    cat_test = test[CAT_COLS].astype(str).values
    p_cat_parts = []
    t_cat_parts = []
    for seed in SEEDS:
        log(f"Train Cat seed {seed} ...")
        p, t, _, _ = run_cat_cv(
            X3,
            y,
            Xte3,
            folds_by_seed[seed],
            names,
            cat_train,
            cat_test,
            N_SELECT,
            seed,
            CAT_PARAMS,
            300,
        )
        p_cat_parts.append(p)
        t_cat_parts.append(t)
        log(f"Cat seed {seed} {_metrics(y, p, last)}")
        np.savez_compressed(
            os.path.join(REPORT_DIR, "v5_cat_parts.npz"),
            y=y,
            last=last,
            **{f"oof_cat_s{s}": pp for s, pp in zip(SEEDS[: len(p_cat_parts)], p_cat_parts)},
            **{f"test_cat_s{s}": tt for s, tt in zip(SEEDS[: len(t_cat_parts)], t_cat_parts)},
        )

    p_cat3 = np.mean(p_cat_parts, axis=0)
    t_cat3 = np.mean(t_cat_parts, axis=0)
    log(f"Cat 3-seed N500 {_metrics(y, p_cat3, last)}")
    np.savez_compressed(
        os.path.join(REPORT_DIR, "v5_cat3.npz"),
        y=y,
        last=last,
        oof_cat3=p_cat3,
        test_cat3=t_cat3,
        **{f"oof_cat_s{s}": pp for s, pp in zip(SEEDS, p_cat_parts)},
        **{f"test_cat_s{s}": tt for s, tt in zip(SEEDS, t_cat_parts)},
    )

    # Blend with the existing v2 models.
    z2 = np.load(os.path.join(REPORT_DIR, "v2_predictions.npz"))
    zclean = np.load(os.path.join(REPORT_DIR, "v2_clean_predictions.npz"))
    p_v2 = zclean["oof_final"]
    folds42 = folds_by_seed[42]
    preds = {
        "lgb3_v3_500": (p_lgb3, t_lgb3),
        "cat3_v3_500": (p_cat3, t_cat3),
        "lgb_v2_3seed": (z2["oof_lgb_v2"], z2["test_lgb_v2"]),
        "cat_v2": (z2["oof_cat_v2"], z2["test_cat_v2"]),
    }

    def blend(keys: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        P = np.clip(np.column_stack([preds[k][0] for k in keys]), 1e-6, 1 - 1e-6)
        T = np.clip(np.column_stack([preds[k][1] for k in keys]), 1e-6, 1 - 1e-6)
        return expit(_logit(P).mean(axis=1)), expit(_logit(T).mean(axis=1))

    candidate_keys = [
        ["lgb3_v3_500"],
        ["lgb3_v3_500", "cat3_v3_500"],
        ["lgb3_v3_500", "cat3_v3_500", "lgb_v2_3seed"],
        ["lgb3_v3_500", "cat3_v3_500", "lgb_v2_3seed", "cat_v2"],
        ["lgb3_v3_500", "cat_v2", "lgb_v2_3seed"],
    ]
    reports = {}
    for keys in candidate_keys:
        p, t = blend(keys)
        name = "+".join(keys)
        r = _blend_report(name, p, t, y, last, p_v2, folds42)
        reports[name] = r
        ps = r["_p_shift_oof"]
        log(
            f"Blend {name}: AUC={r['auc']:.6f} LL={r['logloss']:.6f} "
            f"lastAUC={r['last_auc']:.6f} lastLL={r['last_logloss']:.6f} "
            f"calAUC={roc_auc_score(y, ps):.6f} calLL={log_loss(y, ps):.6f} "
            f"lastCalAUC={roc_auc_score(y[last], ps[last]):.6f} "
            f"lastCalLL={log_loss(y[last], ps[last]):.6f} "
            f"plattCV={r['platt_cv_ll']:.6f}"
        )

    def proxy(r: dict) -> float:
        return 0.6 * r["platt_cv_ll"] + 0.4 * (1.0 - r["auc"])

    best_name = min(reports, key=lambda k: proxy(reports[k]))
    best = reports[best_name]
    log(f"Best {best_name} proxy={proxy(best):.6f}")

    p_test = np.clip(best["_p_shift_test"], 1e-6, 1 - 1e-6)
    sub = pd.DataFrame({ID: test[ID].values, "Target": p_test})
    sub = sample[[ID]].merge(sub, on=ID, how="left")
    assert sub["Target"].notna().all()
    assert (sub[ID].values == sample[ID].values).all()
    out_path = os.path.join(ROOT, "submission_v5.csv")
    sub.to_csv(out_path, index=False, float_format="%.8f")
    log(f"Wrote {out_path} mean={p_test.mean():.6f} min={p_test.min():.6f} max={p_test.max():.6f}")

    clean = {
        k: {kk: vv for kk, vv in r.items() if not kk.startswith("_")}
        for k, r in reports.items()
    }
    report = {
        "lgb3_v3_500": _metrics(y, p_lgb3, last),
        "cat3_v3_500": _metrics(y, p_cat3, last),
        "blends": clean,
        "best_blend": best_name,
        "best_proxy": float(proxy(best)),
        "best_shift": {
            "platt_slope": best["platt_slope"],
            "platt_intercept": best["platt_intercept"],
            "prior_shift": best["prior_shift"],
            "test_mean_raw": best["test_mean_raw"],
            "test_mean_shift": best["test_mean_shift"],
        },
        "submission_file": "submission_v5.csv",
        "runtime_seconds": float(elapsed()),
    }
    with open(os.path.join(REPORT_DIR, "improve_v5.json"), "w") as f:
        json.dump(report, f, indent=2)

    lines = [
        "# v5: 3-seed LGB/Cat on v3 features",
        "",
        f"- LGB 3-seed N500: AUC `{report['lgb3_v3_500']['auc']:.6f}`, LL `{report['lgb3_v3_500']['logloss']:.6f}`",
        f"- Cat 3-seed N500: AUC `{report['cat3_v3_500']['auc']:.6f}`, LL `{report['cat3_v3_500']['logloss']:.6f}`",
        "",
        "| blend | AUC | LogLoss | last AUC | last LL | Platt CV LL |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for k, r in clean.items():
        lines.append(
            f"| {k} | {r['auc']:.6f} | {r['logloss']:.6f} | {r['last_auc']:.6f} | "
            f"{r['last_logloss']:.6f} | {r['platt_cv_ll']:.6f} |"
        )
    lines += ["", f"Best: `{best_name}`. Candidate: `submission_v5.csv`.", ""]
    with open(os.path.join(REPORT_DIR, "improve_v5.md"), "w") as f:
        f.write("\n".join(lines))

    log(f"Done. Total runtime {elapsed():.1f}s")


if __name__ == "__main__":
    main()
