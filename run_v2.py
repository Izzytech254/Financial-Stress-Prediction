"""30-minute v2 experiment: leak-free feature screening, confirmation, ensemble.

Uses frozen folds, fold-internal feature selection, saved baseline predictions,
and reduced-budget CatBoost. Writes submission only if the paired AUC gain
exceeds 1 SE and LogLoss does not degrade beyond noise.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

import lightgbm as lgb

from features_v2 import build_v2_features
from solution import (
    CAT_COLS,
    DATA_DIR,
    ID,
    RANDOM_STATE,
    REPORT_DIR,
    ROOT,
    TARGET,
    cross_val_weight_auc,
    log,
    make_cat,
    make_lgb,
    select_features,
)

SEEDS = [42, 2026, 7]
N_FOLDS = 5
N_SELECT = 320
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
CAT_PARAMS = {
    "learning_rate": 0.05,
    "depth": 7,
    "l2_leaf_reg": 5.0,
    "subsample": 0.9,
    "rsm": 0.7,
    "random_strength": 0.5,
}
T0 = time.time()


def elapsed() -> float:
    return time.time() - T0


def _logit(p: np.ndarray) -> np.ndarray:
    return logit(np.clip(p, 1e-6, 1 - 1e-6))


def _py(obj):
    if isinstance(obj, dict):
        return {k: _py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_py(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def stack(
    base: Tuple[pd.DataFrame, pd.DataFrame, List[str]],
    extras: List[Tuple[pd.DataFrame, pd.DataFrame]],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    btr, bte, names = base
    tr_parts = [btr]
    te_parts = [bte]
    all_names = list(names)
    for etr, ete in extras:
        tr_parts.append(etr)
        te_parts.append(ete)
        all_names.extend(list(etr.columns))
    Xtr = pd.concat(tr_parts, axis=1).values.astype(np.float32)
    Xte = pd.concat(te_parts, axis=1).values.astype(np.float32)
    return Xtr, Xte, all_names


def summarize(
    y: np.ndarray,
    oof: np.ndarray,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    base_oof: np.ndarray | None = None,
) -> dict:
    aucs = [float(roc_auc_score(y[va], oof[va])) for _, va in folds]
    lls = [float(log_loss(y[va], oof[va])) for _, va in folds]
    out = {
        "auc": float(roc_auc_score(y, oof)),
        "logloss": float(log_loss(y, oof)),
        "fold_aucs": aucs,
        "fold_loglosses": lls,
        "fold_auc_mean": float(np.mean(aucs)),
        "fold_auc_sd": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
    }
    if base_oof is not None:
        da = [a - float(roc_auc_score(y[va], base_oof[va])) for a, (_, va) in zip(aucs, folds)]
        dl = [l - float(log_loss(y[va], base_oof[va])) for l, (_, va) in zip(lls, folds)]
        out["delta_auc_mean"] = float(np.mean(da))
        out["delta_auc_se"] = float(np.std(da, ddof=1) / np.sqrt(len(da)))
        out["delta_ll_mean"] = float(np.mean(dl))
        out["delta_ll_se"] = float(np.std(dl, ddof=1) / np.sqrt(len(dl)))
    return out


def run_lgb_cv(
    X: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    folds_by_seed: Dict[int, List[Tuple[np.ndarray, np.ndarray]]],
    seeds: List[int],
) -> Tuple[np.ndarray, np.ndarray, List[float], List[float]]:
    oof = np.zeros(len(y), dtype=np.float64)
    test_pred = np.zeros(len(X_test), dtype=np.float64)
    fold_aucs, fold_lls = [], []
    n_fits = len(seeds) * N_FOLDS
    for seed in seeds:
        oof_seed = np.zeros(len(y), dtype=np.float64)
        for tr, va in folds_by_seed[seed]:
            idx, _ = select_features(X[tr], y[tr], N_SELECT, seed=RANDOM_STATE)
            model = make_lgb(LGB_PARAMS, seed)
            model.fit(
                X[tr][:, idx],
                y[tr],
                eval_set=[(X[va][:, idx], y[va])],
                eval_metric="auc",
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
            p_va = model.predict_proba(X[va][:, idx])[:, 1]
            oof_seed[va] = p_va
            test_pred += model.predict_proba(X_test[:, idx])[:, 1] / n_fits
            fold_aucs.append(float(roc_auc_score(y[va], p_va)))
            fold_lls.append(float(log_loss(y[va], p_va)))
        oof += oof_seed / len(seeds)
    return oof, test_pred, fold_aucs, fold_lls


def _is_onehot(name: str) -> bool:
    return any(name.startswith(c + "_") for c in CAT_COLS)


def run_cat_cv(
    X: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    names: List[str],
    cat_train: np.ndarray,
    cat_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[float], List[float]]:
    oof = np.zeros(len(y), dtype=np.float64)
    test_pred = np.zeros(len(X_test), dtype=np.float64)
    fold_aucs, fold_lls = [], []
    for tr, va in folds:
        idx, _ = select_features(X[tr], y[tr], N_SELECT, seed=RANDOM_STATE)
        num_idx = [i for i in idx if not _is_onehot(names[i])]
        if len(num_idx) < 20:
            num_idx = [i for i in range(X.shape[1]) if not _is_onehot(names[i])][:N_SELECT]
        cols = [names[i] for i in num_idx]
        Xtr_cb = pd.DataFrame(X[tr][:, num_idx], columns=cols)
        Xva_cb = pd.DataFrame(X[va][:, num_idx], columns=cols)
        Xte_cb = pd.DataFrame(X_test[:, num_idx], columns=cols)
        for j, c in enumerate(CAT_COLS):
            Xtr_cb[c] = cat_train[tr, j]
            Xva_cb[c] = cat_train[va, j]
            Xte_cb[c] = cat_test[:, j]
        cat_idx = list(range(len(num_idx), len(num_idx) + len(CAT_COLS)))
        model = make_cat(CAT_PARAMS, RANDOM_STATE, iterations=300)
        model.fit(
            Xtr_cb,
            y[tr],
            eval_set=(Xva_cb, y[va]),
            use_best_model=True,
            cat_features=cat_idx,
        )
        p_va = model.predict_proba(Xva_cb)[:, 1]
        oof[va] = p_va
        test_pred += model.predict_proba(Xte_cb)[:, 1] / len(folds)
        fold_aucs.append(float(roc_auc_score(y[va], p_va)))
        fold_lls.append(float(log_loss(y[va], p_va)))
    return oof, test_pred, fold_aucs, fold_lls


def paired(y: np.ndarray, p_new: np.ndarray, p_base: np.ndarray,
           folds: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[float, float, float, float]:
    da, dl = [], []
    for _, va in folds:
        da.append(float(roc_auc_score(y[va], p_new[va]) - roc_auc_score(y[va], p_base[va])))
        dl.append(float(log_loss(y[va], p_new[va]) - log_loss(y[va], p_base[va])))
    return (
        float(np.mean(da)),
        float(np.std(da, ddof=1) / np.sqrt(len(da))),
        float(np.mean(dl)),
        float(np.std(dl, ddof=1) / np.sqrt(len(dl))),
    )


def platt_cv_ll(p: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(y), dtype=np.float64)
    for tr, va in skf.split(p, y):
        lr = LogisticRegression(C=1e6, max_iter=1000, solver="lbfgs")
        lr.fit(_logit(p[tr]).reshape(-1, 1), y[tr])
        oof[va] = lr.predict_proba(_logit(p[va]).reshape(-1, 1))[:, 1]
    return float(log_loss(y, oof)), oof


def main() -> None:
    os.makedirs(REPORT_DIR, exist_ok=True)
    if os.path.exists(os.path.join(ROOT, "submission.csv")):
        shutil.copyfile(
            os.path.join(ROOT, "submission.csv"),
            os.path.join(ROOT, "submission_baseline.csv"),
        )
        log("Backed up submission.csv -> submission_baseline.csv")

    train = pd.read_csv(os.path.join(DATA_DIR, "Train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "Test.csv"))
    sample_sub = pd.read_csv(os.path.join(DATA_DIR, "SampleSubmission.csv"))
    log(f"Loaded Train={train.shape} Test={test.shape}")

    feats = build_v2_features(train, test)
    y = feats["y"]
    base = feats["base"]
    groups = feats["train_groups"]
    log(
        f"Features built: base={base[0].shape[1]} "
        f"cust_tr={feats['cust_tr'][0].shape[1]} "
        f"cust_tt={feats['cust_tt'][0].shape[1]} "
        f"history={feats['history'][0].shape[1]} elapsed={elapsed():.1f}s"
    )

    folds_by_seed = {
        s: list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=s).split(np.zeros(len(y)), y))
        for s in SEEDS
    }
    group_folds = list(
        StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=RANDOM_STATE).split(
            np.zeros(len(y)), y, groups
        )
    )
    fold_arrays = {}
    for s in SEEDS:
        for i, (tr, va) in enumerate(folds_by_seed[s]):
            fold_arrays[f"tr_{s}_{i}"] = tr
            fold_arrays[f"va_{s}_{i}"] = va
    for i, (tr, va) in enumerate(group_folds):
        fold_arrays[f"gtr_{i}"] = tr
        fold_arrays[f"gva_{i}"] = va
    np.savez_compressed(os.path.join(REPORT_DIR, "folds_v2.npz"), **fold_arrays)

    z = np.load(os.path.join(REPORT_DIR, "preds_2seed.npz"))
    p_base = z["oof_final"]
    baseline_auc = float(roc_auc_score(y, p_base))
    baseline_ll = float(log_loss(y, p_base))
    log(f"Baseline OOF AUC={baseline_auc:.5f} LL={baseline_ll:.5f}")

    feature_sets: Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]] = {
        "baseline": stack(base, []),
        "cust_tr": stack(base, [feats["cust_tr"]]),
        "cust_tt": stack(base, [feats["cust_tt"]]),
        "history": stack(base, [feats["history"]]),
    }

    screening: Dict[str, dict] = {}
    screening_oof: Dict[str, np.ndarray] = {}
    for name in ["baseline", "cust_tr", "cust_tt", "history"]:
        X, Xte, _ = feature_sets[name]
        oof, test_pred, aucs, lls = run_lgb_cv(X, y, Xte, folds_by_seed, [42])
        screening_oof[name] = oof
        ref = screening_oof["baseline"] if name != "baseline" else None
        res = summarize(y, oof, folds_by_seed[42], ref)
        screening[name] = res
        log(
            f"Screen {name}: AUC={res['auc']:.5f} LL={res['logloss']:.5f} "
            f"delta_vs_lgb={res.get('delta_auc_mean', 0):+.5f} "
            f"+/-{res.get('delta_auc_se', 0):.5f} elapsed={elapsed():.1f}s"
        )

    best_cust = max(["cust_tr", "cust_tt"], key=lambda k: screening[k]["auc"])
    if elapsed() < 14 * 60:
        feature_sets["combined"] = stack(base, [feats[best_cust], feats["history"]])
        X, Xte, names = feature_sets["combined"]
        oof, test_pred, aucs, lls = run_lgb_cv(X, y, Xte, folds_by_seed, [42])
        screening_oof["combined"] = oof
        res = summarize(y, oof, folds_by_seed[42], screening_oof["baseline"])
        screening["combined"] = res
        log(
            f"Screen combined({best_cust}+history): AUC={res['auc']:.5f} LL={res['logloss']:.5f} "
            f"delta_vs_lgb={res['delta_auc_mean']:+.5f} +/-{res['delta_auc_se']:.5f} "
            f"elapsed={elapsed():.1f}s"
        )
    else:
        log("Skipping combined screen to protect the 30-minute budget.")

    best_name = max(screening, key=lambda k: screening[k]["auc"])
    log(f"Best screening feature set: {best_name} (AUC={screening[best_name]['auc']:.5f})")

    if best_name == "baseline" or screening[best_name]["auc"] <= screening["baseline"]["auc"]:
        log("No feature set beat the baseline LightGBM screen; stopping and keeping baseline submission.")
        best_name = "baseline"

    confirm = None
    p_lgb = None
    t_lgb = None
    group_res = None
    cat_res = None
    p_cat = None
    t_cat = None
    final = None

    if best_name != "baseline":
        X, Xte, names = feature_sets[best_name]
        p_lgb, t_lgb, aucs, lls = run_lgb_cv(X, y, Xte, folds_by_seed, SEEDS)
        all_folds = [f for s in SEEDS for f in folds_by_seed[s]]
        confirm = summarize(y, p_lgb, all_folds, p_base)
        log(
            f"Confirm {best_name}: 3-repeat OOF AUC={confirm['auc']:.5f} LL={confirm['logloss']:.5f} "
            f"delta_vs_baseline={confirm['delta_auc_mean']:+.5f} +/-{confirm['delta_auc_se']:.5f} "
            f"elapsed={elapsed():.1f}s"
        )

        p_group, t_group, gaucs, glls = run_lgb_cv(X, y, Xte, {42: group_folds}, [42])
        group_res = summarize(y, p_group, group_folds, p_base)
        log(
            f"GroupKFold {best_name}: AUC={group_res['auc']:.5f} LL={group_res['logloss']:.5f} "
            f"elapsed={elapsed():.1f}s"
        )

        gain = confirm["delta_auc_mean"]
        se = confirm["delta_auc_se"]
        ll_gain = confirm["delta_ll_mean"]
        ll_se = confirm["delta_ll_se"]
        proceed = (gain > se) and (ll_gain <= max(ll_se, 1e-4))
        log(
            f"Confirm gate: delta_auc={gain:+.5f} > SE={se:.5f}? {gain > se}; "
            f"delta_ll={ll_gain:+.5f} <= SE={ll_se:.5f}? {ll_gain <= max(ll_se, 1e-4)}"
        )
        if not proceed:
            log("Feature set did not clear the confirmation gate; keeping baseline submission.")
        else:
            if elapsed() < 20 * 60:
                cat_train = train[CAT_COLS].astype(str).values
                cat_test = test[CAT_COLS].astype(str).values
                try:
                    p_cat, t_cat, caucs, clls = run_cat_cv(
                        X, y, Xte, folds_by_seed[42], names, cat_train, cat_test
                    )
                    cat_res = summarize(y, p_cat, folds_by_seed[42], p_base)
                    log(
                        f"CatBoost: AUC={cat_res['auc']:.5f} LL={cat_res['logloss']:.5f} "
                        f"elapsed={elapsed():.1f}s"
                    )
                except Exception as exc:
                    log(f"CatBoost failed/skipped: {exc}")
            else:
                log("Skipping CatBoost: elapsed budget exceeded 20 minutes.")

            models = [
                ("lgb_v2", p_lgb, t_lgb),
                ("xgb_base", z["oof_xgb"], z["test_xgb"]),
                ("hist_base", z["oof_hist"], z["test_hist"]),
            ]
            if p_cat is not None:
                models.append(("cat_v2", p_cat, t_cat))
            P = np.clip(np.column_stack([m[1] for m in models]), 1e-6, 1 - 1e-6)
            T = np.clip(np.column_stack([m[2] for m in models]), 1e-6, 1 - 1e-6)

            p_mean_logit = expit(_logit(P).mean(axis=1))
            auc_mean_logit = float(roc_auc_score(y, p_mean_logit))
            w_cv_auc, w = cross_val_weight_auc(P, y, seed=RANDOM_STATE)
            p_weighted = np.clip(P @ w, 1e-6, 1 - 1e-6)
            if w_cv_auc > auc_mean_logit + 0.0005:
                p_blend = p_weighted
                t_blend = np.clip(T @ w, 1e-6, 1 - 1e-6)
                blend_mode = "weighted_prob"
                blend_est = w_cv_auc
            else:
                p_blend = p_mean_logit
                t_blend = expit(_logit(T).mean(axis=1))
                blend_mode = "mean_logit"
                blend_est = auc_mean_logit

            ll_cv_before = float(log_loss(y, p_blend))
            ll_cv_after, p_cal_cv = platt_cv_ll(p_blend, y)
            lr = LogisticRegression(C=1e6, max_iter=1000, solver="lbfgs")
            lr.fit(_logit(p_blend).reshape(-1, 1), y)
            p_cal = lr.predict_proba(_logit(p_blend).reshape(-1, 1))[:, 1]
            t_cal = lr.predict_proba(_logit(t_blend).reshape(-1, 1))[:, 1]
            apply_cal = ll_cv_after <= ll_cv_before + 1e-6
            p_final = p_cal_cv if apply_cal else p_blend
            t_final = t_cal if apply_cal else t_blend
            final_auc = float(roc_auc_score(y, p_final))
            da, dase, dl, dlse = paired(y, p_final, p_base, folds_by_seed[42])
            accept = (da > dase) and (dl <= max(dlse, 1e-4)) and (final_auc > baseline_auc)
            final = {
                "feature_set": best_name,
                "models": [m[0] for m in models],
                "blend_mode": blend_mode,
                "blend_cv_auc_estimate": float(blend_est),
                "blend_weights": dict(zip([m[0] for m in models], w.tolist())) if blend_mode == "weighted_prob" else None,
                "logloss_cv_before_calibration": float(ll_cv_before),
                "logloss_cv_after_calibration": float(ll_cv_after),
                "calibration_applied": bool(apply_cal),
                "final_oof_auc": final_auc,
                "final_oof_logloss_cv": float(ll_cv_after if apply_cal else ll_cv_before),
                "delta_auc_vs_baseline": float(da),
                "delta_auc_se": float(dase),
                "delta_ll_vs_baseline": float(dl),
                "delta_ll_se": float(dlse),
                "accepted": bool(accept),
            }
            log(f"Final blend: AUC={final_auc:.5f} mode={blend_mode} "
                f"delta_auc={da:+.5f} +/-{dase:.5f} delta_ll={dl:+.5f} +/-{dlse:.5f} "
                f"accepted={accept}")

            np.savez_compressed(
                os.path.join(REPORT_DIR, "v2_predictions.npz"),
                y=y,
                oof_lgb_v2=p_lgb,
                test_lgb_v2=t_lgb,
                oof_cat_v2=p_cat if p_cat is not None else np.array([]),
                test_cat_v2=t_cat if t_cat is not None else np.array([]),
                oof_blend=p_blend,
                test_blend=t_blend,
                oof_final=p_final,
                test_final=t_final,
            )

            if accept:
                submission = pd.DataFrame({ID: test[ID].values, "Target": np.clip(t_final, 1e-6, 1 - 1e-6)})
                submission = sample_sub[[ID]].merge(submission, on=ID, how="left")
                assert submission["Target"].notna().all()
                assert (submission[ID].values == sample_sub[ID].values).all()
                submission.to_csv(
                    os.path.join(ROOT, "submission.csv"), index=False, float_format="%.8f"
                )
                log("Accepted improvement; submission.csv updated.")
            else:
                log("Final blend did not clear acceptance; baseline submission retained.")

    report = {
        "baseline": {"auc": baseline_auc, "logloss": baseline_ll},
        "feature_counts": {
            "base": int(base[0].shape[1]),
            "cust_tr": int(feats["cust_tr"][0].shape[1]),
            "cust_tt": int(feats["cust_tt"][0].shape[1]),
            "history": int(feats["history"][0].shape[1]),
        },
        "screening": screening,
        "best_screening": best_name,
        "confirm": confirm,
        "groupkfold": group_res,
        "catboost": cat_res,
        "final": final,
        "feature_names": names if best_name != "baseline" else base[2],
        "config": {
            "seeds": SEEDS,
            "n_folds": N_FOLDS,
            "n_selected": N_SELECT,
            "lgb_params": LGB_PARAMS,
            "cat_params": CAT_PARAMS,
            "customer_key": "hash of 8 profile columns",
            "test_rows_in_features": best_name in ("cust_tt", "combined"),
        },
        "runtime_seconds": round(elapsed(), 1),
    }
    with open(os.path.join(REPORT_DIR, "cv_results_v2.json"), "w") as f:
        json.dump(_py(report), f, indent=2)

    rows = [
        "| Hypothesis | Feature set | OOF AUC | OOF LL | dAUC vs LGB base | SE | dLL | Decision |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for name, r in screening.items():
        rows.append(
            f"| {name} | {name} | {r['auc']:.5f} | {r['logloss']:.5f} | "
            f"{r.get('delta_auc_mean', 0):+.5f} | {r.get('delta_auc_se', 0):.5f} | "
            f"{r.get('delta_ll_mean', 0):+.5f} | {'keep' if r['auc'] >= screening['baseline']['auc'] else 'reject'} |"
        )
    if confirm:
        rows.append(
            f"| confirm | {best_name} | {confirm['auc']:.5f} | {confirm['logloss']:.5f} | "
            f"{confirm['delta_auc_mean']:+.5f} | {confirm['delta_auc_se']:.5f} | "
            f"{confirm['delta_ll_mean']:+.5f} | {'kept' if final and final['accepted'] else 'not accepted'} |"
        )
    if final:
        rows.append(
            f"| final blend | {final['feature_set']} | {final['final_oof_auc']:.5f} | "
            f"{final['final_oof_logloss_cv']:.5f} | {final['delta_auc_vs_baseline']:+.5f} | "
            f"{final['delta_auc_se']:.5f} | {final['delta_ll_vs_baseline']:+.5f} | "
            f"{'ACCEPT' if final['accepted'] else 'keep baseline'} |"
        )
    with open(os.path.join(REPORT_DIR, "experiment_log_v2.md"), "w") as f:
        f.write("# v2 Experiment Log\n\n")
        f.write(f"Baseline: AUC {baseline_auc:.5f}, LogLoss {baseline_ll:.5f}. ")
        f.write("Frozen 5-fold seeds 42/2026/7. Feature selection is fold-internal.\n\n")
        f.write("\n".join(rows) + "\n")

    summary_lines = [
        "# v2 Executive Summary",
        "",
        f"- Baseline OOF AUC: **{baseline_auc:.5f}**; LogLoss: **{baseline_ll:.5f}**.",
    ]
    if final:
        summary_lines.append(
            f"- Final OOF AUC: **{final['final_oof_auc']:.5f}**; "
            f"cross-validated LogLoss: **{final['final_oof_logloss_cv']:.5f}**; "
            f"accepted: **{final['accepted']}**."
        )
        summary_lines.append(
            f"- Paired AUC delta vs baseline: {final['delta_auc_vs_baseline']:+.5f} "
            f"+/- {final['delta_auc_se']:.5f}; LogLoss delta: {final['delta_ll_vs_baseline']:+.5f} "
            f"+/- {final['delta_ll_se']:.5f}."
        )
    else:
        summary_lines.append("- No feature set cleared the confirmation gate; baseline retained.")
    summary_lines += [
        f"- Best screening feature set: `{best_name}`.",
        f"- GroupKFold AUC: {group_res['auc']:.5f}." if group_res else "- GroupKFold was not run.",
        f"- CatBoost OOF AUC: {cat_res['auc']:.5f}." if cat_res else "- CatBoost was not included.",
        "",
        "## Interpretation",
        "",
        "Customer-level features were evaluated with leave-one-row-out statistics from features only. "
        "The `tt` variant uses unlabeled test rows; the `tr` variant uses Train rows only. "
        "Random-row CV is the test proxy because every test customer appears in Train. GroupKFold is a "
        "customer-memorization sensitivity check, not the primary selection criterion.",
        "",
        "## Residual risks",
        "",
        "- The 30-minute budget precludes nested Optuna; both models use fixed hyperparameters.",
        "- CatBoost uses one 5-fold repeat, so its contribution is estimated with fewer folds.",
        "- The baseline LogLoss is in-sample calibrated; new LogLoss is cross-validated.",
        "- Small gains may remain within fold noise; the acceptance gate is deliberately conservative.",
    ]
    with open(os.path.join(REPORT_DIR, "executive_summary_v2.md"), "w") as f:
        f.write("\n".join(summary_lines) + "\n")

    log(f"Wrote reports. Total runtime: {elapsed():.1f}s")


if __name__ == "__main__":
    main()
