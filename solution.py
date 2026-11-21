"""Financial Stress Prediction Challenge - end-to-end solution.

Predicts the probability that a mobile-money customer experiences liquidity
stress in the next 30 days from six months of transaction behaviour.

Pipeline:
  1. Feature engineering (trends, ratios, flow composition, affordability)
  2. Fast XGBoost importance-based feature selection
  3. Optuna tuning for XGBoost / LightGBM / CatBoost / HistGradientBoosting
  4. Repeated stratified CV with multi-seed bagging
  5. OOF-evaluated blending + Platt calibration
  6. Submission generation

Run:
    .venv/bin/python solution.py
"""

from __future__ import annotations

import json
import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.special import logit, expit

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

import lightgbm as lgb
import optuna
import xgboost as xgb
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
REPORT_DIR = os.path.join(ROOT, "reports")
TARGET = "liquidity_stress_next_30d"
ID = "ID"
MONTHS = list(range(1, 7))
CAT_COLS = ["gender", "region", "smartphone", "segment", "earning_pattern"]
TYPES = {
    "paybill": "companies",
    "merchantpay": "merchants",
    "transfer_from_bank": "banks",
    "mm_send": "recipients",
    "received": "senders",
    "deposit": "agents",
    "withdraw": "agents",
}

N_SELECTED_FEATURES = 320
N_FOLDS = 5
CV_SEEDS = [42, 2026]
CATBOOST_SEEDS = [42]
N_JOBS = 8
RANDOM_STATE = 42
EPS = 1.0

# Tuning budgets (seconds). Optuna stops after the timeout or n_trials.
TUNING_BUDGET = {
    "xgb": (170, 20),
    "lgb": (120, 20),
    "cat": (150, 10),
    "hist": (70, 12),
}


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _slope(mat: np.ndarray) -> np.ndarray:
    x = np.arange(mat.shape[1], dtype=np.float64)
    x = x - x.mean()
    return (mat * x).sum(axis=1) / (x * x).sum()


def _safe_log1p(mat: np.ndarray) -> np.ndarray:
    return np.sign(mat) * np.log1p(np.abs(mat))


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def build_features(
    train: pd.DataFrame, test: pd.DataFrame, clip: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, List[str]]:
    """Build the engineered feature matrix for train and test."""
    n_train = len(train)
    df = pd.concat([train.drop(columns=[TARGET]), test], axis=0, ignore_index=True)

    feats: Dict[str, np.ndarray] = {}

    # Raw numeric features. Monetary/count features are log1p transformed.
    raw_cols = [c for c in df.columns if c not in [ID] + CAT_COLS]
    for c in raw_cols:
        v = pd.to_numeric(df[c], errors="coerce").astype(np.float64).values
        if c.startswith("m") and any(k in c for k in ("total_value", "highest_amount", "daily_avg_bal")):
            v = _safe_log1p(v)
        elif c == "arpu":
            v = np.log1p(np.clip(v, 0, None))
        feats[c] = v

    # Longitudinal statistics for every transaction type / metric.
    for t, cp in TYPES.items():
        metrics = ["volume", "total_value", "highest_amount", cp]
        vol_sum = np.zeros(len(df), dtype=np.float64)
        for i in MONTHS:
            vol_sum += pd.to_numeric(df[f"m{i}_{t}_volume"], errors="coerce").values
        for m in metrics:
            cols = [f"m{i}_{t}_{m}" for i in MONTHS]
            mat = np.column_stack([pd.to_numeric(df[c], errors="coerce").values for c in cols]).astype(np.float64)
            if "total_value" in m or "highest_amount" in m:
                mat = _safe_log1p(mat)
            s = mat.sum(axis=1)
            mu = mat.mean(axis=1)
            sd = mat.std(axis=1)
            mn = mat.min(axis=1)
            mx = mat.max(axis=1)
            recent = mat[:, :3].mean(axis=1)
            older = mat[:, 3:].mean(axis=1)
            m1 = mat[:, 0]
            m6 = mat[:, 5]
            w = np.array([6, 5, 4, 3, 2, 1], dtype=np.float64)
            rw = mat @ w / w.sum()
            key = f"{t}_{m}"
            active = (mat > 0).sum(axis=1)

            feats[f"{key}_sum"] = s
            feats[f"{key}_mean"] = mu
            feats[f"{key}_std"] = sd
            feats[f"{key}_min"] = mn
            feats[f"{key}_max"] = mx
            feats[f"{key}_slope"] = _slope(mat)
            feats[f"{key}_m1"] = m1
            feats[f"{key}_m6"] = m6
            feats[f"{key}_m1_m6_ratio"] = (m1 + EPS) / (m6 + EPS)
            feats[f"{key}_m1_mean_ratio"] = (m1 + EPS) / (mu + EPS)
            feats[f"{key}_recent"] = recent
            feats[f"{key}_older"] = older
            feats[f"{key}_recent_older_ratio"] = (recent + EPS) / (older + EPS)
            feats[f"{key}_recency_weighted"] = rw
            feats[f"{key}_cv"] = sd / (mu + EPS)
            feats[f"{key}_active_months"] = active
            feats[f"{key}_zero_months"] = (mat <= 0).sum(axis=1)

            if m in ("total_value", "highest_amount"):
                feats[f"{key}_mean_txn"] = s / (vol_sum + EPS)
            if m == cp:
                feats[f"{key}_per_volume"] = s / (vol_sum + EPS)
                feats[f"{key}_per_active_month"] = s / (active + EPS)

    # Balance features.
    bals = np.column_stack([pd.to_numeric(df[f"m{i}_daily_avg_bal"], errors="coerce").values for i in MONTHS]).astype(np.float64)
    bals = np.log1p(np.clip(bals, 0, None))
    b_mu = bals.mean(axis=1)
    b_sd = bals.std(axis=1)
    b_recent = bals[:, :3].mean(axis=1)
    b_older = bals[:, 3:].mean(axis=1)
    feats["bal_mean2"] = b_mu
    feats["bal_std2"] = b_sd
    feats["bal_cv2"] = b_sd / (b_mu + EPS)
    feats["bal_min2"] = bals.min(axis=1)
    feats["bal_max2"] = bals.max(axis=1)
    feats["bal_slope2"] = _slope(bals)
    feats["bal_m1_m6_ratio2"] = (bals[:, 0] + EPS) / (bals[:, 5] + EPS)
    feats["bal_m1_mean_ratio2"] = (bals[:, 0] + EPS) / (b_mu + EPS)
    feats["bal_recent_older_ratio2"] = (b_recent + EPS) / (b_older + EPS)
    feats["bal_recent_older_diff2"] = b_recent - b_older
    feats["bal_m1_m2_diff2"] = bals[:, 0] - bals[:, 1]
    feats["bal_m1_m2_ratio2"] = (bals[:, 0] + EPS) / (bals[:, 1] + EPS)

    # Monthly inflow / outflow and affordability.
    flow: Dict[str, np.ndarray] = {}
    for i in MONTHS:
        inflow = (
            df[f"m{i}_deposit_total_value"].values
            + df[f"m{i}_received_total_value"].values
            + df[f"m{i}_transfer_from_bank_total_value"].values
        )
        outflow = (
            df[f"m{i}_withdraw_total_value"].values
            + df[f"m{i}_mm_send_total_value"].values
            + df[f"m{i}_paybill_total_value"].values
            + df[f"m{i}_merchantpay_total_value"].values
        )
        flow[f"in{i}"] = np.log1p(np.clip(inflow, 0, None))
        flow[f"out{i}"] = np.log1p(np.clip(outflow, 0, None))
        flow[f"net{i}"] = _safe_log1p(inflow - outflow)
        flow[f"out_in{i}"] = (outflow + EPS) / (inflow + EPS)
        flow[f"runway{i}"] = np.clip(df[f"m{i}_daily_avg_bal"].values / (outflow / 30.0 + EPS), 0, 1000)
        flow[f"out_per_bal{i}"] = (outflow + EPS) / (df[f"m{i}_daily_avg_bal"].values + EPS)
        flow[f"in_per_bal{i}"] = (inflow + EPS) / (df[f"m{i}_daily_avg_bal"].values + EPS)

    for name in ["in", "out", "net", "out_in", "runway", "out_per_bal", "in_per_bal"]:
        mat = np.column_stack([flow[f"{name}{i}"] for i in MONTHS])
        feats[f"flow_{name}_mean"] = mat.mean(axis=1)
        feats[f"flow_{name}_std"] = mat.std(axis=1)
        feats[f"flow_{name}_min"] = mat.min(axis=1)
        feats[f"flow_{name}_max"] = mat.max(axis=1)
        feats[f"flow_{name}_slope"] = _slope(mat)
        feats[f"flow_{name}_m1"] = mat[:, 0]
        feats[f"flow_{name}_m1_m6_ratio"] = (mat[:, 0] + EPS) / (mat[:, 5] + EPS)
        feats[f"flow_{name}_recent"] = mat[:, :3].mean(axis=1)
        feats[f"flow_{name}_older"] = mat[:, 3:].mean(axis=1)
        feats[f"flow_{name}_recent_older_ratio"] = (mat[:, :3].mean(axis=1) + EPS) / (mat[:, 3:].mean(axis=1) + EPS)

    # Six-month flow composition.
    def raw_sum(prefix: str) -> np.ndarray:
        cols = [f"m{i}_{prefix}_total_value" for i in MONTHS]
        return np.column_stack([pd.to_numeric(df[c], errors="coerce").values for c in cols]).astype(np.float64).sum(axis=1)

    inflow_parts = {k: raw_sum(k) for k in ["deposit", "received", "transfer_from_bank"]}
    outflow_parts = {k: raw_sum(k) for k in ["withdraw", "mm_send", "paybill", "merchantpay"]}
    inflow = sum(inflow_parts.values())
    outflow = sum(outflow_parts.values())
    bal_level = np.expm1(b_mu) + EPS
    feats["total_inflow"] = np.log1p(np.clip(inflow, 0, None))
    feats["total_outflow"] = np.log1p(np.clip(outflow, 0, None))
    feats["net_flow"] = _safe_log1p(inflow - outflow)
    feats["inflow_outflow_ratio"] = (inflow + EPS) / (outflow + EPS)
    feats["total_flow_per_bal"] = (inflow + outflow) / bal_level
    feats["net_flow_per_bal"] = (inflow - outflow) / bal_level
    feats["net_flow_per_inflow"] = (inflow - outflow) / (inflow + EPS)
    for k, v in inflow_parts.items():
        feats[f"share_in_{k}"] = (v + EPS) / (inflow + EPS)
        feats[f"{k}_per_bal"] = (v + EPS) / bal_level
    for k, v in outflow_parts.items():
        feats[f"share_out_{k}"] = (v + EPS) / (outflow + EPS)
        feats[f"{k}_per_bal"] = (v + EPS) / bal_level
    feats["withdraw_deposit_ratio"] = (outflow_parts["withdraw"] + EPS) / (inflow_parts["deposit"] + EPS)
    feats["received_sent_ratio"] = (inflow_parts["received"] + EPS) / (outflow_parts["mm_send"] + EPS)
    feats["deposit_share_of_inflow2"] = (inflow_parts["deposit"] + EPS) / (inflow + EPS)
    feats["arpu_per_bal2"] = (np.log1p(df["arpu"].values) + EPS) / (b_mu + EPS)
    feats["activity_x_arpu"] = df["x_90_d_activity_rate"].values * np.log1p(df["arpu"].values)
    feats["activity_x_bal"] = df["x_90_d_activity_rate"].values * b_mu

    # Aggregate transaction counts.
    all_vol = np.zeros(len(df), dtype=np.float64)
    for t in TYPES:
        for i in MONTHS:
            all_vol += pd.to_numeric(df[f"m{i}_{t}_volume"], errors="coerce").values
    feats["total_txn_volume"] = all_vol
    feats["txn_per_activity"] = all_vol / (df["x_90_d_activity_rate"].values + 0.01)
    feats["txn_per_bal"] = all_vol / bal_level
    feats["txn_per_inflow"] = all_vol / (inflow + EPS)

    # Categorical one-hot encoding.
    X_cat = pd.get_dummies(df[CAT_COLS].astype(str), columns=CAT_COLS, drop_first=True, dtype=np.float64)
    X = pd.concat([pd.DataFrame(feats), X_cat], axis=1)
    X = X.replace([np.inf, -np.inf], np.nan).astype(np.float32)
    X = X.fillna(0.0)

    # Clip extreme outliers using training quantiles only.
    if clip:
        q_lo = X.iloc[:n_train].quantile(0.001)
        q_hi = X.iloc[:n_train].quantile(0.999)
        X = X.clip(lower=q_lo, upper=q_hi, axis=1)

    X_train = X.iloc[:n_train].reset_index(drop=True)
    X_test = X.iloc[n_train:].reset_index(drop=True)
    y = train[TARGET].astype(int).values
    return X_train, X_test, y, list(X.columns)


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------
def select_features(
    X: np.ndarray, y: np.ndarray, n_features: int, seed: int = RANDOM_STATE
) -> Tuple[List[int], np.ndarray]:
    """Fast gain-based feature selection on a subsample."""
    rng = np.random.default_rng(seed)
    n = min(9000, len(X))
    idx = rng.choice(len(X), n, replace=False)
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.06,
        max_depth=6,
        min_child_weight=5,
        subsample=0.9,
        colsample_bytree=0.7,
        tree_method="hist",
        max_bin=64,
        importance_type="gain",
        n_jobs=N_JOBS,
        random_state=seed,
        eval_metric="logloss",
    )
    model.fit(X[idx], y[idx], verbose=False)
    imp = model.feature_importances_
    order = np.argsort(imp)[::-1]
    return order[: min(n_features, len(order))].tolist(), imp


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def make_xgb(params: dict, seed: int, n_estimators: int = 800) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=n_estimators,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        max_bin=int(params.get("max_bin", 128)),
        learning_rate=float(params.get("learning_rate", 0.04)),
        max_depth=int(params.get("max_depth", 7)),
        min_child_weight=float(params.get("min_child_weight", 5)),
        subsample=float(params.get("subsample", 0.9)),
        colsample_bytree=float(params.get("colsample_bytree", 0.7)),
        reg_alpha=float(params.get("reg_alpha", 1.0)),
        reg_lambda=float(params.get("reg_lambda", 5.0)),
        gamma=float(params.get("gamma", 0.0)),
        n_jobs=N_JOBS,
        random_state=seed,
        early_stopping_rounds=50,
    )


def make_lgb(params: dict, seed: int, n_estimators: int = 900) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        n_estimators=n_estimators,
        objective="binary",
        learning_rate=float(params.get("learning_rate", 0.05)),
        num_leaves=int(params.get("num_leaves", 31)),
        max_depth=int(params.get("max_depth", 9)),
        min_child_samples=int(params.get("min_child_samples", 40)),
        subsample=float(params.get("subsample", 0.9)),
        colsample_bytree=float(params.get("colsample_bytree", 0.7)),
        reg_alpha=float(params.get("reg_alpha", 1.0)),
        reg_lambda=float(params.get("reg_lambda", 5.0)),
        max_bin=int(params.get("max_bin", 127)),
        force_col_wise=True,
        n_jobs=N_JOBS,
        random_state=seed,
        verbosity=-1,
    )


def make_cat(params: dict, seed: int, iterations: int = 700) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=iterations,
        learning_rate=float(params.get("learning_rate", 0.05)),
        depth=int(params.get("depth", 7)),
        l2_leaf_reg=float(params.get("l2_leaf_reg", 5.0)),
        subsample=float(params.get("subsample", 0.9)),
        rsm=float(params.get("rsm", 0.7)),
        random_strength=float(params.get("random_strength", 1.0)),
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=seed,
        thread_count=N_JOBS,
        od_type="Iter",
        od_wait=50,
        verbose=False,
        allow_writing_files=False,
    )


def make_hist(params: dict, seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=600,
        learning_rate=float(params.get("learning_rate", 0.05)),
        max_leaf_nodes=int(params.get("max_leaf_nodes", 63)),
        min_samples_leaf=int(params.get("min_samples_leaf", 40)),
        l2_regularization=float(params.get("l2_regularization", 5.0)),
        max_bins=int(params.get("max_bins", 128)),
        early_stopping=True,
        validation_fraction=0.12,
        n_iter_no_change=30,
        random_state=seed,
    )


# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------
def tune_model(name: str, X: np.ndarray, y: np.ndarray) -> dict:
    log(f"Tuning {name} ...")
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    timeout, n_trials = TUNING_BUDGET[name]

    def objective(trial: optuna.Trial) -> float:
        if name == "xgb":
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.075, log=True),
                "max_depth": trial.suggest_int("max_depth", 4, 9),
                "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
                "subsample": trial.suggest_float("subsample", 0.7, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.9),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 20.0, log=True),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            }
            model = make_xgb(params, RANDOM_STATE)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        elif name == "lgb":
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.08, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 127),
                "max_depth": trial.suggest_int("max_depth", 5, 12),
                "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
                "subsample": trial.suggest_float("subsample", 0.7, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.9),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 20.0, log=True),
            }
            model = make_lgb(params, RANDOM_STATE)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                eval_metric="auc",
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
        elif name == "cat":
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.08, log=True),
                "depth": trial.suggest_int("depth", 5, 9),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
                "subsample": trial.suggest_float("subsample", 0.7, 1.0),
                "rsm": trial.suggest_float("rsm", 0.5, 0.9),
                "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
            }
            model = make_cat(params, RANDOM_STATE)
            model.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)
        else:
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.08, log=True),
                "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 127),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 20, 100),
                "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 20.0, log=True),
                "max_bins": trial.suggest_categorical("max_bins", [128, 255]),
            }
            model = make_hist(params, RANDOM_STATE)
            model.fit(X_tr, y_tr)

        p = model.predict_proba(X_va)[:, 1]
        return float(roc_auc_score(y_va, p))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    best = dict(study.best_params)
    log(f"{name}: best holdout AUC={study.best_value:.5f} params={best}")
    return best


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------
@dataclass
class ModelResult:
    name: str
    oof: np.ndarray
    test: np.ndarray
    auc: float
    logloss: float
    fold_aucs: List[float] = field(default_factory=list)
    fold_loglosses: List[float] = field(default_factory=list)
    params: dict = field(default_factory=dict)


def run_cv(
    name: str,
    params: dict,
    X: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    seeds: List[int],
) -> ModelResult:
    log(f"Training {name} with {len(seeds)} seed(s) x {N_FOLDS} folds ...")
    oof_all = np.zeros(len(y), dtype=np.float64)
    test_all = np.zeros(len(X_test), dtype=np.float64)
    fold_aucs, fold_lls = [], []
    n_runs = 0

    for seed in seeds:
        oof_seed = np.zeros(len(y), dtype=np.float64)
        test_seed = np.zeros(len(X_test), dtype=np.float64)
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_va, y_va = X[va_idx], y[va_idx]
            if name == "xgb":
                model = make_xgb(params, seed)
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            elif name == "lgb":
                model = make_lgb(params, seed)
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_va, y_va)],
                    eval_metric="auc",
                    callbacks=[lgb.early_stopping(50, verbose=False)],
                )
            elif name == "cat":
                model = make_cat(params, seed)
                model.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)
            else:
                model = make_hist(params, seed)
                model.fit(X_tr, y_tr)

            p_va = model.predict_proba(X_va)[:, 1]
            p_te = model.predict_proba(X_test)[:, 1]
            oof_seed[va_idx] = p_va
            test_seed += p_te / N_FOLDS
            fold_aucs.append(roc_auc_score(y_va, p_va))
            fold_lls.append(log_loss(y_va, p_va))
            log(f"  {name} seed={seed} fold={fold}: AUC={fold_aucs[-1]:.5f} LL={fold_lls[-1]:.5f}")
        oof_all += oof_seed
        test_all += test_seed
        n_runs += 1

    oof_all /= n_runs
    test_all /= n_runs
    auc = roc_auc_score(y, oof_all)
    ll = log_loss(y, oof_all)
    log(f"{name}: OOF AUC={auc:.5f} OOF LogLoss={ll:.5f}")
    return ModelResult(
        name=name,
        oof=oof_all,
        test=test_all,
        auc=auc,
        logloss=ll,
        fold_aucs=fold_aucs,
        fold_loglosses=fold_lls,
        params=params,
    )


# ---------------------------------------------------------------------------
# Blending / calibration
# ---------------------------------------------------------------------------
def _fast_auc(y: np.ndarray, p: np.ndarray) -> float:
    r = pd.Series(p).rank().values
    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def optimize_weights(
    P: np.ndarray, y: np.ndarray, n_samples: int = 3000, seed: int = RANDOM_STATE
) -> np.ndarray:
    """Random Dirichlet search for blend weights maximizing OOF AUC."""
    rng = np.random.default_rng(seed)
    best_auc, best_w = -1.0, None
    n_models = P.shape[1]
    for _ in range(n_samples):
        w = rng.dirichlet(np.ones(n_models))
        auc = _fast_auc(y, P @ w)
        if auc > best_auc:
            best_auc, best_w = auc, w
    return best_w


def cross_val_weight_auc(P: np.ndarray, y: np.ndarray, seed: int = RANDOM_STATE) -> Tuple[float, np.ndarray]:
    """Honest weight-search estimate via 5-fold cross-validation."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    aucs = []
    for tr_idx, va_idx in skf.split(P, y):
        w = optimize_weights(P[tr_idx], y[tr_idx], n_samples=800, seed=seed)
        aucs.append(_fast_auc(y[va_idx], P[va_idx] @ w))
    w_full = optimize_weights(P, y, n_samples=3000, seed=seed)
    return float(np.mean(aucs)), w_full


def blend_models(results: List[ModelResult], y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict, dict]:
    names = [r.name for r in results]
    P_oof = np.column_stack([r.oof for r in results])
    P_test = np.column_stack([r.test for r in results])
    P_oof = np.clip(P_oof, 1e-6, 1 - 1e-6)
    P_test = np.clip(P_test, 1e-6, 1 - 1e-6)
    L_oof = logit(P_oof)
    L_test = logit(P_test)

    candidates: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    # Mean probability.
    p = P_oof.mean(axis=1)
    t = P_test.mean(axis=1)
    candidates["mean_prob"] = (p, t)

    # Mean logit.
    p = expit(L_oof.mean(axis=1))
    t = expit(L_test.mean(axis=1))
    candidates["mean_logit"] = (p, t)

    # Mean rank (normalized).
    p = np.column_stack([pd.Series(P_oof[:, j]).rank(pct=True).values for j in range(P_oof.shape[1])]).mean(axis=1)
    t = np.column_stack([pd.Series(P_test[:, j]).rank(pct=True).values for j in range(P_test.shape[1])]).mean(axis=1)
    candidates["mean_rank"] = (np.clip(p, 1e-6, 1 - 1e-6), np.clip(t, 1e-6, 1 - 1e-6))

    # Logistic stack. Cross-validated OOF estimate is used for comparison.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    stack_oof = np.zeros(len(y), dtype=np.float64)
    for tr_idx, va_idx in skf.split(L_oof, y):
        meta = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
        meta.fit(L_oof[tr_idx], y[tr_idx])
        stack_oof[va_idx] = meta.predict_proba(L_oof[va_idx])[:, 1]
    meta = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    meta.fit(L_oof, y)
    stack_test = meta.predict_proba(L_test)[:, 1]
    candidates["logistic_stack"] = (stack_oof, stack_test)

    # CV-estimated weighted average (weights selected on OOF).
    w_cv_auc, w = cross_val_weight_auc(P_oof, y, seed=RANDOM_STATE)
    p = P_oof @ w
    t = P_test @ w
    candidates["weighted_prob"] = (np.clip(p, 1e-6, 1 - 1e-6), np.clip(t, 1e-6, 1 - 1e-6))

    # Evaluate candidates.
    comparison = {}
    for key, (p, _) in candidates.items():
        comparison[key] = {
            "auc_oof": float(roc_auc_score(y, p)),
            "logloss_oof": float(log_loss(y, p)),
        }
    comparison["weighted_prob"]["auc_cv_estimate"] = float(w_cv_auc)

    # Pick the best probability-producing candidate by OOF AUC. Rank blend is
    # allowed only if it beats the best probability blend by a clear margin.
    prob_keys = ["mean_prob", "mean_logit", "logistic_stack", "weighted_prob"]
    best_key = max(prob_keys, key=lambda k: comparison[k]["auc_oof"])
    best_auc = comparison[best_key]["auc_oof"]
    if comparison["mean_rank"]["auc_oof"] > best_auc + 0.001:
        best_key = "mean_rank"
    log(f"Blend comparison: {json.dumps(comparison, indent=2)}")
    log(f"Selected blend: {best_key}")
    p_oof, p_test = candidates[best_key]
    return p_oof, p_test, comparison, {"selected": best_key, "weights": w.tolist(), "names": names}


def calibrate(oof: np.ndarray, test: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Platt scaling on the blend logit (monotonic -> preserves AUC)."""
    o = logit(np.clip(oof, 1e-6, 1 - 1e-6)).reshape(-1, 1)
    t = logit(np.clip(test, 1e-6, 1 - 1e-6)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=1000, solver="lbfgs")
    lr.fit(o, y)
    o_cal = lr.predict_proba(o)[:, 1]
    t_cal = lr.predict_proba(t)[:, 1]
    info = {
        "coef": float(lr.coef_[0][0]),
        "intercept": float(lr.intercept_[0]),
        "auc_before": float(roc_auc_score(y, oof)),
        "auc_after": float(roc_auc_score(y, o_cal)),
        "logloss_before": float(log_loss(y, oof)),
        "logloss_after": float(log_loss(y, o_cal)),
    }
    if info["logloss_after"] <= info["logloss_before"] + 1e-6 and info["auc_after"] >= info["auc_before"] - 1e-5:
        log(f"Calibration applied: LL {info['logloss_before']:.5f} -> {info['logloss_after']:.5f}")
        info["applied"] = True
        return o_cal, t_cal, info
    log("Calibration rejected; using raw blend probabilities.")
    info["applied"] = False
    return oof, test, info


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    t_start = time.time()
    os.makedirs(REPORT_DIR, exist_ok=True)

    log("Loading data ...")
    train = pd.read_csv(os.path.join(DATA_DIR, "Train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "Test.csv"))
    sample_sub = pd.read_csv(os.path.join(DATA_DIR, "SampleSubmission.csv"))

    log(f"Train={train.shape} Test={test.shape} target_rate={train[TARGET].mean():.4f}")
    X_train_df, X_test_df, y, feature_names = build_features(train, test)
    log(f"Engineered features: {X_train_df.shape[1]}")

    X_full = X_train_df.values.astype(np.float32)
    X_test_full = X_test_df.values.astype(np.float32)

    # Feature selection.
    selected_idx, importance = select_features(X_full, y, N_SELECTED_FEATURES, seed=RANDOM_STATE)
    selected_names = [feature_names[i] for i in selected_idx]
    X = X_full[:, selected_idx]
    X_test = X_test_full[:, selected_idx]
    log(f"Selected {len(selected_idx)} features for modeling")

    imp_df = pd.DataFrame({"feature": feature_names, "importance": importance}).sort_values("importance", ascending=False)
    imp_df.to_csv(os.path.join(REPORT_DIR, "feature_importance.csv"), index=False)

    # Tune each model.
    tuned = {}
    for name in ["xgb", "lgb", "cat", "hist"]:
        try:
            tuned[name] = tune_model(name, X, y)
        except Exception as exc:  # keep the pipeline alive if one tuner fails
            log(f"Tuning failed for {name}: {exc}; using defaults.")
            tuned[name] = {}

    # Cross-validated model training.
    results: List[ModelResult] = []
    results.append(run_cv("xgb", tuned["xgb"], X, y, X_test, CV_SEEDS))
    results.append(run_cv("lgb", tuned["lgb"], X, y, X_test, CV_SEEDS))
    results.append(run_cv("cat", tuned["cat"], X, y, X_test, CATBOOST_SEEDS))
    results.append(run_cv("hist", tuned["hist"], X, y, X_test, CV_SEEDS))

    # Blend and calibrate.
    oof_blend, test_blend, comparison, blend_info = blend_models(results, y)
    oof_final, test_final, calibration_info = calibrate(oof_blend, test_blend, y)
    test_final = np.clip(test_final, 1e-6, 1 - 1e-6)

    final_auc = float(roc_auc_score(y, oof_final))
    final_ll = float(log_loss(y, oof_final))
    log(f"FINAL OOF AUC={final_auc:.5f} LogLoss={final_ll:.5f}")
    log(f"Test prediction mean={test_final.mean():.4f} min={test_final.min():.6f} max={test_final.max():.6f}")

    # Submission.
    assert len(test_final) == len(test)
    submission = pd.DataFrame({ID: test[ID].values, "Target": test_final})
    submission = sample_sub[[ID]].merge(submission, on=ID, how="left")
    assert submission["Target"].notna().all(), "Missing predictions after merge"
    assert (submission[ID].values == sample_sub[ID].values).all(), "ID order mismatch"
    out_path = os.path.join(ROOT, "submission.csv")
    submission.to_csv(out_path, index=False, float_format="%.8f")
    log(f"Saved submission to {out_path}")

    # Reports.
    report = {
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "target_rate": float(train[TARGET].mean()),
        "n_engineered_features": int(X_train_df.shape[1]),
        "n_selected_features": int(len(selected_idx)),
        "selected_features": selected_names,
        "tuning": {k: v for k, v in tuned.items()},
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
        "runtime_seconds": round(time.time() - t_start, 1),
    }
    with open(os.path.join(REPORT_DIR, "cv_results.json"), "w") as f:
        json.dump(report, f, indent=2)
    log(f"Saved report to {os.path.join(REPORT_DIR, 'cv_results.json')}")
    log(f"Total runtime: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
