from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.special import log_ndtr, logit, ndtr, ndtri
from sklearn.metrics import log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports"
TARGET = "liquidity_stress_next_30d"
ID = "ID"
PROFILE = [
    "arpu",
    "age",
    "gender",
    "region",
    "smartphone",
    "segment",
    "earning_pattern",
    "x_90_d_activity_rate",
]
CATS = ["gender", "region", "smartphone", "segment", "earning_pattern"]
WEIGHTS = np.array([5.0, 3.0, 3.0, 2.0, 4.0, 2.0, 1.0]) / 20.0
RATE = 0.15
SIGMA = 0.35
EPS = 1e-15
MZ_SLOPE = 1.0 / SIGMA


def load_data(data_dir: str | Path = DATA_DIR):
    root = Path(data_dir)
    train = pd.read_csv(root / "Train.csv")
    test = pd.read_csv(root / "Test.csv")
    sample = pd.read_csv(root / "SampleSubmission.csv")
    assert train.shape == (40000, 184), train.shape
    assert test.shape == (30000, 183), test.shape
    assert (sample[ID].to_numpy() == test[ID].to_numpy()).all()
    return train, test, sample


def snapshot_dates(train: pd.DataFrame, test: pd.DataFrame):
    both = pd.concat([train[PROFILE], test[PROFILE]], ignore_index=True)
    customer = both.groupby(PROFILE, sort=False).ngroup().to_numpy()
    assert (customer[: len(train)] == np.repeat(np.arange(10000), 4)).all()
    assert (customer[len(train) :] == np.repeat(np.arange(10000), 3)).all()
    d_train = np.tile(np.arange(4), 10000)
    d_test = np.tile(np.arange(3), 10000) + 4
    y = train[TARGET].to_numpy()
    assert (pd.Series(y).groupby(d_train).sum() == 1500).all()
    return d_train, d_test


def risk_factors(df: pd.DataFrame) -> np.ndarray:
    def months(stem: str) -> np.ndarray:
        return df[[f"m{i}_{stem}" for i in range(1, 7)]].to_numpy(np.float64)

    bal = months("daily_avg_bal")
    withdraw = months("withdraw_total_value")
    income = (
        months("received_total_value")
        + months("deposit_total_value")
        + months("transfer_from_bank_total_value")
    )
    spend = months("paybill_total_value") + months("merchantpay_total_value")
    outflow = withdraw + spend + months("mm_send_total_value")
    recent, older = slice(0, 3), slice(3, 6)
    out = np.column_stack(
        [
            np.maximum(1 - (bal[:, recent].mean(1) + 1) / (bal[:, older].mean(1) + 1), 0),
            np.maximum(
                1 - (income[:, recent].mean(1) + 1) / (income[:, older].mean(1) + 1),
                0,
            ),
            (withdraw[:, recent].mean(1) + 1) / (withdraw[:, older].mean(1) + 1),
            (spend[:, recent].mean(1) + 1) / (spend[:, older].mean(1) + 1),
            (outflow[:, recent].sum(1) + 1) / (income[:, recent].sum(1) + 1),
            bal.std(axis=1),
            1 - df["x_90_d_activity_rate"].to_numpy(np.float64),
        ]
    )
    assert np.isfinite(out).all()
    return out


def solve_intercept(lin: np.ndarray, rate: float = RATE) -> float:
    return brentq(lambda c: ndtr(c + lin).mean() - rate, -10.0, 10.0, xtol=1e-14)


def calibrated(lin: np.ndarray, d: np.ndarray) -> np.ndarray:
    out = np.empty(len(lin), dtype=np.float64)
    for k in np.unique(d):
        m = d == k
        out[m] = ndtr(solve_intercept(lin[m]) + lin[m])
    return np.clip(out, EPS, 1 - EPS)


def month_z_score(R: np.ndarray, d_all: np.ndarray) -> np.ndarray:
    capped = np.minimum(R, np.quantile(R, 0.99, axis=0))
    Z = np.empty_like(capped)
    for k in np.unique(d_all):
        m = d_all == k
        Z[m] = (capped[m] - capped[m].mean(0)) / capped[m].std(0)
    return Z @ WEIGHTS


def global_z_score(R: np.ndarray) -> np.ndarray:
    capped = np.minimum(R, np.quantile(R, 0.99, axis=0))
    return ((capped - capped.mean(0)) / capped.std(0)) @ WEIGHTS


def per_month_std(s: np.ndarray, d_all: np.ndarray) -> np.ndarray:
    sd = np.array([s[d_all == k].std() for k in range(int(d_all.max()) + 1)])
    return s / sd[d_all]


def probit_eta(p: np.ndarray) -> np.ndarray:
    return ndtri(np.clip(p, EPS, 1 - EPS))


def per_date(eta: np.ndarray, d: np.ndarray) -> np.ndarray:
    out = np.empty(len(eta), dtype=np.float64)
    for k in np.unique(d):
        m = d == k
        out[m] = ndtr(solve_intercept(eta[m]) + eta[m])
    return np.clip(out, EPS, 1 - EPS)


def init_of(lin: np.ndarray, d_all: np.ndarray, idx: np.ndarray):
    eta = np.empty(len(idx), dtype=np.float64)
    for k in np.unique(d_all[idx]):
        m = d_all[idx] == k
        eta[m] = solve_intercept(lin[idx][m]) + lin[idx][m]
    return logit(np.clip(ndtr(eta), EPS, 1 - EPS)), eta


def fit_slope(s: np.ndarray, y: np.ndarray, dates: np.ndarray) -> float:
    levels = np.unique(dates)
    X = np.c_[(dates[:, None] == levels[None, :]).astype(float), s]

    def nll(w: np.ndarray):
        eta = X @ w
        lp, lq = log_ndtr(eta), log_ndtr(-eta)
        log_phi = -0.5 * eta**2 - 0.5 * np.log(2 * np.pi)
        grad_eta = np.where(y == 1, -np.exp(log_phi - lp), np.exp(log_phi - lq))
        return -(y * lp + (1 - y) * lq).sum(), X.T @ grad_eta

    res = minimize(
        nll,
        np.r_[np.full(len(levels), -1.56), 2.87],
        jac=True,
        method="L-BFGS-B",
        options={"ftol": 1e-15, "gtol": 1e-10, "maxiter": 10000},
    )
    assert res.success, res.message
    return float(res.x[-1])


def features(
    all_rows: pd.DataFrame,
    R: np.ndarray,
    drop_cols: Sequence[str] = (),
) -> np.ndarray:
    drop = [ID] + CATS + list(drop_cols)
    num = all_rows.drop(columns=drop)
    values = num.to_numpy(np.float64)
    X = np.sign(values) * np.log1p(np.abs(values))
    cats = pd.get_dummies(all_rows[CATS], dtype=float).to_numpy()
    caps = np.quantile(R, 0.99, axis=0)
    return np.c_[X, cats, np.minimum(R, caps)].astype(np.float32)


def month_scaled(X: np.ndarray, d_all: np.ndarray, how: str) -> np.ndarray:
    out = np.empty(X.shape, dtype=np.float32)
    for k in np.unique(d_all):
        m = d_all == k
        if how == "z":
            sd = np.nanstd(X[m], 0)
            out[m] = (X[m] - np.nanmean(X[m], 0)) / np.where(sd > 0, sd, 1)
        elif how == "rank":
            out[m] = pd.DataFrame(X[m]).rank(pct=True).to_numpy()
        else:
            out[m] = X[m]
    return out


def composite(y: np.ndarray, p: np.ndarray) -> Tuple[float, float, float]:
    ll = float(log_loss(y, p))
    auc = float(roc_auc_score(y, p))
    return 0.6 * (1.0 - ll / 0.595) + 0.4 * auc, ll, auc


def structural_oof_mz(s_mz: np.ndarray, d_tr: np.ndarray) -> np.ndarray:
    n = len(d_tr)
    lin = MZ_SLOPE * s_mz[:n]
    out = np.empty(n, dtype=np.float64)
    for k in range(4):
        m = d_tr == k
        out[m] = calibrated(lin[m], d_tr[m])
    return out


def structural_test_mz(s_mz: np.ndarray, d_te: np.ndarray) -> np.ndarray:
    n = len(s_mz) - len(d_te)
    return calibrated(MZ_SLOPE * s_mz[n:], d_te)


def structural_oof_global(s: np.ndarray, d_tr: np.ndarray) -> np.ndarray:
    n = len(d_tr)
    lin = MZ_SLOPE * s[:n]
    out = np.empty(n, dtype=np.float64)
    for k in range(4):
        m = d_tr == k
        out[m] = calibrated(lin[m], d_tr[m])
    return out


def structural_oof_avg(
    s_global: np.ndarray,
    s_mz: np.ndarray,
    y: np.ndarray,
    d_tr: np.ndarray,
    d_all: np.ndarray,
) -> np.ndarray:
    n = len(d_tr)
    parts = [s_global, per_month_std(s_global, d_all), s_mz]
    out = np.empty(n, dtype=np.float64)
    for k in range(4):
        fit = d_tr != k
        held = d_tr == k
        etas = []
        for s in parts:
            slope = fit_slope(s[:n][fit], y[fit], d_tr[fit])
            etas.append(probit_eta(calibrated(slope * s[:n][held], d_tr[held])))
        out[held] = calibrated(np.mean(etas, axis=0), d_tr[held])
    return out


def validate_submission(path: str | Path, sample: pd.DataFrame) -> dict:
    sub = pd.read_csv(path)
    assert list(sub.columns) == [ID, "Target"], list(sub.columns)
    assert len(sub) == len(sample) == 30000, len(sub)
    assert sub[ID].is_unique
    assert (sub[ID].to_numpy() == sample[ID].to_numpy()).all()
    p = sub["Target"].to_numpy(np.float64)
    assert np.isfinite(p).all()
    assert ((p > 0) & (p < 1)).all()
    return {
        "rows": int(len(sub)),
        "mean": float(p.mean()),
        "min": float(p.min()),
        "max": float(p.max()),
    }


def write_json(path: str | Path, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True))
