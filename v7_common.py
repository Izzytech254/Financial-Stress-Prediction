from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import log_ndtr

from v6_common import (
    DATA_DIR,
    EPS,
    ID,
    REPORT_DIR,
    SIGMA,
    TARGET,
    calibrated,
    composite,
    features,
    init_of,
    load_data,
    month_scaled,
    month_z_score,
    risk_factors,
    snapshot_dates,
    solve_intercept,
    structural_oof_mz,
    validate_submission,
    write_json,
)

EXTRA4 = [
    "m5_deposit_agents",
    "m2_merchantpay_merchants",
    "m5_withdraw_highest_amount",
    "m4_mm_send_recipients",
]
EXTRA3_NOM5 = [c for c in EXTRA4 if c != "m5_deposit_agents"]

PREFIX10 = [
    "m5_deposit_agents",
    "m2_merchantpay_merchants",
    "m5_withdraw_highest_amount",
    "m4_paybill_volume",
    "m6_mm_send_total_value",
    "m4_transfer_from_bank_volume",
    "m4_transfer_from_bank_banks",
    "m3_withdraw_agents",
    "m1_paybill_total_value",
    "m6_transfer_from_bank_banks",
]
RANK3_10 = [
    "m5_deposit_agents",
    "m2_merchantpay_merchants",
    "m4_transfer_from_bank_volume",
    "m5_withdraw_highest_amount",
    "m4_paybill_volume",
    "m4_transfer_from_bank_banks",
    "m6_mm_send_total_value",
    "m6_transfer_from_bank_banks",
    "m3_withdraw_agents",
    "m5_merchantpay_highest_amount",
]
RANK3_9 = RANK3_10[:9]
RANK3_8 = RANK3_10[:8]
RANK3_7 = RANK3_10[:7]
RANK3_6 = RANK3_10[:6]
RANK3_5 = RANK3_10[:5]
RANK3_4 = RANK3_10[:4]
PREFIX4 = PREFIX10[:4]
PREFIX7 = PREFIX10[:7]
PREFIX8 = PREFIX10[:8]

EXTRA_SETS = {
    "extra4": EXTRA4,
    "nom5": EXTRA3_NOM5,
    "prefix4": PREFIX4,
    "prefix7": PREFIX7,
    "prefix8": PREFIX8,
    "prefix10": PREFIX10,
    "rank3_4": RANK3_4,
    "rank3_5": RANK3_5,
    "rank3_6": RANK3_6,
    "rank3_7": RANK3_7,
    "rank3_8": RANK3_8,
    "rank3_9": RANK3_9,
    "rank3_10": RANK3_10,
    "rank3_10_nom5": [c for c in RANK3_10 if c != "m5_deposit_agents"],
    "none": [],
}


def extra_transform(
    all_rows: pd.DataFrame,
    columns: Sequence[str],
    d_all: np.ndarray,
    kind: str = "global_rank",
):
    """Transform extra factors without using labels."""
    if not columns:
        return np.empty((len(all_rows), 0), dtype=np.float64)
    values = all_rows[list(columns)].to_numpy(np.float64)
    if kind == "global_rank":
        Z = pd.DataFrame(values).rank(pct=True).to_numpy()
    elif kind == "rank":
        Z = np.empty_like(values)
        for k in np.unique(d_all):
            m = d_all == k
            Z[m] = pd.DataFrame(values[m]).rank(pct=True).to_numpy()
    elif kind == "rank_sq":
        Z = pd.DataFrame(values).rank(pct=True).to_numpy() ** 2
    elif kind.startswith("rank"):
        power = float(kind[4:])
        Z = pd.DataFrame(values).rank(pct=True).to_numpy() ** power
    elif kind == "z_cap":
        capped = np.minimum(values, np.quantile(values, 0.99, axis=0))
        Z = np.empty_like(capped)
        for k in np.unique(d_all):
            m = d_all == k
            mean = capped[m].mean(0)
            sd = capped[m].std(0)
            Z[m] = (capped[m] - mean) / np.where(sd > 0, sd, 1.0)
    elif kind == "sqrt":
        Z = np.sqrt(np.maximum(values, 0))
    elif kind == "raw":
        Z = values.copy()
    elif kind == "log1p":
        Z = np.log1p(values)
    else:
        raise ValueError(f"unknown extra transform: {kind}")
    assert np.isfinite(Z).all()
    return Z


def _extra_nll(
    w: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    offset: np.ndarray,
):
    eta = X @ w + offset
    lp, lq = log_ndtr(eta), log_ndtr(-eta)
    log_phi = -0.5 * eta**2 - 0.5 * np.log(2 * np.pi)
    grad_eta = np.where(y == 1, -np.exp(log_phi - lp), np.exp(log_phi - lq))
    return -(y * lp + (1 - y) * lq).sum(), X.T @ grad_eta


def fit_extra_gamma(
    s_mz: np.ndarray,
    Z: np.ndarray,
    y: np.ndarray,
    d_tr: np.ndarray,
    fit_mask: np.ndarray,
    sigma: float = SIGMA,
    fit_mode: str = "fixed",
):
    """Fit extra-factor coefficients with the base score as offset.

    ``fixed`` holds the per-snapshot base intercepts fixed, which regularizes the
    extra coefficients and validated better than fitting intercepts jointly.
    """
    levels = np.unique(d_tr[fit_mask])
    D = (d_tr[fit_mask][:, None] == levels[None, :]).astype(np.float64)
    offset = s_mz[fit_mask] / sigma
    if fit_mode == "fixed":
        c = np.array(
            [solve_intercept(s_mz[fit_mask & (d_tr == k)] / sigma) for k in levels]
        )
        X = Z[fit_mask]
        offset = offset + D @ c
        w0 = np.zeros(Z.shape[1])
    elif fit_mode == "free":
        c = np.full(len(levels), -1.56)
        X = np.c_[D, Z[fit_mask]]
        w0 = np.r_[c, np.zeros(Z.shape[1])]
    else:
        raise ValueError(f"unknown fit mode: {fit_mode}")

    res = minimize(
        _extra_nll,
        w0,
        args=(X, y[fit_mask], offset),
        jac=True,
        method="L-BFGS-B",
        options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 20000, "maxls": 50},
    )
    if not res.success:
        res2 = minimize(
            _extra_nll,
            res.x,
            args=(X, y[fit_mask], offset),
            jac=True,
            method="BFGS",
            options={"gtol": 1e-8, "maxiter": 20000},
        )
        if np.isfinite(res2.fun) and res2.fun <= res.fun + 1e-8:
            res = res2
    assert np.isfinite(res.x).all(), res.message
    if fit_mode == "fixed":
        gamma = res.x
        intercept = c
    else:
        gamma = res.x[-Z.shape[1] :] if Z.shape[1] else np.empty(0)
        intercept = res.x[: len(levels)]
    return gamma, intercept, res


def extra_lin(s_mz: np.ndarray, Z: np.ndarray, gamma: np.ndarray, sigma: float = SIGMA):
    return s_mz / sigma + (Z @ gamma if len(gamma) else 0.0)


def structural_oof_extra(
    s_mz: np.ndarray,
    Z: np.ndarray,
    y: np.ndarray,
    d_tr: np.ndarray,
    sigma: float = SIGMA,
    fit_mode: str = "fixed",
):
    """Leave-one-month-out structural predictions with extra linear factors."""
    n = len(d_tr)
    lin_oof = np.empty(n, dtype=np.float64)
    gammas = []
    intercepts = []
    for k in range(4):
        fit = d_tr != k
        gamma, intercept, _ = fit_extra_gamma(
            s_mz[:n], Z[:n], y, d_tr, fit, sigma, fit_mode
        )
        lin = extra_lin(s_mz[:n], Z[:n], gamma, sigma)
        lin_oof[d_tr == k] = lin[d_tr == k]
        gammas.append(gamma)
        intercepts.append(intercept)
    p_oof = calibrated(lin_oof, d_tr)
    return p_oof, lin_oof, np.asarray(gammas), intercepts


def structural_test_extra(
    s_mz: np.ndarray,
    Z: np.ndarray,
    y: np.ndarray,
    d_tr: np.ndarray,
    d_te: np.ndarray,
    sigma: float = SIGMA,
    fit_mode: str = "fixed",
):
    """Fit extra factors on all training months and predict the test snapshots."""
    n = len(d_tr)
    gamma, intercept, _ = fit_extra_gamma(
        s_mz[:n],
        Z[:n],
        y,
        d_tr,
        np.ones(n, dtype=bool),
        sigma,
        fit_mode,
    )
    lin_all = extra_lin(s_mz, Z, gamma, sigma)
    p_test = calibrated(lin_all[n:], d_te)
    return p_test, lin_all, gamma, intercept


def build_extra_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    d_tr: np.ndarray,
    d_te: np.ndarray,
    variant: str,
    transform: str = "global_rank",
):
    all_rows = pd.concat([train.drop(columns=TARGET), test], ignore_index=True)
    d_all = np.r_[d_tr, d_te]
    R = risk_factors(all_rows)
    s_mz = month_z_score(R, d_all)
    columns = EXTRA_SETS[variant]
    Z = extra_transform(all_rows, columns, d_all, transform)
    return all_rows, d_all, R, s_mz, Z


def build_residual_features(
    all_rows: pd.DataFrame,
    R: np.ndarray,
    d_all: np.ndarray,
    drop_cols: Sequence[str] = (),
    month_scale: str = "rank",
):
    X = features(all_rows, R, drop_cols)
    if month_scale != "none":
        X = month_scaled(X, d_all, month_scale)
    return X


def feature_names(all_rows: pd.DataFrame, drop_cols: Sequence[str] = ()):
    from v6_common import CATS

    numeric = [c for c in all_rows.columns if c not in [ID, *CATS, *drop_cols]]
    cats = [
        f"{c}_{v}"
        for c in CATS
        for v in sorted(all_rows[c].unique())
    ]
    factors = [f"F{i}" for i in range(7)]
    return numeric + cats + factors


def extra_summary(
    p_base: np.ndarray,
    p_extra: np.ndarray,
    y: np.ndarray,
    d_tr: np.ndarray,
):
    base_score, base_ll, base_auc = composite(y, p_base)
    extra_score, extra_ll, extra_auc = composite(y, p_extra)
    per_month = []
    for k in range(4):
        m = d_tr == k
        s_b, ll_b, auc_b = composite(y[m], p_base[m])
        s_e, ll_e, auc_e = composite(y[m], p_extra[m])
        per_month.append(
            {
                "snapshot": k,
                "base_composite": s_b,
                "extra_composite": s_e,
                "gain": s_e - s_b,
                "base_log_loss": ll_b,
                "extra_log_loss": ll_e,
                "base_auc": auc_b,
                "extra_auc": auc_e,
            }
        )
    return {
        "base": {"composite": base_score, "log_loss": base_ll, "auc": base_auc},
        "extra": {"composite": extra_score, "log_loss": extra_ll, "auc": extra_auc},
        "gain": extra_score - base_score,
        "per_month": per_month,
    }
