"""Feature engineering v3: v2 history features plus volatility/trend features."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from features_v2 import build_history_features
from solution import EPS, MONTHS, TARGET, TYPES


def _safe_log1p(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def _slope(mat: np.ndarray) -> np.ndarray:
    """Linear slope from oldest (m6) to newest (m1)."""
    x = np.arange(mat.shape[1], dtype=np.float64)
    x = x - x.mean()
    y = mat[:, ::-1]
    y = y - y.mean(axis=1, keepdims=True)
    return (y * x).sum(axis=1) / (x * x).sum()


def _longest_recent_run(mask: np.ndarray) -> np.ndarray:
    """Longest consecutive True run starting from the newest column (m1)."""
    out = np.zeros(mask.shape[0], dtype=np.float64)
    active = np.ones(mask.shape[0], dtype=bool)
    for j in range(mask.shape[1]):
        active &= mask[:, j]
        out += active
    return out


def build_history_features_v3(
    train: pd.DataFrame, test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    hist_tr, hist_te = build_history_features(train, test)
    df = pd.concat([train.drop(columns=[TARGET]), test], ignore_index=True)
    n_train = len(train)

    bals_raw = np.column_stack(
        [pd.to_numeric(df[f"m{i}_daily_avg_bal"], errors="coerce").values for i in MONTHS]
    ).astype(np.float64)
    bals = np.log1p(np.clip(bals_raw, 0, None))
    monthly_in = np.column_stack(
        [df[f"m{i}_deposit_total_value"].values + df[f"m{i}_received_total_value"].values
         + df[f"m{i}_transfer_from_bank_total_value"].values for i in MONTHS]
    ).astype(np.float64)
    monthly_out = np.column_stack(
        [df[f"m{i}_withdraw_total_value"].values + df[f"m{i}_mm_send_total_value"].values
         + df[f"m{i}_paybill_total_value"].values + df[f"m{i}_merchantpay_total_value"].values
         for i in MONTHS]
    ).astype(np.float64)
    monthly_vol = np.column_stack(
        [sum(pd.to_numeric(df[f"m{i}_{t}_volume"], errors="coerce").values for t in TYPES)
         for i in MONTHS]
    ).astype(np.float64)

    net = monthly_in - monthly_out
    net_log = _safe_log1p(net)
    in_log = np.log1p(np.clip(monthly_in, 0, None))
    out_log = np.log1p(np.clip(monthly_out, 0, None))
    vol_log = np.log1p(np.clip(monthly_vol, 0, None))
    runway = np.clip(bals_raw / (monthly_out / 30.0 + EPS), 0, 1000)
    runway_log = np.log1p(runway)
    out_in = monthly_out / (monthly_in + EPS)

    feats: Dict[str, np.ndarray] = {}

    # Balance volatility and trend.
    feats["h3_bal_std"] = bals.std(axis=1)
    feats["h3_bal_range"] = bals.max(axis=1) - bals.min(axis=1)
    feats["h3_bal_mean_abs_change"] = np.abs(np.diff(bals[:, ::-1], axis=1)).mean(axis=1)
    dbal = np.diff(bals[:, ::-1], axis=1)
    feats["h3_bal_max_drop"] = np.maximum(-dbal.min(axis=1), 0.0)
    feats["h3_bal_max_rise"] = np.maximum(dbal.max(axis=1), 0.0)
    feats["h3_bal_slope"] = _slope(bals)
    feats["h3_bal_slope_norm"] = feats["h3_bal_slope"] / (bals.mean(axis=1) + EPS)
    feats["h3_bal_recent_vs_old"] = (bals[:, :3].mean(axis=1) + EPS) / (
        bals[:, 3:].mean(axis=1) + EPS
    ) - 1.0
    feats["h3_bal_m1_over_m2"] = (bals[:, 0] + EPS) / (bals[:, 1] + EPS)
    feats["h3_bal_m2_over_m3"] = (bals[:, 1] + EPS) / (bals[:, 2] + EPS)

    # Net flow.
    feats["h3_net_mean"] = net_log.mean(axis=1)
    feats["h3_net_std"] = net_log.std(axis=1)
    feats["h3_net_min"] = net_log.min(axis=1)
    feats["h3_net_max"] = net_log.max(axis=1)
    feats["h3_net_slope"] = _slope(net_log)
    feats["h3_net_negative_months"] = (net < 0).sum(axis=1)
    feats["h3_net_consec_negative"] = _longest_recent_run(net < 0)
    feats["h3_net_recent_share"] = net[:, :3].sum(axis=1) / (
        np.abs(net).sum(axis=1) + EPS
    )

    # Inflow / outflow volatility and trend.
    feats["h3_inflow_slope"] = _slope(in_log)
    feats["h3_outflow_slope"] = _slope(out_log)
    feats["h3_inflow_max_drop"] = np.maximum(-np.diff(in_log[:, ::-1], axis=1).min(axis=1), 0.0)
    feats["h3_outflow_max_spike"] = np.maximum(np.diff(out_log[:, ::-1], axis=1).max(axis=1), 0.0)
    feats["h3_inflow_recent_vs_old"] = (monthly_in[:, :3].mean(axis=1) + EPS) / (
        monthly_in[:, 3:].mean(axis=1) + EPS
    ) - 1.0
    feats["h3_outflow_recent_vs_old"] = (monthly_out[:, :3].mean(axis=1) + EPS) / (
        monthly_out[:, 3:].mean(axis=1) + EPS
    ) - 1.0
    feats["h3_out_in_ratio_mean"] = out_in.mean(axis=1)
    feats["h3_out_in_ratio_m1"] = out_in[:, 0]
    feats["h3_out_in_ratio_slope"] = _slope(out_in)

    # Runway.
    feats["h3_runway_std"] = runway_log.std(axis=1)
    feats["h3_runway_min"] = runway_log.min(axis=1)
    feats["h3_runway_max"] = runway_log.max(axis=1)
    feats["h3_runway_slope"] = _slope(runway_log)
    feats["h3_runway_recent_vs_old"] = (runway[:, :3].mean(axis=1) + EPS) / (
        runway[:, 3:].mean(axis=1) + EPS
    ) - 1.0

    # Activity.
    feats["h3_volume_slope"] = _slope(vol_log)
    feats["h3_volume_recent_share"] = monthly_vol[:, :3].sum(axis=1) / (
        monthly_vol.sum(axis=1) + EPS
    )
    feats["h3_volume_cv"] = monthly_vol.std(axis=1) / (monthly_vol.mean(axis=1) + EPS)

    # Interactions.
    feats["h3_drawdown_x_outflow_cv"] = (
        (bals.max(axis=1) - bals[:, 0]) / (bals.max(axis=1) + EPS)
    ) * (monthly_out.std(axis=1) / (monthly_out.mean(axis=1) + EPS))
    feats["h3_netneg_x_drawdown"] = feats["h3_net_negative_months"] * (
        (bals.max(axis=1) - bals[:, 0]) / (bals.max(axis=1) + EPS)
    )
    feats["h3_balstd_x_outflowcv"] = feats["h3_bal_std"] * (
        monthly_out.std(axis=1) / (monthly_out.mean(axis=1) + EPS)
    )

    new_df = pd.DataFrame(feats)
    out = pd.concat([hist_tr, new_df.iloc[:n_train].reset_index(drop=True)], axis=1)
    out_te = pd.concat([hist_te, new_df.iloc[n_train:].reset_index(drop=True)], axis=1)
    return out, out_te
