"""Feature engineering v2: original features + customer-level and history features.

All customer aggregates use features only (never labels). They are available in
two variants:
  tr: aggregates from Train rows only. Test rows use the customer's Train stats.
  tt: aggregates from Train+Test rows, using leave-one-row-out mean/std.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from solution import CAT_COLS, EPS, MONTHS, TARGET, TYPES, build_features


def _safe_log1p(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def _sum_6m(df: pd.DataFrame, prefixes: list[str]) -> np.ndarray:
    out = np.zeros(len(df), dtype=np.float64)
    for p in prefixes:
        for i in MONTHS:
            out += pd.to_numeric(df[f"m{i}_{p}_total_value"], errors="coerce").values
    return out


def _family_values(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    bals_raw = np.column_stack(
        [pd.to_numeric(df[f"m{i}_daily_avg_bal"], errors="coerce").values for i in MONTHS]
    ).astype(np.float64)
    bals = np.log1p(np.clip(bals_raw, 0, None))
    inflow_raw = _sum_6m(df, ["deposit", "received", "transfer_from_bank"])
    outflow_raw = _sum_6m(df, ["withdraw", "mm_send", "paybill", "merchantpay"])
    monthly_in = np.column_stack(
        [df[f"m{i}_deposit_total_value"].values + df[f"m{i}_received_total_value"].values
         + df[f"m{i}_transfer_from_bank_total_value"].values for i in MONTHS]
    ).astype(np.float64)
    monthly_out = np.column_stack(
        [df[f"m{i}_withdraw_total_value"].values + df[f"m{i}_mm_send_total_value"].values
         + df[f"m{i}_paybill_total_value"].values + df[f"m{i}_merchantpay_total_value"].values
         for i in MONTHS]
    ).astype(np.float64)
    runway = np.clip(
        bals_raw / (monthly_out.mean(axis=1, keepdims=True) / 30.0 + EPS), 0, 1000
    ).mean(axis=1)
    total_vol = np.zeros(len(df), dtype=np.float64)
    for t in TYPES:
        for i in MONTHS:
            total_vol += pd.to_numeric(df[f"m{i}_{t}_volume"], errors="coerce").values

    vals = {
        "bal_mean": bals.mean(axis=1),
        "bal_m1": bals[:, 0],
        "inflow": np.log1p(np.clip(inflow_raw, 0, None)),
        "outflow": np.log1p(np.clip(outflow_raw, 0, None)),
        "net": _safe_log1p(inflow_raw - outflow_raw),
        "withdraw": np.log1p(_sum_6m(df, ["withdraw"])),
        "paybill": np.log1p(_sum_6m(df, ["paybill"])),
        "merchantpay": np.log1p(_sum_6m(df, ["merchantpay"])),
        "mm_send": np.log1p(_sum_6m(df, ["mm_send"])),
        "received": np.log1p(_sum_6m(df, ["received"])),
        "deposit": np.log1p(_sum_6m(df, ["deposit"])),
        "total_vol": np.log1p(total_vol),
        "runway": np.log1p(runway),
    }
    return vals


def _customer_key(df: pd.DataFrame) -> np.ndarray:
    return pd.util.hash_pandas_object(df[CAT_COLS + ["arpu", "age", "x_90_d_activity_rate"]],
                                     index=False).values


def _stats_tt(
    values: np.ndarray, groups: np.ndarray, n_groups: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    counts = np.bincount(groups, minlength=n_groups).astype(np.float64)
    sums = np.bincount(groups, weights=values, minlength=n_groups)
    sumsqs = np.bincount(groups, weights=values * values, minlength=n_groups)
    g = groups
    cnt = counts[g]
    mean = (sums[g] - values) / np.maximum(cnt - 1.0, 1.0)
    var = (sumsqs[g] - values * values) / np.maximum(cnt - 1.0, 1.0) - mean * mean
    std = np.sqrt(np.maximum(var, 0.0))
    s = pd.Series(values)
    gg = pd.Series(groups)
    mn = s.groupby(gg).transform("min").values
    mx = s.groupby(gg).transform("max").values
    return mean, std, mn, mx


def _stats_tr(
    values: np.ndarray, groups: np.ndarray, n_groups: int, n_train: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    g_tr = groups[:n_train]
    v_tr = values[:n_train]
    counts = np.bincount(g_tr, minlength=n_groups).astype(np.float64)
    sums = np.bincount(g_tr, weights=v_tr, minlength=n_groups)
    sumsqs = np.bincount(g_tr, weights=v_tr * v_tr, minlength=n_groups)
    g = groups
    is_train = np.arange(len(values)) < n_train
    cnt = counts[g]
    mean = np.empty(len(values), dtype=np.float64)
    var = np.empty(len(values), dtype=np.float64)
    mean[is_train] = (sums[g[is_train]] - values[is_train]) / np.maximum(
        cnt[is_train] - 1.0, 1.0
    )
    var[is_train] = (
        (sumsqs[g[is_train]] - values[is_train] ** 2)
        / np.maximum(cnt[is_train] - 1.0, 1.0)
        - mean[is_train] ** 2
    )
    mean[~is_train] = sums[g[~is_train]] / np.maximum(cnt[~is_train], 1.0)
    var[~is_train] = (
        sumsqs[g[~is_train]] / np.maximum(cnt[~is_train], 1.0)
        - mean[~is_train] ** 2
    )
    std = np.sqrt(np.maximum(var, 0.0))
    s_tr = pd.Series(v_tr).groupby(g_tr).min()
    x_tr = pd.Series(v_tr).groupby(g_tr).max()
    gids = np.arange(n_groups)
    mn = s_tr.reindex(gids).values[g]
    mx = x_tr.reindex(gids).values[g]
    return mean, std, mn, mx


def build_customer_features(
    train: pd.DataFrame, test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    df = pd.concat([train.drop(columns=[TARGET]), test], ignore_index=True)
    n_train = len(train)
    key = _customer_key(df)
    groups, _ = pd.factorize(key)
    n_groups = int(groups.max()) + 1
    vals = _family_values(df)
    families = list(vals.keys())
    minmax_families = {"bal_mean", "bal_m1", "inflow", "outflow", "runway"}

    tr_feats: Dict[str, np.ndarray] = {}
    tt_feats: Dict[str, np.ndarray] = {}
    for name in families:
        v = np.asarray(vals[name], dtype=np.float64)
        m_tt, s_tt, mn_tt, mx_tt = _stats_tt(v, groups, n_groups)
        m_tr, s_tr, mn_tr, mx_tr = _stats_tr(v, groups, n_groups, n_train)
        rank_tt = pd.Series(v).groupby(groups).rank(pct=True).values
        for prefix, m, s, mn, mx, rank in [
            ("ctt", m_tt, s_tt, mn_tt, mx_tt, rank_tt),
            ("ctr", m_tr, s_tr, mn_tr, mx_tr, None),
        ]:
            target = tt_feats if prefix == "ctt" else tr_feats
            target[f"{prefix}_{name}_mean"] = m
            target[f"{prefix}_{name}_std"] = s
            target[f"{prefix}_{name}_dev"] = v - m
            target[f"{prefix}_{name}_z"] = (v - m) / (s + EPS)
            target[f"{prefix}_{name}_ratio"] = (v + EPS) / (m + EPS)
            if rank is not None:
                target[f"{prefix}_{name}_rank"] = rank
            if name in minmax_families:
                target[f"{prefix}_{name}_min"] = mn
                target[f"{prefix}_{name}_max"] = mx
    tr_df = pd.DataFrame(tr_feats)
    tt_df = pd.DataFrame(tt_feats)
    return (
        tr_df.iloc[:n_train].reset_index(drop=True),
        tr_df.iloc[n_train:].reset_index(drop=True),
        tt_df.iloc[:n_train].reset_index(drop=True),
        tt_df.iloc[n_train:].reset_index(drop=True),
        groups[:n_train],
        key,
    )


def build_history_features(
    train: pd.DataFrame, test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
    runway = np.clip(
        bals_raw / (monthly_out / 30.0 + EPS), 0, 1000
    )

    feats: Dict[str, np.ndarray] = {}
    seq = bals[:, ::-1]
    peak = np.maximum.accumulate(seq, axis=1)
    dd = (peak - seq) / (np.abs(peak) + EPS)
    feats["hist_bal_max_drawdown"] = dd.max(axis=1)
    feats["hist_bal_latest_drawdown"] = (bals.max(axis=1) - bals[:, 0]) / (
        bals.max(axis=1) + EPS
    )
    feats["hist_bal_m1_over_max"] = (bals[:, 0] + EPS) / (bals.max(axis=1) + EPS)
    feats["hist_bal_min_over_mean"] = (bals.min(axis=1) + EPS) / (bals.mean(axis=1) + EPS)
    decline = (bals[:, :-1] < bals[:, 1:])
    consec = np.zeros(len(df), dtype=np.float64)
    active = np.ones(len(df), dtype=bool)
    for j in range(5):
        active &= decline[:, j]
        consec += active
    feats["hist_bal_consec_declines"] = consec
    feats["hist_bal_decline_months"] = decline.sum(axis=1)
    feats["hist_bal_near_zero_10"] = (bals_raw < 10).sum(axis=1)
    feats["hist_bal_near_zero_100"] = (bals_raw < 100).sum(axis=1)
    feats["hist_net_positive_months"] = (monthly_in > monthly_out).sum(axis=1)

    for kind, mat in [("inflow", monthly_in), ("outflow", monthly_out)]:
        total = mat.sum(axis=1, keepdims=True) + EPS
        share = mat / total
        feats[f"hist_{kind}_hhi"] = (share * share).sum(axis=1)
        feats[f"hist_{kind}_entropy"] = -(share * np.log(share + EPS)).sum(axis=1)
    feats["hist_inflow_recent_share"] = monthly_in[:, :3].sum(axis=1) / (
        monthly_in.sum(axis=1) + EPS
    )
    feats["hist_outflow_recent_share"] = monthly_out[:, :3].sum(axis=1) / (
        monthly_out.sum(axis=1) + EPS
    )
    feats["hist_runway_min_over_mean"] = (runway.min(axis=1) + EPS) / (
        runway.mean(axis=1) + EPS
    )
    feats["hist_runway_m1_over_mean"] = (runway[:, 0] + EPS) / (
        runway.mean(axis=1) + EPS
    )
    feats["hist_inflow_cv"] = monthly_in.std(axis=1) / (monthly_in.mean(axis=1) + EPS)
    feats["hist_outflow_cv"] = monthly_out.std(axis=1) / (monthly_out.mean(axis=1) + EPS)
    feats["hist_out_in_m1_over_mean"] = ((monthly_out[:, 0] + EPS) / (monthly_in[:, 0] + EPS)) / (
        (monthly_out.mean(axis=1) + EPS) / (monthly_in.mean(axis=1) + EPS)
    )

    out = pd.DataFrame(feats)
    return out.iloc[:n_train].reset_index(drop=True), out.iloc[n_train:].reset_index(drop=True)


def build_v2_features(
    train: pd.DataFrame, test: pd.DataFrame
) -> Dict[str, object]:
    base_tr, base_te, y, base_names = build_features(train, test, clip=False)
    ctr_tr, ctr_te, ctt_tr, ctt_te, train_groups, keys = build_customer_features(train, test)
    hist_tr, hist_te = build_history_features(train, test)
    return {
        "base": (base_tr, base_te, base_names),
        "cust_tr": (ctr_tr, ctr_te),
        "cust_tt": (ctt_tr, ctt_te),
        "history": (hist_tr, hist_te),
        "y": y,
        "train_groups": train_groups,
        "keys": keys,
    }
