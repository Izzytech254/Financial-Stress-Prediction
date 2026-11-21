from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from v6_common import month_scaled

COUNTER = {
    "paybill": "companies",
    "merchantpay": "merchants",
    "transfer_from_bank": "banks",
    "mm_send": "recipients",
    "received": "senders",
    "deposit": "agents",
    "withdraw": "agents",
}
INFLOW = ("received", "deposit", "transfer_from_bank")
OUTFLOW = ("paybill", "merchantpay", "mm_send", "withdraw")
SPEND = ("paybill", "merchantpay", "mm_send", "withdraw")


def tx(rows: pd.DataFrame, month: int, family: str, metric: str) -> np.ndarray:
    suffix = COUNTER[family] if metric == "counter" else metric
    return rows[f"m{month}_{family}_{suffix}"].to_numpy(np.float64)


def balance(rows: pd.DataFrame, month: int) -> np.ndarray:
    return rows[f"m{month}_daily_avg_bal"].to_numpy(np.float64)


def flows(rows: pd.DataFrame, months: Sequence[int]):
    bal = np.column_stack([balance(rows, m) for m in months])
    inc = np.column_stack(
        [sum(tx(rows, m, f, "total_value") for f in INFLOW) for m in months]
    )
    out = np.column_stack(
        [sum(tx(rows, m, f, "total_value") for f in OUTFLOW) for m in months]
    )
    return bal, inc, out, inc - out


def log_slope(a: np.ndarray) -> np.ndarray:
    x = np.arange(a.shape[1], dtype=np.float64)
    y = np.log1p(np.maximum(a, 0))
    xc = x - x.mean()
    denom = (xc * xc).sum()
    return (y * xc).sum(axis=1) / denom


def cv(a: np.ndarray) -> np.ndarray:
    return a.std(axis=1) / (np.abs(a.mean(axis=1)) + 1.0)


def recent_ratio(a: np.ndarray) -> np.ndarray:
    split = max(1, a.shape[1] // 2)
    return (a[:, :split].mean(axis=1) + 1.0) / (a[:, split:].mean(axis=1) + 1.0)


def drawdown(a: np.ndarray) -> np.ndarray:
    peak = np.maximum.accumulate(a, axis=1)
    return (peak - a) / (peak + 1.0)


def consecutive_declines(a: np.ndarray) -> np.ndarray:
    diff = np.diff(a, axis=1)
    declined = diff < 0
    out = np.zeros(a.shape[0], dtype=np.float64)
    run = np.zeros(a.shape[0], dtype=np.float64)
    for j in range(declined.shape[1]):
        run = np.where(declined[:, j], run + 1.0, 0.0)
        out = np.maximum(out, run)
    return out


def spend_shares(rows: pd.DataFrame, months: Sequence[int]):
    vals = np.column_stack([tx(rows, m, f, "total_value") for f in SPEND for m in months])
    total = vals.sum(axis=1) + 1.0
    shares = (vals + 1.0) / (vals.sum(axis=1)[:, None] + len(SPEND) * len(months))
    hhi = (shares * shares).sum(axis=1)
    entropy = -(shares * np.log(shares)).sum(axis=1)
    return hhi, entropy, total


def build_extra(
    rows: pd.DataFrame,
    d_all: np.ndarray,
    groups: Iterable[str],
    months: Sequence[int] = (1, 2, 3, 4, 5, 6),
):
    groups = list(groups)
    months = list(months)
    bal, inc, out, net = flows(rows, months)
    cols: list[np.ndarray] = []
    names: list[str] = []

    def add(name: str, values: np.ndarray) -> None:
        cols.append(np.asarray(values, dtype=np.float64))
        names.append(name)

    if "liquidity" in groups:
        for j, m in enumerate(months):
            add(f"X_liq_runway_m{m}", bal[:, j] / (out[:, j] + 1.0))
            add(f"X_liq_cover_m{m}", inc[:, j] / (out[:, j] + 1.0))
            add(f"X_liq_net_m{m}", net[:, j] / (bal[:, j] + 1.0))
            add(f"X_liq_bal_inc_m{m}", bal[:, j] / (inc[:, j] + 1.0))
        runway = bal / (out + 1.0)
        add("X_liq_runway_mean", runway.mean(axis=1))
        add("X_liq_runway_std", runway.std(axis=1))
        add("X_liq_cover_mean", (inc / (out + 1.0)).mean(axis=1))

    if "trend" in groups:
        for key, a in (("bal", bal), ("inc", inc), ("out", out), ("net", net)):
            add(f"X_trend_slope_{key}", log_slope(a))
            add(f"X_trend_cv_{key}", cv(a))
            add(f"X_trend_recent_{key}", recent_ratio(a))

    if "stress" in groups:
        add("X_stress_drawdown_bal", drawdown(bal)[:, -1])
        add("X_stress_drawdown_recent", drawdown(bal[:, -3:])[:, -1])
        add("X_stress_declines_bal", consecutive_declines(bal))
        add("X_stress_declines_net", consecutive_declines(net))
        add("X_stress_nearzero_10", (bal < 10).sum(axis=1).astype(np.float64))
        add("X_stress_nearzero_100", (bal < 100).sum(axis=1).astype(np.float64))
        add("X_stress_negnet", (net < 0).sum(axis=1).astype(np.float64))
        add("X_stress_min_bal", bal.min(axis=1))

    if "concentration" in groups:
        hhi, entropy, total = spend_shares(rows, months)
        add("X_conc_hhi", hhi)
        add("X_conc_entropy", entropy)
        add("X_conc_total", total)
        runway = bal / (out + 1.0)
        cover = inc / (out + 1.0)
        dd = drawdown(bal)[:, -1]
        add("X_int_runway_x_outcv", runway.mean(axis=1) * cv(out))
        add("X_int_dd_x_balcv", dd * cv(bal))
        add("X_int_negnet_x_outcv", (net < 0).sum(axis=1) * cv(out))
        add("X_int_cover_x_incslope", cover.mean(axis=1) * log_slope(inc))

    if not cols:
        return np.empty((len(rows), 0), dtype=np.float32), []
    X = np.column_stack(cols)
    X = month_scaled(X, d_all, "rank").astype(np.float32)
    return X, names
