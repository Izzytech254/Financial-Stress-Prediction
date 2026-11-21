from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from v6_common import month_scaled

INFLOW = ("received", "deposit", "transfer_from_bank")
OUTFLOW = ("paybill", "merchantpay", "mm_send", "withdraw")
FAMILIES = ("paybill", "merchantpay", "transfer_from_bank", "mm_send", "received", "deposit", "withdraw")


def tx(rows: pd.DataFrame, month: int, family: str, metric: str) -> np.ndarray:
    suffix = {
        "paybill": "companies",
        "merchantpay": "merchants",
        "transfer_from_bank": "banks",
        "mm_send": "recipients",
        "received": "senders",
        "deposit": "agents",
        "withdraw": "agents",
    }[family] if metric == "counter" else metric
    return rows[f"m{month}_{family}_{suffix}"].to_numpy(np.float64)


def balance(rows: pd.DataFrame, month: int) -> np.ndarray:
    return rows[f"m{month}_daily_avg_bal"].to_numpy(np.float64)


def flows(rows: pd.DataFrame, months: Sequence[int]):
    bal = np.column_stack([balance(rows, m) for m in months])
    inc = np.column_stack([sum(tx(rows, m, f, "total_value") for f in INFLOW) for m in months])
    out = np.column_stack([sum(tx(rows, m, f, "total_value") for f in OUTFLOW) for m in months])
    spend = np.column_stack([sum(tx(rows, m, f, "total_value") for f in ("paybill", "merchantpay")) for m in months])
    return bal, inc, out, spend, inc - out


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


def chronological(a: np.ndarray) -> np.ndarray:
    return a[:, ::-1]


def safe_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.float64) / (np.abs(np.asarray(b, dtype=np.float64)) + 1.0)


def additive_stats(a: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "mean": a.mean(axis=1),
        "std": a.std(axis=1),
        "min": a.min(axis=1),
        "max": a.max(axis=1),
        "last": a[:, -1],
    }


def ewma_last(a: np.ndarray, alpha: float) -> np.ndarray:
    out = a[:, 0].copy()
    for j in range(1, a.shape[1]):
        out = (1.0 - alpha) * out + alpha * a[:, j]
    return out


def peer_relative(
    rows: pd.DataFrame,
    d_all: np.ndarray,
    values: np.ndarray,
    group_col: str,
    names: Sequence[str],
):
    keys = rows[group_col].where(rows[group_col].notna(), "__MISSING__").astype(str).to_numpy()
    frame = pd.DataFrame(values, columns=list(names))
    frame.insert(0, "peer", keys)
    frame.insert(0, "date", d_all)
    grouped = frame.groupby(["date", "peer"], sort=False)
    means = grouped[list(names)].transform("mean").to_numpy(np.float64)
    stds = grouped[list(names)].transform(lambda s: s.std(ddof=0)).to_numpy(np.float64)
    ranks = grouped[list(names)].rank(pct=True).to_numpy(np.float64) - 0.5
    z = np.where(stds > 1e-8, (values - means) / (stds + 1e-8), 0.0)
    return z, ranks


def build_extra_v2(
    rows: pd.DataFrame,
    d_all: np.ndarray,
    groups: Iterable[str],
    months: Sequence[int] = (1, 2, 3, 4, 5, 6),
):
    groups = list(groups)
    months = list(months)
    bal, inc, out, spend, net = flows(rows, months)
    cols: list[np.ndarray] = []
    names: list[str] = []

    def add(name: str, values: np.ndarray) -> None:
        v = np.asarray(values, dtype=np.float64)
        v = np.where(np.isfinite(v), v, 0.0)
        cols.append(v)
        names.append(name)

    # existing groups (same logic as features_cv)
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
        vals = np.column_stack([tx(rows, m, f, "total_value") for f in OUTFLOW for m in months])
        total = vals.sum(axis=1) + 1.0
        shares = (vals + 1.0) / (vals.sum(axis=1)[:, None] + len(OUTFLOW) * len(months))
        hhi = (shares * shares).sum(axis=1)
        entropy = -(shares * np.log(shares)).sum(axis=1)
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

    # new groups
    if "cross_ratio" in groups:
        inc_s = inc.sum(1) + 1.0
        out_s = out.sum(1) + 1.0
        spend_s = spend.sum(1) + 1.0
        bal_s = bal.sum(1) + 1.0
        withdraw = np.column_stack([tx(rows, m, "withdraw", "total_value") for m in months]).sum(1) + 1.0
        deposit = np.column_stack([tx(rows, m, "deposit", "total_value") for m in months]).sum(1) + 1.0
        recv = np.column_stack([tx(rows, m, "received", "total_value") for m in months]).sum(1) + 1.0
        add("X_cr_out_inc", out_s / inc_s)
        add("X_cr_spend_inc", spend_s / inc_s)
        add("X_cr_withdraw_inc", withdraw / inc_s)
        add("X_cr_bal_inc", bal_s / inc_s)
        add("X_cr_deposit_inc", deposit / inc_s)
        add("X_cr_recv_inc", recv / inc_s)
        add("X_cr_net_inc", (inc_s - out_s) / inc_s)
        add("X_cr_spend_out", spend_s / out_s)
        add("X_cr_withdraw_out", withdraw / out_s)
        # recent 3 vs older 3
        r, o = slice(0,3), slice(3,6)
        add("X_cr_inc_r_o", (inc[:,r].mean(1)+1)/(inc[:,o].mean(1)+1))
        add("X_cr_out_r_o", (out[:,r].mean(1)+1)/(out[:,o].mean(1)+1))
        add("X_cr_bal_r_o", (bal[:,r].mean(1)+1)/(bal[:,o].mean(1)+1))
        add("X_cr_spend_r_o", (spend[:,r].mean(1)+1)/(spend[:,o].mean(1)+1))
        add("X_cr_net_r_o", (net[:,r].mean(1)+1)/(net[:,o].mean(1)+1))

    if "volatility" in groups:
        for key, a in (("bal", bal), ("inc", inc), ("out", out), ("net", net)):
            add(f"X_vol_range_{key}", a.max(1) - a.min(1))
            add(f"X_vol_iqr_{key}", np.percentile(a,75,axis=1) - np.percentile(a,25,axis=1))
            add(f"X_vol_max_{key}", a.max(1))
            add(f"X_vol_min_{key}", a.min(1))
            add(f"X_vol_last_{key}", a[:,0])  # m1
            add(f"X_vol_first_{key}", a[:,-1]) # m6
            add(f"X_vol_diff_last_first_{key}", a[:,0] - a[:,-1])

    if "month_shape" in groups:
        # per-family volume/value trends recent vs old
        for fam in FAMILIES:
            v = np.column_stack([tx(rows,m,fam,"total_value") for m in months])
            add(f"X_ms_slope_{fam}", log_slope(v))
            add(f"X_ms_cv_{fam}", cv(v))
            add(f"X_ms_ro_{fam}", recent_ratio(v))
            add(f"X_ms_zero_{fam}", (v<=0).sum(1).astype(np.float64))
        # zero months counts overall
        add("X_ms_zero_inc", (inc<=0).sum(1).astype(np.float64))
        add("X_ms_zero_out", (out<=0).sum(1).astype(np.float64))

    if "profile_num" in groups:
        arpu = rows["arpu"].to_numpy(np.float64)
        age = rows["age"].to_numpy(np.float64)
        act = rows["x_90_d_activity_rate"].to_numpy(np.float64)
        inc_s = inc.sum(1) + 1.0
        bal_s = bal.mean(1) + 1.0
        add("X_pn_arpu_log", np.log1p(arpu))
        add("X_pn_age", age)
        add("X_pn_activity", act)
        add("X_pn_inc_per_arpu", inc_s / (arpu + 1.0))
        add("X_pn_bal_per_arpu", bal_s / (arpu + 1.0))
        add("X_pn_age_x_act", age * act)
        add("X_pn_arpu_x_act", arpu * act)

    if "diff2_flows" in groups:
        series = {"bal": bal, "inc": inc, "out": out, "net": net}
        for key, raw in series.items():
            a = chronological(raw)
            d1 = np.diff(a, axis=1)
            d2 = np.diff(d1, axis=1)
            nd1 = safe_ratio(d1, a[:, :-1])
            nd2 = safe_ratio(d2, d1[:, :-1])
            for label, arr in (("d1", d1), ("d2", d2), ("nd1", nd1), ("nd2", nd2)):
                stats = additive_stats(arr)
                for stat, values in stats.items():
                    add(f"X_d2_{key}_{label}_{stat}", values)

    if "window_ewma" in groups:
        series = {"bal": bal, "inc": inc, "out": out, "net": net}
        for key, raw in series.items():
            a = chronological(raw)
            e3 = ewma_last(a, 0.3)
            e6 = ewma_last(a, 0.6)
            older = a[:, :-1].mean(axis=1)
            add(f"X_we_{key}_last2_first2", safe_ratio(a[:, -2:].mean(axis=1), a[:, :2].mean(axis=1)))
            add(f"X_we_{key}_latest_prev5", safe_ratio(a[:, -1] - older, older))
            add(f"X_we_{key}_ewma3", e3)
            add(f"X_we_{key}_ewma6", e6)
            add(f"X_we_{key}_ewma3_minus_6", safe_ratio(e3 - e6, e6))
            add(f"X_we_{key}_latest_minus_ewma3", safe_ratio(a[:, -1] - e3, e3))
            add(f"X_we_{key}_range", (a.max(axis=1) - a.min(axis=1)) / (np.abs(a.mean(axis=1)) + 1.0))

    if "bal_shape_v3" in groups:
        a = chronological(bal)
        mean = a.mean(axis=1)
        median = np.median(a, axis=1)
        span = a.max(axis=1) - a.min(axis=1)
        latest = a[:, -1]
        peak_index = a.argmax(axis=1) / (a.shape[1] - 1)
        trough_index = a.argmin(axis=1) / (a.shape[1] - 1)
        add("X_bs_range_norm", span / (np.abs(mean) + 1.0))
        add("X_bs_min_norm", a.min(axis=1) / (np.abs(mean) + 1.0))
        add("X_bs_max_norm", a.max(axis=1) / (np.abs(mean) + 1.0))
        add("X_bs_std_norm", a.std(axis=1) / (np.abs(mean) + 1.0))
        add("X_bs_below_half_mean", (a < 0.5 * mean[:, None]).sum(axis=1).astype(np.float64))
        add("X_bs_below_mean", (a < mean[:, None]).sum(axis=1).astype(np.float64))
        add("X_bs_below_median", (a < median[:, None]).sum(axis=1).astype(np.float64))
        add("X_bs_peak_index", peak_index)
        add("X_bs_trough_index", trough_index)
        add("X_bs_peak_to_latest", safe_ratio(a.max(axis=1) - latest, span))
        add("X_bs_recovery_from_trough", safe_ratio(latest - a.min(axis=1), span))
        add("X_bs_latest_vs_median", safe_ratio(latest - median, median))
        add("X_bs_increase_months", (np.diff(a, axis=1) > 0).sum(axis=1).astype(np.float64))

    if "arpu_scale_v3" in groups:
        arpu = rows["arpu"].to_numpy(np.float64)
        series = {
            "bal": bal,
            "inc": inc,
            "out": out,
            "net": net,
            "spend": spend,
            "withdraw": np.column_stack([tx(rows, m, "withdraw", "total_value") for m in months]),
            "deposit": np.column_stack([tx(rows, m, "deposit", "total_value") for m in months]),
        }
        denom = np.abs(arpu) + 1.0
        for key, raw in series.items():
            scaled = raw / denom[:, None]
            stats = additive_stats(scaled)
            for stat, values in stats.items():
                add(f"X_as_{key}_{stat}", values)
            add(f"X_as_{key}_total", raw.sum(axis=1) / denom)

    if "peer_label_free" in groups:
        peer_names = [
            "bal_m1",
            "inc_m1",
            "out_m1",
            "net_m1",
            "bal_mean",
            "out_mean",
            "net_mean",
            "bal_range",
            "inc_cv",
            "out_cv",
        ]
        peer_values = np.column_stack(
            [
                bal[:, 0],
                inc[:, 0],
                out[:, 0],
                net[:, 0],
                bal.mean(axis=1),
                out.mean(axis=1),
                net.mean(axis=1),
                (bal.max(axis=1) - bal.min(axis=1)) / (np.abs(bal.mean(axis=1)) + 1.0),
                inc.std(axis=1) / (np.abs(inc.mean(axis=1)) + 1.0),
                out.std(axis=1) / (np.abs(out.mean(axis=1)) + 1.0),
            ]
        )
        for peer_col in ("segment", "region", "earning_pattern"):
            z, rank = peer_relative(rows, d_all, peer_values, peer_col, peer_names)
            for j, name in enumerate(peer_names):
                add(f"X_peer_{peer_col}_{name}_z", z[:, j])
                add(f"X_peer_{peer_col}_{name}_rank", rank[:, j])

    if "activity_types_v3" in groups:
        active = np.stack([np.column_stack([tx(rows, m, f, "total_value") for m in months]) > 0 for f in FAMILIES], axis=2)
        recent = active[:, -3:, :]
        older = active[:, :3, :]
        for j, fam in enumerate(FAMILIES):
            a = active[:, :, j]
            add(f"X_at_{fam}_zero_count", (~a).sum(axis=1).astype(np.float64))
            add(f"X_at_{fam}_zero_change", (~a[:, -3:]).sum(axis=1) - (~a[:, :3]).sum(axis=1))
            add(f"X_at_{fam}_recent_active", a[:, -3:].sum(axis=1).astype(np.float64))
            add(f"X_at_{fam}_latest_active", a[:, -1].astype(np.float64))
        types = active.sum(axis=2)
        recent_types = types[:, -2:]
        older_types = types[:, :2]
        add("X_at_types_latest", types[:, -1].astype(np.float64))
        add("X_at_types_recent2_mean", recent_types.mean(axis=1))
        add("X_at_types_first2_mean", older_types.mean(axis=1))
        add("X_at_types_change", recent_types.mean(axis=1) - older_types.mean(axis=1))
        add("X_at_types_range", types.max(axis=1) - types.min(axis=1))
        add("X_at_types_mean", types.mean(axis=1))
        add("X_at_types_std", types.std(axis=1))
        add("X_at_new_recent", (recent.sum(axis=1) > 0).sum(axis=1) - (older.sum(axis=1) > 0).sum(axis=1))

    if not cols:
        return np.empty((len(rows), 0), dtype=np.float32), []
    X = np.column_stack(cols)
    # month-wise rank scaling like existing (unsupervised, within snapshot)
    X = month_scaled(X, d_all, "rank").astype(np.float32)
    return X, names
