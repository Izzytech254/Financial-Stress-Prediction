from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import log_loss, roc_auc_score

from v6_common import (
    DATA_DIR,
    REPORT_DIR,
    TARGET,
    composite,
    load_data,
    month_z_score,
    per_date,
    probit_eta,
    risk_factors,
    snapshot_dates,
    validate_submission,
    write_json,
)
from v6_residual import seeds_for
from v8_screen import build_variant


class ListwiseMLP(nn.Module):
    def __init__(self, d: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def predict(model, X, idx):
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(X[idx])).cpu().numpy()


def fit_temperature(raw, y, d):
    best = None
    for t in np.linspace(0.2, 3.0, 57):
        p = per_date(probit_eta(np.clip(1.0 / (1.0 + np.exp(-t * raw)), 1e-15, 1 - 1e-15)), d)
        ll = log_loss(y, p)
        if best is None or ll < best[0]:
            best = (ll, float(t))
    return best[1]


D_TRAIN = None


def main() -> None:
    global D_TRAIN
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--variant", default="drop_m5_all")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--tag", default="listwise_drop_m5_all")
    ap.add_argument("--report", default=None)
    ap.add_argument("--preds", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    start = time.time()
    seeds = seeds_for(args.seeds)
    train, test, sample = load_data(args.data_dir)
    d_tr, d_te = snapshot_dates(train, test)
    D_TRAIN = d_tr
    y = train[TARGET].to_numpy()
    n = len(train)
    all_rows = pd.concat([train.drop(columns=TARGET), test], ignore_index=True)
    d_all = np.r_[d_tr, d_te]

    R = risk_factors(all_rows)
    s_mz = month_z_score(R, d_all)
    lin = (2.0 / 0.35) * s_mz
    X, names, drop_cols, month_scale, extra_cols = build_variant(
        args.variant, all_rows, d_all, R, None
    )
    X = np.c_[X, lin].astype(np.float32)
    names = names + ["struct_lin"]
    mu = X.mean(0)
    sd = X.std(0)
    sd[sd == 0] = 1.0
    X = ((X - mu) / sd).astype(np.float32)

    tag = args.tag
    report_path = Path(args.report or REPORT_DIR / f"v9_{tag}.json")
    preds_path = Path(args.preds) if args.preds else None
    out_path = Path(args.out) if args.out else None

    print(
        f"listwise variant={args.variant} tag={tag} features={X.shape[1]} "
        f"seeds={len(seeds)} scale={month_scale}",
        flush=True,
    )

    oof_raw = np.zeros(n, dtype=np.float64)
    best_epochs: list[int] = []
    for k in range(4):
        a = np.flatnonzero(d_tr != k)
        b = np.flatnonzero(d_tr == k)
        ik = [j for j in range(4) if j != k][-1]
        fit = a[d_tr[a] != ik]
        val = a[d_tr[a] == ik]
        fit_months = np.unique(d_tr[fit])
        val_month = int(np.unique(d_tr[val])[0])

        seeds_raw = []
        epochs_used = []
        for seed in seeds:
            set_seed(seed)
            model = ListwiseMLP(X.shape[1], args.hidden)
            opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
            best_state = None
            best_auc = -1.0
            best_epoch = 1
            for epoch in range(1, args.epochs + 1):
                model.train()
                for mth in fit_months:
                    m = fit[d_tr[fit] == mth]
                    xb = torch.from_numpy(X[m])
                    yb = torch.from_numpy(y[m].astype(np.float32))
                    opt.zero_grad()
                    logits = model(xb)
                    logp = F.log_softmax(logits, dim=0)
                    loss = -logp[yb > 0.5].mean()
                    loss.backward()
                    opt.step()
                raw_val = predict(model, X, val)
                auc = roc_auc_score(y[val], raw_val)
                if auc > best_auc:
                    best_auc = auc
                    best_epoch = epoch
                    best_state = {k2: v.detach().clone() for k2, v in model.state_dict().items()}
                elif epoch - best_epoch >= args.patience:
                    break
            model.load_state_dict(best_state)
            best_epochs.append(best_epoch)
            epochs_used.append(best_epoch)
            final = ListwiseMLP(X.shape[1], args.hidden)
            final.load_state_dict(best_state)
            opt2 = torch.optim.Adam(final.parameters(), lr=args.lr, weight_decay=args.wd)
            final.train()
            for epoch in range(best_epoch):
                for mth in np.unique(d_tr[a]):
                    m = a[d_tr[a] == mth]
                    xb = torch.from_numpy(X[m])
                    yb = torch.from_numpy(y[m].astype(np.float32))
                    opt2.zero_grad()
                    logits = final(xb)
                    logp = F.log_softmax(logits, dim=0)
                    loss = -logp[yb > 0.5].mean()
                    loss.backward()
                    opt2.step()
            seeds_raw.append(predict(final, X, b))
        raw_b = np.mean(seeds_raw, axis=0)
        oof_raw[b] = raw_b
        p_b = per_date(
            probit_eta(np.clip(1.0 / (1.0 + np.exp(-raw_b)), 1e-15, 1 - 1e-15)),
            d_tr[b],
        )
        s_k, ll_k, auc_k = composite(y[b], p_b)
        print(
            f"month {k} composite={s_k:.9f} ll={ll_k:.9f} auc={auc_k:.9f} "
            f"epochs={epochs_used}",
            flush=True,
        )

    raw_z = np.empty_like(oof_raw)
    for k in range(4):
        m = d_tr == k
        sd = oof_raw[m].std()
        raw_z[m] = (oof_raw[m] - oof_raw[m].mean()) / (sd if sd > 0 else 1.0)
    t = fit_temperature(raw_z, y, d_tr)
    p_res = per_date(
        probit_eta(np.clip(1.0 / (1.0 + np.exp(-t * raw_z)), 1e-15, 1 - 1e-15)),
        d_tr,
    )
    score_res, ll_res, auc_res = composite(y, p_res)
    per_month = []
    for k in range(4):
        m = d_tr == k
        s_k, ll_k, auc_k = composite(y[m], p_res[m])
        per_month.append({"snapshot": k, "composite": s_k, "log_loss": ll_k, "auc": auc_k})
    print(
        f"listwise LOMO composite={score_res:.9f} ll={ll_res:.9f} auc={auc_res:.9f} "
        f"temperature={t:.3f}",
        flush=True,
    )

    p_test = None
    stats = None
    if args.test:
        tr = np.arange(n)
        te = np.arange(n, n + len(test))
        final_epochs = max(1, int(np.median(best_epochs)))
        seeds_test = []
        for seed in seeds:
            set_seed(seed)
            model = ListwiseMLP(X.shape[1], args.hidden)
            opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
            model.train()
            for epoch in range(final_epochs):
                for mth in range(4):
                    m = np.flatnonzero(d_tr == mth)
                    xb = torch.from_numpy(X[m])
                    yb = torch.from_numpy(y[m].astype(np.float32))
                    opt.zero_grad()
                    logits = model(xb)
                    logp = F.log_softmax(logits, dim=0)
                    loss = -logp[yb > 0.5].mean()
                    loss.backward()
                    opt.step()
            seeds_test.append(predict(model, X, te))
        raw_test = np.mean(seeds_test, axis=0)
        raw_test_z = np.empty_like(raw_test)
        for k in (4, 5, 6):
            m = d_te == k
            sd = raw_test[m].std()
            raw_test_z[m] = (raw_test[m] - raw_test[m].mean()) / (sd if sd > 0 else 1.0)
        p_test = per_date(
            probit_eta(
                np.clip(1.0 / (1.0 + np.exp(-t * raw_test_z)), 1e-15, 1 - 1e-15)
            ),
            d_te,
        )
        if out_path is None:
            out_path = Path(f"submission_v9_{tag}.csv")
        out = sample.copy()
        out["Target"] = p_test
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False, float_format="%.17g")
        stats = validate_submission(out_path, sample)

    if preds_path is not None:
        preds_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "y": y,
            "oof": p_res,
            "d_tr": d_tr,
            "d_te": d_te,
            "rounds": np.asarray(best_epochs),
            "s_mz": s_mz,
        }
        if p_test is not None:
            payload["test"] = p_test
        np.savez_compressed(preds_path, **payload)

    elapsed = time.time() - start
    payload = {
        "model": "v9_listwise",
        "variant": args.variant,
        "tag": tag,
        "feature_count": int(X.shape[1]),
        "seeds": list(seeds),
        "n_seeds": len(seeds),
        "best_epochs": best_epochs,
        "temperature": float(t),
        "validation": {"composite": score_res, "log_loss": ll_res, "auc": auc_res},
        "per_month": per_month,
        "elapsed_seconds": elapsed,
        "preds": str(preds_path.resolve()) if preds_path else None,
        "submission": str(out_path.resolve()) if out_path else None,
    }
    write_json(report_path, payload)
    print(f"report={report_path}", flush=True)


if __name__ == "__main__":
    main()
