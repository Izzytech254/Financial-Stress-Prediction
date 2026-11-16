from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from v6_common import (
    DATA_DIR,
    ID,
    MZ_SLOPE,
    REPORT_DIR,
    TARGET,
    composite,
    global_z_score,
    load_data,
    month_z_score,
    risk_factors,
    snapshot_dates,
    structural_oof_avg,
    structural_oof_global,
    structural_oof_mz,
    structural_test_mz,
    validate_submission,
    write_json,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--out", default="submission_v6_structural.csv")
    ap.add_argument("--report", default=str(REPORT_DIR / "v6_structural.json"))
    ap.add_argument("--preds", default=str(REPORT_DIR / "v6_structural_predictions.npz"))
    args = ap.parse_args()

    train, test, sample = load_data(args.data_dir)
    d_tr, d_te = snapshot_dates(train, test)
    y = train[TARGET].to_numpy()
    n = len(train)
    all_rows = pd.concat([train.drop(columns=TARGET), test], ignore_index=True)
    d_all = np.r_[d_tr, d_te]

    R = risk_factors(all_rows)
    s_mz = month_z_score(R, d_all)
    s_global = global_z_score(R)

    p_oof = structural_oof_mz(s_mz, d_tr)
    p_test = structural_test_mz(s_mz, d_te)
    p_global = structural_oof_global(s_global, d_tr)
    p_avg = structural_oof_avg(s_global, s_mz, y, d_tr, d_all)

    score, ll, auc = composite(y, p_oof)
    score_global, ll_global, auc_global = composite(y, p_global)
    score_avg, ll_avg, auc_avg = composite(y, p_avg)

    per_month = []
    for k in range(4):
        m = d_tr == k
        s_k, ll_k, auc_k = composite(y[m], p_oof[m])
        per_month.append(
            {
                "snapshot": k,
                "rows": int(m.sum()),
                "positives": int(y[m].sum()),
                "composite": s_k,
                "log_loss": ll_k,
                "auc": auc_k,
            }
        )

    out = sample.copy()
    out["Target"] = p_test
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, float_format="%.17g")
    stats = validate_submission(args.out, sample)

    Path(args.preds).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.preds,
        y=y,
        oof=p_oof,
        test=p_test,
        d_tr=d_tr,
        d_te=d_te,
        s_mz=s_mz,
        p_global_oof=p_global,
        p_avg_oof=p_avg,
    )

    payload = {
        "model": "structural_mz",
        "sigma": 0.35,
        "weights": [5, 3, 3, 2, 4, 2, 1],
        "validation": {"composite": score, "log_loss": ll, "auc": auc},
        "per_month": per_month,
        "reference": {
            "global_z": {
                "composite": score_global,
                "log_loss": ll_global,
                "auc": auc_global,
            },
            "avg": {
                "composite": score_avg,
                "log_loss": ll_avg,
                "auc": auc_avg,
            },
        },
        "test": {
            **stats,
            "snapshot_means": [
                float(p_test[d_te == k].mean()) for k in (4, 5, 6)
            ],
        },
        "submission": str(Path(args.out).resolve()),
    }
    write_json(args.report, payload)

    lines = [
        "# v6 Structural Probit",
        "",
        f"- Validation composite: `{score:.9f}`",
        f"- Validation LogLoss: `{ll:.9f}`",
        f"- Validation AUC: `{auc:.9f}`",
        f"- Global-z reference: `{score_global:.9f}`",
        f"- AVG reference: `{score_avg:.9f}`",
        "",
        "| Snapshot | Rows | Positives | Composite | LogLoss | AUC |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in per_month:
        lines.append(
            f"| {row['snapshot']} | {row['rows']} | {row['positives']} | "
            f"{row['composite']:.9f} | {row['log_loss']:.9f} | {row['auc']:.9f} |"
        )
    lines.extend(
        [
            "",
            f"- Test mean: `{stats['mean']:.12f}`",
            f"- Test snapshot means: `{[round(float(p_test[d_te == k].mean()), 12) for k in (4, 5, 6)]}`",
            f"- Submission: `{args.out}`",
        ]
    )
    Path(args.report).with_suffix(".md").write_text("\n".join(lines) + "\n")

    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
