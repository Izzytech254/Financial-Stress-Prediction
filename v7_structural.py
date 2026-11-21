from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from v6_common import (
    DATA_DIR,
    REPORT_DIR,
    TARGET,
    composite,
    load_data,
    snapshot_dates,
    structural_oof_mz,
    validate_submission,
    write_json,
)
from v7_common import (
    EXTRA_SETS,
    build_extra_features,
    extra_summary,
    structural_oof_extra,
    structural_test_extra,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--variant", default="extra4", choices=sorted(EXTRA_SETS))
    ap.add_argument(
        "--transform",
        default="auto",
        choices=[
            "auto",
            "global_rank",
            "rank",
            "rank1.5",
            "rank2",
            "rank2.5",
            "rank3",
            "rank4",
            "rank_sq",
            "sqrt",
            "raw",
            "log1p",
            "z_cap",
        ],
    )
    ap.add_argument("--fit-mode", default="fixed", choices=["fixed", "free"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--report", default=None)
    ap.add_argument("--preds", default=None)
    args = ap.parse_args()

    transform = args.transform
    if transform == "auto":
        if args.variant.startswith("rank3"):
            transform = "rank3"
        elif args.variant == "nom5":
            transform = "sqrt"
        else:
            transform = "global_rank"

    out_path = Path(args.out or f"submission_v7_struct_{args.variant}.csv")
    report_path = Path(args.report or REPORT_DIR / f"v7_struct_{args.variant}.json")
    preds_path = Path(args.preds or REPORT_DIR / f"v7_struct_{args.variant}_predictions.npz")

    train, test, sample = load_data(args.data_dir)
    d_tr, d_te = snapshot_dates(train, test)
    y = train[TARGET].to_numpy()
    all_rows, d_all, R, s_mz, Z = build_extra_features(
        train, test, d_tr, d_te, args.variant, transform
    )
    columns = EXTRA_SETS[args.variant]

    p_base = structural_oof_mz(s_mz, d_tr)
    p_oof, lin_oof, gammas, intercepts = structural_oof_extra(
        s_mz, Z, y, d_tr, fit_mode=args.fit_mode
    )
    p_test, lin_all, gamma_full, intercept_full = structural_test_extra(
        s_mz, Z, y, d_tr, d_te, fit_mode=args.fit_mode
    )

    summary = extra_summary(p_base, p_oof, y, d_tr)
    test_snapshot_means = [
        float(p_test[d_te == k].mean()) for k in (4, 5, 6)
    ]

    out = sample.copy()
    out["Target"] = p_test
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, float_format="%.17g")
    stats = validate_submission(out_path, sample)

    preds_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        preds_path,
        y=y,
        oof=p_oof,
        test=p_test,
        base_oof=p_base,
        d_tr=d_tr,
        d_te=d_te,
        s_mz=s_mz,
        Z=Z,
        gammas=gammas,
        gamma_full=gamma_full,
        lin_oof=lin_oof,
        lin_all=lin_all,
        extra_columns=np.asarray(columns, dtype=object),
    )

    payload = {
        "model": "v7_structural_extra",
        "variant": args.variant,
        "transform": transform,
        "fit_mode": args.fit_mode,
        "extra_columns": columns,
        "sigma": 0.35,
        "base_weights": [5, 3, 3, 2, 4, 2, 1],
        "validation": summary,
        "gammas": [g.tolist() for g in gammas],
        "gamma_full": gamma_full.tolist(),
        "intercepts": [np.asarray(i).tolist() for i in intercepts],
        "intercept_full": np.asarray(intercept_full).tolist(),
        "test": {
            **stats,
            "snapshot_means": test_snapshot_means,
        },
        "submission": str(out_path.resolve()),
    }
    write_json(report_path, payload)

    lines = [
        "# v7 Structural Extra",
        "",
        f"- Variant: `{args.variant}`",
        f"- Transform: `{transform}`",
        f"- Fit mode: `{args.fit_mode}`",
        f"- Extra columns: `{columns}`",
        f"- Base LOMO: `{summary['base']['composite']:.9f}`",
        f"- Extra LOMO: `{summary['extra']['composite']:.9f}`",
        f"- Gain: `{summary['gain']:+.9f}`",
        f"- Extra LogLoss: `{summary['extra']['log_loss']:.9f}`",
        f"- Extra AUC: `{summary['extra']['auc']:.9f}`",
        f"- Full-train gamma: `{gamma_full.tolist()}`",
        "",
        "| Snapshot | Base | Extra | Gain |",
        "|---:|---:|---:|---:|",
    ]
    for row in summary["per_month"]:
        lines.append(
            f"| {row['snapshot']} | {row['base_composite']:.9f} | "
            f"{row['extra_composite']:.9f} | {row['gain']:+.9f} |"
        )
    lines.extend(
        [
            "",
            f"- Test mean: `{stats['mean']:.12f}`",
            f"- Test snapshot means: `{[round(v, 12) for v in test_snapshot_means]}`",
            f"- Submission: `{out_path}`",
        ]
    )
    report_path.with_suffix(".md").write_text("\n".join(lines) + "\n")

    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
