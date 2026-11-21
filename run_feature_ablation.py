from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from frozen_cv import evaluate_folds, load_folds, paired_delta
from v6_common import REPORT_DIR

DEFAULT_GROUPS = ("liquidity", "trend", "stress", "concentration")


def run_group(group: str, args: argparse.Namespace) -> Path:
    tag = f"feat_{group}_s{args.seeds}"
    slug = tag + ("_" + args.months.replace(",", "") if args.months != "1,2,3,4,5,6" else "")
    preds = Path(args.out_dir) / f"{slug}_predictions.npz"
    if preds.exists() and not args.force:
        return preds
    cmd = [
        sys.executable, "baseline_cv.py",
        "--variant", args.variant,
        "--seeds", str(args.seeds),
        "--threads", str(args.threads),
        "--feature-groups", group,
        "--feature-months", args.months,
        "--tag", tag,
        "--report", str(Path(args.out_dir) / f"{slug}.json"),
        "--preds", str(preds),
    ]
    subprocess.run(cmd, check=True)
    return preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=str(REPORT_DIR / "baseline_cv_smoke_s5_predictions.npz"))
    ap.add_argument("--variant", default="drop_m5_all")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--groups", default=",".join(DEFAULT_GROUPS))
    ap.add_argument("--months", default="1,2,3,4,5,6")
    ap.add_argument("--out-dir", default=str(REPORT_DIR))
    ap.add_argument("--report", default=str(REPORT_DIR / "ablation_features.json"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    folds = load_folds()
    baseline = np.load(args.baseline)
    base_oof = baseline["oof"]
    y = baseline["y"]
    base_metrics = evaluate_folds(y, base_oof, folds)
    results = {"baseline": base_metrics["overall"], "experiments": []}
    print("baseline", json.dumps(base_metrics["overall"], sort_keys=True), flush=True)

    for group in [g for g in args.groups.split(",") if g]:
        preds_path = run_group(group, args)
        z = np.load(preds_path)
        oof = z["oof"]
        metrics = evaluate_folds(y, oof, folds)
        delta = paired_delta(base_oof, oof, y, folds)
        entry = {
            "group": group,
            "months": args.months,
            "proxy": metrics["overall"]["proxy"],
            "log_loss": metrics["overall"]["log_loss"],
            "auc": metrics["overall"]["auc"],
            "delta": delta["delta"],
            "se": delta["se"],
            "accept": delta["accept"],
            "preds": str(preds_path),
        }
        results["experiments"].append(entry)
        print(json.dumps(entry, sort_keys=True), flush=True)

    Path(args.report).write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
