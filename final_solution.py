from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PY = ".venv/bin/python"
PRED = "reports/baseline_cv_drop_m5_all_s50_predictions.npz"

STAGES = {
    "manifest": [
        PY, "pipeline.py", "--stage", "manifest",
    ],
    "audit": [
        PY, "step1_analysis.py",
    ],
    "baseline": [
        PY, "pipeline.py", "--stage", "baseline",
    ],
    "features": [
        PY, "pipeline.py", "--stage", "features",
    ],
    "features_test": [
        PY, "pipeline.py", "--stage", "features_test",
    ],
    "features_preds": [
        PY, "pipeline.py", "--stage", "features_preds",
    ],
    "tune": [
        PY, "pipeline.py", "--stage", "tune",
    ],
    "final_optuna": [
        PY, "final_optuna.py", "--trials", "100", "--search-seeds", "3",
        "--final-seeds", "10", "--threads", "4", "--tag", "final_tuned_lgb100",
    ],
    "models_cv": [
        PY, "models_cv.py", "--kinds", "xgb,cat", "--seeds", "5",
        "--threads", "4", "--folds", "all", "--test",
    ],
    "finalize": [
        PY, "finalize_candidates.py",
        "--pred", f"baseline={PRED}",
        "--pred", "trend_conc=reports/feat_trend_conc_s50_predictions.npz",
        "--pred", "tuned=reports/tuned_trend_conc_s10_predictions.npz",
        "--pred", "final_tuned=reports/final_tuned_lgb100_predictions.npz",
        "--pred", "cat=reports/final_cat_s5_all_predictions.npz",
        "--pred", "xgb=reports/final_xgb_s5_all_predictions.npz",
        "--baseline", "baseline",
        "--b-candidate", "lgb_equal_logit",
        "--report", "reports/final_candidates.json",
        "--out-prefix", "submission",
    ],
    "signals_struct": [
        PY, "v7_structural.py", "--variant", "rank3_10",
    ],
    "signals_v8": [
        PY, "v8_screen.py", "--variant", "base", "--seeds", "50", "--threads", "4",
        "--test", "--init-preds", "reports/v7_struct_rank3_10_predictions.npz",
        "--tag", "base_init_rank3_s50",
        "--preds", "reports/v8_base_init_rank3_s50_predictions.npz",
    ],
    "diagnostics": [
        PY, "calibration_lab.py",
    ],
    "calibration": [
        PY, "calibration_suite.py",
    ],
    "blends": [
        PY, "blend_suite.py",
    ],
    "selection": [
        PY, "final_selection_v2.py",
    ],
    "finalize_v2": [
        PY, "finalize_v2.py",
    ],
    "verify_baseline": [
        PY, "verify_submission_B.py",
    ],
}

ORDER = [
    "manifest", "audit", "baseline", "features", "features_test",
    "features_preds", "tune", "final_optuna", "models_cv", "signals_struct",
    "signals_v8", "finalize", "diagnostics", "calibration", "blends",
    "selection", "finalize_v2", "verify_baseline", "report",
]


def fmt(value, digits=9):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_report() -> None:
    from finalize_v2 import write_report as write_v2_report
    write_v2_report()
    return
    data = json.loads(Path("reports/final_candidates.json").read_text())
    rows = sorted(data["rows"], key=lambda r: r["proxy"], reverse=True)
    a = next(r for r in rows if r["name"] == data["submission_A"])
    b = next(r for r in rows if r["name"] == data["submission_B"])
    base = next(r for r in rows if r["name"] == "baseline")
    baseline_public = 0.744671204
    rank1_public = 0.744833171
    a_public = baseline_public + (a["proxy"] - base["proxy"])
    b_public = baseline_public + (b["proxy"] - base["proxy"])

    lines = [
        "# Final report",
        "",
        "## Outcome",
        "",
        f"- Submission A: `{a['submission']}` — `{a['name']}`, frozen-CV proxy `{fmt(a['proxy'])}`.",
        f"- Submission B: `{b['submission']}` — `{b['name']}`, frozen-CV proxy `{fmt(b['proxy'])}`.",
        "- A is the best-CV candidate by paired repeated-CV score. B is the raw hedge with the best CV score among non-rank-calibrated multi-seed blends.",
        "- No rank-1 claim is made: neither final A nor B has been scored publicly.",
        "",
        "## Rules and data",
        "",
        "- The Zindi competition/rules URLs returned the SPA text `404 not found`; exact team, submission, and external-data restrictions could not be read. This solution uses only the provided `Train.csv`, `Test.csv`, and `SampleSubmission.csv`.",
        "- Train `(40000, 184)`, test `(30000, 183)`, target `liquidity_stress_next_30d`, 15% positives per snapshot.",
        "- Same 10,000 customers appear in train (4 snapshots) and test (3 later snapshots); each snapshot has exactly 1,500 positives.",
        "- No missing values, duplicate rows/IDs, or constant columns; clean RF adversarial validation AUC `0.5000837858333333`.",
        "- Metric: `score = 0.6 * (1 - LogLoss / 0.595) + 0.4 * ROC_AUC`.",
        "",
        "## Validation protocol",
        "",
        "- Frozen 5-fold x 3-seed stratified CV: seeds `42/2026/7`, manifest in `reports/frozen_cv_manifest.json`, folds in `reports/folds_v2.npz`.",
        "- Selection required paired delta greater than one standard error over the 15 fold/repeat scores.",
        "- Plain in-sample isotonic/temperature calibration was rejected after nested CV gains failed to transfer publicly (`0.739335283` public for a CV-selected calibrated blend).",
        "- Within-date rank isotonic calibration was retained because it is nested in repeated CV, preserves per-snapshot 15% priors, and avoids the extreme clipping that hurt plain isotonic.",
        "",
        "## Final candidate table",
        "",
        "| Candidate | Family | Members | Weights | OOF LogLoss | OOF AUC | Proxy | Fold std | Delta vs baseline | ± SE | Accept |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        weights = "—" if r["weights"] is None else ", ".join(f"{w:.3f}" for w in r["weights"])
        lines.append(
            f"| `{r['name']}` | {r['family']} | {', '.join(r['members'])} | {weights} | "
            f"{fmt(r['log_loss'])} | {fmt(r['auc'])} | {fmt(r['proxy'])} | {fmt(r['fold_std'])} | "
            f"{fmt(r['delta_vs_baseline'])} | {fmt(r['delta_se'])} | {r['delta_accept']} |"
        )

    lines += [
        "",
        "## Ranked CV gains vs baseline",
        "",
        "| Rank | Change | Proxy gain | Decision/use |",
        "|---:|---|---:|---|",
        f"| 1 | Within-date rank isotonic on all-model logit blend | +{a['delta_vs_baseline']:.9f} | A; public risk remains untested |",
        "| 2 | Within-date rank isotonic on tuned + final-tuned equal logit blend | +0.000540516 | close runner-up |",
        "| 3 | 100-trial Optuna tuned LightGBM | +0.000219020 | accepted raw model component |",
        "| 4 | Combine trend + concentration features | +0.000139840 | accepted feature group |",
        "| 5 | Trend features only | +0.000103370 | accepted feature group |",
        "| 6 | Concentration features only | +0.000012970 | weak accepted feature group |",
        "| 7 | 50-seed baseline variance reduction | +0.000003020 | accepted variance reduction |",
        "| 8 | Liquidity ratio features | -0.000029720 | rejected |",
        "| 9 | Stress features | -0.000002450 | rejected |",
        "",
        "New feature groups `cross_ratio`, `volatility`, `month_shape`, and `profile_num` were rejected in one-repeat ablations; every addition was below noise or negative. Structural probit weight fitting, rank transforms of latent inputs, and fitted sigma were also rejected; fixed sigma `0.35` remained best.",
        "",
        "## Public-score estimate",
        "",
        f"- Known public baseline: `{baseline_public:.9f}`; rank 1: `{rank1_public:.9f}`; gap: `{rank1_public - baseline_public:.9f}`.",
        f"- Mechanical CV-to-public transfer for A would give `{a_public:.9f}`; for B `{b_public:.9f}`.",
        "- This transfer is not reliable: raw CV gains of roughly `+0.0002` did not appear on the public leaderboard, and plain isotonic calibration collapsed publicly despite strong nested CV.",
        "- Honest expectation: B is likely around the existing public plateau `0.74465–0.7448`; A has higher upside (`0.7448–0.7453` if the rank map transfers) and meaningful downside if it behaves like plain calibration. Submit A first if one public attempt is available; submit B as the conservative hedge.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "python -m venv .venv",
        "source .venv/bin/activate",
        "pip install -r requirements.txt",
        "python final_solution.py --stage all",
        "```",
        "",
    ]
    Path("reports/FINAL_REPORT.md").write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=[*ORDER, "all"], default="report")
    args = ap.parse_args()
    stages = ORDER if args.stage == "all" else [args.stage]
    for stage in stages:
        if stage == "report":
            write_report()
            print("+ report: reports/FINAL_REPORT.md", flush=True)
            continue
        cmd = STAGES[stage]
        print(f"+ {stage}: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    sys.exit(main())
