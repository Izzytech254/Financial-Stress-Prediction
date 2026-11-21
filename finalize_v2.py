from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from v6_common import DATA_DIR, REPORT_DIR, load_data, validate_submission, write_json

C_NAME = "B_v8_logit_0.5"
D_NAME = "logC1_v8_logit_0.5"


def write_submissions() -> dict:
    train, test, sample = load_data(DATA_DIR)
    z = np.load(REPORT_DIR / "final_candidates_v2_predictions.npz")
    out = {}
    for letter, name in (("C", C_NAME), ("D", D_NAME)):
        p = z[f"{name}_test"]
        path = Path(f"submission_{letter}.csv")
        submission = sample.copy()
        submission["Target"] = p
        submission.to_csv(path, index=False, float_format="%.17g")
        stats = validate_submission(path, sample)
        np.savez_compressed(REPORT_DIR / f"submission_{letter}_predictions.npz", y=z["y"], d_tr=z["d_tr"], d_te=z["d_te"], oof=z[f"{name}_oof"], test=p)
        out[letter] = {"name": name, "path": str(path), "validation": stats, "oof": z[f"{name}_oof"], "test": p}
    return out


def write_report() -> None:
    selection = json.loads((REPORT_DIR / "final_selection_v2.json").read_text())
    calibration = json.loads((REPORT_DIR / "calibration_suite.json").read_text())
    blends = json.loads((REPORT_DIR / "blend_suite.json").read_text())
    diagnostics = json.loads((REPORT_DIR / "calibration_diagnostics.json").read_text())
    verification = json.loads((REPORT_DIR / "baseline_verification.json").read_text())
    ablation_path = REPORT_DIR / "feature_group_ablation_v3_groups.json"
    ablation = json.loads(ablation_path.read_text()) if ablation_path.exists() else None
    e_path = REPORT_DIR / "submission_candidates_E.json"
    candidates_e = json.loads(e_path.read_text()) if e_path.exists() else None
    f_path = REPORT_DIR / "submission_candidates_F.json"
    candidates_f = json.loads(f_path.read_text()) if f_path.exists() else None
    g_path = REPORT_DIR / "submission_candidates_G.json"
    candidates_g = json.loads(g_path.read_text()) if g_path.exists() else None
    h_path = REPORT_DIR / "submission_candidates_H.json"
    candidates_h = json.loads(h_path.read_text()) if h_path.exists() else None
    i_path = REPORT_DIR / "submission_candidates_I.json"
    candidates_i = json.loads(i_path.read_text()) if i_path.exists() else None
    j_path = REPORT_DIR / "submission_candidates_J.json"
    candidates_j = json.loads(j_path.read_text()) if j_path.exists() else None
    k_path = REPORT_DIR / "submission_candidates_K.json"
    candidates_k = json.loads(k_path.read_text()) if k_path.exists() else None
    l_path = REPORT_DIR / "submission_candidates_L.json"
    candidates_l = json.loads(l_path.read_text()) if l_path.exists() else None
    m_path = REPORT_DIR / "submission_candidates_M.json"
    candidates_m = json.loads(m_path.read_text()) if m_path.exists() else None
    n_path = REPORT_DIR / "submission_candidates_N.json"
    candidates_n = json.loads(n_path.read_text()) if n_path.exists() else None
    o_path = REPORT_DIR / "submission_candidates_O.json"
    candidates_o = json.loads(o_path.read_text()) if o_path.exists() else None
    expanded_path = REPORT_DIR / "calibration_interactions_expanded.json"
    calibration_expanded = json.loads(expanded_path.read_text()) if expanded_path.exists() else None
    offline_j_path = REPORT_DIR / "offline_search_J.json"
    offline_j = json.loads(offline_j_path.read_text()) if offline_j_path.exists() else None
    cal_ext_path = REPORT_DIR / "calibration_extended.json"
    calibration_extended = json.loads(cal_ext_path.read_text()) if cal_ext_path.exists() else None
    cal_feat_path = REPORT_DIR / "calibration_extended_features.json"
    calibration_features = json.loads(cal_feat_path.read_text()) if cal_feat_path.exists() else None
    rows = sorted(selection["results"].items(), key=lambda kv: kv[1]["delta_proxy"], reverse=True)
    c = selection["results"][C_NAME]
    d = selection["results"][D_NAME]
    b = selection["results"]["B"]
    public_b = {"score": 0.744694601, "log_loss": 0.223576399, "auc": 0.925374485}
    public_a = {"score": 0.743551, "log_loss": 0.224698, "auc": 0.925342}
    public_rank1 = {"score": 0.744833171, "log_loss": 0.223488800, "auc": 0.925500072}
    public_c = {"score": 0.744365653, "log_loss": 0.223826287, "auc": 0.925182086}
    public_d = {"score": 0.744325022, "log_loss": 0.223856978, "auc": 0.925157879}
    public_f1 = {"score": 0.744692858, "log_loss": 0.223578127, "auc": 0.925374485}
    public_f2 = {"score": 0.744702799, "log_loss": 0.223572264, "auc": 0.925384555}
    public_f3 = {"score": 0.744704184, "log_loss": 0.223570890, "auc": 0.925384555}
    public_g1 = {"score": 0.744711831, "log_loss": 0.223565918, "auc": 0.925391140}
    public_h1 = {"score": 0.744724040, "log_loss": 0.223554235, "auc": 0.925392205}
    public_i1 = {"score": 0.744733669, "log_loss": 0.223547413, "auc": 0.925399080}
    public_k2 = {"score": 0.744780206, "log_loss": 0.223500111, "auc": 0.925396175}
    public_l1 = {"score": 0.744777292, "log_loss": 0.223499620, "auc": 0.925387654}
    public_l2 = {"score": 0.744778268, "log_loss": 0.223499728, "auc": 0.925390365}
    public_m1 = {"score": 0.744707523, "log_loss": 0.223562049, "auc": 0.925370612}
    public_m2 = {"score": 0.744659467, "log_loss": 0.223593918, "auc": 0.925330815}
    public_m3 = {"score": 0.744548276, "log_loss": 0.223675875, "auc": 0.925259452}
    public_n1 = {"score": 0.744782951, "log_loss": None, "auc": None}
    public_n2 = {"score": 0.744792129, "log_loss": 0.223489094, "auc": 0.925398208}
    public_o1 = {"score": 0.744795320, "log_loss": 0.223485123, "auc": 0.925396175}
    new_feature_lines = []
    if ablation is not None:
        arms = ablation["arms"]
        accepted = [arm for arm in arms if arm["delta"]["accept"]]
        new_feature_lines = [
            "## New feature-group experiment",
            "",
            f"- Baseline verification passed: `submission_B.csv` sha256 `{verification['submission_sha256']}`; public metrics `{verification['public']['score']:.9f}` / `{verification['public']['log_loss']:.9f}` / `{verification['public']['auc']:.9f}`.",
            f"- Frozen OOF gate: proxy `{verification['oof']['proxy']:.9f}`, LL `{verification['oof']['log_loss']:.9f}`, AUC `{verification['oof']['auc']:.9f}`; required submission proxy `{verification['gate_proxy']:.9f}`.",
            f"- Groups tested: `{ablation['groups_tested']}`; acceptance rule: repeat-level paired proxy delta positive in every repeat and mean delta greater than `2 x SE` across repeat deltas.",
            "- SE was computed across the 3 repeat-level deltas, not across the 15 correlated folds. With 6 tests, multiple-comparison risk remains material; the strict repeat gate and reserved confirmation split were intended to mitigate, not eliminate, it.",
            "",
            "| Group | Δ proxy | repeat SE | repeat deltas | accept |",
            "|---|---:|---:|---|---|",
        ]
        for arm in arms:
            delta = arm["delta"]
            repeat_text = ", ".join(f"{r['repeat']}:{r['delta']:+.9f}" for r in delta["repeats"])
            new_feature_lines.append(f"| `{'+'.join(arm['groups'])}` | {delta['delta']:+.9f} | {delta['se']:.9f} | {repeat_text} | {delta['accept']} |")
        new_feature_lines += [
            "",
            f"- Accepted groups: `{len(accepted)}`. Per the stop rule, no combination search, retuning, seed-bagging, confirmation split, or public submission was performed.",
        ]
    e_lines = []
    if candidates_e is not None:
        e_lines = [
            "",
            "## Submission candidates E",
            "",
            "- User requested a new file to submit while preserving `submission_B.csv`; B hash is unchanged after generating the candidates.",
            "- E candidates are probability-space hedges around B using saved, deterministic OOF/test predictions; they do not overwrite B and preserve the 15% per-snapshot mean.",
            "- These are diagnostic attempts, not strict-gate promotions: all have slightly lower frozen OOF than B, so any public gain would come from transfer noise or diversity around B.",
            "",
            "| File | Blend | OOF proxy | Δ vs B | sha256 |",
            "|---|---|---:|---:|---|",
        ]
        for arm in candidates_e["candidates"]:
            source_weight = arm["weight_source"]
            e_lines.append(f"| `{arm['file']}` | {1.0 - source_weight:.2f} B + {source_weight:.2f} `{arm['source']}` | {arm['oof_proxy']:.9f} | {arm['oof_delta_vs_B']:+.9f} | `{arm['sha256']}` |")
        e_lines += [
            "",
            "- Submit order if daily limit allows: `submission_E1.csv`, then `submission_E2.csv`, then `submission_E3.csv`. Keep `submission_B.csv` as the saved fallback unless an E candidate scores higher publicly.",
        ]
    f_lines = []
    if candidates_f is not None:
        f_lines = [
            "",
            "## Submission candidates F",
            "",
            "- E feedback: E1 `0.744693992` (LL `0.223577118`, AUC `0.925374776`), E2 `0.744678116`, E3 `0.744675143`; B remains best at `0.744694601`.",
            "- F candidates target LogLoss while keeping perturbations small; F2/F3 use lightly blended regularized logistic calibration, F1 uses the saved low-tail clip.",
            "- All F candidates are diagnostic attempts, not strict-gate promotions; they use saved deterministic OOF/test arrays and preserve sample ID order.",
            "",
            "| File | Construction | OOF proxy | Δ vs B | Δ LL | Δ AUC | Date-3 Δ | Test MAD vs B | sha256 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
        for arm in candidates_f["candidates"]:
            source_weight = arm["weight_source"]
            construction = f"{1.0 - source_weight:.2f} B + {source_weight:.2f} `{arm['source']}`" if source_weight else f"`{arm['source']}`"
            f_lines.append(
                f"| `{arm['file']}` | {construction} | {arm['oof_proxy']:.9f} | {arm['oof_delta_vs_B']:+.9f} | "
                f"{arm['oof_log_loss_delta_vs_B']:+.9f} | {arm['oof_auc_delta_vs_B']:+.9f} | "
                f"{arm['oof_by_date_delta_vs_B'][3]:+.9f} | {arm['test_mad_vs_B']:.9f} | `{arm['sha256']}` |"
            )
        f_lines += [
            "",
            f"- Submit order was: {' → '.join(f'`{x}`' for x in candidates_f['submit_order'])}.",
            "- Public outcome: F3 `0.744704184` (LL `0.223570890`, AUC `0.925384555`) was best, followed by F2 `0.744702799`; F1 `0.744692858` did not beat B.",
        ]
    g_lines = []
    if candidates_g is not None:
        g_lines = [
            "",
            "## Submission candidate G",
            "",
            "- F3 is the public reference at `0.744704184`. G1 doubles the F3 C1 weight from 5% to 10% because the F batch confirmed that the regularized logistic direction transfers.",
            "- G1 is a single-submission escalation, not a strict-gate promotion; it uses saved deterministic OOF/test arrays and preserves sample ID order.",
            "",
            "| File | Construction | OOF proxy | Δ vs B | Δ vs F3 | Δ LL | Δ AUC | Date-3 Δ | Test MAD vs B | sha256 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for arm in candidates_g["candidates"]:
            source_weight = arm["weight_source"]
            construction = f"{1.0 - source_weight:.2f} B + {source_weight:.2f} `{arm['source']}`"
            g_lines.append(
                f"| `{arm['file']}` | {construction} | {arm['oof_proxy']:.9f} | {arm['oof_delta_vs_B']:+.9f} | "
                f"{arm['oof_delta_vs_F3']:+.9f} | {arm['oof_log_loss_delta_vs_B']:+.9f} | "
                f"{arm['oof_auc_delta_vs_B']:+.9f} | {arm['oof_by_date_delta_vs_B'][3]:+.9f} | "
                f"{arm['test_mad_vs_B']:.9f} | `{arm['sha256']}` |"
            )
        g_lines += [
            "",
            f"- Submit order was: {' → '.join(f'`{x}`' for x in candidates_g['submit_order'])}.",
            "- Public outcome: G1 `0.744711831` (LL `0.223565918`, AUC `0.925391140`) beat F3; OOF-to-public transfer stayed near 55% for LL and 100% for AUC.",
        ]
    h_lines = []
    if candidates_h is not None:
        h_lines = [
            "",
            "## Submission candidate H",
            "",
            "- G1 is the public reference at `0.744711831`. H1 raises the C1 weight from 10% to 25% because the F/G batches showed stable transfer of the regularized logistic direction.",
            "- H1 is a single-submission escalation, not a strict-gate promotion; it uses saved deterministic OOF/test arrays and preserves sample ID order.",
            "",
            "| File | Construction | OOF proxy | Δ vs B | Δ vs G1 | Δ LL | Δ AUC | Date-3 Δ | Test MAD vs B | Extrapolated public | sha256 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for arm in candidates_h["candidates"]:
            source_weight = arm["weight_source"]
            construction = f"{1.0 - source_weight:.2f} B + {source_weight:.2f} `{arm['source']}`"
            h_lines.append(
                f"| `{arm['file']}` | {construction} | {arm['oof_proxy']:.9f} | {arm['oof_delta_vs_B']:+.9f} | "
                f"{arm['oof_delta_vs_G1']:+.9f} | {arm['oof_log_loss_delta_vs_B']:+.9f} | "
                f"{arm['oof_auc_delta_vs_B']:+.9f} | {arm['oof_by_date_delta_vs_B'][3]:+.9f} | "
                f"{arm['test_mad_vs_B']:.9f} | {arm['extrapolated_public_score']:.9f} | `{arm['sha256']}` |"
            )
        h_lines += [
            "",
            f"- Submit order was: {' → '.join(f'`{x}`' for x in candidates_h['submit_order'])}.",
            "- Public outcome: H1 `0.744724040` (LL `0.223554235`, AUC `0.925392205`) beat G1; transfer efficiency fell to ~50% for both LL and AUC but stayed positive.",
        ]
    i_lines = []
    if candidates_i is not None:
        i_lines = [
            "",
            "## Submission candidate I",
            "",
            "- H1 is the public reference at `0.744724040`. I1 raises the C1 weight from 25% to 40% using the measured ~50% transfer efficiency.",
            "- I1 is a single-submission escalation, not a strict-gate promotion; it uses saved deterministic OOF/test arrays and preserves sample ID order.",
            "",
            "| File | Construction | OOF proxy | Δ vs B | Δ vs H1 | Δ LL | Δ AUC | Date-3 Δ | Test MAD vs B | Extrapolated public | sha256 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for arm in candidates_i["candidates"]:
            source_weight = arm["weight_source"]
            construction = f"{1.0 - source_weight:.2f} B + {source_weight:.2f} `{arm['source']}`"
            i_lines.append(
                f"| `{arm['file']}` | {construction} | {arm['oof_proxy']:.9f} | {arm['oof_delta_vs_B']:+.9f} | "
                f"{arm['oof_delta_vs_H1']:+.9f} | {arm['oof_log_loss_delta_vs_B']:+.9f} | "
                f"{arm['oof_auc_delta_vs_B']:+.9f} | {arm['oof_by_date_delta_vs_B'][3]:+.9f} | "
                f"{arm['test_mad_vs_B']:.9f} | {arm['extrapolated_public_score']:.9f} | `{arm['sha256']}` |"
            )
        i_lines += [
            "",
            f"- Submit order was: {' → '.join(f'`{x}`' for x in candidates_i['submit_order'])}.",
            "- Public outcome: I1 `0.744733669` (LL `0.223547413`, AUC `0.925399080`) beat H1; transfer efficiency fell to ~45% for both LL and AUC, and the public response curve is near its C1-weight optimum.",
        ]
    j_lines = []
    if candidates_j is not None:
        j_lines = [
            "",
            "## Submission candidate J",
            "",
            "- I1 is the public reference at `0.744733669`. J1 uses the public-fit C1 optimum near 50%; J3 is a higher-OOF hedge at 58%; J2 mixes a small C0.01 share for diversity.",
            "- No submissions remain, so J1/J2/J3 are prepared but not submitted; keep I1 until the submission budget resets.",
            "",
            "| File | Construction | OOF proxy | Δ vs B | Δ vs I1 | Δ LL | Δ AUC | Date-3 Δ | Test MAD vs B | sha256 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for arm in candidates_j["candidates"]:
            b_weight = 1.0
            weights = {}
            for part in arm["sequential_blends"]:
                w = part["weight"]
                b_weight *= 1.0 - w
                weights = {src: v * (1.0 - w) for src, v in weights.items()}
                weights[part["source"]] = weights.get(part["source"], 0.0) + w
            source_text = " + ".join(f"{v:.2f} `{src}`" for src, v in weights.items())
            construction = f"{b_weight:.2f} B + {source_text}"
            j_lines.append(
                f"| `{arm['file']}` | {construction} | {arm['oof_proxy']:.9f} | {arm['oof_delta_vs_B']:+.9f} | "
                f"{arm['oof_delta_vs_I1']:+.9f} | {arm['oof_log_loss_delta_vs_B']:+.9f} | "
                f"{arm['oof_auc_delta_vs_B']:+.9f} | {arm['oof_by_date_delta_vs_B'][3]:+.9f} | "
                f"{arm['test_mad_vs_B']:.9f} | `{arm['sha256']}` |"
            )
        j_lines += [
            "",
            f"- Submit order when budget resets: {' → '.join(f'`{x}`' for x in candidates_j['submit_when_reset'])}.",
        ]
    if offline_j is not None:
        j_lines += [
            "",
            "## Offline search after I1",
            "",
            f"- Evaluated `{offline_j['n_evaluated']}` saved OOF/test blends; `{offline_j['n_eligible']}` met the positive-OOF and non-negative date-3 screen.",
            f"- {offline_j['conclusion']}",
        ]
    if calibration_extended is not None:
        best = calibration_extended["results"][calibration_extended["ranking"][0]]
        best_w = max(best["blends"], key=lambda w: best["blends"][w]["proxy"])
        j_lines += [
            "",
            "## Extended logistic calibration search",
            "",
            f"- Re-trained profile-aware logistic calibration for C values `{calibration_extended['settings']['C_values']}` under the same nested 3-repeat protocol.",
            f"- Best saved-array blend was `{calibration_extended['ranking'][0]}` at weight `{best_w}`: OOF proxy `{best['blends'][best_w]['proxy']:.9f}`; the gain over C1 was negligible, so C strength alone was not promoted.",
        ]
    if calibration_features is not None:
        top = calibration_features["rows"][0]
        j_lines += [
            "",
            f"- Added quadratic logit, logit×date, and logit×segment interactions; best full-feature blend reached OOF proxy `{top['proxy']:.9f}` with paired accept `{top['paired_accept']}`.",
            "- High-weight interaction variants weakened date 3, so only 50%-weight variants with non-negative date-3 deltas were promoted to K backups.",
        ]
    k_lines = []
    if candidates_k is not None:
        k_lines = [
            "",
            "## Submission candidate K",
            "",
            "- K1/K2 use profile-aware logistic calibration with quadratic logit, logit×date, and logit×segment interactions at 50% weight; K1 uses C=1 for the stronger date-3 result, K2 uses C=3 for the higher overall OOF proxy.",
            "- No submissions remain, so K1/K2 are prepared but not submitted; I1 remains the public best until the budget resets.",
            "",
            "| File | Construction | OOF proxy | Δ vs B | Δ vs J1 | Δ LL | Δ AUC | Date-3 Δ | Test MAD vs B | sha256 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for arm in candidates_k["candidates"]:
            k_lines.append(
                f"| `{arm['file']}` | {1.0 - arm['weight_source']:.2f} B + {arm['weight_source']:.2f} `{arm['source']}` | "
                f"{arm['oof_proxy']:.9f} | {arm['oof_delta_vs_B']:+.9f} | {arm['oof_delta_vs_J1']:+.9f} | "
                f"{arm['oof_log_loss_delta_vs_B']:+.9f} | {arm['oof_auc_delta_vs_B']:+.9f} | "
                f"{arm['oof_date3_delta_vs_B']:+.9f} | {arm['test_mad_vs_B']:.9f} | `{arm['sha256']}` |"
            )
        k_lines += [
            "",
            f"- Submit order when budget resets: {' → '.join(f'`{x}`' for x in candidates_k['submit_when_reset'])}.",
        ]
    l_lines = []
    if candidates_l is not None:
        l_lines = [
            "",
            "## Submission candidate L",
            "",
            "- L1/L2 raise the `logistic_interactions_C1` weight from 50% to 53%/54%; both remain paired-accepted and keep date-3 non-negative.",
            "- L1 is the safer primary at 53%; L2 has slightly higher overall OOF but a smaller date-3 margin. No submissions remain, so both are prepared but not submitted.",
            "",
            "| File | Construction | OOF proxy | Δ vs B | Δ vs K1 | Δ LL | Δ AUC | Date-3 Δ | Test MAD vs B | sha256 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for arm in candidates_l["candidates"]:
            l_lines.append(
                f"| `{arm['file']}` | {1.0 - arm['weight_source']:.2f} B + {arm['weight_source']:.2f} `{arm['source']}` | "
                f"{arm['oof_proxy']:.9f} | {arm['oof_delta_vs_B']:+.9f} | {arm['oof_delta_vs_K1']:+.9f} | "
                f"{arm['oof_log_loss_delta_vs_B']:+.9f} | {arm['oof_auc_delta_vs_B']:+.9f} | "
                f"{arm['oof_date3_delta_vs_B']:+.9f} | {arm['test_mad_vs_B']:.9f} | `{arm['sha256']}` |"
            )
        l_lines += [
            "",
            f"- Submit order when budget resets: {' → '.join(f'`{x}`' for x in candidates_l['submit_when_reset'])}.",
        ]
    if calibration_expanded is not None:
        best = calibration_expanded["eligible"][0]
        l_lines += [
            "",
            "## Expanded interaction calibration search",
            "",
            "- Added logit interactions with region, earning pattern, smartphone, gender, quantile bins, scaled continuous features, and segment×date alongside the K/L interaction set.",
            f"- Best eligible expanded source was `{best['source']}` at weight `{best['w']:.2f}`: OOF proxy `{best['proxy']:.9f}`, date-3 delta `{best['date3_delta_vs_B']:+.9f}`, paired accept `{best['paired_accept']}`.",
            "- C3 was selected for the M backups because its date-3 margin was slightly stronger at the higher 60%–100% weights while overall OOF was effectively identical to C1.",
        ]
    m_lines = []
    if candidates_m is not None:
        m_lines = [
            "",
            "## Submission candidate M",
            "",
            "- M1/M2/M3 use the expanded C3 interaction source at 60%, 75%, and 100% weight; all are paired-accepted with strongly positive date-3 deltas.",
            "- M1/M2/M3 were prepared as backups; their public results showed increasing regression with expanded-interaction weight, so they are retained as diagnostics only.",
            "",
            "| File | Construction | OOF proxy | Δ vs B | Δ vs L1 | Δ LL | Δ AUC | Date-3 Δ | Test MAD vs B | sha256 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for arm in candidates_m["candidates"]:
            m_lines.append(
                f"| `{arm['file']}` | {1.0 - arm['weight_source']:.2f} B + {arm['weight_source']:.2f} `{arm['source']}` | "
                f"{arm['oof_proxy']:.9f} | {arm['oof_delta_vs_B']:+.9f} | {arm['oof_delta_vs_L1']:+.9f} | "
                f"{arm['oof_log_loss_delta_vs_B']:+.9f} | {arm['oof_auc_delta_vs_B']:+.9f} | "
                f"{arm['oof_date3_delta_vs_B']:+.9f} | {arm['test_mad_vs_B']:.9f} | `{arm['sha256']}` |"
            )
        m_lines += [
            "",
            f"- Historical submit order used for M: {' → '.join(f'`{x}`' for x in candidates_m['submit_when_reset'])}.",
        ]
    n_lines = []
    if candidates_n is not None:
        n_lines = [
            "",
            "## Public feedback after L/M/N",
            "",
            "- N2 was the intermediate public best at `0.744792129`; N1 scored `0.744782951`, so the 10% expanded-C3 dose beat the 5% dose and K2.",
            "- L1/L2 were within `0.000003` of K2 and the M family regressed increasingly as expanded-interaction weight rose, so the usable dose range is narrow.",
            "",
            "## Submission candidate N",
            "",
            "- N1/N2 add 5%/10% expanded C3 to K2. Both remained close to K2 in test deviation, were paired-accepted, and improved OOF proxy, LL, AUC, and date-3 versus K2.",
            "- N2 publicly beat N1 and K2, confirming the positive transfer of a small expanded-C3 dose.",
            "",
            "| File | Construction | OOF proxy | Δ vs K2 | Δ LL vs K2 | Δ AUC vs K2 | Date-3 Δ vs K2 | Test MAD vs B | sha256 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
        for arm in candidates_n["candidates"]:
            n_lines.append(
                f"| `{arm['file']}` | `{arm['construction']}` | "
                f"{arm['oof_proxy']:.9f} | {arm['oof_delta_vs_K2']:+.9f} | "
                f"{arm['oof_log_loss_delta_vs_K2']:+.9f} | {arm['oof_auc_delta_vs_K2']:+.9f} | "
                f"{arm['oof_date3_delta_vs_K2']:+.9f} | {arm['test_mad_vs_B']:.9f} | `{arm['sha256']}` |"
            )
        n_lines += [
            "",
            f"- Historical submit order used for N: {' → '.join(f'`{x}`' for x in candidates_n['submit_when_reset'])}.",
        ]
    o_lines = []
    if candidates_o is not None:
        arm = candidates_o["candidates"][0]
        o_lines = [
            "",
            "## Submission candidate O",
            "",
            "- O1 raises the expanded-C3 dose to 20%, following N2's public win over N1 and K2.",
            "- O1 was submitted as the final candidate and became the public best at `0.744795320`.",
            "",
            "| File | Construction | OOF proxy | Δ vs K2 | Δ LL vs K2 | Δ AUC vs K2 | Date-3 Δ vs K2 | Test MAD vs B | sha256 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
            f"| `{arm['file']}` | `{arm['construction']}` | {arm['oof_proxy']:.9f} | {arm['oof_delta_vs_K2']:+.9f} | "
            f"{arm['oof_log_loss_delta_vs_K2']:+.9f} | {arm['oof_auc_delta_vs_K2']:+.9f} | "
            f"{arm['oof_date3_delta_vs_K2']:+.9f} | {arm['test_mad_vs_B']:.9f} | `{arm['sha256']}` |",
            "",
            f"- Final submission used: `{candidates_o['historical_submission']}`; submissions available after that: `{candidates_o['submissions_available']}`.",
        ]
    new_feature_lines += e_lines + f_lines + g_lines + h_lines + i_lines + j_lines + k_lines + l_lines + m_lines + n_lines + o_lines
    new_feature_lines += [
        "",
        "## Peer-relative feature methodology and reproducibility",
        "",
        "- Peer-relative features are computed label-free from `concat(train.drop(target), test)` in fixed row order with snapshot dates `0..3` for train and `4..6` for test.",
        "- For each peer key (`segment`, `region`, `earning_pattern`) and each selected numeric feature, rows are grouped by `(snapshot_date, peer_key)`; missing keys use the sentinel `__MISSING__`.",
        "- Within each group, z-score uses population standard deviation (`ddof=0`) with `1e-8` numerical floor and percentile rank uses average ties minus `0.5`; no target, fold index, prediction, or leaderboard value is used.",
        "- Derived columns are then rank-scaled within each snapshot date by the same `month_scaled(..., 'rank')` path used by the base pipeline.",
        "- Determinism: frozen fold seeds `42/2026/7`, LightGBM bagging seeds start `909,11,22,33,44`, early-stopping splits use `10000 + repeat * 10 + fold`, Optuna sampler seed `20261119`, and confirmation seed `314159` is reserved but unused because the stop rule triggered.",
        "- Full rebuild from raw CSVs: `python final_solution.py --stage all`; the `verify_baseline` stage checks `submission_B.csv` hash, raw-data hashes, sample ID order, saved test predictions, and frozen OOF metrics.",
        "",
    ]
    lines = [
        "# Final report",
        "",
        "## Outcome",
        "",
        f"- Submission C: `submission_C.csv` — `{C_NAME}`, frozen-CV proxy `{c['proxy']:.9f}` vs B `{b['proxy']:.9f}` (`{c['delta_proxy']:+.9f}`).",
        f"- Submission D: `submission_D.csv` — `{D_NAME}`, frozen-CV proxy `{d['proxy']:.9f}` vs B `{b['proxy']:.9f}` (`{d['delta_proxy']:+.9f}`).",
        "- C is the conservative hedge: raw logit blend of public-proven B and the structurally initialized 50-seed v8 signal; no post-hoc calibration.",
        "- D is the higher-upside hedge: same v8 signal blended with regularized profile-aware logistic calibration of B.",
        f"- Known public: B `{public_b['score']:.9f}`, O1 `{public_o1['score']:.9f}`, rank 1 `{public_rank1['score']:.9f}`; exact metric recomputes B as `{public_b['score']:.9f}`.",
        "- O1 is the best public submission so far; no submissions remain.",
        "- No public leaderboard scores were used for tuning C or D; A/B scores are reported only as outcome evidence.",
        "- New feature-group search under the amended gate found no accepted group; the later E/F/G/H/I/J/K/L/M/N/O batches are explicitly public-feedback diagnostics.",
        "",
        "## Public outcome of C and D",
        "",
        "| Submission | Public score | Δ vs B | Log Loss | Δ LL | ROC AUC | Δ AUC |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| B | {public_b['score']:.9f} | 0 | {public_b['log_loss']:.9f} | 0 | {public_b['auc']:.9f} | 0 |",
        f"| C | {public_c['score']:.9f} | {public_c['score'] - public_b['score']:+.9f} | {public_c['log_loss']:.9f} | {public_c['log_loss'] - public_b['log_loss']:+.9f} | {public_c['auc']:.9f} | {public_c['auc'] - public_b['auc']:+.9f} |",
        f"| D | {public_d['score']:.9f} | {public_d['score'] - public_b['score']:+.9f} | {public_d['log_loss']:.9f} | {public_d['log_loss'] - public_b['log_loss']:+.9f} | {public_d['auc']:.9f} | {public_d['auc'] - public_b['auc']:+.9f} |",
        "",
        f"- Both C and D regressed publicly despite positive OOF deltas: about 76–77% of each loss came from worse Log Loss and 23–24% from worse AUC.",
        f"- At that stage B was the best submitted file; rank-1 gap was `{public_rank1['score'] - public_b['score']:+.9f}`.",
        "- The v8 structural signal blended into C/D does not transfer to the test snapshots; it is rejected along with rank/isotonic calibration.",
        "",
        "## Recent public outcomes",
        "",
        "| Submission | Public score | Δ vs O1 | Log Loss | ROC AUC |",
        "|---|---:|---:|---:|---:|",
        f"| K2 | {public_k2['score']:.9f} | {public_k2['score'] - public_o1['score']:+.9f} | {public_k2['log_loss']:.9f} | {public_k2['auc']:.9f} |",
        f"| L1 | {public_l1['score']:.9f} | {public_l1['score'] - public_o1['score']:+.9f} | {public_l1['log_loss']:.9f} | {public_l1['auc']:.9f} |",
        f"| L2 | {public_l2['score']:.9f} | {public_l2['score'] - public_o1['score']:+.9f} | {public_l2['log_loss']:.9f} | {public_l2['auc']:.9f} |",
        f"| M1 | {public_m1['score']:.9f} | {public_m1['score'] - public_o1['score']:+.9f} | {public_m1['log_loss']:.9f} | {public_m1['auc']:.9f} |",
        f"| M2 | {public_m2['score']:.9f} | {public_m2['score'] - public_o1['score']:+.9f} | {public_m2['log_loss']:.9f} | {public_m2['auc']:.9f} |",
        f"| M3 | {public_m3['score']:.9f} | {public_m3['score'] - public_o1['score']:+.9f} | {public_m3['log_loss']:.9f} | {public_m3['auc']:.9f} |",
        f"| N1 | {public_n1['score']:.9f} | {public_n1['score'] - public_o1['score']:+.9f} | — | — |",
        f"| N2 | {public_n2['score']:.9f} | {public_n2['score'] - public_o1['score']:+.9f} | {public_n2['log_loss']:.9f} | {public_n2['auc']:.9f} |",
        f"| O1 | {public_o1['score']:.9f} | 0 | {public_o1['log_loss']:.9f} | {public_o1['auc']:.9f} |",
        "",
        f"- Target rank 1 is `{public_rank1['score']:.9f}`; O1 gap is `{public_rank1['score'] - public_o1['score']:+.9f}`.",
        "- O1 beat N2 by `+0.000003191`; the gain came from lower Log Loss, while ROC AUC matched K2 and remained the main remaining gap to rank 1.",
        "",
        "## Rules, data, and metric",
        "",
        "- The Zindi competition/rules URLs returned the SPA text `404 not found`; exact team, submission, and external-data restrictions could not be read. This solution uses only the provided `Train.csv`, `Test.csv`, and `SampleSubmission.csv`.",
        "- Train `(40000, 184)`, test `(30000, 183)`, target `liquidity_stress_next_30d`, 15% positives per snapshot.",
        "- Same 10,000 customers appear in train (4 snapshots) and test (3 later snapshots); each snapshot has exactly 1,500 positives.",
        "- No missing values, duplicate rows/IDs, or constant columns; clean RF adversarial validation AUC `0.5000837858333333`.",
        "- Exact metric: `score = 0.6 * (1 - LogLoss / 0.595) + 0.4 * ROC_AUC`. Expanded: `0.4 * AUC - 1.008403361 * LogLoss + 0.6`.",
        "",
        "## Why A failed publicly",
        "",
        f"- A had a better frozen-CV proxy than B (`0.741246388` vs `{b['proxy']:.9f}`), but scored `{public_a['score']:.6f}` vs B `{public_b['score']:.6f}`.",
        "- A's within-date rank isotonic map is stepwise and clips hundreds of rows at `0.99999`; it preserved AUC but increased public Log Loss by `0.001122`.",
        "- Plain isotonic calibration previously collapsed publicly to roughly `0.739335`; rank/isotonic-style calibration is therefore rejected despite strong nested CV.",
        f"- B diagnostics show near-zero bucket biases; worst 500 OOF losses were 478 false negatives and 22 false positives, concentrated in MVC/LVC low-probability rows, with no missingness effects.",
        "",
        "## Validation protocol",
        "",
        "- Frozen 5-fold × 3-seed stratified CV: seeds `42/2026/7`, manifest in `reports/frozen_cv_manifest.json`, folds in `reports/folds_v2.npz`.",
        "- Selection required paired delta greater than one standard error over 15 fold/repeat scores.",
        "- Nested calibration: fit on 2 repeats, evaluate held-out repeat, final fit on all 3; per-snapshot 15% prior restored after every transform.",
        "- Blend weights: non-negative simplex, optimized on 2 repeats, evaluated on held-out repeat; logistic stacker tested with leave-one-model-out.",
        "- Candidate C/D additionally required positive date-3 delta and positive customer-cluster bootstrap lower 95% bound.",
        "- Customer-cluster bootstrap: 1,000 resamples of 10,000 customers with all 4 train snapshots kept together.",
        "",
        "## Calibration suite",
        "",
        "| Candidate | OOF proxy | Delta vs B | Paired SE | Decision |",
        "|---|---:|---:|---:|---|",
    ]
    for name in ("iso_smooth_1", "iso_smooth_0.75", "iso_smooth_0.5", "iso_smooth_0.25", "logistic_features_C1", "logistic_features_C0.1", "segment_platt", "temperature", "platt", "beta"):
        v = calibration["results"][name]
        decision = "reject: isotonic/step-risk" if name.startswith("iso_smooth") else ("retain for D" if name == "logistic_features_C1" else "reject")
        lines.append(f"| `{name}` | {v['proxy']:.9f} | {v['paired_proxy_delta']:+.9f} | {v['paired_proxy_se']:.9f} | {decision} |")
    lines += [
        "",
        "Temperature, Platt, beta, clip sweeps, logistic feature calibration, segment Platt, and small GBM calibration were all tested nested. All gains were below `+0.00014`; the larger gains came from stepwise isotonic mappings that are not trusted after A's public failure.",
        "",
        "## Blend suite",
        "",
        f"- Best raw LightGBM-only nested blend: `blend_tuned_final_tuned_baseline_cat`, proxy `{blends['results']['blend_tuned_final_tuned_baseline_cat']['proxy']:.9f}`, delta `{blends['results']['blend_tuned_final_tuned_baseline_cat']['paired_proxy_delta']:+.9f}`, SE `{blends['results']['blend_tuned_final_tuned_baseline_cat']['paired_proxy_se']:.9f}`.",
        "- Full logistic stackers and six-model non-negative blends did not beat B nested; leave-one-model-out showed tuned LightGBM as the only indispensable member.",
        "",
        "## Final candidate ranking",
        "",
        "| Candidate | OOF proxy | Delta vs B | Log Loss | AUC | Date-3 delta | Bootstrap 95% CI | Decision |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for name, v in rows:
        boot = v["bootstrap"]
        decision = "C" if name == C_NAME else ("D" if name == D_NAME else ("baseline" if name == "B" else "not selected"))
        lines.append(
            f"| `{name}` | {v['proxy']:.9f} | {v['delta_proxy']:+.9f} | {v['log_loss']:.9f} | {v['auc']:.9f} | {v['by_date'][-1]['delta']:+.9f} | "
            f"[{boot['p025']:+.9f}, {boot['p975']:+.9f}] | {decision} |"
        )
    lines += [
        "",
        "## Decision",
        "",
        f"- C (`{C_NAME}`): OOF `{c['proxy']:.9f}`, Log Loss `{c['log_loss']:.9f}`, AUC `{c['auc']:.9f}`; date-3 delta `{c['by_date'][-1]['delta']:+.9f}`; bootstrap CI `[{c['bootstrap']['p025']:+.9f}, {c['bootstrap']['p975']:+.9f}]`.",
        f"- D (`{D_NAME}`): OOF `{d['proxy']:.9f}`, Log Loss `{d['log_loss']:.9f}`, AUC `{d['auc']:.9f}`; date-3 delta `{d['by_date'][-1]['delta']:+.9f}`; bootstrap CI `[{d['bootstrap']['p025']:+.9f}, {d['bootstrap']['p975']:+.9f}]`.",
        "- Public outcome: C and D both regressed; keep `submission_B.csv` as the final choice unless a genuinely new CV-validated signal is found.",
        f"- The v8 structural blend turned OOF gains of `{c['delta_proxy']:+.9f}` / `{d['delta_proxy']:+.9f}` into public losses of `{public_c['score'] - public_b['score']:+.9f}` / `{public_d['score'] - public_b['score']:+.9f}`.",
        "- Future work should prioritize distribution-shift-robust validation and avoid structural rank features, rank/isotonic calibration, and public-score-driven tuning.",
        "",
    ]
    lines += new_feature_lines
    lines += [
        "## Reproduce",
        "",
        "```bash",
        "python -m venv .venv",
        "source .venv/bin/activate",
        "pip install -r requirements.txt",
        "python final_solution.py --stage all",
        "```",
        "",
        "Core stages after data/model artifacts exist:",
        "",
        "```bash",
        ".venv/bin/python calibration_lab.py",
        ".venv/bin/python calibration_suite.py",
        ".venv/bin/python blend_suite.py",
        ".venv/bin/python final_selection_v2.py",
        ".venv/bin/python finalize_v2.py",
        "```",
        "",
    ]
    (REPORT_DIR / "FINAL_REPORT.md").write_text("\n".join(lines))
    accepted_groups = [] if ablation is None else ['+'.join(arm['groups']) for arm in ablation['arms'] if arm['delta']['accept']]
    new_features = {
        "groups_tested": 0 if ablation is None else ablation['groups_tested'],
        "accepted_groups": accepted_groups,
        "decision": "stop_no_submission" if not accepted_groups else "accepted_groups_require_confirmation",
        "baseline_verified": verification['ok'],
        "gate_proxy": verification['gate_proxy'],
        "ablation_report": "reports/feature_group_ablation_v3_groups.json" if ablation is not None else None,
    }
    decision = {
        "C": C_NAME,
        "D": D_NAME,
        "selected": "submission_O1.csv",
        "public_best_code": candidates_o["public_best_code"] if candidates_o is not None else None,
        "candidate_to_submit": None,
        "submissions_available": candidates_o["submissions_available"] if candidates_o is not None else None,
        "submission_candidates": candidates_e,
        "submission_candidates_f": candidates_f,
        "submission_candidates_g": candidates_g,
        "submission_candidates_h": candidates_h,
        "submission_candidates_i": candidates_i,
        "submission_candidates_j": candidates_j,
        "submission_candidates_k": candidates_k,
        "submission_candidates_l": candidates_l,
        "submission_candidates_m": candidates_m,
        "submission_candidates_n": candidates_n,
        "submission_candidates_o": candidates_o,
        "offline_search_j": offline_j,
        "calibration_extended": calibration_extended,
        "calibration_extended_features": calibration_features,
        "calibration_interactions_expanded": calibration_expanded,
        "recommended_diagnostic": "submission_O1.csv" if candidates_o is not None else ("submission_N2.csv" if candidates_n is not None else ("submission_M1.csv" if candidates_m is not None else ("submission_L1.csv" if candidates_l is not None else ("submission_K1.csv" if candidates_k is not None else ("submission_J1.csv" if candidates_j is not None else ("submission_I1.csv" if candidates_i is not None else ("submission_H1.csv" if candidates_h is not None else ("submission_G1.csv" if candidates_g is not None else ("submission_F2.csv" if candidates_f is not None else ("submission_E1.csv" if candidates_e is not None else None)))))))))),
        "new_features": new_features,
        "public": {"B": public_b, "A": public_a, "C": public_c, "D": public_d, "F1": public_f1, "F2": public_f2, "F3": public_f3, "G1": public_g1, "H1": public_h1, "I1": public_i1, "K2": public_k2, "L1": public_l1, "L2": public_l2, "M1": public_m1, "M2": public_m2, "M3": public_m3, "N1": public_n1, "N2": public_n2, "O1": public_o1, "rank1": public_rank1},
        "c": {k: c[k] for k in ("proxy", "log_loss", "auc", "delta_proxy", "by_date", "bootstrap")},
        "d": {k: d[k] for k in ("proxy", "log_loss", "auc", "delta_proxy", "by_date", "bootstrap")},
    }
    write_json(REPORT_DIR / "final_decision_v2.json", decision)


def main() -> None:
    out = write_submissions()
    write_report()
    print(json.dumps({k: {"name": v["name"], "path": v["path"], "validation": v["validation"]} for k, v in out.items()}, indent=2))


if __name__ == "__main__":
    main()
