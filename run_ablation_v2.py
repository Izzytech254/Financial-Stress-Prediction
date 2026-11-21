from __future__ import annotations
import argparse, json, time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from features_cv_v2 import build_extra_v2
from frozen_cv import evaluate_folds, load_folds, paired_delta_by_repeat, save_predictions
from v6_common import DATA_DIR, MZ_SLOPE, TARGET, init_of, load_data, month_z_score, per_date, probit_eta, risk_factors, snapshot_dates, write_json
from v6_residual import bag, early_stopping_rounds, make_params, scaled_rounds, seeds_for
from v8_screen import build_variant

BASE_GROUPS = ['trend', 'concentration']


def inner_split(y, d, seed):
    from sklearn.model_selection import StratifiedKFold
    strata = y * 4 + d
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
    fit, val = next(skf.split(np.arange(len(y)), strata))
    return fit, val


def run(groups, seeds_n=5, tag='exp', variant='drop_m5_all'):
    start = time.time()
    folds = load_folds()
    seeds = seeds_for(seeds_n)
    train, test, _ = load_data(DATA_DIR)
    d_tr, d_te = snapshot_dates(train, test)
    y = train[TARGET].to_numpy()
    n = len(train)
    all_rows = pd.concat([train.drop(columns=TARGET), test], ignore_index=True)
    d_all = np.r_[d_tr, d_te]
    R = risk_factors(all_rows)
    s_mz = month_z_score(R, d_all)
    lin = MZ_SLOPE * s_mz
    X, names, drop_cols, month_scale, extra_cols = build_variant(variant, all_rows, d_all, R, None)
    if groups:
        Xe, ne = build_extra_v2(all_rows, d_all, groups)
        X = np.c_[X, Xe].astype(np.float32)
        names = names + ne
    configs = [{}, {"num_leaves": 15, "max_depth": 4}, {"min_data_in_leaf": 100}, {"learning_rate": 0.01}, {"learning_rate": 0.05}]
    repeat_ids = sorted({f[0] for f in folds})
    rep_pos = {s: j for j, s in enumerate(repeat_ids)}
    oof_reps = np.empty((len(repeat_ids), n), dtype=np.float64)
    rounds = np.zeros((len(configs), len(folds)), dtype=np.int32)
    for i, (rep, fold, a, b) in enumerate(folds):
        fit_rel, val_rel = inner_split(y[a], d_tr[a], 10000 + rep * 10 + fold)
        fit = a[fit_rel]
        val = a[val_rel]
        init_fit = init_of(lin, d_all, fit)[0]
        init_val = init_of(lin, d_all, val)[0]
        init_a = init_of(lin, d_all, a)[0]
        init_b = init_of(lin, d_all, b)[0]
        raws = []
        for c, cfg in enumerate(configs):
            params = make_params(cfg, 8)
            best = early_stopping_rounds(X[fit], y[fit], init_fit, X[val], y[val], init_val, params, 600, 60)
            rounds[c, i] = best
            raws.append(bag(X[a], y[a], init_a, X[b], max(1, best), params, seeds))
        combined = expit(init_b + np.mean(raws, axis=0))
        oof_reps[rep_pos[rep], b] = per_date(probit_eta(combined), d_tr[b])
        if fold == 0:
            print(f'{tag} rep={rep} i={i + 1}/{len(folds)} feats={X.shape[1]}', flush=True)
    oof = np.mean(oof_reps, axis=0)
    m = evaluate_folds(y, oof, folds)
    print(f'{tag} proxy={m["overall"]["proxy"]:.9f} ll={m["overall"]["log_loss"]:.9f} auc={m["overall"]["auc"]:.9f} se={m["std_proxy"] / np.sqrt(15):.9f} time={time.time() - start:.1f}s', flush=True)
    return oof_reps, oof, m, {'groups': groups, 'n_seeds': seeds_n, 'feature_count': X.shape[1], 'time': time.time() - start, 'features': names}


def safe_tag(value):
    return ''.join(c if c.isalnum() else '_' for c in value).strip('_')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--add', default='')
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--variant', default='drop_m5_all')
    ap.add_argument('--mode', choices=['single', 'combo'], default='single')
    ap.add_argument('--tag', default='')
    ap.add_argument('--report-only', action='store_true')
    args = ap.parse_args()
    adds = [g for g in args.add.split(',') if g]
    tag = args.tag or f'{args.mode}_s{args.seeds}'
    folds = load_folds()
    repeat_ids = sorted({f[0] for f in folds})
    train, test, _ = load_data(DATA_DIR)
    d_tr, d_te = snapshot_dates(train, test)
    y = train[TARGET].to_numpy()
    if args.report_only:
        report_path = Path(f'reports/feature_group_ablation_{safe_tag(tag)}.json')
        out = json.loads(report_path.read_text())
        base_reps = np.load(f'reports/ablation_{safe_tag(tag)}_base_predictions.npz')['oof_repeats']
        for result in out['arms']:
            arm_name = '+'.join(result['groups'])
            arm_tag = f'{tag}_{safe_tag(arm_name)}'
            new_reps = np.load(f'reports/ablation_{safe_tag(arm_tag)}_predictions.npz')['oof_repeats']
            result['delta'] = paired_delta_by_repeat(base_reps, new_reps, y, folds, repeats=repeat_ids)
            print(f'{arm_name} delta={result["delta"]["delta"]:.9f} se={result["delta"]["se"]:.9f} accept={result["delta"]["accept"]}', flush=True)
            print(' repeats', [round(r['delta'], 9) for r in result['delta']['repeats']], flush=True)
        write_json(report_path, out)
        return
    base_reps, base_oof, base_m, base_info = run(BASE_GROUPS, args.seeds, f'{tag}_base', args.variant)
    save_predictions(f'reports/ablation_{safe_tag(tag)}_base_predictions.npz', {'y': y, 'oof': base_oof, 'oof_repeats': base_reps, 'd_tr': d_tr})
    results = []
    if adds:
        arms = [[g] for g in adds] if args.mode == 'single' else [adds]
        for arm in arms:
            arm_name = '+'.join(arm)
            arm_tag = f'{tag}_{safe_tag(arm_name)}'
            new_reps, new_oof, new_m, new_info = run(BASE_GROUPS + arm, args.seeds, arm_tag, args.variant)
            delta = paired_delta_by_repeat(base_reps, new_reps, y, folds, repeats=repeat_ids)
            save_predictions(f'reports/ablation_{safe_tag(arm_tag)}_predictions.npz', {'y': y, 'base_oof': base_oof, 'new_oof': new_oof, 'oof_repeats': new_reps, 'd_tr': d_tr, 'groups': np.asarray(arm)})
            print(f'{arm_name} delta={delta["delta"]:.9f} se={delta["se"]:.9f} accept={delta["accept"]}', flush=True)
            results.append({'groups': arm, 'metrics': new_m, 'delta': delta, 'info': new_info})
    out = {
        'tag': tag,
        'base_groups': BASE_GROUPS,
        'base': base_m,
        'base_info': base_info,
        'arms': results,
        'groups_tested': len(results),
        'multiple_comparison_risk': 'repeat-level 2xSE gate and a held-out confirmation split mitigate but do not eliminate false positives from multiple tests',
        'acceptance_rule': 'delta positive in every repeat and mean delta > 2x SE across repeat-level deltas',
    }
    write_json(f'reports/feature_group_ablation_{safe_tag(tag)}.json', out)
    print(f'groups_tested={len(results)}', flush=True)


if __name__ == '__main__':
    main()
