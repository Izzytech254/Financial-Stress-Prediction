from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from v6_common import DATA_DIR, REPORT_DIR, composite

FOLDS_PATH = REPORT_DIR / "folds_v2.npz"
REPEATS = (42, 2026, 7)
MANIFEST_PATH = REPORT_DIR / "frozen_cv_manifest.json"


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_folds(
    path: str | Path = FOLDS_PATH, repeats: Sequence[int] = REPEATS
) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    z = np.load(path)
    out = []
    for seed in repeats:
        for fold in range(5):
            tr = z[f"tr_{seed}_{fold}"]
            va = z[f"va_{seed}_{fold}"]
            if set(tr) & set(va):
                raise ValueError(f"overlap in fold {seed}-{fold}")
            out.append((int(seed), int(fold), tr, va))
    return out


def write_manifest(
    folds_path: str | Path = FOLDS_PATH,
    manifest_path: str | Path = MANIFEST_PATH,
    data_dir: str | Path = DATA_DIR,
    repeats: Sequence[int] = REPEATS,
) -> dict:
    data_dir = Path(data_dir)
    manifest = {
        "folds_path": str(folds_path),
        "folds_sha256": file_hash(folds_path),
        "repeats": list(repeats),
        "folds_per_repeat": 5,
        "train_sha256": file_hash(data_dir / "Train.csv"),
        "test_sha256": file_hash(data_dir / "Test.csv"),
        "sample_sha256": file_hash(data_dir / "SampleSubmission.csv"),
        "metric": "0.6*(1-log_loss/0.595)+0.4*auc",
        "data_rows": {"train": 40000, "test": 30000},
    }
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def evaluate_folds(
    y: np.ndarray,
    preds: np.ndarray,
    folds: Iterable[tuple[int, int, np.ndarray, np.ndarray]],
) -> dict:
    rows = []
    for seed, fold, _, va in folds:
        score, ll, auc = composite(y[va], preds[va])
        rows.append(
            {
                "repeat": seed,
                "fold": fold,
                "n": int(len(va)),
                "proxy": score,
                "log_loss": ll,
                "auc": auc,
            }
        )
    proxy, ll, auc = composite(y, preds)
    return {
        "overall": {"proxy": proxy, "log_loss": ll, "auc": auc},
        "folds": rows,
        "mean_proxy": float(np.mean([r["proxy"] for r in rows])),
        "std_proxy": float(np.std([r["proxy"] for r in rows], ddof=1)),
        "mean_log_loss": float(np.mean([r["log_loss"] for r in rows])),
        "std_log_loss": float(np.std([r["log_loss"] for r in rows], ddof=1)),
        "mean_auc": float(np.mean([r["auc"] for r in rows])),
        "std_auc": float(np.std([r["auc"] for r in rows], ddof=1)),
    }


def paired_delta(
    base_preds: np.ndarray,
    new_preds: np.ndarray,
    y: np.ndarray,
    folds: Iterable[tuple[int, int, np.ndarray, np.ndarray]],
) -> dict:
    deltas = []
    for _, _, _, va in folds:
        s0, _, _ = composite(y[va], base_preds[va])
        s1, _, _ = composite(y[va], new_preds[va])
        deltas.append(s1 - s0)
    arr = np.asarray(deltas, dtype=np.float64)
    mean = float(arr.mean())
    se = float(arr.std(ddof=1) / np.sqrt(len(arr)))
    return {"delta": mean, "se": se, "accept": bool(mean > se), "n": len(arr)}


def paired_delta_by_repeat(
    base_preds: np.ndarray,
    new_preds: np.ndarray,
    y: np.ndarray,
    folds: Iterable[tuple[int, int, np.ndarray, np.ndarray]],
    repeats: Sequence[int] = REPEATS,
    threshold_multiplier: float = 2.0,
) -> dict:
    folds = list(folds)
    base_arr = np.asarray(base_preds)
    new_arr = np.asarray(new_preds)
    if base_arr.ndim == 2:
        if base_arr.shape != new_arr.shape or base_arr.shape[0] != len(repeats):
            raise ValueError(f"repeat prediction shape mismatch: {base_arr.shape} vs {new_arr.shape} vs {len(repeats)}")
        base_reps = base_arr
        new_reps = new_arr
    else:
        base_reps = np.repeat(base_arr[None, :], len(repeats), axis=0)
        new_reps = np.repeat(new_arr[None, :], len(repeats), axis=0)
    rows = []
    for i, repeat in enumerate(repeats):
        idx = np.concatenate([va for rep, _, _, va in folds if rep == repeat])
        base_score, base_ll, base_auc = composite(y[idx], base_reps[i][idx])
        new_score, new_ll, new_auc = composite(y[idx], new_reps[i][idx])
        rows.append(
            {
                "repeat": int(repeat),
                "base_proxy": base_score,
                "new_proxy": new_score,
                "delta": new_score - base_score,
                "delta_log_loss": new_ll - base_ll,
                "delta_auc": new_auc - base_auc,
                "n": int(len(idx)),
            }
        )
    deltas = np.asarray([r["delta"] for r in rows], dtype=np.float64)
    mean = float(deltas.mean())
    se = float(deltas.std(ddof=1) / np.sqrt(len(deltas))) if len(deltas) > 1 else 0.0
    positive_everywhere = bool(np.all(deltas > 0))
    accept = bool(positive_everywhere and mean > threshold_multiplier * se)
    return {
        "delta": mean,
        "se": se,
        "threshold_multiplier": threshold_multiplier,
        "threshold": threshold_multiplier * se,
        "positive_everywhere": positive_everywhere,
        "accept": accept,
        "n_repeats": len(rows),
        "repeats": rows,
    }


def save_predictions(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-manifest", action="store_true")
    ap.add_argument("--manifest", default=str(MANIFEST_PATH))
    ap.add_argument("--folds", default=str(FOLDS_PATH))
    args = ap.parse_args()
    if args.write_manifest:
        manifest = write_manifest(args.folds, args.manifest)
        print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
