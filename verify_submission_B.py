from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from frozen_cv import evaluate_folds, load_folds
from v6_common import DATA_DIR, TARGET, load_data, write_json

EXPECTED_SUBMISSION_SHA256 = "5278fb77528a3701f733cef622b2c19c4df04d4321b893d1e581a98909ff4a2c"
EXPECTED_PUBLIC = {"score": 0.744694601, "log_loss": 0.223576399, "auc": 0.925374485}
EXPECTED_OOF = {"proxy": 0.7408630782405067, "log_loss": 0.22639926185594203, "auc": 0.922912137254902}


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    train, test, sample = load_data(DATA_DIR)
    submission = pd.read_csv("submission_B.csv")
    predictions = np.load("reports/submission_B_predictions.npz")
    decision = json.loads(Path("reports/final_decision_v2.json").read_text())
    candidates = json.loads(Path("reports/final_candidates.json").read_text())
    candidate = next(row for row in candidates["rows"] if row.get("submission") == "submission_B.csv")
    metrics = evaluate_folds(predictions["y"], predictions["oof"], load_folds())
    oof = metrics["overall"]
    submission_hash = sha256("submission_B.csv")
    target = submission["Target"].to_numpy(np.float64)
    saved_test = np.asarray(predictions["test"], dtype=np.float64)
    id_order_ok = bool(np.array_equal(submission["ID"].to_numpy(), sample["ID"].to_numpy()))
    prediction_match = bool(np.allclose(target, saved_test, rtol=0.0, atol=1e-15))
    public_match = all(abs(decision["public"]["B"][key] - value) < 1e-9 for key, value in EXPECTED_PUBLIC.items())
    oof_match = all(abs(oof[key] - value) < 1e-9 for key, value in EXPECTED_OOF.items())
    candidate_match = bool(
        candidate["name"] == "lgb_equal_logit"
        and candidate["members"] == ["tuned", "final_tuned"]
        and candidate["weights"] == [0.5, 0.5]
    )
    checks = {
        "submission_hash": submission_hash == EXPECTED_SUBMISSION_SHA256,
        "id_order": id_order_ok,
        "prediction_match": prediction_match,
        "public_mapping": public_match,
        "candidate_mapping": candidate_match,
        "oof_metrics": oof_match,
        "row_count": len(submission) == len(test) == 30000,
        "target_column": list(submission.columns) == ["ID", "Target"],
    }
    payload = {
        "ok": all(checks.values()),
        "checks": checks,
        "submission_sha256": submission_hash,
        "raw_sha256": {
            "train": sha256(Path(DATA_DIR) / "Train.csv"),
            "test": sha256(Path(DATA_DIR) / "Test.csv"),
            "sample": sha256(Path(DATA_DIR) / "SampleSubmission.csv"),
            "folds": sha256("reports/folds_v2.npz"),
        },
        "prediction_max_abs_diff": float(np.max(np.abs(target - saved_test))),
        "submission_stats": {"min": float(target.min()), "mean": float(target.mean()), "max": float(target.max())},
        "public": decision["public"]["B"],
        "oof": oof,
        "gate_proxy": float(oof["proxy"] + 0.001),
        "selected": decision["selected"],
        "candidate": {"name": candidate["name"], "members": candidate["members"], "weights": candidate["weights"]},
    }
    write_json("reports/baseline_verification.json", payload)
    print(json.dumps({"ok": payload["ok"], "gate_proxy": payload["gate_proxy"], "oof": oof}, indent=2, sort_keys=True))
    if not payload["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
