"""Validate submission.csv against SampleSubmission.csv."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sub = pd.read_csv(os.path.join(ROOT, "submission.csv"))
sample = pd.read_csv(os.path.join(ROOT, "data", "SampleSubmission.csv"))

assert list(sub.columns) == ["ID", "Target"], f"columns={list(sub.columns)}"
assert len(sub) == len(sample) == 30000, f"rows={len(sub)} expected=30000"
assert sub["ID"].is_unique, "duplicate IDs"
assert (sub["ID"].values == sample["ID"].values).all(), "ID order mismatch"
assert sub["Target"].notna().all(), "NaN predictions"
assert np.isfinite(sub["Target"].values).all(), "non-finite predictions"
assert sub["Target"].between(0, 1).all(), "predictions outside [0,1]"

print("submission.csv OK")
print("rows:", len(sub))
print("columns:", list(sub.columns))
print("target mean:", float(sub["Target"].mean()))
print("target min/max:", float(sub["Target"].min()), float(sub["Target"].max()))
