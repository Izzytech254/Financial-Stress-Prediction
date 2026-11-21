"""End-to-end v2 entry point.

Runs the frozen-fold screening/confirmation pipeline, then finalizes the
accepted leak-free LightGBM + CatBoost blend and writes submission.csv.

Run:
    .venv/bin/python solution_v2.py
"""

from __future__ import annotations

import os
import runpy

ROOT = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    runpy.run_path(os.path.join(ROOT, "run_v2.py"), run_name="__main__")
    runpy.run_path(os.path.join(ROOT, "finalize_v2_clean.py"), run_name="__main__")
