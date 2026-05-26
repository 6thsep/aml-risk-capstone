"""
QM640 Capstone: Master Pipeline Runner
Executes all analysis scripts in sequence and reports timing.

Usage: python run_all.py
"""

import subprocess
import sys
import os
import time

ROOT = os.path.dirname(os.path.abspath(__file__))

STEPS = [
    ("EDA",                  "src/02_eda.py"),
    ("Feature Preparation",  "src/03_feature_preparation.py"),
    ("RQ1: LASSO",          "src/04_model_rq1_lasso.py"),
    ("RQ2: Classification", "src/05_model_rq2_classification.py"),
    ("RQ3: Moderation",     "src/06_model_rq3_moderation.py"),
    ("RQ4: Clustering",     "src/07_model_rq4_clustering.py"),
]


def run_step(label: str, script: str) -> bool:
    path = os.path.join(ROOT, script)
    if not os.path.exists(path):
        print(f"  [SKIP] {path} not found")
        return True

    print(f"\n{'='*60}")
    print(f"  STEP: {label}")
    print(f"  Script: {script}")
    print(f"{'='*60}")

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, path],
        cwd=ROOT,
        capture_output=False,
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n  [FAILED] ({elapsed:.1f}s): returncode {result.returncode}")
        return False

    print(f"\n  [done] ({elapsed:.1f}s)")
    return True


def main():
    print("QM640 AML Risk Capstone: Full Pipeline")
    print(f"Python: {sys.executable}")
    print(f"Root:   {ROOT}")

    t_start = time.time()
    for label, script in STEPS:
        ok = run_step(label, script)
        if not ok:
            print(f"\nPipeline aborted at step: {label}")
            sys.exit(1)

    total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  ALL STEPS COMPLETE  ({total:.1f}s total)")
    print(f"  Figures → outputs/figures/")
    print(f"  Results → outputs/results/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
