"""
run_pipeline.py
───────────────
Runs the full kite-ideal pipeline end to end.

Usage:
    python run_pipeline.py

Stages:
    1. data_generator.py  — creates raw messy CSVs
    2. cleaner.py         — fixes all data quality issues
    3. transformer.py     — scores all rider × kite combinations
    4. analyzer.py        — finds optimal quiver per rider
"""

import subprocess
import sys
import time

STAGES = [
    ("Data generation",  "data_generator.py"),
    ("Cleaning",         "cleaner.py"),
    ("Transformation",   "transformer.py"),
    ("Analysis",         "analyzer.py"),
]


def run_stage(label, script):
    print(f"\n{'═' * 52}")
    print(f"  Stage: {label}")
    print(f"{'═' * 52}")
    start  = time.time()
    result = subprocess.run([sys.executable, script], capture_output=False)
    elapsed = round(time.time() - start, 1)

    if result.returncode != 0:
        print(f"\n[✗] {label} failed. Pipeline stopped.")
        sys.exit(1)

    print(f"\n  Completed in {elapsed}s")


if __name__ == "__main__":
    print("\n🪁  kite-ideal pipeline starting...")
    total_start = time.time()

    for label, script in STAGES:
        run_stage(label, script)

    total = round(time.time() - total_start, 1)
    print(f"\n{'═' * 52}")
    print(f"  Pipeline complete in {total}s")
    print(f"  Recommendations → outputs/quiver_recommendations.csv")
    print(f"  Charts          → outputs/charts/")
    print(f"{'═' * 52}\n")