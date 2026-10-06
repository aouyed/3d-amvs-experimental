#!/usr/bin/env python3
"""Single entry point for the 3D-AMV pipeline (July 1 2020 CLIMCAPS).

Usage:
    python run_pipeline.py download   # raw CLIMCAPS L2 + ERA5 (resumable)
    python run_pipeline.py run        # collocate -> retrieval -> coarsening
    python run_pipeline.py analyze    # error metrics + figures
    python run_pipeline.py all        # download + run + analyze
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

ERA5 = os.path.join(ROOT, "data", "raw", "era5", "era5_uv_20200701.nc")
NPZ = os.path.join(ROOT, "data", "processed", "l2_collocated_july1.npz")
CKPT = os.path.join(ROOT, "data", "processed", "retrieval_ckpt.npz")
COARSENED = [
    os.path.join(ROOT, "data", "processed", f"wind_coarsened_{p}_july1.nc")
    for p in ("ascending", "descending")
]


def _step(script):
    print(f"\n=== {script} ===", flush=True)
    rc = subprocess.run([sys.executable, script], cwd=ROOT).returncode
    if rc != 0:
        sys.exit(f"`{script}` failed (exit {rc}) — stopping.")


def _need(paths, hint):
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        for p in missing:
            print(f"  missing: {p}")
        print(f"  hint: {hint}")
        return False
    return True


def download():
    _step("download_l2.py")              # resumable: skips granules already present
    if not os.path.exists(ERA5):
        _step("download_era5.py")


def run():
    if not _need([ERA5], "run `python run_pipeline.py download` first"):
        return
    _step("collocate_full_day.py")
    if not _need([NPZ], "collocation failed"):
        return
    _step("run_retrieval_all_levels.py")
    if not _need([CKPT], "retrieval failed"):
        return
    _step("run_vertical_coarsening.py")


def analyze():
    if not _need(COARSENED, "run `python run_pipeline.py run` first"):
        return
    _step("error_analysis.py")
    _step("plot_coarsened.py")
    _step("plot_histograms.py")


def all_():
    download()
    run()
    analyze()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["download", "run", "analyze", "all"])
    args = parser.parse_args()
    {"download": download, "run": run, "analyze": analyze, "all": all_}[args.stage]()
