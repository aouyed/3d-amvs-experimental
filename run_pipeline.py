#!/usr/bin/env python3
"""Single entry point for the 3D-AMV pipeline (July 1 2020 CLIMCAPS).

Usage:
    python run_pipeline.py download     # raw CLIMCAPS L2 + ERA5 (resumable)
    python run_pipeline.py run          # collocate -> retrieval -> coarsening
    python run_pipeline.py analyze      # error metrics + figures
    python run_pipeline.py all          # download + run + analyze
    python run_pipeline.py credentials  # re-enter Earthdata/CDS credentials
"""
import argparse
import os
import subprocess
import sys

import setup_credentials as creds

ROOT = os.path.dirname(os.path.abspath(__file__))

ERA5 = os.path.join(ROOT, "data", "raw", "era5", "era5_uv_20200701.nc")
NPZ = os.path.join(ROOT, "data", "processed", "l2_collocated_july1.npz")
CKPT = os.path.join(ROOT, "data", "processed", "retrieval_ckpt.npz")
COARSENED = [
    os.path.join(ROOT, "data", "processed", f"wind_coarsened_{p}_july1.nc")
    for p in ("ascending", "descending")
]


def _step(script, hint=None):
    print(f"\n=== {script} ===", flush=True)
    rc = subprocess.run([sys.executable, script], cwd=ROOT).returncode
    if rc != 0:
        msg = f"`{script}` failed (exit {rc})"
        if hint:
            msg += f"\n  hint: {hint}"
        sys.exit(msg + " — stopping.")


def _need(paths, hint):
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        for p in missing:
            print(f"  missing: {p}")
        print(f"  hint: {hint}")
        return False
    return True


def _require_creds(name, ok, prompt):
    if ok():
        return
    if not sys.stdin.isatty():
        print(f"[credentials] {name} credentials missing — run "
              f"`python setup_credentials.py` manually (see README 'Credentials').",
              file=sys.stderr)
        sys.exit(f"missing {name} credentials")
    prompt()


def download():
    _require_creds("Earthdata", creds.earthdata_ok, creds.prompt_earthdata)
    _step("download_l2.py",              # resumable: skips granules already present
          hint="If this was an authentication failure, run "
               "`python run_pipeline.py credentials` to re-enter your Earthdata "
               "credentials, then re-run `download`.")
    if not os.path.exists(ERA5):
        _require_creds("CDS", creds.cds_ok, creds.prompt_cds)
        _step("download_era5.py",
              hint="If this was an authentication failure, run "
                   "`python run_pipeline.py credentials` to re-enter your CDS "
                   "credentials, then re-run `download`.")


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
    _step("plot_quiver_all_levels.py")
    _step("plot_histograms.py")


def all_():
    download()
    run()
    analyze()


def credentials():
    print("\n=== setup_credentials.py --force ===", flush=True)
    rc = subprocess.run([sys.executable, "setup_credentials.py", "--force"], cwd=ROOT).returncode
    if rc != 0:
        sys.exit("`setup_credentials.py --force` failed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["download", "run", "analyze", "all", "credentials"])
    args = parser.parse_args()
    {"download": download, "run": run, "analyze": analyze, "all": all_,
     "credentials": credentials}[args.stage]()
