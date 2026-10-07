#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Error metrics + plots for the coarsened 3D-AMV winds vs ERA5.

Reads the coarsened output written by src/vertical_coarsening.py
({tag}_{date}_{orbit}.nc), where orbit "am" = descending and "pm" = ascending.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import config
from parameters import parameters

PLOTS = os.path.join(config.DATA_PROCESSED, "plots")
os.makedirs(PLOTS, exist_ok=True)

# "am" = descending, "pm" = ascending (measured local solar time)
ORBITS = {"descending": "am", "ascending": "pm"}


def compute_metrics(u, v, u_era5, v_era5):
    """Return per-level error metrics (pure)."""
    m = np.isfinite(u) & np.isfinite(v) & np.isfinite(u_era5) & np.isfinite(v_era5)
    u, v, ue, ve = u[m], v[m], u_era5[m], v_era5[m]
    if len(u) == 0:
        return None
    err = np.sqrt((u - ue) ** 2 + (v - ve) ** 2)
    spd = np.sqrt(u ** 2 + v ** 2)
    spd_e = np.sqrt(ue ** 2 + ve ** 2)
    return {
        "n": int(len(u)),
        "rmsvd": float(np.sqrt(np.mean(err ** 2))),
        "u_bias": float(np.mean(u - ue)),
        "v_bias": float(np.mean(v - ve)),
        "corr": float(np.corrcoef(spd, spd_e)[0, 1]) if len(u) > 1 else np.nan,
        "u": u,
        "v": v,
        "u_era5": ue,
        "v_era5": ve,
        "spd": spd,
        "spd_era5": spd_e,
    }


def load_coarsened(suf=""):
    """Load the coarsened output for both passes (I/O)."""
    dsets = {}
    for pname in ("ascending", "descending"):
        f = os.path.join(config.DATA_PROCESSED, f"wind_coarsened_{pname}_july1{suf}.nc")
        dsets[pname] = xr.open_dataset(f).sel(satellite="snpp")
    return dsets


def plot(metrics, per_level, suf=""):
    """Write figures (side effect)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    for pname, (plevs, rmsvd) in per_level.items():
        ax.plot(rmsvd, plevs, "-o", label=pname)
    ax.set_xlabel("RMSVD (m/s)")
    ax.set_ylabel("pressure (hPa)")
    ax.invert_yaxis()
    ax.set_title("RMSVD vs ERA5 by level")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.scatter(metrics["spd_era5"], metrics["spd"], s=2, alpha=0.15)
    lim = [0, max(np.nanpercentile(metrics["spd_era5"], 99), np.nanpercentile(metrics["spd"], 99))]
    ax.plot(lim, lim, "r--", lw=1, label="1:1")
    ax.set_xlabel("ERA5 speed (m/s)")
    ax.set_ylabel("retrieved speed (m/s)")
    ax.set_title(f"speed comparison (corr {metrics['corr']:.2f}, RMSVD {metrics['rmsvd']:.1f} m/s)")
    ax.legend(); ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, f"error_analysis{suf}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    suf = f"_{sys.argv[1]}" if len(sys.argv) > 1 else ""
    dsets = load_coarsened(suf)

    per_level = {}
    all_metrics = []
    print("=" * 100)
    print(f"{'pass':11s} {'plev (hPa)':>10s} {'n':>6s} {'RMSVD':>7s} {'u bias':>7s} {'v bias':>7s} {'corr(speed)':>11s}")
    print("-" * 100)
    for pname, ds in dsets.items():
        plevs = ds.plev.values
        rmsvd_list, plev_list = [], []
        for i, pl in enumerate(plevs):
            m = compute_metrics(
                ds["u"].isel(plev=i).values,
                ds["v"].isel(plev=i).values,
                ds["u_era5"].isel(plev=i).values,
                ds["v_era5"].isel(plev=i).values,
            )
            if m is None or m["n"] < 50:
                continue
            print(f"{pname:11s} {pl:10.1f} {m['n']:6d} {m['rmsvd']:7.2f} {m['u_bias']:+7.2f} {m['v_bias']:+7.2f} {m['corr']:11.3f}")
            rmsvd_list.append(m["rmsvd"])
            plev_list.append(pl)
            all_metrics.append(m)
        per_level[pname] = (np.array(plev_list), np.array(rmsvd_list))
        ds.close()

    print("=" * 100)
    combined = {
        key: np.concatenate([m[key] for m in all_metrics])
        for key in ["u", "v", "u_era5", "v_era5", "spd", "spd_era5"]
    }
    overall = compute_metrics(combined["u"], combined["v"], combined["u_era5"], combined["v_era5"])
    print(f"OVERALL: n={overall['n']} | RMSVD {overall['rmsvd']:.2f} m/s | speed corr {overall['corr']:.3f}")

    plot(overall, per_level, suf)
    print(f"wrote {os.path.join(PLOTS, f'error_analysis{suf}.png')}")


if __name__ == "__main__":
    main()
