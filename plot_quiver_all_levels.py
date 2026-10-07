#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quiver maps of every coarsened 3D-AMV level (retrieved vs ERA5), with RMSVD.

Reads wind_coarsened_{ascending,descending}_july1.nc (produced by
run_vertical_coarsening.py) and writes one two-panel quiver figure per
coarsened pressure level and pass, annotated with the per-level RMSVD.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import xarray as xr

ROOT = os.path.dirname(os.path.abspath(__file__))
PLOTS = os.path.join(ROOT, "data", "processed", "plots")
os.makedirs(PLOTS, exist_ok=True)

SUF = f"_{sys.argv[1]}" if len(sys.argv) > 1 else ""

FILES = {
    "ascending": os.path.join(ROOT, "data", "processed", f"wind_coarsened_ascending_july1{SUF}.nc"),
    "descending": os.path.join(ROOT, "data", "processed", f"wind_coarsened_descending_july1{SUF}.nc"),
}


def rmsvd(u, v, ue, ve):
    """RMS vector difference over pixels where all four fields are valid."""
    m = np.isfinite(u) & np.isfinite(v) & np.isfinite(ue) & np.isfinite(ve)
    if not m.any():
        return np.nan
    err = np.sqrt((u - ue) ** 2 + (v - ve) ** 2)
    return float(np.sqrt(np.mean(err[m] ** 2)))


def one_ax(ax, lon, lat, u, v, title, scale=240):
    ax.coastlines()
    ax.set_global()
    ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)
    Q = ax.quiver(lon, lat, u, v, transform=ccrs.PlateCarree(), scale=scale)
    ax.quiverkey(Q, 0.9, 0.95, 10, "10 m/s", labelpos="E", coordinates="figure")
    ax.set_title(title, fontsize=9)


for pname, f in FILES.items():
    ds = xr.open_dataset(f).sel(satellite="snpp")
    for pl in ds.plev.values:
        lev = float(pl)
        d = ds.sel(plev=lev)
        # per-level RMSVD on the full-resolution field (before plotting coarsening)
        r = rmsvd(d["u"].values, d["v"].values,
                  d["u_era5"].values, d["v_era5"].values)
        # coarsen for plotting density
        d = d.coarsen(latitude=4, longitude=4, boundary="trim").mean()
        X, Y = np.meshgrid(d.longitude.values, d.latitude.values)

        fig, axes = plt.subplots(1, 2, figsize=(16, 5),
                                 subplot_kw={"projection": ccrs.PlateCarree()})
        one_ax(axes[0], X, Y, d["u"].values, d["v"].values,
               f"{pname} retrieved {lev:.0f} hPa")
        one_ax(axes[1], X, Y, d["u_era5"].values, d["v_era5"].values,
               f"ERA5 {lev:.0f} hPa")
        fig.suptitle(f"RMSVD = {r:.2f} m/s", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        out = os.path.join(PLOTS, f"quiver_{pname}_{lev:.0f}hPa{SUF}.png")
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}  (RMSVD {r:.2f} m/s)")
    ds.close()
