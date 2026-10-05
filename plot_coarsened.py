#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quiver maps: retrieved vs ERA5 at representative coarsened levels."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import xarray as xr

ROOT = os.path.dirname(os.path.abspath(__file__))
PLOTS = os.path.join(ROOT, "data", "processed", "plots")
os.makedirs(PLOTS, exist_ok=True)

FILES = {
    "ascending": os.path.join(ROOT, "data", "processed", "wind_coarsened_ascending_july1.nc"),
    "descending": os.path.join(ROOT, "data", "processed", "wind_coarsened_descending_july1.nc"),
}
TARGETS = [850.0, 500.0, 250.0]   # representative levels (hPa)


def one_ax(ax, lon, lat, u, v, title, scale=120):
    ax.coastlines()
    ax.set_global()
    ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)
    Q = ax.quiver(lon, lat, u, v, transform=ccrs.PlateCarree(), scale=scale)
    ax.quiverkey(Q, 0.9, 0.95, 10, "10 m/s", labelpos="E", coordinates="figure")
    ax.set_title(title, fontsize=9)


for pname, f in FILES.items():
    ds = xr.open_dataset(f).sel(satellite="snpp")
    for tgt in TARGETS:
        d = ds.sel(plev=tgt, method="nearest")
        pl = float(d.plev)
        # coarsen for plotting density
        d = d.coarsen(latitude=4, longitude=4, boundary="trim").mean()
        X, Y = np.meshgrid(d.longitude.values, d.latitude.values)

        fig, axes = plt.subplots(1, 2, figsize=(16, 5),
                                 subplot_kw={"projection": ccrs.PlateCarree()})
        one_ax(axes[0], X, Y, d["u"].values, d["v"].values, f"{pname} retrieved {pl:.0f} hPa")
        one_ax(axes[1], X, Y, d["u_era5"].values, d["v_era5"].values, f"ERA5 {pl:.0f} hPa")
        fig.tight_layout()
        out = os.path.join(PLOTS, f"compare_{pname}_{pl:.0f}hPa.png")
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")
    ds.close()
