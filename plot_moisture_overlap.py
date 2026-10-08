#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot SNPP / NOAA-20 water-vapor fields and their 5-min-swath overlap.

Descending pass only, ~700 hPa level, from the L2-regridded (no vertical
coarsening) collocated product. Overlap mirrors the wind retrieval: 5-minute
windows over 24 h, SNPP offset +50 min.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(ROOT, "data", "processed", "l2_collocated_july1_noqc.npz")
PLOTS = os.path.join(ROOT, "data", "processed", "plots")
os.makedirs(PLOTS, exist_ok=True)

lat_e = np.arange(-90.0, 90.0001, 1.0)     # 181 edges
lon_e = np.arange(-180.0, 180.0001, 1.0)   # 361 edges

z = np.load(NPZ)
obs = z["obs"]          # (sat, pass, lat, lon) datetime64[ns]
q = z["q_mean"]         # (sat, pass, level, lat, lon)
plev = z["plev"]

k = int(np.argmin(np.abs(plev - 700.0)))
hpa = float(plev[k])

PASS = 0                # descending
SWATH_MIN = 5
SNPP_OFFSET_MIN = 50
NWIN = 24 * 60 // SWATH_MIN

allobs = obs[:, PASS, :, :]
valid = allobs[~np.isnat(allobs)]
mind = valid.min()

overlap = np.zeros((180, 360), dtype=bool)
for i in range(NWIN):
    start = mind + np.timedelta64(i * SWATH_MIN, "m")
    end = mind + np.timedelta64((i + 1) * SWATH_MIN, "m")
    snpp_start = start + np.timedelta64(SNPP_OFFSET_MIN, "m")
    snpp_end = end + np.timedelta64(SNPP_OFFSET_MIN, "m")
    j1 = (obs[1, PASS] >= start) & (obs[1, PASS] <= end)
    snpp = (obs[0, PASS] >= snpp_start) & (obs[0, PASS] <= snpp_end)
    overlap |= (j1 & snpp)

snpp_h = q[0, PASS, k].astype(float)
j1_h = q[1, PASS, k].astype(float)
snpp_h[~np.isfinite(snpp_h)] = np.nan
j1_h[~np.isfinite(j1_h)] = np.nan
overlap_h = np.where(overlap, j1_h, np.nan)

allvals = np.concatenate([snpp_h[np.isfinite(snpp_h)], j1_h[np.isfinite(j1_h)]])
vmax = float(np.nanpercentile(allvals, 98))
vmin = 0.0

titles = [f"SNPP {hpa:.0f} hPa", f"NOAA-20 {hpa:.0f} hPa",
          f"NOAA-20 in overlap cells"]

fig, axes = plt.subplots(1, 3, figsize=(18, 5),
                         subplot_kw={"projection": ccrs.PlateCarree()})
for ax, data, title in zip(axes, [snpp_h, j1_h, overlap_h], titles):
    ax.coastlines()
    ax.set_global()
    ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)
    im = ax.pcolormesh(lon_e, lat_e, data, transform=ccrs.PlateCarree(),
                       cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=11)

cb = fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.046, pad=0.08,
                  label="specific humidity (kg/kg)")
out = os.path.join(PLOTS, "moisture_overlap_700hPa.png")
fig.savefig(out, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"level {k} = {hpa:.1f} hPa | overlap cells = {int(overlap.sum())} | vmax = {vmax:.4f}")
print(f"wrote {out}")
