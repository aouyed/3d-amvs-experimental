#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Error analysis + plots for the coarsened 3D-AMV winds vs ERA5."""

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

files = {
    "ascending": os.path.join(ROOT, "data", "processed", "wind_coarsened_ascending_july1.nc"),
    "descending": os.path.join(ROOT, "data", "processed", "wind_coarsened_descending_july1.nc"),
}

all_u, all_ue, all_v, all_ve = [], [], [], []
print("=" * 100)
print(f"{'pass':11s} {'plev (hPa)':>10s} {'n':>6s} {'RMSVD':>7s} {'u bias':>7s} {'v bias':>7s} {'corr(speed)':>11s}")
print("-" * 100)
per_level = {}
for pname, f in files.items():
    ds = xr.open_dataset(f).sel(satellite="snpp")
    plevs = ds.plev.values
    rmsvd_list, plev_list = [], []
    for i, pl in enumerate(plevs):
        u = ds["u"].isel(plev=i).values
        v = ds["v"].isel(plev=i).values
        ue = ds["u_era5"].isel(plev=i).values
        ve = ds["v_era5"].isel(plev=i).values
        m = np.isfinite(u) & np.isfinite(v) & np.isfinite(ue) & np.isfinite(ve)
        if m.sum() < 50:
            continue
        u, v, ue, ve = u[m], v[m], ue[m], ve[m]
        spd = np.sqrt(u**2 + v**2)
        spd_e = np.sqrt(ue**2 + ve**2)
        rmsvd = float(np.sqrt(np.mean((u - ue)**2 + (v - ve)**2)))
        u_bias = float(np.mean(u - ue))
        v_bias = float(np.mean(v - ve))
        corr = float(np.corrcoef(spd, spd_e)[0, 1])
        print(f"{pname:11s} {pl:10.1f} {m.sum():6d} {rmsvd:7.2f} {u_bias:+7.2f} {v_bias:+7.2f} {corr:11.3f}")
        rmsvd_list.append(rmsvd)
        plev_list.append(pl)
        all_u.append(u); all_ue.append(ue); all_v.append(v); all_ve.append(ve)
    per_level[pname] = (np.array(plev_list), np.array(rmsvd_list))
    ds.close()

print("=" * 100)
all_u = np.concatenate(all_u); all_ue = np.concatenate(all_ue)
all_v = np.concatenate(all_v); all_ve = np.concatenate(all_ve)
spd = np.sqrt(all_u**2 + all_v**2); spd_e = np.sqrt(all_ue**2 + all_ve**2)
overall_rmsvd = float(np.sqrt(np.mean((all_u - all_ue)**2 + (all_v - all_ve)**2)))
overall_corr = float(np.corrcoef(spd, spd_e)[0, 1])
print(f"OVERALL: n={len(spd):d} | RMSVD {overall_rmsvd:.2f} m/s | speed corr {overall_corr:.3f}")

# ---- Fig 1: RMSVD vs pressure + scatter ----
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax = axes[0]
for pname, (pl, rv) in per_level.items():
    ax.plot(rv, pl, "-o", label=pname)
ax.set_xlabel("RMSVD (m/s)")
ax.set_ylabel("pressure (hPa)")
ax.invert_yaxis()
ax.set_title("RMSVD vs ERA5 by level")
ax.legend(); ax.grid(alpha=0.3)

ax = axes[1]
ax.scatter(spd_e, spd, s=2, alpha=0.15)
lim = [0, max(np.nanpercentile(spd_e, 99), np.nanpercentile(spd, 99))]
ax.plot(lim, lim, "r--", lw=1, label="1:1")
ax.set_xlabel("ERA5 speed (m/s)")
ax.set_ylabel("retrieved speed (m/s)")
ax.set_title(f"speed comparison (corr {overall_corr:.2f}, RMSVD {overall_rmsvd:.1f} m/s)")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "error_analysis.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"wrote {os.path.join(PLOTS, 'error_analysis.png')}")
