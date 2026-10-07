#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""2D histogram (speed vs ERA5 speed) for OLD (L2-derived q) vs NEW (L3 spec_hum)."""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xarray as xr

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "data", "processed")
PLOTS = os.path.join(OUT, "plots")
os.makedirs(PLOTS, exist_ok=True)

EDGES = [0.0, 30.0]
BINS = 100


def collect(files):
    spd, spd_e, u, v, ue, ve = [], [], [], [], [], []
    for f in files:
        ds = xr.open_dataset(f).sel(satellite="snpp")
        u.append(ds["u"].values.ravel()); v.append(ds["v"].values.ravel())
        ue.append(ds["u_era5"].values.ravel()); ve.append(ds["v_era5"].values.ravel())
        ds.close()
    u = np.concatenate(u); v = np.concatenate(v); ue = np.concatenate(ue); ve = np.concatenate(ve)
    m = np.isfinite(u) & np.isfinite(v) & np.isfinite(ue) & np.isfinite(ve)
    u, v, ue, ve = u[m], v[m], ue[m], ve[m]
    spd = np.sqrt(u**2 + v**2); spd_e = np.sqrt(ue**2 + ve**2)
    rmsvd = float(np.sqrt(np.mean((u - ue)**2 + (v - ve)**2)))
    corr = float(np.corrcoef(spd, spd_e)[0, 1])
    return spd, spd_e, rmsvd, corr, len(spd)


def hist2d(spd, spd_e):
    xb = np.linspace(EDGES[0], EDGES[1], BINS + 1)
    hm, _, _ = np.histogram2d(spd, spd_e, bins=[xb, xb])
    return (100 * hm / hm.sum()).T


new = collect([os.path.join(OUT, f"wind_coarsened_{p}_july1.nc") for p in ("ascending", "descending")])
l2q_files = [os.path.join(OUT, f"wind_coarsened_{p}_july1_l2q.nc") for p in ("ascending", "descending")]
if not all(os.path.exists(f) for f in l2q_files):
    print("SKIP: L2-QC (_l2q) comparison files not present; run the no-QC L2 variant to generate them.")
    raise SystemExit(0)
old = collect(l2q_files)

print(f"{'':12s} {'n':>7s} {'RMSVD':>7s} {'speed corr':>11s}")
print(f"{'OLD (L2 q)':12s} {old[4]:7d} {old[2]:7.2f} {old[3]:11.3f}")
print(f"{'NEW (L3 q)':12s} {new[4]:7d} {new[2]:7.2f} {new[3]:11.3f}")

fig, axes = plt.subplots(2, 1, figsize=(6, 10))
for ax, data, title, r in [
    (axes[0], new, "NEW (L3 spec_hum)", new[3]),
    (axes[1], old, "OLD (L2-derived q)", old[3]),
]:
    im = ax.imshow(hist2d(data[0], data[1]), vmin=0, vmax=1.5e-1,
                   extent=[EDGES[0], EDGES[1], EDGES[0], EDGES[1]],
                   aspect="auto", origin="lower", cmap="CMRmap_r")
    ax.set_xlabel("speed (m/s)"); ax.set_ylabel("ERA5 speed (m/s)")
    ax.set_title(title)
    ax.text(0.65, 0.9, f"r = {r:.2f}", transform=ax.transAxes, fontsize=12)
    ax.plot([0, 30], [0, 30], "k--", lw=0.7)

cbar_ax = fig.add_axes([0.25, 0.02, 0.5, 0.02])
fig.colorbar(im, cax=cbar_ax, orientation="horizontal", label="percent")
fig.tight_layout()
out = os.path.join(PLOTS, "2d_hist_l3q_vs_l2q.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"wrote {out}")
