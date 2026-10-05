#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vertical coarsening via the existing vertical_coarsening.vertical_coarse.

Loads the all-levels retrieval checkpoint, builds a per-pass (plev, lat, lon,
satellite) dataset, then applies compute() filter + coarsen(plev=5).median().
"""

import os
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import vertical_coarsening as vc

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "data", "processed")

TAG = sys.argv[1] if len(sys.argv) > 1 else ""
SUF = f"_{TAG}" if TAG else ""

c = np.load(os.path.join(OUT, f"retrieval_ckpt{SUF}.npz"))
U, V, UERA, VERA = c["U"], c["V"], c["UERA"], c["VERA"]   # (sat, pass, level, lat, lon)

z = np.load(os.path.join(OUT, f"l2_collocated_july1{SUF}.npz"))
obs = z["obs"]      # (sat, pass, lat, lon)
q = z["q_mean"]     # (sat, pass, level, lat, lon)
plev = z["plev"]    # hPa

LAT = np.arange(-89.5, 90.0, 1.0)
LON = np.arange(-179.5, 180.0, 1.0)
li = np.where((LAT >= -60.0) & (LAT <= 60.0))[0]
lj = np.where((LON >= -180.0) & (LON <= 180.0))[0]
lat_r = LAT[li]
lon_r = LON[lj]
NY, NX = len(lat_r), len(lon_r)
NLEV = len(plev)

THRESH = 10.0
N_LAYERS = 5

print("=" * 80)
for p, pname in enumerate(["ascending", "descending"]):
    u = U[:, p].transpose(1, 2, 3, 0)        # (level, lat, lon, sat)
    v = V[:, p].transpose(1, 2, 3, 0)
    ue = UERA[:, p].transpose(1, 2, 3, 0)
    ve = VERA[:, p].transpose(1, 2, 3, 0)
    hum = q[:, p][:, :, li, :][:, :, :, lj].transpose(1, 2, 3, 0)     # (level, lat, lon, sat)
    obs_reg = obs[:, p][:, li, :][:, :, lj]                          # (sat, lat, lon)
    obs_b = np.broadcast_to(obs_reg.transpose(1, 2, 0), (NLEV, NY, NX, 2)).copy()

    ds = xr.Dataset(
        {
            "u": (["plev", "latitude", "longitude", "satellite"], u),
            "v": (["plev", "latitude", "longitude", "satellite"], v),
            "u_era5": (["plev", "latitude", "longitude", "satellite"], ue),
            "v_era5": (["plev", "latitude", "longitude", "satellite"], ve),
            "humidity_overlap": (["plev", "latitude", "longitude", "satellite"], hum),
            "obs_time": (["plev", "latitude", "longitude", "satellite"], obs_b),
        },
        coords={"plev": plev, "latitude": lat_r, "longitude": lon_r,
                "satellite": ["snpp", "j1"]},
    )

    # RMSVD before coarsening (unfiltered, all levels)
    err = np.sqrt((u - ue)**2 + (v - ve)**2)
    rmsvd_all = float(np.sqrt(np.nanmean(err**2)))

    ds_c = vc.vertical_coarse(ds, THRESH, N_LAYERS)   # existing filter + coarsen

    u_c = ds_c["u"].sel(satellite="snpp").values
    v_c = ds_c["v"].sel(satellite="snpp").values
    ue_c = ds_c["u_era5"].sel(satellite="snpp").values
    ve_c = ds_c["v_era5"].sel(satellite="snpp").values
    err_c = np.sqrt((u_c - ue_c)**2 + (v_c - ve_c)**2)
    rmsvd_c = float(np.sqrt(np.nanmean(err_c**2)))
    spd = np.sqrt(u_c**2 + v_c**2)

    print(f"{pname:11s} | levels {NLEV} -> {ds_c.sizes.get('plev')} (coarsen x{N_LAYERS}) | "
          f"RMSVD all-levels {rmsvd_all:5.1f} -> coarsened {rmsvd_c:5.1f} m/s | "
          f"speed median {np.nanmedian(spd):5.1f} m/s | p90 {np.nanpercentile(spd,90):5.1f} m/s")

    # drop obs_time (datetime64 NaT) before saving to avoid CF encode issues
    ds_out = ds_c.drop_vars("obs_time")
    ds_out.attrs["title"] = f"3D-AMV winds vertically coarsened (x{N_LAYERS}), {pname}, July 1 2020"
    ds_out.attrs["thresh"] = THRESH
    out = os.path.join(OUT, f"wind_coarsened_{pname}_july1{SUF}.nc")
    ds_out.to_netcdf(out)
    print(f"  wrote {out}")

print("=" * 80)
