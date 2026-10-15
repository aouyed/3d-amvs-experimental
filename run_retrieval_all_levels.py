#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optical-flow retrieval on all H2O levels (>= 100 hPa cutoff).

Reads data/processed/l2_collocated_july1{SUF}.npz (built by the collocation step),
runs the 5-min swath retrieval (amv_calculators), and checkpoints U/V/UERA/VERA
per (pass, level) to data/processed/retrieval_ckpt{SUF}.npz (resumable).

Vertical coarsening is a separate step: run_vertical_coarsening.py.
"""

import os
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import amv_calculators as calc
import config
from parameters import parameters

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "data", "processed")

TAG = sys.argv[1] if len(sys.argv) > 1 else ""
SUF = f"_{TAG}" if TAG else ""
CKPT = os.path.join(OUT, f"retrieval_ckpt{SUF}.npz")

z = np.load(os.path.join(OUT, f"l2_collocated_july1{SUF}.npz"))
obs = z["obs"]          # (sat, pass, lat, lon) datetime64
q = z["q_mean"]         # (sat, pass, level, lat, lon)
cnt = z["cnt"]
plev = z["plev"]        # hPa, 56 levels

LAT = np.arange(-89.5, 90.0, 1.0)
LON = np.arange(-179.5, 180.0, 1.0)
PASSES = ["descending", "ascending"]

li = np.where((LAT >= -60.0) & (LAT <= 60.0))[0]
lj = np.where((LON >= -180.0) & (LON <= 180.0))[0]
lat_r = LAT[li]
lon_r = LON[lj]

NSAT, NPASS, NLEV = 2, 2, len(plev)
NY, NX = len(lat_r), len(lon_r)

# --- checkpoint init / load ---
if os.path.exists(CKPT):
    c = np.load(CKPT)
    U, V, UERA, VERA = c["U"], c["V"], c["UERA"], c["VERA"]
    done = c["done"]
else:
    U = np.full((NSAT, NPASS, NLEV, NY, NX), np.nan, dtype=np.float32)
    V = np.full_like(U, np.nan)
    UERA = np.full_like(U, np.nan)
    VERA = np.full_like(U, np.nan)
    done = np.zeros((NPASS, NLEV), dtype=bool)

era5 = xr.open_dataset(os.path.join(ROOT, "data", "raw", "era5", "era5_uv_20200701.nc"))


def build_era5_model(hpa):
    ds = era5.sel(pressure_level=hpa, method="nearest")
    ds = ds.drop_vars(["expver", "number"], errors="ignore")
    ds = ds.drop("pressure_level")
    ds = ds.rename({"valid_time": "time"})
    ds = ds.assign_coords(longitude=((ds.longitude + 180) % 360) - 180)
    ds = ds.reindex(longitude=np.sort(ds.longitude.values))
    return ds


param = parameters()
param.set_alg("tvl1")
param.set_Lambda(0.15)

remaining = int((~done).sum())
print(f"remaining combos: {remaining}/{NPASS*NLEV}", flush=True)

for p, pname in enumerate(PASSES):
    for k in range(NLEV):
        if done[p, k]:
            continue
        hpa = float(plev[k])

        q_snpp = q[0, p, k][np.ix_(li, lj)]
        q_j1 = q[1, p, k][np.ix_(li, lj)]
        t_snpp = obs[0, p][np.ix_(li, lj)]
        t_j1 = obs[1, p][np.ix_(li, lj)]

        hum = np.stack([q_snpp, q_j1], axis=-1)
        obs_arr = np.empty((NY, NX, 2), dtype="datetime64[ns]")
        obs_arr[..., 0] = t_snpp.astype("datetime64[ns]")
        obs_arr[..., 1] = t_j1.astype("datetime64[ns]")

        ds_unit = xr.Dataset(
            {"specific_humidity_mean": (["latitude", "longitude", "satellite"], hum),
             "obs_time": (["latitude", "longitude", "satellite"], obs_arr)},
            coords={"latitude": lat_r, "longitude": lon_r, "satellite": ["snpp", "j1"]},
        )
        ds_unit = calc.prepare_ds(ds_unit)
        df = ds_unit.to_dataframe()

        ds_model = build_era5_model(hpa)

        for start, end in calc.swath_initializer(
            ds_unit, config.SWATH_MINUTES, config.SWATH_HOURS
        ):
            ds_snpp = ds_unit.loc[{"satellite": "snpp"}]
            ds_j1 = ds_unit.loc[{"satellite": "j1"}]
            ds_merged, ds_snpp, ds_j1, df_snpp = calc.prepare_patch(
                ds_snpp, ds_j1, ds_model, start, end
            )
            if df_snpp.shape[0] > config.MIN_PATCH_PIXELS:
                df, _, _, _ = calc.amv_calculator(ds_merged, df, param)

        ds_r = xr.Dataset.from_dataframe(df)
        # store UNFILTERED u/v/u_era5/v_era5 (vertical_coarse applies the filter)
        for si, sn in enumerate(["snpp", "j1"]):
            U[si, p, k] = ds_r["u"].sel(satellite=sn).squeeze().values
            V[si, p, k] = ds_r["v"].sel(satellite=sn).squeeze().values
            UERA[si, p, k] = ds_r["u_era5"].sel(satellite=sn).squeeze().values
            VERA[si, p, k] = ds_r["v_era5"].sel(satellite=sn).squeeze().values

        done[p, k] = True
        np.savez(CKPT, U=U, V=V, UERA=UERA, VERA=VERA, done=done)
        print(f"  {pname} {hpa:6.1f} hPa done  ({int(done.sum())}/{NPASS*NLEV})", flush=True)

era5.close()
print("ALL LEVELS DONE", flush=True)
