#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collocate July 1 2020:
  * specific humidity  <- L3 gridded product (already 1-deg, QC'd), `spec_hum`
  * observation time   <- L2 swath FOVs, nearest-neighbour binned to 1-deg

Output: data/processed/l2_collocated_july1.npz
  obs  (sat, pass, lat, lon)   datetime64
  q_mean (sat, pass, level, lat, lon)  = L3 spec_hum (already averaged)
  cnt  (sat, pass, lat, lon)   number of L2 FOVs per cell
  plev (level,)                hPa
"""

import os

import numpy as np
import xarray as xr
from netCDF4 import Dataset as NCDataset

ROOT = os.path.dirname(os.path.abspath(__file__))
L2 = os.path.join(ROOT, "data", "raw", "l2")
L3_DIR = os.path.join(ROOT, "data", "raw", "CLIMCAPS_july")
OUT = os.path.join(ROOT, "data", "processed")

H2O_CUTOFF_PA = 10000.0
EPOCH = np.datetime64("1993-01-01T00:00:00")
REF = np.datetime64("2020-01-01T00:00:00")

LAT = np.arange(-89.5, 90.0, 1.0)
LON = np.arange(-179.5, 180.0, 1.0)

# --- 1. Grid L2 obs_time per (satellite, asc_flag pass) -------------------
obs_sum = np.zeros((2, 2, 180, 360), dtype=np.float64)
cnt = np.zeros((2, 2, 180, 360), dtype=np.int32)

manifests = [(0, "manifest_snpp.txt"), (1, "manifest_j1.txt")]
for si, mf in manifests:
    for line in open(os.path.join(L2, mf)):
        gid = line.strip().split("\t")[0]
        nc = NCDataset(os.path.join(L2, gid))
        lat = nc.variables["lat"][:].ravel()
        lon = nc.variables["lon"][:].ravel()
        raw = nc.variables["obs_time_tai93"][:].ravel()
        qc = nc.variables["qc_flag"][:].ravel()
        asc = nc.variables["asc_flag"][:]
        n_xtrack = nc.variables["h2o_vap_mol_lay"][:].shape[1]
        nc.close()

        asc_per = np.repeat(asc, n_xtrack)
        t = np.full(raw.shape, np.datetime64("NaT"), dtype="datetime64[ns]")
        v = ~np.isnan(raw)
        t[v] = EPOCH + np.round(raw[v] * 1e9).astype(np.int64).astype("timedelta64[ns]")

        i = np.round(lat + 89.5).astype(int)
        j = np.round(lon + 179.5).astype(int)
        ok = (qc <= 1) & (i >= 0) & (i < 180) & (j >= 0) & (j < 360) & ~np.isnat(t)
        t_s = (t - REF) / np.timedelta64(1, "s")

        for p in (0, 1):
            m = ok & (asc_per == p)
            if m.any():
                np.add.at(obs_sum[si, p], (i[m], j[m]), t_s[m])
                np.add.at(cnt[si, p], (i[m], j[m]), 1)

obs = np.full((2, 2, 180, 360), np.datetime64("NaT"), dtype="datetime64[ns]")
mean_t = np.full((2, 2), np.datetime64("NaT"), dtype="datetime64[ns]")
for si in range(2):
    for p in range(2):
        m = cnt[si, p] > 0
        obs[si, p][m] = REF + (obs_sum[si, p][m] / cnt[si, p][m] * 1e9).astype(np.int64).astype("timedelta64[ns]")
        mean_t[si, p] = obs[si, p][m].astype("datetime64[ns]").astype(np.int64).mean().astype("datetime64[ns]")
print("L2 mean obs_time per pass (snpp):", mean_t[0], flush=True)

# --- 2. Load L3 spec_hum ---------------------------------------------------
snpp_f = os.path.join(L3_DIR, "SNDR.SNPP.CRIMSS.20200701.D01.L3_CLIMCAPS_QCC.std.v02_38.G.210219221503.nc")
j1_f = os.path.join(L3_DIR, "SNDR.J1.CRIMSS.20200701.D01.L3_CLIMCAPS_QCC.std.v02_38.G.210219134411.nc")
l3 = [xr.open_dataset(snpp_f), xr.open_dataset(j1_f)]

l3_t = l3[0].obs_time_tai93.values
print("L3 obs_time_tai93:", l3_t, flush=True)

# map L3 orbit_pass -> L2 asc_flag:
#   orbit_pass timedelta ~13h = ascending (asc_flag=1), ~1h = descending (asc_flag=0)
orbit_h = l3[0].orbit_pass.values.astype("timedelta64[h]").astype(int)
map_l3_to_l2 = [1 if h > 6 else 0 for h in orbit_h]
print("L3 orbit_pass hours:", orbit_h, "-> L2 asc_flag:", map_l3_to_l2, flush=True)

# levels >= cutoff
ap_h2o = l3[0].air_pres_h2o.values          # Pa
p_idx = np.where(ap_h2o >= H2O_CUTOFF_PA)[0]
plev = ap_h2o[p_idx] / 100.0
NLEV = len(p_idx)
print(f"levels kept: {NLEV} ({plev[0]:.0f} .. {plev[-1]:.0f} hPa)", flush=True)

# lat orientation
if l3[0].lat.values[0] > l3[0].lat.values[-1]:
    l3 = [x.reindex(lat=list(reversed(x.lat))) for x in l3]

# build q_mean (sat, pass, level, lat, lon)
q_mean = np.full((2, 2, NLEV, 180, 360), np.nan, dtype=np.float32)
for si in range(2):
    sh = l3[si].spec_hum                    # (orbit_pass, air_pres_h2o, lat, lon)
    for l3_idx in range(2):
        p = map_l3_to_l2[l3_idx]
        q_mean[si, p] = sh.isel(orbit_pass=l3_idx).isel(air_pres_h2o=p_idx).values  # (NLEV,180,360)
    l3[si].close()

np.savez_compressed(os.path.join(OUT, "l2_collocated_july1.npz"),
                    obs=obs, q_mean=q_mean, cnt=cnt, plev=plev)
print("saved l2_collocated_july1.npz", flush=True)
