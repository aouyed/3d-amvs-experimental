#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collocate July 1 2020 specific humidity from L2 CLIMCAPS retrievals with
NO quality filtering (per Ouyed et al. 2023: "we did not apply any quality
filtering ahead of gridding").

  * specific humidity  <- L2 `h2o_vap_mol_lay` (molecules/m2) converted to
                          kg/kg via hydrostatic layer mass, then averaged
                          onto a 1x1 degree grid (all valid retrievals,
                          regardless of qc_flag).
  * observation time   <- L2 swath FOV midtimes, binned to 1 degree.

Output: data/processed/l2_collocated_july1_noqc.npz
  obs     (sat, pass, lat, lon)   datetime64[ns]
  q_mean  (sat, pass, level, lat, lon)  kg/kg
  cnt     (sat, pass, lat, lon)   number of L2 FOVs per cell
  plev    (level,)                hPa  (levels >= 100 hPa)
"""

import os

import numpy as np
from netCDF4 import Dataset as NCDataset

ROOT = os.path.dirname(os.path.abspath(__file__))
L2 = os.path.join(ROOT, "data", "raw", "l2")
OUT = os.path.join(ROOT, "data", "processed")

H2O_CUTOFF_PA = 10000.0          # keep layers with mid-pressure >= 100 hPa
M_H2O = 0.018015                 # kg / mol
N_A = 6.02214076e23              # molecules / mol
G = 9.80665                      # m / s^2

EPOCH = np.datetime64("1993-01-01T00:00:00")
REF = np.datetime64("2020-01-01T00:00:00")

LAT = np.arange(-89.5, 90.0, 1.0)
LON = np.arange(-179.5, 180.0, 1.0)

# --- fixed L2 layer geometry (same for every granule) -----------------------
ref_f = os.path.join(L2, "manifest_j1.txt")
with open(ref_f) as fh:
    ref_gid = fh.readline().strip().split("\t")[0]
ref = NCDataset(os.path.join(L2, ref_gid))
air_pres_lay = ref.variables["air_pres_lay"][:]            # Pa, mid-layer
bnds = ref.variables["air_pres_lay_bnds"][:]               # (100, 2) Pa
ref.close()

dp = bnds[:, 1] - bnds[:, 0]                               # Pa per layer
p_idx = np.where(air_pres_lay >= H2O_CUTOFF_PA)[0]
plev = air_pres_lay[p_idx] / 100.0                         # hPa
NLEV = len(p_idx)
# q = N_H2O * M_H2O * g / (dp * N_A)   (kg/kg)
factor = (M_H2O * G / (dp * N_A))[p_idx]                   # (NLEV,)
print(f"levels kept: {NLEV} ({plev[0]:.0f} .. {plev[-1]:.0f} hPa)", flush=True)

# --- accumulators -----------------------------------------------------------
q_sum = np.zeros((2, 2, NLEV, 180, 360), dtype=np.float64)
q_cnt = np.zeros((2, 2, NLEV, 180, 360), dtype=np.int32)
obs_sum = np.zeros((2, 2, 180, 360), dtype=np.float64)
cnt = np.zeros((2, 2, 180, 360), dtype=np.int32)

manifests = [(0, "manifest_snpp.txt"), (1, "manifest_j1.txt")]
for si, mf in manifests:
    n_gran = 0
    with open(os.path.join(L2, mf)) as fh:
        lines = fh.readlines()
    for line in lines:
        gid = line.strip().split("\t")[0]
        nc = NCDataset(os.path.join(L2, gid))
        lat = np.ma.filled(nc.variables["lat"][:].astype(np.float64), np.nan).ravel()
        lon = np.ma.filled(nc.variables["lon"][:].astype(np.float64), np.nan).ravel()
        raw = np.ma.filled(nc.variables["obs_time_tai93"][:].astype(np.float64), np.nan).ravel()
        asc = nc.variables["asc_flag"][:]                   # (atrack,)
        h = nc.variables["h2o_vap_mol_lay"][:]             # masked (atrack, xtrack, 100)
        n_xtrack = h.shape[1]
        nc.close()

        h = np.ma.filled(h.astype(np.float64), np.nan).reshape(-1, 100)[:, p_idx]
        q = h * factor[None, :]                             # (N, NLEV) kg/kg

        asc_per = np.repeat(asc, n_xtrack)
        i = np.round(lat + 89.5).astype(np.int32)
        j = np.round(lon + 179.5).astype(np.int32)
        inb = (i >= 0) & (i < 180) & (j >= 0) & (j < 360)

        # observation time -> seconds since REF
        t = np.full(raw.shape, np.nan, dtype=np.float64)
        v = np.isfinite(raw)
        t[v] = (EPOCH + np.round(raw[v] * 1e9).astype(np.int64).astype("timedelta64[ns]")
                - REF) / np.timedelta64(1, "s")

        for p in (0, 1):
            m = inb & np.isfinite(t) & (asc_per == p)
            if m.any():
                np.add.at(obs_sum[si, p], (i[m], j[m]), t[m])
                np.add.at(cnt[si, p], (i[m], j[m]), 1)
            for k in range(NLEV):
                mq = m & np.isfinite(q[:, k])
                if mq.any():
                    np.add.at(q_sum[si, p, k], (i[mq], j[mq]), q[mq, k])
                    np.add.at(q_cnt[si, p, k], (i[mq], j[mq]), 1)
        n_gran += 1
    print(f"  processed {n_gran} granules for {'SNPP' if si == 0 else 'J1'}", flush=True)

# --- finalize ---------------------------------------------------------------
obs = np.full((2, 2, 180, 360), np.datetime64("NaT"), dtype="datetime64[ns]")
mean_t = np.full((2, 2), np.datetime64("NaT"), dtype="datetime64[ns]")
for si in range(2):
    for p in range(2):
        m = cnt[si, p] > 0
        obs[si, p][m] = REF + (obs_sum[si, p][m] / cnt[si, p][m] * 1e9).astype(np.int64).astype("timedelta64[ns]")
        mean_t[si, p] = obs[si, p][m].astype("datetime64[ns]").astype(np.int64).mean().astype("datetime64[ns]")
print("mean obs_time per pass (snpp):", mean_t[0], flush=True)

q_mean = np.full((2, 2, NLEV, 180, 360), np.nan, dtype=np.float32)
for si in range(2):
    for p in range(2):
        for k in range(NLEV):
            m = q_cnt[si, p, k] > 0
            q_mean[si, p, k][m] = (q_sum[si, p, k][m] / q_cnt[si, p, k][m]).astype(np.float32)

# quick diagnostics
print("q_mean finite fraction: %.3f" % np.isfinite(q_mean).mean(), flush=True)
print("q_mean range (kg/kg): %.4f .. %.4f" % (np.nanmin(q_mean), np.nanmax(q_mean)), flush=True)

np.savez_compressed(os.path.join(OUT, "l2_collocated_july1_noqc.npz"),
                    obs=obs, q_mean=q_mean, cnt=cnt, plev=plev)
print("saved l2_collocated_july1_noqc.npz", flush=True)
