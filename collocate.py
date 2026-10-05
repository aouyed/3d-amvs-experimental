#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collocate specific humidity (L2-derived or L3 gridded) with L2 observation times
and write them in the schema expected by src/main.py.

Humidity source is selected by config.HUMIDITY_SOURCE:
  * "l3" -> NASA L3 gridded `spec_hum` (on `air_pres_h2o`).
  * "l2" -> derived from L2 `h2o_vap_mol_lay` via to_specific_humidity().

Observation time is L2 swath FOV midtimes, binned to 1 degree (qc <= 1).

Output: data/processed/real_water_vapor_{source}_july.nc
  specific_humidity_mean  (day, plev, time, latitude, longitude, satellite)
  obs_time                (day, plev, time, latitude, longitude, satellite)
  time = ["am", "pm"]  (am = descending, pm = ascending)
"""

import glob
import os
import sys

import numpy as np
import xarray as xr
from netCDF4 import Dataset as NCDataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import config

L2_DIR = os.path.join(config.DATA_RAW, "l2")
L3_DIR = os.path.join(config.DATA_RAW, "CLIMCAPS_july")

EPOCH = np.datetime64("1993-01-01T00:00:00")
REF = np.datetime64("2020-01-01T00:00:00")

LAT = np.arange(-89.5, 90.0, 1.0)
LON = np.arange(-179.5, 180.0, 1.0)

M_H2O = 0.018015                 # kg / mol
N_A = 6.02214076e23              # molecules / mol
G = 9.80665                      # m / s^2

# pass index 0 = asc_flag 0 = descending = "am"
# pass index 1 = asc_flag 1 = ascending  = "pm"
PASS_TO_TIME = ["am", "pm"]


def map_orbit_pass(orbit_hours):
    """Map L3 `orbit_pass` (hours) to L2 `asc_flag` (pure).

    orbit_pass ~13 h -> ascending (asc_flag=1); ~1 h -> descending (asc_flag=0).
    """
    return [1 if h > 6 else 0 for h in orbit_hours]


def to_specific_humidity(h2o_vap_mol_lay, dp):
    """Convert H2O column (molecules/m2) to specific humidity (kg/kg) (pure).

    q = N_H2O * M_H2O * g / (dp * N_A), where dp is the layer pressure
    thickness in Pa. Broadcasts dp over the trailing axis of h2o_vap_mol_lay.
    """
    return h2o_vap_mol_lay * M_H2O * G / (dp * N_A)


def grid_obs_time(lat, lon, t_sec, asc):
    """Bin one granule's FOV midtimes to a 1x1 grid per pass (pure).

    Args:
        lat, lon: flattened FOV coordinates (degrees).
        t_sec: FOV midtime in seconds since REF.
        asc: flattened ascending-orbit flag (0 or 1) per FOV.

    Returns:
        obs_sum (2, 180, 360) float64, cnt (2, 180, 360) int32.
    """
    # NaN coordinates are mapped out of bounds so they are excluded cleanly.
    lat_f = np.where(np.isfinite(lat), lat, -999.0)
    lon_f = np.where(np.isfinite(lon), lon, -999.0)
    i = np.round(lat_f + 89.5).astype(np.int32)
    j = np.round(lon_f + 179.5).astype(np.int32)
    inb = (i >= 0) & (i < 180) & (j >= 0) & (j < 360) & np.isfinite(t_sec)

    obs_sum = np.zeros((2, 180, 360), dtype=np.float64)
    cnt = np.zeros((2, 180, 360), dtype=np.int32)
    for p in (0, 1):
        m = inb & (asc == p)
        if m.any():
            np.add.at(obs_sum[p], (i[m], j[m]), t_sec[m])
            np.add.at(cnt[p], (i[m], j[m]), 1)
    return obs_sum, cnt


def load_l2_obs():
    """Grid L2 FOV midtimes to 1 degree (I/O).

    Returns:
        obs (sat, pass, lat, lon) datetime64[ns].
    """
    obs_sum = np.zeros((2, 2, 180, 360), dtype=np.float64)
    cnt = np.zeros((2, 2, 180, 360), dtype=np.int32)

    manifests = [(0, "manifest_snpp.txt"), (1, "manifest_j1.txt")]
    for si, mf in manifests:
        with open(os.path.join(L2_DIR, mf)) as fh:
            lines = fh.readlines()
        for line in lines:
            gid = line.strip().split("\t")[0]
            nc = NCDataset(os.path.join(L2_DIR, gid))
            lat = nc.variables["lat"][:].astype(np.float64).ravel()
            lon = nc.variables["lon"][:].astype(np.float64).ravel()
            raw = nc.variables["obs_time_tai93"][:].astype(np.float64).ravel()
            qc = nc.variables["qc_flag"][:].ravel()
            asc = nc.variables["asc_flag"][:]
            n_xtrack = nc.variables["h2o_vap_mol_lay"][:].shape[1]
            nc.close()

            t = np.full(raw.shape, np.nan, dtype=np.float64)
            v = np.isfinite(raw)
            t[v] = (EPOCH + np.round(raw[v] * 1e9).astype(np.int64).astype("timedelta64[ns]")
                    - REF) / np.timedelta64(1, "s")
            asc_per = np.repeat(asc, n_xtrack)

            ok = qc <= 1
            o, c = grid_obs_time(lat[ok], lon[ok], t[ok], asc_per[ok])
            obs_sum[si] += o
            cnt[si] += c

    obs = np.full((2, 2, 180, 360), np.datetime64("NaT"), dtype="datetime64[ns]")
    for si in range(2):
        for p in range(2):
            m = cnt[si, p] > 0
            obs[si, p][m] = REF + (obs_sum[si, p][m] / cnt[si, p][m] * 1e9).astype(np.int64).astype("timedelta64[ns]")
    return obs


def load_l3_spec_hum(cutoff_pa):
    """Load L3 gridded `spec_hum` on levels >= cutoff (I/O).

    Returns:
        q_mean (sat, pass, level, lat, lon) kg/kg, plev (level,) hPa.
    """
    files = glob.glob(os.path.join(L3_DIR, "*L3*.nc"))
    assert len(files) == 2, f"expected 2 L3 files, found {len(files)}"
    # Order matters: satellite index 0 must be SNPP, index 1 must be J1
    # (matches the L2 manifests and the retrieval's snpp/j1 labels).
    snpp_f = [f for f in files if "SNPP" in os.path.basename(f)][0]
    j1_f = [f for f in files if "J1" in os.path.basename(f)][0]
    l3 = [xr.open_dataset(snpp_f), xr.open_dataset(j1_f)]

    orbit_h = l3[0].orbit_pass.values.astype("timedelta64[h]").astype(int)
    map_l3_to_l2 = map_orbit_pass(orbit_h)

    ap_h2o = l3[0].air_pres_h2o.values
    p_idx = np.where(ap_h2o >= cutoff_pa)[0]
    plev = ap_h2o[p_idx] / 100.0
    nlev = len(p_idx)

    if l3[0].lat.values[0] > l3[0].lat.values[-1]:
        l3 = [x.reindex(lat=list(reversed(x.lat))) for x in l3]

    q_mean = np.full((2, 2, nlev, 180, 360), np.nan, dtype=np.float32)
    for si in range(2):
        sh = l3[si].spec_hum
        for l3_idx in range(2):
            p = map_l3_to_l2[l3_idx]
            q_mean[si, p] = sh.isel(orbit_pass=l3_idx).isel(air_pres_h2o=p_idx).values
        l3[si].close()

    return q_mean, plev


def load_l2_spec_hum(cutoff_pa):
    """Derive specific humidity from L2 `h2o_vap_mol_lay` and grid to 1 deg (I/O).

    Returns:
        q_mean (sat, pass, level, lat, lon) kg/kg, plev (level,) hPa.
    """
    # Fixed L2 layer geometry (identical for every granule).
    with open(os.path.join(L2_DIR, "manifest_j1.txt")) as fh:
        ref_gid = fh.readline().strip().split("\t")[0]
    ref = NCDataset(os.path.join(L2_DIR, ref_gid))
    air_pres_lay = ref.variables["air_pres_lay"][:]
    bnds = ref.variables["air_pres_lay_bnds"][:]
    ref.close()

    dp = bnds[:, 1] - bnds[:, 0]                       # Pa per layer
    p_idx = np.where(air_pres_lay >= cutoff_pa)[0]
    plev = air_pres_lay[p_idx] / 100.0                 # hPa
    nlev = len(p_idx)

    q_sum = np.zeros((2, 2, nlev, 180, 360), dtype=np.float64)
    q_cnt = np.zeros((2, 2, nlev, 180, 360), dtype=np.int32)

    manifests = [(0, "manifest_snpp.txt"), (1, "manifest_j1.txt")]
    for si, mf in manifests:
        with open(os.path.join(L2_DIR, mf)) as fh:
            lines = fh.readlines()
        for line in lines:
            gid = line.strip().split("\t")[0]
            nc = NCDataset(os.path.join(L2_DIR, gid))
            lat = nc.variables["lat"][:].astype(np.float64).ravel()
            lon = nc.variables["lon"][:].astype(np.float64).ravel()
            qc = nc.variables["qc_flag"][:].ravel()
            asc = nc.variables["asc_flag"][:]
            h = nc.variables["h2o_vap_mol_lay"][:]
            n_xtrack = h.shape[1]
            nc.close()

            h = np.ma.filled(h.astype(np.float64), np.nan).reshape(-1, 100)[:, p_idx]
            q = to_specific_humidity(h, dp[p_idx])      # (N, nlev) kg/kg

            asc_per = np.repeat(asc, n_xtrack)
            lat_f = np.where(np.isfinite(lat), lat, -999.0)
            lon_f = np.where(np.isfinite(lon), lon, -999.0)
            i = np.round(lat_f + 89.5).astype(np.int32)
            j = np.round(lon_f + 179.5).astype(np.int32)
            inb = (i >= 0) & (i < 180) & (j >= 0) & (j < 360)
            ok = inb & (qc <= 1)

            for p in (0, 1):
                m = ok & (asc_per == p)
                if m.any():
                    for k in range(nlev):
                        mq = m & np.isfinite(q[:, k])
                        if mq.any():
                            np.add.at(q_sum[si, p, k], (i[mq], j[mq]), q[mq, k])
                            np.add.at(q_cnt[si, p, k], (i[mq], j[mq]), 1)

    q_mean = np.full((2, 2, nlev, 180, 360), np.nan, dtype=np.float32)
    for si in range(2):
        for p in range(2):
            for k in range(nlev):
                m = q_cnt[si, p, k] > 0
                q_mean[si, p, k][m] = (q_sum[si, p, k][m] / q_cnt[si, p, k][m]).astype(np.float32)

    return q_mean, plev


def write_input_nc(q_mean, obs, plev, path, lat):
    """Write the authors'-schema input NetCDF (I/O)."""
    nlev = len(plev)
    nlat = len(lat)

    q = np.empty((nlev, 2, nlat, 360, 2), dtype=np.float32)
    obs_arr = np.empty((nlev, 2, nlat, 360, 2), dtype="datetime64[ns]")
    for p in range(2):
        q[:, p] = q_mean[:, p].transpose(1, 2, 3, 0)          # (level, lat, lon, sat)
        ob = np.stack([obs[0, p], obs[1, p]], axis=-1)        # (lat, lon, sat)
        obs_arr[:, p] = ob[None, :, :, :]

    ds = xr.Dataset(
        {
            "specific_humidity_mean": (["plev", "time", "latitude", "longitude", "satellite"], q),
            "obs_time": (["plev", "time", "latitude", "longitude", "satellite"], obs_arr),
        },
        coords={
            "plev": plev,
            "time": PASS_TO_TIME,
            "latitude": lat,
            "longitude": LON,
            "satellite": ["snpp", "j1"],
        },
    )
    ds = ds.expand_dims("day").assign_coords(day=[np.datetime64("2020-07-01")])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ds.to_netcdf(path)
    print(f"wrote {path}", flush=True)


def main():
    loaders = {"l3": load_l3_spec_hum, "l2": load_l2_spec_hum}
    if config.HUMIDITY_SOURCE not in loaders:
        raise ValueError(f"unknown HUMIDITY_SOURCE: {config.HUMIDITY_SOURCE!r}")
    q_mean, plev = loaders[config.HUMIDITY_SOURCE](config.H2O_CUTOFF_PA)
    obs = load_l2_obs()
    lat_idx = np.where(np.abs(LAT) <= config.LAT_BOUND)[0]
    lat = LAT[lat_idx]
    q_mean = q_mean[:, :, :, lat_idx, :]
    obs = obs[:, :, lat_idx, :]
    write_input_nc(q_mean, obs, plev, config.INPUT_FILE, lat)


if __name__ == "__main__":
    main()
