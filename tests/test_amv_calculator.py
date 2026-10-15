#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Smoke test: amv_calculator produces finite winds on a synthetic patch."""

import os
import sys

import numpy as np
import xarray as xr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import amv_calculators as calc  # noqa: E402
from parameters import parameters  # noqa: E402


def test_amv_calculator_finite_u_v():
    n = 20
    lat = np.linspace(-10.0, 10.0, n)
    lon = np.linspace(-10.0, 10.0, n)
    yy, xx = np.mgrid[0:n, 0:n]

    # smooth, shifted humidity fields so optical flow has structure
    hum_j1 = (0.01 + 0.0001 * (xx + yy)).astype(np.float32)
    hum_snpp = (0.01 + 0.0001 * (xx + 2.0 + yy)).astype(np.float32)
    hum = np.stack([hum_j1, hum_snpp], axis=-1)  # satellite order: j1, snpp

    t0 = np.datetime64("2020-07-01T00:00:00")
    obs_j1 = np.full((n, n), t0, dtype="datetime64[ns]")
    obs_snpp = np.full((n, n), t0 + np.timedelta64(50, "m"), dtype="datetime64[ns]")
    obs = np.stack([obs_j1, obs_snpp], axis=-1)

    ds_unit = xr.Dataset(
        {"specific_humidity_mean": (["latitude", "longitude", "satellite"], hum),
         "obs_time": (["latitude", "longitude", "satellite"], obs)},
        coords={"latitude": lat, "longitude": lon, "satellite": ["j1", "snpp"]},
    )
    ds_unit = calc.prepare_ds(ds_unit)
    df = ds_unit.to_dataframe()

    # merged patch with ERA5 winds as (lat, lon), mirroring _attach_era5
    ds_merged = xr.Dataset(
        {"specific_humidity_mean": (["latitude", "longitude", "satellite"], hum),
         "obs_time": (["latitude", "longitude", "satellite"], obs),
         "u": (["latitude", "longitude"], np.zeros((n, n), dtype=np.float32)),
         "v": (["latitude", "longitude"], np.zeros((n, n), dtype=np.float32))},
        coords={"latitude": lat, "longitude": lon, "satellite": ["j1", "snpp"]},
    )

    param = parameters()
    param.set_alg("tvl1")
    df, _, _, _ = calc.amv_calculator(ds_merged, df, param)

    assert np.isfinite(df["u"].values).any()
    assert np.isfinite(df["v"].values).any()
