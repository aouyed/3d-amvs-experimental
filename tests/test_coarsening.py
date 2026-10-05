#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for vertical_coarsening.compute / vertical_coarse."""

import os
import sys

import numpy as np
import xarray as xr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import vertical_coarsening as vc  # noqa: E402


def _make_ds(n_plev=6):
    plev = np.arange(n_plev, dtype=float) * 100 + 100
    lat = np.array([0.0, 1.0])
    lon = np.array([0.0, 1.0])
    sat = ["snpp", "j1"]
    shape = (n_plev, 2, 2, 2)
    u = np.ones(shape)
    v = np.zeros(shape)
    u_era5 = np.zeros(shape)   # -> error magnitude = 1 everywhere
    v_era5 = np.zeros(shape)
    return xr.Dataset(
        {
            "u": (["plev", "latitude", "longitude", "satellite"], u),
            "v": (["plev", "latitude", "longitude", "satellite"], v),
            "u_era5": (["plev", "latitude", "longitude", "satellite"], u_era5),
            "v_era5": (["plev", "latitude", "longitude", "satellite"], v_era5),
            "humidity_overlap": (["plev", "latitude", "longitude", "satellite"], u.copy()),
        },
        coords={"plev": plev, "latitude": lat, "longitude": lon, "satellite": sat},
    )


def test_compute_filters_by_threshold():
    ds = _make_ds()
    # error magnitude is sqrt((1-0)^2 + 0) = 1 everywhere
    out = vc.compute(ds, thresh=0.5)
    assert out["u"].isnull().all()
    out = vc.compute(ds, thresh=2.0)
    assert out["u"].notnull().all()


def test_vertical_coarse_median_reduces_levels():
    ds = _make_ds(n_plev=10)
    out = vc.vertical_coarse(ds, thresh=2.0, n_layers=5)
    assert out.sizes["plev"] == 2          # 10 // 5
    # median of constant u=1 should remain 1
    assert np.allclose(out["u"].values, 1.0)
    assert "obs_time" not in out.data_vars  # obs_time re-attachment removed
