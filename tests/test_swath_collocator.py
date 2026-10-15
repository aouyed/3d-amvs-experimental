#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for swath_collocator (interval builder + patch collocation)."""

import os
import sys

import numpy as np
import xarray as xr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import swath_collocator as sc  # noqa: E402

START = np.datetime64("2020-07-01T00:00:00")


def _sat_ds(hum, obs_time, sat="snpp"):
    """Build a single-satellite dataset on a small lat/lon grid.

    Mirrors ds_unit.loc[{"satellite": s}] in the retrieval loop: no satellite
    dimension, but a scalar "satellite" coordinate (so to_dataframe() exposes a
    "satellite" column for _to_patch_dataframe's set_index).
    """
    lat = np.array([-1.0, 0.0, 1.0])
    lon = np.array([-1.0, 0.0, 1.0])
    return xr.Dataset(
        {"specific_humidity_mean": (["latitude", "longitude"], hum),
         "obs_time": (["latitude", "longitude"], obs_time)},
        coords={"latitude": lat, "longitude": lon, "satellite": sat},
    )


def _model_ds():
    lat = np.array([-1.0, 0.0, 1.0])
    lon = np.array([-1.0, 0.0, 1.0])
    time = np.array([START + np.timedelta64(25, "m")], dtype="datetime64[ns]")
    return xr.Dataset(
        {"u": (["time", "latitude", "longitude"], np.zeros((1, 3, 3))),
         "v": (["time", "latitude", "longitude"], np.zeros((1, 3, 3)))},
        coords={"time": time, "latitude": lat, "longitude": lon},
    )


def test_swath_intervals_count_and_spacing():
    obs = np.array(["2020-07-01T00:00:00", "2020-07-01T01:00:00"], dtype="datetime64[ns]")
    ints = sc.swath_intervals(obs, dmins=5, swath_hours=24)
    assert len(ints) == 288
    assert ints[0][0] == obs.min()
    assert ints[0][1] == obs.min() + np.timedelta64(5, "m")
    for a, b in zip(ints, ints[1:]):          # contiguous, no gaps/overlap
        assert a[1] == b[0]


def test_swath_intervals_ignores_nat():
    obs = np.array(["NaT", "2020-07-01T03:00:00"], dtype="datetime64[ns]")
    ints = sc.swath_intervals(obs, dmins=5, swath_hours=1)
    assert ints[0][0] == np.datetime64("2020-07-01T03:00:00")


def test_collocate_patch_selects_window_and_applies_snpp_offset():
    hum = np.ones((3, 3))
    end = START + np.timedelta64(5, "m")

    t_j1 = np.full((3, 3), np.datetime64("NaT"), dtype="datetime64[ns]")
    t_j1[0, 0] = START + np.timedelta64(1, "m")
    t_j1[1, 1] = START + np.timedelta64(2, "m")

    t_snpp = np.full((3, 3), np.datetime64("NaT"), dtype="datetime64[ns]")
    t_snpp[0, 0] = START + np.timedelta64(50, "m")   # only in-window cells
    t_snpp[1, 1] = START + np.timedelta64(51, "m")

    ds_merged, _, _, df_snpp = sc.collocate_patch(
        _sat_ds(hum, t_snpp, sat="snpp"), _sat_ds(hum, t_j1, sat="j1"), _model_ds(), START, end
    )

    assert df_snpp.shape[0] == 2                  # exactly the two overlapping cells
    idx = df_snpp.index.droplevel("satellite")
    assert set(map(tuple, idx.values)) == {(-1.0, -1.0), (0.0, 0.0)}
    assert "u" in ds_merged and "v" in ds_merged  # ERA5 attached


def test_collocate_patch_drops_mismatched_satellite_cells():
    hum = np.ones((3, 3))
    end = START + np.timedelta64(5, "m")
    t_j1 = np.full((3, 3), np.datetime64("NaT"), dtype="datetime64[ns]")
    t_j1[0, 0] = START + np.timedelta64(1, "m")
    t_snpp = np.full((3, 3), np.datetime64("NaT"), dtype="datetime64[ns]")
    t_snpp[0, 0] = START + np.timedelta64(49, "m")   # outside +50min window

    _, _, _, df_snpp = sc.collocate_patch(
        _sat_ds(hum, t_snpp, sat="snpp"), _sat_ds(hum, t_j1, sat="j1"), _model_ds(), START, end
    )
    assert df_snpp.shape[0] == 0
