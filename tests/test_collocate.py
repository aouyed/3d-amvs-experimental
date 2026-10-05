#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for collocate.py pure functions."""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import collocate  # noqa: E402


def test_map_orbit_pass():
    assert collocate.map_orbit_pass([13, 1, 14, 0]) == [1, 0, 1, 0]
    assert collocate.map_orbit_pass([6]) == [0]
    assert collocate.map_orbit_pass([7]) == [1]


def test_grid_obs_time_binning():
    lat = np.array([-0.5, 0.5, 1.5])
    lon = np.array([-0.5, 0.5, 1.5])
    t_sec = np.array([100.0, 200.0, 300.0])
    asc = np.array([0, 0, 1])

    obs_sum, cnt = collocate.grid_obs_time(lat, lon, t_sec, asc)

    # FOV0: lat index 89, lon index 179, pass 0
    assert obs_sum[0, 89, 179] == 100.0
    assert cnt[0, 89, 179] == 1
    # FOV1: lat index 90, lon index 180, pass 0
    assert obs_sum[0, 90, 180] == 200.0
    assert cnt[0, 90, 180] == 1
    # FOV2: lat index 91, lon index 181, pass 1
    assert obs_sum[1, 91, 181] == 300.0
    assert cnt[1, 91, 181] == 1
    # nothing else accumulated
    assert cnt.sum() == 3


def test_grid_obs_time_ignores_out_of_bounds_and_nan():
    lat = np.array([-99.0, 0.0, np.nan])
    lon = np.array([0.0, 0.0, 0.0])
    t_sec = np.array([1.0, 2.0, 3.0])
    asc = np.array([0, 0, 0])

    obs_sum, cnt = collocate.grid_obs_time(lat, lon, t_sec, asc)

    # only FOV1 is in bounds and finite
    assert cnt.sum() == 1
    assert cnt[0, 90, 180] == 1


def test_to_specific_humidity():
    h = np.array([1e23, 2e23])
    dp = np.array([10000.0, 20000.0])
    q = collocate.to_specific_humidity(h, dp)
    expected = h * collocate.M_H2O * collocate.G / (dp * collocate.N_A)
    assert np.allclose(q, expected)


def test_to_specific_humidity_sanity():
    q = collocate.to_specific_humidity(np.array([1e23]), np.array([10000.0]))
    assert 0.0 < q[0] < 0.05
