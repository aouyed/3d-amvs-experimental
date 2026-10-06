#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 15:58:19 2022

@author: aouyed
"""
import numpy as np


def compute(ds, thresh):
    """Filter to winds with error magnitude below `thresh` (returns a new dataset)."""
    error_mag = np.sqrt((ds["u"] - ds["u_era5"]) ** 2 + (ds["v"] - ds["v_era5"]) ** 2)
    return ds.where(error_mag < thresh)


def vertical_coarse(ds, thresh, n_layers):
    ds = ds.reindex(plev=list(reversed(ds.plev)))
    ds = compute(ds, thresh)
    ds_c = (
        ds[["u", "v", "u_era5", "v_era5", "humidity_overlap"]]
        .coarsen(plev=n_layers, boundary="trim")
        .median()
    )
    return ds_c
