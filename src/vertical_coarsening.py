#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 15:58:19 2022

@author: aouyed
"""
import os
import sys

import xarray as xr
import pandas as pd
import numpy as np
import config
import datetime
from tqdm import tqdm
from parameters import parameters


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


def main(param, orbit=None):
    dates = param.dates
    orbits = [orbit] if orbit is not None else ("am", "pm")

    for date in tqdm(dates):
        date_string = date.strftime("%m_%d_%Y")
        thresh = param.thresh
        for orbit in orbits:
            raw_f = os.path.join(
                config.DATA_PROCESSED,
                f"{param.alg}_lambda_{param.Lambda}_{date_string}_{orbit}.nc",
            )
            ds = xr.open_dataset(raw_f)
            ds = vertical_coarse(ds, thresh, param.plev_coarse)
            out = os.path.join(
                config.DATA_PROCESSED,
                f"{param.tag}_{date_string}_{orbit}.nc",
            )
            ds.to_netcdf(out)


if __name__ == "__main__":
    orbit = sys.argv[1] if len(sys.argv) > 1 else None
    param = parameters()
    main(param, orbit)
