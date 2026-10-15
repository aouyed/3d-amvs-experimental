#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Swath/time collocation for the 3D-AMV retrieval (pure functions, no I/O).

Extracted verbatim from amv_calculators.swath_initializer / prepare_patch so the
collocation step can be unit-tested without NetCDF data. Behaviour is unchanged.
"""

import numpy as np
import xarray as xr

import config

LABEL = "specific_humidity_mean"


def swath_intervals(obs_time, dmins, swath_hours):
    """Return list of contiguous [start, end] windows covering the swath.

    Args:
        obs_time: 1-D datetime64 array (may contain NaT).
        dmins: window width in minutes.
        swath_hours: total swath duration in hours.
    """
    number = (swath_hours * 60) / dmins
    mind = np.nanmin(obs_time)
    times = np.arange(dmins, dmins * number, dmins)
    intervals = [[mind, mind + np.timedelta64(dmins, "m")]]
    for time in times:
        time = int(round(time))
        intervals.append(
            [mind + np.timedelta64(time, "m"), mind + np.timedelta64(time + dmins, "m")]
        )
    return intervals


def _to_patch_dataframe(ds_unit):
    """Convert a patch unit to a DataFrame with satellite in the index."""
    return (
        ds_unit.to_dataframe()
        .dropna(subset=[LABEL])
        .set_index("satellite", append=True)
    )


def _attach_era5(ds_merged, ds_model, model_t):
    """Interpolate ERA5 onto the patch cells and merge."""
    ds_model = ds_model.sel(
        time=model_t,
        latitude=ds_merged["latitude"].values,
        longitude=ds_merged["longitude"].values,
        method="nearest",
    )
    ds_model["latitude"] = ds_merged["latitude"].values
    ds_model["longitude"] = ds_merged["longitude"].values
    ds_model = ds_model.drop("time")
    return xr.merge([ds_merged, ds_model])


def collocate_patch(ds_snpp, ds_j1, ds_model, start, end,
                    snpp_offset_min=config.SNPP_OFFSET_MIN,
                    model_midtime_min=config.MODEL_MIDTIME_MIN):
    """Select the overlapping 5-min patch and attach ERA5.

    J1 kept in [start, end]; SNPP kept in [start+offset, end+offset]; only
    cells valid in *both* satellites survive. Returns
    (ds_merged, ds_snpp, ds_j1, df_snpp).
    """
    ds_j1 = ds_j1.where((ds_j1.obs_time >= start) & (ds_j1.obs_time <= end))
    model_t = start + np.timedelta64(model_midtime_min, "m")
    snpp_start = start + np.timedelta64(snpp_offset_min, "m")
    snpp_end = end + np.timedelta64(snpp_offset_min, "m")
    ds_snpp = ds_snpp.where((ds_snpp.obs_time >= snpp_start) & (ds_snpp.obs_time <= snpp_end))

    cond1 = xr.ufuncs.logical_not(xr.ufuncs.isnan(ds_j1["obs_time"]))
    cond2 = xr.ufuncs.logical_not(xr.ufuncs.isnan(ds_snpp["obs_time"]))
    ds_j1_unit = ds_j1[[LABEL, "obs_time"]].where(cond1 & cond2)
    ds_snpp_unit = ds_snpp[[LABEL, "obs_time"]].where(cond1 & cond2)

    df_j1 = _to_patch_dataframe(ds_j1_unit)
    df_snpp = _to_patch_dataframe(ds_snpp_unit)
    ds_j1 = xr.Dataset.from_dataframe(df_j1)
    ds_snpp = xr.Dataset.from_dataframe(df_snpp)
    ds_merged = _attach_era5(xr.merge([ds_j1, ds_snpp]), ds_model, model_t)

    return ds_merged, ds_snpp, ds_j1, df_snpp
