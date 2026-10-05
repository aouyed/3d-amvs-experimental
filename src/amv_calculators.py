#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 15:17:31 2021

@author: aouyed
"""
import cv2
import xarray as xr
import numpy as np
import inpainter

import config


LABEL = "specific_humidity_mean"


def optical_flow(frame0, frame, alg):
    """Compute optical flow between two frames (cv2 wrapper).

    Args:
        frame0 (numpy.ndarray): First frame.
        frame (numpy.ndarray): Second frame.
        alg (str): 'deepflow' or 'tvl1'.

    Returns:
        tuple: (flowx, flowy) pixel displacements.
    """
    if frame0.shape != frame.shape:
        frame = np.resize(frame, frame0.shape)

    nframe0 = cv2.normalize(src=frame0, dst=None, alpha=0, beta=255,
                            norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    nframe = cv2.normalize(src=frame, dst=None, alpha=0, beta=255,
                           norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    if alg == "deepflow":
        of = cv2.optflow.createOptFlow_DeepFlow()
    else:
        of = cv2.optflow.DualTVL1OpticalFlow_create()

    flowd = of.calc(nframe0, nframe, None)
    return flowd[:, :, 0], flowd[:, :, 1]


def frame_retreiver_ns(ds):
    """
    Retrieve and process a frame from the dataset using inpainter.

    Args:
        ds (xarray.Dataset): The dataset containing the frame.

    Returns:
        numpy.ndarray: The processed frame.
    """
    frame = np.squeeze(ds[LABEL].values)
    frame = inpainter.drop_nan(frame)
    return frame


def prepare_ds(ds):
    """
    Prepare the dataset by adding empty variables for calculations.

    Args:
        ds (xarray.Dataset): The dataset to prepare.

    Returns:
        xarray.Dataset: The prepared dataset.
    """
    ds["humidity_overlap"] = xr.full_like(
        ds["specific_humidity_mean"], fill_value=np.nan
    )
    ds["flowx"] = xr.full_like(ds["specific_humidity_mean"], fill_value=np.nan)
    ds["flowy"] = xr.full_like(ds["specific_humidity_mean"], fill_value=np.nan)
    ds["u"] = xr.full_like(ds["specific_humidity_mean"], fill_value=np.nan)
    ds["v"] = xr.full_like(ds["specific_humidity_mean"], fill_value=np.nan)
    ds["u_era5"] = xr.full_like(ds["specific_humidity_mean"], fill_value=np.nan)
    ds["v_era5"] = xr.full_like(ds["specific_humidity_mean"], fill_value=np.nan)
    ds["dt_inv"] = xr.full_like(ds["specific_humidity_mean"], fill_value=np.nan)

    return ds


def swath_initializer(ds, dmins, swath_hours):
    """
    Initialize swath time intervals.

    Args:
        ds (xarray.Dataset): The dataset containing observation times.
        dmins (int): Interval in minutes.
        swath_hours (int): Total swath duration in hours.

    Returns:
        list: List of swath time intervals.
    """
    number = (swath_hours * 60) / dmins
    mind = np.nanmin(ds["obs_time"].values)
    times = np.arange(dmins, dmins * number, dmins)
    swathes = []
    swathes.append([mind, mind + np.timedelta64(dmins, "m")])
    for time in times:
        time = int(round(time))
        swathes.append(
            [mind + np.timedelta64(time, "m"), mind + np.timedelta64(time + dmins, "m")]
        )
    return swathes


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


def prepare_patch(ds_snpp, ds_j1, ds_model, start, end):
    """Select the overlapping 5-min patch and attach ERA5.

    Returns:
        ds_merged, ds_snpp, ds_j1, df_snpp
    """
    ds_j1 = ds_j1.where((ds_j1.obs_time >= start) & (ds_j1.obs_time <= end))
    model_t = start + np.timedelta64(config.MODEL_MIDTIME_MIN, "m")
    snpp_start = start + np.timedelta64(config.SNPP_OFFSET_MIN, "m")
    snpp_end = end + np.timedelta64(config.SNPP_OFFSET_MIN, "m")
    ds_snpp = ds_snpp.where((ds_snpp.obs_time >= snpp_start) & (ds_snpp.obs_time <= snpp_end))

    cond1 = xr.ufuncs.logical_not(xr.ufuncs.isnan(ds_j1["obs_time"]))
    cond2 = xr.ufuncs.logical_not(xr.ufuncs.isnan(ds_snpp["obs_time"]))
    ds_j1_unit = ds_j1[["specific_humidity_mean", "obs_time"]].where(cond1 & cond2)
    ds_snpp_unit = ds_snpp[["specific_humidity_mean", "obs_time"]].where(cond1 & cond2)

    df_j1 = _to_patch_dataframe(ds_j1_unit)
    df_snpp = _to_patch_dataframe(ds_snpp_unit)
    ds_j1 = xr.Dataset.from_dataframe(df_j1)
    ds_snpp = xr.Dataset.from_dataframe(df_snpp)
    ds_merged = _attach_era5(xr.merge([ds_j1, ds_snpp]), ds_model, model_t)

    return ds_merged, ds_snpp, ds_j1, df_snpp


def flow_calculator(ds_snpp, ds_j1, ds_merged, param):
    """
    Calculate flow components and update datasets.

    Args:
        ds_snpp (xarray.Dataset): SNPP dataset.
        ds_j1 (xarray.Dataset): J1 dataset.
        ds_merged (xarray.Dataset): Merged dataset.
        param (parameters): Parameters object.

    Returns:
        tuple: Updated SNPP and J1 datasets.
    """
    frame0 = frame_retreiver_ns(ds_j1)
    frame = frame_retreiver_ns(ds_snpp)
    flowx, flowy = optical_flow(frame0, frame, param.alg)
    flowx = np.expand_dims(flowx, axis=2)
    flowy = np.expand_dims(flowy, axis=2)
    ds_snpp["flowx"] = (["latitude", "longitude", "satellite"], flowx)
    ds_snpp["flowy"] = (["latitude", "longitude", "satellite"], flowy)
    ds_j1["flowx"] = (["latitude", "longitude", "satellite"], flowx)
    ds_j1["flowy"] = (["latitude", "longitude", "satellite"], flowy)
    frame = np.expand_dims(frame, axis=2)
    frame0 = np.expand_dims(frame0, axis=2)
    dt = (
        ds_merged["obs_time"].loc[{"satellite": "snpp"}]
        - ds_merged["obs_time"].loc[{"satellite": "j1"}]
    )
    dt_int = dt.values.astype("timedelta64[s]").astype(np.int32)
    with np.errstate(divide="ignore", invalid="ignore"):
        dt_inv = 1.0 / dt_int
    dt_inv[~np.isfinite(dt_inv)] = np.nan
    dx_conv = abs(np.cos(np.deg2rad(ds_merged.latitude)))
    ds_merged["dt_inv"] = (["latitude", "longitude"], dt_inv)

    ds_snpp["dt_inv"] = ds_merged["dt_inv"]
    ds_snpp["u"] = config.SCALE_DEG * dx_conv * ds_merged["dt_inv"] * ds_snpp["flowx"]
    ds_snpp["v"] = config.SCALE_DEG * ds_merged["dt_inv"] * ds_snpp["flowy"]

    ds_j1["dt_inv"] = ds_merged["dt_inv"]
    ds_j1["u"] = config.SCALE_DEG * dx_conv * ds_merged["dt_inv"] * ds_j1["flowx"]
    ds_j1["v"] = config.SCALE_DEG * ds_merged["dt_inv"] * ds_j1["flowy"]
    return ds_snpp, ds_j1


def fill_winds(df, df_sat):
    """
    Fill wind/flow columns in `df` from satellite data in place.

    Args:
        df (pandas.DataFrame): DataFrame to fill (mutated in place).
        df_sat (pandas.DataFrame): Satellite data.
    """
    mask = df.index.isin(df_sat.index)
    df.loc[mask, "humidity_overlap"] = df_sat["specific_humidity_mean"]
    df.loc[mask, "flowx"] = df_sat["flowx"]
    df.loc[mask, "flowy"] = df_sat["flowy"]
    df.loc[mask, "u"] = df_sat["u"]
    df.loc[mask, "v"] = df_sat["v"]
    df.loc[mask, "dt_inv"] = df_sat["dt_inv"]


def fill_era5(df, df_sat, df_m):
    """
    Fill ERA5 wind columns in `df` in place.

    Args:
        df (pandas.DataFrame): DataFrame to fill (mutated in place).
        df_sat (pandas.DataFrame): Satellite data.
        df_m (pandas.DataFrame): Model data.
    """
    mask = df.index.isin(df_sat.index)
    df.loc[mask, "u_era5"] = df_m["u"]
    df.loc[mask, "v_era5"] = df_m["v"]


def _compute_flow(ds_merged, param):
    """Compute optical flow and u/v for both satellites on a patch."""
    ds_snpp = ds_merged.loc[{"satellite": "snpp"}].expand_dims("satellite")
    ds_j1 = ds_merged.loc[{"satellite": "j1"}].expand_dims("satellite")
    return flow_calculator(ds_snpp, ds_j1, ds_merged, param)


def amv_calculator(ds_merged, df, param):
    """
    Calculate atmospheric motion vectors (AMVs).

    Args:
        ds_merged (xarray.Dataset): Merged dataset.
        df (pandas.DataFrame): DataFrame to update.
        param (parameters): Parameters object.

    Returns:
        tuple: Updated DataFrame, SNPP dataset, J1 dataset, and merged dataset.
    """
    ds_snpp, ds_j1 = _compute_flow(ds_merged, param)
    df_model = (
        ds_merged.loc[{"satellite": "snpp"}].drop("satellite").to_dataframe().dropna()
    )
    df_snpp = ds_snpp.to_dataframe().reorder_levels(
        ["latitude", "longitude", "satellite"]
    ).sort_index()
    df_j1 = ds_j1.to_dataframe().reorder_levels(
        ["latitude", "longitude", "satellite"]
    ).sort_index()
    fill_winds(df, df_snpp)
    fill_winds(df, df_j1)
    fill_era5(df, df_j1, df_model)
    fill_era5(df, df_snpp, df_model)

    return df, ds_snpp, ds_j1, ds_merged
