#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download ERA5 hourly u/v winds for July 1 2020 (CDS API)."""

import os

import cdsapi

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw", "era5")
os.makedirs(OUT_DIR, exist_ok=True)

c = cdsapi.Client()
c.retrieve(
    "reanalysis-era5-pressure-levels",
    {
        "product_type": "reanalysis",
        "variable": ["u_component_of_wind", "v_component_of_wind"],
        "pressure_level": ["850", "700", "500", "400"],
        "year": "2020",
        "month": "07",
        "day": "01",
        "time": [f"{h:02d}:00" for h in range(24)],
        "format": "netcdf",
    },
    os.path.join(OUT_DIR, "era5_uv_20200701.nc"),
)
print("done")
