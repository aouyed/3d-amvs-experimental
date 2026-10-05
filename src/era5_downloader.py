#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stub replacement for the removed ``era5_downloader`` module.

The original module downloaded and coarsened ERA5 reanalysis fields into
``data/interim/coarse_model_MM_DD_YYYY.nc`` (u / v on pressure levels) which
``main.model_loader()`` reads back for the ERA5 "truth" comparison and for the
model reference times.

This stub exists so that ``import era5_downloader`` (used at the top of
``main.py``) succeeds. It does not perform any real data acquisition.
"""

import warnings


def downloader(day, **kwargs):
    """Raise a clear error instead of silently doing nothing."""
    raise NotImplementedError(
        "ERA5 downloading is not implemented in this stub. "
        "Provide data/interim/coarse_model_*.nc files manually, or restore the "
        "real era5_downloader implementation."
    )
