import os
from datetime import datetime

import numpy as np

MONTH = datetime(2020, 7, 1)
month_string = MONTH.strftime("%B").lower()
QC = "noqc"
THRESHOLDS = [100]

# Repository-relative paths (robust to the current working directory).
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
DATA_RAW = os.path.join(DATA_DIR, "raw")
DATA_PROCESSED = os.path.join(DATA_DIR, "processed")
DATA_INTERIM = os.path.join(DATA_DIR, "interim")

# Humidity source: "l3" (NASA gridded spec_hum) or "l2" (derived from L2 h2o_vap_mol_lay).
HUMIDITY_SOURCE = "l3"
# Input written by collocate.py: specific_humidity_mean + obs_time (unified schema).
INPUT_FILE = os.path.join(DATA_PROCESSED, f"real_water_vapor_{HUMIDITY_SOURCE}_july.nc")
# Keep humidity layers with mid-pressure >= 100 hPa.
H2O_CUTOFF_PA = 10000.0
# Restrict wind vectors to |latitude| <= LAT_BOUND (degrees) for error metrics.
LAT_BOUND = 60.0

# Retrieval geometry / timing (shared by main.py and amv_calculators.py).
EARTH_RADIUS = 6371000.0
DEG_TO_RAD = np.deg2rad(1)
SCALE_DEG = EARTH_RADIUS * DEG_TO_RAD      # meters per degree (latitude / equator lon)
SWATH_MINUTES = 5
SWATH_HOURS = 24
SNPP_OFFSET_MIN = 50
MODEL_MIDTIME_MIN = 25
MIN_PATCH_PIXELS = 100
# GEODESICS={'swath':[(-47.5, -60), (45, -30),'latitude'],
# 'equator':[(6.5, -149.5),(6.5, 4.5),'longitude']}
# GEODESICS={'swath':[(-50, -130), (46, -91),'latitude'],
#          'equator':[(6.5, -149.5),(6.5, 4.5),'longitude']}

# GEODESICS={'swath':[(-35.4, 28.75), (49.74, 66.13),'latitude'],
#          'equator':[(-36.5, -127.7),(55.5, -95.5),'latitude']}
GEODESICS = {
    "swath": [(-52.4, -127.6), (46.3, -89.4), "latitude"],
    "equator": [(-36.5, -127.7), (55.5, -95.5), "latitude"],
}
PRESSURES = [850, 700, 500, 400]
# TAG='full_thick_plev_tlv1'
TAG = "filtered_thick_plev_tlv1"
# TAG='filtered_4_thick_plev_tlv1'

# TAG='full_nn_tlv1'
