import os

import numpy as np

# Repository-relative paths (robust to the current working directory).
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
DATA_PROCESSED = os.path.join(DATA_DIR, "processed")

# Retrieval geometry / timing.
EARTH_RADIUS = 6371000.0
DEG_TO_RAD = np.deg2rad(1)
SCALE_DEG = EARTH_RADIUS * DEG_TO_RAD      # meters per degree (latitude / equator lon)
SWATH_MINUTES = 5
SWATH_HOURS = 24
SNPP_OFFSET_MIN = 50
MODEL_MIDTIME_MIN = 25
MIN_PATCH_PIXELS = 100

# Vertical coarsening.
COARSEN_THRESH = 10.0
COARSEN_LAYERS = 5
