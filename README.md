# 3D winds retrieval (experimental branch)

Original code associated with:

> Ouyed, A., Smith, N., Zeng, X., Galarneau Jr, T., Su, H., & Dixon, R. D. (2023).
> *Global three-dimensional water vapor feature-tracking for horizontal winds using
> hyperspectral infrared sounder data from overlapped tracks of two satellites.*
> Geophysical Research Letters, 50(7), e2022GL101830.
> https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022GL101830

This is a **private experimental development repository** that mirrors
`aouyed/3d-amvs` (added as the `upstream` remote). Experimental work is committed
here and, once validated, is intended to be pushed to the public `aouyed/3d-amvs`.

## What this branch adds

Reproducible CLIMCAPS → 3D-wind pipeline for July 1, 2020 (SNPP + NOAA-20):

| Script | Purpose |
|---|---|
| `download_l2.py` | Download CLIMCAPS L2 granules (GES DISC) |
| `download_era5.py` | Download ERA5 u/v (CDS API) |
| `collocate_full_day.py` | Grid L2 obs_time + L3 `spec_hum` (QCC) to 1° |
| `collocate_l2_noqc.py` | Grid L2 `h2o_vap_mol_lay`-derived `spec_hum` to 1° **with no QC** |
| `run_retrieval_all_levels.py` | Per-level 5-min swath optical-flow retrieval (resumable) |
| `run_vertical_coarsening.py` | Gross-error filter (δ=10 m/s) + 5-level median coarsening |
| `error_analysis.py` | RMSVD / bias / speed-corr vs ERA5 |
| `plot_coarsened.py`, `plot_histograms.py` | Quiver + histogram figures |

## Key finding

The paper gridded CLIMCAPS L2 water vapor to 1° **without quality filtering**
("we did not apply any quality filtering ahead of gridding"). The published L3
products are only available QC'd (comprehensive `QCC` or specific `QCS`), so a
true no-QC field must be re-gridded from L2. Raw (unfiltered) RMSVD vs ERA5 is
~16–25 m/s; the δ=10 m/s gross-error filter + 5-level coarsening collapses this
to ~6 m/s, which is therefore a post-filter artifact rather than raw skill.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Dependencies & reproducibility

`requirements.txt` lists the **direct** dependencies with **pinned versions**
(`==`) so installs are repeatable. This follows pip's guidance that pinning
"protects you from bugs or incompatibilities in newly released versions"; the
critical pin here is `opencv-contrib-python-headless==4.10.0.84`, which provides
`cv2.optflow` (TV-L1 / DualTVL1) used by the retrieval.

`requirements.txt` intentionally omits transitive dependencies (they are
resolved automatically by pip) and developer-only tools. For a byte-for-byte
frozen environment, run:

```bash
pip freeze > requirements.lock.txt
```

Data (`data/`, `3d_amvs_old/`) and credentials (`~/.netrc`, `~/.cdsapirc`) are
gitignored and must be obtained/provided locally — never committed.
