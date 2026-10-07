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

## Quick start

One command per stage, run in order (each stage skips work already done):

```bash
.venv/bin/python run_pipeline.py download   # 1. raw CLIMCAPS L2 + ERA5 (resumable)
.venv/bin/python run_pipeline.py run        # 2. collocate -> retrieval -> coarsening
.venv/bin/python run_pipeline.py analyze    # 3. error metrics + figures
```

Or everything at once:

```bash
.venv/bin/python run_pipeline.py all
```

| Command | Runs | Produces |
|---|---|---|
| `download` | `download_l2.py`, `download_era5.py` | `data/raw/l2/*`, `data/raw/era5/era5_uv_20200701.nc` |
| `run` | `collocate_full_day.py` → `run_retrieval_all_levels.py` → `run_vertical_coarsening.py` | `data/processed/l2_collocated_july1.npz` → `retrieval_ckpt.npz` → `wind_coarsened_{ascending,descending}_july1.nc` |
| `analyze` | `error_analysis.py`, `plot_quiver_all_levels.py`, `plot_histograms.py` | metrics + `data/processed/plots/*.png` |

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

### Credentials

The `download` stage needs two credential files (both are gitignored — never commit them):

- **Earthdata** — `~/.netrc`:
  ```
  machine urs.earthdata.nasa.gov
      login <username>
      password <password>
  ```
- **CDS API** — `~/.cdsapirc`:
  ```
  url: https://cds.climate.copernicus.eu/api
  key: <API_KEY>
  ```

`run_pipeline.py download` (and `all`) prompts interactively for whichever is missing. You can also run `python setup_credentials.py` manually, or `python setup_credentials.py --check` to inspect. In non-interactive/CI runs, create the files yourself — the pipeline prints instructions and exits instead of prompting.

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
