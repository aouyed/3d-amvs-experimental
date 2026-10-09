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
python -m venv .venv                        # 0. create the Python environment (once)
.venv/bin/pip install -r requirements.txt   # install dependencies

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

Command `run_pipeline.py download` (and `all`) prompts interactively for whichever is missing. 

To change credentials (e.g. after an authentication failure), run `python run_pipeline.py credentials`.


