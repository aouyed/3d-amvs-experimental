#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Random-vector (null model) baseline for the 3D-AMV winds.

Reproduces the null model from the public 3d-amvs repository
(src/random_vectors.py, commit d98d4bd) with the NaN-masking bug fixed, and
evaluates it with the same speed-correlation / RMSVD metric used by
error_analysis.py. A single QC threshold (10 m/s) is applied.

Reads the coarsened output wind_coarsened_{ascending,descending}_july1{suf}.nc
(single wind field per pass; the satellite dim is broadcast). Produces:
  - a per-level + overall metrics table on stdout
  - random_vectors_compare{suf}.png              (RMSVD-by-level + speed scatter)
  - quiver_random_{pass}_{lev}hPa{suf}.png       (retrieved | random | ERA5)

Usage:
    python random_vectors.py          # default (L3-fed) coarsened product
    python random_vectors.py noqc     # no-QC (L2-regridded) product
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import xarray as xr

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data", "processed")
PLOTS = os.path.join(DATA, "plots")
os.makedirs(PLOTS, exist_ok=True)

SUF = f"_{sys.argv[1]}" if len(sys.argv) > 1 else ""

PASSES = ("ascending", "descending")
THRESH = 10.0                          # single QC threshold (m/s)
QUIVER_LEVELS = [850.0, 500.0, 200.0]  # hPa (nearest coarsened level is used)
SEED = 0


def compute_metrics(u, v, u_era5, v_era5):
    """Per-level/overall metrics; mirrors error_analysis.compute_metrics."""
    m = np.isfinite(u) & np.isfinite(v) & np.isfinite(u_era5) & np.isfinite(v_era5)
    u, v, ue, ve = u[m], v[m], u_era5[m], v_era5[m]
    if len(u) == 0:
        return None
    err = np.sqrt((u - ue) ** 2 + (v - ve) ** 2)
    spd = np.sqrt(u ** 2 + v ** 2)
    spd_e = np.sqrt(ue ** 2 + ve ** 2)
    return {
        "n": int(len(u)),
        "rmsvd": float(np.sqrt(np.mean(err ** 2))),
        "u_bias": float(np.mean(u - ue)),
        "v_bias": float(np.mean(v - ve)),
        "corr": float(np.corrcoef(spd, spd_e)[0, 1]) if len(u) > 1 else np.nan,
        "u": u, "v": v, "u_era5": ue, "v_era5": ve, "spd": spd, "spd_era5": spd_e,
    }


def scalar(m):
    if m is None:
        return None
    return {k: m[k] for k in ("n", "rmsvd", "u_bias", "v_bias", "corr")}


def rmsvd(u, v, ue, ve):
    m = np.isfinite(u) & np.isfinite(v) & np.isfinite(ue) & np.isfinite(ve)
    if not m.any():
        return np.nan
    err = np.sqrt((u - ue) ** 2 + (v - ve) ** 2)
    return float(np.sqrt(np.mean(err[m] ** 2)))


def load_snpp(pname):
    f = os.path.join(DATA, f"wind_coarsened_{pname}_july1{SUF}.nc")
    return xr.open_dataset(f).sel(satellite="snpp")


def make_random_3d(ds, rng):
    """uniform(-20,20) noise masked to real coverage, then |error| < THRESH."""
    u_real = ds["u"].values
    v_real = ds["v"].values
    u_era5 = ds["u_era5"].values
    v_era5 = ds["v_era5"].values

    u_r = rng.uniform(-20.0, 20.0, size=u_real.shape)
    v_r = rng.uniform(-20.0, 20.0, size=v_real.shape)

    # corrected bug: mask coverage holes with np.isnan (was: == np.nan no-op)
    u_r[np.isnan(u_real)] = np.nan
    v_r[np.isnan(v_real)] = np.nan

    err = np.sqrt((u_r - u_era5) ** 2 + (v_r - v_era5) ** 2)
    keep = err < THRESH
    return np.where(keep, u_r, np.nan), np.where(keep, v_r, np.nan)


def real_metrics(ds):
    u = ds["u"].values
    v = ds["v"].values
    u_era5 = ds["u_era5"].values
    v_era5 = ds["v_era5"].values
    plevs = ds["plev"].values
    per_level = {pl: scalar(compute_metrics(u[i], v[i], u_era5[i], v_era5[i]))
                 for i, pl in enumerate(plevs)}
    overall = scalar(compute_metrics(u, v, u_era5, v_era5))
    return per_level, overall


def one_ax(ax, lon, lat, u, v, title, scale=240):
    ax.coastlines()
    ax.set_global()
    ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)
    Q = ax.quiver(lon, lat, u, v, transform=ccrs.PlateCarree(), scale=scale)
    ax.quiverkey(Q, 0.9, 0.95, 10, "10 m/s", labelpos="E", coordinates="figure")
    ax.set_title(title, fontsize=9)


def main():
    rng = np.random.default_rng(SEED)
    results = {}
    print("=" * 100)
    print(f"RANDOM-VECTOR NULL MODEL (thresh={THRESH:g} m/s)  suffix='{SUF or '(default)'}'")
    print("=" * 100)

    for pname in PASSES:
        ds = load_snpp(pname)
        u_r, v_r = make_random_3d(ds, rng)
        plevs = ds["plev"].values

        rl = {}
        for i, pl in enumerate(plevs):
            rl[pl] = scalar(compute_metrics(u_r[i], v_r[i],
                                            ds["u_era5"].values[i],
                                            ds["v_era5"].values[i]))
        ro = scalar(compute_metrics(u_r, v_r, ds["u_era5"].values, ds["v_era5"].values))
        pl_, po = real_metrics(ds)
        results[pname] = dict(
            plevs=plevs, random_per_level=rl, random_overall=ro,
            real_per_level=pl_, real_overall=po,
            u_random=u_r, v_random=v_r,
            u_era5=ds["u_era5"].values, v_era5=ds["v_era5"].values,
        )
        ds.close()

        print("-" * 100)
        print(f"{pname}  --  random @ {THRESH:g} m/s")
        print(f"{'plev (hPa)':>10s} {'n':>7s} {'RMSVD':>7s} {'u bias':>7s} {'v bias':>7s} {'corr(speed)':>11s}")
        for pl in plevs:
            m = rl[pl]
            if m is None or m["n"] < 50:
                continue
            print(f"{pl:10.1f} {m['n']:7d} {m['rmsvd']:7.2f} {m['u_bias']:+7.2f} {m['v_bias']:+7.2f} {m['corr']:11.3f}")
        if ro is not None:
            print(f"{'OVERALL':>10s} {ro['n']:7d} {ro['rmsvd']:7.2f} {ro['u_bias']:+7.2f} {ro['v_bias']:+7.2f} {ro['corr']:11.3f}")
        if po is not None:
            print(f"{'REAL overall':>10s} {po['n']:7d} {po['rmsvd']:7.2f} {po['u_bias']:+7.2f} {po['v_bias']:+7.2f} {po['corr']:11.3f}")

    # combined random (both passes) for the scatter + summary
    u_all = np.concatenate([results[p]["u_random"] for p in PASSES])
    v_all = np.concatenate([results[p]["v_random"] for p in PASSES])
    ue_all = np.concatenate([results[p]["u_era5"] for p in PASSES])
    ve_all = np.concatenate([results[p]["v_era5"] for p in PASSES])
    comb = compute_metrics(u_all, v_all, ue_all, ve_all)
    print("=" * 100)
    print(f"RANDOM combined: n={comb['n']} RMSVD={comb['rmsvd']:.2f} m/s speed-corr={comb['corr']:.3f}")

    # ---- compare figure ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ax = axes[0]
    for pname in PASSES:
        r = results[pname]
        plevs = r["plevs"]
        rr = [r["random_per_level"][p]["rmsvd"] if (r["random_per_level"][p] is not None and r["random_per_level"][p]["n"] >= 50) else np.nan for p in plevs]
        pr = [r["real_per_level"][p]["rmsvd"] if (r["real_per_level"][p] is not None and r["real_per_level"][p]["n"] >= 50) else np.nan for p in plevs]
        ax.plot(rr, plevs, "--o", label=f"{pname} random@{THRESH:g} m/s")
        ax.plot(pr, plevs, "-s", label=f"{pname} real")
    ax.set_xlabel("RMSVD (m/s)")
    ax.set_ylabel("pressure (hPa)")
    ax.invert_yaxis()
    ax.set_title("RMSVD by level: real AMV vs random null model")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.scatter(comb["spd_era5"], comb["spd"], s=2, alpha=0.15)
    lim = [0, max(np.nanpercentile(comb["spd_era5"], 99), np.nanpercentile(comb["spd"], 99))]
    ax.plot(lim, lim, "r--", lw=1, label="1:1")
    ax.set_xlabel("ERA5 speed (m/s)")
    ax.set_ylabel("random speed (m/s)")
    ax.set_title(f"random speed vs ERA5 @{THRESH:g} m/s (corr {comb['corr']:.2f}, RMSVD {comb['rmsvd']:.1f} m/s, n={comb['n']})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(PLOTS, f"random_vectors_compare{SUF}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)

    # ---- quiver figures ----
    for pname in PASSES:
        r = results[pname]
        f = os.path.join(DATA, f"wind_coarsened_{pname}_july1{SUF}.nc")
        ds = xr.open_dataset(f).sel(satellite="snpp")
        for tgt in QUIVER_LEVELS:
            i = int(np.argmin(np.abs(r["plevs"] - tgt)))
            pl = float(r["plevs"][i])
            d = ds.isel(plev=i)
            d = d.assign(
                u_rand=(("latitude", "longitude"), r["u_random"][i]),
                v_rand=(("latitude", "longitude"), r["v_random"][i]),
            )
            rr_ = rmsvd(d["u"].values, d["v"].values, d["u_era5"].values, d["v_era5"].values)
            rg_ = rmsvd(r["u_random"][i], r["v_random"][i], d["u_era5"].values, d["v_era5"].values)

            dc = d.coarsen(latitude=4, longitude=4, boundary="trim").mean()
            X, Y = np.meshgrid(dc.longitude.values, dc.latitude.values)

            fig, axes = plt.subplots(1, 3, figsize=(20, 5),
                                     subplot_kw={"projection": ccrs.PlateCarree()})
            one_ax(axes[0], X, Y, dc["u"].values, dc["v"].values, f"{pname} retrieved {pl:.0f} hPa")
            one_ax(axes[1], X, Y, dc["u_rand"].values, dc["v_rand"].values, f"{pname} random @{THRESH:g} m/s")
            one_ax(axes[2], X, Y, dc["u_era5"].values, dc["v_era5"].values, f"ERA5 {pl:.0f} hPa")
            fig.suptitle(f"RMSVD vs ERA5: retrieved {rr_:.2f} m/s | random@{THRESH:g} m/s {rg_:.2f} m/s", fontsize=12)
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            out = os.path.join(PLOTS, f"quiver_random_{pname}_{pl:.0f}hPa{SUF}.png")
            fig.savefig(out, dpi=130, bbox_inches="tight")
            plt.close(fig)
            print("wrote", out)
        ds.close()

    print("DONE")


if __name__ == "__main__":
    main()
