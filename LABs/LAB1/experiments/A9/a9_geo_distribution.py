# -*- coding: utf-8 -*-
"""A9: geographic distribution of the world train/valid splits.

Questions
- How much of the globe does each split cover, and how concentrated is it?
- How many samples per country, and how do train/valid differ?
- Is valid a near-neighbour subset of train (as Q7 suggested)?

Outputs
- experiments/A9/a9_results.json
- experiments/A9/figures/a9_maps.{png,pdf}
- experiments/A9/figures/a9_country_counts.{png,pdf}
- experiments/A9/figures/a9_coverage.{png,pdf}
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(parents=True, exist_ok=True)
R_EARTH = 6371.0


def load_split(split):
    df = pd.read_csv(ROOT / f"datasets/world/{split}.csv")
    return df


def haversine_matrix(lat1, lon1, lat2, lon2):
    """Pairwise great-circle distance in km, shape (len(lat1), len(lat2))."""
    lat1 = np.radians(np.asarray(lat1, dtype=np.float64))[:, None]
    lon1 = np.radians(np.asarray(lon1, dtype=np.float64))[:, None]
    lat2 = np.radians(np.asarray(lat2, dtype=np.float64))[None, :]
    lon2 = np.radians(np.asarray(lon2, dtype=np.float64))[None, :]
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * R_EARTH * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def nn_distances(src, dst, same=False):
    d = haversine_matrix(src["latitude"], src["longitude"],
                         dst["latitude"], dst["longitude"])
    if same:
        np.fill_diagonal(d, np.inf)
    return d.min(axis=1)


def gini(counts):
    x = np.sort(np.asarray(counts, dtype=np.float64))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return float("nan")
    return float((2.0 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def hhi(counts):
    p = np.asarray(counts, dtype=np.float64)
    p = p / p.sum()
    return float(np.sum(p ** 2))


def grid_stats(df, lat_edges, lon_edges, lon_centers, lat_centers):
    """Count samples in 10-degree cells; area weight = cos(mid latitude)."""
    lat_idx = np.clip(np.digitize(df["latitude"], lat_edges) - 1, 0, len(lat_centers) - 1)
    lon_idx = np.clip(np.digitize(df["longitude"], lon_edges) - 1, 0, len(lon_centers) - 1)
    counts = np.zeros((len(lat_centers), len(lon_centers)), dtype=np.int64)
    np.add.at(counts, (lat_idx, lon_idx), 1)
    area = np.cos(np.radians(lat_centers))[:, None] * np.ones((1, len(lon_centers)))
    occupied = counts > 0
    total_area = area.sum()
    covered_area = area[occupied].sum()
    occ_counts = counts[occupied].astype(np.float64)
    order = np.argsort(occ_counts)[::-1]
    sorted_counts = occ_counts[order]
    return {
        "counts": counts,
        "area": area,
        "occupied_cells": int(occupied.sum()),
        "total_cells": int(counts.size),
        "area_fraction": float(covered_area / total_area),
        "counts_occupied": occ_counts,
        "gini_occupied": gini(occ_counts),
        "hhi_occupied": hhi(occ_counts),
        "effective_cells": float(1.0 / hhi(occ_counts)) if hhi(occ_counts) > 0 else float("nan"),
        "top1_share": float(sorted_counts[0] / sorted_counts.sum()),
        "top5_share": float(sorted_counts[:5].sum() / sorted_counts.sum()),
    }


def main():
    tr = load_split("train")
    va = load_split("valid")
    print(f"[data] train {len(tr)} / valid {len(va)}")

    # --- countries ---
    ct = tr["country"].value_counts()
    cv = va["country"].value_counts()
    all_countries = sorted(set(ct.index) | set(cv.index))
    country_rows = []
    for c in all_countries:
        country_rows.append({
            "country": c,
            "train": int(ct.get(c, 0)),
            "valid": int(cv.get(c, 0)),
        })
    country_df = pd.DataFrame(country_rows).sort_values(["train", "valid"], ascending=False)
    valid_only = sorted(set(cv.index) - set(ct.index))
    train_only = sorted(set(ct.index) - set(cv.index))
    both = sorted(set(ct.index) & set(cv.index))
    country_stats = {
        "n_train_countries": int(len(ct)),
        "n_valid_countries": int(len(cv)),
        "n_both": int(len(both)),
        "n_valid_only": int(len(valid_only)),
        "n_train_only": int(len(train_only)),
        "valid_only": valid_only,
        "valid_only_samples": int(sum(cv[c] for c in valid_only)),
        "train_only_samples": int(sum(ct[c] for c in train_only)),
        "gini_train": gini(ct.values),
        "gini_valid": gini(cv.values),
        "hhi_train": hhi(ct.values),
        "hhi_valid": hhi(cv.values),
        "top10_share_train": float(ct.head(10).sum() / ct.sum()),
        "top10_share_valid": float(cv.head(10).sum() / cv.sum()),
        "top1_share_train": float(ct.head(1).sum() / ct.sum()),
        "top1_share_valid": float(cv.head(1).sum() / cv.sum()),
        "countries_with_1_sample_train": int((ct == 1).sum()),
        "countries_with_1_sample_valid": int((cv == 1).sum()),
        "countries_with_ge5_train": int((ct >= 5).sum()),
        "countries_with_ge5_valid": int((cv >= 5).sum()),
    }

    # --- nearest neighbours (degrees / km) ---
    nn_tr = nn_distances(tr, tr, same=True)
    nn_va = nn_distances(va, tr, same=False)
    nn_stats = {
        "train_to_train": {
            "median_km": float(np.median(nn_tr)),
            "mean_km": float(np.mean(nn_tr)),
            "p90_km": float(np.percentile(nn_tr, 90)),
            "max_km": float(np.max(nn_tr)),
        },
        "valid_to_train": {
            "median_km": float(np.median(nn_va)),
            "mean_km": float(np.mean(nn_va)),
            "p90_km": float(np.percentile(nn_va, 90)),
            "max_km": float(np.max(nn_va)),
            "frac_within_100km": float(np.mean(nn_va < 100.0)),
            "frac_within_500km": float(np.mean(nn_va < 500.0)),
            "frac_within_1000km": float(np.mean(nn_va < 1000.0)),
        },
    }

    # --- latitude bands (10 deg), area-normalised ---
    lat_edges = np.arange(-90.0, 90.0001, 10.0)
    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2.0
    lon_edges = np.arange(-180.0, 180.0001, 10.0)
    lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2.0
    tr_counts, _ = np.histogram(tr["latitude"], bins=lat_edges)
    va_counts, _ = np.histogram(va["latitude"], bins=lat_edges)
    band_area = np.cos(np.radians(lat_centers))
    lat_stats = {
        "lat_centers": lat_centers.tolist(),
        "train_counts": tr_counts.tolist(),
        "valid_counts": va_counts.tolist(),
        "train_per_area": (tr_counts / band_area).tolist(),
        "valid_per_area": (va_counts / band_area).tolist(),
        "north_frac_train": float(np.mean(tr["latitude"] > 0)),
        "north_frac_valid": float(np.mean(va["latitude"] > 0)),
    }

    grid_tr = grid_stats(tr, lat_edges, lon_edges, lon_centers, lat_centers)
    grid_va = grid_stats(va, lat_edges, lon_edges, lon_centers, lat_centers)
    grid_summary = {
        split: {k: v for k, v in g.items() if k not in ("counts", "area", "counts_occupied")}
        for split, g in (("train", grid_tr), ("valid", grid_va))
    }
    grid_summary["train"]["top5_share"] = grid_tr["top5_share"]
    grid_summary["valid"]["top5_share"] = grid_va["top5_share"]

    results = {
        "protocol": {
            "n_train": int(len(tr)),
            "n_valid": int(len(va)),
            "grid": "10 deg x 10 deg; area weight = cos(mid latitude); coverage is approximate",
            "distance": "haversine, R=6371 km",
        },
        "countries": country_stats,
        "country_table": country_rows,
        "nearest_neighbour": nn_stats,
        "latitude_bands": lat_stats,
        "grid": grid_summary,
    }
    with open(HERE / "a9_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("[stats]", json.dumps({k: country_stats[k] for k in
          ("n_train_countries", "n_valid_countries", "n_valid_only", "n_train_only")}, ensure_ascii=False))
    print("[nn] valid->train median %.1f km, p90 %.1f, max %.1f" % (
        nn_stats["valid_to_train"]["median_km"],
        nn_stats["valid_to_train"]["p90_km"],
        nn_stats["valid_to_train"]["max_km"]))

    # ================= figures =================
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9})
    land = cfeature.NaturalEarthFeature("physical", "land", "110m",
                                        facecolor="#f2f2f0", edgecolor="none")

    # ---- fig 1: maps ----
    fig = plt.figure(figsize=(13.2, 6.4))
    panels = [
        ("Train (n=500)", tr, "#2166ac"),
        ("Valid (n=130)", va, "#d95f02"),
    ]
    for i, (title, df, color) in enumerate(panels):
        ax = fig.add_subplot(2, 2, i + 1, projection=ccrs.PlateCarree())
        ax.set_global()
        ax.add_feature(land, zorder=0)
        ax.coastlines("110m", linewidth=0.45, color="#666666", zorder=1)
        ax.scatter(df["longitude"], df["latitude"], s=7, c=color, alpha=0.72,
                   linewidths=0, transform=ccrs.PlateCarree(), zorder=2)
        ax.set_title(f"{title}", fontsize=10)
        gl = ax.gridlines(draw_labels=True, linewidth=0.25, color="#cccccc", alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 7}
        gl.ylabel_style = {"size": 7}
    ax = fig.add_subplot(2, 2, (3, 4), projection=ccrs.PlateCarree())
    ax.set_global()
    ax.add_feature(land, zorder=0)
    ax.coastlines("110m", linewidth=0.45, color="#666666", zorder=1)
    ax.scatter(tr["longitude"], tr["latitude"], s=6, c="#8da0cb", alpha=0.55,
               linewidths=0, transform=ccrs.PlateCarree(), label="train", zorder=2)
    ax.scatter(va["longitude"], va["latitude"], s=14, c="#d95f02", alpha=0.9,
               linewidths=0.3, edgecolors="white", transform=ccrs.PlateCarree(),
               label="valid", zorder=3)
    ax.legend(loc="lower left", fontsize=7, framealpha=0.9)
    ax.set_title("Overlay: valid points sit mostly on/near train coverage", fontsize=10)
    gl = ax.gridlines(draw_labels=True, linewidth=0.25, color="#cccccc", alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 7}
    gl.ylabel_style = {"size": 7}
    fig.suptitle("A9: global coverage of the world splits (110m coastlines; no country borders)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"a9_maps.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # ---- fig 2: per-country counts ----
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 5.0))
    top = ct.head(15).index.tolist()
    y = np.arange(len(top))
    tr_vals = [int(ct.get(c, 0)) for c in top]
    va_vals = [int(cv.get(c, 0)) for c in top]
    ax = axes[0]
    ax.barh(y - 0.19, tr_vals, height=0.36, color="#2166ac", label="train")
    ax.barh(y + 0.19, va_vals, height=0.36, color="#d95f02", label="valid")
    ax.set_yticks(y)
    ax.set_yticklabels(top, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("samples")
    ax.set_title("Top-15 countries by train count")
    ax.legend(fontsize=7)
    ax.grid(axis="x", alpha=0.25)

    ax = axes[1]
    maxc = max(ct.max(), cv.max())
    bins = np.geomspace(0.5, maxc + 0.5, 12)
    ax.hist(ct.values, bins=bins, color="#2166ac", alpha=0.65, label="train", log=True)
    ax.hist(cv.values, bins=bins, color="#d95f02", alpha=0.65, label="valid", log=True)
    ax.set_xscale("log")
    ax.set_xlabel("samples per country (log bins)")
    ax.set_ylabel("number of countries (log)")
    ax.set_title("Country-count distribution\n(train 158 / valid 78 countries)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25, which="both")

    ax = axes[2]
    x = country_df["train"].to_numpy() + 1
    yv = country_df["valid"].to_numpy() + 1
    is_vo = country_df["train"].to_numpy() == 0
    is_to = country_df["valid"].to_numpy() == 0
    ax.scatter(x[~is_vo & ~is_to], yv[~is_vo & ~is_to], s=14, color="#666666",
               alpha=0.7, label="both")
    ax.scatter(x[is_vo], yv[is_vo], s=26, color="#d95f02", marker="^",
               label="valid-only (19)")
    ax.scatter(x[is_to], yv[is_to], s=18, color="#2166ac", marker="v",
               alpha=0.6, label="train-only (99)")
    lim = max(x.max(), yv.max()) * 1.1
    ax.plot([1, lim], [1, lim], ls="--", lw=0.8, color="#999999")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("train count + 1 (log)")
    ax.set_ylabel("valid count + 1 (log)")
    ax.set_title("Per-country coverage, train vs valid")
    ax.text(0.02, 0.02, "valid-only: 19 countries / 20 samples\ntrain-only: 99 countries / 187 samples",
            transform=ax.transAxes, fontsize=6.5, color="#555555", va="bottom")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.25, which="both")
    fig.suptitle("A9: country-level sample distribution", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"a9_country_counts.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # ---- fig 3: coverage / concentration ----
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.8))
    ax = axes[0]
    ax.plot(lat_centers, tr_counts / band_area, "-o", ms=3, color="#2166ac", label="train")
    ax.plot(lat_centers, va_counts / band_area, "-s", ms=3, color="#d95f02", label="valid")
    ax.set_xlabel("latitude (deg)")
    ax.set_ylabel("samples per cos(lat) unit")
    ax.set_title("Latitude coverage per unit area")
    ax.set_xticks(np.arange(-90, 91, 30))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)

    ax = axes[1]
    xs_tr = np.sort(nn_tr)
    xs_va = np.sort(nn_va)
    ax.plot(xs_tr, np.linspace(0, 1, len(xs_tr)), color="#2166ac", label="train -> train")
    ax.plot(xs_va, np.linspace(0, 1, len(xs_va)), color="#d95f02", label="valid -> train")
    ax.axvline(np.median(nn_tr), color="#2166ac", ls=":", lw=0.9)
    ax.axvline(np.median(nn_va), color="#d95f02", ls=":", lw=0.9)
    ax.set_xscale("log")
    ax.set_xlabel("nearest-neighbour distance (km, log)")
    ax.set_ylabel("ECDF")
    ax.set_title("Nearest-neighbour distance\n(medians: %.0f / %.0f km)" % (
        np.median(nn_tr), np.median(nn_va)))
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=7)

    ax = axes[2]
    for name, g, color in (("train", grid_tr, "#2166ac"), ("valid", grid_va, "#d95f02")):
        c = np.sort(g["counts_occupied"])[::-1]
        cum = np.cumsum(c) / c.sum()
        xfrac = np.arange(1, len(c) + 1) / len(c)
        ax.plot(xfrac, cum, color=color,
                label=f"{name}: {g['occupied_cells']} cells, Gini {g['gini_occupied']:.2f}")
    ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color="#999999")
    ax.set_xlabel("fraction of occupied 10-deg cells (sorted by count)")
    ax.set_ylabel("cumulative sample share")
    ax.set_title("Spatial concentration (Lorenz)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    fig.suptitle("A9: coverage density and spatial concentration", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"a9_coverage.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print("[done] wrote", HERE / "a9_results.json")
    print("[grid] train occupied %d (%.1f%% area), valid %d (%.1f%% area)" % (
        grid_tr["occupied_cells"], 100 * grid_tr["area_fraction"],
        grid_va["occupied_cells"], 100 * grid_va["area_fraction"]))


if __name__ == "__main__":
    main()
