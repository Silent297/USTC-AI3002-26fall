# -*- coding: utf-8 -*-
"""C1 评估：真实 JSONL → 指标 / 参考对照 / 配对统计 / 3 张图（本地 CPU，无 torch）。

输入：--run-dir（含 train.jsonl / valid.jsonl / run_meta.json）
前置自检：复算 world raw L11 探针锚点（valid MSE 150.4277、train 116.9279，±1e-3；
大圆中位 valid 953.8 km / train 662.4 km），不符即中止（防资产错配）。

口径（与指南3 §4.2/§4.3 及课程一致）：
- 有效回答 = tier ∈ {json, embedded, regex} 且 in_range；无效不进 MSE 分母；
- MSE = 有效条数上 (Δlat²+Δlon²)/2 的平均；周期经度 MSE 为附加指标（经度差回绕）；
- 距离 = 大圆 haversine（R=6371 km）；100 km 命中率分母 = 全部输入条数；
- 参考数字来自同学参考报告（非官方），仅做对照线。

输出：<run-dir>/c1_metrics.json + <fig-dir>/c1_{distance_cdf,valid_map,metric_bars}.{png,pdf}
地图合规：仅 110m 海岸线，严禁国界线。
"""
import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

import cartopy.crs as ccrs

from c1_lib import coord_sq_error, haversine_km, wrap_lon_deg

ROOT = Path(__file__).resolve().parent.parent.parent
WORLD_DIR = ROOT / "datasets" / "world"
PROBE_PATH = ROOT / "results/slurm_layers/world/weights/world_layer11_extra0/weights.npz"
DEFAULT_RUN_DIR = ROOT / "results/slurm_qwen_direct/qwen_direct_v1"
DEFAULT_FIG_DIR = ROOT / "experiments" / "C1" / "figures"

SEED = 20261016
N_BOOT = 10000
TIER_ORDER = ("json", "embedded", "regex", "out_of_range", "failed")

# 探针锚点（本地已复算：q6/q7 报告口径）
ANCHOR = {
    "train": {"mse": 116.9279, "median_km": 662.4},
    "valid": {"mse": 150.4277, "median_km": 953.8},
}
ANCHOR_MSE_TOL = 1e-3
ANCHOR_MEDIAN_TOL_KM = 0.5

# 参考数字：指南3 §4.2/§4.3（同学参考报告，非官方）
REFERENCE = {
    "train": {"n_total": 500, "n_valid": 495, "mse": 161.072281, "periodic_mse": 109.56,
              "median_km": 15.20, "mean_km": 201.97, "p90_km": 332.04, "hit100_all": 0.6920},
    "valid": {"n_total": 130, "n_valid": 128, "mse": 513.512888, "periodic_mse": 17.77,
              "median_km": 35.78, "mean_km": 159.00, "p90_km": 439.10, "hit100_all": 0.6385},
    "combined": {"n_total": 630, "n_valid": 623, "mse": 233.483835, "periodic_mse": 90.70,
                 "median_km": 19.75, "mean_km": 193.14, "p90_km": 362.83, "hit100_all": 0.6810},
}


def utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--metrics-out", type=Path, default=None,
                        help="默认 <run-dir>/c1_metrics.json")
    parser.add_argument("--skip-anchor", action="store_true", help="跳过探针锚点自检（仅调试用）")
    return parser.parse_args()


def load_world(split):
    frame = pd.read_csv(WORLD_DIR / f"{split}.csv")
    truth = frame[["latitude", "longitude"]].to_numpy(np.float64)
    features = np.load(WORLD_DIR / f"{split}.layer11.npy").astype(np.float64)
    probe_weights = np.load(PROBE_PATH)
    probe = features @ probe_weights["weights"].T + probe_weights["bias"]
    return frame, truth, probe


def anchor_check():
    result = {}
    for split in ("train", "valid"):
        _, truth, probe = load_world(split)
        mse = float(np.mean(coord_sq_error(probe[:, 0], probe[:, 1], truth[:, 0], truth[:, 1])))
        distances = haversine_km(probe[:, 0], probe[:, 1], truth[:, 0], truth[:, 1])
        median_km = float(np.median(distances))
        n_lon_wrap = int(np.sum(np.abs(probe[:, 1] - truth[:, 1]) > 180.0))
        expected = ANCHOR[split]
        problems = []
        if abs(mse - expected["mse"]) > ANCHOR_MSE_TOL:
            problems.append(f"{split} MSE {mse:.4f} != anchor {expected['mse']}")
        if abs(median_km - expected["median_km"]) > ANCHOR_MEDIAN_TOL_KM:
            problems.append(f"{split} median {median_km:.1f} km != anchor {expected['median_km']}")
        if n_lon_wrap != 0:
            problems.append(f"{split} has {n_lon_wrap} rows with |dlon|>180")
        result[split] = {"mse": mse, "median_km": median_km, "n_lon_wrap": n_lon_wrap,
                         "mean_km": float(np.mean(distances)), "p90_km": float(np.percentile(distances, 90)),
                         "max_km": float(np.max(distances))}
        status = "OK" if not problems else "FAIL"
        print(f"anchor[{split}] {status}: MSE={mse:.4f} ({expected['mse']}), "
              f"median={median_km:.1f} km ({expected['median_km']}), |dlon|>180: {n_lon_wrap}")
        if problems:
            raise SystemExit("Anchor self-check failed: " + "; ".join(problems))
    return result


def load_records(path):
    records = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                records.append(json.loads(line))
    records.sort(key=lambda record: record["row"])
    return records


def summarize(records, n_total):
    tiers = Counter(record["tier"] for record in records)
    in_range = [record for record in records if record["in_range"]]
    summary = {
        "n_total": n_total,
        "n_lines": len(records),
        "tiers": {tier: tiers.get(tier, 0) for tier in TIER_ORDER},
        "n_valid": len(in_range),
        "efficiency": len(in_range) / n_total if n_total else float("nan"),
        "strict_rate": tiers.get("json", 0) / n_total if n_total else float("nan"),
        "loose_rate": sum(tiers.get(tier, 0) for tier in ("json", "embedded", "regex")) / n_total
        if n_total else float("nan"),
    }
    if not in_range:
        return summary, None
    pred_lat = np.array([record["pred_lat"] for record in in_range], dtype=np.float64)
    pred_lon = np.array([record["pred_lon"] for record in in_range], dtype=np.float64)
    true_lat = np.array([record["lat_true"] for record in in_range], dtype=np.float64)
    true_lon = np.array([record["lon_true"] for record in in_range], dtype=np.float64)
    sq_raw = coord_sq_error(pred_lat, pred_lon, true_lat, true_lon)
    sq_periodic = coord_sq_error(pred_lat, pred_lon, true_lat, true_lon, periodic_lon=True)
    distances = haversine_km(pred_lat, pred_lon, true_lat, true_lon)
    summary.update({
        "mse": float(np.mean(sq_raw)),
        "periodic_mse": float(np.mean(sq_periodic)),
        "rmse": float(np.sqrt(np.mean(sq_raw))),
        "dist_mean_km": float(np.mean(distances)),
        "dist_median_km": float(np.median(distances)),
        "dist_p90_km": float(np.percentile(distances, 90)),
        "dist_max_km": float(np.max(distances)),
        "hit100_valid_rate": float(np.mean(distances <= 100.0)),
        "hit100_all_rate": float(np.sum(distances <= 100.0) / n_total) if n_total else float("nan"),
        "n_dlon_wrap": int(np.sum(np.abs(pred_lon - true_lon) > 180.0)),
    })
    return summary, {"pred_lat": pred_lat, "pred_lon": pred_lon, "true_lat": true_lat,
                     "true_lon": true_lon, "sq_raw": sq_raw, "sq_periodic": sq_periodic,
                     "distances": distances, "in_range": in_range}


def top_contributions(records, top_k=5):
    in_range = [record for record in records if record["in_range"]]
    if not in_range:
        return []
    errors = []
    for record in in_range:
        sq = float(coord_sq_error(record["pred_lat"], record["pred_lon"],
                                  record["lat_true"], record["lon_true"]))
        dlon_raw = record["pred_lon"] - record["lon_true"]
        distance = float(haversine_km(record["pred_lat"], record["pred_lon"],
                                      record["lat_true"], record["lon_true"]))
        errors.append({"row": record["row"], "sentence": record["sentence"], "country": record["country"],
                       "lat_true": record["lat_true"], "lon_true": record["lon_true"],
                       "pred_lat": record["pred_lat"], "pred_lon": record["pred_lon"],
                       "dlon_raw": float(dlon_raw), "distance_km": distance, "sq_error": sq})
    total = sum(item["sq_error"] for item in errors)
    for item in errors:
        item["share"] = item["sq_error"] / total if total else 0.0
    errors.sort(key=lambda item: item["sq_error"], reverse=True)
    return errors[:top_k]


def paired_bootstrap(valid_records, probe, check_anchor_off=False):
    subset = [record for record in valid_records if record["in_range"]]
    if not subset:
        return None
    rows = np.array([record["row"] for record in subset])
    true_lat = np.array([record["lat_true"] for record in subset], dtype=np.float64)
    true_lon = np.array([record["lon_true"] for record in subset], dtype=np.float64)
    direct_lat = np.array([record["pred_lat"] for record in subset], dtype=np.float64)
    direct_lon = np.array([record["pred_lon"] for record in subset], dtype=np.float64)
    probe_lat = probe[rows, 0]
    probe_lon = probe[rows, 1]

    d_direct = haversine_km(direct_lat, direct_lon, true_lat, true_lon)
    d_probe = haversine_km(probe_lat, probe_lon, true_lat, true_lon)
    se_direct = coord_sq_error(direct_lat, direct_lon, true_lat, true_lon)
    se_probe = coord_sq_error(probe_lat, probe_lon, true_lat, true_lon)

    n = len(rows)
    rng = np.random.default_rng(SEED)
    index = rng.integers(0, n, size=(N_BOOT, n))
    dist_diff = d_probe[index].mean(axis=1) - d_direct[index].mean(axis=1)
    se_diff = se_probe[index].mean(axis=1) - se_direct[index].mean(axis=1)

    def pack(diff, direct, probe_values):
        return {
            "n": int(n),
            "mean_direct": float(np.mean(direct)),
            "mean_probe": float(np.mean(probe_values)),
            "mean_diff_probe_minus_direct": float(np.mean(probe_values) - np.mean(direct)),
            "ci95_probe_minus_direct": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
            "n_direct_closer": int(np.sum(direct < probe_values)),
            "n_probe_closer": int(np.sum(probe_values < direct)),
            "n_ties": int(np.sum(direct == probe_values)),
        }

    return {
        "rows": rows.tolist(),
        "distance_km": pack(dist_diff, d_direct, d_probe),
        "sq_error": pack(se_diff, se_direct, se_probe),
        "note": "diff = probe - direct; CI95 = percentile bootstrap over rows (10k, seed fixed)",
    }


def entity_name(sentence):
    text = sentence.strip()
    prefix = "What's the longitude and latitude of "
    if text.startswith(prefix):
        text = text[len(prefix):]
    if text.endswith("?"):
        text = text[:-1]
    return text


def fig_distance_cdf(fig_dir, direct_distances, probe_distances):
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    series = [
        ("Qwen direct - valid", direct_distances["valid"], "#d62728", "-"),
        ("Qwen direct - train", direct_distances["train"], "#ff9896", "-"),
        ("probe (raw L11) - valid", probe_distances["valid"], "#1f77b4", "--"),
        ("probe (raw L11) - train", probe_distances["train"], "#aec7e8", "--"),
    ]
    for label, values, color, style in series:
        if values is None or len(values) == 0:
            continue
        ordered = np.sort(values)
        cdf = np.arange(1, len(ordered) + 1) / len(ordered)
        ax.plot(ordered, cdf, style, color=color, linewidth=1.6,
                label=f"{label}  (n={len(ordered)}, median {np.median(ordered):.1f} km)")
    ax.set_xscale("log")
    ax.set_xlim(1, 30000)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("great-circle distance to ground truth (km, log scale)")
    ax.set_ylabel("empirical CDF")
    ax.set_title("C1: positioning error CDF - Qwen direct output vs linear probe")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(fig_dir / "c1_distance_cdf.png", dpi=150)
    fig.savefig(fig_dir / "c1_distance_cdf.pdf")
    plt.close(fig)


def fig_valid_map(fig_dir, valid_records, valid_distances, top_items):
    subset = [(record, distance) for record, distance in zip(
        [record for record in valid_records if record["in_range"]], valid_distances)]
    lons = np.array([record["pred_lon"] for record, _ in subset])
    lats = np.array([record["pred_lat"] for record, _ in subset])
    distances = np.array([distance for _, distance in subset])
    true_lon = np.array([record["lon_true"] for record, _ in subset])
    true_lat = np.array([record["lat_true"] for record, _ in subset])

    fig = plt.figure(figsize=(12.6, 7.0))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_global()
    ax.coastlines(resolution="110m", linewidth=0.5, color="0.4")
    grid = ax.gridlines(draw_labels=True, linewidth=0.3, color="0.85", alpha=0.7)
    grid.top_labels = grid.right_labels = False
    grid.xlabel_style = {"size": 8}
    grid.ylabel_style = {"size": 8}

    ax.scatter(true_lon, true_lat, s=16, facecolor="none", edgecolor="0.35", linewidths=0.8,
               transform=ccrs.PlateCarree(), zorder=3, label="ground truth")
    vmax = max(1000.0, float(np.percentile(distances, 90)))
    norm = LogNorm(vmin=1.0, vmax=vmax)
    scatter = ax.scatter(lons, lats, c=np.clip(distances, 1.0, None), cmap="viridis",
                         norm=norm, s=34, edgecolor="k", linewidths=0.3,
                         transform=ccrs.PlateCarree(), zorder=4,
                         label="Qwen direct prediction (color = distance)")
    colorbar = fig.colorbar(scatter, ax=ax, shrink=0.7, pad=0.02)
    colorbar.set_label("great-circle distance (km, log scale)", fontsize=9)

    for item in top_items:
        label = f"{entity_name(item['sentence'])}\n{item['distance_km']:.0f} km"
        if item["pred_lat"] < -55:  # near the bottom edge: box above-left to clear the colorbar
            dx, dy, ha = -12, 10, "right"
        else:
            dx, dy, ha = 12, -8, "left"
        ax.annotate(label, xy=(item["pred_lon"], item["pred_lat"]), xytext=(dx, dy),
                    textcoords="offset points", fontsize=7, ha=ha,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.5", alpha=0.85),
                    arrowprops=dict(arrowstyle="-", color="0.4", linewidth=0.7),
                    transform=ccrs.PlateCarree(), zorder=6)

    n_valid = len(subset)
    median_km = float(np.median(distances))
    ax.set_title(f"C1: Qwen3-4B-Base direct coordinates on world valid "
                 f"(n={n_valid}/130 valid; median {median_km:.1f} km)\n"
                 f"hollow circles = truth; dots = prediction, colored by great-circle distance; top-5 errors annotated",
                 fontsize=11)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(fig_dir / "c1_valid_map.png", dpi=150)
    fig.savefig(fig_dir / "c1_valid_map.pdf")
    plt.close(fig)


def fig_metric_bars(fig_dir, metrics, probe_stats):
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.6))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    ax = axes[0]
    splits = ["train", "valid"]
    bottoms = np.zeros(len(splits))
    for tier_index, tier in enumerate(TIER_ORDER):
        shares = [100.0 * metrics[split]["tiers"].get(tier, 0) / metrics[split]["n_total"] for split in splits]
        ax.bar(splits, shares, bottom=bottoms, color=colors[tier_index], label=tier)
        for position, share in enumerate(shares):
            if share >= 2.0:
                ax.text(position, bottoms[position] + share / 2.0, f"{share:.1f}", ha="center",
                        va="center", fontsize=8,
                        color="white" if tier in ("json", "failed") else "black")
        bottoms += np.array(shares)
    ax.set_title("answer parse tiers (% of all inputs)", fontsize=10)
    ax.set_ylabel("share of inputs (%)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=7.5, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.42))

    ax = axes[1]
    positions = np.arange(len(splits))
    width = 0.34
    raw = [metrics[split]["mse"] for split in splits]
    periodic = [metrics[split]["periodic_mse"] for split in splits]
    ax.bar(positions - width / 2, raw, width, color=colors[0], label="raw MSE")
    ax.bar(positions + width / 2, periodic, width, color=colors[1], label="periodic-lon MSE")
    for split_index, split in enumerate(splits):
        reference = REFERENCE[split]
        ax.scatter([split_index - width / 2], [reference["mse"]], marker="D", s=28, color="k", zorder=5,
                   label="classmate report, unofficial" if split_index == 0 else None)
        ax.scatter([split_index + width / 2], [reference["periodic_mse"]], marker="D", s=28, color="k", zorder=5)
        ax.scatter([split_index - width / 2], [probe_stats[split]["mse"]], marker="*", s=90, color="crimson",
                   zorder=5, label="probe MSE anchor (ours)" if split_index == 0 else None)
        for value, xpos in ((raw[split_index], split_index - width / 2),
                            (periodic[split_index], split_index + width / 2)):
            ax.text(xpos, value * 1.12, f"{value:.1f}", ha="center", fontsize=7.5)
    ax.set_yscale("log")
    ax.set_xticks(positions, splits)
    ax.set_title("coordinate MSE (deg$^2$, note log scale)", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper left")

    ax = axes[2]
    medians = [metrics[split]["dist_median_km"] for split in splits]
    p90s = [metrics[split]["dist_p90_km"] for split in splits]
    ax.bar(positions - width / 2, medians, width, color=colors[0], label="median")
    ax.bar(positions + width / 2, p90s, width, color=colors[1], label="P90")
    for split_index, split in enumerate(splits):
        reference = REFERENCE[split]
        ax.scatter([split_index - width / 2], [reference["median_km"]], marker="D", s=28, color="k", zorder=5,
                   label="classmate report, unofficial" if split_index == 0 else None)
        ax.scatter([split_index + width / 2], [reference["p90_km"]], marker="D", s=28, color="k", zorder=5)
        ax.scatter([split_index - width / 2], [probe_stats[split]["median_km"]], marker="*", s=90, color="crimson",
                   zorder=5, label="probe median (ours)" if split_index == 0 else None)
        for value, xpos in ((medians[split_index], split_index - width / 2),
                            (p90s[split_index], split_index + width / 2)):
            ax.text(xpos, value * 1.12, f"{value:.0f}", ha="center", fontsize=7.5)
    ax.set_yscale("log")
    ax.set_xticks(positions, splits)
    ax.set_title("great-circle distance (km, log scale)", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper left")

    fig.suptitle("C1: Qwen3-4B-Base direct output vs reference/probe (classmate report numbers are unofficial)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(fig_dir / "c1_metric_bars.png", dpi=150)
    fig.savefig(fig_dir / "c1_metric_bars.pdf")
    plt.close(fig)


def main():
    args = parse_args()
    run_dir = args.run_dir
    fig_dir = args.fig_dir
    fig_dir.mkdir(parents=True, exist_ok=True)
    metrics_out = args.metrics_out or (run_dir / "c1_metrics.json")

    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    expected = meta["expected_rows"]
    print(f"run-dir: {run_dir}")
    print(f"meta: prompt_template ok={meta.get('prompt_template') is not None}, "
          f"expected_rows={expected}, tiers={meta.get('tiers')}")

    anchors = {} if args.skip_anchor else anchor_check()

    train_records = load_records(run_dir / "train.jsonl")
    valid_records = load_records(run_dir / "valid.jsonl")
    train_summary, train_pack = summarize(train_records, expected["train"])
    valid_summary, valid_pack = summarize(valid_records, expected["valid"])

    pooled_distances = []
    if train_pack is not None:
        pooled_distances.append(train_pack["distances"])
    if valid_pack is not None:
        pooled_distances.append(valid_pack["distances"])
    n_total_all = expected["train"] + expected["valid"]
    if pooled_distances:
        pooled = np.concatenate(pooled_distances)
        n_valid_all = len(pooled)
        pooled_sq = np.concatenate([pack["sq_raw"] for pack in (train_pack, valid_pack) if pack is not None])
        pooled_periodic = np.concatenate([pack["sq_periodic"] for pack in (train_pack, valid_pack) if pack is not None])
        combined = {
            "n_total": n_total_all,
            "n_valid": int(n_valid_all),
            "mse": float(np.mean(pooled_sq)),
            "periodic_mse": float(np.mean(pooled_periodic)),
            "dist_median_km": float(np.median(pooled)),
            "dist_mean_km": float(np.mean(pooled)),
            "dist_p90_km": float(np.percentile(pooled, 90)),
            "hit100_all_rate": float(np.sum(pooled <= 100.0) / n_total_all),
        }
    else:
        combined = {"n_total": n_total_all, "n_valid": 0}

    print()
    for split, summary in (("train", train_summary), ("valid", valid_summary)):
        if summary.get("n_valid"):
            print(f"[{split}] strict={summary['strict_rate']:.4f} loose={summary['loose_rate']:.4f} "
                  f"valid={summary['n_valid']}/{summary['n_total']} MSE={summary['mse']:.4f} "
                  f"periodic={summary['periodic_mse']:.4f} median={summary['dist_median_km']:.2f} km "
                  f"P90={summary['dist_p90_km']:.1f} hit100(all)={summary['hit100_all_rate']:.4f}")
        print(f"[{split}] tiers={summary['tiers']}")
    if combined.get("n_valid"):
        print(f"[combined] valid={combined['n_valid']}/{combined['n_total']} MSE={combined['mse']:.4f} "
              f"periodic={combined['periodic_mse']:.4f} median={combined['dist_median_km']:.2f} km "
              f"hit100(all)={combined['hit100_all_rate']:.4f}")

    _, _, probe_valid = load_world("valid")
    _, _, probe_train = load_world("train")
    probe_distances = {
        "valid": haversine_km(probe_valid[:, 0], probe_valid[:, 1],
                              pd.read_csv(WORLD_DIR / "valid.csv")["latitude"].to_numpy(np.float64),
                              pd.read_csv(WORLD_DIR / "valid.csv")["longitude"].to_numpy(np.float64)),
        "train": haversine_km(probe_train[:, 0], probe_train[:, 1],
                              pd.read_csv(WORLD_DIR / "train.csv")["latitude"].to_numpy(np.float64),
                              pd.read_csv(WORLD_DIR / "train.csv")["longitude"].to_numpy(np.float64)),
    }
    paired = paired_bootstrap(valid_records, probe_valid)
    if paired is not None:
        distance_diff = paired["distance_km"]
        print(f"[paired valid n={distance_diff['n']}] mean dist direct={distance_diff['mean_direct']:.1f} km "
              f"probe={distance_diff['mean_probe']:.1f} km; "
              f"diff(probe-direct)={distance_diff['mean_diff_probe_minus_direct']:.1f} "
              f"CI95=[{distance_diff['ci95_probe_minus_direct'][0]:.1f},{distance_diff['ci95_probe_minus_direct'][1]:.1f}]; "
              f"direct closer {distance_diff['n_direct_closer']}/{distance_diff['n']}")

    top_train = top_contributions(train_records)
    top_valid = top_contributions(valid_records)
    tomanivi_pattern = False
    if top_valid:
        first = top_valid[0]
        tomanivi_pattern = first["share"] >= 0.9 and abs(first["dlon_raw"]) > 180.0
        print(f"[top1 valid] {first['country']}: share={first['share']:.4f} dlon_raw={first['dlon_raw']:.3f} "
              f"dist={first['distance_km']:.1f} km (Tomanivi pattern: {tomanivi_pattern})")

    if valid_pack is not None:
        fig_distance_cdf(fig_dir, {"valid": valid_pack["distances"], "train": train_pack["distances"] if train_pack is not None else None}, probe_distances)
        fig_valid_map(fig_dir, valid_records, valid_pack["distances"], top_valid)
    fig_metric_bars(fig_dir, {"train": train_summary, "valid": valid_summary},
                    {"train": {"mse": ANCHOR["train"]["mse"],
                               "median_km": float(np.median(probe_distances["train"]))},
                     "valid": {"mse": ANCHOR["valid"]["mse"],
                               "median_km": float(np.median(probe_distances["valid"]))}})

    output = {
        "generated_utc": utcnow(),
        "run_dir": str(run_dir),
        "run_meta": {"argv": meta.get("argv"), "gen": meta.get("gen"), "limit": meta.get("limit"),
                     "revision_note": meta.get("revision_note"), "tiers": meta.get("tiers"),
                     "env": meta.get("env"), "tokenizer_info": meta.get("tokenizer_info")},
        "anchors": anchors,
        "anchor_expected": ANCHOR,
        "splits": {"train": train_summary, "valid": valid_summary},
        "combined": combined,
        "reference_classmate_report_unofficial": REFERENCE,
        "paired_valid_vs_probe": paired,
        "probe_distances_km": {split: {"median": float(np.median(values)), "mean": float(np.mean(values)),
                                       "p90": float(np.percentile(values, 90))}
                               for split, values in probe_distances.items()},
        "top_contributions": {"train": top_train, "valid": top_valid},
        "checks": {"valid_top1_tomanivi_pattern": tomanivi_pattern},
    }
    with metrics_out.open("w", encoding="utf-8") as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2)
    print(f"\nWrote {metrics_out}")
    print("figures:", ", ".join(sorted(path.name for path in fig_dir.glob("c1_*.png"))))


if __name__ == "__main__":
    main()
