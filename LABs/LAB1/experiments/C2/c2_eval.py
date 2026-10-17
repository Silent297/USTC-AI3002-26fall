# -*- coding: utf-8 -*-
"""C2 本地评估：变体 × 层指标、配对 bootstrap、分层，出 3-4 张图。

输入：results/slurm_c2/<run>/c2_metrics.json（集群训练产物，含逐条 valid 预测）
对照口径：
- 主口径 = std13-L11（5 个变体齐全，与 world 主网格同配置）；
  raw11 作历史最优配置（world 150.4277）的辅助对拍（nameonly 在该配置下发散）；
- 配对 Δ = 变体 − 参照（正 = 变体更差），bootstrap 95% CI（10k，seed 固定）；
- 预测漂移 = 两个 prompt 条件下预测点之间的大圆距离（衡量 prompt 对答案的影响）；
- 分层 = 国家在 train 见/未见（Q7 口径）。
输出：c2_analysis.json + figures/c2_{layer_curves,variant_bars,paired_effects}.{png,pdf}
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "C1"))
from c1_lib import haversine_km  # noqa: E402

SEED = 20261017
N_BOOT = 10000
VARIANT_ORDER = ("origcountry", "origlast", "nocountry", "nameonly", "rephrase")
VARIANT_LABEL = {
    "origcountry": "orig (country token)",
    "origlast": "orig (last token)",
    "nocountry": "no country name",
    "nameonly": "name only (no instruction)",
    "rephrase": "rephrased question",
}
VARIANT_SHORT = {
    "origcountry": "orig (country)",
    "origlast": "orig (last)",
    "nocountry": "no country",
    "nameonly": "name only",
    "rephrase": "rephrase",
}
# (name, reference, test)：Δ = test − reference（正 = 变体更差）
PAIRS = (
    ("instruction removed", "origcountry", "nameonly"),
    ("country name removed", "origlast", "nocountry"),
    ("question rephrased", "origlast", "rephrase"),
    ("token position (last vs country)", "origcountry", "origlast"),
)
HEADLINE_SETS = ("std13", "raw11")
HEADLINE_LAYER = 11


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return [float("nan"), float("nan")]
    idx = rng.integers(0, len(values), size=(N_BOOT, len(values)))
    means = values[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def index_records(records: list[dict]) -> dict:
    indexed = {}
    for record in records:
        if record.get("finite") and "pred_valid" in record:
            indexed[(record["variant"], record["probe_set"], record["layer"])] = record
    return indexed


def get(indexed, variant, probe_set, layer):
    return indexed.get((variant, probe_set, layer))


def predict_array(record) -> np.ndarray:
    return np.asarray(record["pred_valid"], dtype=np.float64)


def distances(record, truth: np.ndarray) -> np.ndarray:
    pred = predict_array(record)
    return haversine_km(pred[:, 0], pred[:, 1], truth[:, 0], truth[:, 1])


def load_strata(world_dir: Path) -> dict:
    train = pd.read_csv(world_dir / "train.csv")
    valid = pd.read_csv(world_dir / "valid.csv")
    seen_countries = set(train["country"])
    return {
        "country_seen": np.array([country in seen_countries for country in valid["country"]]),
        "countries": valid["country"].tolist(),
    }


def headline_table(indexed, truth, strata, probe_sets=HEADLINE_SETS) -> dict:
    table = {}
    for probe_set in probe_sets:
        layer = HEADLINE_LAYER if probe_set == "std13" else 11
        rows = {}
        for variant in VARIANT_ORDER:
            record = get(indexed, variant, probe_set, layer)
            if record is None:
                continue
            dist = distances(record, truth)
            rows[variant] = {
                "layer": layer,
                "mse": record["mse"],
                "periodic_mse": record["periodic_mse"],
                "dist_median_km": float(np.median(dist)),
                "dist_mean_km": float(np.mean(dist)),
                "dist_p90_km": float(np.percentile(dist, 90)),
                "hit100_rate": float(np.mean(dist <= 100.0)),
            }
        table[probe_set] = rows
    return table


def pair_effects(indexed, truth, strata, rng) -> list[dict]:
    effects = []
    seen = strata["country_seen"]
    for probe_set in HEADLINE_SETS:
        layer = HEADLINE_LAYER if probe_set == "std13" else 11
        for name, ref_variant, test_variant in PAIRS:
            ref = get(indexed, ref_variant, probe_set, layer)
            test = get(indexed, test_variant, probe_set, layer)
            if ref is None or test is None:
                continue
            ref_dist = distances(ref, truth)
            test_dist = distances(test, truth)
            delta = test_dist - ref_dist
            ref_pred = predict_array(ref)
            test_pred = predict_array(test)
            shift = haversine_km(ref_pred[:, 0], ref_pred[:, 1], test_pred[:, 0], test_pred[:, 1])
            effects.append(
                {
                    "pair": name,
                    "reference": ref_variant,
                    "test": test_variant,
                    "probe_set": probe_set,
                    "layer": layer,
                    "n": int(len(delta)),
                    "mean_delta_km": float(np.mean(delta)),
                    "median_delta_km": float(np.median(delta)),
                    "ci95_delta_km": bootstrap_ci(delta, rng),
                    "test_closer": int(np.sum(test_dist < ref_dist)),
                    "reference_closer": int(np.sum(ref_dist < test_dist)),
                    "mean_shift_km": float(np.mean(shift)),
                    "median_shift_km": float(np.median(shift)),
                    "ci95_shift_km": bootstrap_ci(shift, rng),
                    "mean_ref_km": float(np.mean(ref_dist)),
                    "mean_test_km": float(np.mean(test_dist)),
                    "strata": {
                        "country_seen": {
                            "n": int(np.sum(seen)),
                            "mean_delta_km": float(np.mean(delta[seen])) if seen.any() else None,
                            "ci95_delta_km": bootstrap_ci(delta[seen], rng),
                        },
                        "country_unseen": {
                            "n": int(np.sum(~seen)),
                            "mean_delta_km": float(np.mean(delta[~seen])) if (~seen).any() else None,
                            "ci95_delta_km": bootstrap_ci(delta[~seen], rng),
                        },
                    },
                }
            )
    return effects


def layer_curves(indexed, truth) -> dict:
    curves = {}
    for variant in VARIANT_ORDER:
        rows = []
        for (v, probe_set, layer), record in sorted(indexed.items(), key=lambda item: item[0][2]):
            if v == variant and probe_set == "std13":
                rows.append(
                    {
                        "layer": layer,
                        "mse": record["mse"],
                        "periodic_mse": record["periodic_mse"],
                        "dist_median_km": record["dist_median_km"],
                    }
                )
        if rows:
            curves[variant] = rows
    return curves


def fig_layer_curves(fig_dir, curves):
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for index, variant in enumerate(VARIANT_ORDER):
        rows = curves.get(variant)
        if not rows:
            continue
        layers = [row["layer"] for row in rows]
        mse = [row["mse"] for row in rows]
        median = [row["dist_median_km"] for row in rows]
        color = colors[index % len(colors)]
        axes[0].plot(layers, mse, marker="o", markersize=4, linewidth=1.5, color=color, label=VARIANT_LABEL[variant])
        axes[1].plot(layers, median, marker="o", markersize=4, linewidth=1.5, color=color, label=VARIANT_LABEL[variant])
    for ax in axes:
        ax.axvline(11, color="0.6", linestyle=":", linewidth=1.0)
        ax.set_xlabel("decoder block (0-based)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7.5)
    axes[0].set_ylabel("valid raw MSE")
    axes[0].set_title("raw MSE vs layer")
    axes[1].set_ylabel("median great-circle distance (km)")
    axes[1].set_title("distance vs layer")
    fig.suptitle("C2: coordinate decodability across prompt variants (z-scored features, world train→valid)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(fig_dir / "c2_layer_curves.png", dpi=150)
    fig.savefig(fig_dir / "c2_layer_curves.pdf")
    plt.close(fig)


def fig_variant_bars(fig_dir, headline, distances_by_variant):
    """std13-L11（5 变体齐全）汇总：MSE、median/P90、距离 CDF。"""
    rows = headline.get("std13", {})
    variants = [variant for variant in VARIANT_ORDER if variant in rows]
    if not variants:
        return
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    x = np.arange(len(variants))
    width = 0.38
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.8))

    ax = axes[0]
    mse = [rows[var]["mse"] for var in variants]
    ax.bar(x, mse, 0.55, color=colors[0])
    ax.set_yscale("log")
    for position, value in enumerate(mse):
        ax.text(position, value * 1.03, f"{value:.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, [VARIANT_SHORT[v] for v in variants], rotation=22, ha="right", fontsize=8)
    ax.set_ylabel("raw MSE (log scale)")
    ax.set_title("coordinate MSE")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    median = [rows[var]["dist_median_km"] for var in variants]
    p90 = [rows[var]["dist_p90_km"] for var in variants]
    ax.bar(x - width / 2, median, width, color=colors[0], label="median")
    ax.bar(x + width / 2, p90, width, color=colors[1], label="P90")
    for position, value in enumerate(median):
        ax.text(position - width / 2, value * 1.02, f"{value:.0f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x, [VARIANT_SHORT[v] for v in variants], rotation=22, ha="right", fontsize=8)
    ax.set_ylabel("great-circle distance (km)")
    ax.set_title("median / P90 distance to truth")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[2]
    for index, variant in enumerate(variants):
        dist = np.sort(distances_by_variant[variant])
        fraction = np.arange(1, len(dist) + 1) / len(dist)
        ax.plot(dist, fraction, linewidth=1.6, color=colors[index % len(colors)],
                label=VARIANT_SHORT[variant])
    ax.axvline(1000, color="0.6", linestyle=":", linewidth=1.0)
    ax.set_xlabel("great-circle distance (km)")
    ax.set_ylabel("cumulative fraction of valid items")
    ax.set_title("distance CDF")
    ax.set_xlim(0, 8000)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)

    fig.suptitle("C2: prompt variants with z-scored features, std13 L11 probe (world valid n=130)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(fig_dir / "c2_variant_bars.png", dpi=150)
    fig.savefig(fig_dir / "c2_variant_bars.pdf")
    plt.close(fig)


def fig_paired_effects(fig_dir, effects):
    if not effects:
        return
    labels, deltas, delta_cis, shifts, shift_cis = [], [], [], [], []
    for effect in effects:
        labels.append(f"{effect['pair']}\n[{effect['probe_set']}]")
        deltas.append(effect["mean_delta_km"])
        delta_cis.append(effect["ci95_delta_km"])
        shifts.append(effect["mean_shift_km"])
        shift_cis.append(effect["ci95_shift_km"])
    y = np.arange(len(effects))[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 0.62 * len(effects) + 2.2), sharey=True)
    for ax, values, cis, title, color in (
        (axes[0], deltas, delta_cis, "error change: mean Δ distance to truth (km)\npositive = variant worse", "#d62728"),
        (axes[1], shifts, shift_cis, "answer shift: mean distance between predictions (km)\nhow much the prompt moves the answer", "#1f77b4"),
    ):
        ax.errorbar(
            values,
            y,
            xerr=[[value - ci[0] for value, ci in zip(values, cis)], [ci[1] - value for value, ci in zip(values, cis)]],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=3,
            markersize=5,
        )
        ax.axvline(0, color="0.4", linewidth=0.8)
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("km")
        ax.grid(axis="x", alpha=0.3)
    axes[0].set_yticks(y, labels, fontsize=8)
    fig.suptitle("C2: paired prompt-variant effects (world valid n=130, bootstrap 95% CI)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(fig_dir / "c2_paired_effects.png", dpi=150)
    fig.savefig(fig_dir / "c2_paired_effects.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metrics", type=Path, default=ROOT / "results/slurm_c2/c2_v1/c2_metrics.json")
    parser.add_argument("--world-dir", type=Path, default=ROOT / "datasets/world")
    parser.add_argument("--fig-dir", type=Path, default=Path(__file__).with_name("figures"))
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("c2_analysis.json"))
    args = parser.parse_args()

    payload = json.loads(args.metrics.read_text(encoding="utf-8"))
    indexed = index_records(payload["records"])
    truth = np.asarray(payload.get("valid_truth"), dtype=np.float64)
    if truth.size == 0:
        truth = pd.read_csv(args.world_dir / "valid.csv")[["latitude", "longitude"]].to_numpy(np.float64)
    strata = load_strata(args.world_dir)
    rng = np.random.default_rng(SEED)

    headline = headline_table(indexed, truth, strata)
    effects = pair_effects(indexed, truth, strata, rng)
    curves = layer_curves(indexed, truth)
    distances_by_variant = {
        variant: distances(record, truth)
        for variant in VARIANT_ORDER
        if (record := get(indexed, variant, "std13", HEADLINE_LAYER)) is not None
    }
    missing_raw11 = [variant for variant in VARIANT_ORDER if get(indexed, variant, "raw11", 11) is None]

    anchors = {
        "origcountry_raw11_mse": next(
            (record["mse"] for (v, s, l), record in indexed.items() if v == "origcountry" and s == "raw11"),
            None,
        ),
        "origcountry_std13_L11_mse": next(
            (record["mse"] for (v, s, l), record in indexed.items()
             if v == "origcountry" and s == "std13" and l == HEADLINE_LAYER),
            None,
        ),
    }
    analysis = {
        "generated_utc": utcnow(),
        "metrics_path": str(args.metrics),
        "n_valid": int(len(truth)),
        "anchors": anchors,
        "anchors_expected": {"origcountry_raw11_mse": 150.42772354798632,
                             "origcountry_std13_L11_mse": 167.04130708058523},
        "missing_raw11": missing_raw11,
        "headline": headline,
        "layer_curves": curves,
        "pair_effects": effects,
    }
    args.out.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")

    fig_dir = args.fig_dir
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_layer_curves(fig_dir, curves)
    fig_variant_bars(fig_dir, headline, distances_by_variant)
    fig_paired_effects(fig_dir, effects)

    print(f"metrics: {args.metrics} ({len(payload['records'])} runs, {len(indexed)} usable)")
    print(f"anchors: raw11={anchors['origcountry_raw11_mse']}, std13-L11={anchors['origcountry_std13_L11_mse']}")
    if missing_raw11:
        print(f"raw11 missing (training failed): {missing_raw11}")
    for probe_set, rows in headline.items():
        print(f"\n== headline [{probe_set}] ==")
        for variant, row in rows.items():
            print(
                f"  {variant:<12} MSE={row['mse']:.3f} periodic={row['periodic_mse']:.3f} "
                f"median={row['dist_median_km']:.1f} km P90={row['dist_p90_km']:.1f} "
                f"hit100={100 * row['hit100_rate']:.1f}%"
            )
    print("\n== paired effects (Δ = test − reference) ==")
    for effect in effects:
        lo, hi = effect["ci95_delta_km"]
        slo, shi = effect["ci95_shift_km"]
        print(
            f"  {effect['pair']:<34} [{effect['probe_set']}] Δ={effect['mean_delta_km']:+.1f} km "
            f"CI[{lo:+.1f},{hi:+.1f}]  closer {effect['test_closer']}/{effect['n']}  "
            f"shift={effect['mean_shift_km']:.1f} km CI[{slo:.1f},{shi:.1f}]"
        )
    print(f"\nWrote {args.out}")
    print(f"figures: {fig_dir}/c2_{{layer_curves,variant_bars,paired_effects}}.{{png,pdf}}")


if __name__ == "__main__":
    main()
