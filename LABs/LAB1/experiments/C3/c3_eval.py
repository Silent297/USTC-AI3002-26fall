# -*- coding: utf-8 -*-
"""C3 本地评估：训练集 × 层在四个评估集上的指标、配对 bootstrap、严格未见子集。

输入：results/slurm_c3/<run>/c3_metrics.json（含每个 (train_set, layer) 的四套逐条预测）。
主口径：std13（z-score + lr 2e-3 + 2000 epochs + lambda1=1e-3 + lambda2=1），
层选择以原课程协议 L11 为主；L2 是 orig_valid 上事后最优层，L32 是
new_unseen 上的最优层，均明确标注为 descriptive。

关键口径：
- 原任务 orig_valid（n=130）按国家字符串是否出现在 world train 分三层：
  seen 110 / unseen-in-C3 12 / unseen-in-both 8（Q7 的 seen/unseen 细分）；
- GeoNames test_unseen 的“未见”只对 C3 训练成立：30 国中 22 国在 world train
  出现过。因此另设 strict8 = test_unseen 中 8 个在 world train 完全没有
  国家字符串的国家（260 条），以及 hard6 = 再去掉 Niger/Palestine 的 6 国
  212 条（world train 含 'Chad / Nigeria / Niger / Cameroon'、'Israel/Palestine'
  这类复合字符串，可能含这两国的城市）；
- 配对 Δ 的 95% CI 同时给 item bootstrap 与 country-cluster bootstrap
  （按国家整簇重采样；国家数少时后者仅作敏感性参考）。

输出：c3_analysis.json + figures/c3_{layer_curves,transfer_bars,paired_effects,
strata_unseen}.{png,pdf}
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "C1"))
from c1_lib import coord_sq_error, haversine_km  # noqa: E402

SEED = 20261017
N_BOOT = 10000
TRAIN_ORDER = ("orig", "new", "new_noleak", "union", "union_noleak")
TRAIN_SOLID = ("orig", "new", "union")
TRAIN_LABEL = {
    "orig": "orig (world 500)",
    "new": "new (GeoNames 8,382)",
    "new_noleak": "new, no leak (8,368)",
    "union": "union (8,882)",
    "union_noleak": "union, no leak (8,868)",
}
TRAIN_SHORT = {
    "orig": "orig 500",
    "new": "new 8.4k",
    "union": "union 8.9k",
}
TRAIN_COLOR = {
    "orig": "#4d4d4d",
    "new": "#1f77b4",
    "new_noleak": "#9ecae1",
    "union": "#d62728",
    "union_noleak": "#f4a3a3",
}
EVAL_ORDER = ("orig_valid", "new_valid", "new_test", "new_unseen")
EVAL_LABEL = {
    "orig_valid": "world valid (130)",
    "new_valid": "GeoNames valid (1,050)",
    "new_test": "GeoNames test (1,050)",
    "new_unseen": "GeoNames unseen (1,518)",
}
LAYER_PRIMARY = 11
LAYER_ORIG_BEST = 2
LAYER_UNSEEN_BEST = 32
COUNTRY_MEAN_BASELINE = 113.77


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_payload(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["_index"] = {
        (record["train_set"], record["layer"]): record
        for record in payload["records"]
        if record.get("finite") and "predictions" in record
    }
    payload["_truth"] = {name: np.asarray(values, dtype=np.float64) for name, values in payload["eval_truth"].items()}
    return payload


def eval_countries(eval_name: str, splits_dir: Path, world_dir: Path) -> list[str]:
    split = {"orig_valid": "valid", "new_valid": "valid", "new_test": "test", "new_unseen": "test_unseen"}[eval_name]
    source = world_dir if eval_name == "orig_valid" else splits_dir
    return pd.read_csv(source / f"{split}.csv")["country"].tolist()


def eval_masks(eval_name: str, countries: list[str], splits_dir: Path, world_dir: Path) -> dict[str, np.ndarray]:
    if eval_name != "orig_valid":
        return {"all": np.ones(len(countries), dtype=bool)}
    world_train = pd.read_csv(world_dir / "train.csv")
    c3_train = pd.read_csv(splits_dir / "train.csv")
    seen = set(world_train["country"])
    in_c3 = set(c3_train["country"])
    return {
        "all": np.ones(len(countries), dtype=bool),
        "seen": np.array([country in seen for country in countries]),
        "unseen_in_c3": np.array([(country not in seen) and (country in in_c3) for country in countries]),
        "unseen_in_both": np.array([(country not in seen) and (country not in in_c3) for country in countries]),
    }


def strict_masks(countries: list[str], world_dir: Path) -> dict[str, np.ndarray]:
    world_train = pd.read_csv(world_dir / "train.csv")
    seen = set(world_train["country"])
    country_array = np.asarray(countries)
    strict = np.array([country not in seen for country in countries])
    hard = strict & ~np.isin(country_array, ["Niger", "Palestine"])
    return {"strict8": strict, "hard6": hard}


def per_item_errors(record: dict, eval_name: str, truth: np.ndarray):
    pred = np.asarray(record["predictions"][eval_name], dtype=np.float64)
    sq = coord_sq_error(pred[:, 0], pred[:, 1], truth[:, 0], truth[:, 1])
    dist = haversine_km(pred[:, 0], pred[:, 1], truth[:, 0], truth[:, 1])
    return sq, dist


def summarize(values_sq: np.ndarray, values_dist: np.ndarray) -> dict:
    if len(values_sq) == 0:
        return {"n": 0}
    return {
        "n": int(len(values_sq)),
        "mse": float(np.mean(values_sq)),
        "dist_median_km": float(np.median(values_dist)),
        "dist_mean_km": float(np.mean(values_dist)),
        "dist_p90_km": float(np.percentile(values_dist, 90)),
        "dist_max_km": float(np.max(values_dist)),
        "hit100_rate": float(np.mean(values_dist <= 100.0)),
    }


def bootstrap_item_ci(values: np.ndarray, rng: np.random.Generator) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return [float("nan"), float("nan")]
    idx = rng.integers(0, len(values), size=(N_BOOT, len(values)))
    means = values[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def bootstrap_cluster_ci(values: np.ndarray, clusters: np.ndarray, rng: np.random.Generator) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    labels, inverse = np.unique(np.asarray(clusters), return_inverse=True)
    if len(labels) <= 1 or len(values) == 0:
        return [float("nan"), float("nan")]
    groups = [values[inverse == index] for index in range(len(labels))]
    means = np.empty(N_BOOT, dtype=np.float64)
    for step in range(N_BOOT):
        pick = rng.integers(0, len(labels), size=len(labels))
        means[step] = np.concatenate([groups[index] for index in pick]).mean()
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def make_contrast(payload, eval_name, mask_name, mask, countries, test, ref, layer, group, label) -> dict:
    record_test = payload["_index"].get((test, layer))
    record_ref = payload["_index"].get((ref, layer))
    if record_test is None or record_ref is None:
        raise SystemExit(f"missing record: {(test, layer)} or {(ref, layer)}")
    truth = payload["_truth"][eval_name]
    test_sq, test_dist = per_item_errors(record_test, eval_name, truth)
    ref_sq, ref_dist = per_item_errors(record_ref, eval_name, truth)
    delta_sq = (test_sq - ref_sq)[mask]
    delta_dist = (test_dist - ref_dist)[mask]
    cluster_labels = np.asarray(countries)[mask]
    rng = np.random.default_rng(SEED)
    return {
        "group": group,
        "label": label,
        "eval": eval_name,
        "mask": mask_name,
        "test": test,
        "reference": ref,
        "layer": int(layer),
        "n": int(mask.sum()),
        "n_countries": int(len(set(cluster_labels))),
        "mse_test": float(np.mean(test_sq[mask])),
        "mse_reference": float(np.mean(ref_sq[mask])),
        "delta_mse": float(np.mean(delta_sq)),
        "ci95_item_mse": bootstrap_item_ci(delta_sq, rng),
        "ci95_cluster_mse": bootstrap_cluster_ci(delta_sq, cluster_labels, rng),
        "dist_mean_test_km": float(np.mean(test_dist[mask])),
        "dist_mean_reference_km": float(np.mean(ref_dist[mask])),
        "delta_dist_km": float(np.mean(delta_dist)),
        "ci95_item_dist_km": bootstrap_item_ci(delta_dist, rng),
        "ci95_cluster_dist_km": bootstrap_cluster_ci(delta_dist, cluster_labels, rng),
        "dist_median_test_km": float(np.median(test_dist[mask])),
        "dist_median_reference_km": float(np.median(ref_dist[mask])),
        "test_closer": int(np.sum(test_dist[mask] < ref_dist[mask])),
        "reference_closer": int(np.sum(ref_dist[mask] < test_dist[mask])),
    }


def best_layers(payload) -> dict:
    table = {}
    for train_set in TRAIN_ORDER:
        table[train_set] = {}
        for eval_name in EVAL_ORDER + ("strict8", "hard6"):
            rows = [
                (layer, record)
                for (name, layer), record in payload["_index"].items()
                if name == train_set
            ]
            if eval_name in ("strict8", "hard6"):
                # 严格子集的 MSE 需要逐条重算；放在 strict_unseen 表中
                continue
            layer, record = min(rows, key=lambda item: item[1]["metrics"][eval_name]["mse"])
            table[train_set][eval_name] = {
                "layer": int(layer),
                "mse": float(record["metrics"][eval_name]["mse"]),
            }
    return table


def strict_summary(payload, masks, countries) -> dict:
    result = {}
    for mask_name, mask in masks.items():
        subset_countries = np.asarray(countries)[mask]
        result[mask_name] = {
            "n": int(mask.sum()),
            "countries": sorted(set(subset_countries.tolist())),
            "per_country_n": {
                country: int(np.sum(subset_countries == country)) for country in sorted(set(subset_countries.tolist()))
            },
            "by_train_set": {},
        }
        for train_set in TRAIN_ORDER:
            result[mask_name]["by_train_set"][train_set] = {}
            for layer in (LAYER_ORIG_BEST, LAYER_PRIMARY, LAYER_UNSEEN_BEST):
                record = payload["_index"].get((train_set, layer))
                if record is None:
                    continue
                sq, dist = per_item_errors(record, "new_unseen", payload["_truth"]["new_unseen"])
                entry = summarize(sq[mask], dist[mask])
                entry["per_country_mse"] = {
                    country: float(np.mean(sq[mask & (np.asarray(countries) == country)]))
                    for country in sorted(set(subset_countries.tolist()))
                }
                result[mask_name]["by_train_set"][train_set][str(layer)] = entry
    return result


def layer_curves(payload) -> dict:
    curves = {}
    for train_set in TRAIN_ORDER:
        curves[train_set] = {}
        for eval_name in EVAL_ORDER:
            rows = []
            for layer in sorted(layer for name, layer in payload["_index"] if name == train_set):
                metrics = payload["_index"][(train_set, layer)]["metrics"][eval_name]
                rows.append(
                    {
                        "layer": int(layer),
                        "mse": float(metrics["mse"]),
                        "periodic_mse": float(metrics["periodic_mse"]),
                        "dist_median_km": float(metrics["dist_median_km"]),
                        "dist_p90_km": float(metrics["dist_p90_km"]),
                    }
                )
            curves[train_set][eval_name] = rows
    return curves


def fig_layer_curves(fig_dir: Path, curves: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.2))
    layers = sorted({row["layer"] for rows in next(iter(curves.values())).values() for row in rows})
    for ax, eval_name in zip(axes.flat, EVAL_ORDER):
        for train_set in TRAIN_ORDER:
            rows = curves[train_set][eval_name]
            style = "-" if train_set in TRAIN_SOLID else "--"
            ax.plot(
                [row["layer"] for row in rows],
                [row["mse"] for row in rows],
                marker="o" if train_set in TRAIN_SOLID else None,
                markersize=3.5,
                linewidth=1.6 if train_set in TRAIN_SOLID else 1.1,
                linestyle=style,
                color=TRAIN_COLOR[train_set],
                label=TRAIN_LABEL[train_set],
                alpha=1.0 if train_set in TRAIN_SOLID else 0.8,
            )
        ax.axvline(LAYER_PRIMARY, color="0.55", linestyle=":", linewidth=1.0)
        ax.axvline(LAYER_ORIG_BEST, color="0.75", linestyle=":", linewidth=0.8)
        if eval_name == "orig_valid":
            ax.axhline(COUNTRY_MEAN_BASELINE, color="0.4", linestyle="--", linewidth=1.0)
            ax.text(layers[-1], COUNTRY_MEAN_BASELINE * 1.03, "country-mean oracle 113.8",
                    ha="right", va="bottom", fontsize=7, color="0.35")
        ax.set_yscale("log")
        ax.set_xlabel("decoder block (0-based)")
        ax.set_ylabel("raw coordinate MSE (log)")
        ax.set_title(f"{EVAL_LABEL[eval_name]}", fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xticks(layers[::2])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("C3: probe error by training set and layer (std13; dotted L11, lighter dotted L2)",
                 fontsize=11, y=0.995)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.962), ncol=5,
               fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    fig.savefig(fig_dir / "c3_layer_curves.png", dpi=150)
    fig.savefig(fig_dir / "c3_layer_curves.pdf")
    plt.close(fig)


def fig_transfer_bars(fig_dir: Path, payload, masks, countries) -> None:
    eval_names = ("orig_valid", "new_valid", "new_test", "new_unseen", "strict8")
    labels = ("world\n130", "Geo valid\n1,050", "Geo test\n1,050",
              "Geo unseen\n1,518", "strict\n260 / 8")
    values = {}
    for layer in (LAYER_ORIG_BEST, LAYER_PRIMARY, LAYER_UNSEEN_BEST):
        values[layer] = {}
        for eval_name in eval_names:
            for train_set in TRAIN_SOLID:
                record = payload["_index"][(train_set, layer)]
                if eval_name == "strict8":
                    sq, _ = per_item_errors(record, "new_unseen", payload["_truth"]["new_unseen"])
                    values[layer][(eval_name, train_set)] = float(np.mean(sq[masks["strict8"]]))
                else:
                    values[layer][(eval_name, train_set)] = float(record["metrics"][eval_name]["mse"])

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.0), sharey=True)
    x = np.arange(len(eval_names))
    width = 0.26
    for ax, layer in zip(axes, (LAYER_ORIG_BEST, LAYER_PRIMARY, LAYER_UNSEEN_BEST)):
        for index, train_set in enumerate(TRAIN_SOLID):
            vals = [values[layer][(eval_name, train_set)] for eval_name in eval_names]
            positions = x + (index - 1) * width
            ax.bar(positions, vals, width, color=TRAIN_COLOR[train_set], label=TRAIN_LABEL[train_set])
            for position, value in zip(positions, vals):
                ax.text(position, value * 1.04, f"{value:.0f}", ha="center", va="bottom", fontsize=6.5, rotation=90)
        ax.set_yscale("log")
        ax.set_ylim(top=ax.get_ylim()[1] * 1.3)
        ax.set_xticks(x, labels, fontsize=7.5)
        ax.set_title(f"layer {layer}" + (" (selected on orig valid)" if layer == LAYER_ORIG_BEST else
                                         " (course L11)" if layer == LAYER_PRIMARY else
                                         " (best on full unseen)"), fontsize=9.5)
        ax.set_ylabel("raw coordinate MSE (log)")
        ax.grid(axis="y", alpha=0.3)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.suptitle("C3: fixed-layer comparison across evaluation sets (lower is better)", fontsize=11, y=0.995)
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=3,
               fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(fig_dir / "c3_transfer_bars.png", dpi=150)
    fig.savefig(fig_dir / "c3_transfer_bars.pdf")
    plt.close(fig)


def fig_paired_effects(fig_dir: Path, contrasts: list[dict]) -> None:
    groups = (("original", "A. original task (world valid, n=130)"),
              ("new", "B. GeoNames task and strict unseen subsets"))
    fig, axes = plt.subplots(1, 2, figsize=(15.8, 10.0), gridspec_kw={"width_ratios": [1.0, 1.15]})
    for ax, (group, title) in zip(axes, groups):
        rows = [row for row in contrasts if row["group"] == group]
        y = np.arange(len(rows))[::-1]
        for position, row in zip(y, rows):
            item_lo, item_hi = row["ci95_item_mse"]
            cluster_lo, cluster_hi = row["ci95_cluster_mse"]
            ax.plot([cluster_lo, cluster_hi], [position, position], color="0.62", linewidth=5.0,
                    solid_capstyle="butt", alpha=0.55, zorder=1)
            ax.errorbar(
                row["delta_mse"], position,
                xerr=[[row["delta_mse"] - item_lo], [item_hi - row["delta_mse"]]],
                fmt="o", color="#b2182b", ecolor="#b2182b", capsize=3, markersize=5, zorder=3,
            )
        ax.axvline(0, color="0.3", linewidth=0.9)
        ax.grid(axis="x", alpha=0.3)
        ax.set_yticks(y, [row["label"] for row in rows], fontsize=7.5)
        ax.set_xlabel("paired Δ MSE (test − reference; <0 = test better)")
        ax.set_title(title, fontsize=10)
        ax.set_ylim(-0.7, len(rows) - 0.3)
    fig.suptitle("C3: paired effects (negative = second training set is better)", fontsize=11, y=0.995)
    fig.legend(
        [plt.Line2D([], [], color="#b2182b", marker="o", linewidth=1.2),
         plt.Line2D([], [], color="0.62", linewidth=5.0, alpha=0.55)],
        ["mean Δ with item-bootstrap 95% CI", "country-cluster bootstrap 95% CI"],
        loc="upper center", bbox_to_anchor=(0.5, 0.962), ncol=2, fontsize=8, frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.925), w_pad=3.0)
    fig.savefig(fig_dir / "c3_paired_effects.png", dpi=150)
    fig.savefig(fig_dir / "c3_paired_effects.pdf")
    plt.close(fig)


def fig_strata_unseen(fig_dir: Path, payload, strict: dict, strata: dict, countries) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.9))
    ax = axes[0]
    mask_order = ("seen", "unseen_in_c3", "unseen_in_both")
    mask_label = ("seen in\nworld train", "unseen in world,\nseen in C3", "unseen\nin both")
    x = np.arange(len(mask_order))
    width = 0.26
    for index, train_set in enumerate(TRAIN_SOLID):
        vals = [strata[train_set][mask_name][str(LAYER_PRIMARY)]["mse"] for mask_name in mask_order]
        positions = x + (index - 1) * width
        ax.bar(positions, vals, width, color=TRAIN_COLOR[train_set], label=TRAIN_SHORT[train_set])
        for position, value in zip(positions, vals):
            ax.text(position, value * 1.02, f"{value:.0f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x, [f"{label}\n(n={strata['n'][name]})" for label, name in
                      zip(mask_label, mask_order)], fontsize=8)
    ax.set_ylabel("raw coordinate MSE")
    ax.set_title(f"world valid strata, layer {LAYER_PRIMARY}", fontsize=9.5)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.12)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7.5)

    country_names = sorted(strict["strict8"]["per_country_n"])
    for ax, layer in zip(axes[1:], (LAYER_PRIMARY, LAYER_UNSEEN_BEST)):
        for country in country_names:
            vals = [strict["strict8"]["by_train_set"][train_set][str(layer)]["per_country_mse"][country]
                    for train_set in TRAIN_SOLID]
            ax.plot([TRAIN_SHORT[name] for name in TRAIN_SOLID], vals, marker="o", markersize=4,
                    linewidth=1.2, label=country)
        ax.set_yscale("log")
        ax.set_ylabel("per-country MSE (log)")
        ax.set_title(f"strict unseen: per-country MSE, layer {layer}", fontsize=9.5)
        ax.grid(alpha=0.3)
        ax.tick_params(axis="x", labelsize=8)
    handles, legend_labels = axes[1].get_legend_handles_labels()
    fig.suptitle(
        "C3: augmentation helps countries absent from the original training set, with heterogeneous per-country effects",
        fontsize=10.5, y=0.995,
    )
    fig.legend(handles, legend_labels, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=8,
               fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 0.925))
    fig.savefig(fig_dir / "c3_strata_unseen.png", dpi=150)
    fig.savefig(fig_dir / "c3_strata_unseen.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metrics", type=Path, default=ROOT / "results/slurm_c3/c3_v1/c3_metrics.json")
    parser.add_argument("--splits-dir", type=Path, default=Path(__file__).with_name("data") / "splits")
    parser.add_argument("--world-dir", type=Path, default=ROOT / "datasets" / "world")
    parser.add_argument("--fig-dir", type=Path, default=Path(__file__).with_name("figures"))
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("c3_analysis.json"))
    args = parser.parse_args()

    payload = load_payload(args.metrics)
    truth = payload["_truth"]
    countries_by_eval = {
        eval_name: eval_countries(eval_name, args.splits_dir, args.world_dir) for eval_name in EVAL_ORDER
    }
    masks_by_eval = {
        eval_name: eval_masks(eval_name, countries_by_eval[eval_name], args.splits_dir, args.world_dir)
        for eval_name in EVAL_ORDER
    }
    unseen_countries = countries_by_eval["new_unseen"]
    strict = strict_masks(unseen_countries, args.world_dir)

    curves = layer_curves(payload)
    best = best_layers(payload)
    strict_table = strict_summary(payload, strict, unseen_countries)

    strata = {"n": {name: int(mask.sum()) for name, mask in masks_by_eval["orig_valid"].items()}}
    for train_set in TRAIN_SOLID + ("new_noleak", "union_noleak"):
        strata[train_set] = {}
        for mask_name, mask in masks_by_eval["orig_valid"].items():
            strata[train_set][mask_name] = {}
            for layer in (LAYER_ORIG_BEST, LAYER_PRIMARY):
                record = payload["_index"][(train_set, layer)]
                sq, dist = per_item_errors(record, "orig_valid", truth["orig_valid"])
                strata[train_set][mask_name][str(layer)] = summarize(sq[mask], dist[mask])

    contrasts = []
    orig_masks = masks_by_eval["orig_valid"]
    orig_countries = countries_by_eval["orig_valid"]

    def add_orig(mask_name, test, ref, layer, label):
        contrasts.append(make_contrast(
            payload, "orig_valid", mask_name, orig_masks[mask_name], orig_countries, test, ref, layer,
            "original", label,
        ))

    for layer in (LAYER_ORIG_BEST, LAYER_PRIMARY):
        add_orig("all", "union", "orig", layer, f"union−orig | overall L{layer}")
    for mask_name, label in (("seen", "seen (110)"), ("unseen_in_c3", "unseen→C3 (12)"),
                             ("unseen_in_both", "unseen both (8)")):
        for layer in (LAYER_ORIG_BEST, LAYER_PRIMARY):
            add_orig(mask_name, "union", "orig", layer, f"union−orig | {label} L{layer}")
    for layer in (LAYER_ORIG_BEST, LAYER_PRIMARY):
        add_orig("all", "new", "orig", layer, f"new−orig | overall L{layer}")
        add_orig("all", "union", "new", layer, f"union−new | overall L{layer}")
        # 泄漏控制：noleak 与完整增广的差异应接近 0
        add_orig("all", "union_noleak", "union", layer, f"leak: union_noleak−union L{layer}")

    def add_new(eval_name, mask_name, mask, test, ref, layer, label):
        contrasts.append(make_contrast(
            payload, eval_name, mask_name, mask, countries_by_eval[eval_name], test, ref, layer,
            "new", label,
        ))

    all_new = masks_by_eval["new_unseen"]["all"]
    for eval_name in ("new_valid", "new_test"):
        short = "valid" if eval_name == "new_valid" else "test"
        add_new(eval_name, "all", masks_by_eval[eval_name]["all"], "union", "new", LAYER_PRIMARY,
                f"union−new | Geo {short} L{LAYER_PRIMARY}")
        add_new(eval_name, "all", masks_by_eval[eval_name]["all"], "new", "orig", LAYER_PRIMARY,
                f"new−orig | Geo {short} L{LAYER_PRIMARY}")
    for test, ref in (("union", "new"), ("new", "orig")):
        add_new("new_unseen", "all", all_new, test, ref, LAYER_PRIMARY,
                f"{test}−{ref} | unseen all L{LAYER_PRIMARY}")
    for mask_name in ("strict8", "hard6"):
        for test, ref in (("union", "new"), ("new", "orig"), ("union", "orig")):
            for layer in (LAYER_PRIMARY, LAYER_UNSEEN_BEST):
                add_new("new_unseen", mask_name, strict[mask_name], test, ref, layer,
                        f"{test}−{ref} | {mask_name} L{layer}")
    add_new("new_test", "all", masks_by_eval["new_test"]["all"], "union_noleak", "union", LAYER_PRIMARY,
            f"leak: union_noleak−union | Geo test L{LAYER_PRIMARY}")
    add_new("new_test", "all", masks_by_eval["new_test"]["all"], "new_noleak", "new", LAYER_PRIMARY,
            f"leak: new_noleak−new | Geo test L{LAYER_PRIMARY}")

    analysis = {
        "generated_utc": utcnow(),
        "metrics_path": str(args.metrics),
        "n_boot": N_BOOT,
        "seed": SEED,
        "anchors": payload.get("anchor_checks", []),
        "datasets": {
            "orig_valid": {
                "n": len(orig_countries),
                "n_countries": len(set(orig_countries)),
                "strata_n": strata["n"],
            },
            "new_unseen": {
                "n": len(unseen_countries),
                "n_countries": len(set(unseen_countries)),
                "strict8_n": int(strict["strict8"].sum()),
                "hard6_n": int(strict["hard6"].sum()),
                "strict8_countries": sorted(set(np.asarray(unseen_countries)[strict["strict8"]].tolist())),
                "hard6_countries": sorted(set(np.asarray(unseen_countries)[strict["hard6"]].tolist())),
                "world_train_overlap_countries": sorted(
                    set(unseen_countries) & set(pd.read_csv(args.world_dir / "train.csv")["country"])
                ),
            },
        },
        "best_layers": best,
        "layer_curves": curves,
        "strata": strata,
        "strict_unseen": strict_table,
        "contrasts": contrasts,
        "leakage": [row for row in contrasts if "leak" in row["label"]],
        "headline_layer11": {
            train_set: {
                eval_name: payload["_index"][(train_set, LAYER_PRIMARY)]["metrics"][eval_name]
                for eval_name in EVAL_ORDER
            }
            for train_set in TRAIN_ORDER
        },
        "headline_layer2": {
            train_set: {
                eval_name: payload["_index"][(train_set, LAYER_ORIG_BEST)]["metrics"][eval_name]
                for eval_name in EVAL_ORDER
            }
            for train_set in TRAIN_ORDER
        },
    }
    args.out.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")

    args.fig_dir.mkdir(parents=True, exist_ok=True)
    fig_layer_curves(args.fig_dir, curves)
    fig_transfer_bars(args.fig_dir, payload, strict, unseen_countries)
    fig_paired_effects(args.fig_dir, contrasts)
    fig_strata_unseen(args.fig_dir, payload, strict_table, strata, unseen_countries)

    print(f"metrics: {args.metrics} ({len(payload['records'])} runs, {len(payload['_index'])} usable)")
    failed_anchor = [check for check in payload.get("anchor_checks", []) if not check["ok"]]
    print(f"anchors: {len(payload.get('anchor_checks', [])) - len(failed_anchor)}/{len(payload.get('anchor_checks', []))} ok")
    print("\n== best layer per train set / eval (MSE) ==")
    for train_set, row in best.items():
        print(f"  {train_set:13s} " + "  ".join(
            f"{eval_name}=L{entry['layer']}({entry['mse']:.1f})" for eval_name, entry in row.items()
        ))
    print("\n== orig_valid strata (MSE, layer 11) ==")
    for mask_name in ("seen", "unseen_in_c3", "unseen_in_both"):
        print(f"  {mask_name:14s} n={strata['n'][mask_name]:3d} " + "  ".join(
            f"{train_set}={strata[train_set][mask_name][str(LAYER_PRIMARY)]['mse']:.1f}"
            for train_set in TRAIN_SOLID
        ))
    print("\n== strict unseen (layer 11 / layer 32) ==")
    for mask_name in ("strict8", "hard6"):
        for layer in (LAYER_PRIMARY, LAYER_UNSEEN_BEST):
            print(f"  {mask_name} L{layer}: " + "  ".join(
                f"{train_set}={strict_table[mask_name]['by_train_set'][train_set][str(layer)]['mse']:.1f}"
                for train_set in TRAIN_SOLID
            ))
    print("\n== paired contrasts (ΔMSE, item CI / cluster CI) ==")
    for row in contrasts:
        item = row["ci95_item_mse"]
        cluster = row["ci95_cluster_mse"]
        print(
            f"  {row['label']:<46s} Δ={row['delta_mse']:+8.1f} "
            f"item[{item[0]:+7.1f},{item[1]:+7.1f}] cluster[{cluster[0]:+7.1f},{cluster[1]:+7.1f}] "
            f"Δdist={row['delta_dist_km']:+7.1f}km"
        )
    print(f"\nWrote {args.out}")
    print(f"figures: {args.fig_dir}/c3_{{layer_curves,transfer_bars,paired_effects,strata_unseen}}.{{png,pdf}}")


if __name__ == "__main__":
    main()
