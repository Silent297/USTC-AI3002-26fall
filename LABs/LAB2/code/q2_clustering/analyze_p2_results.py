"""T0017: verify retrieved P2 files, independently audit ARI and make figures.

No fitting, GPU execution or quality-metric replay. Original artifacts are read-only.
"""
import argparse
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

import extras
import p2_robustness as p2
from utils import results


def inventory(root):
    return {f.relative_to(root).as_posix(): results.sha256(f)
            for f in sorted(root.rglob("*")) if f.is_file()}


def contingency(a, b):
    _, a = np.unique(a, return_inverse=True)
    _, b = np.unique(b, return_inverse=True)
    table = np.zeros((int(a.max()) + 1, int(b.max()) + 1), dtype=np.int64)
    np.add.at(table, (a, b), 1)
    return table


def ari_from_counts(a, b):
    """Independent contingency formula (not sklearn's metric implementation)."""
    table = contingency(a, b)
    choose2 = lambda x: np.sum(x * (x - 1) // 2)
    observed = float(choose2(table))
    left, right = float(choose2(table.sum(axis=1))), float(choose2(table.sum(axis=0)))
    total = len(a) * (len(a) - 1) / 2
    expected = left * right / total
    upper = (left + right) / 2
    return 1.0 if upper == expected else (observed - expected) / (upper - expected)


def aligned_difference(a, b, subset):
    table = contingency(a, b)
    left, right = linear_sum_assignment(-table)
    _, ca = np.unique(a, return_inverse=True)
    _, cb = np.unique(b, return_inverse=True)
    mapping = np.full(table.shape[1], -1, dtype=np.int64)
    mapping[right] = left
    changed = np.flatnonzero(ca != mapping[cb])
    return dict(ari=float(adjusted_rand_score(a, b)), different_rows=len(changed),
                different_fraction=float(len(changed) / len(a)),
                different_subset_rows=len(np.intersect1d(changed, subset)))


def run(source, output):
    source = source.resolve()
    output = output.resolve()
    p2.require(source != output and source not in output.parents, "output must be outside source")
    original = inventory(source)
    output = results.create_output(output)
    summary = p2.analyze(source / "p2", output / "analysis")
    experiment = source / "p2/experiment"
    meta, cfg, report, rows, labels, restarts, _ = p2.collect(source / "p2")
    p2.require(not meta["development"] and len(rows) == 60 and len(restarts) == 180,
               "expected formal P2 60-candidate/180-restart output")
    source_checks = {}
    for name in p2.SOURCES:
        snapshot = (source / "p2/source_snapshot" / name).read_bytes().replace(b"\r\n", b"\n")
        local = (p2.PACKAGE / name).read_bytes().replace(b"\r\n", b"\n")
        source_checks[name] = snapshot == local
    p2.require(all(source_checks.values()), "local/snapshot code mismatch after newline normalization")
    audit = extras.check_output(SimpleNamespace(output=experiment, input=p2.INPUT, development=False))
    cluster_manifest = p2.read_json(experiment / "manifest.json")
    p2.require(cluster_manifest["completed_runs"] == cluster_manifest["successful_fits"] == 60
               and cluster_manifest["status"] == "completed", "unexpected experiment manifest")
    saved_summary = p2.read_json(source / "p2_analysis/summary.json")
    p2.require({k: v for k, v in summary.items() if k != "source"} ==
               {k: v for k, v in saved_summary.items() if k != "source"}, "cluster/local summary mismatch")
    csv_equal = {name: (source / "p2_analysis" / name).read_bytes() == (output / "analysis" / name).read_bytes()
                 for name in ("candidates.csv", "restarts.csv", "within_method_ari.csv",
                              "cross_method_ari.csv", "per_seed_k.csv", "paired_method_metrics.csv")}
    p2.require(all(csv_equal.values()), "cluster/local CSV mismatch")
    ari_errors = []
    for method in p2.METHODS:
        for k in cfg["candidate_k"]:
            members = [labels[f"{method}_k{k}_seed{s}"] for s in cfg["seeds"]]
            for a, b in itertools.combinations(members, 2):
                ari_errors.append(abs(ari_from_counts(a, b) - adjusted_rand_score(a, b)))
    subset = np.load(experiment / "diagnostic_indices.npy", allow_pickle=False)
    comparisons = []
    for k in cfg["candidate_k"]:
        for seed in cfg["seeds"]:
            a, b = (labels[f"{method}_k{k}_seed{seed}"] for method in p2.METHODS)
            ari_errors.append(abs(ari_from_counts(a, b) - adjusted_rand_score(a, b)))
            comparisons.append(dict(k=k, seed=seed, **aligned_difference(a, b, subset)))
    p2.require(max(ari_errors) < 1e-12 and len(ari_errors) == 300, "independent ARI audit failed")
    identities = {(r["method"], r["k"], r["seed"]): r for r in rows}
    margins, wins = [], []
    for method in p2.METHODS:
        for seed in cfg["seeds"]:
            for k in (8, 12):
                margins.append(dict(method=method, seed=seed, neighbor_k=k,
                    silhouette_k10_minus_neighbor=identities[method, 10, seed]["silhouette"] -
                                                  identities[method, k, seed]["silhouette"]))
            for metric in p2.data.METRICS:
                candidates = [identities[method, k, seed] for k in cfg["candidate_k"]]
                function = min if metric == "davies_bouldin" else max
                best = function(r[metric] for r in candidates)
                tied = sorted(r["k"] for r in candidates if abs(r[metric] - best) <= 1e-6)
                wins.append(dict(method=method, seed=seed, metric=metric, chosen_k=tied[0], tied_k=json.dumps(tied)))
    baseline = p2.PACKAGE / "outputs_eval_T0008_20261119_223025_177703"
    baseline_checks = []
    baseline_hash = results.sha256(baseline / "clusters.npy")
    for method in p2.METHODS:
        for k in cfg["candidate_k"]:
            for seed in (17, 42):
                key = f"{method}_k{k}_seed{seed}"
                old = np.load(baseline / "candidates" / key / "clusters.npy", allow_pickle=False)
                baseline_checks.append(dict(key=key, ari=float(adjusted_rand_score(old, labels[key]))))
    old_selected = np.load(baseline / "clusters.npy", allow_pickle=False)
    new_selected = np.load(experiment / "clusters.npy", allow_pickle=False)
    selected_comparison = aligned_difference(old_selected, new_selected, subset)
    p2.write_csv(output / "cross_method_alignment.csv", comparisons)
    p2.write_csv(output / "silhouette_margins.csv", margins)
    p2.write_csv(output / "all_metric_k_wins.csv", wins)
    p2.write_csv(output / "baseline_partition_checks.csv", baseline_checks)
    details = dict(quality_metrics_recomputed=False, optimizer_updates=0,
        total_ari_checked=len(ari_errors), independent_ari_max_abs_error=max(ari_errors),
        source_snapshot_normalized_equal=source_checks, cluster_local_csv_equal=csv_equal,
        exported_output_check=audit, hashed_artifacts=len(p2.read_json(source / "p2/artifact_hashes.json")),
        original_files=len(original), original_unchanged=False,
        baseline_clusters_sha256=baseline_hash, p2_clusters_sha256=results.sha256(experiment / "clusters.npy"),
        baseline_vs_p2_export=selected_comparison,
        baseline_same_seed_partition_ari=baseline_checks,
        pipeline_selection=report["selection"], elapsed_seconds=report["elapsed_seconds"],
        restart_stop_reasons={method: {str(k): {reason: sum(r["method"] == method and r["k"] == k
            and r["stop_reason"] == reason for r in restarts) for reason in sorted({r["stop_reason"] for r in restarts})}
            for k in cfg["candidate_k"]} for method in p2.METHODS})
    # Show every seed; no seed is dropped from the plotted quality distribution.
    colors = dict(kmeans="#1766a3", gmm="#cf6817")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.1))
    for ax, metric, title in zip(axes, p2.data.METRICS, ("Silhouette (higher)", "CH (higher)", "DB (lower)")):
        for method, offset in (("kmeans", -.16), ("gmm", .16)):
            means = []
            for index, k in enumerate(cfg["candidate_k"]):
                values = [identities[method, k, s][metric] for s in cfg["seeds"]]
                ax.scatter(index + offset + np.linspace(-.055, .055, len(values)), values,
                           s=22, color=colors[method], alpha=.7)
                means.append(np.mean(values))
            ax.plot(np.arange(3) + offset, means, marker="D", label=method, color=colors[method], linewidth=1)
        ax.set(xticks=np.arange(3), xticklabels=cfg["candidate_k"], xlabel="K", title=title)
        ax.grid(axis="y", alpha=.2)
        if metric == "calinski_harabasz":
            ax.set_yscale("log")
    axes[0].legend()
    fig.suptitle("P2 / Job190299: 10 seeds per group, n_init=3; dots=seeds, diamonds=means")
    fig.tight_layout()
    fig.savefig(output / "p2_metrics.png", dpi=180)
    plt.close(fig)
    matrix_labels = [f"{m}_k10_seed{s}" for m in p2.METHODS for s in cfg["seeds"]]
    matrix = np.asarray([[adjusted_rand_score(labels[a], labels[b]) for b in matrix_labels] for a in matrix_labels])
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), gridspec_kw=dict(width_ratios=[1, 1.4]))
    groups = summary["groups"]
    axes[0].scatter(np.arange(6), [g["ari"]["median"] for g in groups], marker="D", label="median")
    axes[0].scatter(np.arange(6), [g["ari"]["min"] for g in groups], marker="v", label="minimum")
    axes[0].set(xticks=np.arange(6), xticklabels=[f"{g['method']}\nK={g['k']}" for g in groups],
                ylim=(.55, 1.02), ylabel="Pairwise ARI", title="Within group: all 45 seed pairs")
    axes[0].tick_params(axis="x", labelsize=8)
    axes[0].grid(axis="y", alpha=.2)
    axes[0].legend()
    im = axes[1].imshow(matrix, vmin=.87, vmax=1, cmap="viridis")
    tick_labels = [f"{'KM' if m == 'kmeans' else 'GM'}:{s}" for m in p2.METHODS for s in cfg["seeds"]]
    axes[1].set(xticks=range(20), yticks=range(20), xticklabels=tick_labels, yticklabels=tick_labels,
                title="K=10: within and cross method ARI")
    axes[1].tick_params(axis="both", labelsize=6.5)
    plt.setp(axes[1].get_xticklabels(), rotation=90)
    axes[1].axhline(9.5, color="white", linewidth=.8)
    axes[1].axvline(9.5, color="white", linewidth=.8)
    fig.colorbar(im, ax=axes[1], shrink=.8)
    fig.tight_layout()
    fig.savefig(output / "p2_stability.png", dpi=180)
    plt.close(fig)
    p2.require(original == inventory(source), "original run changed during analysis")
    p2.require(baseline_hash == results.sha256(baseline / "clusters.npy"), "baseline export changed")
    details["original_unchanged"] = True
    results.save_json(output / "verification.json", details)
    print(json.dumps(dict(output=str(output), artifacts_verified=details["hashed_artifacts"],
                          original_files_unchanged=len(original), independent_ari_error=max(ari_errors),
                          export_comparison=selected_comparison), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.output)
