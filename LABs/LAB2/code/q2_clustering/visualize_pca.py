"""PCA display of frozen Q2 labels; fit on the fixed 3000-row subset only.

Lightweight CPU analysis: no clustering, no GPU, no full-data PCA fitting.
All 56000 rows are projected into one shared coordinate system.
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

PACKAGE = Path(__file__).resolve().parent
BASE = PACKAGE / "outputs_eval_T0008_20261119_223025_177703"
P2 = PACKAGE / "outputs_T0016_20261123_224500_190299/p2/experiment"
FEATURES = PACKAGE / "data/features.npz"
FEATURE_HASH = "a07578f6e1ab9d17667b987b25b1b9bc1f06a5f1f678a127e31dc78bc5ff0b38"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def basis(sample):
    X = np.asarray(sample, dtype=np.float64)
    mean = X.mean(axis=0)
    centered = X - mean
    covariance = centered.T @ centered / (len(X) - 1)
    eigenvalues, vectors = np.linalg.eigh(covariance)
    components = vectors[:, -2:][:, ::-1].T.copy()
    # Fix eigenvector signs for reproducible display orientation.
    for component in components:
        if component[np.abs(component).argmax()] < 0:
            component *= -1
    values = eigenvalues[-2:][::-1].copy()
    total = float(np.trace(covariance))
    require(total > 0 and np.isfinite(components).all(), "invalid PCA covariance")
    np.testing.assert_allclose(components @ components.T, np.eye(2), atol=1e-12)
    np.testing.assert_allclose(covariance @ components.T, components.T * values, rtol=1e-10, atol=1e-12)
    return mean, components, values / total, values, total


def toy_check():
    X = np.array([[2., 0., 0.], [-2., 0., 0.], [0., 1., 0.], [0., -1., 0.]])
    before = X.copy()
    mean, components, ratio, _, _ = basis(X)
    np.testing.assert_allclose(ratio, [.8, .2], atol=1e-12)
    np.testing.assert_allclose(((X - mean) @ components.T) @ components + mean, X, atol=1e-12)
    np.testing.assert_array_equal(before, X)


def load_partition(directory, key):
    records = read_json(directory / "runs.json")
    row = next(r for r in records if r["key"] == key)
    path = directory / "candidates" / key / "clusters.npy"
    require(row["status"] == "succeeded" and digest(path) == row["clusters_sha256"], key + " label identity mismatch")
    labels = np.load(path, allow_pickle=False)
    require(labels.shape == (56000,) and labels.dtype == np.int32 and set(np.unique(labels)) == set(range(10)),
            key + " invalid labels")
    return labels, path, row


def align(reference, labels):
    table = np.zeros((10, 10), dtype=np.int64)
    np.add.at(table, (reference, labels), 1)
    left, right = linear_sum_assignment(-table)
    mapping = np.empty(10, dtype=np.int32)
    mapping[right] = left
    return mapping[labels], mapping


def run(output):
    require(not output.exists(), "choose a new output directory")
    toy_check()
    require(digest(FEATURES) == FEATURE_HASH, "feature hash mismatch")
    cfg = read_json(BASE / "config.json")
    selection = read_json(BASE / "selection.json")
    require(selection["method"] == "kmeans" and selection["k"] == 10, "unexpected selected clustering")
    indices_path = BASE / "diagnostic_indices.npy"
    indices = np.load(indices_path, allow_pickle=False)
    expected = np.sort(np.random.default_rng(cfg["silhouette_seed"]).choice(56000, 3000, replace=False))
    require(cfg["silhouette_sample_size"] == 3000 and np.array_equal(indices, expected), "evaluation subset mismatch")
    selected, selected_path, selected_row = load_partition(BASE, selection["run_key"])
    require(digest(BASE / "clusters.npy") == digest(selected_path), "selected export mismatch")
    standard, standard_path, standard_row = load_partition(BASE, "gmm_k10_seed17")
    different, different_path, different_row = load_partition(P2, "gmm_k10_seed20261123")
    input_paths = [FEATURES, indices_path, selected_path, standard_path, different_path, BASE / "clusters.npy",
                   BASE / "selection.json", BASE / "config.json", BASE / "runs.json", P2 / "runs.json"]
    original = {str(p): digest(p) for p in input_paths}
    with np.load(FEATURES, allow_pickle=False) as archive:
        require(set(archive.files) == {"X"}, "unexpected feature archive")
        X = archive["X"]
    require(X.shape == (56000, 512) and X.dtype == np.float32 and np.isfinite(X).all(), "invalid features")
    output.mkdir(parents=True)
    with threadpool_limits(limits=4):
        mean, components, sample_ratio, eigenvalues, sample_total = basis(X[indices])
        full_mean = X.mean(axis=0, dtype=np.float64)
        scores = np.empty((56000, 2), dtype=np.float64)
        full_total = 0.
        for start in range(0, len(X), 4096):
            block = X[start:start + 4096].astype(np.float64)
            scores[start:start + len(block)] = (block - mean) @ components.T
            full_total += float(np.square(block - full_mean).sum())
        full_total /= len(X) - 1
        full_ratio = scores.var(axis=0, ddof=1) / full_total
        subset_scores = (X[indices].astype(np.float64) - mean) @ components.T
        np.testing.assert_allclose(scores[indices], subset_scores, atol=1e-12)
        np.testing.assert_allclose(subset_scores.var(axis=0, ddof=1), eigenvalues, rtol=1e-10, atol=1e-12)
    require(np.isfinite(scores).all() and 0 < sample_ratio.sum() <= 1 + 1e-12
            and 0 < full_ratio.sum() <= 1 + 1e-12, "invalid variance ratios")
    np.save(output / "embedding.npy", scores)
    np.save(output / "fit_indices.npy", indices)
    np.savez_compressed(output / "pca_basis.npz", mean=mean, components=components,
                        eigenvalues=eigenvalues, sample_variance_ratio=sample_ratio,
                        full_projected_variance_ratio=full_ratio)
    standard_aligned, standard_mapping = align(selected, standard)
    different_aligned, different_mapping = align(selected, different)
    subsets = [selected, standard_aligned, different_aligned]
    names = ["KMeans K=10, seed17 (selected)", "GMM K=10, seed17", "GMM K=10, seed20261123 (P2)"]
    # Identical row order and colors across panels; labels do not affect PCA fitting.
    order = np.random.default_rng(20261124).permutation(len(X))
    cmap = plt.get_cmap("tab10")
    norm = BoundaryNorm(np.arange(-.5, 10.5), 10)
    span = np.ptp(scores, axis=0)
    limits = [(scores[:, j].min() - .03 * span[j], scores[:, j].max() + .03 * span[j]) for j in range(2)]

    def panel(ax, labels, title):
        scatter = ax.scatter(scores[order, 0], scores[order, 1], c=labels[order], cmap=cmap,
                             norm=norm, s=2, alpha=.4, linewidths=0, rasterized=True)
        ax.set(title=title, xlabel=f"PC1 ({sample_ratio[0]:.2%} of fit-subset variance)",
               ylabel=f"PC2 ({sample_ratio[1]:.2%} of fit-subset variance)",
               xlim=limits[0], ylim=limits[1])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=.15)
        return scatter

    fig, ax = plt.subplots(figsize=(8.2, 6.2), constrained_layout=True)
    scatter = panel(ax, selected, names[0])
    colorbar = fig.colorbar(scatter, ax=ax, ticks=range(10), shrink=.8)
    colorbar.set_label("Predicted cluster ID (not a true class label)")
    fig.suptitle(f"PCA fit: fixed 3000 rows; display: all 56000 rows\n"
                 f"Fit-subset variance retained {sample_ratio.sum():.2%}; full-data variance captured {full_ratio.sum():.2%}")
    fig.savefig(output / "q2_pca_selected.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), constrained_layout=True)
    for ax, labels, name in zip(axes, subsets, names):
        scatter = panel(ax, labels, name)
    fig.colorbar(scatter, ax=axes, ticks=range(10), shrink=.6, label="Cluster ID aligned to selected KMeans")
    fig.suptitle(f"Shared PCA coordinates; raw centered features, no standardization/whitening\n"
                 f"Fit on 3000 fixed rows, project all 56000; retained fit variance {sample_ratio.sum():.2%}")
    fig.savefig(output / "q2_pca_comparison.png", dpi=180)
    plt.close(fig)
    require(original == {str(p): digest(p) for p in input_paths}, "an input file changed during analysis")
    summary = dict(task="T0020", python=platform.python_version(), numpy=np.__version__, device="cpu",
        computation="PCA basis fitted on fixed 3000-row evaluation subset; all rows projected, no clustering",
        preprocessing="center only; no per-feature standardization; no whitening", compute_dtype="float64",
        fit_rows=3000, projected_rows=56000, original_dimensions=512, silhouette_seed=cfg["silhouette_seed"],
        draw_order_seed=20261124, sample_explained_variance_ratio=sample_ratio.tolist(),
        sample_explained_variance_ratio_sum=float(sample_ratio.sum()),
        full_data_projected_variance_ratio=full_ratio.tolist(), full_data_projected_variance_ratio_sum=float(full_ratio.sum()),
        full_data_note="Variance captured by sample-fitted directions, not a full-data fitted PCA optimum.",
        labels_used_for_pca=False, true_labels_used=False, original_input_files_unchanged=True,
        source_hashes=original, analysis_source_sha256=digest(Path(__file__).resolve()),
        partitions=[{k: row[k] for k in ("key", "method", "k", "seed", "clusters_sha256", "silhouette", "calinski_harabasz", "davies_bouldin")}
                    for row in (selected_row, standard_row, different_row)],
        gmm_seed17_mapping=standard_mapping.tolist(), gmm_p2_mapping=different_mapping.tolist(),
        gmm_seed17_different_rows=int(np.count_nonzero(selected != standard_aligned)),
        gmm_p2_different_rows=int(np.count_nonzero(selected != different_aligned)),
        gmm_seed17_ari=float(adjusted_rand_score(selected, standard)),
        gmm_p2_ari=float(adjusted_rand_score(selected, different)))
    save_json(output / "summary.json", summary)
    save_json(output / "artifact_hashes.json", {p.name: digest(p) for p in output.iterdir() if p.is_file()})
    print(json.dumps({k: summary[k] for k in ("fit_rows", "projected_rows", "sample_explained_variance_ratio", "sample_explained_variance_ratio_sum",
                     "full_data_projected_variance_ratio_sum", "gmm_seed17_different_rows", "gmm_p2_different_rows")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output.resolve())
