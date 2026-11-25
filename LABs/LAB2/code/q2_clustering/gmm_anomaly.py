"""Frozen GMM diagnostics; no fitting, no changes to assignment implementation.

local: archived labels/parameters/PCA + fixed 3000-row CPU density diagnosis.
run: Slurm CUDA full-row density diagnosis and frozen 2x2 interventions.
Both modes use direct (x-mu)^2 float64 scores; metrics in run use original
float32 CUDA evaluation kernels and original common silhouette subset.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
import torch
from threadpoolctl import threadpool_limits

PACKAGE = Path(__file__).resolve().parent
P2 = PACKAGE / "outputs_T0016_20261123_224500_190299/p2/experiment"
PCA = PACKAGE / "outputs_T0020_20261124_01"
FEATURES = PACKAGE / "data/features.npz"
KEYS = ("gmm_k10_seed17", "gmm_k10_seed20261123")
FEATURE_HASH = "a07578f6e1ab9d17667b987b25b1b9bc1f06a5f1f678a127e31dc78bc5ff0b38"
INDEX_HASH = "70d72e904d21c308ec6d82b35d9dbbfe4b79a16c2d317305205726bb0ed5b60a"
MODEL_HASHES = (
    "d93983ab450192513f02a77d8de5aca4c9ebb06155a7ab07c4cf8ca334444ff0",
    "30cec0cdb7fc01f8f529bbe7642f9fffcd2dcab4f9f6c327979a2bd4d7181f15")
LABEL_HASHES = (
    "9d98f7c06a7176051bd1e5c70731c1ae706047df34e2eb2771ce47cab93435dc",
    "cd48d86588beb1ea205192d64eb6385f53443b70a17c8c18512d1e1a7d53833a")
TERMS = ("weight", "normalization", "mahalanobis", "entropy")
VARIANTS = ("original", "uniform_weights", "shared_sphere", "uniform_shared")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                     allow_nan=False) + "\n", encoding="utf-8")


def align(reference, labels, k=10):
    """Maximum overlap permutation. The reference is NOT ground truth."""
    table = np.zeros((k, k), dtype=np.int64)
    np.add.at(table, (reference, labels), 1)
    left, right = linear_sum_assignment(-table)
    mapping = np.empty(k, dtype=np.int64)
    mapping[right] = left
    aligned = mapping[labels]
    return aligned, mapping, table[:, np.argsort(mapping)]


def variants(model):
    """Predefined scalar: prior-weighted mean original variance across dimensions.

    Fixed before effects are evaluated, no tuning, no re-estimation of means.
    """
    w, mu, v = [model[n].astype(np.float64) for n in ("weights", "means", "variances")]
    sphere = float(np.sum(w[:, None] * v) / (w.sum() * v.shape[1]))
    uniform = np.full_like(w, 1 / len(w))
    shared = np.full_like(v, sphere)
    return dict(original=(w, mu, v), uniform_weights=(uniform, mu, v),
                shared_sphere=(w, mu, shared), uniform_shared=(uniform, mu, shared)), sphere


@torch.no_grad()
def score_points(X, parameters, device="cpu", batch=256):
    """Stable direct-difference scores and exact responsibility decomposition.

    q=log(weight) - .5*sum(log(2pi*v)) - .5*sum((x-mu)^2/v).
    log p = E_r[log weight] + E_r[log normalization]
            + E_r[-.5 Mahalanobis] + H(r).
    No quadratic expansion; never subtract large similar squared norms.
    """
    w, mu, v = [torch.as_tensor(a, dtype=torch.float64, device=device) for a in parameters]
    require(bool((w > 0).all() and (v > 0).all()), "nonpositive GMM parameters")
    logw = w.log()
    normalizer = -.5 * (v.log().sum(1) + mu.shape[1] * math.log(2 * math.pi))
    chunks = {key: [] for key in ("ll", "labels", *TERMS)}
    for start in range(0, len(X), batch):
        x = torch.as_tensor(X[start:start + batch], dtype=torch.float64, device=device)
        distance = -.5 * ((x[:, None, :] - mu[None, :, :]).square() / v).sum(2)
        q = logw + normalizer + distance
        ll = torch.logsumexp(q, 1)
        logr = q - ll[:, None]
        r = logr.exp()
        values = dict(ll=ll, labels=q.argmax(1), weight=(r * logw).sum(1),
                      normalization=(r * normalizer).sum(1),
                      mahalanobis=(r * distance).sum(1), entropy=-(r * logr).sum(1))
        torch.testing.assert_close(r.sum(1), torch.ones_like(ll), atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(sum(values[t] for t in TERMS), ll, atol=1e-8, rtol=1e-10)
        for key, val in values.items():
            require(bool(torch.isfinite(val).all()), "nonfinite score " + key)
            chunks[key].append(val.cpu().numpy())
    return {key: np.concatenate(values) for key, values in chunks.items()}


def legacy_check(X, model, stable, implementation, device):
    """Same rows for float32 expanded and float64 direct scores, no refit.

    Full mode deliberately calls the original E-step once on all rows, matching
    original matrix shapes; GPU/library versions may still affect replay.
    """
    X32 = torch.as_tensor(X, dtype=torch.float32, device=device)
    args = [torch.as_tensor(model[n], dtype=torch.float32, device=device)
            for n in ("weights", "means", "variances")]
    with torch.no_grad():
        r, ll = implementation.gmm_estep(X32, *args)
    require(bool(torch.isfinite(r).all()) and math.isfinite(float(ll)), "legacy E-step nonfinite")
    return dict(expanded_float32_mean_ll=float(ll), direct_float64_mean_ll=float(stable["ll"].mean()),
                mean_ll_difference=float(ll) - float(stable["ll"].mean()),
                assignment_disagreements=int(np.count_nonzero(r.argmax(1).cpu().numpy() != stable["labels"])),
                maximum_responsibility_sum_error=float((r.sum(1) - 1).abs().max()))


def group_contributions(normal, special, changed):
    delta = special["ll"] - normal["ll"]
    answer = {}
    for name, mask in (("all", np.ones(len(delta), bool)), ("changed", changed), ("unchanged", ~changed)):
        n = int(mask.sum())
        answer[name] = dict(n=n, mean_delta_ll=float(delta[mask].mean()) if n else None,
                            median_delta_ll=float(np.median(delta[mask])) if n else None,
                            delta_quantiles_05_95_99=np.quantile(delta[mask], [.05, .95, .99]).tolist() if n else None,
                            sum_delta_ll=float(delta[mask].sum()),
                            contribution_to_total_mean=float(delta[mask].sum() / len(delta)),
                            positive_fraction=float((delta[mask] > 0).mean()) if n else None,
                            terms={t: float((special[t][mask] - normal[t][mask]).sum() / len(delta)) for t in TERMS})
    np.testing.assert_allclose(sum(answer[g]["contribution_to_total_mean"] for g in ("changed", "unchanged")),
                               delta.mean(), atol=1e-9)
    return answer


def source_equal(a, b):
    return a.read_bytes().replace(b"\r\n", b"\n") == b.read_bytes().replace(b"\r\n", b"\n")


def load_inputs(p2, pca):
    paths = [FEATURES, p2 / "runs.json", p2 / "plan.json", p2 / "config.json",
             p2 / "diagnostic_indices.npy"]
    require(digest(FEATURES) == FEATURE_HASH and digest(paths[-1]) == INDEX_HASH, "input identity mismatch")
    plan, cfg = read_json(p2 / "plan.json"), read_json(p2 / "config.json")
    require(plan["input_sha256"] == FEATURE_HASH and not cfg["standardize"]
            and cfg["dtype"] == "float32" and plan["device"] == "cuda", "unexpected P2 protocol")
    require(digest(p2 / "config.json") == plan["config_sha256"], "configuration hash mismatch")
    indices = np.load(paths[-1], allow_pickle=False)
    expected = np.sort(np.random.default_rng(20261011).choice(56000, 3000, replace=False))
    require(cfg["silhouette_seed"] == 20261011 and cfg["silhouette_sample_size"] == 3000
            and np.array_equal(indices, expected), "subset protocol mismatch")
    records = read_json(p2 / "runs.json")
    models, labels, rows = [], [], []
    for key, mh, lh in zip(KEYS, MODEL_HASHES, LABEL_HASHES):
        row = next(r for r in records if r["key"] == key)
        folder = p2 / "candidates" / key
        mp, lp = folder / "model.npz", folder / "clusters.npy"
        require(row["status"] == "succeeded" and digest(mp) == row["model_sha256"] == mh
                and digest(lp) == row["clusters_sha256"] == lh, key + " archive mismatch")
        with np.load(mp, allow_pickle=False) as z:
            require(np.all(z["preprocessing_mean"] == 0) and np.all(z["preprocessing_scale"] == 1),
                    "nonidentity preprocessing")
            model = {n: z[n].copy() for n in ("weights", "means", "variances")}
        require(model["means"].shape == model["variances"].shape == (10, 512)
                and model["weights"].shape == (10,) and all(np.isfinite(a).all() for a in model.values())
                and np.all(model["weights"] > 0) and np.all(model["variances"] > 0)
                and np.isclose(model["weights"].sum(), 1), "invalid model")
        lab = np.load(lp, allow_pickle=False)
        require(lab.shape == (56000,) and np.issubdtype(lab.dtype, np.integer)
                and np.array_equal(np.unique(lab), np.arange(10)), "invalid labels")
        models.append(model); labels.append(lab); rows.append(row)
        paths.extend([mp, lp, folder / "fit.json"])
    with np.load(FEATURES, allow_pickle=False) as z:
        require(z.files == ["X"], "unexpected feature archive")
        X = z["X"]
    require(X.shape == (56000, 512) and X.dtype == np.float32 and np.isfinite(X).all(), "invalid features")
    # Use current evaluator only if it matches the frozen P2 source (LF/CRLF permitted).
    snapshot = p2.parent / "source_snapshot"
    for name in ("submission.py", "evaluate.py", "utils/runtime.py", "utils/data.py"):
        require(source_equal(PACKAGE / name, snapshot / name), "source drift: " + name)
        paths.extend([PACKAGE / name, snapshot / name])
    require(digest(snapshot / "submission.py") == plan["implementation_sha256"], "implementation provenance mismatch")
    scores = None
    if pca is not None:
        hashes = read_json(pca / "artifact_hashes.json")
        for name in ("embedding.npy", "fit_indices.npy", "pca_basis.npz", "summary.json"):
            require(digest(pca / name) == hashes[name], "PCA hash mismatch: " + name)
            paths.append(pca / name)
        require(digest(pca / "fit_indices.npy") == INDEX_HASH, "PCA subset mismatch")
        scores = np.load(pca / "embedding.npy", allow_pickle=False)
        require(scores.shape == (56000, 2) and np.isfinite(scores).all(), "invalid embedding")
    return X, models, labels, rows, indices, scores, paths


def load_implementation():
    spec = importlib.util.spec_from_file_location("frozen_gmm_kernels", PACKAGE / "submission.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def structure_plots(output, scores, labels, models):
    aligned, mapping, table = align(*labels)
    changed = aligned != labels[0]
    np.save(output / "changed_mask.npy", changed)
    np.save(output / "contingency.npy", table)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    order = np.random.default_rng(20261125).permutation(len(scores))
    for ax, lab, name in zip(axes[:2], (labels[0], aligned), ("GMM seed17", "GMM seed20261123 (aligned)")):
        ax.scatter(scores[order, 0], scores[order, 1], c=lab[order], cmap="tab10", vmin=0, vmax=9,
                   s=2, alpha=.4, linewidths=0, rasterized=True)
        ax.set_title(name)
    axes[2].scatter(*scores[~changed].T, c="lightgray", s=2, alpha=.25, linewidths=0)
    axes[2].scatter(*scores[changed].T, c="#d62728", s=3, alpha=.6, linewidths=0)
    axes[2].set_title(f"Changed assignments: {changed.sum()}/{len(changed)}")
    for ax in axes:
        ax.set(xlabel="PC1 (sample-fitted)", ylabel="PC2 (sample-fitted)")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(scores[:, 0].min() - .5, scores[:, 0].max() + .5)
        ax.set_ylim(scores[:, 1].min() - .5, scores[:, 1].max() + .5)
    fig.suptitle("Same archived PCA basis; captures 25.54% of full-data variance")
    fig.savefig(output / "gmm_anomaly_partition.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 6.5), constrained_layout=True)
    im = ax.imshow(np.ma.masked_equal(table, 0), norm=LogNorm(vmin=1, vmax=table.max()), cmap="Blues")
    for i, j in np.ndindex(table.shape):
        if table[i, j]:
            ax.text(j, i, str(table[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if table[i, j] > 1000 else "black")
    ax.set(xticks=range(10), yticks=range(10), xlabel="seed20261123 component (aligned ID)",
           ylabel="seed17 component (reference ID)", title="Full-data contingency; zero cells blank; no ground truth")
    fig.colorbar(im, ax=ax, label="Number of rows (log scale)")
    fig.savefig(output / "gmm_anomaly_contingency.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    reorder = np.argsort(mapping)
    for offset, model, ind, color, title in zip((-.18, .18), models, (np.arange(10), reorder),
                                                ("#1f77b4", "#d62728"), ("seed17", "seed20261123")):
        boxes = axes[0].boxplot(model["variances"][ind].T, positions=np.arange(10) + offset,
                                widths=.3, patch_artist=True, showfliers=False,
                                manage_ticks=False, whis=(0, 100))
        for box in boxes["boxes"]:
            box.set_facecolor(color); box.set_alpha(.55)
        axes[1].bar(np.arange(10) + offset, model["weights"][ind], width=.34, label=title, color=color, alpha=.7)
    axes[0].set(yscale="log", ylabel="Diagonal variance across 512 dimensions",
                title="Per-component distributions (whiskers min/max)")
    axes[0].axhline(1e-4, color="black", linestyle="--", label="reg_covar = 1e-4")
    axes[1].set(ylabel="Mixture weight", title="Original mixture weights; matched component IDs")
    for ax in axes:
        ax.set(xticks=range(10), xlabel="Component ID aligned to seed17")
        ax.legend()
    fig.savefig(output / "gmm_anomaly_parameters.png", dpi=180); plt.close(fig)
    return changed, mapping, table


def density_plot(output, normal, special, changed, scope):
    delta = special["ll"] - normal["ll"]
    groups = group_contributions(normal, special, changed)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    # Shared bins on a symmetric log scale retain tails without hiding the bulk.
    transformed = np.sign(delta) * np.log1p(np.abs(delta) / 10)
    transformed_bins = np.linspace(transformed.min() - .01, transformed.max() + .01, 70)
    bins = np.sign(transformed_bins) * 10 * np.expm1(np.abs(transformed_bins))
    for mask, name, color in ((~changed, "Unchanged", "#1f77b4"), (changed, "Changed", "#d62728")):
        if mask.any():
            axes[0].hist(delta[mask], bins=bins, histtype="step", linewidth=1.5,
                         label=f"{name} (n={mask.sum()})", color=color)
    axes[0].axvline(0, color="black", linewidth=.7)
    axes[0].set(xscale="symlog", xlabel="Delta log density (symlog): seed20261123 - seed17", ylabel="Sample count")
    axes[0].legend()
    for off, g, color in ((-.17, "unchanged", "#1f77b4"), (.17, "changed", "#d62728")):
        axes[1].bar(np.arange(4) + off, [groups[g]["terms"][t] for t in TERMS], width=.32, label=g, color=color)
    axes[1].set(xticks=range(4), xticklabels=TERMS,
                ylabel="Contribution to total mean delta log density")
    axes[1].axhline(0, color="black", linewidth=.7); axes[1].legend()
    fig.suptitle(f"{scope}; direct float64; responsibility-weighted exact decomposition")
    fig.savefig(output / "gmm_anomaly_density.png", dpi=180); plt.close(fig)
    return groups


def seal(output, paths, before, summary):
    require(all(digest(p) == before[str(p)] for p in paths), "an input changed during analysis")
    summary["inputs_unchanged"] = True
    summary["input_sha256"] = before
    save_json(output / "summary.json", summary)
    save_json(output / "artifact_hashes.json", {str(p.relative_to(output)): digest(p)
              for p in sorted(output.rglob("*")) if p.is_file() and p.name != "artifact_hashes.json"})


def execute(args):
    start = time.perf_counter()
    require(not args.output.exists(), "choose a new output directory")
    full = args.command == "run"
    if full:
        require(bool(os.environ.get("SLURM_JOB_ID")), "run must execute inside a Slurm allocation")
        require(torch.cuda.is_available() and "A100" in torch.cuda.get_device_name(0), "run requires Slurm A100")
    torch.set_num_threads(4)
    device = "cuda" if full else "cpu"
    X, models, labels, rows, indices, scores, paths = load_inputs(args.p2, None if full else args.pca)
    paths.extend([Path(__file__), PACKAGE / "run_gmm_anomaly_T0023.sbatch"])
    before = {str(p): digest(p) for p in paths}
    args.output.mkdir(parents=True, exist_ok=False)
    snap = args.output / "source_snapshot"; snap.mkdir()
    for name in ("gmm_anomaly.py", "evaluate.py", "submission.py", "run_gmm_anomaly_T0023.sbatch"):
        shutil.copy2(PACKAGE / name, snap / name)
    shutil.copytree(PACKAGE / "utils", snap / "utils", ignore=shutil.ignore_patterns("__pycache__"))
    implementation = load_implementation()
    aligned, mapping, table = align(*labels)
    changed = aligned != labels[0]
    params, sphere = variants(models[1])
    # Persist the complete intervention definition before computing effects.
    protocol = dict(task="T0023", refit=False, means_fixed=True, shared_spherical_variance=sphere,
                    sphere_definition="sum_k(pi_k * sum_d(v_kd)) / (sum_k(pi_k) * D), original parameters only",
                    interventions=list(VARIANTS), score_dtype="float64", score_formula="direct (x-mu)^2/v",
                    evaluation_dtype="float32", silhouette_rows=3000, silhouette_seed=20261011,
                    subset_distance_scope="within subset", CH_DB_scope="full original 512-dimensional features",
                    input_sha256=before)
    save_json(args.output / "protocol.json", protocol)
    summary = dict(task="T0023", mode=args.command, device=device, python=platform.python_version(),
                   torch=torch.__version__, numpy=np.__version__, cuda=torch.version.cuda,
                   gpu=torch.cuda.get_device_name(0) if full else None,
                   slurm={k: os.environ.get(k) for k in ("SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_JOB_NAME")},
                   mapping_special_to_normal=mapping.tolist(), changed_rows=int(changed.sum()),
                   archived_pair_ARI=float(adjusted_rand_score(*labels)),
                   shared_spherical_variance=sphere,
                   archived_runs=[{k: r[k] for k in ("key", "objective", "best_restart", "converged",
                                                   "silhouette", "calinski_harabasz", "davies_bouldin")} for r in rows])
    try:
        summary["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PACKAGE, text=True).strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        summary["git_commit"] = None
    use_indices = np.arange(len(X)) if full else indices
    data = X if full else X[indices]
    np.save(args.output / "density_indices.npy", use_indices)
    normal = score_points(data, tuple(models[0][n] for n in ("weights", "means", "variances")), device)
    special = score_points(data, params["original"], device)
    points = {"normal_" + k: v for k, v in normal.items()}
    points.update({"special_" + k: v for k, v in special.items()})
    np.savez_compressed(args.output / "point_diagnostics.npz", **points)
    summary["density_rows"] = len(data)
    summary["density_groups"] = density_plot(args.output, normal, special, changed[use_indices],
                                               "All 56000 rows" if full else "Fixed 3000-row subset ONLY")
    summary["same_rows_numerical_check"] = {}
    for name, model, stable, lab in zip(("normal", "special"), models, (normal, special), labels):
        check = legacy_check(data, model, stable, implementation, device)
        check["direct64_disagreements_with_archive"] = int(np.count_nonzero(stable["labels"] != lab[use_indices]))
        summary["same_rows_numerical_check"][name] = check
    if not full:
        structure_plots(args.output, scores, labels, models)
        summary["parameters"] = [{"key": key, "min_variance": float(m["variances"].min()),
                                   "median_variance": float(np.median(m["variances"])),
                                   "max_variance": float(m["variances"].max()),
                                   "fraction_variance_below_2reg": float((m["variances"] < 2e-4).mean()),
                                   "component_median_variances": np.median(m["variances"], axis=1).tolist(),
                                   "weights": m["weights"].tolist()}
                                  for key, m in zip(KEYS, models)]
        summary["full_interventions_executed"] = False
    else:
        from utils import runtime
        from evaluate import evaluate_partition
        runtime.configure("cuda", "float32", 4)
        implementation = load_implementation()  # refresh the module's device binding
        all_labels = dict(archived_normal=labels[0], archived_special=labels[1])
        density_means = {}
        for name in VARIANTS:
            out = special if name == "original" else score_points(X, params[name], device)
            all_labels[name] = out["labels"].astype(np.int32)
            density_means[name] = float(out["ll"].mean())
        np.savez_compressed(args.output / "intervention_labels.npz", **all_labels)
        results = {}
        for name, lab in all_labels.items():
            result = evaluate_partition(X, lab, indices, implementation)
            if name in density_means:
                result["direct64_mean_log_density"] = density_means[name]
            result["ARI_to_archived_normal"] = float(adjusted_rand_score(labels[0], lab))
            result["ARI_to_archived_special"] = float(adjusted_rand_score(labels[1], lab))
            result["changed_vs_special_aligned"] = int(np.count_nonzero(align(labels[1], lab)[0] != labels[1]))
            # Means retain component identities: raw changes isolate intervention
            # effects from any legacy32/direct64 baseline assignment differences.
            result["changed_vs_special_raw_ids"] = int(np.count_nonzero(lab != labels[1]))
            result["changed_vs_original_direct64_raw_ids"] = int(np.count_nonzero(lab != all_labels["original"]))
            result["ARI_to_original_direct64"] = float(adjusted_rand_score(all_labels["original"], lab))
            results[name] = result
        # Test the sufficient-condition identity independently of density scoring.
        nearest = []
        mu = torch.as_tensor(models[1]["means"], dtype=torch.float64, device=device)
        for i in range(0, len(X), 256):
            block = torch.as_tensor(X[i:i + 256], dtype=torch.float64, device=device)
            nearest.append((block[:, None] - mu).square().sum(2).argmin(1).cpu().numpy())
        np.testing.assert_array_equal(np.concatenate(nearest), all_labels["uniform_shared"])
        summary["interventions"] = results
        summary["archived_metric_replay_differences"] = {
            name: {metric: (results[name][metric] - row[metric]) if results[name][metric] is not None else None
                   for metric in ("silhouette", "calinski_harabasz", "davies_bouldin")}
            for name, row in zip(("archived_normal", "archived_special"), rows)}
        summary["all_metrics_ok"] = all(m["status"] == "ok" for r in results.values() for m in r["metrics"].values())
        summary["nearest_frozen_mean_identity_verified"] = True
        # Summary is sealed even when metrics failed: preserve evidence, return nonzero.
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
        for ax, metric in zip(axes, ("silhouette", "calinski_harabasz", "davies_bouldin")):
            values = [results[n][metric] for n in VARIANTS]
            ax.bar(range(4), [v if v is not None else np.nan for v in values])
            ax.set(xticks=range(4), xticklabels=VARIANTS, title=metric)
            ax.tick_params(axis="x", labelrotation=35)
        fig.suptitle("Frozen anomalous means; direct64 assignment, original CUDA float32 metrics")
        fig.savefig(args.output / "gmm_anomaly_interventions.png", dpi=180); plt.close(fig)
    summary["elapsed_seconds"] = time.perf_counter() - start
    seal(args.output, paths, before, summary)
    print(json.dumps({k: summary[k] for k in ("task", "mode", "changed_rows", "density_rows", "elapsed_seconds")}), flush=True)
    if full:
        require(summary["all_metrics_ok"], "one or more metrics failed; retain sealed summary")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("local", "run"))
    parser.add_argument("--p2", type=Path, default=P2)
    parser.add_argument("--pca", type=Path, default=PCA)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        execute(args)


if __name__ == "__main__":
    main()
