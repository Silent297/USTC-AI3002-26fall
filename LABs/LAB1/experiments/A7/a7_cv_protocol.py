# -*- coding: utf-8 -*-
"""A7: 5-fold CV protocol supplement (world / truth, all 13 layers).

Question: is our single train/valid-based layer selection reproducible under a
proper K-fold protocol, and how well does the CV layer ranking transfer to the
valid split?

Protocol (fixed before looking at the results):
- 5-fold CV on the TRAIN split only, 3 independent shuffles (seeds 0,1,2).
  truth: StratifiedKFold on the label; world: StratifiedKFold on latitude
  quintiles (world has no class labels; latitude strata keep every fold
  geographically covering the full range).
- Fold-wise z-scoring: mean/std from the training folds only (no leakage).
- Estimators identical to A1 single-layer protocol:
  world: closed-form dual ridge, lambda2 = 1.0;
  truth: sklearn lbfgs logistic, C = 1/(2*n_train_fold*lambda2), lambda2 = 1e-3.
- Metrics: world element-mean MSE + median haversine; truth accuracy + logloss.
- Reference columns are A1 singles (same protocol) train/valid metrics from
  experiments/A1/a1_results.json; the valid split is never used for selection.
- CV is a random split of train (same distribution); valid is OOD (Q8: truth
  valid has 3.5x misspellings + 77.4% unseen subjects; world valid has 20
  unseen countries). Absolute CV-vs-valid gaps mix sampling and distribution
  shift, so the layer-RANKING transfer is what this experiment tests.

Outputs:
- experiments/A7/a7_results.json
- experiments/A7/a7_summary.csv
- experiments/A7/figures/a7_cv_curves.{png,pdf}
- experiments/A7/figures/a7_cv_vs_valid.{png,pdf}
- experiments/A7/figures/a7_fold_stability.{png,pdf}
- experiments/A7/figures/a7_world_error_tails.{png,pdf}
"""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]   # LABs/LAB1
HERE = Path(__file__).resolve().parent       # experiments/A7
FIG = HERE / "figures"
A1_RESULTS = ROOT / "experiments/A1/a1_results.json"

LAYERS = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 34, 35]
N_FOLDS = 5
N_REPEATS = 3
SEED = 0
WORLD_LAMBDA2 = 1.0
TRUTH_LAMBDA2 = 1e-3

ANCHOR_WORLD_Z_L2 = 157.6186   # A1 single L2, valid MSE
ANCHOR_TRUTH_Z_L20 = 0.8667    # A1 single L20, valid accuracy


def ridge_dual_fit_predict(Xtr, ytr, Xva, lambda2):
    """Closed-form ridge in dual form (same objective as A1 / model.py)."""
    ybar = ytr.mean(axis=0)
    yc = ytr - ybar
    K = Xtr @ Xtr.T
    n = K.shape[0]
    alpha = np.linalg.solve(K + lambda2 * n * np.eye(n), yc)
    return (Xva @ (Xtr.T @ alpha)) + ybar


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=np.float64))
                              for v in (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2.0) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2.0) ** 2)
    return 6371.0 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def logloss(y, prob):
    p = np.clip(prob, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def zscore_fold(Xtr, Xva):
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd = np.where(sd == 0.0, 1.0, sd)
    return (Xtr - mu) / sd, (Xva - mu) / sd


def world_cv(X, y):
    """5-fold CV with latitude-quintile stratification; returns fold rows + OOF rows."""
    strata = pd.qcut(pd.Series(y[:, 0]), N_FOLDS, labels=False).to_numpy()
    fold_rows, oof_rows, oof_preds = [], [], []
    for repeat in range(N_REPEATS):
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED + repeat)
        oof = np.full_like(y, np.nan)
        for fold, (tr, va) in enumerate(skf.split(np.arange(len(y)), strata)):
            Xt, Xv = zscore_fold(X[tr], X[va])
            pred = ridge_dual_fit_predict(Xt, y[tr], Xv, WORLD_LAMBDA2)
            oof[va] = pred
            dist = haversine_km(pred[:, 0], pred[:, 1], y[va, 0], y[va, 1])
            fold_rows.append({"repeat": repeat, "fold": fold, "n": int(len(va)),
                              "mse": float(((pred - y[va]) ** 2).mean()),
                              "median_km": float(np.median(dist))})
        dist = haversine_km(oof[:, 0], oof[:, 1], y[:, 0], y[:, 1])
        oof_rows.append({"repeat": repeat,
                         "mse": float(((oof - y) ** 2).mean()),
                         "median_km": float(np.median(dist))})
        oof_preds.append(oof)
    return fold_rows, oof_rows, oof_preds


def truth_cv(X, y):
    """5-fold CV stratified on label; returns fold rows + OOF rows."""
    fold_rows, oof_rows, oof_preds = [], [], []
    for repeat in range(N_REPEATS):
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED + repeat)
        oof = np.full(len(y), np.nan)
        max_iter_seen, n_converged = 0, 0
        for fold, (tr, va) in enumerate(skf.split(np.arange(len(y)), y)):
            Xt, Xv = zscore_fold(X[tr], X[va])
            C = 1.0 / (2.0 * len(tr) * TRUTH_LAMBDA2)
            clf = LogisticRegression(C=C, solver="lbfgs", max_iter=2000, tol=1e-4)
            clf.fit(Xt, y[tr])
            prob = clf.predict_proba(Xv)[:, 1]
            oof[va] = prob
            n_iter = int(clf.n_iter_[0])
            max_iter_seen = max(max_iter_seen, n_iter)
            n_converged += int(n_iter < 2000)
            fold_rows.append({"repeat": repeat, "fold": fold, "n": int(len(va)),
                              "acc": float(np.mean((prob >= 0.5) == y[va])),
                              "logloss": logloss(y[va], prob),
                              "n_iter": n_iter})
        oof_rows.append({"repeat": repeat,
                         "acc": float(np.mean((oof >= 0.5) == y)),
                         "logloss": logloss(y, oof),
                         "max_n_iter": max_iter_seen,
                         "n_converged": n_converged})
        oof_preds.append(oof)
    return fold_rows, oof_rows, oof_preds


def summarize_cv(fold_rows, oof_rows, keys):
    out = {}
    for key in keys:
        fold_vals = np.array([r[key] for r in fold_rows], dtype=np.float64)
        oof_vals = np.array([r[key] for r in oof_rows], dtype=np.float64)
        out[key] = {"mean": float(oof_vals.mean()),
                    "std_repeat": float(oof_vals.std(ddof=1)) if len(oof_vals) > 1 else 0.0,
                    "fold_mean": float(fold_vals.mean()),
                    "std_fold": float(fold_vals.std(ddof=1)),
                    "oof_repeats": [float(v) for v in oof_vals],
                    "folds": [float(v) for v in fold_vals]}
    return out


def load_a1_singles(task):
    with A1_RESULTS.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    return {int(r["layers"][0]): r for r in payload[task] if r["kind"] == "single"}


def rank_values(values, lower_better):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values if lower_better else -values, kind="stable")
    ranks = np.empty(len(values), dtype=int)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


def selection_block(records, metric, valid_key, lower_better, valid_lower_better):
    cv_vals = np.array([r["cv"][metric]["mean"] for r in records])
    valid_vals = np.array([r["valid"][valid_key] for r in records])
    cv_best = int(np.argmin(cv_vals) if lower_better else np.argmax(cv_vals))
    valid_best = int(np.argmin(valid_vals) if valid_lower_better else np.argmax(valid_vals))
    return {
        "metric": metric,
        "cv_best_layer": int(records[cv_best]["layer"]),
        "cv_best_cv": float(cv_vals[cv_best]),
        "cv_best_valid": float(valid_vals[cv_best]),
        "valid_best_layer": int(records[valid_best]["layer"]),
        "valid_best_valid": float(valid_vals[valid_best]),
        "valid_best_cv": float(cv_vals[valid_best]),
        "valid_regret_of_cv_selection": float(valid_vals[cv_best] - valid_vals[valid_best]),
        "cv_regret_of_valid_selection": float(cv_vals[valid_best] - cv_vals[cv_best]),
        "same_layer": bool(cv_best == valid_best),
    }


def run_task(task, data):
    if task == "world":
        fold_rows_all, oof_rows_all, keys = [], [], ("mse", "median_km")
        cv_fn = world_cv
    else:
        fold_rows_all, oof_rows_all, keys = [], [], ("acc", "logloss")
        cv_fn = truth_cv

    refs = load_a1_singles(task)
    records = []
    for i, layer in enumerate(LAYERS, 1):
        t0 = time.time()
        X = np.load(ROOT / f"datasets/{task}/train.layer{layer}.npy").astype(np.float64)
        fold_rows, oof_rows, oof_preds = cv_fn(X, data["y"])
        del X
        cv = summarize_cv(fold_rows, oof_rows, keys)
        ref = refs[layer]
        rec = {"layer": layer,
               "cv": cv,
               "train": {k: float(ref[k]) for k in
                         (("train_mse",) if task == "world" else ("train_acc",))},
               "valid": ({k: float(ref[k]) for k in ("valid_mse", "dist_median_km")}
                         if task == "world" else
                         {k: float(ref[k]) for k in ("valid_acc", "valid_logloss")})}
        if task == "truth":
            rec["n_iter_max"] = int(max(r["n_iter"] for r in fold_rows))
            rec["n_converged"] = int(sum(1 for r in fold_rows if r["n_iter"] < 2000))
        records.append(rec)
        headline = cv["mse"]["mean"] if task == "world" else cv["acc"]["mean"]
        print(f"[{task}] {i}/{len(LAYERS)} L{layer:<2d} cv={headline:.4f} "
              f"(fold sd {cv[keys[0]]['std_fold']:.4f}) "
              f"valid={rec['valid']['valid_mse' if task == 'world' else 'valid_acc']:.4f} "
              f"({time.time() - t0:.1f}s)", flush=True)
        fold_rows_all.extend([dict(r, layer=layer) for r in fold_rows])
        oof_rows_all.extend([dict(r, layer=layer) for r in oof_rows])
        data.setdefault("oof_by_layer", {})[layer] = oof_preds[0]

    # rank matrices for the stability figure: rank of each layer in each fold
    fold_keys = ("mse",) if task == "world" else ("acc",)
    rank_mat = np.full((len(LAYERS), N_REPEATS * N_FOLDS), np.nan)
    for j in range(N_REPEATS * N_FOLDS):
        vals = [next(r for r in fold_rows_all if r["layer"] == L
                     and r["repeat"] == j // N_FOLDS and r["fold"] == j % N_FOLDS)[fold_keys[0]]
                for L in LAYERS]
        rank_mat[:, j] = rank_values(vals, lower_better=(task == "world"))

    winners = []
    for j in range(N_REPEATS * N_FOLDS):
        winners.append(int(LAYERS[int(np.argmin(rank_mat[:, j]))]))
    winner_counts = {int(L): int(winners.count(L)) for L in sorted(set(winners))}

    lower = task == "world"
    corr = {
        "cv_vs_valid": spearmanr([r["cv"]["mse" if task == "world" else "acc"]["mean"] for r in records],
                                 [r["valid"]["valid_mse" if task == "world" else "valid_acc"] for r in records]),
        "train_vs_valid": spearmanr([list(r["train"].values())[0] for r in records],
                                    [r["valid"]["valid_mse" if task == "world" else "valid_acc"] for r in records]),
    }
    if task == "world":
        corr["cv_median_km_vs_valid_median_km"] = spearmanr(
            [r["cv"]["median_km"]["mean"] for r in records],
            [r["valid"]["dist_median_km"] for r in records])
        corr["cv_mse_vs_cv_median_km"] = spearmanr(
            [r["cv"]["mse"]["mean"] for r in records],
            [r["cv"]["median_km"]["mean"] for r in records])
    if task == "truth":
        corr["cv_logloss_vs_valid_acc"] = spearmanr(
            [r["cv"]["logloss"]["mean"] for r in records],
            [r["valid"]["valid_acc"] for r in records])
        corr["cv_acc_vs_valid_logloss"] = spearmanr(
            [r["cv"]["acc"]["mean"] for r in records],
            [r["valid"]["valid_logloss"] for r in records])

    result = {
        "layers": LAYERS,
        "records": records,
        "rank_matrix": rank_mat.tolist(),
        "fold_winners": winners,
        "winner_counts": winner_counts,
        "correlations": {k: {"rho": float(v.statistic), "p": float(v.pvalue)}
                         for k, v in corr.items()},
        "selection": selection_block(records, "mse" if lower else "acc",
                                     "valid_mse" if lower else "valid_acc", lower, lower),
    }
    if task == "truth":
        result["selection_logloss"] = selection_block(records, "logloss", "valid_acc", True, False)
    return result


def fig_cv_curves(world, truth):
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.4))
    x = np.arange(len(LAYERS))
    labels = [f"L{L}" for L in LAYERS]
    panels = [
        (axes[0], world, "mse", "train_mse", "valid_mse", True,
         "world: MSE (deg$^2$, lower is better)", True, "cv_vs_valid"),
        (axes[1], truth, "acc", "train_acc", "valid_acc", False,
         "truth: accuracy (higher is better)", False, "cv_vs_valid"),
    ]
    for ax, res, metric, train_key, valid_key, logy, title, lower, corr_key in panels:
        train = [r["train"][train_key] for r in res["records"]]
        cv = [r["cv"][metric]["mean"] for r in res["records"]]
        cv_sd = [r["cv"][metric]["std_fold"] for r in res["records"]]
        valid = [r["valid"][valid_key] for r in res["records"]]
        ax.plot(x, train, ":", color="0.55", marker=".", markersize=4,
                linewidth=1.2, label="train, in-sample (A1)")
        ax.errorbar(x, cv, yerr=cv_sd, color="tab:blue", marker="o", markersize=4,
                    capsize=2.5, linewidth=1.4, label="5-fold CV, OOF (3 repeats)")
        ax.plot(x, valid, "--", color="tab:orange", marker="s", markersize=4,
                linewidth=1.4, label="valid, OOD (A1 protocol)")
        if logy:
            ax.set_yscale("log")
        cv_best = int(np.argmin(cv) if lower else np.argmax(cv))
        valid_best = int(np.argmin(valid) if lower else np.argmax(valid))
        ax.axvline(cv_best, color="tab:blue", alpha=0.25, linewidth=6)
        ax.axvline(valid_best, color="tab:orange", alpha=0.18, linewidth=6)
        rho = res["correlations"][corr_key]
        ax.set_title(f"{title}\nSpearman(CV, valid) = {rho['rho']:.2f} "
                     f"(p = {rho['p']:.3f}); CV best L{LAYERS[cv_best]}, "
                     f"valid best L{LAYERS[valid_best]}", fontsize=9.5)
        ax.set_xticks(x, labels, fontsize=8)
        ax.set_xlabel("layer", fontsize=9)
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.legend(fontsize=7.5, loc="best")
    fig.suptitle("A7: 5-fold CV on train vs the single train/valid split "
                 "(same closed-form protocol; CV z-scored per fold)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG / "a7_cv_curves.png", dpi=150)
    fig.savefig(FIG / "a7_cv_curves.pdf")
    plt.close(fig)


def fig_cv_vs_valid(world, truth):
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.9))
    specs = [
        (axes[0], world, "mse", "valid_mse", "cv_vs_valid",
         "world: CV MSE vs valid MSE", "CV OOF MSE", "valid MSE"),
        (axes[1], truth, "acc", "valid_acc", "cv_vs_valid",
         "truth: CV accuracy vs valid accuracy", "CV OOF accuracy", "valid accuracy"),
        (axes[2], truth, "logloss", "valid_acc", "cv_logloss_vs_valid_acc",
         "truth: CV logloss vs valid accuracy\n(direction-inverted: lower logloss = higher accuracy)",
         "CV OOF logloss", "valid accuracy"),
    ]
    for ax, res, cv_key, valid_key, corr_key, title, xlabel, ylabel in specs:
        xs = np.array([r["cv"][cv_key]["mean"] for r in res["records"]])
        ys = np.array([r["valid"][valid_key] for r in res["records"]])
        ax.scatter(xs, ys, s=26, color="tab:blue", alpha=0.8)
        for r, xv, yv in zip(res["records"], xs, ys):
            ax.annotate(f"L{r['layer']}", (xv, yv), xytext=(3, 2),
                        textcoords="offset points", fontsize=6.5, color="0.25")
        rho = res["correlations"][corr_key]
        ax.set_title(f"{title}\nSpearman $\\rho$ = {rho['rho']:.2f} (p = {rho['p']:.3f})",
                     fontsize=9.5)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(alpha=0.25, linewidth=0.6)
    fig.suptitle("A7: layer rankings under CV vs the OOD valid split across 13 layers",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIG / "a7_cv_vs_valid.png", dpi=150)
    fig.savefig(FIG / "a7_cv_vs_valid.pdf")
    plt.close(fig)


def fig_fold_stability(world, truth):
    fig, axes = plt.subplots(2, 1, figsize=(12.6, 8.2))
    labels = [f"L{L}" for L in LAYERS]
    for ax, res, title in [
        (axes[0], world, "world: per-fold layer rank by fold MSE (1 = best)"),
        (axes[1], truth, "truth: per-fold layer rank by fold accuracy (1 = best)"),
    ]:
        mat = np.array(res["rank_matrix"])
        im = ax.imshow(mat, aspect="auto", cmap="viridis_r", vmin=1, vmax=len(LAYERS))
        for j, winner in enumerate(res["fold_winners"]):
            i = LAYERS.index(winner)
            ax.scatter([j], [i], marker="o", s=16, facecolor="none",
                       edgecolor="black", linewidth=0.8)
        for boundary in range(N_FOLDS, N_REPEATS * N_FOLDS, N_FOLDS):
            ax.axvline(boundary - 0.5, color="w", linewidth=1.2)
        ax.set_yticks(range(len(LAYERS)), labels, fontsize=8)
        ax.set_xticks(range(N_REPEATS * N_FOLDS),
                      [f"R{j // N_FOLDS}F{j % N_FOLDS}" for j in range(N_REPEATS * N_FOLDS)],
                      fontsize=6.5, rotation=0)
        ax.set_title(title, fontsize=9.5)
        fig.colorbar(im, ax=ax, shrink=0.85, pad=0.015, label="rank")
    fig.suptitle("A7: fold-level layer-ranking stability (○ = fold winner; "
                 "white lines separate the 3 repeats)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG / "a7_fold_stability.png", dpi=150)
    fig.savefig(FIG / "a7_fold_stability.pdf")
    plt.close(fig)


def ecdf(values):
    v = np.sort(np.asarray(values, dtype=np.float64))
    return v, np.arange(1, len(v) + 1) / len(v)


def fig_world_error_tails(world_res, world_data):
    """Why does CV mean MSE far exceed valid MSE while medians agree?"""
    df_va = pd.read_csv(ROOT / "datasets/world/valid.csv")
    y = world_data["y"]
    yv = df_va[["latitude", "longitude"]].to_numpy(np.float64)
    sel = world_res["selection"]
    layers = sorted({int(sel["cv_best_layer"]), int(sel["valid_best_layer"])})
    colors = {int(sel["cv_best_layer"]): "tab:blue", int(sel["valid_best_layer"]): "tab:orange"}

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.0))
    stats = {}
    for layer in layers:
        X = np.load(ROOT / f"datasets/world/train.layer{layer}.npy").astype(np.float64)
        Xv = np.load(ROOT / f"datasets/world/valid.layer{layer}.npy").astype(np.float64)
        mu, sd = X.mean(axis=0), X.std(axis=0)
        sd = np.where(sd == 0.0, 1.0, sd)
        pred_va = ridge_dual_fit_predict((X - mu) / sd, y, (Xv - mu) / sd, WORLD_LAMBDA2)
        oof = world_data["oof_by_layer"][layer]
        se_cv = ((oof - y) ** 2).mean(axis=1)
        se_va = ((pred_va - yv) ** 2).mean(axis=1)
        km_cv = haversine_km(oof[:, 0], oof[:, 1], y[:, 0], y[:, 1])
        km_va = haversine_km(pred_va[:, 0], pred_va[:, 1], yv[:, 0], yv[:, 1])
        stats[layer] = {"se_cv": se_cv, "se_va": se_va, "km_cv": km_cv, "km_va": km_va,
                        "mse_cv": float(se_cv.mean()), "mse_va": float(se_va.mean()),
                        "med_se_cv": float(np.median(se_cv)), "med_se_va": float(np.median(se_va)),
                        "med_km_cv": float(np.median(km_cv)), "med_km_va": float(np.median(km_va))}
        top_cv = np.sort(se_cv)[::-1][:5]
        top_va = np.sort(se_va)[::-1][:5]
        trim_va = float(np.sort(se_va)[::-1][5:].mean())
        print(f"[world tails] L{layer}: CV MSE {stats[layer]['mse_cv']:.1f} "
              f"(med SE {stats[layer]['med_se_cv']:.1f}) vs valid MSE "
              f"{stats[layer]['mse_va']:.1f} (med SE {stats[layer]['med_se_va']:.1f}); "
              f"median km {stats[layer]['med_km_cv']:.0f} vs {stats[layer]['med_km_va']:.0f}; "
              f"top5 SSE share {top_cv.sum() / se_cv.sum() * 100:.1f}% (n={len(se_cv)}) vs "
              f"{top_va.sum() / se_va.sum() * 100:.1f}% (n={len(se_va)}); "
              f"valid MSE excl top5 {trim_va:.1f}; max SE {se_cv.max():.0f} vs {se_va.max():.0f}",
              flush=True)

    for layer in layers:
        s, c = stats[layer], colors[layer]
        axes[0].plot(*ecdf(s["se_cv"]), color=c, linestyle="-", linewidth=1.5,
                     label=f"L{layer} CV OOF (R0): MSE {s['mse_cv']:.0f}, median SE {s['med_se_cv']:.0f}")
        axes[0].plot(*ecdf(s["se_va"]), color=c, linestyle="--", linewidth=1.5,
                     label=f"L{layer} valid: MSE {s['mse_va']:.0f}, median SE {s['med_se_va']:.0f}")
        axes[1].plot(*ecdf(s["km_cv"]), color=c, linestyle="-", linewidth=1.5,
                     label=f"L{layer} CV OOF (R0): median {s['med_km_cv']:.0f} km")
        axes[1].plot(*ecdf(s["km_va"]), color=c, linestyle="--", linewidth=1.5,
                     label=f"L{layer} valid: median {s['med_km_va']:.0f} km")
    for ax, xlabel in [(axes[0], "per-sample squared error (deg$^2$, log scale)"),
                       (axes[1], "great-circle distance (km, log scale)")]:
        ax.set_xscale("log")
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("ECDF", fontsize=9)
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.legend(fontsize=7, loc="upper left")
    axes[0].set_title("squared error: heavy tails drive the mean", fontsize=9.5)
    axes[1].set_title("great-circle distance: typical cases agree", fontsize=9.5)
    fig.suptitle("A7 world: CV vs valid per-sample errors (solid = CV OOF, dashed = valid; "
                 f"L{sel['cv_best_layer']} = CV best, L{sel['valid_best_layer']} = valid best)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG / "a7_world_error_tails.png", dpi=150)
    fig.savefig(FIG / "a7_world_error_tails.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=("world", "truth"),
                        default=("world", "truth"))
    args = parser.parse_args()

    FIG.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    payload = {
        "layers": LAYERS, "n_folds": N_FOLDS, "n_repeats": N_REPEATS,
        "seeds": [SEED + r for r in range(N_REPEATS)],
        "world_lambda2": WORLD_LAMBDA2, "truth_lambda2": TRUTH_LAMBDA2,
        "world_stratify": "latitude quintiles", "truth_stratify": "label",
        "protocol_note": "CV on train only; fold-wise z-score; same estimators as A1 singles; "
                         "valid reference from experiments/A1/a1_results.json",
        "env": {"python": platform.python_version(), "numpy": np.__version__},
    }
    results = {}
    world_data = None

    for task in args.tasks:
        df = pd.read_csv(ROOT / f"datasets/{task}/train.csv")
        y = (df[["latitude", "longitude"]].to_numpy(np.float64) if task == "world"
             else df["label"].to_numpy(np.float64))
        data = {"y": y}
        results[task] = run_task(task, data)
        if task == "world":
            world_data = data
        payload[task] = results[task]

    if "world" in results:
        assert abs(results["world"]["records"][0]["valid"]["valid_mse"] - ANCHOR_WORLD_Z_L2) < 1e-3, \
            "world L2 valid anchor mismatch (A1 reference changed?)"
        sel = results["world"]["selection"]
        print(f"[world] CV best L{sel['cv_best_layer']} (valid there {sel['cv_best_valid']:.2f}); "
              f"valid best L{sel['valid_best_layer']} ({sel['valid_best_valid']:.2f})", flush=True)
    if "truth" in results:
        assert abs(results["truth"]["records"][6]["valid"]["valid_acc"] - ANCHOR_TRUTH_Z_L20) < 1e-4, \
            "truth L20 valid anchor mismatch (A1 reference changed?)"
        sel = results["truth"]["selection"]
        print(f"[truth] CV best L{sel['cv_best_layer']} (valid there {sel['cv_best_valid']:.4f}); "
              f"valid best L{sel['valid_best_layer']} ({sel['valid_best_valid']:.4f})", flush=True)

    # summary CSV
    rows = []
    for task in results:
        res = results[task]
        metric = "mse" if task == "world" else "acc"
        valid_key = "valid_mse" if task == "world" else "valid_acc"
        cv_ranks = rank_values([r["cv"][metric]["mean"] for r in res["records"]],
                               lower_better=(task == "world"))
        valid_ranks = rank_values([r["valid"][valid_key] for r in res["records"]],
                                  lower_better=(task == "world"))
        for r, cr, vr in zip(res["records"], cv_ranks, valid_ranks):
            rows.append({"task": task, "layer": r["layer"],
                         "train": list(r["train"].values())[0],
                         "cv_mean": r["cv"][metric]["mean"],
                         "cv_std_fold": r["cv"][metric]["std_fold"],
                         "valid": r["valid"][valid_key],
                         "cv_rank": int(cr), "valid_rank": int(vr)})
    pd.DataFrame(rows).to_csv(HERE / "a7_summary.csv", index=False)

    if "world" in results and "truth" in results:
        fig_cv_curves(results["world"], results["truth"])
        fig_cv_vs_valid(results["world"], results["truth"])
        fig_fold_stability(results["world"], results["truth"])
        fig_world_error_tails(results["world"], world_data)
        print("[figs] wrote 4 figures", flush=True)

    payload["elapsed_s"] = round(time.time() - t_start, 1)
    with (HERE / "a7_results.json").open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=1)
    print(f"[done] {payload['elapsed_s']}s; wrote {HERE / 'a7_results.json'}", flush=True)


if __name__ == "__main__":
    main()
