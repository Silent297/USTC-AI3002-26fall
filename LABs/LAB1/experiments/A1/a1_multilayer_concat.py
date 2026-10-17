# -*- coding: utf-8 -*-
"""A1: multi-layer concatenation probe (world lat/lon + truth T/F).

Question: does concatenating hidden states from several layers beat the best
single-layer linear probe?

Protocol (fixed before looking at valid):
- Features: each layer z-scored with TRAINING statistics only; blocks are
  z-scored independently and then concatenated (per guide2 §3 suggestion).
- world: closed-form ridge in dual form (exact solution of the same objective
  as our `model.py` LinearRegressionModel without L1). PRIMARY protocol uses a
  fixed lambda2 = 1.0 for every combo (same regularization budget as the
  historical std single-layer config; zero per-combo selection bias); as a
  SECONDARY robustness check, a per-combo lambda2 is selected by repeated
  (5x4) train-internal CV and its valid metric is recorded too. All 13
  singles, all 78 pairs and 6 fixed triples are enumerated.
- truth: logistic regression (sklearn lbfgs) with C = 1/(2*N*lambda2),
  lambda2 = 1e-3 (same regularization budget as the historical best GD run);
  z-scored features; same 13 singles / 78 pairs / 6 triples.
- The valid split is used ONLY for reporting, never for selection.
- Anchors for self-check / reference (same layers, historical pipelines):
  world z L11 lambda2=1.0 -> valid MSE ~167.0; world z L2 -> ~157.6;
  world raw L11 submission probe -> 150.43; truth GD L20 -> acc 0.8967.

Outputs:
- experiments/A1/a1_results.json  (all combos + protocol + env)
- experiments/A1/figures/a1_concat_heatmaps.{png,pdf}
- experiments/A1/figures/a1_best_combos.{png,pdf}
- experiments/A1/figures/a1_corr_matrices.{png,pdf}
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
from sklearn.model_selection import RepeatedKFold

ROOT = Path(__file__).resolve().parents[2]   # LABs/LAB1
HERE = Path(__file__).resolve().parent       # experiments/A1
FIG = HERE / "figures"

LAYERS = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 34, 35]
WORLD_FIXED_LAMBDA2 = 1.0
WORLD_LAM_GRID = [0.1, 0.3, 1.0, 3.0, 10.0]
TRUTH_LAMBDA2 = 1e-3
N_FOLDS = 5
N_REPEATS = 4
SEED = 0

# Fixed triples (no selection on valid): shallow, shallow+deep, mid, deep mixes.
TRIPLES = [
    (2, 5, 8), (2, 5, 11), (2, 11, 35),
    (17, 20, 23), (20, 23, 26), (32, 34, 35),
]

ANCHOR_WORLD_Z_L11 = 167.0     # GD, lr=3e-3, ep2000, lambda2=1.0
ANCHOR_WORLD_Z_L2 = 157.6      # GD full-budget, lambda2=1.0
ANCHOR_WORLD_RAW_L11 = 150.43  # historical best submission probe
ANCHOR_TRUTH_GD_L20 = 0.8967   # historical GD best (raw features, L1+L2)
GUIDE_ANCHORS = {"knn": 125.27, "mlp128_64": 152.70, "concat_L11_L35": 153.63}


def combo_label(layers):
    return "+".join(f"L{L}" for L in layers)


def load_task(task):
    data = {}
    for split in ("train", "valid"):
        layers = {}
        for layer in LAYERS:
            arr = np.load(ROOT / f"datasets/{task}/{split}.layer{layer}.npy")
            layers[layer] = arr.astype(np.float64)
        df = pd.read_csv(ROOT / f"datasets/{task}/{split}.csv")
        if task == "world":
            y = df[["latitude", "longitude"]].to_numpy(np.float64)
        else:
            y = df["label"].to_numpy(np.float64)
        data[split] = {"X": layers, "y": y}
    return data


def block_stats(block_list):
    mus, sds = [], []
    for X in block_list:
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd = np.where(sd == 0.0, 1.0, sd)
        mus.append(mu)
        sds.append(sd)
    return mus, sds


def build_concat(layers_dict, layers, indices, mus, sds):
    return np.hstack([(layers_dict[L][indices] - mu) / sd
                      for L, mu, sd in zip(layers, mus, sds)])


def ridge_dual_fit_predict(Xtr, ytr, Xva, lambda2):
    """Exact ridge in dual form, same objective as model.py's LinearRegressionModel:
    min (1/N)|y - Xw - b|^2 + lambda2*|w|^2  <=>  (K + N*lambda2 I) alpha = y - ybar, w = X^T alpha.
    """
    ybar = ytr.mean(axis=0)
    yc = ytr - ybar
    K = Xtr @ Xtr.T
    n = K.shape[0]
    alpha = np.linalg.solve(K + lambda2 * n * np.eye(n), yc)
    return (Xva @ (Xtr.T @ alpha)) + ybar, Xtr.T @ alpha, ybar


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=np.float64))
                              for v in (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2.0) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2.0) ** 2)
    return 6371.0 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def world_combo(data, layers, folds):
    Xtr_layers = {L: data["train"]["X"][L] for L in layers}
    y = data["train"]["y"]
    n = len(y)
    yva = data["valid"]["y"]
    nva = len(yva)

    def fit_at(lambda2):
        mus, sds = block_stats([Xtr_layers[L] for L in layers])
        Xtr = build_concat(Xtr_layers, layers, np.arange(n), mus, sds)
        Xva = build_concat(data["valid"]["X"], layers, np.arange(nva), mus, sds)
        pred_va, w, b = ridge_dual_fit_predict(Xtr, y, Xva, lambda2)
        return pred_va, Xtr @ w + b

    # primary: fixed regularization budget
    pred_va, pred_tr = fit_at(WORLD_FIXED_LAMBDA2)
    out = {"combo": combo_label(layers),
           "kind": "single" if len(layers) == 1 else ("pair" if len(layers) == 2 else "triple"),
           "layers": list(layers), "lambda2": WORLD_FIXED_LAMBDA2,
           "train_mse": float(((pred_tr - y) ** 2).mean()),
           "valid_mse": float(((pred_va - yva) ** 2).mean())}
    dist = haversine_km(pred_va[:, 0], pred_va[:, 1], yva[:, 0], yva[:, 1])
    out["dist_median_km"] = float(np.median(dist))
    out["dist_p90_km"] = float(np.percentile(dist, 90))

    # secondary: repeated-CV lambda2 selection (robustness only)
    cv_sse = {lam: 0.0 for lam in WORLD_LAM_GRID}
    cv_count = 0
    for tr_idx, va_idx in folds:
        mus, sds = block_stats([Xtr_layers[L][tr_idx] for L in layers])
        Xtr = build_concat(Xtr_layers, layers, tr_idx, mus, sds)
        Xva = build_concat(Xtr_layers, layers, va_idx, mus, sds)
        for lam in WORLD_LAM_GRID:
            pred, _, _ = ridge_dual_fit_predict(Xtr, y[tr_idx], Xva, lam)
            cv_sse[lam] += float(((pred - y[va_idx]) ** 2).sum())
        cv_count += len(va_idx) * 2
    cv_mse = {lam: cv_sse[lam] / cv_count for lam in WORLD_LAM_GRID}
    best_lam = min(cv_mse, key=cv_mse.get)
    pred_va_sel, pred_tr_sel = fit_at(best_lam)
    out["lambda2_cvsel"] = best_lam
    out["cv_mse"] = float(cv_mse[best_lam])
    out["valid_mse_cvsel"] = float(((pred_va_sel - yva) ** 2).mean())
    out["train_mse_cvsel"] = float(((pred_tr_sel - y) ** 2).mean())
    return out, pred_va


def truth_fit_predict(Xtr, ytr, Xva, C):
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(C=C, solver="lbfgs", max_iter=2000, tol=1e-4)
    clf.fit(Xtr, ytr)
    prob = clf.predict_proba(Xva)[:, 1]
    prob_tr = clf.predict_proba(Xtr)[:, 1]
    return prob, prob_tr, int(clf.n_iter_[0])


def logloss(y, prob):
    p = np.clip(prob, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def truth_combo(data, layers):
    y = data["train"]["y"]
    yva = data["valid"]["y"]
    mus, sds = block_stats([data["train"]["X"][L] for L in layers])
    Xtr = build_concat(data["train"]["X"], layers, np.arange(len(y)), mus, sds)
    Xva = build_concat(data["valid"]["X"], layers, np.arange(len(yva)), mus, sds)
    C = 1.0 / (2.0 * len(y) * TRUTH_LAMBDA2)
    prob, prob_tr, n_iter = truth_fit_predict(Xtr, y, Xva, C)
    return {"combo": combo_label(layers),
            "kind": "single" if len(layers) == 1 else ("pair" if len(layers) == 2 else "triple"),
            "layers": list(layers), "C": C,
            "valid_acc": float(np.mean((prob >= 0.5) == yva)),
            "valid_logloss": logloss(yva, prob),
            "train_acc": float(np.mean((prob_tr >= 0.5) == y)),
            "n_iter": n_iter}, prob


def anchor_check(data):
    """Fixed-lambda2=1.0 closed-form fits for L11 / L2, to cross-check the GD anchors."""
    y = data["train"]["y"]
    yva = data["valid"]["y"]
    out = {}
    for layer in (11, 2):
        mus, sds = block_stats([data["train"]["X"][layer]])
        Xtr = build_concat(data["train"]["X"], [layer], np.arange(len(y)), mus, sds)
        Xva = build_concat(data["valid"]["X"], [layer], np.arange(len(yva)), mus, sds)
        pred, _, _ = ridge_dual_fit_predict(Xtr, y, Xva, 1.0)
        out[layer] = float(((pred - yva) ** 2).mean())
    return out


def all_combos():
    singles = [(L,) for L in LAYERS]
    pairs = [(LAYERS[i], LAYERS[j]) for i in range(len(LAYERS)) for j in range(i + 1, len(LAYERS))]
    return singles, pairs


def run_world(data, singles_only):
    singles, pairs = all_combos()
    rkf = RepeatedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS, random_state=SEED)
    folds = list(rkf.split(np.arange(len(data["train"]["y"]))))
    results, preds = [], {}
    jobs = list(singles) + ([] if singles_only else list(pairs))
    if not singles_only:
        jobs += TRIPLES
    print(f"[world] {len(jobs)} combos x {len(folds)} CV fits x {len(WORLD_LAM_GRID)} lambdas "
          f"(fixed-budget primary lambda2={WORLD_FIXED_LAMBDA2})", flush=True)
    for i, layers in enumerate(jobs, 1):
        t0 = time.time()
        res, pred_va = world_combo(data, list(layers), folds)
        results.append(res)
        if len(layers) == 1:
            preds[layers[0]] = pred_va
        print(f"[world] {i}/{len(jobs)} {res['combo']:16s} kind={res['kind']:6s} "
              f"valid_fixed={res['valid_mse']:8.2f} med_km={res['dist_median_km']:7.1f} | "
              f"cvsel lam={res['lambda2_cvsel']:<4g} valid={res['valid_mse_cvsel']:8.2f} "
              f"({time.time() - t0:.1f}s)", flush=True)
    return results, preds


def run_truth(data, singles_only):
    singles, pairs = all_combos()
    jobs = list(singles) + ([] if singles_only else list(pairs))
    if not singles_only:
        jobs += TRIPLES
    results, probs = [], {}
    print(f"[truth] {len(jobs)} combos (fixed C)", flush=True)
    for i, layers in enumerate(jobs, 1):
        t0 = time.time()
        res, prob = truth_combo(data, list(layers))
        results.append(res)
        if len(layers) == 1:
            probs[layers[0]] = prob
        print(f"[truth] {i}/{len(jobs)} {res['combo']:16s} kind={res['kind']:6s} "
              f"acc={res['valid_acc']:.4f} ll={res['valid_logloss']:.4f} "
              f"n_iter={res['n_iter']:4d} ({time.time() - t0:.1f}s)", flush=True)
    return results, probs


def fig_concat_heatmaps(world_results, truth_results):
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.2))
    n = len(LAYERS)
    pos = {L: i for i, L in enumerate(LAYERS)}

    for ax, results, metric, title, fmt, cmap in [
        (axes[0], world_results, "valid_mse",
         "A1: pair-concat valid MSE (world, z-scored, fixed $\\lambda_2$=1.0)", "{:.1f}", "viridis"),
        (axes[1], truth_results, "valid_acc",
         "A1: pair-concat valid accuracy (truth)", "{:.3f}", "viridis"),
    ]:
        mat = np.full((n, n), np.nan)
        for r in results:
            if r["kind"] == "single":
                i = pos[r["layers"][0]]
                mat[i, i] = r[metric]
            elif r["kind"] == "pair":
                i, j = pos[r["layers"][0]], pos[r["layers"][1]]
                mat[j, i] = r[metric]
        im = ax.imshow(mat, cmap=cmap, origin="lower")
        for j in range(n):
            for i in range(n):
                if np.isfinite(mat[j, i]):
                    val = mat[j, i]
                    vmin, vmax = np.nanmin(mat), np.nanmax(mat)
                    frac = (val - vmin) / (vmax - vmin + 1e-12)
                    color = "white" if frac < 0.45 else "black"
                    ax.text(i, j, fmt.format(val), ha="center", va="center",
                            fontsize=5.6, color=color)
        ax.set_xticks(range(n), [f"L{L}" for L in LAYERS], fontsize=8)
        ax.set_yticks(range(n), [f"L{L}" for L in LAYERS], fontsize=8)
        ax.set_xlabel("second layer j", fontsize=9)
        ax.set_ylabel("first layer i", fontsize=9)
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    fig.suptitle("diagonal = single layer; lower triangle = pair concat", fontsize=9, y=0.015)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG / "a1_concat_heatmaps.png", dpi=150)
    fig.savefig(FIG / "a1_concat_heatmaps.pdf")
    plt.close(fig)


def _top_bars(ax, results, metric, best_single, anchor, anchor_label, xlabel, better="low", n_pairs=8, n_triples=3):
    pairs = sorted([r for r in results if r["kind"] == "pair"], key=lambda r: r[metric],
                   reverse=(better == "high"))[:n_pairs]
    triples = sorted([r for r in results if r["kind"] == "triple"], key=lambda r: r[metric],
                     reverse=(better == "high"))[:n_triples]
    rows = pairs + triples
    labels = [r["combo"] + (" (3)" if r["kind"] == "triple" else "") for r in rows]
    values = [r[metric] for r in rows]
    ypos = np.arange(len(rows))
    colors = ["tab:blue"] * len(pairs) + ["tab:orange"] * len(triples)
    ax.barh(ypos, values, color=colors, alpha=0.85)
    ax.set_yticks(ypos, labels, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(best_single, color="0.3", linestyle="--", linewidth=1.2,
               label=f"best single (this protocol) {best_single:.3g}")
    ax.axvline(anchor, color="tab:red", linestyle=":", linewidth=1.5,
               label=anchor_label)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.legend(fontsize=7.5, loc="lower right")


def fig_best_combos(world_results, truth_results):
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.2))
    best_w = min((r["valid_mse"] for r in world_results if r["kind"] == "single"))
    _top_bars(axes[0], world_results, "valid_mse", best_w, ANCHOR_WORLD_RAW_L11,
              f"submission anchor raw L11 {ANCHOR_WORLD_RAW_L11}",
              "valid MSE (deg$^2$; lower is better)")
    axes[0].set_title("A1: top pairs/triples vs best single (world)", fontsize=10)
    best_t = max((r["valid_acc"] for r in truth_results if r["kind"] == "single"))
    _top_bars(axes[1], truth_results, "valid_acc", best_t, ANCHOR_TRUTH_GD_L20,
              f"historical GD L20 {ANCHOR_TRUTH_GD_L20:.4f}",
              "valid accuracy (higher is better)", better="high")
    axes[1].set_title("A1: top pairs/triples vs best single (truth)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "a1_best_combos.png", dpi=150)
    fig.savefig(FIG / "a1_best_combos.pdf")
    plt.close(fig)


def fig_corr_matrices(world_data, world_preds, truth_data, truth_probs):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.6))
    n = len(LAYERS)

    yw = world_data["valid"]["y"]
    pw = np.stack([world_preds[L].reshape(-1) for L in LAYERS])   # (13, 260)
    rw = np.stack([(world_preds[L] - yw).reshape(-1) for L in LAYERS])
    corr_pred_w = np.corrcoef(pw)
    corr_res_w = np.corrcoef(rw)

    yt = truth_data["valid"]["y"]
    pt = np.stack([truth_probs[L] for L in LAYERS])               # (13, 300)
    rt = np.stack([truth_probs[L] - yt for L in LAYERS])
    corr_pred_t = np.corrcoef(pt)
    corr_res_t = np.corrcoef(rt)

    for ax, mat, title in [
        (axes[0], corr_res_w, "world: residual correlation (flattened lat/lon)"),
        (axes[1], corr_res_t, "truth: residual correlation"),
    ]:
        im = ax.imshow(mat, cmap="coolwarm", vmin=-1, vmax=1, origin="lower")
        ax.set_xticks(range(n), [f"L{L}" for L in LAYERS], fontsize=7.5)
        ax.set_yticks(range(n), [f"L{L}" for L in LAYERS], fontsize=7.5)
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    fig.suptitle("single-layer probes on valid: prediction corr mean "
                 f"(world {corr_pred_w[np.triu_indices(n, 1)].mean():.3f}, "
                 f"truth {corr_pred_t[np.triu_indices(n, 1)].mean():.3f})",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG / "a1_corr_matrices.png", dpi=150)
    fig.savefig(FIG / "a1_corr_matrices.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=("world", "truth"), default=("world", "truth"))
    parser.add_argument("--singles-only", action="store_true", help="smoke test: skip pairs/triples")
    args = parser.parse_args()

    FIG.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    payload = {
        "layers": LAYERS, "world_fixed_lambda2": WORLD_FIXED_LAMBDA2,
        "world_lambda_grid": WORLD_LAM_GRID,
        "truth_lambda2": TRUTH_LAMBDA2, "n_folds": N_FOLDS, "n_repeats": N_REPEATS,
        "seed": SEED,
        "triples": [list(t) for t in TRIPLES],
        "anchors": {"world_raw_l11": ANCHOR_WORLD_RAW_L11,
                    "world_z_l11_lam1": ANCHOR_WORLD_Z_L11,
                    "world_z_l2_lam1": ANCHOR_WORLD_Z_L2,
                    "truth_gd_l20": ANCHOR_TRUTH_GD_L20,
                    "guide": GUIDE_ANCHORS},
        "env": {"python": platform.python_version(), "numpy": np.__version__},
    }
    world_results = truth_results = None
    world_data = world_preds = truth_data = truth_probs = None

    if "world" in args.tasks:
        world_data = load_task("world")
        world_results, world_preds = run_world(world_data, args.singles_only)
        payload["world"] = world_results
        anchors = anchor_check(world_data)
        payload["world_anchor_check"] = anchors
        print(f"[world] anchor check (closed form, lambda2=1.0): "
              f"L11={anchors[11]:.2f} (GD anchor {ANCHOR_WORLD_Z_L11:.2f}), "
              f"L2={anchors[2]:.2f} (GD anchor {ANCHOR_WORLD_Z_L2:.2f})", flush=True)

    if "truth" in args.tasks:
        truth_data = load_task("truth")
        truth_results, truth_probs = run_truth(truth_data, args.singles_only)
        payload["truth"] = truth_results

    if world_results and truth_results:
        fig_concat_heatmaps(world_results, truth_results)
        fig_best_combos(world_results, truth_results)
        fig_corr_matrices(world_data, world_preds, truth_data, truth_probs)
        print("[figs] wrote 3 figures", flush=True)

    payload["elapsed_s"] = round(time.time() - t_start, 1)
    with (HERE / "a1_results.json").open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=1)
    print(f"[done] {time.time() - t_start:.1f}s; wrote {HERE / 'a1_results.json'}", flush=True)


if __name__ == "__main__":
    main()
