# -*- coding: utf-8 -*-
"""A6: how dense is the L11/L20/L23 representation? Nonzero coefficients vs lambda1.

Protocol
- Features z-scored with train statistics only.
- Objective matches model.py, but solved with FISTA + soft-thresholding
  (proximal L1), NOT the sign-subgradient of model.py.
    linear  : (1/N)||Xw+b-y||^2 + lam1*||w||_1 + lam2*||w||^2
    logistic: (1/N) sum log(1+exp(z)) - y*z + lam1*||w||_1 + lam2*||w||^2
- lambda2 fixed at the A1 anchors: world 1.0, truth 1e-3.
- lambda1 grid spans [1e-5, ~1.15] * lambda1_max, where lambda1_max is the
  analytic zero-solution threshold: max |gradient of the smooth loss at w=0|.
- Path is fitted from large to small lambda1 with warm starts.
- Valid is used only for reporting; no selection.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(parents=True, exist_ok=True)
R_EARTH = 6371.0

TASKS = {
    "world_L11": {"task": "world", "layer": 11, "lam2": 1.0, "kind": "linear"},
    "truth_L20": {"task": "truth", "layer": 20, "lam2": 1e-3, "kind": "logistic"},
    "truth_L23": {"task": "truth", "layer": 23, "lam2": 1e-3, "kind": "logistic"},
}
GRIDS = {
    "world_L11": [1e-4, 1e-3, 1e-2, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0,
                  50.0, 70.0, 91.40907791149256, 105.0],
    "truth_L20": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.05,
                  0.1, 0.15, 0.2, 0.2259724848572042, 0.26],
    "truth_L23": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.05,
                  0.1, 0.15, 0.18796523553038672, 0.22, 0.26],
}
ANCHORS = {
    "world_L11": {"valid_mse": 167.17072274253437, "median_km": 923.1205912354363},
    "truth_L20": {"valid_acc": 0.8666666666666667, "valid_logloss": 0.4834606151866668},
    "truth_L23": {"valid_acc": 0.85, "valid_logloss": 0.486512503446484},
}


def load_task(name):
    cfg = TASKS[name]
    out = {}
    for split in ("train", "valid"):
        X = np.load(ROOT / f"datasets/{cfg['task']}/{split}.layer{cfg['layer']}.npy").astype(np.float64)
        df = pd.read_csv(ROOT / f"datasets/{cfg['task']}/{split}.csv")
        if cfg["kind"] == "linear":
            y = df[["latitude", "longitude"]].to_numpy(np.float64)
        else:
            y = df["label"].to_numpy(np.float64)
        out[split] = {"X": X, "y": y}
    mu = out["train"]["X"].mean(axis=0)
    sd = out["train"]["X"].std(axis=0)
    sd = np.where(sd == 0.0, 1.0, sd)
    for split in out:
        out[split]["X"] = (out[split]["X"] - mu) / sd
    return out


def soft(x, t):
    return np.sign(x) * np.maximum(np.abs(x) - t, 0.0)


def power_lmax(X, iters=60, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=X.shape[1])
    v /= np.linalg.norm(v)
    n = 1.0
    for _ in range(iters):
        v = X.T @ (X @ v)
        n = np.linalg.norm(v)
        v /= n
    return float(n)


def smooth_grad(X, y, w, b, lam2, kind):
    z = X @ w + b
    if kind == "linear":
        r = z - y
        f = np.sum(r * r) / len(y) + lam2 * np.sum(w * w)
        g = 2.0 / len(y) * (X.T @ r) + 2.0 * lam2 * w
        gb = 2.0 / len(y) * r.sum(axis=0)
    else:
        f = np.mean(np.logaddexp(0.0, z) - y * z) + lam2 * np.sum(w * w)
        p = 0.5 * (1.0 + np.tanh(0.5 * z))
        g = X.T @ (p - y) / len(y) + 2.0 * lam2 * w
        gb = np.mean(p - y)
    return f, g, gb


def smooth_loss(X, y, w, b, lam2, kind):
    z = X @ w + b
    if kind == "linear":
        r = z - y
        return float(np.sum(r * r) / len(y) + lam2 * np.sum(w * w))
    return float(np.mean(np.logaddexp(0.0, z) - y * z) + lam2 * np.sum(w * w))


def fista(X, y, lam1, lam2, kind, w0, b0, step, max_iter=3000, tol=1e-7):
    w = np.array(w0, dtype=np.float64, copy=True)
    b = np.array(b0, dtype=np.float64, copy=True)
    z = w.copy()
    zb = b
    tk = 1.0
    f_hist = []
    rel = np.inf
    for it in range(max_iter):
        fz, gz, gbz = smooth_grad(X, y, z, zb, lam2, kind)
        w_new = soft(z - step * gz, step * lam1)
        b_new = zb - step * gbz
        f_new = smooth_loss(X, y, w_new, b_new, lam2, kind)
        if not np.isfinite(f_new):
            return w, b, it + 1, f_new, rel, False
        tk_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * tk * tk))
        z = w_new + ((tk - 1.0) / tk_new) * (w_new - w)
        zb = b_new + ((tk - 1.0) / tk_new) * (b_new - b)
        w, b, tk = w_new, b_new, tk_new
        f_hist.append(f_new)
        if it >= 40 and it % 10 == 0:
            f_ref = f_hist[-40]
            rel = abs(f_new - f_ref) / max(1.0, abs(f_new))
            if rel < tol:
                return w, b, it + 1, f_new, rel, True
    return w, b, max_iter, f_new, rel, False


def metrics(X, y, w, b, kind):
    z = X @ w + b
    if kind == "linear":
        p = z
        mse = float(np.mean((p - y) ** 2))
        lat1 = np.radians(p[:, 0]); lon1 = np.radians(p[:, 1])
        lat2 = np.radians(y[:, 0]); lon2 = np.radians(y[:, 1])
        a = (np.sin((lat2 - lat1) / 2) ** 2
             + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
        d = 2.0 * R_EARTH * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
        return {"mse": mse, "median_km": float(np.median(d)), "mean_km": float(np.mean(d))}
    prob = np.round(0.5 * (1.0 + np.tanh(0.5 * z)), 12)
    acc = float(np.mean((prob >= 0.5) == y))
    ll = float(-np.mean(y * np.log(np.clip(prob, 1e-12, 1.0))
                       + (1.0 - y) * np.log(np.clip(1.0 - prob, 1e-12, 1.0))))
    return {"acc": acc, "logloss": ll}


def intercept_only(X, y, kind):
    if kind == "linear":
        w = np.zeros((X.shape[1], y.shape[1]))
        b = y.mean(axis=0)
    else:
        w = np.zeros(X.shape[1])
        p = float(y.mean())
        b = np.float64(np.log(p / (1.0 - p)))
    return w, b


def sign_subgradient_world(X, y, lam1, lam2, epochs=1500, lr=None):
    """model.py-style sign() subgradient, only to show it never yields exact zeros."""
    N, D = X.shape
    if lr is None:
        lmax = power_lmax(X)
        lr = N / (2.0 * lmax + 2.0 * lam2 * N)  # 1/L of the smooth part
    w = np.zeros((D, y.shape[1]))
    b = y.mean(axis=0)
    for _ in range(epochs):
        r = X @ w + b - y
        g = 2.0 / N * (X.T @ r) + lam1 * np.sign(w) + 2.0 * lam2 * w
        gb = 2.0 / N * r.sum(axis=0)
        w -= lr * g
        b -= lr * gb
    return w, b, float(lr)


def nonzero_dims(w, thr=0.0):
    """Hidden dimensions carrying at least one nonzero weight (world probes are (D,2))."""
    w = np.asarray(w)
    if w.ndim == 1:
        return int(np.count_nonzero(np.abs(w) > thr))
    return int(np.count_nonzero(np.any(np.abs(w) > thr, axis=1)))


def wkey(name, lam1):
    return f"{name}_lam{lam1:.10g}"


def run_path(name, max_iter, tol):
    cfg = TASKS[name]
    data = load_task(name)
    X, y = data["train"]["X"], data["train"]["y"]
    Xv, yv = data["valid"]["X"], data["valid"]["y"]
    lmax = power_lmax(X)
    if cfg["kind"] == "linear":
        L = 2.0 * lmax / len(y) + 2.0 * cfg["lam2"]
    else:
        L = 0.25 * lmax / len(y) + 2.0 * cfg["lam2"]
    step = 1.0 / L

    # analytic lambda1 at which w=0 is optimal
    if cfg["kind"] == "linear":
        b0 = y.mean(axis=0)
        g0 = 2.0 / len(y) * (X.T @ (b0 - y))
    else:
        b0 = np.float64(np.log(y.mean() / (1.0 - y.mean())))
        p0 = 1.0 / (1.0 + np.exp(-b0))
        g0 = X.T @ (p0 - y) / len(y)
    lam1_max = float(np.max(np.abs(g0)))
    grid = sorted(GRIDS[name])  # ascending
    records = []
    weights = {}
    w, b = intercept_only(X, y, cfg["kind"])
    start = time.time()
    for lam1 in sorted(grid, reverse=True):
        if lam1 >= lam1_max:
            w, b = intercept_only(X, y, cfg["kind"])
            iters, rel, conv = 0, 0.0, True
            f = smooth_loss(X, y, w, b, cfg["lam2"], cfg["kind"])
        else:
            w, b, iters, f, rel, conv = fista(
                X, y, lam1, cfg["lam2"], cfg["kind"], w, b, step,
                max_iter=max_iter, tol=tol)
        tr = metrics(X, y, w, b, cfg["kind"])
        va = metrics(Xv, yv, w, b, cfg["kind"])
        weights[wkey(name, lam1)] = np.array(w, dtype=np.float64, copy=True)
        nz = nonzero_dims(w, 0.0)
        rec = {
            "lambda1": float(lam1),
            "lambda1_over_max": float(lam1 / lam1_max),
            "nonzero": nz,
            "nonzero_entries": int(np.count_nonzero(w)),
            "nonzero_1e-6": nonzero_dims(w, 1e-6),
            "frac_nonzero": float(nz / w.shape[0]),
            "l1_norm": float(np.sum(np.abs(w))),
            "l2_norm": float(np.sqrt(np.sum(w ** 2))),
            "intercept": np.asarray(b, dtype=np.float64).tolist(),
            "train": tr, "valid": va,
            "objective": float(f), "iters": int(iters), "rel_obj": float(rel),
            "converged": bool(conv), "elapsed_s": float(time.time() - start),
        }
        records.append(rec)
        print(f"[{name}] lam1={lam1:.4g} frac={lam1/lam1_max:.3g} nz={rec['nonzero']:4d} "
              f"({100*rec['frac_nonzero']:5.1f}%) it={iters:4d} conv={conv} "
              f"train={tr['mse'] if cfg['kind']=='linear' else tr['acc']:.4f} "
              f"valid={va['mse'] if cfg['kind']=='linear' else va['acc']:.4f}")
    return {
        "config": cfg, "lam1_max": lam1_max, "lmax_gram": lmax, "L": L, "step": step,
        "anchor": ANCHORS[name], "records": records, "weights": weights,
    }


def effective_number(w):
    w = np.abs(np.asarray(w, dtype=np.float64))
    if w.ndim == 1:
        w = w[:, None]
    vals = []
    for j in range(w.shape[1]):
        col = w[:, j]
        vals.append(float(col.sum() ** 2 / max(np.sum(col * col), 1e-30)))
    return float(np.mean(vals))


def make_figures(results, wstore):
    results = {k: v for k, v in results.items() if "records" in v}
    colors = {"world_L11": "#2166ac", "truth_L20": "#d95f02", "truth_L23": "#7570b3"}
    labels = {"world_L11": "world L11", "truth_L20": "truth L20", "truth_L23": "truth L23"}

    # ---- fig 1: density curves ----
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))
    for name, res in results.items():
        rec = sorted(res["records"], key=lambda r: r["lambda1_over_max"])
        x = [r["lambda1_over_max"] for r in rec]
        nz = [r["nonzero"] for r in rec]
        fr = [100 * r["frac_nonzero"] for r in rec]
        axes[0].plot(x, nz, "-o", ms=3, color=colors[name], label=labels[name])
        axes[1].plot(x, fr, "-o", ms=3, color=colors[name], label=labels[name])
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel(r"$\lambda_1 / \lambda_1^{max}$ (log)")
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=7.5)
    axes[0].axhline(2560, color="#999999", ls="--", lw=0.8)
    axes[0].axhline(1280, color="#bbbbbb", ls=":", lw=0.8)
    axes[0].set_ylabel("nonzero coefficients")
    axes[0].set_title("Nonzero count vs sparsity pressure")
    axes[1].set_ylabel("% nonzero")
    axes[1].set_title("Fraction of the 2560 coefficients kept")
    axes[1].set_ylim(-2, 103)
    fig.suptitle("A6: L1 sparsity pressure vs nonzero coefficients (proximal solver)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"a6_density_curves.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # ---- fig 2: metric vs sparsity ----
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))
    for name, res in results.items():
        rec = sorted(res["records"], key=lambda r: r["lambda1_over_max"])
        x = [r["lambda1_over_max"] for r in rec]
        if "world" in name:
            y = [r["valid"]["mse"] for r in rec]
            ax = axes[0]
            ax.plot(x, y, "-o", ms=3, color=colors[name], label=labels[name])
        else:
            y = [100 * r["valid"]["acc"] for r in rec]
            ax = axes[1]
            ax.plot(x, y, "-o", ms=3, color=colors[name], label=labels[name])
    for ax, ttl in zip(axes, ("world: valid MSE", "truth: valid accuracy (%)")):
        ax.set_xscale("log")
        ax.set_xlabel(r"$\lambda_1 / \lambda_1^{max}$ (log)")
        ax.set_title(ttl)
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=7.5)
        ax.axvline(1.0, color="#999999", ls="--", lw=0.8)
    axes[0].set_ylabel("valid MSE (deg$^2$)")
    axes[1].set_ylabel("valid accuracy (%)")
    # anchors at lambda1=0
    axes[0].axhline(ANCHORS["world_L11"]["valid_mse"], color=colors["world_L11"],
                    ls=":", lw=0.9)
    axes[0].annotate("L2 anchor 167.17", xy=(1e-5, ANCHORS["world_L11"]["valid_mse"]),
                     xytext=(0, 6), textcoords="offset points", fontsize=6.5,
                     color=colors["world_L11"])
    # zoom on the mild-L1 region where valid MSE dips below the L2 anchor
    world = results.get("world_L11")
    if world:
        rec = sorted(world["records"], key=lambda r: r["lambda1_over_max"])
        rec = [r for r in rec if r["lambda1_over_max"] <= 0.03]
        axin = axes[0].inset_axes([0.52, 0.52, 0.44, 0.4])
        axin.plot([r["lambda1_over_max"] for r in rec],
                  [r["valid"]["mse"] for r in rec], "-o", ms=2.5,
                  color=colors["world_L11"])
        axin.axhline(ANCHORS["world_L11"]["valid_mse"], color=colors["world_L11"],
                     ls=":", lw=0.8)
        axin.set_xscale("log")
        axin.set_ylim(160, 190)
        axin.grid(alpha=0.25, which="both")
        axin.tick_params(labelsize=6)
        axin.set_title(r"zoom: $\lambda_1/\lambda_1^{max}\leq 0.03$", fontsize=6.5)
    for nm in ("truth_L20", "truth_L23"):
        axes[1].axhline(100 * ANCHORS[nm]["valid_acc"], color=colors[nm], ls=":", lw=0.9)
    fig.suptitle("A6: valid metric vs L1 strength (valid only for reporting)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"a6_metric_vs_l1.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # ---- fig 3: coefficient structure ----
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))
    name = "truth_L20"
    sel = [1e-4, 1e-2, 5e-2]
    for lam, color in zip(sel, ("#2166ac", "#d95f02", "#7570b3")):
        key = wkey(name, lam)
        if key not in wstore:
            continue
        w = wstore[key]
        nz = np.abs(w[np.abs(w) > 0])
        axes[0].hist(nz, bins=np.geomspace(max(nz.min(), 1e-12), nz.max(), 40),
                     histtype="step", lw=1.2, color=color,
                     label=rf"$\lambda_1={lam:g}$ ({len(nz)} nonzero)")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("|w| (log)")
    axes[0].set_ylabel("count (log)")
    axes[0].set_title("truth L20: coefficient magnitudes")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.25, which="both")

    for name2, res in results.items():
        recs = sorted(res["records"], key=lambda r: r["lambda1_over_max"])
        x = [r["lambda1_over_max"] for r in recs]
        eff = []
        for r in recs:
            key = wkey(name2, r["lambda1"])
            eff.append(effective_number(wstore[key]) if key in wstore else np.nan)
        axes[1].plot(x, eff, "-o", ms=3, color=colors[name2], label=labels[name2])
    axes[1].plot([1e-5, 1], [2560, 2560], ls="--", lw=0.8, color="#999999", label="2560 coeffs")
    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"$\lambda_1 / \lambda_1^{max}$ (log)")
    axes[1].set_ylabel(r"effective number $(\sum|w|)^2/\sum w^2$")
    axes[1].set_title("Coefficient-mass concentration")
    axes[1].grid(alpha=0.25, which="both")
    axes[1].legend(fontsize=7)
    fig.suptitle("A6: what L1 removes (coefficient structure)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"a6_coef_structure.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_tables(results):
    with open(HERE / "a6_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    rows = []
    for name, res in results.items():
        if name == "solver_check" or "records" not in res:
            continue
        for r in res["records"]:
            row = {"task": name, "lambda1": r["lambda1"],
                   "lambda1_over_max": r["lambda1_over_max"],
                   "nonzero": r["nonzero"], "frac_nonzero": r["frac_nonzero"],
                   "iters": r["iters"], "converged": r["converged"]}
            for split in ("train", "valid"):
                for k, v in r[split].items():
                    row[f"{split}_{k}"] = v
            rows.append(row)
    pd.DataFrame(rows).to_csv(HERE / "a6_l1_density.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--replot", action="store_true")
    ap.add_argument("--recompute-metrics", action="store_true")
    ap.add_argument("--max-iter", type=int, default=3000)
    ap.add_argument("--tol", type=float, default=1e-7)
    args = ap.parse_args()
    if args.replot or args.recompute_metrics:
        with open(HERE / "a6_results.json", encoding="utf-8") as f:
            results = json.load(f)
        wstore = dict(np.load(HERE / "a6_weights.npz"))
    if args.recompute_metrics:
        for name in TASKS:
            data = load_task(name)
            for split in ("train", "valid"):
                data[split]["kind"] = TASKS[name]["kind"]
            for rec in results[name]["records"]:
                w = wstore[wkey(name, rec["lambda1"])]
                b = np.asarray(rec["intercept"], dtype=np.float64)
                rec["train"] = metrics(data["train"]["X"], data["train"]["y"], w, b, TASKS[name]["kind"])
                rec["valid"] = metrics(data["valid"]["X"], data["valid"]["y"], w, b, TASKS[name]["kind"])
            print(f"[recompute] {name} done")
        save_tables(results)
    if args.replot or args.recompute_metrics:
        make_figures(results, wstore)
        print("[replot] wrote figures")
        return
    if args.quick:
        for k in list(GRIDS):
            g = sorted(GRIDS[k])
            GRIDS[k] = [g[-1], g[len(g) // 2], g[0]]
        args.max_iter = min(args.max_iter, 300)
    t0 = time.time()
    results = {}
    wstore = {}
    for name in TASKS:
        results[name] = run_path(name, args.max_iter, args.tol)
        wstore.update(results[name].pop("weights"))
    np.savez_compressed(HERE / "a6_weights.npz", **wstore)
    # sign-subgradient check on world L11
    data = load_task("world_L11")
    w_sg, b_sg, lr_sg = sign_subgradient_world(
        data["train"]["X"], data["train"]["y"], 0.01, 1.0, epochs=800)
    results["solver_check"] = {
        "task": "world_L11", "lambda1": 0.01, "lambda2": 1.0,
        "sign_subgradient_nonzero": int(np.count_nonzero(w_sg)),
        "sign_subgradient_nonzero_1e-6": int(np.sum(np.abs(w_sg) > 1e-6)),
        "sign_subgradient_lr": lr_sg, "epochs": 800,
        "sign_subgradient_train_mse": float(np.mean(
            (data["train"]["X"] @ w_sg + b_sg - data["train"]["y"]) ** 2)),
        "proximal_nonzero_at_0.01": next((
            r["nonzero"] for r in results["world_L11"]["records"]
            if abs(r["lambda1"] - 0.01) < 1e-12), None),
        "proximal_train_mse_at_0.01": next((
            r["train"]["mse"] for r in results["world_L11"]["records"]
            if abs(r["lambda1"] - 0.01) < 1e-12), None),
    }
    with open(HERE / "a6_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    rows = []
    for name, res in results.items():
        if name == "solver_check":
            continue
        for r in res["records"]:
            row = {"task": name, "lambda1": r["lambda1"],
                   "lambda1_over_max": r["lambda1_over_max"],
                   "nonzero": r["nonzero"], "frac_nonzero": r["frac_nonzero"],
                   "iters": r["iters"], "converged": r["converged"]}
            for split in ("train", "valid"):
                for k, v in r[split].items():
                    row[f"{split}_{k}"] = v
            rows.append(row)
    pd.DataFrame(rows).to_csv(HERE / "a6_l1_density.csv", index=False)
    make_figures(results, wstore)
    print(f"[done] {time.time()-t0:.1f}s; wrote {HERE/'a6_results.json'}")


if __name__ == "__main__":
    main()
