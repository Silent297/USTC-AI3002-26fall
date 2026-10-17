# -*- coding: utf-8 -*-
"""A2: sklearn model family vs the linear probe ("is linear readout enough?").

Protocol (train/valid fixed; the valid split is used ONLY for reporting):
- features: z-scored single layer with TRAIN statistics; world uses L11
  (submission layer) and L2 (best single layer under the std protocol);
  truth uses L20 (peak layer).
- every model's hyperparameters are chosen by train-internal RepeatedKFold CV
  over a small grid; then refit on full train. SVR targets are z-scored per
  output with train stats (wrapper inverse-transforms before scoring).
- grids are sized from a local fit-time benchmark on truth z L20 (5088x2560):
  SVC 49.3 s/fit, LinearSVC 42.1 s/fit, HistGB 50 s/fit, RF(100) 8.6 s/fit,
  ET(100) 3.6 s/fit, KNN 0.3 s -> expensive families get few configs.
- in-protocol linear anchors (from A1 at fixed budget): world z L11 MSE 167.17,
  world z L2 157.62; truth z L20 acc 0.8667. Historical anchors: world raw L11
  150.43 (submission probe), truth raw GD L20 0.8967. Guide (student reference
  report, non-official): world KNN 125.27 / MLP-128-64 152.70 / their ridge
  167.17 (valid-selection protocol), truth RBF_SVC 87.00%.

Outputs: experiments/A2/figures/a2_model_bars.{png,pdf}, experiments/A2/a2_results.json
"""
import argparse
import json
import platform
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor,
                              HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.exceptions import ConvergenceWarning
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import RepeatedKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR, LinearSVC

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"

SEED = 0
WORLD_LAYERS = [11, 2]          # submission layer first, then best std single
TRUTH_LAYER = 20
WORLD_N_REPEATS = 2
TRUTH_N_REPEATS = 1             # truth fits are expensive (SVC/LinearSVC ~50 s/fit); 5 folds only

ANCHORS = {
    "world": {"linear_in_protocol_l11": 167.17, "linear_in_protocol_l2": 157.62,
              "submission_raw_l11": 150.43, "guide_knn": 125.27,
              "guide_mlp128_64": 152.70},
    "truth": {"linear_in_protocol_l20": 0.8667, "gd_raw_l20": 0.8967,
              "guide_rbf_svc": 0.8700},
}
GUIDE_NOTE = "student reference report (non-official)"


def load_layer(task, split, layer):
    X = np.load(ROOT / f"datasets/{task}/{split}.layer{layer}.npy").astype(np.float64)
    return X


def load_target(task, split):
    df = pd.read_csv(ROOT / f"datasets/{task}/{split}.csv")
    if task == "world":
        return df[["latitude", "longitude"]].to_numpy(np.float64)
    return df["label"].to_numpy(np.int64)


def zscore(train_X, X):
    mu = train_X.mean(axis=0)
    sd = train_X.std(axis=0)
    sd = np.where(sd == 0.0, 1.0, sd)
    return (X - mu) / sd


def world_models(D):
    g1 = 1.0 / D
    # SVR is target-scale sensitive (lat vs lon spans differ); z-score targets
    # per output with train stats, wrapper inverse-transforms before scoring.
    svr = lambda: MultiOutputRegressor(
        TransformedTargetRegressor(regressor=SVR(kernel="rbf", cache_size=1000),
                                   transformer=StandardScaler()))
    return [
        ("OLS", lambda: LinearRegression(), [{}]),
        ("Ridge", lambda: Ridge(), [{"alpha": a} for a in (50, 150, 500, 1500)]),
        ("KNN", lambda: KNeighborsRegressor(weights="distance", n_jobs=-1),
         [{"n_neighbors": k} for k in (3, 5, 10, 20)]),
        ("SVR-RBF", svr,
         [{"estimator__regressor__C": c, "estimator__regressor__gamma": g}
          for c in (1, 10, 100) for g in (g1, 3 * g1)]),
        ("KernelRidge-RBF", lambda: KernelRidge(kernel="rbf"),
         [{"alpha": a, "gamma": g} for a in (0.1, 1, 10) for g in (g1, 3 * g1)]),
        ("RandomForest", lambda: RandomForestRegressor(n_estimators=500, random_state=SEED, n_jobs=-1), [{}]),
        ("ExtraTrees", lambda: ExtraTreesRegressor(n_estimators=500, random_state=SEED, n_jobs=-1), [{}]),
        # HistGB has no native multi-output; wrap per coordinate
        ("HistGradientBoosting", lambda: MultiOutputRegressor(HistGradientBoostingRegressor(random_state=SEED)), [{}]),
    ]


def truth_models(D):
    g1 = 1.0 / D
    return [
        ("LogisticRegression", lambda: LogisticRegression(solver="lbfgs", max_iter=2000, tol=1e-4),
         [{"C": c} for c in (0.03, 0.0983, 0.3, 1.0)]),
        ("LinearSVC", lambda: LinearSVC(dual="auto", max_iter=5000),
         [{"C": c} for c in (0.1, 1.0)]),
        ("SVC-RBF", lambda: SVC(kernel="rbf", cache_size=1000),
         [{"C": c, "gamma": g} for c, g in ((1, g1), (10, g1), (10, 3 * g1))]),
        ("KNN", lambda: KNeighborsClassifier(weights="distance", n_jobs=-1),
         [{"n_neighbors": k} for k in (3, 5, 10, 20)]),
        ("RandomForest", lambda: RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1), [{}]),
        ("ExtraTrees", lambda: ExtraTreesClassifier(n_estimators=500, random_state=SEED, n_jobs=-1), [{}]),
        ("HistGradientBoosting", lambda: HistGradientBoostingClassifier(random_state=SEED), [{}]),
    ]


def score_world(est, X, y):
    return float(((est.predict(X) - y) ** 2).mean())


def score_truth(est, X, y):
    return float(np.mean(est.predict(X) == y))


def evaluate(task, name, make_est, grid, Xtr, ytr, Xva, yva, folds, scorer):
    best = None
    for params in grid:
        sse, count = 0.0, 0
        for tr_idx, va_idx in folds:
            est = make_est().set_params(**params)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                est.fit(Xtr[tr_idx], ytr[tr_idx])
            sse += scorer(est, Xtr[va_idx], ytr[va_idx]) * len(va_idx)
            count += len(va_idx)
        cv = sse / count
        if best is None or cv < best[0]:
            best = (cv, params)
    cv, params = best
    est = make_est().set_params(**params)
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        est.fit(Xtr, ytr)
    fit_s = time.time() - t0
    out = {"model": name, "params": {k: (float(v) if isinstance(v, (int, float)) else v)
                                     for k, v in params.items()},
           "cv_score": cv, "fit_s": round(fit_s, 1),
           "train_score": scorer(est, Xtr, ytr), "valid_score": scorer(est, Xva, yva)}
    return out


def run_task(task, layer, out_key, single_layer):
    Xtr_raw = load_layer(task, "train", layer)
    Xva_raw = load_layer(task, "valid", layer)
    Xtr = zscore(Xtr_raw, Xtr_raw)
    Xva = zscore(Xtr_raw, Xva_raw)
    ytr = load_target(task, "train")
    yva = load_target(task, "valid")
    D = Xtr.shape[1]

    n_repeats = WORLD_N_REPEATS if task == "world" else TRUTH_N_REPEATS
    folds = list(RepeatedKFold(n_splits=5, n_repeats=n_repeats, random_state=SEED).split(Xtr))
    models = world_models(D) if task == "world" else truth_models(D)
    scorer = score_world if task == "world" else score_truth

    print(f"[{task} L{layer}] {len(models)} models, {len(folds)} selection fits per config; "
          f"metric={'valid MSE' if task == 'world' else 'valid acc'}", flush=True)
    results = []
    for name, make_est, grid in models:
        res = evaluate(task, name, make_est, grid, Xtr, ytr, Xva, yva, folds, scorer)
        res["layer"] = layer
        results.append(res)
        print(f"[{task} L{layer}] {name:18s} params={res['params']} "
              f"cv={res['cv_score']:.4f} train={res['train_score']:.4f} "
              f"valid={res['valid_score']:.4f} ({res['fit_s']:.1f}s)", flush=True)
    return results, {"n_train": int(len(ytr)), "n_valid": int(len(yva)), "dim": int(D),
                     "n_cv_fits": len(folds)}


def fig_bars(world_l11, world_l2, truth_l20):
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.8), gridspec_kw={"width_ratios": [1.15, 1.15, 1.0]})
    panels = [
        (axes[0], world_l11, "valid MSE (deg$^2$)", "world L11 (submission layer)", "low"),
        (axes[1], world_l2, "valid MSE (deg$^2$)", "world L2 (best std single)", "low"),
        (axes[2], truth_l20, "valid accuracy", "truth L20 (peak layer)", "high"),
    ]
    for ax, results, xlabel, title, better in panels:
        rows = sorted(results, key=lambda r: r["valid_score"], reverse=(better == "high"))
        names = [r["model"] for r in rows]
        vals = [r["valid_score"] for r in rows]
        ypos = np.arange(len(rows))
        ax.barh(ypos, vals, color="tab:blue", alpha=0.85)
        for pos, r in zip(ypos, rows):
            label = ", ".join(f"{k.split('__')[-1]}={v:g}" for k, v in r["params"].items()) or "default"
            ax.text(r["valid_score"], pos, "  " + label, va="center", fontsize=7, color="0.25")
        ax.set_yticks(ypos, names, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_title(title, fontsize=10)
        task = "world" if "world" in title else "truth"
        if task == "world":
            ref = ANCHORS["world"]["linear_in_protocol_l11"] if "L11" in title else ANCHORS["world"]["linear_in_protocol_l2"]
            ax.axvline(ref, color="0.25", linestyle="--", linewidth=1.2,
                       label=f"linear ridge (A1) {ref:.1f}")
            ax.axvline(ANCHORS["world"]["submission_raw_l11"], color="tab:red", linestyle=":", linewidth=1.4,
                       label=f"submission raw L11 {ANCHORS['world']['submission_raw_l11']}")
            ax.axvline(ANCHORS["world"]["guide_knn"], color="tab:green", linestyle="-.", linewidth=1.0,
                       label=f"guide KNN {ANCHORS['world']['guide_knn']}*")
        else:
            ax.axvline(ANCHORS["truth"]["linear_in_protocol_l20"], color="0.25", linestyle="--", linewidth=1.2,
                       label=f"linear LR (A1) {ANCHORS['truth']['linear_in_protocol_l20']:.4f}")
            ax.axvline(ANCHORS["truth"]["gd_raw_l20"], color="tab:red", linestyle=":", linewidth=1.4,
                       label=f"GD raw L20 {ANCHORS['truth']['gd_raw_l20']:.4f}")
        ax.legend(fontsize=7.5, loc="lower right")
    fig.suptitle(f"* {GUIDE_NOTE}; their protocol selects on valid - shown for scale only", fontsize=8.5, y=1.00)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIG / "a2_model_bars.png", dpi=150)
    fig.savefig(FIG / "a2_model_bars.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=("world", "truth"), default=("world", "truth"))
    args = parser.parse_args()

    FIG.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    payload = {"seed": SEED, "anchors": ANCHORS, "guide_note": GUIDE_NOTE,
               "env": {"python": platform.python_version(), "numpy": np.__version__}}
    world_l11 = world_l2 = truth_l20 = None

    if "world" in args.tasks:
        world_l11, info = run_task("world", 11, "world_l11", True)
        payload["world_l11"] = {"results": world_l11, "info": info}
        world_l2, info = run_task("world", 2, "world_l2", True)
        payload["world_l2"] = {"results": world_l2, "info": info}
    if "truth" in args.tasks:
        truth_l20, info = run_task("truth", TRUTH_LAYER, "truth_l20", True)
        payload["truth_l20"] = {"results": truth_l20, "info": info}

    if world_l11 and world_l2 and truth_l20:
        fig_bars(world_l11, world_l2, truth_l20)
        print("[figs] wrote a2_model_bars", flush=True)

    payload["elapsed_s"] = round(time.time() - t0, 1)
    with (HERE / "a2_results.json").open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=1)
    print(f"[done] {time.time() - t0:.1f}s; wrote {HERE / 'a2_results.json'}", flush=True)


if __name__ == "__main__":
    main()
