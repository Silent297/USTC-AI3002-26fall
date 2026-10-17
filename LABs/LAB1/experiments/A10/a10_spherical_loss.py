"""A10: spherical (theta^2) loss vs an MSE control for the world L11 linear probe.

Protocol follows guide3 section 5, with one methodological change: the guide
selects hyperparameters on valid; we select them on train-only repeated K-fold
CV (5 folds x 3 seeds, stratified by latitude quintiles) and touch valid only
for the final report.  Features are z-scored with training-fold statistics.

Model: p_deg = 90 * (A z + a), A in R^{2x2560}, a in R^2.
Losses:
  MSE control: L = (1/2N) sum_i ||p_i - y_i||^2 + lam * ||A||_F^2
  Spherical:   L = (1/N) sum_i theta_eps,i^2 + lam * ||A||_F^2
               theta_eps = atan2(sqrt(||u x u_hat||^2 + eps), u . u_hat)
The spherical objective is invariant under the branch
(phi, lambda) -> (180 - phi, lambda + 180), which is a different coordinate
representation of the same 3-D point and may fall outside the legal lat range.
We report raw coordinate MSE (course metric), branch-canonicalised MSE and a
truth-dependent periodic diagnostic, plus out-of-range counts.

Usage:
  python experiments/A10/a10_spherical_loss.py --smoke
  python experiments/A10/a10_spherical_loss.py
  python experiments/A10/a10_spherical_loss.py --hybrid
  python experiments/A10/a10_spherical_loss.py --replot
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "experiments" / "A10"

TASK = "world"
LAYER = 11
N_FOLDS = 5
SEEDS = (0, 1, 2)
LRS = (1e-3, 3e-3)
LAMS = (0.0, 1e-3, 1e-2)
MAX_EPOCHS = 1200
PATIENCE = 150
OUT_SCALE = 90.0
EPS = 1e-12
R_EARTH = 6371.0
HYBRID_BETAS = (0.01, 0.1, 0.2, 0.3, 0.5, 1.0)

ANCHOR_MSE = 167.17072274253437
ANCHOR_MEDIAN_KM = 923.1205912354363
GUIDE_OLD = {"mse_mse": 207.46, "mse_mean_km": 1342.5,
             "sphere_mse": 164.17, "sphere_mean_km": 1173.5}

METRIC_KEYS = (
    "mse_raw", "mse_canon", "mse_periodic",
    "dist_median_km", "dist_mean_km", "dist_rmse_km", "dist_p90_km",
    "lat_mae_canon", "lon_mae_canon", "theta2_mean",
    "n_lat_oob", "n_lon_oob", "n_reflect", "reflect_rate",
)


def load_world():
    Xtr = np.load(ROOT / f"datasets/{TASK}/train.layer{LAYER}.npy").astype(np.float64)
    Xva = np.load(ROOT / f"datasets/{TASK}/valid.layer{LAYER}.npy").astype(np.float64)
    ytr = pd.read_csv(ROOT / f"datasets/{TASK}/train.csv")[["latitude", "longitude"]].to_numpy(np.float64)
    yva = pd.read_csv(ROOT / f"datasets/{TASK}/valid.csv")[["latitude", "longitude"]].to_numpy(np.float64)
    assert Xtr.shape == (500, 2560) and Xva.shape == (130, 2560), (Xtr.shape, Xva.shape)
    return Xtr, ytr, Xva, yva


def zscore_stats(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd == 0.0, 1.0, sd)
    return mu, sd


def zscore_apply(X, mu, sd):
    return (X - mu) / sd


def unit_vectors_torch(rad):
    lat, lon = rad[:, 0], rad[:, 1]
    return torch.stack([torch.cos(lat) * torch.cos(lon),
                        torch.cos(lat) * torch.sin(lon),
                        torch.sin(lat)], dim=1)


def spherical_theta_torch(p_deg, y_deg):
    u = unit_vectors_torch(torch.deg2rad(p_deg))
    v = unit_vectors_torch(torch.deg2rad(y_deg))
    c = (u * v).sum(dim=1).clamp(-1.0, 1.0)
    cross = torch.cross(u, v, dim=1)
    s = torch.sqrt((cross ** 2).sum(dim=1) + EPS)
    return torch.atan2(s, c)


def objective_terms(model, z, y_deg, kind, lam, beta):
    p = model(z)
    if kind == "mse":
        data = 0.5 * ((p - y_deg) ** 2).sum(dim=1).mean()
    else:
        theta = spherical_theta_torch(p, y_deg)
        data = (theta ** 2).mean()
        if kind == "hybrid":
            drad = torch.deg2rad(p) - torch.deg2rad(y_deg)
            data = data + beta * 0.5 * (drad ** 2).sum(dim=1).mean()
    reg = lam * (model.A ** 2).sum()
    return data + reg, data


def monitor_metric(p, y_deg, kind):
    if kind == "mse":
        return (0.5 * ((p - y_deg) ** 2).sum(dim=1).mean()).item()
    return (spherical_theta_torch(p, y_deg).mean() * R_EARTH).item()


class Probe(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.A = nn.Parameter(torch.zeros(2, dim))
        self.a = nn.Parameter(torch.zeros(2))

    def forward(self, z):
        return OUT_SCALE * (z @ self.A.T + self.a)


def reset_model(model, seed):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        model.A.normal_(0.0, 0.001, generator=g)
        model.a.zero_()


def check_sphere_gradient():
    torch.manual_seed(123)
    y = torch.tensor([[10.0, 20.0], [-30.0, 170.0], [0.0, -180.0],
                      [45.0, 0.0], [-60.0, -120.0], [89.0, 10.0],
                      [12.0, 20.5]], dtype=torch.float64)
    p0 = torch.tensor([[10.0, 20.0], [-30.0, 170.0], [0.0, -180.0],
                       [45.0, 0.0], [-60.0, -120.0], [89.0, 10.0],
                       [12.0, 20.0]], dtype=torch.float64)
    p = (p0 + 0.01 * torch.randn_like(p0)).requires_grad_(True)
    loss = (spherical_theta_torch(p, y) ** 2).mean()
    loss.backward()
    grad_auto = p.grad.clone()

    pr = torch.deg2rad(p.detach())
    yr = torch.deg2rad(y)
    u = unit_vectors_torch(pr)
    v = unit_vectors_torch(yr)
    c = (u * v).sum(dim=1).clamp(-1.0, 1.0)
    cross = torch.cross(u, v, dim=1)
    s = torch.sqrt((cross ** 2).sum(dim=1) + EPS)
    theta = torch.atan2(s, c)
    dldc = -2.0 * theta / s
    phi_t, lam_t = yr[:, 0], yr[:, 1]
    phi_p, lam_p = pr[:, 0], pr[:, 1]
    dlam = lam_p - lam_t
    dc_dphi = torch.sin(phi_t) * torch.cos(phi_p) - torch.cos(phi_t) * torch.sin(phi_p) * torch.cos(dlam)
    dc_dlam = -torch.cos(phi_t) * torch.cos(phi_p) * torch.sin(dlam)
    grad_analytic = torch.stack([dldc * dc_dphi / len(p),
                                 dldc * dc_dlam / len(p)], dim=1) * (torch.pi / 180.0)
    err = (grad_auto - grad_analytic).abs().max().item()
    assert err < 1e-6, f"spherical gradient mismatch: max abs err {err:.3e}"
    return err


def canonicalize_points(p):
    """Map each point to the legal latitude branch |lat| <= 90.

    (phi, lam) and (180 - phi, lam + 180) map to the same unit vector; the
    spherical loss cannot distinguish them.  This returns the legal-latitude
    representative plus a mask of samples that needed a reflection.  Longitude
    is left in whatever range the reflection produces; periodic longitude
    effects are handled separately in coord_metrics.
    """
    p = np.asarray(p, dtype=np.float64)
    lat = p[:, 0].copy()
    lon = p[:, 1].copy()
    lat_mod = ((lat + 180.0) % 360.0) - 180.0
    flip = (lat_mod > 90.0) | (lat_mod < -90.0)
    lat_c = np.where(lat_mod > 90.0, 180.0 - lat_mod,
                     np.where(lat_mod < -90.0, -180.0 - lat_mod, lat_mod))
    lon_c = np.where(flip, lon + 180.0, lon)
    return np.column_stack([lat_c, lon_c]), flip


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=np.float64))
                              for v in (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2.0) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2.0) ** 2)
    return 2.0 * R_EARTH * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def theta_rad_np(p_deg, y_deg):
    p = np.radians(np.asarray(p_deg, dtype=np.float64))
    y = np.radians(np.asarray(y_deg, dtype=np.float64))
    u = np.stack([np.cos(p[:, 0]) * np.cos(p[:, 1]),
                  np.cos(p[:, 0]) * np.sin(p[:, 1]),
                  np.sin(p[:, 0])], axis=1)
    v = np.stack([np.cos(y[:, 0]) * np.cos(y[:, 1]),
                  np.cos(y[:, 0]) * np.sin(y[:, 1]),
                  np.sin(y[:, 0])], axis=1)
    c = np.sum(u * v, axis=1)
    cross = np.cross(u, v)
    s = np.sqrt(np.sum(cross ** 2, axis=1) + EPS)
    return np.arctan2(s, c)


def coord_metrics(p, y):
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    assert p.shape == y.shape and np.isfinite(p).all(), "non-finite or shape-mismatched predictions"
    pc, flip = canonicalize_points(p)

    mse_raw = float(np.mean((p - y) ** 2))
    mse_canon = float(np.mean((pc - y) ** 2))
    err_stack = []
    for shift in (-360.0, 0.0, 360.0):
        q = pc.copy()
        q[:, 1] += shift
        err_stack.append(np.mean((q - y) ** 2, axis=1))
    mse_periodic = float(np.mean(np.min(np.stack(err_stack, axis=1), axis=1)))

    d = haversine_km(p[:, 0], p[:, 1], y[:, 0], y[:, 1])
    theta = theta_rad_np(p, y)
    return {
        "n": int(len(p)),
        "mse_raw": mse_raw,
        "mse_canon": mse_canon,
        "mse_periodic": mse_periodic,
        "dist_median_km": float(np.median(d)),
        "dist_mean_km": float(np.mean(d)),
        "dist_rmse_km": float(np.sqrt(np.mean(d ** 2))),
        "dist_p90_km": float(np.percentile(d, 90)),
        "lat_mae_canon": float(np.mean(np.abs(pc[:, 0] - y[:, 0]))),
        "lon_mae_canon": float(np.mean(np.abs(pc[:, 1] - y[:, 1]))),
        "theta2_mean": float(np.mean(theta ** 2)),
        "n_lat_oob": int(np.sum((p[:, 0] < -90.0) | (p[:, 0] > 90.0))),
        "n_lon_oob": int(np.sum((p[:, 1] < -180.0) | (p[:, 1] > 180.0))),
        "n_reflect": int(np.sum(flip)),
        "reflect_rate": float(np.mean(flip)),
    }


def train_fold(Xtr, ytr, Xva, yva, kind, lr, lam, beta, seed, max_epochs, patience):
    model = Probe(Xtr.shape[1])
    reset_model(model, seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.eval()
    with torch.no_grad():
        best = float(monitor_metric(model(Xva), yva, kind))
    best_epoch = 0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    bad = 0
    diverged = False
    curve = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        loss, _ = objective_terms(model, Xtr, ytr, kind, lam, beta)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        model.eval()
        with torch.no_grad():
            metric = float(monitor_metric(model(Xva), yva, kind))
        if not np.isfinite(float(loss.detach())) or not np.isfinite(metric):
            diverged = True
            break
        curve.append(metric)
        if metric < best - 1e-9:
            best = metric
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        ptr = model(Xtr).numpy().astype(np.float64)
        pva = model(Xva).numpy().astype(np.float64)
    rec = {
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "epochs_run": int(len(curve)),
        "best_monitor": float(best),
        "diverged": bool(diverged),
        "weight_norm": float(model.A.detach().norm().item()),
        "train": coord_metrics(ptr, ytr.detach().numpy().astype(np.float64)),
        "val": coord_metrics(pva, yva.detach().numpy().astype(np.float64)),
    }
    return rec, curve, ptr, pva


def aggregate_curves(curves):
    maxlen = max(len(c) for c in curves)
    arr = np.full((len(curves), maxlen), np.nan)
    for i, c in enumerate(curves):
        arr[i, :len(c)] = c
    out = {"mean": np.nanmean(arr, axis=0),
           "q25": np.nanpercentile(arr, 25, axis=0),
           "q75": np.nanpercentile(arr, 75, axis=0)}
    return {k: [None if not np.isfinite(v) else float(v) for v in vals]
            for k, vals in out.items()}


def run_cv(X, y, kind, lr, lam, beta, seeds, n_folds, max_epochs, patience):
    strata = pd.qcut(pd.Series(y[:, 0]), n_folds, labels=False, duplicates="drop").to_numpy()
    fold_records = []
    curves = []
    oof_raw = {}
    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        oof = np.full_like(y, np.nan)
        for fold, (tr, va) in enumerate(skf.split(X, strata)):
            mu, sd = zscore_stats(X[tr])
            Xtr = torch.tensor(zscore_apply(X[tr], mu, sd), dtype=torch.float32)
            Xva = torch.tensor(zscore_apply(X[va], mu, sd), dtype=torch.float32)
            ytr = torch.tensor(y[tr], dtype=torch.float32)
            yva = torch.tensor(y[va], dtype=torch.float32)
            rec, curve, _, pva = train_fold(Xtr, ytr, Xva, yva, kind, lr, lam, beta,
                                            seed, max_epochs, patience)
            rec["fold"] = int(fold)
            fold_records.append(rec)
            curves.append(curve)
            oof[va] = pva
        assert np.isfinite(oof).all(), f"non-finite OOF predictions for seed {seed}"
        oof_raw[int(seed)] = oof

    agg = {}
    oof_metrics = {int(s): coord_metrics(oof_raw[int(s)], y) for s in seeds}
    for key in METRIC_KEYS:
        vals = [oof_metrics[int(s)][key] for s in seeds]
        agg[f"cv_oof_{key}_mean"] = float(np.mean(vals))
        agg[f"cv_oof_{key}_std"] = float(np.std(vals))
    best_epochs = [r["best_epoch"] for r in fold_records]
    agg["best_epoch_median"] = float(np.median(best_epochs))
    agg["best_epoch_min"] = int(np.min(best_epochs))
    agg["best_epoch_max"] = int(np.max(best_epochs))
    agg["best_epochs"] = [int(v) for v in best_epochs]
    agg["n_diverged_folds"] = int(sum(r["diverged"] for r in fold_records))
    agg["folds"] = fold_records
    agg["oof_by_seed"] = oof_metrics
    return agg, aggregate_curves(curves), oof_raw


def ridge_dual_fit_predict(Xtr, ytr, Xva, lambda2=1.0):
    ybar = ytr.mean(axis=0)
    yc = ytr - ybar
    K = Xtr @ Xtr.T
    n = K.shape[0]
    alpha = np.linalg.solve(K + lambda2 * n * np.eye(n), yc)
    w = Xtr.T @ alpha
    return Xva @ w + ybar, Xtr @ w + ybar, w, ybar


def run_ridge(Xtr, ytr, Xva, yva):
    mu, sd = zscore_stats(Xtr)
    Xtrz = zscore_apply(Xtr, mu, sd)
    Xvaz = zscore_apply(Xva, mu, sd)
    pred_va, pred_tr, w, b = ridge_dual_fit_predict(Xtrz, ytr, Xvaz, lambda2=1.0)
    return {
        "valid": coord_metrics(pred_va, yva),
        "train": coord_metrics(pred_tr, ytr),
        "pred_va": pred_va,
        "pred_tr": pred_tr,
        "w_z": w,
        "b": b,
        "mu": mu,
        "sd": sd,
    }


def train_final(Xtr, ytr, Xva, kind, lr, lam, beta, seed, epochs):
    mu, sd = zscore_stats(Xtr)
    Xtrz = torch.tensor(zscore_apply(Xtr, mu, sd), dtype=torch.float32)
    Xvaz = torch.tensor(zscore_apply(Xva, mu, sd), dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32)
    model = Probe(Xtr.shape[1])
    reset_model(model, seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        loss, _ = objective_terms(model, Xtrz, yt, kind, lam, beta)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
    model.eval()
    with torch.no_grad():
        ptr = model(Xtrz).numpy().astype(np.float64)
        pva = model(Xvaz).numpy().astype(np.float64)
    weights = {"A_deg": OUT_SCALE * model.A.detach().numpy().copy(),
               "a_deg": OUT_SCALE * model.a.detach().numpy().copy(),
               "weight_norm": float(model.A.detach().norm().item())}
    return model, ptr, pva, weights


def ensemble_predictions(preds, kind):
    if kind == "sphere":
        u = unit_vectors_np(np.deg2rad(np.stack(preds, axis=0)))
        um = u.mean(axis=0)
        um /= np.linalg.norm(um, axis=1, keepdims=True)
        lat = np.rad2deg(np.arcsin(np.clip(um[:, 2], -1.0, 1.0)))
        lon = np.rad2deg(np.arctan2(um[:, 1], um[:, 0]))
        return np.column_stack([lat, lon])
    return np.mean(np.stack(preds, axis=0), axis=0)


def unit_vectors_np(rad):
    lat, lon = rad[..., 0], rad[..., 1]
    return np.stack([np.cos(lat) * np.cos(lon),
                     np.cos(lat) * np.sin(lon),
                     np.sin(lat)], axis=-1)


def run_final(kind, lr, lam, beta, epochs, seeds, Xtr, ytr, Xva, yva):
    per_seed = {}
    preds = []
    weights = {}
    for seed in seeds:
        _, ptr, pva, w = train_final(Xtr, ytr, Xva, kind, lr, lam, beta, seed, epochs)
        per_seed[str(seed)] = {
            "train": coord_metrics(ptr, ytr),
            "valid": coord_metrics(pva, yva),
            "epochs": int(epochs),
            "weight_norm": w["weight_norm"],
        }
        preds.append(pva)
        weights[int(seed)] = w
    ens = ensemble_predictions(preds, kind)
    per_seed["ensemble"] = {"valid": coord_metrics(ens, yva)}
    summary = {}
    for key in METRIC_KEYS:
        vals = [per_seed[str(s)]["valid"][key] for s in seeds]
        summary[key + "_mean"] = float(np.mean(vals))
        summary[key + "_std"] = float(np.std(vals))
    return {"per_seed": per_seed, "seed_mean_std": summary,
            "preds": preds, "ensemble_pred": ens, "weights": weights}


def select_best(cv_results, kind):
    key = "cv_oof_mse_raw_mean" if kind == "mse" else "cv_oof_dist_mean_km_mean"
    clean = {k: v for k, v in cv_results.items() if v["n_diverged_folds"] == 0}
    pool = clean or cv_results
    return min(pool, key=lambda k: cv_results[k][key])


def config_key(lr, lam):
    return f"lr{lr:g}_lam{lam:g}"


def summarize_cv(cv):
    return {k: v for k, v in cv.items() if k != "folds"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--hybrid", action="store_true")
    parser.add_argument("--replot", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        outdir = OUTDIR / "_smoke"
        seeds = (0,)
        lrs = (1e-3,)
        lams = (0.0,)
        n_folds, max_epochs, patience = 2, 60, 20
    else:
        outdir = OUTDIR
        seeds = SEEDS
        lrs = LRS
        lams = LAMS
        n_folds, max_epochs, patience = N_FOLDS, MAX_EPOCHS, PATIENCE
    outdir.mkdir(parents=True, exist_ok=True)

    if args.replot:
        payload = json.loads((outdir / "a10_results.json").read_text(encoding="utf-8"))
        pred_data = np.load(outdir / "a10_predictions.npz")
        make_figures(payload, pred_data, outdir / "figures")
        return

    torch.set_num_threads(4)
    t0 = time.time()
    grad_err = check_sphere_gradient()
    print(f"[A10] spherical gradient check OK (max abs err {grad_err:.2e})", flush=True)

    Xtr, ytr, Xva, yva = load_world()
    ridge = run_ridge(Xtr, ytr, Xva, yva)
    assert abs(ridge["valid"]["mse_raw"] - ANCHOR_MSE) < 1e-3, ridge["valid"]["mse_raw"]
    assert abs(ridge["valid"]["dist_median_km"] - ANCHOR_MEDIAN_KM) < 0.05, ridge["valid"]["dist_median_km"]
    print(f"[A10] ridge anchor OK: MSE {ridge['valid']['mse_raw']:.4f}, "
          f"median {ridge['valid']['dist_median_km']:.4f} km", flush=True)

    kinds = ["mse", "sphere"]
    grid = [(lr, lam) for lr in lrs for lam in lams]
    all_cv = {k: {} for k in kinds}
    curve_store = {}
    oof_store = {}
    for kind in kinds:
        for lr, lam in grid:
            t_cfg = time.time()
            cv, curve, oof_raw = run_cv(Xtr, ytr, kind, lr, lam, 0.0, seeds,
                                        n_folds, max_epochs, patience)
            key = config_key(lr, lam)
            all_cv[kind][key] = cv
            curve_store[(kind, key)] = curve
            oof_store[(kind, key)] = oof_raw
            print(f"[{kind}] {key}: OOF MSE {cv['cv_oof_mse_raw_mean']:.1f}"
                  f"±{cv['cv_oof_mse_raw_std']:.1f}, mean km {cv['cv_oof_dist_mean_km_mean']:.1f}"
                  f"±{cv['cv_oof_dist_mean_km_std']:.1f}, refl {cv['cv_oof_reflect_rate_mean']*100:.1f}%,"
                  f" ep med {cv['best_epoch_median']:.0f}, div {cv['n_diverged_folds']}"
                  f" ({time.time()-t_cfg:.1f}s)", flush=True)

    selected = {}
    pred_store = {}
    pred_seed_store = {}
    weight_store = {}
    oof_selected = {}
    for kind in kinds:
        best_key = select_best(all_cv[kind], kind)
        cv = all_cv[kind][best_key]
        lr = float(best_key.split("_")[0][2:])
        lam = float(best_key.split("_")[1][3:])
        epochs = max(1, int(round(cv["best_epoch_median"])))
        final = run_final(kind, lr, lam, 0.0, epochs, seeds, Xtr, ytr, Xva, yva)
        selected[kind] = {
            "lr": lr, "lambda2": lam, "beta": 0.0,
            "final_epochs": epochs,
            "cv": summarize_cv(cv),
            "cv_curve": curve_store[(kind, best_key)],
            "final": {k: v for k, v in final.items() if k not in ("preds", "ensemble_pred", "weights")},
        }
        pred_store[kind] = final["ensemble_pred"]
        pred_seed_store[kind] = {int(s): final["preds"][i] for i, s in enumerate(seeds)}
        weight_store[kind] = final["weights"]
        oof_selected[kind] = oof_store[(kind, best_key)]
        print(f"[{kind}] selected {best_key}, final {epochs} ep: "
              f"valid MSE {final['seed_mean_std']['mse_raw_mean']:.1f}"
              f"±{final['seed_mean_std']['mse_raw_std']:.1f}, "
              f"median {final['seed_mean_std']['dist_median_km_mean']:.1f} km, "
              f"mean {final['seed_mean_std']['dist_mean_km_mean']:.1f} km, "
              f"refl {final['seed_mean_std']['n_reflect_mean']:.1f}", flush=True)

    if args.hybrid and not args.smoke:
        sphere_lr = selected["sphere"]["lr"]
        sphere_lam = selected["sphere"]["lambda2"]
        all_cv["hybrid"] = {}
        for beta in HYBRID_BETAS:
            t_cfg = time.time()
            cv, curve, oof_raw = run_cv(Xtr, ytr, "hybrid", sphere_lr, sphere_lam, beta,
                                        seeds, n_folds, max_epochs, patience)
            key = f"beta{beta:g}"
            all_cv["hybrid"][key] = cv
            curve_store[("hybrid", key)] = curve
            oof_store[("hybrid", key)] = oof_raw
            print(f"[hybrid] {key}: OOF MSE {cv['cv_oof_mse_raw_mean']:.1f}, "
                  f"mean km {cv['cv_oof_dist_mean_km_mean']:.1f}, "
                  f"refl {cv['cv_oof_reflect_rate_mean']*100:.1f}%, "
                  f"ep med {cv['best_epoch_median']:.0f}, div {cv['n_diverged_folds']}"
                  f" ({time.time()-t_cfg:.1f}s)", flush=True)
        clean = {k: v for k, v in all_cv["hybrid"].items() if v["n_diverged_folds"] == 0}
        pool_all = clean or all_cv["hybrid"]
        zero_flip = [k for k, v in pool_all.items()
                     if v["cv_oof_reflect_rate_mean"] == 0.0]
        if zero_flip:
            best_key = min(zero_flip, key=lambda k: pool_all[k]["cv_oof_dist_mean_km_mean"])
        else:
            min_flip = min(v["cv_oof_reflect_rate_mean"] for v in pool_all.values())
            pool = [k for k, v in pool_all.items()
                    if v["cv_oof_reflect_rate_mean"] == min_flip]
            best_key = min(pool, key=lambda k: pool_all[k]["cv_oof_dist_mean_km_mean"])
        beta = float(best_key[4:])
        cv = all_cv["hybrid"][best_key]
        epochs = max(1, int(round(cv["best_epoch_median"])))
        final = run_final("hybrid", sphere_lr, sphere_lam, beta, epochs,
                          seeds, Xtr, ytr, Xva, yva)
        selected["hybrid"] = {
            "lr": sphere_lr, "lambda2": sphere_lam, "beta": beta,
            "final_epochs": epochs,
            "cv": summarize_cv(cv),
            "cv_curve": curve_store[("hybrid", best_key)],
            "final": {k: v for k, v in final.items() if k not in ("preds", "ensemble_pred", "weights")},
        }
        pred_store["hybrid"] = final["ensemble_pred"]
        pred_seed_store["hybrid"] = {int(s): final["preds"][i] for i, s in enumerate(seeds)}
        weight_store["hybrid"] = final["weights"]
        oof_selected["hybrid"] = oof_store[("hybrid", best_key)]
        print(f"[hybrid] selected {best_key} at lr={sphere_lr:g} lam={sphere_lam:g}, "
              f"final {epochs} ep: valid MSE {final['seed_mean_std']['mse_raw_mean']:.1f}, "
              f"median {final['seed_mean_std']['dist_median_km_mean']:.1f} km, "
              f"refl {final['seed_mean_std']['n_reflect_mean']:.1f}", flush=True)

    payload = {
        "protocol": {
            "task": TASK, "layer": LAYER, "n_folds": n_folds, "seeds": list(seeds),
            "lrs": list(lrs), "lambdas": list(lams), "max_epochs": max_epochs,
            "patience": patience, "out_scale": OUT_SCALE, "eps": EPS,
            "r_earth_km": R_EARTH,
            "selection": "OOF MSE (mse group) / OOF mean great-circle km (sphere & hybrid)",
            "valid_usage": "final report only; no early stopping or configuration choice",
            "hybrid_betas": list(HYBRID_BETAS) if args.hybrid and not args.smoke else [],
            "guide_old_reference": GUIDE_OLD,
        },
        "anchors": {"ridge_valid_mse": ridge["valid"]["mse_raw"],
                    "ridge_valid_median_km": ridge["valid"]["dist_median_km"]},
        "ridge": {"valid": ridge["valid"], "train": ridge["train"]},
        "cv": all_cv,
        "selected": selected,
        "env": {"python": platform.python_version(), "numpy": np.__version__,
                "torch": torch.__version__, "torch_threads": torch.get_num_threads()},
        "elapsed_s": round(time.time() - t0, 1),
    }
    (outdir / "a10_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    pred_npz = {"y_true": yva, "ridge_pred": ridge["pred_va"], "ridge_pred_train": ridge["pred_tr"]}
    weights_npz = {"ridge_w_z": ridge["w_z"], "ridge_b": ridge["b"],
                   "ridge_mu": ridge["mu"], "ridge_sd": ridge["sd"]}
    for kind in selected:
        pred_npz[f"{kind}_pred_ens"] = pred_store[kind]
        for seed in seeds:
            pred_npz[f"{kind}_pred_seed{seed}"] = pred_seed_store[kind][int(seed)]
            pred_npz[f"{kind}_oof_seed{seed}"] = oof_selected[kind][int(seed)]
            w = weight_store[kind][int(seed)]
            weights_npz[f"{kind}_seed{seed}_A_deg"] = w["A_deg"]
            weights_npz[f"{kind}_seed{seed}_a_deg"] = w["a_deg"]
    np.savez_compressed(outdir / "a10_predictions.npz", **pred_npz)
    np.savez_compressed(outdir / "a10_weights.npz", **weights_npz)

    make_figures(payload, pred_npz, outdir / "figures")
    print(f"[done] {payload['elapsed_s']:.1f}s; wrote {outdir / 'a10_results.json'}", flush=True)


def make_figures(payload, pred_data, figdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                         "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8})

    methods = ["ridge"] + [m for m in ("mse", "sphere", "hybrid") if m in payload["selected"]]
    labels = {"ridge": "Ridge (closed form)", "mse": "MSE control",
              "sphere": "Spherical $\\theta^2$", "hybrid": "Spherical + MSE tie-break"}
    if "hybrid" in payload["selected"]:
        labels["hybrid"] = (f"Hybrid $\\theta^2$+MSE "
                            f"($\\beta$={payload['selected']['hybrid']['beta']:g})")
    short = {"ridge": "Ridge", "mse": "MSE", "sphere": "Sphere",
             "hybrid": "Hybrid"}
    colors = {"ridge": "0.55", "mse": "#3b6fb6", "sphere": "#d97706", "hybrid": "#7c3aed"}
    first_seed = payload["protocol"]["seeds"][0]

    def method_pred(m):
        key = "ridge_pred" if m == "ridge" else f"{m}_pred_seed{first_seed}"
        return np.asarray(pred_data[key], dtype=np.float64)

    # ---- training curves -------------------------------------------------
    n_fits = payload["protocol"]["n_folds"] * len(payload["protocol"]["seeds"])
    specs = [("mse", "MSE control", "fold-val MSE (deg$^2$)", True),
             ("sphere", "Spherical $\\theta^2$", "fold-val mean distance (km)", False)]
    if "hybrid" in payload["selected"]:
        specs.append(("hybrid", "Hybrid $\\theta^2$+MSE", "fold-val mean distance (km)", False))
    fig, axes = plt.subplots(1, len(specs), figsize=(5.0 * len(specs), 3.7),
                             constrained_layout=True)
    for ax, (kind, title, ylabel, logy) in zip(axes, specs):
        sel = payload["selected"][kind]
        c = sel["cv_curve"]
        x = np.arange(1, len(c["mean"]) + 1)
        mean = np.array([np.nan if v is None else v for v in c["mean"]])
        q25 = np.array([np.nan if v is None else v for v in c["q25"]])
        q75 = np.array([np.nan if v is None else v for v in c["q75"]])
        ax.plot(x, mean, color=colors[kind], lw=1.5, label=f"mean over {n_fits} fits")
        ax.fill_between(x, q25, q75, color=colors[kind], alpha=0.18, lw=0, label="IQR over fits")
        ax.axvline(sel["final_epochs"], color="0.2", ls="--", lw=0.9,
                   label=f"final budget {sel['final_epochs']} ep")
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        extra = f", $\\beta$={sel['beta']:g}" if kind == "hybrid" else ""
        ax.set_title(f"{title}: lr={sel['lr']:g}, $\\lambda_2$={sel['lambda2']:g}{extra}")
        if logy:
            ax.set_yscale("log")
        ax.grid(alpha=0.2, lw=0.5)
        ax.legend(frameon=False)
    save_fig(fig, figdir / "a10_training_curves")

    # ---- metric bars -----------------------------------------------------
    panels = [("mse_raw", "valid coord MSE (deg$^2$, raw)", True),
              ("mse_canon", "valid coord MSE (deg$^2$, branch-canon.)", True),
              ("dist_median_km", "median great-circle distance (km)", False),
              ("dist_mean_km", "mean great-circle distance (km)", False)]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.9), constrained_layout=True)
    for ax, (metric, title, logy) in zip(axes, panels):
        vals, errs, ticks, notes = [], [], [], []
        for m in methods:
            if m == "ridge":
                vals.append(payload["ridge"]["valid"][metric])
                errs.append(0.0)
                ticks.append(short[m])
            else:
                ss = payload["selected"][m]["final"]["seed_mean_std"]
                vals.append(ss[metric + "_mean"])
                errs.append(ss[metric + "_std"])
                ticks.append(short[m])
                notes.append(f"{short[m]}: refl {ss['n_reflect_mean']:.1f}, "
                             f"latOOB {ss['n_lat_oob_mean']:.1f}")
        x = np.arange(len(methods))
        bars = ax.bar(x, vals, color=[colors[m] for m in methods], width=0.62,
                      yerr=errs, capsize=3, error_kw={"lw": 0.8, "alpha": 0.7})
        ax.set_xticks(x)
        ax.set_xticklabels(ticks, fontsize=8)
        ax.set_title(title)
        top = max(v + e for v, e in zip(vals, errs))
        if logy:
            ax.set_yscale("log")
            ax.set_ylim(bottom=min(vals) * 0.6, top=top * 2.4)
        else:
            ax.set_ylim(0.0, top * 1.25)
        for b, v, e in zip(bars, vals, errs):
            ax.text(b.get_x() + b.get_width() / 2, (v + e) * (1.03 if logy else 1.02),
                    f"{v:.1f}" if v < 100 else f"{v:.0f}", ha="center", va="bottom", fontsize=7)
        ax.grid(axis="y", alpha=0.2, lw=0.5)
        if metric == "mse_raw" and notes:
            ax.text(0.02, 0.98, "valid, seed mean\n" + "\n".join(notes),
                    transform=ax.transAxes, va="top", ha="left", fontsize=6.4, color="0.25")
    save_fig(fig, figdir / "a10_metric_bars")

    # ---- hybrid beta sweep -----------------------------------------------
    if "hybrid" in payload["cv"]:
        sweep = payload["cv"]["hybrid"]
        keys = sorted(sweep, key=lambda k: float(k[4:]))
        betas = np.array([float(k[4:]) for k in keys])
        refl = np.array([sweep[k]["cv_oof_reflect_rate_mean"] * 100 for k in keys])
        mean_km = np.array([sweep[k]["cv_oof_dist_mean_km_mean"] for k in keys])
        mean_std = np.array([sweep[k]["cv_oof_dist_mean_km_std"] for k in keys])
        mse = np.array([sweep[k]["cv_oof_mse_raw_mean"] for k in keys])
        mse_std = np.array([sweep[k]["cv_oof_mse_raw_std"] for k in keys])
        sel_beta = payload["selected"]["hybrid"]["beta"]
        sphere_mean = payload["selected"]["sphere"]["cv"]["cv_oof_dist_mean_km_mean"]
        sphere_mse = payload["selected"]["sphere"]["cv"]["cv_oof_mse_raw_mean"]
        mse_mean = payload["selected"]["mse"]["cv"]["cv_oof_dist_mean_km_mean"]
        mse_ctrl_mse = payload["selected"]["mse"]["cv"]["cv_oof_mse_raw_mean"]
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
        ax = axes[0]
        ax.plot(betas, refl, "o-", color=colors["hybrid"], lw=1.4, ms=4)
        ax.axvline(sel_beta, color="0.2", ls="--", lw=0.9,
                   label=f"selected $\\beta$={sel_beta:g}")
        ax.set_xscale("log")
        ax.set_xlabel(r"MSE tie-break weight $\beta$")
        ax.set_ylabel("OOF reflection rate (%)")
        ax.set_title("branch flips vs hybrid weight")
        ax.grid(alpha=0.2, lw=0.5)
        ax.legend(frameon=False)
        ax = axes[1]
        ax.errorbar(betas, mean_km, yerr=mean_std, fmt="o-", color=colors["hybrid"],
                    lw=1.4, ms=4, capsize=2)
        ax.axhline(sphere_mean, color=colors["sphere"], ls=":", lw=1.0,
                   label="spherical selected")
        ax.axhline(mse_mean, color=colors["mse"], ls=":", lw=1.0, label="MSE selected")
        ax.axvline(sel_beta, color="0.2", ls="--", lw=0.9)
        ax.set_xscale("log")
        ax.set_xlabel(r"MSE tie-break weight $\beta$")
        ax.set_ylabel("OOF mean distance (km)")
        ax.set_title("distance vs hybrid weight")
        ax.grid(alpha=0.2, lw=0.5)
        ax.legend(frameon=False)
        ax = axes[2]
        ax.errorbar(betas, mse, yerr=mse_std, fmt="o-", color=colors["hybrid"],
                    lw=1.4, ms=4, capsize=2)
        ax.axhline(sphere_mse, color=colors["sphere"], ls=":", lw=1.0,
                   label="spherical selected")
        ax.axhline(mse_ctrl_mse, color=colors["mse"], ls=":", lw=1.0, label="MSE selected")
        ax.axvline(sel_beta, color="0.2", ls="--", lw=0.9)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"MSE tie-break weight $\beta$")
        ax.set_ylabel("OOF coordinate MSE (deg$^2$, raw)")
        ax.set_title("raw coordinate MSE vs hybrid weight")
        ax.grid(alpha=0.2, lw=0.5)
        ax.legend(frameon=False)
        save_fig(fig, figdir / "a10_hybrid_sweep")

    # ---- distance / squared-error ECDF -----------------------------------
    y = np.asarray(pred_data["y_true"], dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), constrained_layout=True)
    for m in methods:
        p = method_pred(m)
        d = haversine_km(p[:, 0], p[:, 1], y[:, 0], y[:, 1])
        xs = np.sort(d)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        axes[0].plot(xs, ys, color=colors[m], lw=1.6, drawstyle="steps-post", label=labels[m])
        se = np.mean((p - y) ** 2, axis=1)
        xs2 = np.sort(np.maximum(se, 1e-8))
        axes[1].plot(xs2, ys, color=colors[m], lw=1.6, drawstyle="steps-post")
        if m != "ridge":
            pc, _ = canonicalize_points(p)
            se2 = np.mean((pc - y) ** 2, axis=1)
            xs3 = np.sort(np.maximum(se2, 1e-8))
            axes[1].plot(xs3, ys, color=colors[m], lw=1.2, ls="--", drawstyle="steps-post")
    axes[0].set_xlabel("great-circle distance (km)")
    axes[0].set_ylabel("ECDF")
    axes[0].set_title(f"valid distance ECDF (130 points, seed {first_seed})")
    axes[0].grid(alpha=0.2, lw=0.5)
    axes[0].legend(frameon=False)
    axes[1].set_xlabel("per-sample coordinate SE (deg$^2$, log scale)")
    axes[1].set_ylabel("ECDF")
    axes[1].set_xscale("log")
    axes[1].set_title("coordinate error ECDF: solid raw, dashed branch-canon.")
    axes[1].grid(alpha=0.2, lw=0.5)
    save_fig(fig, figdir / "a10_distance_ecdf")

    # ---- valid maps ------------------------------------------------------
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    if len(methods) == 1:
        methods_plot = methods
    else:
        methods_plot = [m for m in methods if m != "ridge"]
    fig, axes = plt.subplots(1, len(methods_plot), figsize=(5.3 * len(methods_plot), 3.5),
                             subplot_kw={"projection": ccrs.PlateCarree()},
                             constrained_layout=True)
    if len(methods_plot) == 1:
        axes = [axes]
    all_d = []
    plot_cache = {}
    for m in methods_plot:
        p = method_pred(m)
        pc, flip = canonicalize_points(p)
        d = haversine_km(p[:, 0], p[:, 1], y[:, 0], y[:, 1])
        plot_cache[m] = (pc, flip, d)
        all_d.append(d)
    vmax = max(3000.0, float(np.percentile(np.concatenate(all_d), 95)))
    for ax, m in zip(axes, methods_plot):
        pc, flip, d = plot_cache[m]
        ax.set_global()
        ax.coastlines("110m", linewidth=0.45, color="0.35")
        ax.add_feature(cfeature.LAND, facecolor="0.965", zorder=0)
        ax.scatter(y[:, 1], y[:, 0], s=7, c="0.55", zorder=2,
                   transform=ccrs.PlateCarree(), label="true")
        for i in range(len(y)):
            ax.annotate("", xy=(pc[i, 1], pc[i, 0]), xytext=(y[i, 1], y[i, 0]),
                        arrowprops=dict(arrowstyle="->", color=colors[m], alpha=0.28, lw=0.6),
                        transform=ccrs.PlateCarree(), zorder=2)
        sc = ax.scatter(pc[:, 1], pc[:, 0], s=13, c=d, cmap="viridis",
                        vmin=0.0, vmax=vmax, edgecolors="none", zorder=3)
        if flip.any():
            ax.scatter(pc[flip, 1], pc[flip, 0], s=46, facecolors="none",
                       edgecolors="red", linewidths=0.9, zorder=4,
                       transform=ccrs.PlateCarree(), label="reflected branch")
        cm = coord_metrics(method_pred(m), y)
        ax.set_title(f"{labels[m]}\nmedian {cm['dist_median_km']:.0f} km, "
                     f"mean {cm['dist_mean_km']:.0f} km, refl {cm['n_reflect']}")
        ax.legend(loc="lower left", frameon=True, framealpha=0.8)
    fig.suptitle(f"world L{LAYER}, valid n={len(y)}, seed {first_seed} predictions",
                 fontsize=9.5)
    cbar = fig.colorbar(sc, ax=axes, orientation="horizontal", fraction=0.045, pad=0.06)
    cbar.set_label("great-circle error (km)")
    save_fig(fig, figdir / "a10_valid_map")


def save_fig(fig, path):
    fig.savefig(path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


if __name__ == "__main__":
    main()
