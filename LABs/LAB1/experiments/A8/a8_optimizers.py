# -*- coding: utf-8 -*-
"""A8: optimizer comparison on the same linear/logistic probe objective.

Questions
- Under the same init/objective/steps, how fast are GD, heavy-ball momentum and Adam?
- Where is each optimizer's stability edge, in units of the GD critical lr?
- Do they reach the same final train/valid metrics, or does the optimizer matter?

Protocol
- Tasks: world L11 (linear, lam2=1.0) and truth L20 (logistic, lam2=1e-3).
- Features z-scored with train statistics only; lam1=0 (A6 covers L1).
- Objective exactly as model.py (world: two single-target fits == 1/N ||Y-XW-b||_F^2;
  MSE in tables stays element-mean so it matches the A1 anchor).
- Full-batch torch float64. Head-to-head uses the model.py w=b=0 init; the
  stability sweep uses a tiny random init (sigma=1e-3, seed 1234) so every
  eigendirection is excited and the divergence edge is meaningful.
- lr_crit = 2/L, L = Lipschitz bound of the smooth objective:
    linear  : 2*lmax([X,1])/N + 2*lam2
    logistic: 0.25*lmax([X,1])/N + 2*lam2
  lr is reported as f = lr/lr_crit.
- Reference optimum: closed-form dual ridge (world), L-BFGS (truth).
- Valid is reported only, never used for selection.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
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
}
ANCHORS = {
    "world_L11": {"valid_mse": 167.17072274253437, "median_km": 923.1205912354363},
    "truth_L20": {"valid_acc": 0.8666666666666667, "valid_logloss": 0.4834606151866668},
}
OPTIMIZERS = ("gd", "momentum", "adam")
COLORS = {"gd": "#2166ac", "momentum": "#d95f02", "adam": "#1b9e77"}
LABELS = {"gd": "GD", "momentum": "Momentum 0.9", "adam": "Adam"}
FGRID = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 1.9, 2.2, 3.0]
HEAD_F = 0.9
SWEEP_STEPS = 600
HEAD_STEPS = 8000
DIV_FACTOR = 1e6


def load_task(name):
    cfg = TASKS[name]
    out = {}
    for split in ("train", "valid"):
        X = np.load(ROOT / f"datasets/{cfg['task']}/{split}.layer{cfg['layer']}.npy").astype(np.float64)
        df = pd.read_csv(ROOT / f"datasets/{cfg['task']}/{split}.csv")
        if cfg["kind"] == "linear":
            y = df[["latitude", "longitude"]].to_numpy(np.float64)
        else:
            y = df["label"].to_numpy(np.float64)[:, None]
        out[split] = {"X": X, "y": y}
    mu = out["train"]["X"].mean(axis=0)
    sd = out["train"]["X"].std(axis=0)
    sd = np.where(sd == 0.0, 1.0, sd)
    for split in out:
        out[split]["X"] = (out[split]["X"] - mu) / sd
    return out


def augmented_lmax(X, iters=80, seed=0):
    rng = np.random.default_rng(seed)
    Xa = np.hstack([X, np.ones((len(X), 1))])
    v = rng.normal(size=Xa.shape[1])
    v /= np.linalg.norm(v)
    n = 1.0
    for _ in range(iters):
        v = Xa.T @ (Xa @ v)
        n = float(np.linalg.norm(v))
        v /= n
    return n


def smooth_loss_np(X, y, w, b, lam2, kind):
    z = X @ w + b
    if kind == "linear":
        # model.py fits lat/lon as two single-target models with 1/N scaling;
        # vectorised this is (1/N)||Y - XW - b||_F^2, i.e. sum over outputs.
        return float(np.mean(np.sum((z - y) ** 2, axis=1)) + lam2 * np.sum(w * w))
    return float(np.mean(np.logaddexp(0.0, z) - y * z) + lam2 * np.sum(w * w))


def ridge_reference(X, y, lam2):
    n = len(y)
    ybar = y.mean(axis=0)
    alpha = np.linalg.solve(X @ X.T + lam2 * n * np.eye(n), y - ybar)
    w = X.T @ alpha
    b = ybar
    f = smooth_loss_np(X, y, w, b, lam2, "linear")
    return w, b, f


def logistic_reference(X, y, lam2):
    from scipy.optimize import minimize
    n, d = X.shape
    yv = y[:, 0]

    def fun(theta):
        w = theta[:d]
        b = theta[d]
        z = X @ w + b
        loss = np.mean(np.logaddexp(0.0, z) - yv * z) + lam2 * (w @ w)
        p = 0.5 * (1.0 + np.tanh(0.5 * z))
        g = np.empty(d + 1)
        g[:d] = X.T @ (p - yv) / n + 2.0 * lam2 * w
        g[d] = np.mean(p - yv)
        return loss, g

    res = minimize(fun, np.zeros(d + 1), jac=True, method="L-BFGS-B",
                   options={"maxiter": 3000, "gtol": 1e-10, "ftol": 1e-15, "maxls": 50})
    w = res.x[:d].copy()
    b = np.array([res.x[d]])
    return w, b, float(res.fun), {
        "success": bool(res.success), "nit": int(res.nit),
        "grad_inf": float(np.max(np.abs(fun(res.x)[1]))),
    }


def metrics_np(X, y, w, b, kind):
    z = X @ w + b
    if kind == "linear":
        mse = float(np.mean((z - y) ** 2))
        lat1 = np.radians(z[:, 0]); lon1 = np.radians(z[:, 1])
        lat2 = np.radians(y[:, 0]); lon2 = np.radians(y[:, 1])
        a = (np.sin((lat2 - lat1) / 2.0) ** 2
             + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2.0) ** 2)
        d = 2.0 * R_EARTH * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
        return {"mse": mse, "median_km": float(np.median(d)), "mean_km": float(np.mean(d))}
    z1 = z[:, 0] if z.ndim == 2 else z
    p = 0.5 * (1.0 + np.tanh(0.5 * z1))
    yv = y[:, 0]
    acc = float(np.mean((p >= 0.5) == yv))
    ll = float(-np.mean(yv * np.log(np.clip(p, 1e-12, 1.0))
                       + (1.0 - yv) * np.log(np.clip(1.0 - p, 1e-12, 1.0))))
    return {"acc": acc, "logloss": ll}


def torch_loss(Xt, yt, w, b, kind, lam2):
    z = Xt @ w + b
    if kind == "linear":
        return torch.mean(torch.sum((z - yt) ** 2, dim=1)) + lam2 * torch.sum(w * w)
    return torch.mean(torch.logaddexp(torch.zeros_like(z), z) - yt * z) + lam2 * torch.sum(w * w)


def make_optimizer(name, params, lr):
    if name == "gd":
        return torch.optim.SGD(params, lr=lr)
    if name == "momentum":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-8)
    raise ValueError(name)


def run_optimizer(X, y, kind, lam2, name, lr, steps, record_every=10,
                  early_stop=True, f0=None, init_scale=0.0):
    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(y)
    out_dim = y.shape[1]
    if init_scale > 0:
        gen = torch.Generator().manual_seed(1234)
        w0 = init_scale * torch.randn((X.shape[1], out_dim), generator=gen, dtype=torch.float64)
        b0 = init_scale * torch.randn(out_dim, generator=gen, dtype=torch.float64)
    else:
        w0 = torch.zeros((X.shape[1], out_dim), dtype=torch.float64)
        b0 = torch.zeros(out_dim, dtype=torch.float64)
    w = w0.clone().requires_grad_(True)
    b = b0.clone().requires_grad_(True)
    opt = make_optimizer(name, [w, b], lr)
    loss0 = smooth_loss_np(X, y, w.detach().numpy(), b.detach().numpy(), lam2, kind)
    if f0 is None:
        f0 = loss0
    limit = DIV_FACTOR * max(abs(f0), 1.0)
    history = [(0, float(loss0))]
    diverged = False
    div_step = None
    t0 = time.time()
    recent = []
    stopped = False
    for step in range(1, steps + 1):
        opt.zero_grad()
        loss = torch_loss(Xt, yt, w, b, kind, lam2)
        lv = float(loss.detach())
        if not np.isfinite(lv) or lv > limit:
            diverged = True
            div_step = step
            break
        loss.backward()
        if not (torch.isfinite(w.grad).all() and torch.isfinite(b.grad).all()):
            diverged = True
            div_step = step
            break
        opt.step()
        recent.append(lv)
        if len(recent) > 200:
            recent.pop(0)
        if early_stop and step > 500 and len(recent) == 200:
            ref = recent[0]
            if abs(lv - ref) <= 1e-12 * max(abs(ref), 1.0):
                stopped = True
        if step % record_every == 0 or step == 1:
            history.append((step, float(lv)))
        if stopped:
            break
    elapsed = time.time() - t0
    w_np = w.detach().numpy().copy()
    b_np = b.detach().numpy().copy()
    final_loss = smooth_loss_np(X, y, w_np, b_np, lam2, kind) if not diverged else float("inf")
    return {
        "opt": name, "lr": float(lr), "steps_run": step,
        "diverged": bool(diverged), "div_step": div_step,
        "early_stopped": bool(stopped),
        "final_loss": float(final_loss),
        "history": history if record_every < steps else [(0, float(loss0)), (step, float(final_loss))],
        "elapsed_s": float(elapsed),
        "w": w_np, "b": b_np,
    }


def normalized_gap(loss, f_ref, f0):
    return float(max(loss - f_ref, 0.0) / max(f0 - f_ref, 1e-12))


def steps_to_target(history, f_ref, f0, target):
    for step, loss in history:
        if np.isfinite(loss) and normalized_gap(loss, f_ref, f0) <= target:
            return int(step)
    return None


def run_task(name, head_f, sweep_steps, head_steps):
    cfg = TASKS[name]
    data = load_task(name)
    X, y = data["train"]["X"], data["train"]["y"]
    Xv, yv = data["valid"]["X"], data["valid"]["y"]
    lmax = augmented_lmax(X)
    if cfg["kind"] == "linear":
        L = 2.0 * lmax / len(y) + 2.0 * cfg["lam2"]
    else:
        L = 0.25 * lmax / len(y) + 2.0 * cfg["lam2"]
    lr_crit = 2.0 / L

    t0 = time.time()
    if cfg["kind"] == "linear":
        w_ref, b_ref, f_ref = ridge_reference(X, y, cfg["lam2"])
        ref_info = {"method": "dual ridge (closed form)"}
    else:
        w_ref, b_ref, f_ref, ref_info = logistic_reference(X, y, cfg["lam2"])
        ref_info["method"] = "L-BFGS-B"
    ref_info["train"] = metrics_np(X, y, w_ref, b_ref, cfg["kind"])
    ref_info["valid"] = metrics_np(Xv, yv, w_ref, b_ref, cfg["kind"])
    ref_info["loss"] = float(f_ref)
    ref_info["elapsed_s"] = float(time.time() - t0)
    print(f"[{name}] lmax_aug={lmax:.6g} L={L:.6g} lr_crit={lr_crit:.6g} "
          f"f*={f_ref:.6g} ({ref_info['method']}, {ref_info['elapsed_s']:.1f}s)")
    if cfg["kind"] == "logistic":
        print(f"[{name}] lbfgs success={ref_info['success']} nit={ref_info['nit']} "
              f"grad_inf={ref_info['grad_inf']:.2e}")

    f0 = smooth_loss_np(X, y, np.zeros((X.shape[1], y.shape[1])), np.zeros(y.shape[1]),
                        cfg["lam2"], cfg["kind"])

    # ---- stability sweep (tiny random init excites all eigendirections) ----
    sweep = []
    for opt_name in OPTIMIZERS:
        for f in FGRID:
            lr = f * lr_crit
            res = run_optimizer(X, y, cfg["kind"], cfg["lam2"], opt_name, lr,
                                sweep_steps, record_every=sweep_steps,
                                early_stop=False, f0=f0, init_scale=1e-3)
            gap = normalized_gap(res["final_loss"], f_ref, f0) if not res["diverged"] else None
            sweep.append({
                "opt": opt_name, "f": float(f), "lr": float(lr),
                "final_loss": float(res["final_loss"]),
                "normalized_gap": gap,
                "diverged": bool(res["diverged"]), "div_step": res["div_step"],
            })
            print(f"[{name}] sweep {opt_name:8s} f={f:4.1f} "
                  f"{'DIVERGED@'+str(res['div_step']) if res['diverged'] else 'gap=%.3e' % gap}")

    # ---- does the init change the observed edge? (world/truth probe at f=1.5) ----
    init_probe = {}
    for scale in (0.0, 1e-3):
        res = run_optimizer(X, y, cfg["kind"], cfg["lam2"], "gd", 1.5 * lr_crit,
                            sweep_steps, record_every=sweep_steps, early_stop=False,
                            f0=f0, init_scale=scale)
        init_probe[f"gd_f1.5_init{scale:g}"] = {
            "init_scale": float(scale), "diverged": bool(res["diverged"]),
            "div_step": res["div_step"],
            "normalized_gap": None if res["diverged"] else normalized_gap(res["final_loss"], f_ref, f0),
        }
        print(f"[{name}] probe gd f=1.5 init={scale:g} "
              f"{'DIVERGED@'+str(res['div_step']) if res['diverged'] else 'gap=%.3e' % init_probe[f'gd_f1.5_init{scale:g}']['normalized_gap']}")

    # ---- head to head at the same nominal lr, model.py zero init ----
    head = {}
    for opt_name in OPTIMIZERS:
        lr = head_f * lr_crit
        res = run_optimizer(X, y, cfg["kind"], cfg["lam2"], opt_name, lr,
                            head_steps, record_every=10, early_stop=True, f0=f0,
                            init_scale=0.0)
        gap = normalized_gap(res["final_loss"], f_ref, f0) if not res["diverged"] else None
        history = [(s, normalized_gap(l, f_ref, f0)) for s, l in res["history"]
                   if np.isfinite(l)]
        head[opt_name] = {
            "lr": float(lr), "f": float(head_f), "steps_run": res["steps_run"],
            "diverged": bool(res["diverged"]), "div_step": res["div_step"],
            "early_stopped": bool(res["early_stopped"]),
            "final_loss": float(res["final_loss"]), "normalized_gap": gap,
            "steps_to_1e-2": steps_to_target(res["history"], f_ref, f0, 1e-2),
            "steps_to_1e-3": steps_to_target(res["history"], f_ref, f0, 1e-3),
            "steps_to_1e-4": steps_to_target(res["history"], f_ref, f0, 1e-4),
            "train": metrics_np(X, y, res["w"], res["b"], cfg["kind"]),
            "valid": metrics_np(Xv, yv, res["w"], res["b"], cfg["kind"]),
            "elapsed_s": res["elapsed_s"],
            "history": history,
            "w_l2": float(np.sqrt(np.sum(res["w"] ** 2))),
        }
        print(f"[{name}] head {opt_name:8s} f={head_f} steps={res['steps_run']} "
              f"{'DIVERGED@'+str(res['div_step']) if res['diverged'] else 'gap=%.3e' % gap} "
              f"elapsed={res['elapsed_s']:.1f}s")
        del res

    return {
        "config": cfg, "lmax_aug": float(lmax), "L": float(L), "lr_crit": float(lr_crit),
        "f0": float(f0), "reference": ref_info, "sweep": sweep,
        "init_probe": init_probe, "head": head, "anchor": ANCHORS[name],
    }


def make_figures(results):
    # ---- fig 1: convergence ----
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))
    for ax, (name, res) in zip(axes, results.items()):
        for opt_name in OPTIMIZERS:
            h = res["head"][opt_name]["history"]
            if not h:
                continue
            steps = np.array([p[0] for p in h])
            gaps = np.maximum(np.array([p[1] for p in h]), 1e-14)
            ax.plot(steps, gaps, color=COLORS[opt_name], lw=0.6, alpha=0.22)
            best = np.minimum.accumulate(gaps)
            ax.plot(steps, best, color=COLORS[opt_name], lw=1.6, label=LABELS[opt_name])
            if res["head"][opt_name]["diverged"]:
                d = res["head"][opt_name]["div_step"]
                ax.scatter([d], [1.0], marker="x", s=40, color=COLORS[opt_name], zorder=5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e-14, 2.0)
        ax.set_xlabel("step (log)")
        ax.set_ylabel(r"best-so-far $(f-f^*)/(f_0-f^*)$")
        ax.set_title(f"{name.replace('_', ' ')}: convergence at f = {HEAD_F:g}")
        ax.grid(alpha=0.25, which="both")
        ax.axhline(1e-2, color="#999999", ls=":", lw=0.8)
        ax.axhline(1e-4, color="#bbbbbb", ls=":", lw=0.8)
        ax.legend(fontsize=7.5, loc="lower left")
    fig.suptitle("A8: best-so-far (thick) and instantaneous (thin) loss gap at lr = %.1f x GD critical" % HEAD_F,
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"a8_convergence.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # ---- fig 2: stability ----
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))
    for ax, (name, res) in zip(axes, results.items()):
        for opt_name in OPTIMIZERS:
            rows = [r for r in res["sweep"] if r["opt"] == opt_name]
            x, y = [], []
            for r in rows:
                if r["diverged"]:
                    x.append(r["f"])
                    y.append(10.0)
                else:
                    x.append(r["f"])
                    y.append(max(r["normalized_gap"], 1e-14))
            ax.plot(x, y, "-o", ms=3.5, color=COLORS[opt_name], label=LABELS[opt_name])
            xd = [r["f"] for r in rows if r["diverged"]]
            if xd:
                ax.scatter(xd, [12.0] * len(xd), marker="x", s=42,
                           color=COLORS[opt_name], zorder=5)
        ax.set_yscale("log")
        ax.set_ylim(1e-6, 40)
        ax.set_xlabel(r"$f = \mathrm{lr}/\mathrm{lr}_{crit}$")
        ax.set_ylabel(r"gap after %d steps (diverged = x)" % SWEEP_STEPS)
        ax.set_title(name.replace("_", " "))
        ax.axvline(1.0, color="#555555", ls="--", lw=0.8)
        ax.axvline(1.9, color="#999999", ls=":", lw=0.9)
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=7.5)
    axes[0].text(1.01, 20, "GD edge", fontsize=6.5, color="#555555")
    axes[0].text(1.92, 20, "mom. edge", fontsize=6.5, color="#999999")
    axes[1].text(1.01, 6.0, "f=1: conservative bound,\nnot the observed edge", fontsize=6.5,
                 color="#555555", va="top")
    fig.suptitle("A8: stability region vs normalised learning rate", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"a8_stability.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # ---- fig 3: final metrics ----
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.2))
    specs = [
        (axes[0, 0], "world_L11", "valid", "mse", "world: valid MSE (deg$^2$)", False),
        (axes[0, 1], "world_L11", "valid", "median_km", "world: valid median distance (km)", False),
        (axes[1, 0], "truth_L20", "valid", "acc", "truth: valid accuracy", True),
        (axes[1, 1], "truth_L20", "valid", "logloss", "truth: valid log-loss", False),
    ]
    xpos = np.arange(3)
    for ax, task, split, key, title, as_pct in specs:
        vals = [results[task]["head"][o][split][key] for o in OPTIMIZERS]
        for x, v, o in zip(xpos, vals, OPTIMIZERS):
            ax.plot([x], [v], "o", ms=8, color=COLORS[o], zorder=3)
            txt = f"{100*v:.2f}%" if as_pct else f"{v:.5g}"
            ax.annotate(txt, xy=(x, v), xytext=(0, 7), textcoords="offset points",
                        fontsize=7, ha="center")
        ref = results[task]["reference"][split][key]
        ax.axhline(ref, color="#555555", ls="--", lw=0.9,
                   label="reference %.5g" % ref)
        lo = min(vals + [ref])
        hi = max(vals + [ref])
        pad = max((hi - lo) * 0.8, abs(hi) * 2e-3, 1e-6)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xticks(xpos)
        ax.set_xticklabels([LABELS[o] for o in OPTIMIZERS], fontsize=8)
        ax.set_xlim(-0.5, 2.5)
        ax.set_title(title, fontsize=9.5)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=6.5, loc="lower right")
    fig.suptitle("A8: final valid metrics at f = %.1f x GD critical (dots) vs reference optimum (dashed)" % HEAD_F,
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"a8_final_metrics.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--replot", action="store_true")
    ap.add_argument("--head-f", type=float, default=HEAD_F)
    ap.add_argument("--sweep-steps", type=int, default=SWEEP_STEPS)
    ap.add_argument("--head-steps", type=int, default=HEAD_STEPS)
    args = ap.parse_args()
    torch.set_num_threads(4)
    if args.replot:
        with open(HERE / "a8_results.json", encoding="utf-8") as f:
            results = json.load(f)
        make_figures(results)
        print("[replot] wrote figures from a8_results.json")
        return
    if args.smoke:
        global FGRID
        FGRID = [0.1, 0.9, 1.5, 3.0]
        args.sweep_steps = min(args.sweep_steps, 100)
        args.head_steps = min(args.head_steps, 300)
    t0 = time.time()
    results = {}
    for name in TASKS:
        results[name] = run_task(name, args.head_f, args.sweep_steps, args.head_steps)
    results["protocol"] = {
        "optimizers": list(OPTIMIZERS), "f_grid": FGRID, "head_f": args.head_f,
        "sweep_steps": args.sweep_steps, "head_steps": args.head_steps,
        "lr_crit": "2/L; L = Lipschitz bound of the smooth objective with [X,1]",
        "sweep_init": "w,b ~ N(0,1e-3^2), seed 1234 (excites all eigendirections)",
        "head_init": "w=0, b=0 for all optimizers (model.py init)",
        "dtype": "torch float64, full batch", "threads": 4,
        "lam1": 0.0, "valid_usage": "reporting only",
        "reference": "dual ridge (world), L-BFGS-B (truth)",
    }
    with open(HERE / "a8_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    make_figures(results)
    print(f"[done] {time.time()-t0:.1f}s; wrote {HERE/'a8_results.json'}")


if __name__ == "__main__":
    main()
