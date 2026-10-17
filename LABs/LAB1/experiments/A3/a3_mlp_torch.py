# -*- coding: utf-8 -*-
"""A3: self-trained MLP (torch, CPU) vs the linear probe ("is the linear readout enough?").

Protocol (train/valid fixed; the valid split is used ONLY for reporting):
- features: z-scored single layer with TRAIN statistics; world uses L11
  (submission layer) and L2 (best single layer under the std protocol);
  truth uses L20 (peak layer).
- every run: an internal 10% split of TRAIN is held out for early stopping only
  (state at best internal-val loss is restored); AdamW(lr=1e-3, wd=1e-4),
  batch 128 (world) / 256 (truth), patience 30, max 400/200 epochs; 3 seeds.
- architectures: linear control (same modern recipe, isolates "nonlinearity"
  from "training procedure"), MLP 64, MLP 128-64 (ReLU).
- in-protocol linear anchors (from A1/A2 at fixed budget): world z L11 MSE
  167.17, world z L2 157.62; truth z L20 acc 0.8667. Historical anchors: world
  raw L11 150.43 (submission probe), truth raw GD L20 0.8967. Guide (student
  reference report, non-official, valid-selection protocol): world MLP-128-64
  152.70, truth RBF-SVC 87.00%.

Outputs: experiments/A3/figures/a3_model_bars.{png,pdf},
experiments/A3/figures/a3_curves.{png,pdf}, experiments/A3/a3_results.json
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
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"

SEED_LIST = (0, 1, 2)
ARCHS = ("linear", "mlp64", "mlp128_64")
WORLD_LAYERS = (11, 2)
TRUTH_LAYER = 20
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH = {"world": 128, "truth": 256}
MAX_EPOCHS = {"world": 400, "truth": 200}
PATIENCE = 30
INTERNAL_VAL_FRAC = 0.1

ANCHORS = {
    "world": {"linear_in_protocol_l11": 167.17, "linear_in_protocol_l2": 157.62,
              "submission_raw_l11": 150.43, "guide_mlp128_64": 152.70},
    "truth": {"linear_in_protocol_l20": 0.8667, "gd_raw_l20": 0.8967,
              "guide_rbf_svc": 0.8700},
}
GUIDE_NOTE = "student reference report (non-official); their protocol selects on valid - shown for scale only"

COLORS = {"linear": "0.45", "mlp64": "tab:blue", "mlp128_64": "tab:orange"}


def load_layer(task, split, layer):
    return np.load(ROOT / f"datasets/{task}/{split}.layer{layer}.npy").astype(np.float32)


def load_target(task, split):
    df = pd.read_csv(ROOT / f"datasets/{task}/{split}.csv")
    if task == "world":
        return df[["latitude", "longitude"]].to_numpy(np.float32)
    return df["label"].to_numpy(np.float32)


def zscore(train_X, X):
    mu = train_X.mean(axis=0)
    sd = train_X.std(axis=0)
    sd = np.where(sd == 0.0, 1.0, sd)
    return (X - mu) / sd


def build_model(arch, d_in, d_out):
    if arch == "linear":
        return nn.Linear(d_in, d_out)
    if arch == "mlp64":
        return nn.Sequential(nn.Linear(d_in, 64), nn.ReLU(), nn.Linear(64, d_out))
    if arch == "mlp128_64":
        return nn.Sequential(nn.Linear(d_in, 128), nn.ReLU(),
                             nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, d_out))
    raise ValueError(arch)


def count_params(model):
    return int(sum(p.numel() for p in model.parameters()))


def train_run(task, Xtr, ytr, Xva, yva, arch, seed):
    torch.manual_seed(seed)
    n = len(ytr)
    rng = np.random.RandomState(seed)
    mask_val = rng.rand(n) < INTERNAL_VAL_FRAC
    mask_trn = ~mask_val

    Xt = torch.from_numpy(Xtr[mask_trn])
    Xv = torch.from_numpy(Xtr[mask_val])
    Xa = torch.from_numpy(Xva)
    Xall = torch.from_numpy(Xtr)
    if task == "world":
        yt, yv, ya, yall = (torch.from_numpy(a) for a in
                            (ytr[mask_trn], ytr[mask_val], yva, ytr))
    else:
        yt = torch.from_numpy(ytr[mask_trn]).reshape(-1, 1)
        yv = torch.from_numpy(ytr[mask_val]).reshape(-1, 1)
        ya = torch.from_numpy(yva)
        yall = torch.from_numpy(ytr)

    d_out = 2 if task == "world" else 1
    model = build_model(arch, Xtr.shape[1], d_out)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss() if task == "world" else nn.BCEWithLogitsLoss()
    gen = torch.Generator().manual_seed(seed)
    nt = int(mask_trn.sum())

    curve = []
    best_loss, best_state, best_epoch, bad = float("inf"), None, 0, 0
    for epoch in range(1, MAX_EPOCHS[task] + 1):
        model.train()
        perm = torch.randperm(nt, generator=gen)
        for start in range(0, nt, BATCH[task]):
            idx = perm[start:start + BATCH[task]]
            opt.zero_grad()
            loss = criterion(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            train_loss = criterion(model(Xt), yt).item()
            val_loss = criterion(model(Xv), yv).item()
        curve.append([epoch, train_loss, val_loss])
        if val_loss < best_loss - 1e-9:
            best_loss, best_epoch, bad = val_loss, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_va = model(Xa)
        pred_tr = model(Xall)
        if task == "world":
            valid = float(((pred_va - ya) ** 2).mean().item())
            train = float(((pred_tr - yall) ** 2).mean().item())
        else:
            valid = float(((pred_va.flatten() > 0).to(torch.float32) == ya).to(torch.float32).mean().item())
            train = float(((pred_tr.flatten() > 0).to(torch.float32) == yall).to(torch.float32).mean().item())
    record = {"arch": arch, "seed": seed, "n_params": count_params(model),
              "best_epoch": best_epoch, "epochs_run": len(curve),
              "best_internal_val_loss": best_loss,
              "train_score": train, "valid_score": valid}
    return record, curve


def run_task(task, layer):
    Xtr_raw = load_layer(task, "train", layer)
    Xva_raw = load_layer(task, "valid", layer)
    Xtr = zscore(Xtr_raw, Xtr_raw)
    Xva = zscore(Xtr_raw, Xva_raw)
    ytr = load_target(task, "train")
    yva = load_target(task, "valid")
    metric = "valid MSE" if task == "world" else "valid acc"
    print(f"[{task} L{layer}] {len(ARCHS)} archs x {len(SEED_LIST)} seeds; metric={metric}", flush=True)

    runs, curves = [], {}
    for arch in ARCHS:
        for seed in SEED_LIST:
            t0 = time.time()
            record, curve = train_run(task, Xtr, ytr, Xva, yva, arch, seed)
            record["wall_s"] = round(time.time() - t0, 1)
            runs.append(record)
            curves[(arch, seed)] = curve
            print(f"[{task} L{layer}] {arch:10s} seed={seed} best_ep={record['best_epoch']:3d} "
                  f"ran={record['epochs_run']:3d} train={record['train_score']:.4f} "
                  f"valid={record['valid_score']:.4f} ({record['wall_s']:.1f}s)", flush=True)

    summary = {}
    for arch in ARCHS:
        vals = np.array([r["valid_score"] for r in runs if r["arch"] == arch])
        summary[arch] = {"mean": float(vals.mean()), "std": float(vals.std()), "n": int(len(vals))}
    return runs, curves, summary


def fig_bars(world_l11, world_l2, truth_l20):
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.4))
    panels = [
        (axes[0], world_l11, "world L11 (submission layer)", "valid MSE (deg$^2$)"),
        (axes[1], world_l2, "world L2 (best std single)", "valid MSE (deg$^2$)"),
        (axes[2], truth_l20, "truth L20 (peak layer)", "valid accuracy"),
    ]
    for ax, summary, title, xlabel in panels:
        archs = list(ARCHS)
        means = [summary[a]["mean"] for a in archs]
        stds = [summary[a]["std"] for a in archs]
        ypos = np.arange(len(archs))
        ax.barh(ypos, means, xerr=stds, capsize=3, alpha=0.9,
                color=[COLORS[a] for a in archs])
        span = max(means) - min(0.0, min(means))
        for pos, a in zip(ypos, archs):
            ax.text(summary[a]["mean"] + summary[a]["std"] + 0.015 * span, pos,
                    f"{summary[a]['mean']:.4f} $\\pm$ {summary[a]['std']:.4f}",
                    va="center", fontsize=7.5, color="0.25")
        ax.set_yticks(ypos, archs, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_title(title, fontsize=10)
        if "world" in title:
            ref = ANCHORS["world"]["linear_in_protocol_l11"] if "L11" in title \
                else ANCHORS["world"]["linear_in_protocol_l2"]
            ax.axvline(ref, color="0.25", linestyle="--", linewidth=1.2,
                       label=f"linear ridge (A1) {ref:.1f}")
            ax.axvline(ANCHORS["world"]["submission_raw_l11"], color="tab:red", linestyle=":",
                       linewidth=1.4, label=f"submission raw L11 {ANCHORS['world']['submission_raw_l11']}")
            ax.axvline(ANCHORS["world"]["guide_mlp128_64"], color="tab:green", linestyle="-.",
                       linewidth=1.0, label=f"guide MLP-128-64 {ANCHORS['world']['guide_mlp128_64']}*")
        else:
            ax.axvline(ANCHORS["truth"]["linear_in_protocol_l20"], color="0.25", linestyle="--",
                       linewidth=1.2, label=f"linear LR (A1) {ANCHORS['truth']['linear_in_protocol_l20']:.4f}")
            ax.axvline(ANCHORS["truth"]["gd_raw_l20"], color="tab:red", linestyle=":", linewidth=1.4,
                       label=f"GD raw L20 {ANCHORS['truth']['gd_raw_l20']:.4f}")
            ax.axvline(ANCHORS["truth"]["guide_rbf_svc"], color="tab:green", linestyle="-.",
                       linewidth=1.0, label=f"guide RBF-SVC {ANCHORS['truth']['guide_rbf_svc']:.2f}*")
        ax.legend(fontsize=7.5, loc="lower right")
    fig.suptitle(f"bars = mean $\\pm$ std over seeds {list(SEED_LIST)}; * {GUIDE_NOTE}", fontsize=8.5, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIG / "a3_model_bars.png", dpi=150)
    fig.savefig(FIG / "a3_model_bars.pdf")
    plt.close(fig)


def fig_curves(world_l11, world_l2, truth_l20):
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 4.6))
    panels = [
        (axes[0], world_l11, "world L11: internal-val MSE", "internal-val MSE"),
        (axes[1], world_l2, "world L2: internal-val MSE", "internal-val MSE"),
        (axes[2], truth_l20, "truth L20: internal-val logloss", "internal-val logloss"),
    ]
    for ax, curves, title, ylabel in panels:
        seen = set()
        for (arch, seed), curve in sorted(curves.items()):
            ep = [row[0] for row in curve]
            vl = [max(row[2], 1e-12) for row in curve]
            ax.plot(ep, vl, color=COLORS[arch], alpha=0.6, linewidth=1.0,
                    label=arch if arch not in seen else None)
            seen.add(arch)
            i = int(np.argmin(vl))
            ax.scatter([ep[i]], [vl[i]], color=COLORS[arch], s=22, marker="x", zorder=5)
        ax.set_yscale("log")
        ax.set_xlabel("epoch", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25, linewidth=0.4)
    fig.suptitle("thin lines = seeds 0-2; x = best early-stop epoch "
                 "(internal 10% train-val split, never the report valid set)", fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG / "a3_curves.png", dpi=150)
    fig.savefig(FIG / "a3_curves.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=("world", "truth"), default=("world", "truth"))
    args = parser.parse_args()

    FIG.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    payload = {"seed_list": list(SEED_LIST), "archs": list(ARCHS),
               "config": {"lr": LR, "weight_decay": WEIGHT_DECAY, "batch": BATCH,
                          "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
                          "internal_val_frac": INTERNAL_VAL_FRAC},
               "anchors": ANCHORS, "guide_note": GUIDE_NOTE,
               "env": {"python": platform.python_version(), "numpy": np.__version__,
                       "torch": torch.__version__, "device": "cpu"}}
    wl11 = wl2 = tr20 = None

    if "world" in args.tasks:
        runs, curves, summary = run_task("world", WORLD_LAYERS[0])
        payload["world_l11"] = {"runs": runs, "summary": summary}
        wl11 = curves
        runs, curves, summary = run_task("world", WORLD_LAYERS[1])
        payload["world_l2"] = {"runs": runs, "summary": summary}
        wl2 = curves
    if "truth" in args.tasks:
        runs, curves, summary = run_task("truth", TRUTH_LAYER)
        payload["truth_l20"] = {"runs": runs, "summary": summary}
        tr20 = curves

    if wl11 and wl2 and tr20:
        fig_bars(payload["world_l11"]["summary"], payload["world_l2"]["summary"],
                 payload["truth_l20"]["summary"])
        fig_curves(wl11, wl2, tr20)
        print("[figs] wrote a3_model_bars + a3_curves", flush=True)

    payload["elapsed_s"] = round(time.time() - t0, 1)
    with (HERE / "a3_results.json").open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=1)
    print(f"[done] {time.time() - t0:.1f}s; wrote {HERE / 'a3_results.json'}", flush=True)


if __name__ == "__main__":
    main()
