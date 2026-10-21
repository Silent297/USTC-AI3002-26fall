"""Plot the raw-L2 coarse lr x lambda2 sweep (world, layer 2)."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def fmt(value: float) -> str:
    return f"{value:g}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/l2_sweep/raw_coarse/results.json"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/l2_sweep/raw_coarse"),
    )
    args = parser.parse_args()

    records = json.loads(args.results.read_text(encoding="utf-8"))
    grid_recs = [rec for rec in records if rec["vary"] == "lambda2"]
    div_recs = [rec for rec in records if rec["vary"] == "lr"]

    lrs = sorted({rec["lr"] for rec in grid_recs})
    lams = sorted({rec["lambda2"] for rec in grid_recs})
    table = np.full((len(lrs), len(lams)), np.nan)
    for rec in grid_recs:
        table[lrs.index(rec["lr"]), lams.index(rec["lambda2"])] = rec["metric"]

    best = min((rec for rec in grid_recs if rec["finite"]), key=lambda r: r["metric"])
    print("lr x lambda2 valid MSE")
    print("lr\\lam " + " ".join(f"{fmt(lam):>10}" for lam in lams))
    for i, lr in enumerate(lrs):
        print(f"{fmt(lr):>7} " + " ".join(f"{table[i, j]:10.3f}" for j in range(len(lams))))
    print(f"best: lr={best['lr']:g}, lambda2={best['lambda2']:g}, MSE={best['metric']:.4f}")
    for rec in div_recs:
        print(f"divergence probe: lr={rec['lr']:g}, finite={rec['finite']}, MSE={rec['metric']:.3e}")

    out_dir = args.out_dir
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    ax = axes[0]
    im = ax.imshow(table, cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(lams)))
    ax.set_xticklabels([fmt(lam) for lam in lams])
    ax.set_yticks(range(len(lrs)))
    ax.set_yticklabels([fmt(lr) for lr in lrs])
    ax.set_xlabel(r"$\lambda_2$")
    ax.set_ylabel("learning rate")
    ax.set_title("raw L2 coarse sweep: valid MSE")
    for i in range(len(lrs)):
        for j in range(len(lams)):
            value = table[i, j]
            ax.text(j, i, f"{value:.1f}", ha="center", va="center",
                    color="white" if value > np.nanmedian(table) else "black", fontsize=8.5)
    bi, bj = lrs.index(best["lr"]), lams.index(best["lambda2"])
    ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False,
                               edgecolor="red", linewidth=2.0))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="valid MSE")

    ax = axes[1]
    for lam in lams:
        ys = [table[lrs.index(lr), lams.index(lam)] for lr in lrs]
        ax.plot(lrs, ys, marker="o", linewidth=1.6, label=rf"$\lambda_2$={fmt(lam)}")
    ax.set_xscale("log")
    ax.set_xticks(lrs)
    ax.set_xticklabels([fmt(lr) for lr in lrs])
    ax.minorticks_off()
    ax.set_xlabel("learning rate")
    ax.set_ylabel("valid MSE")
    ax.set_title("raw L2: MSE vs learning rate")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, frameon=False)
    div = div_recs[0]
    ax.text(0.02, 0.06, f"lr={div['lr']:g}: diverged (finite=False)",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=8, color="0.35")

    fig.tight_layout()
    for suffix in (".png", ".pdf"):
        fig.savefig(plot_dir / f"l2_raw_coarse{suffix}", bbox_inches="tight", dpi=200)
    plt.close(fig)

    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["lr", "lambda2", "lambda1", "epochs", "mse", "finite", "seconds"])
        for rec in grid_recs + div_recs:
            writer.writerow([rec["lr"], rec["lambda2"], rec["lambda1"], rec["epochs"],
                             rec["metric"], rec["finite"], rec["seconds"]])
    print(f"wrote {plot_dir / 'l2_raw_coarse.png'}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
