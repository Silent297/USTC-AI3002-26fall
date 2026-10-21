"""Q9 figure: truth L11 hyperparameter sweeps from the first grid (job 88653)."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_grid as pg

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results/slurm_grid/20261009_115559_88653/results.json"
OUT_DIR = ROOT / "results/q9_grid/plots"


def main() -> None:
    records = [
        rec
        for rec in json.loads(RESULTS.read_text(encoding="utf-8"))
        if rec.get("task") == "truth" and rec.get("vary") != "layer"
    ]

    by_vary = {}
    for rec in records:
        by_vary.setdefault(rec["vary"], []).append(rec)

    pg.setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.3), squeeze=False)
    ax_lr, ax_epochs = axes[0]
    ax_l1, ax_l2 = axes[1]

    pg.draw_panel(ax_lr, by_vary["lr"])
    pg.draw_panel(ax_epochs, by_vary["epochs"])
    pg.draw_panel(ax_l1, by_vary["lambda1"], mark_best=False)
    pg.draw_panel(ax_l2, by_vary["lambda2"], mark_best=False)

    for ax, vary, note in (
        (ax_l1, "lambda1", "green diamond: kept λ1=0.001\n(weak regularization)"),
        (ax_l2, "lambda2", "green diamond: kept λ2=0.001\n(flat across all values)"),
    ):
        recs = by_vary[vary]
        values = sorted({rec[vary] for rec in recs})
        x_keep = values.index(0.001)
        y_keep = next(rec["metric"] for rec in recs if rec[vary] == 0.001)
        ax.scatter(
            [x_keep],
            [y_keep],
            marker="D",
            s=70,
            facecolors="none",
            edgecolors="green",
            linewidths=1.4,
            zorder=7,
        )
        ax.text(
            0.97,
            0.72,
            note,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            color="green",
        )

    fig.suptitle("truth L11 hyperparameter sweeps (job 88653)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for path in pg.save_figure(fig, OUT_DIR / "truth_sweep"):
        print(path)


if __name__ == "__main__":
    main()
