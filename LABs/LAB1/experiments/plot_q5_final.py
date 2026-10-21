"""Q5 figure: raw L11 sweeps behind the submitted weights (lr, epochs, lambda1, lambda2)."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_grid as pg

ROOT = Path(__file__).resolve().parent.parent
RAW_RESULTS = ROOT / "results/slurm_grid/20261009_115559_88653/results.json"
LAMBDA1_RESULTS = ROOT / "results/q5_grid/raw_lambda1_results.json"
OUT_DIR = ROOT / "results/q5_grid/plots"

KEEP_RAW_GROUPS = {"world_lr", "world_epochs", "world_lambda2"}


def main() -> None:
    raw = json.loads(RAW_RESULTS.read_text(encoding="utf-8"))
    lambda1 = json.loads(LAMBDA1_RESULTS.read_text(encoding="utf-8"))

    records = []
    for rec in raw:
        if rec.get("task") == "world" and rec.get("group") in KEEP_RAW_GROUPS:
            rec = dict(rec)
            rec["data_tag"] = "raw L11 (final)"
            records.append(rec)
    for rec in lambda1:
        if rec.get("task") == "world":
            rec = dict(rec)
            rec["data_tag"] = "raw L11 (final)"
            records.append(rec)

    by_vary = {}
    for rec in records:
        by_vary.setdefault(rec["vary"], []).append(rec)

    pg.setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.3), squeeze=False)

    ax_lr, ax_epochs = axes[0]
    ax_l1, ax_l2 = axes[1]

    pg.draw_panel(ax_lr, by_vary["lr"])
    ax_lr.text(
        0.97,
        0.72,
        "lr=0.001: diverged (Q3)",
        transform=ax_lr.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        color="0.4",
    )

    pg.draw_panel(ax_epochs, by_vary["epochs"])
    pg.draw_panel(ax_l1, by_vary["lambda1"], mark_best=False)
    ax_l1.set_title("L1 coefficient (λ1)", fontsize=10)
    l1_values = sorted({rec["lambda1"] for rec in by_vary["lambda1"]})
    x_keep = l1_values.index(0.001)
    y_keep = next(rec["metric"] for rec in by_vary["lambda1"] if rec["lambda1"] == 0.001)
    ax_l1.scatter(
        [x_keep],
        [y_keep],
        marker="D",
        s=70,
        facecolors="none",
        edgecolors="green",
        linewidths=1.4,
        zorder=7,
    )
    ax_l1.text(
        0.5,
        0.45,
        "green diamond: kept λ1=0.001\nλ1=0 gives 150.419 (−0.008)",
        transform=ax_l1.transAxes,
        ha="center",
        va="center",
        fontsize=7,
        color="green",
    )

    pg.draw_panel(ax_l2, by_vary["lambda2"])

    fig.suptitle("world L11 final config: raw sweeps", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for path in pg.save_figure(fig, OUT_DIR / "world_grid"):
        print(path)


if __name__ == "__main__":
    main()
