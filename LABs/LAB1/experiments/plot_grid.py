"""Plot the grouped hyperparameter sweep results as per-task figures.

Each group varies exactly one hyperparameter while the others stay pinned
(see hparam_grid.json), so every panel is a single-variable comparison.
Reads the results.json written by run_grid.py and writes combined per-task
grids plus standalone per-group figures, as PNG and PDF.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import blended_transform_factory


# 面板顺序；同一超参数在两张任务图里出现在同一个位置
TITLE_BY_VARY = {
    "lr": "Learning rate",
    "epochs": "Epochs",
    "lambda1": "L1 coefficient (λ1)",
    "lambda2": "L2 coefficient (λ2)",
    "layer": "Hidden layer",
}
LABEL_BY_VARY = {
    "lr": "learning rate",
    "epochs": "epochs",
    "lambda1": "λ1",
    "lambda2": "λ2",
    "layer": "hidden layer index",
}
SHORT_BY_KEY = {
    "lr": "lr",
    "epochs": "epochs",
    "lambda1": "λ1",
    "lambda2": "λ2",
    "layer": "layer",
}


def fmt_value(value: float) -> str:
    # 0.0005 -> "0.0005"、1e-05 -> "0.00005"、0.0 -> "0"
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return text or "0"


def load_records(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def group_records(records: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for record in records:
        groups.setdefault(record["group"], []).append(record)
    return groups


def best_finite(recs: list[dict]) -> dict | None:
    # 只用有限点选最优；'finite' 字段在发散 run 上不可靠，直接检查 metric
    finite = [rec for rec in recs if np.isfinite(rec["metric"])]
    if not finite:
        return None
    if finite[0]["metric_name"] == "mse":
        return min(finite, key=lambda rec: rec["metric"])
    return max(finite, key=lambda rec: rec["metric"])


def setup_style() -> None:
    plt.rcParams.update(
        {
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.unicode_minus": False,
            "savefig.dpi": 200,
        }
    )


def setup_xaxis(ax, recs: list[dict], vary: str) -> dict:
    """按超参数类型设置 x 轴，返回「取值 -> 绘图坐标」映射。"""
    values = sorted({rec[vary] for rec in recs})
    if vary in ("lr", "epochs"):
        ax.set_xscale("log")
        ax.set_xticks(values)
        ax.set_xticklabels([fmt_value(value) for value in values])
        ax.minorticks_off()
        ax.set_xlim(min(values) / 1.25, max(values) * 1.25)
    elif vary in ("lambda1", "lambda2"):
        # λ 含 0 且小值挤在一起，改用等距位置轴 + 真实取值标签
        positions = {value: index for index, value in enumerate(values)}
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels([fmt_value(value) for value in values])
        ax.set_xlim(-0.4, len(values) - 0.6)
        return positions
    else:
        ax.set_xticks(values)
        ax.set_xticklabels([fmt_value(value) for value in values])
        padding = 0.08 * (max(values) - min(values))
        ax.set_xlim(min(values) - padding, max(values) + padding)
    return {value: value for value in values}


def draw_panel(ax, recs: list[dict]) -> None:
    vary = recs[0]["vary"]
    metric_name = recs[0]["metric_name"]
    positions = setup_xaxis(ax, recs, vary)

    finite = sorted(
        (rec for rec in recs if np.isfinite(rec["metric"])),
        key=lambda rec: rec[vary],
    )
    xs = [positions[rec[vary]] for rec in finite]
    ys = [rec["metric"] for rec in finite]
    ax.plot(xs, ys, marker="o", markersize=5, linewidth=1.6, color="#1f77b4")

    if metric_name == "accuracy":
        ax.set_ylim(0.48, 0.92)
        ax.axhline(0.5, color="0.45", linestyle="--", linewidth=1.0)
        ax.text(
            0.02,
            0.505,
            "chance level",
            transform=blended_transform_factory(ax.transAxes, ax.transData),
            ha="left",
            va="bottom",
            fontsize=7,
            color="0.4",
        )
    else:
        # world 的 MSE 只在 150~240 之间，收紧 y 轴才能看清差异；上方留白给发散标注
        vmin, vmax = min(ys), max(ys)
        span = vmax - vmin
        if span <= 0:
            span = max(abs(vmax), 1.0) * 0.1
        ax.set_ylim(vmin - 0.08 * span, vmax + 0.30 * span)
    ax.set_ylabel(f"validation {metric_name}")

    # 发散 run（metric=inf）：不连线，在面板顶部标出
    blend = blended_transform_factory(ax.transData, ax.transAxes)
    for rec in recs:
        if np.isfinite(rec["metric"]):
            continue
        x = positions[rec[vary]]
        ax.scatter([x], [0.90], transform=blend, marker="^", s=55, color="0.5", zorder=5)
        ax.text(
            x,
            0.855,
            f"diverged\n({metric_name}=inf)",
            transform=blend,
            ha="center",
            va="top",
            fontsize=6.5,
            color="0.4",
        )

    best = best_finite(recs)
    if best is not None:
        x_best = positions[best[vary]]
        ax.scatter(
            [x_best],
            [best["metric"]],
            s=90,
            facecolors="none",
            edgecolors="red",
            linewidths=1.4,
            zorder=6,
        )
        # 面板顶部已留白，标注统一放点上方
        ax.annotate(
            f"best: {best['metric']:.4g} @ {vary}={fmt_value(best[vary])}",
            (x_best, best["metric"]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=7,
            color="red",
            zorder=6,
        )

    # 面板小字：把钉死的超参数写出来，让每块图自解释；λ1=λ2 时合并显示
    kept = [key for key in ("lr", "epochs", "lambda1", "lambda2", "layer") if key != vary]
    merge_lambdas = (
        "lambda1" in kept
        and "lambda2" in kept
        and recs[0]["lambda1"] == recs[0]["lambda2"]
    )
    pinned_parts = []
    for key in kept:
        if merge_lambdas and key == "lambda2":
            continue
        if merge_lambdas and key == "lambda1":
            pinned_parts.append(f"λ1=λ2={fmt_value(recs[0]['lambda1'])}")
        else:
            pinned_parts.append(f"{SHORT_BY_KEY[key]}={fmt_value(recs[0][key])}")
    pinned = ", ".join(pinned_parts)
    ax.set_title(TITLE_BY_VARY[vary], fontsize=10)
    ax.text(
        0.03,
        0.97,
        f"fixed: {pinned}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.8,
        color="0.35",
    )
    ax.set_xlabel(LABEL_BY_VARY[vary], fontsize=8.5)


def save_figure(fig, stem: Path) -> list[Path]:
    saved = []
    for suffix in (".png", ".pdf"):
        path = stem.with_name(stem.name + suffix)
        fig.savefig(path, bbox_inches="tight")
        saved.append(path)
    plt.close(fig)
    return saved


def figure_task(records: list[dict], task: str, out_dir: Path) -> list[Path]:
    task_records = [rec for rec in records if rec["task"] == task]
    by_vary: dict[str, list[dict]] = {}
    for rec in task_records:
        by_vary.setdefault(rec["vary"], []).append(rec)

    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    flat = axes.ravel()
    for ax, vary in zip(flat, TITLE_BY_VARY):
        if vary in by_vary:
            draw_panel(ax, by_vary[vary])
        else:
            ax.axis("off")
    for ax in flat[len(TITLE_BY_VARY):]:
        ax.axis("off")

    metric_name = task_records[0]["metric_name"]
    fig.suptitle(f"{task}: hyperparameter effects on validation {metric_name}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return save_figure(fig, out_dir / f"{task}_grid")


def figure_group(recs: list[dict], group: str, out_dir: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    draw_panel(ax, recs)
    ax.set_title(f"{recs[0]['task']}: {TITLE_BY_VARY[recs[0]['vary']]}", fontsize=10)
    fig.tight_layout()
    return save_figure(fig, out_dir / group)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_results = (
        Path(__file__).resolve().parent.parent
        / "results"
        / "slurm_grid"
        / "20261009_115559_88653"
        / "results.json"
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=default_results,
        help="results.json produced by run_grid.py",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="output directory (default: <results dir>/plots)",
    )
    args = parser.parse_args()

    records = load_records(args.results)
    out_dir = args.out_dir or args.results.parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    for task in ("truth", "world"):
        task_records = [rec for rec in records if rec["task"] == task]
        diverged = sum(1 for rec in task_records if not np.isfinite(rec["metric"]))
        print(f"{task}: {len(task_records)} runs, {diverged} diverged")

    saved: list[Path] = []
    for task in ("truth", "world"):
        saved += figure_task(records, task, out_dir)

    per_group_dir = out_dir / "per_group"
    per_group_dir.mkdir(parents=True, exist_ok=True)
    for group, recs in group_records(records).items():
        saved += figure_group(recs, group, per_group_dir)

    print(f"Loaded {len(records)} runs from {args.results}")
    print(f"Wrote {len(saved)} files:")
    for path in saved:
        print(f"  {path}")


if __name__ == "__main__":
    main()
