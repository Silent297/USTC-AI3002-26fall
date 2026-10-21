"""Plot validation metric vs hidden layer from train_layers.py results.

Merges one or more results.json files (e.g. the per-chunk outputs of
submit_layer_train.sh) and recomputes every metric from the saved weights on
the validation split. Each task gets a two-panel figure:

- top:    metric vs hidden layer (per-point error bar = 1 SE of the mean),
          'extra' runs (different config) drawn as separate reference markers;
- bottom: paired difference vs the best layer, with 95% CI — validation
          samples are shared across layers, so the paired test is the honest
          significance criterion (marginal bars overstate uncertainty).

The mean/majority-class baselines are computed from the data and printed in a
footer. PNG + PDF, same style conventions as plot_grid.py.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train23 import load_split  # noqa: E402

COLOR_STD = "#1f77b4"
COLOR_EXTRA = "#d62728"
Z95 = 1.96


def load_records(paths: list[Path]) -> list[dict]:
    merged: dict[tuple, dict] = {}
    for path in paths:
        for record in json.loads(path.read_text(encoding="utf-8")):
            key = (record["task"], record["kind"], record["layer"], record["lr"], record.get("lambda2"))
            merged[key] = record
    return list(merged.values())


REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_weights_path(raw: str | None) -> Path | None:
    """把 results.json 里的 weights_path 解析成本机路径。

    集群写入的记录是 Linux 绝对路径（/home/scc/.../LABs/LAB1/results/...），
    本机不存在：按 "LABs/LAB1/" 标记截取尾部，重新锚定到本仓库根目录。
    """
    if not raw:
        return None
    path = Path(raw)
    if path.exists():
        return path
    marker = "LABs/LAB1/"
    tail = raw.replace("\\", "/")
    index = tail.find(marker)
    if index == -1:
        return None
    candidate = REPO_ROOT / tail[index + len(marker):]
    return candidate if candidate.exists() else None


def per_element_scores(record: dict) -> np.ndarray | None:
    """每个验证样本（world 为每坐标）的误差贡献；缺权重返回 None。"""
    weights_path = resolve_weights_path(record.get("weights_path"))
    if weights_path is None:
        return None
    valid_x, valid_y = load_split(record["task"], "valid", record["layer"])
    saved = np.load(weights_path)
    weights = saved["weights"]
    bias = saved["bias"]
    if record["task"] == "world":
        prediction = valid_x @ weights.T + bias
        return ((prediction - valid_y) ** 2).ravel()
    z = valid_x @ weights + float(bias)
    prob = np.exp(-np.logaddexp(0.0, -z))
    return ((prob >= 0.5) == valid_y).astype(float)


def summarize(scores: np.ndarray) -> tuple[float, float]:
    metric = float(scores.mean())
    se = float(scores.std(ddof=1) / np.sqrt(scores.size))
    return metric, se


def config_label(record: dict) -> str:
    features = "z-scored" if record["standardize"] else "raw"
    return (
        f"{features} feats, lr={record['lr']:g}, "
        f"λ1={record['lambda1']:g}, λ2={record['lambda2']:g}"
    )


def plot_task(
    records: list[dict], task: str, out_dir: Path, include_extra: bool = True
) -> list[Path]:
    task_records = [record for record in records if record["task"] == task]
    std = sorted(
        (rec for rec in task_records if rec["kind"] == "layer" and rec["finite"]),
        key=lambda rec: rec["layer"],
    )
    extras = []
    if include_extra:
        extras = sorted(
            (rec for rec in task_records if rec["kind"] == "extra" and rec["finite"]),
            key=lambda rec: rec["layer"],
        )
    if not std and not extras:
        return []

    std_scores = [(rec, per_element_scores(rec)) for rec in std]
    std_scores = [(rec, scores) for rec, scores in std_scores if scores is not None]
    extra_scores = [(rec, per_element_scores(rec)) for rec in extras]
    extra_scores = [(rec, scores) for rec, scores in extra_scores if scores is not None]
    if not std_scores:
        return []
    metric_name = std_scores[0][0]["metric_name"]
    pick_best = min if metric_name == "mse" else max

    # 参考层 = std 里最优的一层，下方面板画各层与它的配对差
    best_rec, best_scores = pick_best(std_scores, key=lambda item: item[1].mean())
    ref_layer = best_rec["layer"]

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(8.4, 6.0), sharex=True, gridspec_kw={"height_ratios": [2.1, 1.0]}
    )

    # ---- 上：绝对指标曲线 ----
    xs, ys, ses = [], [], []
    for rec, scores in std_scores:
        metric, se = summarize(scores)
        xs.append(rec["layer"]), ys.append(metric), ses.append(se)
    top.errorbar(
        xs, ys, yerr=ses, marker="o", markersize=5, linewidth=1.6,
        elinewidth=0.9, capsize=2.5, color=COLOR_STD,
        label=f"layer runs ({config_label(best_rec)})",
    )

    for rec, scores in extra_scores:
        metric, se = summarize(scores)
        top.errorbar(
            [rec["layer"]], [metric], yerr=se, marker="*", markersize=13,
            linestyle="none", elinewidth=0.9, capsize=2.5, color=COLOR_EXTRA,
            label=f"extra (layer {rec['layer']}: {config_label(rec)})",
        )

    best_metric = float(best_scores.mean())
    top.scatter([ref_layer], [best_metric], s=110, facecolors="none",
                edgecolors=COLOR_EXTRA, linewidths=1.4, zorder=6)
    top.annotate(
        f"best: {best_metric:.4g} @ layer {ref_layer}",
        (ref_layer, best_metric), xytext=(8, 10), textcoords="offset points",
        fontsize=8, color=COLOR_EXTRA, zorder=6,
    )

    values_low = [y - se for y, se in zip(ys, ses)]
    values_high = [y + se for y, se in zip(ys, ses)]
    for _, scores in extra_scores:
        metric, se = summarize(scores)
        values_low.append(metric - se)
        values_high.append(metric + se)
    vmin, vmax = min(values_low), max(values_high)
    span = vmax - vmin or max(abs(vmax), 1.0) * 0.1
    top.set_ylim(vmin - 0.10 * span, vmax + 0.12 * span)
    top.set_ylabel(f"validation {metric_name}")
    top.set_title(f"{task}: validation {metric_name} by hidden layer", fontsize=11)
    top.grid(alpha=0.3)

    # ---- 下：相对参考层的配对差（同批验证样本，配对检验才反映真实显著性）----
    deltas, delta_cis = [], []
    for rec, scores in std_scores:
        diff = scores - best_scores
        mean_diff = float(diff.mean())
        se_diff = float(diff.std(ddof=1) / np.sqrt(diff.size))
        deltas.append(mean_diff)
        delta_cis.append(Z95 * se_diff)
    bottom.errorbar(
        xs, deltas, yerr=delta_cis, marker="o", markersize=4, linewidth=1.4,
        elinewidth=0.9, capsize=2.5, color=COLOR_STD,
    )
    bottom.axhline(0.0, color="0.45", linestyle="--", linewidth=1.0)
    bottom.annotate(
        f"reference: layer {ref_layer} (best)",
        (0.02, 0.06), xycoords="axes fraction", fontsize=7.5, color="0.35",
    )
    bottom.grid(alpha=0.3)
    if metric_name == "mse":
        bottom.set_ylabel(f"ΔMSE vs layer {ref_layer} (95% CI)", fontsize=8.5)
    else:
        bottom.set_ylabel(f"Δaccuracy vs layer {ref_layer} (95% CI)", fontsize=8.5)

    all_layers = sorted({rec["layer"] for rec, _ in std_scores} | {rec["layer"] for rec, _ in extra_scores})
    bottom.set_xticks(all_layers)
    bottom.set_xticklabels([str(layer) for layer in all_layers])
    bottom.set_xlabel("hidden layer index")
    top.legend(fontsize=7.5, loc="best")

    # ---- 页脚：基线数值（从数据实时计算）----
    if task == "world":
        _, valid_y = load_split("world", "valid", ref_layer)
        baseline = float(((valid_y - valid_y.mean(axis=0)) ** 2).mean())
        r2 = 1.0 - best_metric / baseline
        footer = (f"mean-baseline MSE = {baseline:.0f}; best R² = {r2:.4f}  ·  "
                  "top bars: ±1 SE of the mean; bottom bars: 95% CI of the paired difference")
    else:
        _, valid_y = load_split("truth", "valid", ref_layer)
        majority = max(float(valid_y.mean()), 1 - float(valid_y.mean()))
        footer = (f"majority-class accuracy = {majority:.3f}  ·  "
                  "top bars: ±1 SE of the mean; bottom bars: 95% CI of the paired difference")
    fig.text(0.01, 0.005, footer, fontsize=7, color="0.35")

    fig.tight_layout(rect=(0, 0.025, 1, 1))
    saved = []
    for suffix in (".png", ".pdf"):
        path = out_dir / f"{task}_layers{suffix}"
        fig.savefig(path, bbox_inches="tight")
        saved.append(path)
    plt.close(fig)

    print(f"=== {task} ({metric_name}); reference = layer {ref_layer} ===")
    for (rec, scores), delta, ci in zip(std_scores, deltas, delta_cis):
        metric, se = summarize(scores)
        reused_note = " (reused)" if rec.get("reused") else ""
        if metric_name == "mse":
            delta_text = f"{delta:+7.1f} ± {ci:4.1f}"
        else:
            delta_text = f"{delta * 100:+6.2f}pp ± {ci * 100:4.2f}pp"
        print(
            f"  layer {rec['layer']:<3} {metric_name}={metric:.6g} ± {se:.3g}"
            f"   Δ={delta_text}{reused_note}"
        )
    for rec, scores in extra_scores:
        metric, se = summarize(scores)
        print(f"  layer {rec['layer']:<3} [extra] {metric_name}={metric:.6g} ± {se:.3g}  ({config_label(rec)})")
    print(f"  ({footer.split('  ·  ')[0]})")
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        nargs="+",
        required=True,
        help="one or more results.json from train_layers.py (per-chunk files are merged)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="output directory (default: <first results dir>/plots)",
    )
    parser.add_argument("--tasks", default=None, help="comma-separated task filter")
    parser.add_argument(
        "--no-extra",
        action="store_true",
        help="drop 'extra' runs and show the std layer comparison only",
    )
    args = parser.parse_args()

    plt.rcParams.update(
        {
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.unicode_minus": False,
            "savefig.dpi": 200,
        }
    )

    records = load_records(args.results)
    tasks = args.tasks.split(",") if args.tasks else ["world", "truth"]

    out_dir = args.out_dir or args.results[0].parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    for task in tasks:
        saved += plot_task(records, task, out_dir, include_extra=not args.no_extra)

    print(f"\nWrote {len(saved)} files:")
    for path in saved:
        print(f"  {path}")


if __name__ == "__main__":
    main()
