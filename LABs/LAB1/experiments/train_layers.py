"""按超参数列表逐 (task, layer) 训练探针并保存权重。

读取配置（默认 experiments/train_layers.json）：对 tasks 里每个任务的每个层各训练一次，
可选 extra_runs 追加特殊配置的运行。某次训练发散或数值非有限时，自动把 lr 乘
lr_fallback.factor 重试（最多 max_attempts 次），记录最终 lr 与尝试次数。

输出（全部落在 --out-dir 下）：
- results.json    每次运行的完整记录（每完成一个 run 立即重写，作业被超时杀掉也不丢已完成部分）
- summary.csv     扁平表格，便于画层数曲线（同上，实时重写）
- weights/<run>/weights.npz  权重（已折回原始特征空间，可直接提交）

加 --skip-existing 时，已存在 weights.npz 的 run 只复算验证指标、不重训，用于超时后的
断点续跑（假定同一输出目录里的权重来自同一配置）。

训练/评估逻辑复用 run_grid.evaluate（train split 统计量做 z-score，无验证集泄漏）。
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from run_grid import evaluate  # noqa: E402
from train23 import load_split  # noqa: E402


def build_runs(config: dict, tasks_filter: set | None, layers_filter: set | None) -> list:
    runs = []
    for task, block in config["tasks"].items():
        if tasks_filter is not None and task not in tasks_filter:
            continue
        overrides = block.get("overrides", {})
        for layer in block["layers"]:
            if layers_filter is not None and layer not in layers_filter:
                continue
            params = {
                "lr": block["lr"],
                "epochs": block["epochs"],
                "lambda1": block["lambda1"],
                "lambda2": block["lambda2"],
            }
            params.update(overrides.get(str(layer), {}))
            runs.append(
                {
                    "task": task,
                    "layer": layer,
                    "standardize": bool(block.get("standardize", False)),
                    "params": params,
                    "note": "",
                    "kind": "layer",
                    "run_id": f"{task}_layer{layer}",
                }
            )
    for index, extra in enumerate(config.get("extra_runs", [])):
        task = extra["task"]
        layer = extra["layer"]
        if tasks_filter is not None and task not in tasks_filter:
            continue
        if layers_filter is not None and layer not in layers_filter:
            continue
        runs.append(
            {
                "task": task,
                "layer": layer,
                "standardize": bool(extra.get("standardize", False)),
                "params": {key: extra[key] for key in ("lr", "epochs", "lambda1", "lambda2")},
                "note": extra.get("note", ""),
                "kind": "extra",
                "run_id": f"{task}_layer{layer}_extra{index}",
            }
        )
    return runs


def evaluate_saved(task: str, layer: int, weights_path: Path) -> dict:
    """从已保存的权重（原始特征空间）复算验证指标，不重新训练。"""
    valid_x, valid_y = load_split(task, "valid", layer)
    saved = np.load(weights_path)
    weights = saved["weights"]
    bias = saved["bias"]
    if task == "world":
        prediction = valid_x @ weights.T + bias
        metric = float(np.mean((prediction - valid_y) ** 2))
        metric_name = "mse"
    else:
        z = valid_x @ weights + float(bias)
        prob = np.exp(-np.logaddexp(0.0, -z))
        metric = float(np.mean((prob >= 0.5) == valid_y))
        metric_name = "accuracy"
    finite = bool(
        np.all(np.isfinite(weights)) and np.all(np.isfinite(bias)) and np.isfinite(metric)
    )
    return {"metric_name": metric_name, "metric": metric, "finite": finite}


def write_outputs(results_path: Path, summary_path: Path, records: list, fieldnames: list) -> None:
    results_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    with summary_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("train_layers.json"),
        help="超参数列表（默认：与本脚本同目录的 train_layers.json）",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="结果目录：results.json / summary.csv / weights/ 都写到这里",
    )
    parser.add_argument("--tasks", default=None, help="逗号分隔，只跑指定任务（如 world,truth）")
    parser.add_argument("--layers", default=None, help="逗号分隔，只跑指定层（如 11,23）")
    parser.add_argument("--dry-run", action="store_true", help="只列出将要运行的条目，不训练")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="weights/<run_id>/weights.npz 已存在则复算指标直接复用（断点续跑用）",
    )
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    fallback = config.get("lr_fallback", {})
    factor = float(fallback.get("factor", 0.5))
    max_attempts = int(fallback.get("max_attempts", 4))

    tasks_filter = set(args.tasks.split(",")) if args.tasks else None
    layers_filter = {int(value) for value in args.layers.split(",")} if args.layers else None

    runs = build_runs(config, tasks_filter, layers_filter)
    if not runs:
        print("ERROR: 没有匹配的任务/层，什么都没跑", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        print(f"共 {len(runs)} 次训练：")
        for run in runs:
            p = run["params"]
            tail = f"  # {run['note']}" if run["note"] else ""
            print(
                f"  {run['run_id']:<24} task={run['task']:<6} layer={run['layer']:<3} "
                f"std={run['standardize']} lr={p['lr']:g} ep={p['epochs']} "
                f"l1={p['lambda1']:g} l2={p['lambda2']:g}{tail}"
            )
        return

    weights_root = args.out_dir / "weights"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.out_dir / "results.json"
    summary_path = args.out_dir / "summary.csv"
    fieldnames = [
        "task",
        "layer",
        "kind",
        "standardize",
        "lr",
        "initial_lr",
        "attempts",
        "reused",
        "epochs",
        "lambda1",
        "lambda2",
        "metric_name",
        "metric",
        "finite",
        "seconds",
        "weights_path",
        "note",
    ]

    records = []
    failed = 0
    for run in runs:
        task = run["task"]
        layer = run["layer"]
        params = run["params"]
        standardize = run["standardize"]
        weights_path = weights_root / run["run_id"] / "weights.npz"

        label = run["run_id"]
        initial_lr = params["lr"]
        attempts = 0
        elapsed = 0.0
        reused = False
        outcome = None
        if args.skip_existing and weights_path.exists():
            try:
                outcome = evaluate_saved(task, layer, weights_path)
            except Exception as exc:  # 半截/损坏的 npz：删掉重训
                print(f"  {label}: 读取既有 weights.npz 失败（{exc}），改为重新训练")
                outcome = None
            if outcome is not None and not outcome["finite"]:
                print(f"  {label}: 既有权重非有限，删除后重新训练")
                outcome = None
            if outcome is None:
                weights_path.unlink(missing_ok=True)
            else:
                reused = True

        while not reused:
            attempts += 1
            t0 = time.time()
            outcome = evaluate(task, layer, params, weights_path=weights_path, standardize=standardize)
            elapsed += time.time() - t0
            if outcome["finite"] or attempts >= max_attempts:
                break
            print(f"  {label}: 发散/非有限（lr={params['lr']:g}），lr×{factor:g} 后重试")
            params["lr"] = round(params["lr"] * factor, 12)

        record = {
            "task": task,
            "layer": layer,
            "kind": run["kind"],
            "note": run["note"],
            "standardize": standardize,
            "lr": params["lr"],
            "initial_lr": initial_lr,
            "attempts": attempts,
            "reused": reused,
            "epochs": params["epochs"],
            "lambda1": params["lambda1"],
            "lambda2": params["lambda2"],
            **outcome,
            "seconds": round(elapsed, 2),
            "weights_path": str(weights_path),
        }
        records.append(record)
        if not outcome["finite"]:
            # 失败的运行留下的是爆掉的权重，删掉避免被误当有效产物
            if weights_path.exists():
                weights_path.unlink()
            record["weights_path"] = ""
            failed += 1
        # 每完成一个 run 立即落盘，作业被超时杀掉时已完成记录不丢
        write_outputs(results_path, summary_path, records, fieldnames)
        detail = "reused" if reused else f"attempts={attempts}"
        print(
            f"{label:<26} {outcome['metric_name']}={outcome['metric']:.4f} "
            f"finite={outcome['finite']} lr={params['lr']:g} {detail} "
            f"({elapsed:.1f}s)"
        )

    print()
    print("=== 汇总（按层升序）===")
    for task in ("world", "truth"):
        rows = [r for r in records if r["task"] == task and r["kind"] == "layer"]
        rows.sort(key=lambda r: r["layer"])
        for r in rows:
            flag = "" if r["finite"] else "  <-- FAIL"
            detail = "reused" if r.get("reused") else f"attempts={r['attempts']}"
            print(
                f"{task:<6} layer {r['layer']:<3} {r['metric_name']}={r['metric']:.4f} "
                f"lr={r['lr']:g} {detail}{flag}"
            )
    for r in records:
        if r["kind"] == "extra":
            flag = "" if r["finite"] else "  <-- FAIL"
            detail = "reused" if r.get("reused") else f"attempts={r['attempts']}"
            print(
                f"extra  {r['task']}/layer{r['layer']} {r['metric_name']}={r['metric']:.4f} "
                f"lr={r['lr']:g} {detail} ({r['note']}){flag}"
            )

    print()
    print(f"results.json: {results_path}")
    print(f"summary.csv:  {summary_path}")
    print(f"weights:      {weights_root}")

    if failed:
        print(f"ERROR: {failed} 个运行在 {max_attempts} 次尝试后仍非有限", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
