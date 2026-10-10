"""按超参数列表逐 (task, layer) 训练探针并保存权重。

读取配置（默认 experiments/train_layers.json）：对 tasks 里每个任务的每个层各训练一次，
可选 extra_runs 追加特殊配置的运行。某次训练发散或数值非有限时，自动把 lr 乘
lr_fallback.factor 重试（最多 max_attempts 次），记录最终 lr 与尝试次数。

输出（全部落在 --out-dir 下）：
- results.json    每次运行的完整记录
- summary.csv     扁平表格，便于画层数曲线
- weights/<run>/weights.npz  权重（已折回原始特征空间，可直接提交）

训练/评估逻辑复用 run_grid.evaluate（train split 统计量做 z-score，无验证集泄漏）。
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from run_grid import evaluate  # noqa: E402


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
        outcome = None
        while True:
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
        print(
            f"{label:<26} {outcome['metric_name']}={outcome['metric']:.4f} "
            f"finite={outcome['finite']} lr={params['lr']:g} attempts={attempts} "
            f"({elapsed:.1f}s)"
        )

    results_path = args.out_dir / "results.json"
    results_path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    summary_path = args.out_dir / "summary.csv"
    fieldnames = [
        "task",
        "layer",
        "kind",
        "standardize",
        "lr",
        "initial_lr",
        "attempts",
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
    with summary_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print()
    print("=== 汇总（按层升序）===")
    for task in ("world", "truth"):
        rows = [r for r in records if r["task"] == task and r["kind"] == "layer"]
        rows.sort(key=lambda r: r["layer"])
        for r in rows:
            flag = "" if r["finite"] else "  <-- FAIL"
            print(
                f"{task:<6} layer {r['layer']:<3} {r['metric_name']}={r['metric']:.4f} "
                f"lr={r['lr']:g} attempts={r['attempts']}{flag}"
            )
    for r in records:
        if r["kind"] == "extra":
            flag = "" if r["finite"] else "  <-- FAIL"
            print(
                f"extra  {r['task']}/layer{r['layer']} {r['metric_name']}={r['metric']:.4f} "
                f"lr={r['lr']:g} attempts={r['attempts']} ({r['note']}){flag}"
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
