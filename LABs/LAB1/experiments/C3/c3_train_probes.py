# -*- coding: utf-8 -*-
"""C3：在原数据 / GeoNames 增广数据上训练 world 线性探针。

训练集（同一 13 层 std13 协议：z-score + lr 2e-3 + 2000 epochs + λ1=1e-3 + λ2=1）：
    orig            原 world train 500 条
    new             GeoNames train 8382 条
    new_noleak      new 中剔除与原始 world valid 同名同国的城市（防增广“背答案”）
    union           orig + new
    union_noleak    orig + new_noleak

评估集：
    orig_valid / new_valid / new_test / new_unseen

每 (train_set, layer) 训练纬度、经度两个 LinearRegressionModel；发散自动 lr×0.5
重试（≤4 次）。输出逐 run 记录（含四个评估集的逐条预测）到 records/，权重到
weights/，最终合并 c3_metrics.json / summary.csv。--skip_existing 可断点续跑。

锚点：orig 在 orig_valid 上的 13 层 MSE 必须复现 results/slurm_layers/world/
summary.csv 的 std13 历史值（容差 0.5），否则作业失败。
"""
import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "C1"))
from model import LinearRegressionModel  # noqa: E402
from c1_lib import coord_sq_error, haversine_km  # noqa: E402

WORLD_DIR = ROOT / "datasets" / "world"
C3_DIR = Path(__file__).resolve().parent
SPLITS_DIR = C3_DIR / "data" / "splits"
FEATURES_DIR = Path(os.environ.get("C3_FEATURES_DIR", C3_DIR / "features"))

EVAL_SPECS = {
    "orig_valid": ("world", "valid"),
    "new_valid": ("c3", "valid"),
    "new_test": ("c3", "test"),
    "new_unseen": ("c3", "test_unseen"),
}
WORKER: dict = {}


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_labels(source: str, split: str) -> np.ndarray:
    path = (WORLD_DIR if source == "world" else SPLITS_DIR) / f"{split}.csv"
    frame = pd.read_csv(path)
    return frame[["latitude", "longitude"]].to_numpy(dtype=np.float64)


def load_features(source: str, split: str, layer: int) -> np.ndarray:
    root = WORLD_DIR if source == "world" else FEATURES_DIR
    path = root / f"{split}.layer{layer}.npy"
    features = np.load(path, allow_pickle=False)
    if features.shape[1] != 2560:
        raise ValueError(f"{path}: unexpected shape {features.shape}")
    return features.astype(np.float64)


def new_train_mask() -> np.ndarray:
    frame = pd.read_csv(SPLITS_DIR / "train.csv")
    return ~frame["match_orig_valid"].astype(bool).to_numpy()


def train_data(train_set: str, layer: int):
    if train_set == "orig":
        return load_features("world", "train", layer), load_labels("world", "train")
    new_x = load_features("c3", "train", layer)
    new_y = load_labels("c3", "train")
    if train_set == "new":
        return new_x, new_y
    if train_set == "new_noleak":
        mask = new_train_mask()
        return new_x[mask], new_y[mask]
    if train_set == "union":
        return (
            np.vstack((load_features("world", "train", layer), new_x)),
            np.vstack((load_labels("world", "train"), new_y)),
        )
    if train_set == "union_noleak":
        mask = new_train_mask()
        return (
            np.vstack((load_features("world", "train", layer), new_x[mask])),
            np.vstack((load_labels("world", "train"), new_y[mask])),
        )
    raise ValueError(f"unknown train set: {train_set}")


def distance_metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    distances = haversine_km(pred[:, 0], pred[:, 1], truth[:, 0], truth[:, 1])
    return {
        "dist_median_km": float(np.median(distances)),
        "dist_mean_km": float(np.mean(distances)),
        "dist_p90_km": float(np.percentile(distances, 90)),
        "dist_max_km": float(np.max(distances)),
        "hit100_rate": float(np.mean(distances <= 100.0)),
    }


def error_metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    return {
        "mse": float(np.mean(coord_sq_error(pred[:, 0], pred[:, 1], truth[:, 0], truth[:, 1]))),
        "periodic_mse": float(
            np.mean(coord_sq_error(pred[:, 0], pred[:, 1], truth[:, 0], truth[:, 1], periodic_lon=True))
        ),
    }


def unstandardize(weights: np.ndarray, bias: np.float64, mu: np.ndarray, sd: np.ndarray):
    raw_weights = weights / sd
    raw_bias = np.float64(bias - mu @ raw_weights)
    return raw_weights, raw_bias


def fit_and_evaluate(train_set: str, layer: int, config: dict, out_dir: Path, skip_existing: bool) -> dict:
    run_id = f"{train_set}_layer{layer}"
    record_path = out_dir / "records" / f"{run_id}.json"
    weights_path = out_dir / "weights" / run_id / "weights.npz"
    if skip_existing and record_path.exists():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        record["reused"] = True
        return record

    spec = config["probe"]
    train_x, train_y = train_data(train_set, layer)
    mu = train_x.mean(axis=0)
    sd = train_x.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    train_x = (train_x - mu) / sd

    eval_x = {}
    eval_y = {}
    for name, (source, split) in EVAL_SPECS.items():
        x = load_features(source, split, layer)
        eval_x[name] = (x - mu) / sd
        eval_y[name] = load_labels(source, split)

    start = time.time()
    lr = float(spec["lr"])
    attempts = 0
    record = {
        "train_set": train_set,
        "layer": layer,
        "n_train": int(len(train_y)),
        "standardize": True,
        "epochs": spec["epochs"],
        "lambda1": spec["lambda1"],
        "lambda2": spec["lambda2"],
        "initial_lr": lr,
        "attempts": 0,
        "reused": False,
        "finite": True,
        "weights_path": str(weights_path),
    }
    while True:
        attempts += 1
        lat = LinearRegressionModel(train_x.shape[1])
        lon = LinearRegressionModel(train_x.shape[1])
        kwargs = {
            "lr": lr,
            "epochs": spec["epochs"],
            "lambda1": spec["lambda1"],
            "lambda2": spec["lambda2"],
        }
        lat.fit(train_x, train_y[:, 0], **kwargs)
        lon.fit(train_x, train_y[:, 1], **kwargs)
        preds = {
            name: np.column_stack((lat.predict(x), lon.predict(x))) for name, x in eval_x.items()
        }
        finite = bool(
            not lat.diverged
            and not lon.diverged
            and np.all(np.isfinite(lat.weights))
            and np.all(np.isfinite(lon.weights))
            and all(np.isfinite(pred).all() for pred in preds.values())
        )
        if finite or attempts >= 4:
            break
        lr = round(lr * 0.5, 12)

    record["attempts"] = attempts
    record["lr"] = lr
    record["finite"] = finite
    record["seconds"] = round(time.time() - start, 2)
    if finite:
        lat_w, lat_b = unstandardize(lat.weights, lat.bias, mu, sd)
        lon_w, lon_b = unstandardize(lon.weights, lon.bias, mu, sd)
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            weights_path,
            weights=np.stack((lat_w, lon_w)).astype(np.float64),
            bias=np.array((lat_b, lon_b), dtype=np.float64),
        )
        record["metrics"] = {}
        record["predictions"] = {}
        for name, pred in preds.items():
            metrics = error_metrics(pred, eval_y[name])
            metrics.update(distance_metrics(pred, eval_y[name]))
            record["metrics"][name] = metrics
            record["predictions"][name] = np.round(pred, 5).tolist()
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    return record


def init_worker(out_dir: str, config: dict, skip_existing: bool) -> None:
    WORKER["out_dir"] = Path(out_dir)
    WORKER["config"] = config
    WORKER["skip_existing"] = skip_existing


def worker(task: tuple[str, int]) -> dict:
    return fit_and_evaluate(
        task[0], task[1], WORKER["config"], WORKER["out_dir"], WORKER["skip_existing"]
    )


def write_outputs(metrics_path: Path, summary_path: Path, payload: dict) -> None:
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = [
        "train_set", "layer", "n_train", "lr", "initial_lr", "attempts", "reused", "epochs",
        "lambda1", "lambda2", "finite", "seconds", "weights_path",
    ]
    eval_names = list(EVAL_SPECS)
    fields += [f"{name}_mse" for name in eval_names]
    fields += [f"{name}_dist_median_km" for name in eval_names]
    with summary_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in payload["records"]:
            row = dict(record)
            for name in eval_names:
                metrics = record.get("metrics", {}).get(name, {})
                row[f"{name}_mse"] = metrics.get("mse")
                row[f"{name}_dist_median_km"] = metrics.get("dist_median_km")
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("c3_config.json"))
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--train_sets", default=None, help="逗号分隔，仅跑指定训练集")
    parser.add_argument("--layers", default=None, help="逗号分隔，仅跑指定层")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    train_sets = args.train_sets.split(",") if args.train_sets else config["train_sets"]
    layers = [int(v) for v in args.layers.split(",")] if args.layers else config["probe"]["layers"]
    unknown = [name for name in train_sets if name not in config["train_sets"]]
    if unknown:
        raise SystemExit(f"unknown train set: {unknown}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(train_set, layer) for train_set in train_sets for layer in layers]
    print(f"{len(tasks)} tasks, {args.workers} workers, out={args.out_dir}", flush=True)

    eval_truth = {name: load_labels(*spec) for name, spec in EVAL_SPECS.items()}
    records = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=init_worker,
        initargs=(str(args.out_dir), config, args.skip_existing),
    ) as executor:
        futures = {executor.submit(worker, task): task for task in tasks}
        for future in as_completed(futures):
            train_set, layer = futures[future]
            record = future.result()
            records.append(record)
            records.sort(key=lambda item: (train_sets.index(item["train_set"]), item["layer"]))
            status = "FAIL" if not record["finite"] else (
                f"orig_valid={record['metrics']['orig_valid']['mse']:.2f} "
                f"new_test={record['metrics']['new_test']['mse']:.2f} "
                f"unseen={record['metrics']['new_unseen']['mse']:.2f}"
            )
            print(
                f"{train_set}_layer{layer}: {status} lr={record['lr']:g} "
                f"attempts={record['attempts']} ({record['seconds']:.1f}s)",
                flush=True,
            )
            write_outputs(
                args.out_dir / "c3_metrics.json",
                args.out_dir / "summary.csv",
                {
                    "generated_utc": utcnow(),
                    "config": str(args.config),
                    "train_sets": train_sets,
                    "layers": layers,
                    "eval_sets": list(EVAL_SPECS),
                    "eval_truth": {name: np.round(truth, 5).tolist() for name, truth in eval_truth.items()},
                    "records": records,
                },
            )

    payload = {
        "generated_utc": utcnow(),
        "config": str(args.config),
        "train_sets": train_sets,
        "layers": layers,
        "eval_sets": list(EVAL_SPECS),
        "eval_truth": {name: np.round(truth, 5).tolist() for name, truth in eval_truth.items()},
        "records": records,
    }
    checks = []
    tolerance = float(config["anchors"]["tolerance"])
    for record in records:
        if record["train_set"] != "orig" or not record["finite"]:
            continue
        expected = config["anchors"]["orig_std13_mse"].get(str(record["layer"]))
        if expected is None:
            continue
        actual = record["metrics"]["orig_valid"]["mse"]
        checks.append(
            {
                "layer": record["layer"],
                "mse": actual,
                "expected": expected,
                "diff": actual - expected,
                "ok": abs(actual - expected) <= tolerance,
            }
        )
    payload["anchor_checks"] = checks
    write_outputs(args.out_dir / "c3_metrics.json", args.out_dir / "summary.csv", payload)

    failed_anchor = [check for check in checks if not check["ok"]]
    failed_fit = [record for record in records if not record["finite"]]
    print(f"\nanchor checks: {len(checks) - len(failed_anchor)}/{len(checks)} ok")
    for check in checks:
        print(
            f"  orig L{check['layer']}: {check['mse']:.4f} vs {check['expected']:.4f} "
            f"(diff {check['diff']:+.4f}){'' if check['ok'] else '  <-- FAIL'}"
        )
    if failed_fit:
        print(f"WARNING: {len(failed_fit)} runs did not converge finitely")
    if failed_anchor:
        raise SystemExit(f"anchor check failed for {len(failed_anchor)} layers")
    if len(records) != len(tasks):
        raise SystemExit(f"expected {len(tasks)} records, got {len(records)}")
    print(f"\nWrote {args.out_dir / 'c3_metrics.json'}")


if __name__ == "__main__":
    main()
