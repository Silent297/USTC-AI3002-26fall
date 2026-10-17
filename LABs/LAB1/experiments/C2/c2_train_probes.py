# -*- coding: utf-8 -*-
"""C2：对 prompt 变体表征逐 (variant, probe_set, layer) 训练经纬度线性探针。

训练/标准化/权重折回逻辑与 experiments/run_grid.evaluate 一致（同 model.py
LinearRegressionModel、全批量 GD、z-score 只用 train 统计量），超参见
c2_probe_config.json（std13 = world 主网格；raw11 = world 历史最优 extra run）。
坐标数学（周期经度 MSE、haversine）复用 experiments/C1/c1_lib.py，保证与
Q6/Q7/C1 同口径。

锚点：origcountry（原始 prompt）的 std13 逐层 MSE 与 raw11 必须复现
results/slurm_layers/world/summary.csv 的历史值（默认容差 0.5）——同时验证
特征提取（与 datasets/world 现有特征一致）与训练管线无错配。

输出（全部落在 --out_dir 下）：c2_metrics.json / summary.csv / weights/<run_id>/weights.npz
c2_metrics.json 内含每次 run 的 valid 逐条预测（130×2），供本地配对分析出图。
"""
import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "C1"))
from model import LinearRegressionModel  # noqa: E402
from c1_lib import coord_sq_error, haversine_km  # noqa: E402

DEFAULT_LABELS = ROOT / "datasets" / "world"


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_xy(features_root: Path, labels_dir: Path, variant: str, split: str, layer: int):
    feature_path = features_root / variant / f"{split}.layer{layer}.npy"
    features = np.load(feature_path, allow_pickle=False)
    frame = pd.read_csv(labels_dir / f"{split}.csv")
    targets = frame[["latitude", "longitude"]].to_numpy(dtype=np.float64)
    if len(features) != len(targets):
        raise ValueError(f"{feature_path}: {len(features)} rows != {len(targets)} labels")
    return features.astype(np.float64), targets


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


def predict_from_weights(weights_path: Path, valid_x: np.ndarray) -> np.ndarray:
    saved = np.load(weights_path)
    return valid_x @ saved["weights"].T + saved["bias"]


def train_probe(train_x, train_y, valid_x, spec) -> tuple[np.ndarray, np.ndarray, dict]:
    mu = sd = None
    if spec["standardize"]:
        mu = train_x.mean(axis=0)
        sd = train_x.std(axis=0)
        sd = np.where(sd == 0, 1.0, sd)
        train_x = (train_x - mu) / sd
        valid_x = (valid_x - mu) / sd
    fit_kwargs = {
        "lr": spec["lr"],
        "epochs": spec["epochs"],
        "lambda1": spec["lambda1"],
        "lambda2": spec["lambda2"],
    }
    lat = LinearRegressionModel(train_x.shape[1])
    lon = LinearRegressionModel(train_x.shape[1])
    lat.fit(train_x, train_y[:, 0], **fit_kwargs)
    lon.fit(train_x, train_y[:, 1], **fit_kwargs)
    pred = np.column_stack((lat.predict(valid_x), lon.predict(valid_x)))
    finite = bool(
        np.all(np.isfinite(lat.weights))
        and np.all(np.isfinite(lon.weights))
        and np.isfinite(lat.bias)
        and np.isfinite(lon.bias)
        and np.isfinite(pred).all()
        and not lat.diverged
        and not lon.diverged
    )
    weights, bias = lat.weights, lon.weights
    bias_pair = np.array((lat.bias, lon.bias), dtype=np.float64)
    if spec["standardize"]:
        weights, bias = unstandardize(weights, lat.bias, mu, sd)
        lon_weights, lon_bias = unstandardize(lon.weights, lon.bias, mu, sd)
        weights = np.stack((weights, lon_weights))
        bias_pair = np.array((bias, lon_bias), dtype=np.float64)
    else:
        weights = np.stack((weights, lon.weights))
    return pred, np.asarray(weights, dtype=np.float64), bias_pair, finite


def run_one(features_root, labels_dir, variant, set_name, layer, spec, weights_path, skip_existing):
    valid_x, valid_y = load_xy(features_root, labels_dir, variant, "valid", layer)
    record = {
        "variant": variant,
        "probe_set": set_name,
        "layer": layer,
        "standardize": bool(spec["standardize"]),
        "epochs": spec["epochs"],
        "lambda1": spec["lambda1"],
        "lambda2": spec["lambda2"],
        "initial_lr": spec["lr"],
        "attempts": 0,
        "reused": False,
        "finite": True,
        "weights_path": str(weights_path),
    }
    start = time.time()
    if skip_existing and weights_path.exists():
        pred = predict_from_weights(weights_path, valid_x)
        record["reused"] = True
        record["lr"] = spec["lr"]
        record["finite"] = bool(np.isfinite(pred).all())
    else:
        train_x, train_y = load_xy(features_root, labels_dir, variant, "train", layer)
        lr = float(spec["lr"])
        pred = None
        while True:
            record["attempts"] += 1
            attempt_spec = dict(spec, lr=lr)
            pred, weights, bias, finite = train_probe(train_x, train_y, valid_x, attempt_spec)
            if finite or record["attempts"] >= 4:
                break
            lr = round(lr * 0.5, 12)
        record["lr"] = lr
        record["finite"] = finite
        if finite:
            weights_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(weights_path, weights=weights, bias=bias)
        elif weights_path.exists():
            weights_path.unlink()
    record["seconds"] = round(time.time() - start, 2)
    if record["finite"]:
        record.update(error_metrics(pred, valid_y))
        record.update(distance_metrics(pred, valid_y))
        record["pred_valid"] = np.round(pred, 6).tolist()
    return record


def anchor_checks(records, anchors) -> list[dict]:
    tolerance = float(anchors["tolerance"])
    checks = []
    for record in records:
        if record["variant"] != "origcountry" or not record["finite"]:
            continue
        if record["probe_set"] == "raw11" and record["layer"] == 11:
            expected = anchors["world_raw11_mse"]
        elif record["probe_set"] == "std13":
            expected = anchors["world_std13_mse"].get(str(record["layer"]))
        else:
            continue
        if expected is None:
            continue
        diff = record["mse"] - float(expected)
        checks.append(
            {
                "probe_set": record["probe_set"],
                "layer": record["layer"],
                "mse": record["mse"],
                "expected": float(expected),
                "diff": diff,
                "ok": abs(diff) <= tolerance,
            }
        )
    return checks


def write_outputs(metrics_path: Path, summary_path: Path, payload: dict, fieldnames: list) -> None:
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with summary_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(payload["records"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("c2_probe_config.json"))
    parser.add_argument("--features_root", type=Path, default=Path(__file__).with_name("features"))
    parser.add_argument("--labels_dir", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--variants", default=None, help="逗号分隔，只跑指定变体")
    parser.add_argument("--sets", default=None, help="逗号分隔，只跑指定 probe_set")
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    variants = args.variants.split(",") if args.variants else config["variants"]
    set_names = args.sets.split(",") if args.sets else list(config["probe_sets"])
    unknown = [name for name in set_names if name not in config["probe_sets"]]
    if unknown:
        raise SystemExit(f"unknown probe set: {unknown}")

    runs = []
    for variant in variants:
        for set_name in set_names:
            spec = config["probe_sets"][set_name]
            for layer in spec["layers"]:
                runs.append((variant, set_name, layer, spec))
    if not runs:
        raise SystemExit("no runs selected")

    weights_root = args.out_dir / "weights"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.out_dir / "c2_metrics.json"
    summary_path = args.out_dir / "summary.csv"
    fieldnames = [
        "variant", "probe_set", "layer", "standardize", "lr", "initial_lr", "attempts", "reused",
        "epochs", "lambda1", "lambda2", "mse", "periodic_mse", "dist_median_km", "dist_mean_km",
        "dist_p90_km", "dist_max_km", "hit100_rate", "finite", "seconds", "weights_path",
    ]

    extract_report = {}
    report_path = args.features_root / "extract_report.json"
    if report_path.exists():
        extract_report = json.loads(report_path.read_text(encoding="utf-8"))
    valid_truth = pd.read_csv(args.labels_dir / "valid.csv")[["latitude", "longitude"]].to_numpy(np.float64)

    records = []
    for variant, set_name, layer, spec in runs:
        run_id = f"{variant}_{set_name}_layer{layer}"
        weights_path = weights_root / run_id / "weights.npz"
        record = run_one(
            args.features_root, args.labels_dir, variant, set_name, layer, spec, weights_path, args.skip_existing
        )
        records.append(record)
        payload = {
            "generated_utc": utcnow(),
            "config": str(args.config),
            "features_root": str(args.features_root),
            "labels_dir": str(args.labels_dir),
            "variants": variants,
            "probe_sets": set_names,
            "weights_root": str(weights_root),
            "extract_report": extract_report,
            "valid_truth": np.round(valid_truth, 6).tolist(),
            "records": records,
        }
        write_outputs(metrics_path, summary_path, payload, fieldnames)
        status = "reused" if record["reused"] else f"attempts={record['attempts']}"
        metric = f"mse={record['mse']:.4f} med={record['dist_median_km']:.1f}km" if record["finite"] else "FAIL"
        print(f"{run_id:<28} {metric} lr={record['lr']:g} {status} ({record['seconds']:.1f}s)", flush=True)

    checks = anchor_checks(records, config["anchors"])
    payload["anchor_checks"] = checks
    write_outputs(metrics_path, summary_path, payload, fieldnames)
    failed = [check for check in checks if not check["ok"]]
    print(f"\nanchor checks: {len(checks) - len(failed)}/{len(checks)} ok")
    for check in checks:
        print(
            f"  origcountry {check['probe_set']} L{check['layer']}: "
            f"{check['mse']:.4f} vs {check['expected']:.4f} (diff {check['diff']:+.4f})"
            f"{'' if check['ok'] else '  <-- FAIL'}"
        )
    if failed:
        raise SystemExit(f"anchor check failed for {len(failed)} runs")
    print(f"\nWrote {metrics_path}")


if __name__ == "__main__":
    main()
