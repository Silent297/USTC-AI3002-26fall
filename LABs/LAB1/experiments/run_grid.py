"""Run a grouped hyperparameter grid from a JSON config.

Each group varies exactly one axis while the others stay pinned, so every
comparison inside a group is single-variable. Results are printed and written
to a JSON file for later analysis.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import LinearRegressionModel, LogisticRegressionModel
from train23 import load_split


def safe_component(value: object) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in "._=-" else "-" for ch in text)


def evaluate(task: str, layer: int, params: dict, weights_path: Path | None = None) -> dict:
    train_x, train_y = load_split(task, "train", layer)
    valid_x, valid_y = load_split(task, "valid", layer)
    fit_kwargs = {
        "lr": params["lr"],
        "epochs": params["epochs"],
        "lambda1": params["lambda1"],
        "lambda2": params["lambda2"],
    }

    if task == "world":
        lat = LinearRegressionModel(train_x.shape[1])
        lon = LinearRegressionModel(train_x.shape[1])
        lat.fit(train_x, train_y[:, 0], **fit_kwargs)
        lon.fit(train_x, train_y[:, 1], **fit_kwargs)
        pred = np.column_stack((lat.predict(valid_x), lon.predict(valid_x)))
        metric = float(np.mean((pred - valid_y) ** 2))
        finite = bool(
            np.all(np.isfinite(lat.weights))
            and np.all(np.isfinite(lon.weights))
            and np.isfinite(lat.bias)
            and np.isfinite(lon.bias)
        )
        metric_name = "mse"
        if weights_path is not None:
            weights_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                weights_path,
                weights=np.asarray(np.stack((lat.weights, lon.weights)), dtype=np.float64),
                bias=np.asarray(np.array((lat.bias, lon.bias)), dtype=np.float64),
            )
    else:
        model = LogisticRegressionModel(train_x.shape[1])
        model.fit(train_x, train_y, **fit_kwargs)
        prob = model.predict(valid_x)
        metric = float(np.mean((prob >= 0.5) == valid_y))
        finite = bool(np.all(np.isfinite(model.weights)) and np.isfinite(model.bias))
        metric_name = "accuracy"
        if weights_path is not None:
            weights_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                weights_path,
                weights=np.asarray(model.weights, dtype=np.float64),
                bias=np.asarray(model.bias, dtype=np.float64),
            )

    return {"metric_name": metric_name, "metric": metric, "finite": finite}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None,
                        help="Where to write results JSON (default: alongside config)")
    parser.add_argument("--groups", default=None,
                        help="Comma-separated group names to run (default: all)")
    parser.add_argument("--artifacts-dir", type=Path, default=None,
                        help="Optionally save per-run weights.npz files in this directory")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    base = config["base"]
    wanted = set(args.groups.split(",")) if args.groups else None

    results = []
    for group in config["groups"]:
        if wanted is not None and group["name"] not in wanted:
            continue
        print(f"\n=== group: {group['name']} ({group['task']}, layer={group['layer']}) ===")
        vary_key = next(iter(group["vary"]))
        vary_values = group["vary"][vary_key]
        for value in vary_values:
            params = dict(base)
            params.update(group["fixed"])
            params[vary_key] = value
            layer = value if vary_key == "layer" else group["layer"]

            weights_path = None
            if args.artifacts_dir is not None:
                run_id = safe_component(f"{group['name']}__{vary_key}={value}")
                weights_path = args.artifacts_dir / run_id / "weights.npz"

            t0 = time.time()
            outcome = evaluate(group["task"], layer, params, weights_path=weights_path)
            elapsed = time.time() - t0
            record = {
                "group": group["name"],
                "task": group["task"],
                "layer": layer,
                "vary": vary_key,
                vary_key: value,
                **{k: params[k] for k in ("lr", "epochs", "lambda1", "lambda2")},
                **outcome,
                "seconds": round(elapsed, 2),
            }
            if weights_path is not None:
                record["weights_path"] = str(weights_path)
            results.append(record)
            name = outcome["metric_name"]
            print(
                f"  {vary_key}={value!r:<8} {name}={outcome['metric']:.4f} "
                f"finite={outcome['finite']} ({elapsed:.1f}s)"
            )

    out_path = args.out or (args.config.parent / "results.json")
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {len(results)} runs to {out_path}")


if __name__ == "__main__":
    main()
