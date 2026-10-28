"""Train the coordinate or truth probe and save its parameters."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from model import LinearRegressionModel, LogisticRegressionModel


def load_split(task: str, split: str, layer: int) -> tuple[np.ndarray, np.ndarray]:

    data_dir = Path(__file__).parent / "datasets" / task

    features = np.load(data_dir / f"{split}.layer{layer}.npy", allow_pickle=False)
    labels = pd.read_csv(data_dir / f"{split}.csv")
    if task == "world":
        targets = labels[["latitude", "longitude"]].to_numpy(dtype=np.float64)
        # 形状为 (N, 2)
    else:
        targets = labels["label"].to_numpy(dtype=np.float64)
        # 形状为 (N,)
    if len(features) != len(targets):
        raise ValueError("The feature and label files have different row counts")

    return features.astype(np.float64), targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("world", "truth"))
    parser.add_argument("layer", type=int)
    parser.add_argument("output", type=Path, help="Path to the output .npz file")
    # 训练超参数；不传则采用这里的默认值
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate (default 0.001)")
    parser.add_argument("--epochs", type=int, default=10000, help="number of epochs (default 10000)")
    parser.add_argument("--lambda1", type=float, default=0.01, help="L1 regularization (default 0.01)")
    parser.add_argument("--lambda2", type=float, default=0.01, help="L2 regularization (default 0.01)")
    args = parser.parse_args()

    train_x, train_y = load_split(args.task, "train", args.layer)
    valid_x, valid_y = load_split(args.task, "valid", args.layer)

    fit_kwargs = {
        "lr": args.lr,
        "epochs": args.epochs,
        "lambda1": args.lambda1,
        "lambda2": args.lambda2,
    }
    print(
        f"Hyperparameters: lr={args.lr}, epochs={args.epochs}, "
        f"lambda1={args.lambda1}, lambda2={args.lambda2}"
    )

    if args.task == "world":
        latitude_targets = train_y[:, 0]
        longitude_targets = train_y[:, 1]

        latitude_model = LinearRegressionModel(train_x.shape[1])
        longitude_model = LinearRegressionModel(train_x.shape[1])
        latitude_model.fit(train_x, latitude_targets, **fit_kwargs)
        longitude_model.fit(train_x, longitude_targets, **fit_kwargs)

        latitude_predictions = latitude_model.predict(valid_x)
        longitude_predictions = longitude_model.predict(valid_x)
        prediction = np.column_stack((latitude_predictions, longitude_predictions))
        weights = np.stack((latitude_model.weights, longitude_model.weights))
        bias = np.array((latitude_model.bias, longitude_model.bias))
    else:
        model = LogisticRegressionModel(train_x.shape[1])
        model.fit(train_x, train_y, **fit_kwargs)
        prediction = model.predict(valid_x)
        weights = model.weights
        bias = model.bias

    if prediction.shape != valid_y.shape:
        raise ValueError(f"Expected predictions with shape {valid_y.shape}")

    if args.task == "world":
        print(f"Validation MSE: {np.mean((prediction - valid_y) ** 2):.4f}")
    else:
        print(f"Validation accuracy: {np.mean((prediction >= 0.5) == valid_y):.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        args.output,
        weights=np.asarray(weights, dtype=np.float64),
        bias=np.asarray(bias, dtype=np.float64),
    )
    print(f"Saved {args.output} (layer {args.layer})")


if __name__ == "__main__":
    main()
