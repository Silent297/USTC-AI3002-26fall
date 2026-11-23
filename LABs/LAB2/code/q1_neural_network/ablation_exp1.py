"""CPU-only forward validation of a hand-constructed two-peak MLP; no training."""

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from models import MLP


PACKAGE = Path(__file__).resolve().parent
SEED = 20261123


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference(x):
    """Independent NumPy oracle; never calls the network or its hidden features."""
    centers = np.array([[-0.65, 0.0], [0.65, 0.0]], dtype=np.float64)
    distances = np.linalg.norm(x.astype(np.float64)[:, None, :] - centers, ord=1, axis=2)
    return np.maximum(1.0 - distances / 0.6, 0.0).sum(axis=1, keepdims=True)


def construct(dtype):
    model = MLP(2, [8, 4], "relu").to(device="cpu", dtype=dtype)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        for index, center in enumerate((-0.65, 0.65)):
            start = 4 * index
            model.hidden[0].weight[start:start + 4].copy_(
                torch.tensor([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=dtype))
            model.hidden[0].bias[start:start + 4].copy_(
                torch.tensor([-center, center, 0, 0], dtype=dtype))
            model.hidden[1].weight[index, start:start + 4] = -1.0 / 0.6
            model.hidden[1].bias[index] = 1.0
            model.output.weight[0, index] = 1.0
    model.eval().requires_grad_(False)
    return model


def predict(model, x, second_relu=True):
    if second_relu:
        return model(x)
    first = model.activation(model.hidden[0](x))
    return model.output(model.hidden[1](first))


def scores(prediction, target):
    error = prediction.astype(np.float64) - target
    result = {"n": len(target), "mse": float(np.mean(error ** 2)),
              "max_abs_error": float(np.max(np.abs(error)))}
    for name, mask in (("peak", target[:, 0] >= 0.5), ("background", target[:, 0] == 0)):
        result[name + "_n"] = int(mask.sum())
        result[name + "_mse"] = float(np.mean(error[mask] ** 2)) if mask.any() else None
    return result


def datasets():
    with np.load(PACKAGE / "assets/expression.npz", allow_pickle=False) as data:
        points = {"course_train": data["train_x"].copy(), "course_grid": data["grid_x"].copy()}
        original = {"course_train": data["train_y"].copy(), "course_grid": data["grid_y"].copy()}
    axis = np.linspace(-1.5, 1.5, 201)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    points["dense_grid"] = np.column_stack([xx.ravel(), yy.ravel()])
    points["random"] = np.random.default_rng(SEED).uniform(-1.5, 1.5, (10000, 2))
    points["probes"] = np.array([
        [-0.65, 0], [0.65, 0], [0, 0], [-1.5, -1.5], [1.5, 1.5],
        [-1.25, 0], [-0.05, 0], [0.05, 0], [1.25, 0],
        [-0.65, 0.6], [0.65, -0.6], [-0.95, 0.3], [0.95, -0.3],
    ], dtype=np.float64)
    return points, original, axis


def plot_results(output, arrays, axis):
    shape = (len(axis), len(axis))
    target = arrays["float64_dense_grid_target"].reshape(shape)
    full = arrays["float64_dense_grid_full"].reshape(shape)
    ablated = arrays["float64_dense_grid_ablated"].reshape(shape)
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.5), constrained_layout=True)
    values = [target, full, np.abs(full - target), np.abs(ablated - target)]
    titles = ["Target", "Constructed MLP", "Full absolute error", "No second ReLU: absolute error"]
    for ax, value, title in zip(axes, values, titles):
        picture = ax.imshow(value, extent=(-1.5, 1.5, -1.5, 1.5), origin="lower", cmap="viridis")
        ax.set(title=title, xlabel="x1", ylabel="x2")
        fig.colorbar(picture, ax=ax, shrink=0.8)
    fig.suptitle("CPU float64, fixed hand-set weights, no training; each panel has its own color scale")
    fig.savefig(output / "maps.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for key, label, style in (("target", "Target", "k-"), ("full", "Constructed MLP", "C0--"),
                              ("ablated", "No second ReLU", "C1-")):
        ax.plot(arrays["slice_x"][:, 0], arrays["float64_slice_" + key][:, 0], style, label=label)
    ax.set(xlabel="x1 (x2 = 0)", ylabel="Output", title="Second-activation ablation: identical weights")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.savefig(output / "slice.png", dpi=160)
    plt.close(fig)


@torch.inference_mode()
def run(output):
    started = time.perf_counter()
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new, empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(SEED)  # All randomly initialized values are subsequently overwritten.
    points, original, axis = datasets()
    points["slice"] = np.column_stack([np.linspace(-1.5, 1.5, 1001), np.zeros(1001)])
    arrays = {name + "_x": value for name, value in points.items()}
    report = {
        "experiment": "constructed_mlp_second_relu_ablation", "status": "running",
        "training": False, "device": "cpu", "threads": 1, "seed": SEED,
        "widths": [8, 4], "parameters": 65, "args": sys.argv[1:],
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "numpy": np.__version__, "matplotlib": matplotlib.__version__},
        "sources": {str(p.relative_to(PACKAGE)): sha256(p) for p in (
            Path(__file__).resolve(), PACKAGE / "models.py", PACKAGE / "assets/expression.npz")},
        "metrics": {}, "probes": {}, "checks": {},
    }
    for dtype in (torch.float64, torch.float32):
        name = str(dtype).split(".")[-1]
        tolerance = 1e-12 if dtype == torch.float64 else 2e-6
        model = construct(dtype)
        assert sum(p.numel() for p in model.parameters()) == 65
        before = {key: value.clone() for key, value in model.state_dict().items()}
        torch.save(before, output / (name + "_weights.pt"))
        report["metrics"][name] = {}
        checks = {"tolerance": tolerance, "full_matches_reference": True,
                  "ablated_equals_folded_linear": True, "original_labels_match_reference": True}
        for dataset, raw in points.items():
            x = torch.as_tensor(raw, dtype=dtype, device="cpu")
            # Evaluate oracle at the *actual rounded inputs* consumed by this dtype.
            target = reference(x.numpy())
            full = predict(model, x).numpy()
            ablated = predict(model, x, second_relu=False).numpy()
            first = model.activation(model.hidden[0](x))
            weight = model.output.weight @ model.hidden[1].weight
            bias = model.output.weight @ model.hidden[1].bias + model.output.bias
            folded = torch.nn.functional.linear(first, weight, bias).numpy()
            entry = {"full": scores(full, target), "ablated": scores(ablated, target),
                     "folded_max_difference": float(np.max(np.abs(folded - ablated)))}
            checks["full_matches_reference"] &= entry["full"]["max_abs_error"] <= tolerance
            checks["ablated_equals_folded_linear"] &= entry["folded_max_difference"] <= tolerance
            checks["finite"] = checks.get("finite", True) and all(
                np.isfinite(v).all() for v in (target, full, ablated, folded))
            if dataset in original:
                entry["course_label_reference_max_difference"] = float(np.max(np.abs(original[dataset] - target)))
                entry["full_vs_course_labels"] = scores(full, original[dataset])
                checks["original_labels_match_reference"] &= entry["course_label_reference_max_difference"] <= 2e-6
            for kind, values in (("target", target), ("full", full), ("ablated", ablated)):
                arrays[f"{name}_{dataset}_{kind}"] = values
            report["metrics"][name][dataset] = entry
            if dataset == "probes":
                report["probes"][name] = [dict(x=coordinate.tolist(), target=float(t[0]),
                    full=float(f[0]), ablated=float(a[0])) for coordinate, t, f, a in zip(x.numpy(), target, full, ablated)]
        checks["weights_unchanged"] = all(torch.equal(before[key], value) for key, value in model.state_dict().items())
        report["checks"][name] = checks
    np.savez_compressed(output / "predictions.npz", **arrays)
    plot_results(output, arrays, axis)
    report["status"] = "passed" if all(value for c in report["checks"].values()
        for key, value in c.items() if key != "tolerance") else "failed"
    report["elapsed_seconds"] = time.perf_counter() - started
    report["artifacts_sha256"] = {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()}
    (output / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(output), "checks": report["checks"],
                      "elapsed_seconds": report["elapsed_seconds"]}, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(run(parser.parse_args().output))
