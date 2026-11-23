"""Paired-budget experiment: CUDA formal runs, CPU synthetic smoke only."""

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

from models import MLP

PACKAGE = Path(__file__).resolve().parent
DEFAULT_CONFIG = PACKAGE / "ablation_exp2_config.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_config(config):
    # Keep the registered design fixed; smoke uses a separate explicit configuration.
    if (config["structures"] != {"shallow": [16], "deep": [8, 4]}
            or config["seeds"] != [17, 42, 123, 2026, 20261123]
            or config["checkpoints"] != [0, 3000, 9000]
            or config["activation"] != "relu" or config["optimizer"] != "adam"
            or config["learning_rate"] != 0.01 or config["dtype"] != "float32"
            or config["threads"] != 1 or config["history_every"] != 100):
        raise ValueError("Configuration differs from the registered experiment-2 design")
    return config


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def measure(model, data):
    train_error = model(data["train_x"]) - data["train_y"]
    prediction = model(data["grid_x"])
    error = prediction.double() - data["grid_y"].double()
    if not torch.isfinite(error).all() or not torch.isfinite(train_error).all():
        raise FloatingPointError("Non-finite evaluation")
    row = {"train_mse": float(train_error.double().square().mean()),
           "grid_mse": float(error.square().mean()), "max_abs_error": float(error.abs().max())}
    for name, mask in (("peak", data["grid_y"][:, 0] >= 0.5),
                       ("background", data["grid_y"][:, 0] == 0)):
        row[name + "_n"] = int(mask.sum())
        row[name + "_mse"] = float(error[mask].square().mean()) if mask.any() else None
    return row, prediction.detach().cpu().numpy()


def trajectory(widths, seed, data, config, output, device):
    # CPU Linear initialization is saved before moving to CUDA; no hand-set or course initial states.
    torch.manual_seed(seed)
    model = MLP(2, widths, config["activation"]).float().to(device)
    if sum(p.numel() for p in model.parameters()) != 65:
        raise ValueError("Expected 65 parameters")
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    history, rows, predictions = [], [], {}
    synchronize(device)
    started = time.perf_counter()
    for step in range(max(config["checkpoints"]) + 1):
        if step in config["checkpoints"]:
            metrics, prediction = measure(model, data)
            synchronize(device)
            row = dict(step=step, parameters=65, elapsed_seconds=time.perf_counter() - started, **metrics)
            rows.append(row)
            predictions[str(step)] = prediction
            # Save the SAME optimizer at both budgets, including its moment buffers and step.
            torch.save({"step": step, "seed": seed, "widths": widths,
                        "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                        "optimizer": optimizer.state_dict()}, output / f"step_{step}.pt")
            save_json(output / "checkpoints.json", rows)
        if step % config["history_every"] == 0 or step == max(config["checkpoints"]):
            with torch.no_grad():
                value = float((model(data["train_x"]) - data["train_y"]).square().mean())
            if not np.isfinite(value):
                raise FloatingPointError("Non-finite training history")
            history.append({"step": step, "train_mse": value})
        if step == max(config["checkpoints"]):
            break
        optimizer.zero_grad()
        loss = (model(data["train_x"]) - data["train_y"]).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite loss")
        loss.backward()
        optimizer.step()
    save_json(output / "history.json", history)
    np.savez_compressed(output / "predictions.npz", **predictions)
    return rows, model


def load_course_data(device, expected_hash):
    path = PACKAGE / "assets/expression.npz"
    if digest(path) != expected_hash:
        raise ValueError("Course data hash mismatch")
    with np.load(path, allow_pickle=False) as arrays:
        expected = {"train_x": (1024, 2), "train_y": (1024, 1),
                    "grid_x": (4225, 2), "grid_y": (4225, 1)}
        data = {}
        for key, shape in expected.items():
            value = arrays[key]
            if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError(f"Invalid {key}")
            data[key] = torch.from_numpy(value.copy()).to(device)
    return data


def formal_run(config_path, output):
    config = validate_config(json.loads(config_path.read_text(encoding="utf-8")))
    # No formal CPU fallback, even if somebody invokes this script outside sbatch.
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("Formal experiment requires a Slurm job with CUDA; use --smoke locally")
    if "A100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Expected an A100 allocation")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    data = load_course_data(device, config["data_sha256"])
    save_json(output / "config.json", config)
    sources = [Path(__file__).resolve(), PACKAGE / "models.py", PACKAGE / "train.py",
               config_path, PACKAGE / "run_ablation_exp2_T0011.sbatch"]
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for source in sources:
        (snapshot / source.name).write_bytes(source.read_bytes())
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PACKAGE, capture_output=True, text=True, check=True).stdout.strip()
    manifest = {"status": "running", "training": True, "development": False,
        "planned_trajectories": 10, "completed_trajectories": 0, "runs": [],
        "job_id": os.environ["SLURM_JOB_ID"], "commit": commit,
        "config_sha256": digest(config_path), "config_file": config_path.name, "data_sha256": config["data_sha256"],
        "sources_sha256": {p.name: digest(p) for p in sources},
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "device": torch.cuda.get_device_name(0),
            "cuda": torch.version.cuda, "threads": torch.get_num_threads(), "dtype": "float32",
            "deterministic_algorithms": True, "tf32": False}}
    save_json(output / "manifest.json", manifest)
    started = time.perf_counter()
    for name, widths in config["structures"].items():
        for seed in config["seeds"]:
            key = f"{name}_seed{seed}"
            folder = output / key
            folder.mkdir()
            try:
                rows, model = trajectory(widths, seed, data, config, folder, device)
                del model
                entry = dict(key=key, structure=name, seed=seed, status="succeeded", checkpoints=rows)
            except Exception as exc:
                entry = dict(key=key, structure=name, seed=seed, status="failed", error=repr(exc))
            manifest["runs"].append(entry)
            manifest["completed_trajectories"] = sum(r["status"] == "succeeded" for r in manifest["runs"])
            save_json(output / "manifest.json", manifest)
            print(key, entry["status"], flush=True)
    manifest["elapsed_seconds"] = time.perf_counter() - started
    manifest["status"] = "completed" if manifest["completed_trajectories"] == 10 else "partial"
    save_json(output / "manifest.json", manifest)
    return 0 if manifest["status"] == "completed" else 1


def check_output(output):
    config = validate_config(json.loads((output / "config.json").read_text(encoding="utf-8")))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    expected_keys = {f"{name}_seed{seed}" for name in config["structures"] for seed in config["seeds"]}
    if (manifest["status"] != "completed" or len(manifest["runs"]) != 10
            or manifest["completed_trajectories"] != 10
            or {r["key"] for r in manifest["runs"]} != expected_keys):
        raise ValueError("Incomplete trajectory manifest")
    for name, expected in manifest["sources_sha256"].items():
        if digest(output / "source_snapshot" / name) != expected:
            raise ValueError(f"Source snapshot hash mismatch: {name}")
    if digest(output / "source_snapshot" / manifest["config_file"]) != manifest["config_sha256"]:
        raise ValueError("Configuration source hash mismatch")
    if config != json.loads((output / "source_snapshot" / manifest["config_file"]).read_text()):
        raise ValueError("Saved configuration differs from source")
    data = load_course_data(torch.device("cpu"), manifest["data_sha256"])
    grid_y = data["grid_y"].numpy().astype(np.float64)
    table = []
    for entry in manifest["runs"]:
        if entry["status"] != "succeeded":
            raise ValueError("Failed trajectory")
        folder = output / entry["key"]
        if [r["step"] for r in entry["checkpoints"]] != config["checkpoints"]:
            raise ValueError("Incorrect checkpoint steps")
        if entry["checkpoints"] != json.loads((folder / "checkpoints.json").read_text()):
            raise ValueError("Checkpoint metrics disagree with manifest")
        history = json.loads((folder / "history.json").read_text())
        if [r["step"] for r in history] != list(range(0, 9001, 100)):
            raise ValueError("Incorrect history steps")
        if not all(np.isfinite(r["train_mse"]) for r in history):
            raise ValueError("Non-finite history")
        with np.load(folder / "predictions.npz", allow_pickle=False) as arrays:
            if set(arrays.files) != {"0", "3000", "9000"}:
                raise ValueError("Missing predictions")
            for row in entry["checkpoints"]:
                step = row["step"]
                if arrays[str(step)].shape != (4225, 1) or not np.isfinite(arrays[str(step)]).all():
                    raise ValueError("Invalid predictions")
                error = arrays[str(step)].astype(np.float64) - grid_y
                np.testing.assert_allclose(row["grid_mse"], np.mean(error ** 2), rtol=1e-10, atol=1e-12)
                np.testing.assert_allclose(row["max_abs_error"], np.max(np.abs(error)), rtol=1e-10, atol=1e-12)
                saved = torch.load(folder / f"step_{step}.pt", map_location="cpu", weights_only=True)
                if (saved["step"] != step or saved["seed"] != entry["seed"]
                        or saved["widths"] != config["structures"][entry["structure"]]):
                    raise ValueError("Checkpoint metadata mismatch")
                states = saved["optimizer"]["state"].values()
                if step == 0 and saved["optimizer"]["state"]:
                    raise ValueError("Initial optimizer must be empty")
                if step and (not saved["optimizer"]["state"] or any(float(s["step"]) != step for s in states)):
                    raise ValueError("Adam state was reset or checkpoint step is wrong")
                if not all(torch.isfinite(t).all() for t in saved["model"].values()):
                    raise ValueError("Non-finite model")
                MLP(2, saved["widths"], "relu").load_state_dict(saved["model"], strict=True)
                for state in saved["optimizer"]["state"].values():
                    if not all(torch.isfinite(t).all() for t in state.values() if torch.is_tensor(t)):
                        raise ValueError("Non-finite Adam state")
                table.append(dict(structure=entry["structure"], seed=entry["seed"], **row))
    if len(table) != 30:
        raise ValueError("Expected 30 checkpoints, including 10 initial states")
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    statistics = []
    for structure in config["structures"]:
        for step in (3000, 9000):
            selected = [r for r in table if r["structure"] == structure and r["step"] == step]
            values = np.array([r["grid_mse"] for r in selected])
            statistics.append(dict(structure=structure, step=step, n=len(values),
                median=float(np.median(values)), minimum=float(values.min()), maximum=float(values.max())))
    save_json(output / "statistics.json", statistics)
    plot_results(output, table, manifest)
    files = {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob("*"))
             if p.is_file() and p.name != "verification.json"}
    save_json(output / "verification.json", {"passed": True, "trajectories": 10,
        "checkpoints": 30, "budget_results": 20, "artifacts_sha256": files})
    print(json.dumps({"passed": True, "statistics": statistics}, indent=2))
    return 0


def plot_results(output, table, manifest):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, structure in zip(axes, ("shallow", "deep")):
        for seed in (17, 42, 123, 2026, 20261123):
            rows = [r for r in table if r["structure"] == structure and r["seed"] == seed and r["step"] > 0]
            ax.plot([r["step"] for r in rows], [r["grid_mse"] for r in rows], "o-", label=str(seed))
        ax.set(xlabel="Completed updates (same Adam trajectory)", ylabel="Grid MSE", title=structure, yscale="log")
        ax.set_xticks([3000, 9000])
        ax.legend(title="Seed", fontsize=8)
        ax.grid(alpha=.25)
    fig.savefig(output / "paired_budgets.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for entry in manifest["runs"]:
        history = json.loads((output / entry["key"] / "history.json").read_text())
        ax.plot([r["step"] for r in history], [r["train_mse"] for r in history], label=entry["key"], alpha=.8)
    ax.set(xlabel="Completed updates", ylabel="Training MSE", yscale="log", title="All trajectories (no seed filtering)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=.25)
    fig.savefig(output / "training.png", dpi=160)
    plt.close(fig)


def smoke(output):
    """One synthetic CPU trajectory, two updates; never reads course data."""
    from train import train
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    x = torch.tensor([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
    data = dict(train_x=x, train_y=x[:, :1] * .2, grid_x=x, grid_y=x[:, :1] * .2)
    config = dict(activation="relu", learning_rate=.01, checkpoints=[0, 1, 2], history_every=1)
    rows, actual = trajectory([16], 17, data, config, output, torch.device("cpu"))
    torch.manual_seed(17)
    reference = MLP(2, [16], "relu")
    train(reference, x, data["train_y"], steps=2, learning_rate=.01, optimizer="adam")
    if not all(torch.equal(actual.state_dict()[k], value) for k, value in reference.state_dict().items()):
        raise AssertionError("Snapshot instrumentation changed the uninterrupted Adam trajectory")
    for step in (0, 1, 2):
        saved = torch.load(output / f"step_{step}.pt", weights_only=True)
        if step == 0 and saved["optimizer"]["state"]:
            raise AssertionError("Initial state unexpectedly contains Adam moments")
        if step and any(float(s["step"]) != step for s in saved["optimizer"]["state"].values()):
            raise AssertionError("Optimizer state discontinuity")
    save_json(output / "smoke.json", {"passed": True, "development": True,
        "training_updates": 2, "samples": 4, "course_data_read": False,
        "matches_existing_train": True, "steps": [r["step"] for r in rows]})
    print("PASS: synthetic 4-point / 2-update smoke; matches existing train; Adam state continuous")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--check-output", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.smoke:
        result = smoke(args.output)
    elif args.check_output:
        result = check_output(args.output)
    else:
        result = formal_run(args.config, args.output)
    raise SystemExit(result)
