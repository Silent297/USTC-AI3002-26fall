"""Depth/activation experiment: formal Slurm CUDA runs, local synthetic smoke."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

from ablation_exp2 import digest, save_json
from models import MLP
from train import train
from utils.data import load_data
from utils.diagnostics import gradient_profile

PACKAGE = Path(__file__).resolve().parent
CONFIG = PACKAGE / "ablation_exp3_config.json"


def validate_config(config):
    expected = dict(depths=[2, 4, 8], width=32, activations=["sigmoid", "relu"],
        seeds=[17, 42, 123], steps=1000, learning_rate=.05, optimizer="sgd",
        dtype="float32", threads=1, diagnostic_samples=128,
        data_sha256="b3b3d27308694a343dfd1fc5b0376929b9a25f5cdd15f1e54bcd8e319e02977f")
    if config != expected:
        raise ValueError("Configuration differs from the registered experiment-3 design")
    return config


def state_hash(state):
    result = hashlib.sha256()
    for key, value in sorted(state.items()):
        array = value.detach().cpu().contiguous().numpy()
        result.update(json.dumps([key, list(array.shape), str(array.dtype)]).encode())
        result.update(array.tobytes())
    return result.hexdigest()


def state_cpu(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def activation_statistics(model, x):
    values = []
    handles = [layer.register_forward_hook(lambda module, args, value: values.append(value.detach()))
               for layer in model.hidden]
    try:
        with torch.no_grad():
            model(x)
    finally:
        for handle in handles:
            handle.remove()
    rows = []
    with torch.no_grad():
        for layer, z in zip(model.hidden, values):
            a = model.activation(z)
            rows.append(dict(preactivation_mean=float(z.mean()), preactivation_std=float(z.std(unbiased=False)),
                activation_mean=float(a.mean()), activation_std=float(a.std(unbiased=False)),
                inactive_unit_fraction=float((a == 0).all(dim=0).float().mean())
                    if isinstance(model.activation, torch.nn.ReLU) else None,
                weight_frobenius=float(layer.weight.norm())))
    return rows


def capture(model, data, count):
    x, y = data["train_x"][:count], data["train_y"][:count]
    profile = gradient_profile(model, x, y)
    statistics = activation_statistics(model, x)
    for row, stats in zip(profile["layers"], statistics):
        row.update(stats)
    profile["samples"] = len(x)
    profile["output_weight_frobenius"] = float(model.output.weight.detach().norm())
    with torch.no_grad():
        prediction = model(data["grid_x"])
        error = prediction.double() - data["grid_y"].double()
        train_error = model(data["train_x"]).double() - data["train_y"].double()
        metrics = dict(train_mse=float(train_error.square().mean()), grid_mse=float(error.square().mean()),
                       max_abs_error=float(error.abs().max()))
    # JSON validation rejects nonfinite diagnostics, but deliberately preserves None.
    json.dumps(dict(metrics=metrics, gradient_profile=profile), allow_nan=False)
    return dict(metrics=metrics, gradient_profile=profile), prediction.detach().cpu().numpy()


def run_one(initial, depth, seed, activation, data, steps, folder, device, count):
    started = time.perf_counter()
    model = MLP(2, [32] * depth, activation)
    model.load_state_dict(initial, strict=True)
    model = model.to(device=device, dtype=torch.float32)
    stages, predictions = [], {}
    for step in (0, steps):
        if step:
            history = train(model, data["train_x"], data["train_y"], steps=steps,
                            learning_rate=.05, optimizer="sgd", diagnostics=False)
            save_json(folder / "history.json", history)
        row, prediction = capture(model, data, count)
        row.update(step=step)
        stages.append(row)
        predictions[str(step)] = prediction
        torch.save(dict(step=step, depth=depth, seed=seed, activation=activation,
                        initial_hash=state_hash(initial), model=state_cpu(model)), folder / f"step_{step}.pt")
        save_json(folder / "stages.json", stages)
    np.savez_compressed(folder / "predictions.npz", **predictions)
    return dict(depth=depth, seed=seed, activation=activation,
                parameters=sum(p.numel() for p in model.parameters()), initial_hash=state_hash(initial),
                stages=stages, elapsed_seconds=time.perf_counter() - started)


def course_data(expected, device):
    if digest(PACKAGE / "assets/gradient.npz") != expected:
        raise ValueError("Gradient data hash mismatch")
    data = load_data("gradient", device=device)
    shapes = dict(train_x=(1024, 2), train_y=(1024, 1), grid_x=(4225, 2), grid_y=(4225, 1))
    if any(tuple(data[k].shape) != shape or data[k].dtype != torch.float32
           or not torch.isfinite(data[k]).all() for k, shape in shapes.items()):
        raise ValueError("Invalid gradient dataset")
    return data


def formal_run(output):
    config = validate_config(json.loads(CONFIG.read_text()))
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("Formal experiment requires a Slurm job with CUDA; use --smoke locally")
    if "A100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Expected A100")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    data = course_data(config["data_sha256"], device)
    np.save(output / "diagnostic_indices.npy", np.arange(128))
    save_json(output / "config.json", config)
    snapshot = output / "source_snapshot"
    source_files = ["ablation_exp3.py", "ablation_exp3_config.json", "run_ablation_exp3_T0013.sbatch",
                    "ablation_exp2.py", "models.py", "train.py", "utils/data.py", "utils/diagnostics.py"]
    for name in source_files:
        destination = snapshot / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((PACKAGE / name).read_bytes())
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PACKAGE, capture_output=True, text=True, check=True).stdout.strip()
    manifest = dict(status="running", training=True, development=False, job_id=os.environ["SLURM_JOB_ID"], commit=commit,
        planned_runs=18, successful_runs=0, runs=[],
        sources_sha256={name: digest(PACKAGE / name) for name in source_files},
        environment=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
            cuda=torch.version.cuda, device=torch.cuda.get_device_name(0), dtype="float32", threads=1,
            deterministic_algorithms=True, tf32=False))
    save_json(output / "manifest.json", manifest)
    paired_folder = output / "paired_initial"
    paired_folder.mkdir()
    started = time.perf_counter()
    for depth in config["depths"]:
        for seed in config["seeds"]:
            torch.manual_seed(seed)
            initial = state_cpu(MLP(2, [32] * depth, "sigmoid"))
            torch.save(initial, paired_folder / f"depth{depth}_seed{seed}.pt")
            for activation in config["activations"]:
                key = f"depth{depth}_{activation}_seed{seed}"
                folder = output / key
                folder.mkdir()
                try:
                    row = run_one(initial, depth, seed, activation, data, 1000, folder, device, 128)
                    row.update(status="succeeded", key=key)
                except Exception as exc:
                    row = dict(key=key, status="failed", depth=depth, seed=seed,
                               activation=activation, error=repr(exc))
                manifest["runs"].append(row)
                manifest["successful_runs"] = sum(r["status"] == "succeeded" for r in manifest["runs"])
                save_json(output / "manifest.json", manifest)
                print(key, row["status"], flush=True)
    manifest["status"] = "completed" if manifest["successful_runs"] == 18 else "partial"
    manifest["elapsed_seconds"] = time.perf_counter() - started
    save_json(output / "manifest.json", manifest)
    return 0 if manifest["status"] == "completed" else 1


def check_output(output):
    config = validate_config(json.loads((output / "config.json").read_text()))
    manifest = json.loads((output / "manifest.json").read_text())
    expected = {f"depth{d}_{a}_seed{s}" for d in config["depths"]
                for a in config["activations"] for s in config["seeds"]}
    if (manifest["status"] != "completed" or manifest["development"] is not False or manifest["training"] is not True
            or manifest["successful_runs"] != 18 or len(manifest["runs"]) != 18
            or {r["key"] for r in manifest["runs"]} != expected):
        raise ValueError("Incomplete experiment")
    for name, expected_hash in manifest["sources_sha256"].items():
        if digest(output / "source_snapshot" / name) != expected_hash:
            raise ValueError(f"Source hash mismatch: {name}")
    if config != json.loads((output / "source_snapshot/ablation_exp3_config.json").read_text()):
        raise ValueError("Saved configuration mismatch")
    np.testing.assert_array_equal(np.load(output / "diagnostic_indices.npy", allow_pickle=False), np.arange(128))
    data = course_data(config["data_sha256"], torch.device("cpu"))
    table = []
    for row in manifest["runs"]:
        depth, seed, activation = row["depth"], row["seed"], row["activation"]
        if row["key"] != f"depth{depth}_{activation}_seed{seed}" or row["status"] != "succeeded":
            raise ValueError("Run metadata mismatch")
        initial = torch.load(output / "paired_initial" / f"depth{depth}_seed{seed}.pt", map_location="cpu", weights_only=True)
        if state_hash(initial) != row["initial_hash"] or row["parameters"] != 96 + (depth - 1) * 1056 + 33:
            raise ValueError("Paired initial state or parameter count mismatch")
        folder = output / row["key"]
        if row["stages"] != json.loads((folder / "stages.json").read_text()) or [r["step"] for r in row["stages"]] != [0, 1000]:
            raise ValueError("Incorrect stage records")
        history = json.loads((folder / "history.json").read_text())
        if [r["step"] for r in history] != list(range(0, 1001, 100)):
            raise ValueError("Incorrect history")
        json.dumps(history, allow_nan=False)
        with np.load(folder / "predictions.npz", allow_pickle=False) as arrays:
            if set(arrays.files) != {"0", "1000"}:
                raise ValueError("Missing predictions")
            for stage in row["stages"]:
                step = stage["step"]
                saved = torch.load(folder / f"step_{step}.pt", map_location="cpu", weights_only=True)
                if (saved["step"], saved["depth"], saved["seed"], saved["activation"], saved["initial_hash"]) != (
                        step, depth, seed, activation, row["initial_hash"]):
                    raise ValueError("Checkpoint metadata mismatch")
                if step == 0 and state_hash(saved["model"]) != state_hash(initial):
                    raise ValueError("Activation pair does not share identical Linear weights")
                model = MLP(2, [32] * depth, activation)
                model.load_state_dict(saved["model"], strict=True)
                if not all(torch.isfinite(v).all() for v in saved["model"].values()):
                    raise ValueError("Non-finite model")
                prediction = arrays[str(step)]
                if prediction.shape != (4225, 1) or prediction.dtype != np.float32 or not np.isfinite(prediction).all():
                    raise ValueError("Invalid predictions")
                error = prediction.astype(np.float64) - data["grid_y"].numpy().astype(np.float64)
                np.testing.assert_allclose(np.mean(error ** 2), stage["metrics"]["grid_mse"], rtol=1e-10, atol=1e-12)
                profile = stage["gradient_profile"]
                if profile["samples"] != 128 or len(profile["layers"]) != depth:
                    raise ValueError("Incorrect diagnostic sample or layer count")
                json.dumps(profile, allow_nan=False)
                for layer in profile["layers"]:
                    value = layer["relative_to_output"]
                    if profile["output_gradient_rms"] == 0:
                        if value is not None:
                            raise ValueError("Undefined relative gradient must stay null")
                    elif value is None or not np.isclose(value, layer["gradient_rms"] / profile["output_gradient_rms"]):
                        raise ValueError("Inconsistent relative gradient")
                table.append(dict(depth=depth, activation=activation, seed=seed, step=step,
                    parameters=row["parameters"], **stage["metrics"],
                    first_layer_rho=profile["layers"][0]["relative_to_output"],
                    output_gradient_rms=profile["output_gradient_rms"]))
    if len(table) != 36:
        raise ValueError("Expected 36 stage records")
    save_json(output / "summary.json", table)
    plot_results(output, manifest["runs"])
    files = {p.relative_to(output).as_posix(): digest(p) for p in sorted(output.rglob("*"))
             if p.is_file() and p.name != "verification.json"}
    save_json(output / "verification.json", dict(passed=True, runs=18, pairs=9, stages=36, artifacts_sha256=files))
    print(json.dumps(dict(passed=True, runs=18, pairs=9, stages=36), indent=2))
    return 0


def plot_results(output, runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True, sharey=True)
    for stage_index, stage_name in enumerate(("Initial", "Final")):
        for col, depth in enumerate((2, 4, 8)):
            ax = axes[stage_index, col]
            zero, undefined = 0, 0
            for row in runs:
                if row["depth"] != depth:
                    continue
                profile = row["stages"][stage_index]["gradient_profile"]
                ratios = [layer["relative_to_output"] for layer in profile["layers"]]
                zero += sum(r == 0 for r in ratios if r is not None)
                undefined += sum(r is None for r in ratios)
                values = [r if r is not None and r > 0 else np.nan for r in ratios]
                ax.plot(range(1, depth + 1), values, "o-" if row["activation"] == "relu" else "s--",
                        color="C0" if row["activation"] == "relu" else "C1", alpha=.65)
            ax.set(title=f"{stage_name}, {depth} layers", xlabel="Hidden layer", ylabel="Relative gradient RMS", yscale="log")
            ax.text(.02, .02, f"zero={zero}; undefined={undefined}", transform=ax.transAxes, fontsize=8)
            ax.grid(alpha=.25)
    fig.suptitle("ReLU: blue solid; Sigmoid: orange dashed; all 3 seeds. Zero/null stay in JSON.")
    fig.savefig(output / "gradient_profiles.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    first_zero, first_null, mse_zero = 0, 0, 0
    for activation, style in (("sigmoid", "s--"), ("relu", "o-")):
        for seed in (17, 42, 123):
            rows = sorted((r for r in runs if r["activation"] == activation and r["seed"] == seed), key=lambda r:r["depth"])
            first = [r["stages"][0]["gradient_profile"]["layers"][0]["relative_to_output"] for r in rows]
            first_zero += sum(v == 0 for v in first if v is not None)
            first_null += sum(v is None for v in first)
            mse_zero += sum(r["stages"][1]["metrics"]["grid_mse"] == 0 for r in rows)
            axes[0].plot([r["depth"] for r in rows], [v if v is not None and v > 0 else np.nan for v in first],
                         style, label=f"{activation} {seed}")
            axes[1].plot([r["depth"] for r in rows], [r["stages"][1]["metrics"]["grid_mse"] for r in rows], style)
    axes[0].set(title="Initial first-layer relative gradient", ylabel="RMS / output RMS")
    axes[1].set(title="Final grid MSE", ylabel="MSE")
    axes[0].text(.02, .02, f"zero={first_zero}; undefined={first_null}", transform=axes[0].transAxes, fontsize=8)
    axes[1].text(.02, .02, f"zero MSE={mse_zero}; exact values retained in JSON", transform=axes[1].transAxes, fontsize=8)
    for ax in axes:
        ax.set(xlabel="Hidden depth", yscale="log", xticks=[2, 4, 8])
        ax.grid(alpha=.25)
    axes[0].legend(fontsize=7)
    fig.savefig(output / "depth_comparison.png", dpi=160)
    plt.close(fig)


def smoke(output):
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    x = torch.tensor([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
    y = x[:, :1] + .5 * x[:, 1:]
    data = dict(train_x=x, train_y=y, grid_x=x, grid_y=y)
    torch.manual_seed(17)
    initial = state_cpu(MLP(2, [32] * 2, "sigmoid"))
    for activation in ("sigmoid", "relu"):
        folder = output / activation
        folder.mkdir()
        run_one(initial, 2, 17, activation, data, 2, folder, torch.device("cpu"), 4)
        saved = torch.load(folder / "step_2.pt", weights_only=True)
        reference = MLP(2, [32] * 2, activation)
        reference.load_state_dict(initial)
        train(reference, x, y, steps=2, learning_rate=.05)
        if not all(torch.equal(reference.state_dict()[k], value) for k, value in saved["model"].items()):
            raise AssertionError("Diagnostics altered training")
        gradients = [None if p.grad is None else p.grad.clone() for p in reference.parameters()]
        before = state_hash(reference.state_dict())
        capture(reference, data, 4)
        if before != state_hash(reference.state_dict()) or not all(
                (p.grad is None if g is None else torch.equal(p.grad, g)) for p, g in zip(reference.parameters(), gradients)):
            raise AssertionError("Diagnostics changed weights or .grad")
        init = torch.load(folder / "step_0.pt", weights_only=True)
        if state_hash(init["model"]) != state_hash(initial):
            raise AssertionError("Pair initialization mismatch")
    zero = MLP(2, [32] * 2, "relu")
    with torch.no_grad():
        for p in zero.parameters():
            p.zero_()
    profile = gradient_profile(zero, x, torch.zeros_like(y))
    if profile["output_gradient_rms"] != 0 or any(r["relative_to_output"] is not None for r in profile["layers"]):
        raise AssertionError("Undefined gradients were not preserved")
    save_json(output / "smoke.json", dict(passed=True, development=True, samples=4, steps=2,
        paired_initial_equal=True, matches_existing_train=True, diagnostics_preserve_grad=True,
        undefined_ratio_preserved=True, course_data_read=False))
    print("PASS: paired initialization; two-step SGD parity; diagnostics preserve .grad; null ratios")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", action="store_true")
    group.add_argument("--check-output", action="store_true")
    group.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(smoke(args.output) if args.smoke else check_output(args.output) if args.check_output else formal_run(args.output))
