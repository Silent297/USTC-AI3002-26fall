"""P1: paired affine regression, CUDA formal runs and CPU synthetic smoke only."""
import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch import nn

PACKAGE = Path(__file__).resolve().parent
CONFIG = PACKAGE / "p1_config.json"
SOURCES = ("p1_learning_rate.py", "p1_config.json", "run_p1_T0018.sbatch", "train.py", "utils/diagnostics.py")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hashes(root):
    return {p.relative_to(root).as_posix(): sha256(p) for p in sorted(root.rglob("*"))
            if p.is_file() and p != root / "artifact_hashes.json"}


def empty_output(path):
    path = path.resolve()
    require(not path.exists(), "choose a new, nonexistent output directory")
    path.mkdir(parents=True)
    return path


def vector(model):
    return torch.cat([model.weight.detach().reshape(-1), model.bias.detach()])


def reference(data):
    train_x = data["train_x"].detach().cpu().numpy().astype(np.float64)
    train_y = data["train_y"].detach().cpu().numpy().ravel().astype(np.float64)
    A = np.column_stack((train_x, np.ones(len(train_x))))
    theta, _, rank, singular = np.linalg.lstsq(A, train_y, rcond=None)
    known = np.array([1., .5, 0.])
    result = dict(coefficient_order=["w1", "w2", "bias"], least_squares=theta.tolist(),
        rank=int(rank), singular_values=singular.tolist(), hessian_eigenvalues=np.linalg.eigvalsh(2 * A.T @ A / len(A)).tolist())
    for name, coefficients in (("least_squares", theta), ("known", known)):
        result[name + "_parameters"] = coefficients.tolist()
        for part in ("train", "grid"):
            x = data[part + "_x"].detach().cpu().numpy().astype(np.float64)
            y = data[part + "_y"].detach().cpu().numpy().ravel().astype(np.float64)
            result[name + "_" + part + "_mse"] = float(np.mean((x @ coefficients[:2] + coefficients[2] - y) ** 2))
    return result


def trajectory(initial, data, lr, steps, folder):
    model = nn.Linear(2, 1).to(data["train_x"].device)
    model.load_state_dict(initial)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, folder / "initial.pt")
    history, predictions = [], {}
    for step in range(steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = (model(data["train_x"]) - data["train_y"]).square().mean()
        loss.backward()
        theta = vector(model).clone()
        gradient = torch.cat([model.weight.grad.reshape(-1), model.bias.grad]).detach().clone()
        with torch.no_grad():
            grid_prediction = model(data["grid_x"])
            grid_error = grid_prediction.double() - data["grid_y"].double()
            row = dict(step=step, train_mse=float(loss.detach()), grid_mse=float(grid_error.square().mean()),
                       gradient_norm=float(gradient.double().norm()),
                       expected_update_norm=float(lr * gradient.double().norm()))
            for j, name in enumerate(("w1", "w2", "bias")):
                row[name] = float(theta[j])
                row["grad_" + name] = float(gradient[j])
            if step in (0, steps):
                predictions[str(step)] = grid_prediction.detach().cpu().numpy()
        if step < steps:
            optimizer.step()
            delta = vector(model) - theta
            row["actual_update_norm"] = float(delta.double().norm())
        else:
            # Terminal gradient is measured, but no extra optimizer step is taken.
            row["actual_update_norm"] = None
        require(all(v is None or np.isfinite(v) for v in row.values()), "nonfinite trajectory")
        history.append(row)
    with (folder / "history.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, folder / "final.pt")
    np.savez_compressed(folder / "grid_predictions.npz", **predictions)
    return model, history


def validate_config(cfg):
    require(cfg == dict(seeds=[17, 42, 123], learning_rates=[.05, .00001], steps=1000,
        optimizer="sgd", dtype="float32", threads=1,
        data_sha256="b3b3d27308694a343dfd1fc5b0376929b9a25f5cdd15f1e54bcd8e319e02977f"),
        "configuration differs from the registered P1 protocol")


def execute(output, *, toy=None):
    cfg = read_json(CONFIG)
    validate_config(cfg)
    if toy is None:
        require(bool(os.environ.get("SLURM_JOB_ID")), "formal run requires Slurm")
        require(torch.cuda.is_available(), "formal run requires CUDA; no CPU fallback")
        require("A100" in torch.cuda.get_device_name(0), "formal run requires A100")
        path = PACKAGE / "assets/gradient.npz"
        require(sha256(path) == cfg["data_sha256"], "course data hash mismatch")
        with np.load(path, allow_pickle=False) as arrays:
            raw = {k: arrays[k].copy() for k in ("train_x", "train_y", "grid_x", "grid_y")}
        for key, shape in dict(train_x=(1024, 2), train_y=(1024, 1), grid_x=(4225, 2), grid_y=(4225, 1)).items():
            require(raw[key].shape == shape and raw[key].dtype == np.float32 and np.isfinite(raw[key]).all(), key)
        device = torch.device("cuda:0")
    else:
        raw = toy
        cfg.update(steps=2, seeds=[17])
        device = torch.device("cpu")
    output = empty_output(output)
    torch.set_num_threads(cfg["threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data = {k: torch.from_numpy(v).to(device) for k, v in raw.items()}
    # Save the small data used by this run so CPU checks need no GPU replay.
    np.savez_compressed(output / "data.npz", **raw)
    save_json(output / "config.json", cfg)
    save_json(output / "reference.json", reference(data))
    source_hashes = {}
    for name in SOURCES:
        destination = output / "source_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((PACKAGE / name).read_bytes())
        source_hashes[name] = sha256(PACKAGE / name)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PACKAGE, capture_output=True, text=True, check=True).stdout.strip()
    manifest = dict(status="running", development=toy is not None, optimizer_updates_per_run=cfg["steps"],
        planned_trajectories=len(cfg["seeds"]) * 2, runs=[], source_hashes=source_hashes,
        data_sha256=sha256(output / "data.npz"), source_data_sha256=None if toy is not None else cfg["data_sha256"],
        job_id=os.environ.get("SLURM_JOB_ID"), commit=commit,
        environment=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                         device=str(device), gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                         dtype="float32", threads=1, deterministic=True, tf32=False))
    save_json(output / "manifest.json", manifest)
    try:
        for seed in cfg["seeds"]:
            torch.manual_seed(seed)
            initial = {k: v.detach().clone() for k, v in nn.Linear(2, 1).state_dict().items()}
            for label, lr in zip(("normal", "small"), cfg["learning_rates"]):
                key = f"{label}_seed{seed}"
                folder = output / key
                folder.mkdir()
                _, history = trajectory(initial, data, lr, cfg["steps"], folder)
                manifest["runs"].append(dict(key=key, seed=seed, learning_rate=lr, steps=cfg["steps"],
                    initial=history[0], final=history[-1]))
                save_json(output / "manifest.json", manifest)
                print(key, "finished", flush=True)
        require(source_hashes == {name: sha256(PACKAGE / name) for name in SOURCES}, "source changed during run")
        manifest["status"] = "completed"
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save_json(output / "manifest.json", manifest)
        save_json(output / "artifact_hashes.json", hashes(output))


def check(source, output):
    source, output = source.resolve(), output.resolve()
    require(source != output and source not in output.parents, "check output must be outside source")
    before = hashes(source)
    require(before == read_json(source / "artifact_hashes.json"), "artifact hash mismatch")
    cfg, manifest = read_json(source / "config.json"), read_json(source / "manifest.json")
    require(manifest["status"] == "completed", "incomplete run")
    if not manifest["development"]:
        validate_config(cfg)
        require(manifest["source_data_sha256"] == cfg["data_sha256"] and
                manifest["environment"]["gpu"] and "A100" in manifest["environment"]["gpu"], "formal provenance mismatch")
        require(sha256(PACKAGE / "assets/gradient.npz") == cfg["data_sha256"], "local course data mismatch")
    require(manifest["source_hashes"] == {name: sha256(source / "source_snapshot" / name) for name in SOURCES}, "snapshot mismatch")
    require(sha256(source / "data.npz") == manifest["data_sha256"], "data mismatch")
    with np.load(source / "data.npz", allow_pickle=False) as arrays:
        raw = {k: arrays[k] for k in arrays.files}
    if not manifest["development"]:
        with np.load(PACKAGE / "assets/gradient.npz", allow_pickle=False) as arrays:
            require(all(np.array_equal(raw[k], arrays[k]) for k in raw), "saved data differs from course data")
    A = np.column_stack((raw["train_x"].astype(np.float64), np.ones(len(raw["train_x"]))))
    y = raw["train_y"].astype(np.float64)
    expected = {f"{label}_seed{seed}" for seed in cfg["seeds"] for label in ("normal", "small")}
    require(len(manifest["runs"]) == len(expected) and {r["key"] for r in manifest["runs"]} == expected, "missing/duplicate trajectory")
    loaded, errors = {}, []
    for run in manifest["runs"]:
        key = run["key"]
        with (source / key / "history.csv").open(encoding="utf-8", newline="") as stream:
            history = [{k: float(v) if v else None for k, v in row.items()} for row in csv.DictReader(stream)]
        require([r["step"] for r in history] == list(range(cfg["steps"] + 1)), "step count mismatch")
        require(history[-1]["actual_update_norm"] is None, "terminal row has an optimizer update")
        theta = np.array([[r[n] for n in ("w1", "w2", "bias")] for r in history])
        gradient = np.array([[r["grad_" + n] for n in ("w1", "w2", "bias")] for r in history])
        residual = A @ theta.T - y
        independent_gradient = (2 / len(A) * A.T @ residual).T
        np.testing.assert_allclose(gradient, independent_gradient, rtol=1e-4, atol=2e-6)
        np.testing.assert_allclose([r["train_mse"] for r in history], np.mean(residual ** 2, axis=0), rtol=1e-4, atol=1e-10)
        np.testing.assert_allclose([r["gradient_norm"] for r in history], np.linalg.norm(gradient, axis=1), rtol=1e-10, atol=1e-12)
        lr = run["learning_rate"]
        require(key == f"{'normal' if key.startswith('normal') else 'small'}_seed{run['seed']}"
                and run["seed"] in cfg["seeds"], "run seed mismatch")
        require(lr == cfg["learning_rates"][0 if key.startswith("normal") else 1] and run["steps"] == cfg["steps"], "run configuration mismatch")
        require(run["initial"] == history[0] and run["final"] == history[-1], "manifest/history endpoint mismatch")
        np.testing.assert_allclose(theta[1:], theta[:-1] - lr * gradient[:-1], rtol=2e-7, atol=2e-7)
        np.testing.assert_allclose([r["actual_update_norm"] for r in history[:-1]], np.linalg.norm(np.diff(theta, axis=0), axis=1), rtol=2e-6, atol=1e-8)
        np.testing.assert_allclose([r["expected_update_norm"] for r in history], lr * np.linalg.norm(gradient, axis=1), rtol=1e-10, atol=1e-12)
        for name, index in (("initial", 0), ("final", -1)):
            state = torch.load(source / key / (name + ".pt"), weights_only=True, map_location="cpu")
            saved = np.r_[state["weight"].numpy().ravel(), state["bias"].numpy()]
            np.testing.assert_array_equal(saved, theta[index].astype(np.float32))
            prediction = raw["grid_x"].astype(np.float64) @ theta[index, :2] + theta[index, 2]
            with np.load(source / key / "grid_predictions.npz", allow_pickle=False) as predictions:
                np.testing.assert_allclose(predictions[str(0 if index == 0 else cfg["steps"])].ravel(), prediction, rtol=1e-4, atol=5e-7)
        np.testing.assert_allclose([r["grid_mse"] for r in history], np.mean((raw["grid_x"].astype(np.float64) @ theta[:, :2].T + theta[:, 2] - raw["grid_y"].astype(np.float64)) ** 2, axis=0), rtol=1e-4, atol=1e-10)
        loaded[key] = history
        errors.append(float(np.max(np.abs(gradient - independent_gradient))))
    pairs = []
    for seed in cfg["seeds"]:
        a, b = (loaded[f"{label}_seed{seed}"] for label in ("normal", "small"))
        require(all(a[0][k] == b[0][k] for k in ("w1", "w2", "bias", "grad_w1", "grad_w2", "grad_bias", "train_mse")), "unpaired initialization/gradient")
        pairs.append(dict(seed=seed, initial_gradient_norm=a[0]["gradient_norm"],
            normal_initial_update=a[0]["actual_update_norm"], small_initial_update=b[0]["actual_update_norm"],
            actual_initial_update_ratio=a[0]["actual_update_norm"] / b[0]["actual_update_norm"],
            normal_final_train_mse=a[-1]["train_mse"], small_final_train_mse=b[-1]["train_mse"],
            normal_final_grid_mse=a[-1]["grid_mse"], small_final_grid_mse=b[-1]["grid_mse"],
            normal_final_gradient_norm=a[-1]["gradient_norm"], small_final_gradient_norm=b[-1]["gradient_norm"]))
    ref = reference({k: torch.from_numpy(v) for k, v in raw.items()})
    saved_reference = read_json(source / "reference.json")
    require(ref.keys() == saved_reference.keys(), "least-squares reference fields mismatch")
    for key in ref:
        if key in ("rank", "coefficient_order"):
            require(ref[key] == saved_reference[key], "least-squares reference identity mismatch")
        else:
            # NumPy SVD/BLAS need not round identically on Linux and Windows.
            np.testing.assert_allclose(ref[key], saved_reference[key], rtol=1e-10, atol=1e-12)
    require(before == hashes(source), "check modified original files")
    output = empty_output(output)
    save_json(output / "verification.json", dict(passed=True, development=manifest["development"],
        trajectories=len(loaded), gradient_max_abs_error=max(errors), source_unchanged=True,
        hashed_files=len(before), pairs=pairs, reference=ref, optimizer_updates=0,
        note="CPU float64 algebra audit; no GPU metric replay or fitting."))
    print("CHECK PASSED", len(loaded), "trajectories", "max gradient difference", max(errors))


def smoke(output):
    root = empty_output(output)
    x = np.array([[-1., -.5], [0., 1.], [1., 0.], [.5, -.8]], dtype=np.float32)
    toy = dict(train_x=x, train_y=(x[:, :1] + .5 * x[:, 1:]), grid_x=x, grid_y=(x[:, :1] + .5 * x[:, 1:]))
    execute(root / "run", toy=toy)
    check(root / "run", root / "check")
    from train import train
    for label, lr in (("normal", .05), ("small", .00001)):
        folder = root / "run" / f"{label}_seed17"
        model = nn.Linear(2, 1)
        model.load_state_dict(torch.load(folder / "initial.pt", weights_only=True))
        train(model, torch.from_numpy(x), torch.from_numpy(toy["train_y"]), steps=2, learning_rate=lr)
        expected = torch.load(folder / "final.pt", weights_only=True)
        require(all(torch.equal(v, expected[k]) for k, v in model.state_dict().items()), "course SGD parity failed")
    target = root / "run/normal_seed17/history.csv"
    original = target.read_bytes()
    target.write_bytes(original + b"tamper")
    try:
        check(root / "run", root / "forbidden_check")
    except ValueError as error:
        require("hash mismatch" in str(error), "wrong tamper rejection")
    else:
        raise AssertionError("tampering accepted")
    finally:
        target.write_bytes(original)
    print("SMOKE PASSED: paired starts/gradients; 2 rates x 2 updates; course SGD exact parity; independent algebra; tamper rejection")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "smoke", "check"):
        command = sub.add_parser(name)
        command.add_argument("--output", type=Path, required=True)
        if name == "check":
            command.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        execute(args.output)
    elif args.command == "smoke":
        smoke(args.output)
    else:
        check(args.source, args.output)
