"""Read-only CPU verification and analysis of formal experiment-3 artifacts."""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from ablation_exp2 import digest, save_json
from ablation_exp3 import capture, course_data
from models import MLP

PACKAGE = Path(__file__).resolve().parent


def stats(values):
    valid = [v for v in values if v is not None]
    return dict(n=len(values), undefined=len(values) - len(valid), zero=sum(v == 0 for v in valid),
                median=float(np.median(valid)) if valid else None,
                minimum=float(min(valid)) if valid else None, maximum=float(max(valid)) if valid else None)


def analyze(source, output):
    source, output = source.resolve(), output.resolve()
    if output.is_relative_to(source):
        raise ValueError("Analysis output must be outside original artifacts")
    output.mkdir(parents=True, exist_ok=False)
    before = {p.relative_to(source).as_posix(): digest(p) for p in source.rglob("*") if p.is_file()}
    verification = json.loads((source / "verification.json").read_text())
    for name, expected in verification["artifacts_sha256"].items():
        path = (source / name).resolve()
        if not path.is_relative_to(source) or digest(path) != expected:
            raise ValueError(f"Artifact hash mismatch: {name}")
    copied = output / "checked_copy"
    shutil.copytree(source, copied)
    result = subprocess.run([sys.executable, str(PACKAGE / "ablation_exp3.py"),
        "--check-output", "--output", str(copied)], capture_output=True, text=True)
    (output / "check_output.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError("Original checker failed on copied artifacts")
    manifest = json.loads((source / "manifest.json").read_text())
    config = json.loads((source / "config.json").read_text())
    source_checks = {}
    for name, expected in manifest["sources_sha256"].items():
        path = source / "source_snapshot" / name
        same = path.read_bytes().replace(b"\r\n", b"\n") == (PACKAGE / name).read_bytes().replace(b"\r\n", b"\n")
        if digest(path) != expected or not same:
            raise ValueError(f"Source snapshot/current source mismatch: {name}")
        source_checks[name] = dict(hash_matches=True, current_matches_after_lf_normalization=same)
    torch.set_num_threads(1)
    data = course_data(config["data_sha256"], torch.device("cpu"))
    train_mean = float(data["train_y"].double().mean())
    baseline = dict(prediction=train_mean,
        grid_mse=float((data["grid_y"].double() - train_mean).square().mean()),
        train_mse=float((data["train_y"].double() - train_mean).square().mean()),
        grid_target_std=float(data["grid_y"].double().std(unbiased=False)))
    cpu_checks, rows = [], []
    for run in manifest["runs"]:
        folder = source / run["key"]
        with np.load(folder / "predictions.npz", allow_pickle=False) as predictions:
            for stage in run["stages"]:
                step = stage["step"]
                saved = torch.load(folder / f"step_{step}.pt", map_location="cpu", weights_only=True)
                model = MLP(2, [32] * run["depth"], run["activation"])
                model.load_state_dict(saved["model"], strict=True)
                actual, prediction = capture(model, data, 128)
                expected = predictions[str(step)]
                np.testing.assert_allclose(prediction, expected, rtol=1e-4, atol=2e-5)
                absolute_differences, rho_relative_differences = [], []
                original_profile = stage["gradient_profile"]
                np.testing.assert_allclose(actual["gradient_profile"]["output_gradient_rms"],
                    original_profile["output_gradient_rms"], rtol=.005, atol=1e-12)
                for measured, original in zip(actual["gradient_profile"]["layers"], original_profile["layers"]):
                    np.testing.assert_allclose(measured["gradient_rms"], original["gradient_rms"], rtol=.005, atol=1e-12)
                    a, b = measured["relative_to_output"], original["relative_to_output"]
                    if a is None or b is None:
                        if a is not None or b is not None:
                            raise AssertionError("CPU/CUDA disagree about undefined ratio")
                    else:
                        np.testing.assert_allclose(a, b, rtol=.005, atol=1e-10)
                        if b != 0:
                            rho_relative_differences.append(abs(a - b) / abs(b))
                    absolute_differences.append(abs(measured["gradient_rms"] - original["gradient_rms"]))
                cpu_checks.append(dict(key=run["key"], step=step,
                    prediction_max_abs_difference=float(np.max(np.abs(prediction - expected))),
                    gradient_rms_max_abs_difference=max(absolute_differences),
                    rho_max_relative_difference=max(rho_relative_differences, default=0)))
                layers = original_profile["layers"]
                rows.append(dict(key=run["key"], depth=run["depth"], activation=run["activation"],
                    seed=run["seed"], step=step, parameters=run["parameters"],
                    elapsed_seconds=run["elapsed_seconds"], **stage["metrics"],
                    first_rms=layers[0]["gradient_rms"], first_rho=layers[0]["relative_to_output"],
                    output_rms=original_profile["output_gradient_rms"],
                    first_derivative=layers[0]["derivative_mean"],
                    layer_derivatives=[r["derivative_mean"] for r in layers],
                    layer_weight_norms=[r["weight_frobenius"] for r in layers],
                    relu_inactive_fractions=[r["inactive_unit_fraction"] for r in layers],
                    grid_prediction_std=float(expected.astype(np.float64).std()),
                    mse_reduction_vs_constant_percent=100 * (1 - stage["metrics"]["grid_mse"] / baseline["grid_mse"])))
    grouped = []
    for depth in (2, 4, 8):
        for activation in ("sigmoid", "relu"):
            for step in (0, 1000):
                selected = [r for r in rows if (r["depth"], r["activation"], r["step"]) == (depth, activation, step)]
                grouped.append(dict(depth=depth, activation=activation, step=step, n=len(selected),
                    **{key: stats([r[key] for r in selected]) for key in ("grid_mse", "train_mse", "first_rho",
                        "first_rms", "output_rms", "first_derivative", "grid_prediction_std")}))
    zero_count = null_count = 0
    for run in manifest["runs"]:
        for stage in run["stages"]:
            for layer in stage["gradient_profile"]["layers"]:
                value = layer["relative_to_output"]
                zero_count += value == 0 if value is not None else False
                null_count += value is None
    summary = dict(passed=True, job_id=manifest["job_id"], commit=manifest["commit"],
        environment=manifest["environment"], elapsed_seconds=manifest["elapsed_seconds"],
        verified_artifacts=len(verification["artifacts_sha256"]), sources=source_checks,
        constant_baseline=baseline, cpu_checks=cpu_checks, rows=rows, grouped=grouped,
        zero_rho_count=zero_count, undefined_rho_count=null_count,
        original_sha256={name: before[name] for name in ("manifest.json", "verification.json", "summary.json",
            "depth_comparison.png", "gradient_profiles.png")})
    save_json(output / "analysis.json", summary)
    for name in ("gradient_profiles.png", "depth_comparison.png"):
        shutil.copyfile(source / name, output / name)
    after = {p.relative_to(source).as_posix(): digest(p) for p in source.rglob("*") if p.is_file()}
    if before != after:
        raise AssertionError("Original input changed")
    print(json.dumps(dict(passed=True, job_id=summary["job_id"], verified_artifacts=summary["verified_artifacts"],
        constant_baseline=baseline, zero_rho_count=zero_count, undefined_rho_count=null_count,
        prediction_max_abs_difference=max(r["prediction_max_abs_difference"] for r in cpu_checks),
        gradient_rms_max_abs_difference=max(r["gradient_rms_max_abs_difference"] for r in cpu_checks),
        rho_max_relative_difference=max(r["rho_max_relative_difference"] for r in cpu_checks),
        grouped=grouped), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.input, args.output)
