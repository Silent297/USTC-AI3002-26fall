"""Read-only analysis of experiment-2 artifacts; CPU forwards, never training."""

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from ablation_exp2 import digest, save_json
from models import MLP

PACKAGE = Path(__file__).resolve().parent


def plot_comparison(output, paired):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    seeds = [17, 42, 123, 2026, 20261123]
    for index, seed in enumerate(seeds):
        for structure, offset in (("shallow", 0), ("deep", 2)):
            row = next(r for r in paired if r["structure"] == structure and r["seed"] == seed)
            shift = (index - 2) * 0.04
            ax.plot([offset + shift, offset + 1 + shift], [row["grid_3000"], row["grid_9000"]],
                "o-", color=f"C{index}", alpha=.8, label=str(seed) if structure == "shallow" else None)
    for offset, structure in ((0, "shallow"), (2, "deep")):
        for position, field in ((offset, "grid_3000"), (offset + 1, "grid_9000")):
            values = [r[field] for r in paired if r["structure"] == structure]
            ax.plot(position, np.median(values), "k_", markersize=24, markeredgewidth=2)
    ax.set(yscale="log", ylabel="Grid MSE (shared scale)",
           title="Equal 65 parameters: all five seeds; black bars = medians")
    ax.set_xticks([0, 1, 2, 3], ["Shallow\n3000", "Shallow\n9000", "Deep\n3000", "Deep\n9000"])
    ax.legend(title="Seed", ncol=5, loc="lower center", bbox_to_anchor=(.5, 1.08), fontsize=8)
    ax.grid(axis="y", alpha=.25)
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)


def analyze(source, output):
    source, output = source.resolve(), output.resolve()
    if output.is_relative_to(source):
        raise ValueError("Analysis must not write inside the original artifacts")
    output.mkdir(parents=True, exist_ok=False)
    verification = json.loads((source / "verification.json").read_text())
    original_hashes = {str(p.relative_to(source)): digest(p) for p in source.rglob("*") if p.is_file()}
    for name, expected in verification["artifacts_sha256"].items():
        path = (source / name).resolve()
        if not path.is_relative_to(source) or digest(path) != expected:
            raise ValueError(f"Artifact hash mismatch: {name}")
    # The original checker regenerates plots and verification.json, so execute on a copy.
    copied = output / "checked_copy"
    shutil.copytree(source, copied)
    result = subprocess.run([sys.executable, str(PACKAGE / "ablation_exp2.py"),
        "--check-output", "--output", str(copied)], capture_output=True, text=True)
    (output / "check_output.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError("Output checker failed; see check_output.log")
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest["development"] is not False or manifest["training"] is not True:
        raise ValueError("Expected a formal training result")
    source_checks = {}
    for name, expected in manifest["sources_sha256"].items():
        path = source / "source_snapshot" / name
        current = PACKAGE / name
        snapshot = path.read_bytes()
        if digest(path) != expected:
            raise ValueError(f"Source snapshot mismatch: {name}")
        same = snapshot.replace(b"\r\n", b"\n") == current.read_bytes().replace(b"\r\n", b"\n")
        source_checks[name] = dict(hash_matches=True, current_matches_after_lf_normalization=same)
        if not same:
            raise ValueError(f"Current source differs from recorded source: {name}")
    torch.set_num_threads(1)
    with np.load(PACKAGE / "assets/expression.npz", allow_pickle=False) as arrays:
        grid_x = torch.from_numpy(arrays["grid_x"].copy())
    cpu_checks, initialization_checks, rows, paired = [], [], [], []
    for run in manifest["runs"]:
        folder = source / run["key"]
        with np.load(folder / "predictions.npz", allow_pickle=False) as predictions:
            for row in run["checkpoints"]:
                saved = torch.load(folder / f"step_{row['step']}.pt", map_location="cpu", weights_only=True)
                model = MLP(2, saved["widths"], "relu")
                model.load_state_dict(saved["model"], strict=True)
                model.eval()
                with torch.inference_mode():
                    actual = model(grid_x).numpy()
                expected = predictions[str(row["step"])]
                np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=2e-5)
                cpu_checks.append(dict(key=run["key"], step=row["step"],
                    max_abs_difference=float(np.max(np.abs(actual - expected)))))
                if row["step"] == 0:
                    torch.manual_seed(run["seed"])
                    initial = MLP(2, saved["widths"], "relu")
                    differences = [float((initial.state_dict()[k] - value).abs().max())
                                   for k, value in saved["model"].items()]
                    # Windows/Linux kernels need not initialize to identical last bits.
                    tolerance = float(torch.finfo(torch.float32).eps)
                    if max(differences) > tolerance:
                        raise AssertionError("Initial weights differ beyond float32 precision")
                    initialization_checks.append(dict(key=run["key"],
                        max_abs_difference=max(differences), tolerance=tolerance,
                        bitwise_equal=max(differences) == 0))
                rows.append(dict(structure=run["structure"], seed=run["seed"], **row))
        first, last = run["checkpoints"][1:]
        paired.append(dict(structure=run["structure"], seed=run["seed"],
            grid_3000=first["grid_mse"], grid_9000=last["grid_mse"],
            change=last["grid_mse"] - first["grid_mse"],
            improvement_percent=100 * (first["grid_mse"] - last["grid_mse"]) / first["grid_mse"],
            max_error_3000=first["max_abs_error"], max_error_9000=last["max_abs_error"],
            train_3000=first["train_mse"], train_9000=last["train_mse"]))
    statistics = []
    for name in ("shallow", "deep"):
        for step in (3000, 9000):
            selected = [r for r in rows if r["structure"] == name and r["step"] == step]
            record = dict(structure=name, step=step, n=len(selected))
            for metric in ("grid_mse", "train_mse", "peak_mse", "background_mse", "max_abs_error", "elapsed_seconds"):
                values = np.array([r[metric] for r in selected])
                record[metric] = dict(median=float(np.median(values)), minimum=float(values.min()),
                    maximum=float(values.max()), mean=float(values.mean()))
            statistics.append(record)
    comparisons = []
    for step in (3000, 9000):
        shallow = next(r for r in statistics if r["structure"] == "shallow" and r["step"] == step)
        deep = next(r for r in statistics if r["structure"] == "deep" and r["step"] == step)
        comparisons.append(dict(step=step,
            median_ratio_shallow_over_deep=shallow["grid_mse"]["median"] / deep["grid_mse"]["median"],
            worst_deep_better_than_best_shallow=deep["grid_mse"]["maximum"] < shallow["grid_mse"]["minimum"]))
    summary = dict(passed=True, job_id=manifest["job_id"], commit=manifest["commit"],
        source=str(source), environment=manifest["environment"], elapsed_seconds=manifest["elapsed_seconds"],
        verified_artifacts=len(verification["artifacts_sha256"]), sources=source_checks,
        cpu_forward_checks=cpu_checks, initialization_checks=initialization_checks,
        statistics=statistics, paired_budgets=paired, comparisons=comparisons,
        original_sha256={k: original_hashes[k] for k in ("manifest.json", "verification.json", "summary.csv",
            "paired_budgets.png", "training.png")})
    save_json(output / "analysis.json", summary)
    plot_comparison(output, paired)
    with (output / "paired_changes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    for name in ("paired_budgets.png", "training.png"):
        shutil.copyfile(source / name, output / name)
    after = {str(p.relative_to(source)): digest(p) for p in source.rglob("*") if p.is_file()}
    if original_hashes != after:
        raise AssertionError("Original input artifacts changed")
    print(json.dumps(dict(passed=True, job_id=summary["job_id"], verified_artifacts=summary["verified_artifacts"],
        cpu_forward_max_difference=max(r["max_abs_difference"] for r in cpu_checks), comparisons=comparisons,
        paired_budgets=paired), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.input, args.output)
