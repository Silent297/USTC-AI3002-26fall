"""T0019: read-only verification and analysis of retrieved P1 trajectories."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import p1_learning_rate as p1


def inventory(root):
    return {p.relative_to(root).as_posix(): p1.sha256(p)
            for p in sorted(root.rglob("*")) if p.is_file()}


def history(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return [{k: float(v) if v else None for k, v in r.items()} for r in csv.DictReader(stream)]


def csv_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def run(source, output):
    source, output = source.resolve(), output.resolve()
    p1.require(source != output and source not in output.parents, "output must be outside source")
    before = inventory(source)
    output = p1.empty_output(output)
    p1.check(source / "p1", output / "check")
    checked = p1.read_json(output / "check/verification.json")
    cluster_check = p1.read_json(source / "p1_check/verification.json")
    for key in ("passed", "development", "trajectories", "source_unchanged", "hashed_files", "optimizer_updates"):
        p1.require(checked[key] == cluster_check[key], f"cluster/local check mismatch: {key}")
    np.testing.assert_allclose(checked["gradient_max_abs_error"], cluster_check["gradient_max_abs_error"],
                               rtol=1e-8, atol=1e-12)
    for a, b in zip(checked["pairs"], cluster_check["pairs"], strict=True):
        p1.require(a.keys() == b.keys(), "cluster/local pair fields mismatch")
        for key in a:
            np.testing.assert_allclose(a[key], b[key], rtol=1e-10, atol=1e-12)
    meta = p1.read_json(source / "p1/manifest.json")
    cfg = p1.read_json(source / "p1/config.json")
    p1.require(not meta["development"] and len(meta["runs"]) == 6, "expected six formal runs")
    normalized = {}
    for name in p1.SOURCES:
        normalized[name] = ((source / "p1/source_snapshot" / name).read_bytes().replace(b"\r\n", b"\n") ==
                            (p1.PACKAGE / name).read_bytes().replace(b"\r\n", b"\n"))
    p1.require(all(normalized.values()), "source snapshot differs from local code")
    ref = np.array(checked["reference"]["least_squares"])
    rows, loaded, snapshots = [], {}, []
    for record in meta["runs"]:
        h = history(source / "p1" / record["key"] / "history.csv")
        loaded[record["key"]] = h
        a, b = h[0], h[-1]
        rows.append(dict(key=record["key"], seed=record["seed"], learning_rate=record["learning_rate"],
            initial_train_mse=a["train_mse"], final_train_mse=b["train_mse"],
            initial_grid_mse=a["grid_mse"], final_grid_mse=b["grid_mse"],
            train_mse_reduction_percent=100 * (1 - b["train_mse"] / a["train_mse"]),
            initial_gradient_norm=a["gradient_norm"], final_gradient_norm=b["gradient_norm"],
            gradient_retention=b["gradient_norm"] / a["gradient_norm"],
            initial_actual_update=a["actual_update_norm"], last_actual_update=h[-2]["actual_update_norm"],
            zero_updates=sum(r["actual_update_norm"] == 0 for r in h[:-1]),
            parameter_distance_to_ls=float(np.linalg.norm(np.array([b[n] for n in ("w1", "w2", "bias")]) - ref)),
            final_w1=b["w1"], final_w2=b["w2"], final_bias=b["bias"]))
        for step in (0, 1, 10, 100, 1000):
            snapshots.append(dict(key=record["key"], learning_rate=record["learning_rate"], **h[step]))
    csv_rows(output / "trajectories.csv", rows)
    csv_rows(output / "selected_steps.csv", snapshots)
    colors = dict(zip(cfg["seeds"], ("#1766a3", "#ce6815", "#288049")))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    for record in meta["runs"]:
        h = loaded[record["key"]]
        style = "-" if record["key"].startswith("normal") else "--"
        label = f"seed {record['seed']}, lr={record['learning_rate']:g}"
        for ax, field, title in zip(axes, ("train_mse", "gradient_norm", "actual_update_norm"),
                                    ("Training MSE", "Parameter gradient L2 norm", "Actual parameter update L2 norm")):
            points = [(r["step"], r[field]) for r in h if r[field] is not None and r[field] > 0]
            ax.semilogy([x for x, _ in points], [y for _, y in points], style,
                        color=colors[record["seed"]], label=label, linewidth=1.3)
            ax.set(xlabel="Completed SGD updates", title=title)
            ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    axes[2].text(.02, .03, "Terminal row: no update; no zero updates omitted", transform=axes[2].transAxes, fontsize=7)
    p1.require(all(row["zero_updates"] == 0 for row in rows), "update plot annotation must disclose zeros")
    fig.suptitle(f"P1 / Job{meta['job_id']}: affine model, 3 paired seeds; solid lr=0.05, dashed lr=1e-5")
    fig.tight_layout()
    fig.savefig(output / "p1_dynamics.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for seed in cfg["seeds"]:
        h = loaded[f"small_seed{seed}"]
        x = [r["step"] for r in h]
        for ax, field, title in zip(axes, ("train_mse", "gradient_norm"),
                                    ("Small lr: MSE / initial MSE", "Small lr: gradient / initial gradient")):
            ax.plot(x, [r[field] / h[0][field] for r in h], color=colors[seed], label=f"seed {seed}")
            ax.set(xlabel="Completed SGD updates", title=title, ylim=(.975, 1.002))
            ax.grid(alpha=.2)
            ax.legend()
    fig.suptitle("P1 small-learning-rate group: linear-scale zoom (not starting at zero)")
    fig.tight_layout()
    fig.savefig(output / "p1_small_lr.png", dpi=180)
    plt.close(fig)
    p1.require(before == inventory(source), "original files changed during analysis")
    eigenvalues = checked["reference"]["hessian_eigenvalues"]
    p1.save_json(output / "analysis.json", dict(job_id=meta["job_id"], environment=meta["environment"],
        formal_run_commit=meta["commit"], original_files=len(before), original_files_unchanged=True,
        hashed_artifacts=checked["hashed_files"], source_snapshots_equal_after_newline_normalization=normalized,
        gradient_max_abs_error=checked["gradient_max_abs_error"], trajectories=rows,
        pairs=checked["pairs"], reference=checked["reference"],
        hessian_condition=float(max(eigenvalues) / min(eigenvalues)),
        stable_learning_rate_upper_bound=2 / max(eigenvalues), optimizer_updates=0,
        note="All curves read saved histories. No fitting or GPU execution."))
    print("ANALYSIS PASSED", "job", meta["job_id"], "unchanged files", len(before), "output", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.output)
