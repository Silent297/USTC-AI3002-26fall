"""P2 supplementary experiment: fixed ten-seed fits and read-only partition analysis.

Reuses the course pipeline without modifying submission.py or the original config.
Formal fitting is restricted to Slurm/A100. Analysis never refits or recomputes
CUDA quality metrics; it reads saved metrics and computes label-only ARI on CPU.
"""
import argparse
import csv
import itertools
import json
import os
import platform
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score

import extras
from main import run as course_run
from utils import data, results

PACKAGE = Path(__file__).resolve().parent
CONFIG = PACKAGE / "p2_config.json"
INPUT = PACKAGE / "data/features.npz"
INPUT_SHA256 = "a07578f6e1ab9d17667b987b25b1b9bc1f06a5f1f678a127e31dc78bc5ff0b38"
SOURCES = ("p2_robustness.py", "p2_config.json", "run_p2_T0016.sbatch",
           "submission.py", "main.py", "extras.py", "train.py", "evaluate.py",
           "utils/__init__.py", "utils/data.py", "utils/results.py", "utils/runtime.py")
METHODS = ("kmeans", "gmm")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def hashes(root):
    return {path.relative_to(root).as_posix(): results.sha256(path)
            for path in sorted(root.rglob("*")) if path.is_file()
            and path != root / "artifact_hashes.json"}


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def describe(values):
    valid = np.asarray([x for x in values if x is not None], dtype=np.float64)
    require(bool(np.isfinite(valid).all()), "nonfinite statistic")
    return dict(n=len(valid), complete=len(valid) == len(values),
                min=float(valid.min()) if len(valid) else None,
                median=float(np.median(valid)) if len(valid) else None,
                max=float(valid.max()) if len(valid) else None,
                mean=float(valid.mean()) if len(valid) else None,
                std=float(valid.std(ddof=1)) if len(valid) > 1 else None)


def collect(source):
    """Validate every candidate, not only the pipeline's exported winner."""
    recorded_hashes = read_json(source / "artifact_hashes.json")
    require(recorded_hashes == hashes(source), "artifact inventory/hash mismatch")
    meta = read_json(source / "p2_manifest.json")
    require(meta["status"] == "finished", "run did not finish normally")
    require(meta["source_hashes"] == {name: results.sha256(source / "source_snapshot" / name)
                                    for name in SOURCES}, "source snapshot mismatch")
    experiment = source / "experiment"
    report = read_json(experiment / "results.json")
    cfg = data.validate_config(read_json(experiment / "config.json"),
                               development=meta["development"])
    expected_cfg = data.validate_config(meta["configuration"], development=meta["development"])
    require(cfg == expected_cfg == report["config"], "configuration mismatch")
    require(read_json(experiment / "plan.json") == report["plan"], "plan mismatch")
    plan = report["plan"]
    require(plan["input_sha256"] == meta["input_sha256"], "input identity mismatch")
    require(plan["development"] == meta["development"], "development identity mismatch")
    require(plan["device"] == meta["device"], "device identity mismatch")
    require(plan["implementation_sha256"] == meta["source_hashes"]["submission.py"],
            "implementation identity mismatch")
    for name in ("config.json", "fit_config.json"):
        field = "config_sha256" if name == "config.json" else "fit_config_sha256"
        require(results.sha256(experiment / name) == plan[field], f"{name} mismatch")
    require(read_json(experiment / "fit_config.json") == cfg, "fit config differs")
    if not meta["development"]:
        require(meta["input_sha256"] == INPUT_SHA256 and meta["device"] == "cuda",
                "formal input/device differs from P2 protocol")
        require(cfg == data.load_config(source / "source_snapshot/p2_config.json"),
                "formal P2 config differs from source snapshot")
    rows = read_json(experiment / "runs.json")
    require(rows == report["runs"], "runs.json/results.json mismatch")
    expected = {(m, k, s) for m in METHODS for k in cfg["candidate_k"] for s in cfg["seeds"]}
    observed = [(r["method"], r["k"], r["seed"]) for r in rows]
    require(len(observed) == len(expected) and set(observed) == expected,
            "missing, duplicate or unexpected planned candidate")
    n_samples = meta["n_samples"]
    indices = np.load(experiment / "diagnostic_indices.npy", allow_pickle=False)
    expected_indices = np.sort(np.random.default_rng(cfg["silhouette_seed"]).choice(
        n_samples, cfg["silhouette_sample_size"], replace=False))
    require(np.array_equal(indices, expected_indices), "Silhouette subset mismatch")
    partitions, restart_rows, candidate_rows = {}, [], []
    for row in rows:
        key = f'{row["method"]}_k{row["k"]}_seed{row["seed"]}'
        require(row["key"] == key, "invalid candidate key")
        folder = experiment / "candidates" / key
        fit = read_json(folder / "fit.json")
        require(all(row.get(k) == v for k, v in fit["run"].items()), f"{key}: fit record mismatch")
        require(len(fit["restarts"]) == cfg["n_init"], f"{key}: missing restart records")
        require([x["restart"] for x in fit["restarts"]] == list(range(cfg["n_init"])),
                f"{key}: restart identity mismatch")
        for restart in fit["restarts"]:
            restart_rows.append(dict(key=key, method=row["method"], k=row["k"], seed=row["seed"],
                **{name: restart.get(name) for name in ("restart", "restart_seed", "n_iter",
                   "converged", "stop_reason", "objective", "likelihood_decreases", "error")}))
        item = {name: row.get(name) for name in ("key", "method", "k", "seed", "status",
                "actual_k", "converged", "stop_reason", "best_restart", "n_iter", "elapsed_seconds", "error")}
        if row["status"] == "succeeded":
            for filename, field in (("clusters.npy", "clusters_sha256"), ("model.npz", "model_sha256")):
                require(results.sha256(folder / filename) == row[field], f"{key}: {filename} hash mismatch")
            labels = np.load(folder / "clusters.npy", allow_pickle=False)
            require(labels.shape == (n_samples,) and labels.dtype == np.int32 and
                    bool((labels >= 0).all()) and bool((labels < row["k"]).all()),
                    f"{key}: invalid labels")
            ids, sizes = np.unique(labels, return_counts=True)
            require(len(ids) == row["actual_k"] and sizes.tolist() == row["cluster_sizes"]
                    and ids.tolist() == row["cluster_ids"], f"{key}: cluster sizes mismatch")
            partitions[key] = labels
            item.update(min_size=int(sizes.min()), max_size=int(sizes.max()),
                        cluster_sizes=json.dumps(sizes.tolist()))
        for metric in data.METRICS:
            saved = row.get("metrics", {}).get(metric, {})
            value = row.get(metric)
            require((saved.get("status") == "ok" and value is not None and
                     np.isfinite(value) and value == saved.get("value")) or
                    (saved.get("status") != "ok" and value is None), f"{key}: {metric} record invalid")
            item[metric] = value
            item[metric + "_status"] = saved.get("status", "fit_failed")
        candidate_rows.append(item)
    return meta, cfg, report, rows, partitions, restart_rows, candidate_rows


def analyze(source, output):
    source = Path(source).resolve()
    meta, cfg, report, rows, partitions, restarts, candidates = collect(source)
    output = Path(output).resolve()
    require(output != source and source not in output.parents, "analysis output must be outside source")
    output = results.create_output(output)
    grouped, ari_pairs, cross_pairs, winners, paired = [], [], [], [], []
    by_identity = {(r["method"], r["k"], r["seed"]): r for r in rows}
    for method in METHODS:
        for k in cfg["candidate_k"]:
            members = [by_identity[method, k, s] for s in cfg["seeds"]]
            group_pairs = []
            for a, b in itertools.combinations(members, 2):
                value = (float(adjusted_rand_score(partitions[a["key"]], partitions[b["key"]]))
                         if a["key"] in partitions and b["key"] in partitions else None)
                record = dict(method=method, k=k, seed_a=a["seed"], seed_b=b["seed"], ari=value)
                ari_pairs.append(record)
                group_pairs.append(value)
            grouped.append(dict(method=method, k=k, planned=len(members),
                successful=sum(r["status"] == "succeeded" for r in members),
                selected_nonconverged=sum(r["status"] == "succeeded" and not r["converged"] for r in members),
                failed_seeds=[r["seed"] for r in members if r["status"] != "succeeded"],
                actual_k=[r.get("actual_k") for r in members],
                metrics={name: describe([r.get(name) for r in members]) for name in data.METRICS},
                ari=describe(group_pairs)))
        for seed in cfg["seeds"]:
            members = [by_identity[method, k, seed] for k in cfg["candidate_k"]]
            complete = all(r.get("silhouette") is not None for r in members)
            best = max(r["silhouette"] for r in members) if complete else None
            tied = [r["k"] for r in members if abs(r["silhouette"] - best) <= 1e-6] if complete else []
            winners.append(dict(method=method, seed=seed, eligible=complete,
                tied_k=json.dumps(tied), chosen_k=min(tied) if tied else None,
                **{f'k{r["k"]}': r.get("silhouette") for r in members}))
    for k in cfg["candidate_k"]:
        for seed in cfg["seeds"]:
            a, b = (by_identity[m, k, seed] for m in METHODS)
            cross_pairs.append(dict(k=k, seed=seed, ari=(float(adjusted_rand_score(
                partitions[a["key"]], partitions[b["key"]]))
                if a["key"] in partitions and b["key"] in partitions else None)))
            paired.append(dict(k=k, seed=seed, **{metric + "_kmeans_minus_gmm":
                a[metric] - b[metric] if a.get(metric) is not None and b.get(metric) is not None else None
                for metric in data.METRICS}))
    all_success = all(r["status"] == "succeeded" and all(r.get(m) is not None for m in data.METRICS) for r in rows)
    summary = dict(development=meta["development"], all_candidates_and_metrics_ok=all_success,
        planned_candidates=len(rows), planned_restarts=len(rows) * cfg["n_init"],
        successful_candidates=len(partitions), failed_restarts=sum(r["stop_reason"] == "failed" for r in restarts),
        nonconverged_restarts=sum(r["stop_reason"] != "failed" and not r["converged"] for r in restarts),
        configuration=cfg, groups=grouped, pipeline_selection=report["selection"],
        k_win_counts={method: {str(k): sum(r["method"] == method and r["chosen_k"] == k for r in winners)
                              for k in cfg["candidate_k"]} for method in METHODS},
        source=str(source), source_hashes_sha256=results.sha256(source / "artifact_hashes.json"),
        caveats=["45 ARI pairs share 10 fitted runs, not 45 independent trials.",
                 "n_init=3 is internal restart selection, not three external seeds.",
                 "Only the K=8/10/12 neighborhood is tested in formal P2.",
                 "ARI stability and internal metrics do not prove hidden-label accuracy.",
                 "Recorded fitting times are not a controlled runtime benchmark.",
                 "Positive KMeans-minus-GMM favors KMeans for silhouette/CH; negative favors it for DB.",
                 "Analysis reads saved quality metrics; it does not recompute GPU metrics.",
                 "The pipeline export is experimental and does not replace the original final labels."])
    for name, records in (("candidates", candidates), ("restarts", restarts),
                          ("within_method_ari", ari_pairs), ("cross_method_ari", cross_pairs),
                          ("per_seed_k", winners), ("paired_method_metrics", paired)):
        write_csv(output / (name + ".csv"), records)
    results.save_json(output / "summary.json", summary)
    lines = ["# P2 machine summary", "", f"Development: {meta['development']}",
             f"Candidates: {len(partitions)}/{len(rows)}; all metrics valid: {all_success}", "",
             "| Method | K | Success | Silhouette median (min, max) | ARI min / median / max |", "|---|---|---|---|---|"]
    for group in grouped:
        score, ari = group["metrics"]["silhouette"], group["ari"]
        lines.append(f"| {group['method']} | {group['k']} | {group['successful']}/{group['planned']} | "
                     f"{score['median']} ({score['min']}, {score['max']}) | {ari['min']} / {ari['median']} / {ari['max']} |")
    lines += ["", *summary["caveats"]]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    require(read_json(source / "artifact_hashes.json") == hashes(source), "analysis modified source")
    print(json.dumps(dict(output=str(output), successful=len(partitions), planned=len(rows),
                          all_metrics_ok=all_success), indent=2))
    return summary


def execute(output, *, development=False, input_path=INPUT, config_path=CONFIG):
    if not development:
        require(bool(os.environ.get("SLURM_JOB_ID")), "formal run requires a Slurm job")
        require(torch.cuda.is_available(), "formal run requires CUDA; no CPU fallback")
        require("A100" in torch.cuda.get_device_name(0), "formal run requires A100")
        require(results.sha256(input_path) == INPUT_SHA256, "formal feature hash changed")
        require(data.load_config(config_path) == data.load_config(CONFIG), "formal P2 config changed")
    cfg = data.load_config(config_path, development=development)
    device = "cpu" if development else "cuda"
    output = results.create_output(output)
    snapshot = output / "source_snapshot"
    for name in SOURCES:
        destination = snapshot / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PACKAGE / name, destination)
    source_hashes = {name: results.sha256(PACKAGE / name) for name in SOURCES}
    meta = dict(task="T0016", development=development, device=device, configuration=cfg,
        input_sha256=results.sha256(input_path),
        n_samples=90 if development else 56000,
        slurm_job_id=os.environ.get("SLURM_JOB_ID"), python=platform.python_version(),
        torch=torch.__version__, numpy=np.__version__, source_hashes=source_hashes, status="running")
    results.save_json(output / "p2_manifest.json", meta)
    try:
        code = course_run(config_path, output / "experiment", device, input_path=input_path,
                          implementation_path=snapshot / "submission.py", development=development)
        experiment = output / "experiment"
        if (experiment / "clusters.npy").exists():
            audit = extras.check_output(SimpleNamespace(output=experiment, input=input_path, development=development))
            results.save_json(output / "export_check.json", audit)
        require(source_hashes == {name: results.sha256(PACKAGE / name) for name in SOURCES}, "source changed during run")
        meta.update(status="finished", pipeline_exit_code=code)
    except Exception as error:
        meta.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        results.save_json(output / "p2_manifest.json", meta)
        results.save_json(output / "artifact_hashes.json", hashes(output))
    return code


def smoke(output):
    output = results.create_output(output)
    rng = np.random.default_rng(2311)
    X = np.concatenate([rng.normal((c, 0), .15, (30, 2)) for c in (-5, 0, 5)]).astype(np.float32)
    input_path = output / "toy.npz"
    np.savez(input_path, X=X)
    cfg = data.load_config(CONFIG)
    cfg.update(candidate_k=[2, 3], seeds=[17, 42], n_init=2,
               kmeans_max_iter=30, gmm_max_iter=50, silhouette_sample_size=60, threads=1)
    config_path = output / "toy_config.json"
    results.save_json(config_path, cfg)
    require(execute(output / "run", development=True, input_path=input_path, config_path=config_path) == 0, "toy pipeline failed")
    summary = analyze(output / "run", output / "analysis")
    require(summary["all_candidates_and_metrics_ok"] and summary["planned_candidates"] == 8
            and summary["planned_restarts"] == 16, "toy candidate/restart coverage failed")
    labels = np.array([0, 0, 1, 1, 2, 2])
    require(adjusted_rand_score(labels, (labels + 1) % 3) == 1, "ARI label permutation failed")
    require(adjusted_rand_score(labels, [0, 1, 0, 1, 0, 1]) < 0, "ARI distinct partition failed")
    # Keep damaged fixtures in this ignored smoke directory for inspection.
    damaged = output / "damaged_fixture"
    shutil.copytree(output / "run", damaged)
    target = next((damaged / "experiment/candidates").glob("*/clusters.npy"))
    with target.open("ab") as stream:
        stream.write(b"corrupted")
    try:
        collect(damaged)
    except ValueError as error:
        require("hash mismatch" in str(error), "wrong corruption rejection")
    else:
        raise AssertionError("tampering accepted")
    shutil.copyfile(output / "run" / target.relative_to(damaged), target)
    records = read_json(damaged / "experiment/runs.json")[:-1]
    results.save_json(damaged / "experiment/runs.json", records)
    report = read_json(damaged / "experiment/results.json")
    report["runs"] = records
    results.save_json(damaged / "experiment/results.json", report)
    results.save_json(damaged / "artifact_hashes.json", hashes(damaged))
    try:
        collect(damaged)
    except ValueError as error:
        require("planned candidate" in str(error), "wrong missing-seed rejection")
    else:
        raise AssertionError("missing candidate accepted")
    print("SMOKE PASSED: 8 toy candidates / 16 restarts; export/provenance; ARI; tamper and missing-seed rejection")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "smoke", "analyze"):
        command = sub.add_parser(name)
        command.add_argument("--output", type=Path, required=True)
        if name == "analyze":
            command.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "smoke":
        smoke(args.output)
        return 0
    if args.command == "analyze":
        return 0 if analyze(args.source, args.output)["all_candidates_and_metrics_ok"] else 1
    return execute(args.output)


if __name__ == "__main__":
    raise SystemExit(main())
