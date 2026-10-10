#!/usr/bin/env bash
#SBATCH --job-name=lab1-hparam-sweep
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_default
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00

# Submit from the repository root:
#   sbatch LABs/LAB1/scripts/run_hyperparam_sweep.sbatch
#
# This script follows experiments/超参数实验说明.md:
# - grouped hyperparameters live in experiments/hparam_grid.json
# - experiments/run_grid.py runs one controlled variable per group
# - LAB1 only uses numpy/pandas, so this job does not request a GPU.
# 日志集中存放：results/slurm_grid/<时间戳>_<jobid>/ 下含 grid.log 与 .out/.err。

set -uo pipefail

# ---------------------------------------------------------------------------
# User-editable configuration
# ---------------------------------------------------------------------------

REPO_DIR="${HOME}/USTC-AI3002-26fall"
LAB_DIR="${REPO_DIR}/LABs/LAB1"

# 默认跑 hparam_grid.json；用 GRID_CONFIG=hparam_grid_std.json 可切换到其他配置
CONFIG="${LAB_DIR}/experiments/${GRID_CONFIG:-hparam_grid.json}"
RUNNER="${LAB_DIR}/experiments/run_grid.py"
OUTPUT_ROOT="${LAB_DIR}/results/slurm_grid"

# Leave empty to run all groups. To run selected groups:
#   GROUPS_TO_RUN="truth_lr,world_lr"
GROUPS_TO_RUN=""

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

if [[ ! -d "${LAB_DIR}" ]]; then
  echo "ERROR: LAB_DIR does not exist: ${LAB_DIR}" >&2
  exit 1
fi

if [[ ! -f "${CONFIG}" ]]; then
  echo "ERROR: config does not exist: ${CONFIG}" >&2
  exit 1
fi

if [[ ! -f "${RUNNER}" ]]; then
  echo "ERROR: runner does not exist: ${RUNNER}" >&2
  exit 1
fi

CONDA_SH="${HOME}/miniconda3/etc/profile.d/conda.sh"
if [[ ! -f "${CONDA_SH}" ]]; then
  echo "ERROR: conda init script not found: ${CONDA_SH}" >&2
  exit 1
fi

# shellcheck source=/dev/null
set +u
source "${CONDA_SH}"
conda activate ai26 || {
  echo "ERROR: failed to activate conda environment ai26" >&2
  exit 1
}
set -u

export PYTHONIOENCODING="utf-8"

timestamp="$(date +%Y%m%d_%H%M%S)"
run_name="${timestamp}_${SLURM_JOB_ID:-$$}"
run_root="${OUTPUT_ROOT}/${run_name}"
weights_dir="${run_root}/weights"
results_json="${run_root}/results.json"
summary_csv="${run_root}/summary.csv"
final_prompt="${run_root}/final_prompt.md"
grid_log="${run_root}/grid.log"

if ! mkdir -p "${weights_dir}"; then
  echo "ERROR: failed to create output directory: ${run_root}" >&2
  exit 1
fi

# Slurm 的 %x_%j.out/.err 默认落在「提交目录」，随提交位置漂移；
# 建好本次运行目录后立刻把它们移进来，使所有日志集中一处
# （重命名不影响已打开的 fd，后续输出仍会写进本文件）。
if [[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  for stream in out err; do
    slurm_stream="${SLURM_SUBMIT_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.${stream}"
    if [[ -f "${slurm_stream}" ]]; then
      mv "${slurm_stream}" "${run_root}/" || echo "WARNING: failed to archive ${slurm_stream}" >&2
    fi
  done
fi

cd "${LAB_DIR}"

grid_cmd=(
  python experiments/run_grid.py
  --config "${CONFIG}"
  --out "${results_json}"
  --artifacts-dir "${weights_dir}"
)

if [[ -n "${GROUPS_TO_RUN}" ]]; then
  grid_cmd+=(--groups "${GROUPS_TO_RUN}")
fi

echo "Repository: ${REPO_DIR}"
echo "Experiment: ${LAB_DIR}"
echo "Config: ${CONFIG}"
echo "Output root: ${run_root}"
if [[ -n "${GROUPS_TO_RUN}" ]]; then
  echo "Groups: ${GROUPS_TO_RUN}"
else
  echo "Groups: all"
fi
echo

"${grid_cmd[@]}" 2>&1 | tee "${grid_log}"
grid_status="${PIPESTATUS[0]}"

RESULTS_JSON="${results_json}" \
SUMMARY_CSV="${summary_csv}" \
FINAL_PROMPT="${final_prompt}" \
GRID_LOG="${grid_log}" \
python - <<'PY'
import csv
import json
import os
from pathlib import Path

results_path = Path(os.environ["RESULTS_JSON"])
summary_path = Path(os.environ["SUMMARY_CSV"])
prompt_path = Path(os.environ["FINAL_PROMPT"])
grid_log = os.environ["GRID_LOG"]

fieldnames = [
    "group",
    "task",
    "layer",
    "vary",
    "value",
    "lr",
    "epochs",
    "lambda1",
    "lambda2",
    "metric_name",
    "metric",
    "finite",
    "seconds",
    "weights_path",
    "grid_log",
]

if results_path.exists():
    results = json.loads(results_path.read_text(encoding="utf-8"))
else:
    results = []

rows = []
for record in results:
    vary = record.get("vary", "")
    rows.append(
        {
            "group": record.get("group", ""),
            "task": record.get("task", ""),
            "layer": record.get("layer", ""),
            "vary": vary,
            "value": record.get(vary, ""),
            "lr": record.get("lr", ""),
            "epochs": record.get("epochs", ""),
            "lambda1": record.get("lambda1", ""),
            "lambda2": record.get("lambda2", ""),
            "metric_name": record.get("metric_name", ""),
            "metric": record.get("metric", ""),
            "finite": record.get("finite", ""),
            "seconds": record.get("seconds", ""),
            "weights_path": record.get("weights_path", ""),
            "grid_log": grid_log,
        }
    )

with summary_path.open("w", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

best_by_task = {}
for row in rows:
    if not row["finite"] or row["metric"] == "":
        continue
    task = row["task"]
    metric = float(row["metric"])
    current = best_by_task.get(task)
    if current is None:
        best_by_task[task] = row
    elif row["metric_name"] == "mse":
        if metric < float(current["metric"]):
            best_by_task[task] = row
    elif row["metric_name"] == "accuracy":
        if metric > float(current["metric"]):
            best_by_task[task] = row

lines = [
    "# LAB1 Slurm Hyperparameter Grid Summary",
    "",
    "请根据下面这次 LAB1 分组超参数实验结果，分析学习率、训练轮数、L1/L2 正则化和隐表征层数分别对 truth / world 任务的影响，说明你推荐的最佳配置及理由，并给出下一步小范围精调建议。",
    "",
    "## Sweep",
    "",
    f"- Total runs: {len(rows)}",
    f"- Results JSON: {results_path}",
    f"- Summary CSV: {summary_path}",
    f"- Slurm grid log: {grid_log}",
    "",
    "## Best By Task",
    "",
]

for task in ("truth", "world"):
    best = best_by_task.get(task)
    if best is None:
        lines.append(f"- {task}: no finite result with a metric")
        continue
    lines.extend(
        [
            f"- {task}: group={best['group']}, varied {best['vary']}={best['value']}, "
            f"layer={best['layer']}, epochs={best['epochs']}, "
            f"lambda1={best['lambda1']}, lambda2={best['lambda2']}, "
            f"{best['metric_name']}={best['metric']}",
            f"  weights: {best['weights_path']}",
        ]
    )
    if best["vary"] != "lr":
        lines.insert(-2, f"  lr={best['lr']}")

lines.extend(["", "## All Runs", "", "```csv"])
if rows:
    with summary_path.open(encoding="utf-8", newline="") as stream:
        lines.append(stream.read().rstrip())
else:
    lines.append("No results were produced.")
lines.extend(["```", ""])

prompt_path.write_text("\n".join(lines), encoding="utf-8")
PY
summary_status=$?

echo
echo "Grid complete."
echo "Results JSON: ${results_json}"
echo "Summary CSV: ${summary_csv}"
echo "Final prompt: ${final_prompt}"
echo "Grid log: ${grid_log}"

if [[ "${grid_status}" -ne 0 ]]; then
  echo "ERROR: run_grid.py exited with status ${grid_status}" >&2
  exit "${grid_status}"
fi

if [[ "${summary_status}" -ne 0 ]]; then
  echo "ERROR: failed to generate final summary/prompt" >&2
  exit "${summary_status}"
fi

exit 0
