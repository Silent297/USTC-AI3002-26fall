#!/usr/bin/env bash
#SBATCH --job-name=lab1-layer-train
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_default
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=6:00:00

# Submit from the repository root:
#   sbatch LABs/LAB1/scripts/run_layer_train.sbatch
#
# 按 experiments/train_layers.json 的超参数列表，对 world/truth 的 13 个隐表征层逐层训练：
# - world：z-score 标准化 + lr=0.002 + 2000 轮（λ2=1.0）
# - truth：原始特征 + lr=0.005 + 5000 轮
# - 任一 (task, layer) 发散时脚本自动把 lr 减半重试，见 train_layers.py
# - extra_runs 复现 world 历史最优（原始 L11, lr=5e-4, λ2=0.1）
# LAB1 只用 numpy/pandas，不需要 GPU；可用 TRAIN_CONFIG=... 切换配置文件。
#
# 日志集中存放：results/slurm_layers/<时间戳>_<jobid>/ 下含 train.log 与 .out/.err。

set -uo pipefail

# ---------------------------------------------------------------------------
# User-editable configuration
# ---------------------------------------------------------------------------

REPO_DIR="${HOME}/USTC-AI3002-26fall"
LAB_DIR="${REPO_DIR}/LABs/LAB1"

# 默认跑 train_layers.json；用 TRAIN_CONFIG=其他文件名 可切换
CONFIG="${LAB_DIR}/experiments/${TRAIN_CONFIG:-train_layers.json}"
RUNNER="${LAB_DIR}/experiments/train_layers.py"
OUTPUT_ROOT="${LAB_DIR}/results/slurm_layers"

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
run_root="${OUTPUT_ROOT}/${timestamp}_${SLURM_JOB_ID:-$$}"
run_log="${run_root}/train.log"
results_json="${run_root}/results.json"
summary_csv="${run_root}/summary.csv"

if ! mkdir -p "${run_root}"; then
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

echo "Repository: ${REPO_DIR}"
echo "Config: ${CONFIG}"
echo "Output root: ${run_root}"
echo

python "${RUNNER}" \
  --config "${CONFIG}" \
  --out-dir "${run_root}" 2>&1 | tee "${run_log}"
train_status="${PIPESTATUS[0]}"

echo
echo "Layer training complete."
echo "Results JSON: ${results_json}"
echo "Summary CSV: ${summary_csv}"
echo "Weights: ${run_root}/weights/"

if [[ "${train_status}" -ne 0 ]]; then
  echo "ERROR: train_layers.py exited with status ${train_status}" >&2
  exit "${train_status}"
fi

exit 0
