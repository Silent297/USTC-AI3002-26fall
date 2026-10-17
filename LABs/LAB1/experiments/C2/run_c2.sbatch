#!/usr/bin/env bash
#SBATCH --job-name=lab1-c2
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_default
# Students 分区混部 RTX5090 / A100；5090 节点上 ai26 的 torch cu130 会触发 CUDA error 804，钉死 A100。
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00

# 提交（仓库根目录）：
#   sbatch LABs/LAB1/experiments/C2/run_c2.sbatch
#
# C2 = Prompt 变体特征提取 + 探针训练：
#   1) GPU：对 5 个变体 × train/valid 提取 13 层激活（单进程一次前向取全部层；
#      origcountry 与 datasets/world 现有特征做逐层余弦自检，< 0.999 中止）；
#   2) CPU（同节点）：5 变体 × (std13 网格 + raw11 extra) 训练探针，逐 (variant,set,layer)
#      保存权重与逐条 valid 预测；origcountry 的 14 个 run 与 world 历史 MSE 锚点对拍
#      （容差 0.5），不符则作业失败。
#
# 产物：
#   experiments/C2/features/<variant>/<split>.layer<L>.npy   （本地特征，不入 git）
#   results/slurm_c2/<RUN_TAG>/c2_metrics.json / summary.csv / weights/
# 可重复提交：已存在的特征与权重会跳过（断点续跑）。RUN_TAG 默认 c2_v1。

set -uo pipefail

REPO_DIR="${HOME}/USTC-AI3002-26fall"
LAB_DIR="${REPO_DIR}/LABs/LAB1"
MODEL_DIR="${MODEL_DIR:-${HOME}/models/Qwen3-4B-Base}"
RUN_TAG="${RUN_TAG:-c2_v1}"

LOG_ROOT="${LAB_DIR}/results/slurm_c2"

if [[ ! -d "${LAB_DIR}" ]]; then
  echo "ERROR: LAB_DIR does not exist: ${LAB_DIR}" >&2
  exit 1
fi
if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "ERROR: MODEL_DIR does not exist: ${MODEL_DIR}" >&2
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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

run_root="${LOG_ROOT}/${RUN_TAG}"
if ! mkdir -p "${run_root}"; then
  echo "ERROR: failed to create log directory: ${run_root}" >&2
  exit 1
fi

# Slurm 默认把 %x_%j.out/.err 落在提交目录；挪进本次运行目录（已打开 fd 不受影响）
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
echo "Model: ${MODEL_DIR}"
echo "Run tag: ${RUN_TAG}"
echo "Output: ${run_root}"
echo

echo "==== GPU environment ===="
hostname
nvidia-smi || echo "WARNING: nvidia-smi failed"
if ! python - <<'PY'
import torch

print(f"torch {torch.__version__} (built for CUDA {torch.version.cuda})")
if not torch.cuda.is_available():
    raise SystemExit("torch cannot see a CUDA GPU")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
PY
then
  echo "ERROR: no usable CUDA GPU; see README appendix B.3" >&2
  exit 2
fi
echo

# ---------------------------------------------------------------------------
# Step 1: 变体特征提取（GPU）
# ---------------------------------------------------------------------------
echo "==== Step 1: extract variant features ===="
python -u experiments/C2/c2_extract_variants.py \
  --model_dir "${MODEL_DIR}" \
  --data_dir experiments/C2/data \
  --out_dir experiments/C2/features 2>&1 | tee -a "${run_root}/extract.log"
extract_status="${PIPESTATUS[0]}"
if [[ "${extract_status}" -ne 0 ]]; then
  echo "ERROR: feature extraction failed (exit ${extract_status})" >&2
  exit "${extract_status}"
fi

# ---------------------------------------------------------------------------
# Step 2: 探针训练 + 锚点对拍（CPU，同节点）
# ---------------------------------------------------------------------------
echo
echo "==== Step 2: train probes ===="
python -u experiments/C2/c2_train_probes.py \
  --config experiments/C2/c2_probe_config.json \
  --features_root experiments/C2/features \
  --out_dir "${run_root}" \
  --skip_existing 2>&1 | tee -a "${run_root}/train.log"
train_status="${PIPESTATUS[0]}"
if [[ "${train_status}" -ne 0 ]]; then
  echo "ERROR: probe training / anchor check failed (exit ${train_status})" >&2
  exit "${train_status}"
fi

echo
echo "Done."
echo "Metrics: ${run_root}/c2_metrics.json"
echo "Summary: ${run_root}/summary.csv"
echo "Weights: ${run_root}/weights/"
echo "Features (untracked): experiments/C2/features/"

# 收尾自检：文件数量与可读性（特征 5×2×13 = 130 个 npy；权重 5×14 = 70 个 npz）
RUN_ROOT="${run_root}" python - <<'PY'
import os
from pathlib import Path
import numpy as np

feat = sorted(Path("experiments/C2/features").glob("*/*.layer*.npy"))
w = sorted(Path(os.environ["RUN_ROOT"]).glob("weights/*/weights.npz"))
print(f"feature files: {len(feat)}, weight files: {len(w)}")
bad = 0
if len(feat) != 5 * 2 * 13 or len(w) != 5 * 14:
    raise SystemExit("unexpected file counts (expected 130 features, 70 weights)")
for path in feat:
    arr = np.load(path, allow_pickle=False)
    if arr.shape[1] != 2560 or arr.dtype != np.float16 or not np.isfinite(arr).all():
        print("BAD", path); bad += 1
for path in w:
    saved = np.load(path)
    if saved["weights"].shape != (2, 2560) or not np.isfinite(saved["weights"]).all():
        print("BAD", path); bad += 1
if bad:
    raise SystemExit(f"{bad} bad files")
print("final validation: OK")
PY
final_status=$?
if [[ "${final_status}" -ne 0 ]]; then
  echo "ERROR: final validation failed" >&2
  exit "${final_status}"
fi
exit 0
