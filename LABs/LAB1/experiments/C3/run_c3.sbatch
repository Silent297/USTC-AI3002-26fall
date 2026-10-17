#!/usr/bin/env bash
#SBATCH --job-name=lab1-c3
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_default
# Students 分区混部 RTX5090 / A100；5090 节点上 ai26 的 torch cu130 会触发 CUDA error 804，钉死 A100。
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00

# 提交（仓库根目录，先 git pull 拿到 C3 的 CSV）：
#   sbatch LABs/LAB1/experiments/C3/run_c3.sbatch
#
# C3 = GeoNames 数据扩增：
#   1) GPU：对 12000 条 GeoNames prompt 提取 13 层激活（4 个 split，约 800MB，
#      不入 git）；同时复提 datasets/world train/valid 做逐层余弦自检；
#   2) CPU（同节点）：5 个训练集 × 13 层训练 world 探针，评估 4 个集合：
#      orig_valid / new_valid / new_test / new_unseen；orig 的 13 层 MSE 与
#      world 历史 std13 锚点对拍（容差 0.5）。
#
# 产物：
#   experiments/C3/features/                         （本地特征，不入 git）
#   results/slurm_c3/<RUN_TAG>/c3_metrics.json / summary.csv / weights/ / records/
# 可重复提交：已存在的特征、per-run record 会跳过（断点续跑）。RUN_TAG 默认 c3_v1。

set -uo pipefail

REPO_DIR="${HOME}/USTC-AI3002-26fall"
LAB_DIR="${REPO_DIR}/LABs/LAB1"
MODEL_DIR="${MODEL_DIR:-${HOME}/models/Qwen3-4B-Base}"
RUN_TAG="${RUN_TAG:-c3_v1}"

LOG_ROOT="${LAB_DIR}/results/slurm_c3"

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
# Step 1: 特征提取（GPU）
# ---------------------------------------------------------------------------
echo "==== Step 1: extract features ===="
python -u experiments/C3/c3_extract.py \
  --model_dir "${MODEL_DIR}" \
  --data_dir experiments/C3/data/splits \
  --out_dir experiments/C3/features 2>&1 | tee -a "${run_root}/extract.log"
extract_status="${PIPESTATUS[0]}"
if [[ "${extract_status}" -ne 0 ]]; then
  echo "ERROR: feature extraction failed (exit ${extract_status})" >&2
  exit "${extract_status}"
fi

# ---------------------------------------------------------------------------
# Step 2: 探针训练 + 锚点对拍（CPU，同节点）
# ---------------------------------------------------------------------------
# 4 个 worker × 1 个 BLAS 线程；实测比 1 进程 × 4 线程吞吐更好。
echo
echo "==== Step 2: train probes ===="
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
python -u experiments/C3/c3_train_probes.py \
  --config experiments/C3/c3_config.json \
  --out_dir "${run_root}" \
  --workers 4 \
  --skip_existing 2>&1 | tee -a "${run_root}/train.log"
train_status="${PIPESTATUS[0]}"
if [[ "${train_status}" -ne 0 ]]; then
  echo "ERROR: probe training / anchor check failed (exit ${train_status})" >&2
  exit "${train_status}"
fi

echo
echo "Done."
echo "Metrics: ${run_root}/c3_metrics.json"
echo "Summary: ${run_root}/summary.csv"
echo "Weights: ${run_root}/weights/"
echo "Features (untracked): experiments/C3/features/"

# 收尾自检：特征 4×13 = 52 个；权重 5×13 = 65 个（允许发散 run 缺权重）
RUN_ROOT="${run_root}" python - <<'PY'
import json
import os
from pathlib import Path

import numpy as np

feat = sorted(Path("experiments/C3/features").glob("*.layer*.npy"))
weights = sorted(Path(os.environ["RUN_ROOT"]).glob("weights/*/weights.npz"))
records = sorted(Path(os.environ["RUN_ROOT"]).glob("records/*.json"))
metrics_path = Path(os.environ["RUN_ROOT"]) / "c3_metrics.json"
print(f"feature files: {len(feat)}, weight files: {len(weights)}, records: {len(records)}")
bad = 0
if len(feat) != 4 * 13 or len(records) != 5 * 13:
    raise SystemExit("unexpected file counts (expected 52 features, 65 records)")
expected_rows = {"train": 8382, "valid": 1050, "test": 1050, "test_unseen": 1518}
for path in feat:
    arr = np.load(path, allow_pickle=False)
    split = path.name.split(".")[0]
    if arr.shape != (expected_rows[split], 2560) or arr.dtype != np.float16 or not np.isfinite(arr).all():
        print("BAD", path, arr.shape, arr.dtype); bad += 1
for path in weights:
    saved = np.load(path)
    if saved["weights"].shape != (2, 2560) or saved["bias"].shape != (2,) or not np.isfinite(saved["weights"]).all():
        print("BAD", path); bad += 1
payload = json.loads(metrics_path.read_text(encoding="utf-8"))
if len(payload.get("records", [])) != 65:
    print("BAD metrics record count"); bad += 1
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
