#!/usr/bin/env bash
#SBATCH --job-name=lab1-extract
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_default
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00

# Submit from the repository root:
#   sbatch LABs/LAB1/scripts/extract_layers.sbatch
#
# 提取 Qwen3-4B-Base 的 decoder-block 激活，加密「层数」维度的采样点。
# 输出落在 CSV 旁边：datasets/<task>/<split>.layer<L>.npy，
# 与 train23.py:load_split 及 run_grid.py 的读取约定一致。
# 可重复提交：已存在的 .npy 会被跳过，不会重算。

set -uo pipefail

# ---------------------------------------------------------------------------
# User-editable configuration
# ---------------------------------------------------------------------------

REPO_DIR="${HOME}/USTC-AI3002-26fall"
LAB_DIR="${REPO_DIR}/LABs/LAB1"
MODEL_DIR="${MODEL_DIR:-${HOME}/models/Qwen3-4B-Base}"

# 待提取的新层（0-based decoder block），与已有 11/23/35 合并后约每 3 层一个点
LAYERS="${LAYERS:-2 5 8 14 17 20 26 29 32 34}"
TASKS="${TASKS:-truth world}"
SPLITS="${SPLITS:-train valid}"

LOG_ROOT="${LAB_DIR}/results/slurm_extract"

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

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

run_root="${LOG_ROOT}/$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-$$}"
if ! mkdir -p "${run_root}"; then
  echo "ERROR: failed to create log directory: ${run_root}" >&2
  exit 1
fi
extract_log="${run_root}/extract.log"

cd "${LAB_DIR}"

echo "Repository: ${REPO_DIR}"
echo "Model: ${MODEL_DIR}"
echo "Layers: ${LAYERS}"
echo "Tasks: ${TASKS}"
echo "Splits: ${SPLITS}"
echo "Log: ${extract_log}"
echo

# truth 取句末 token；world 取国家 token
token_selection() {
  case "$1" in
    truth) echo "last" ;;
    world) echo "country" ;;
    *) return 1 ;;
  esac
}

ok=0
skipped=0
failures=""

run_one() {
  local task="$1" split="$2" layer="$3"
  local input="${LAB_DIR}/datasets/${task}/${split}.csv"
  local output="${LAB_DIR}/datasets/${task}/${split}.layer${layer}.npy"
  local sel status

  if ! sel="$(token_selection "${task}")"; then
    echo "FAIL  ${task}/${split} layer ${layer}: unknown task" | tee -a "${extract_log}"
    failures+="${task}/${split}/layer${layer}"$'\n'
    return
  fi

  if [[ ! -f "${input}" ]]; then
    echo "FAIL  ${task}/${split} layer ${layer}: input missing: ${input}" | tee -a "${extract_log}"
    failures+="${task}/${split}/layer${layer}"$'\n'
    return
  fi

  if [[ -f "${output}" ]]; then
    echo "SKIP  ${task}/${split} layer ${layer} (already exists)"
    skipped=$((skipped + 1))
    return
  fi

  echo "----- ${task}/${split} layer ${layer} (token-selection=${sel}) -----"
  python extract_intrep.py \
    --model_dir "${MODEL_DIR}" \
    --input "${input}" \
    --layer "${layer}" \
    --token-selection "${sel}" 2>&1 | tee -a "${extract_log}"
  status="${PIPESTATUS[0]}"
  if [[ "${status}" -ne 0 ]]; then
    echo "FAIL  ${task}/${split} layer ${layer} (exit ${status})" | tee -a "${extract_log}"
    failures+="${task}/${split}/layer${layer}"$'\n'
    return
  fi
  ok=$((ok + 1))
}

for task in ${TASKS}; do
  for split in ${SPLITS}; do
    for layer in ${LAYERS}; do
      run_one "${task}" "${split}" "${layer}"
    done
  done
done

echo
echo "Summary: ok=${ok}, skipped=${skipped}"
if [[ -n "${failures}" ]]; then
  echo "Failed combinations:"
  printf '%s' "${failures}"
fi

# 校验所有目标文件可读、形状/类型正确、无 NaN/Inf。
# 校验失败的文件多半是上次作业被中断留下的半截文件，删除后重投即可补齐。
LAB_DIR="${LAB_DIR}" LAYERS="${LAYERS}" TASKS="${TASKS}" SPLITS="${SPLITS}" \
python - <<'PY'
import os
from pathlib import Path

import numpy as np

lab_dir = Path(os.environ["LAB_DIR"])
layers = os.environ["LAYERS"].split()
tasks = os.environ["TASKS"].split()
splits = os.environ["SPLITS"].split()

problems = []
valid = 0
total = len(tasks) * len(splits) * len(layers)
for task in tasks:
    for split in splits:
        for layer in layers:
            path = lab_dir / "datasets" / task / f"{split}.layer{layer}.npy"
            try:
                feats = np.load(path, allow_pickle=False)
                if feats.ndim != 2 or feats.shape[1] != 2560:
                    raise ValueError(f"unexpected shape {feats.shape}")
                if feats.dtype != np.float16:
                    raise ValueError(f"unexpected dtype {feats.dtype}")
                if not np.isfinite(feats).all():
                    raise ValueError("contains non-finite values")
            except Exception as exc:
                problems.append(f"{path}: {exc}")
                print(f"BAD   {task}/{split} layer {layer}")
            else:
                valid += 1
                print(f"OK    {task}/{split} layer {layer}  shape={feats.shape}")

print(f"\nValidated {valid}/{total} feature files")
if problems:
    print("Problems:")
    for problem in problems:
        print(f"  {problem}")
    print("Truncated leftovers: delete the file and resubmit.")
    print("Missing files: just resubmit; already-valid files are skipped.")
    raise SystemExit(1)
PY
validation_status=$?

echo
if [[ -n "${failures}" ]]; then
  echo "ERROR: extraction failures occurred; see the log above" >&2
  exit 1
fi
if [[ "${validation_status}" -ne 0 ]]; then
  echo "ERROR: validation failed" >&2
  exit "${validation_status}"
fi

echo "Done: all feature files present and valid."
exit 0
