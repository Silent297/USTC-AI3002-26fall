#!/usr/bin/env bash
#SBATCH --job-name=lab2-q2-full
#SBATCH --account=stu
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_default
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -uo pipefail

# Slurm 会把批处理脚本复制到 spool 目录执行，不能用 BASH_SOURCE 反推仓库位置。
SCRIPT_DIR=""
for candidate in \
  "${SLURM_SUBMIT_DIR:-}" \
  "${SLURM_SUBMIT_DIR:-}/LABs/LAB2/code/q2_clustering" \
  "$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"; do
  if [[ -n "${candidate}" && -f "${candidate}/extras.py" ]]; then
    SCRIPT_DIR="$(cd "${candidate}" && pwd)"
    break
  fi
done
if [[ -z "${SCRIPT_DIR}" ]]; then
  echo "ERROR: cannot locate code/q2_clustering (extras.py); submit from LABs/LAB2/code/q2_clustering" >&2
  exit 1
fi
LAB_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_ROOT="${LOG_ROOT:-${LAB_ROOT}/logs/T0008}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="${STAMP}_${SLURM_JOB_ID:-local}"
run_root="${LOG_ROOT}/${RUN_ID}"
mkdir -p "${run_root}"
run_log="${run_root}/job.log"

# .out/.err 由 Slurm 在提交目录打开；移入日志目录后 fd 仍续写。
if [[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  for stream in out err; do
    s="${SLURM_SUBMIT_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.${stream}"
    [[ -f "$s" ]] && { mv "$s" "${run_root}/" || echo "WARNING: cannot archive $s" >&2; }
  done
fi

set +u
source "${HOME}/miniconda3/etc/profile.d/conda.sh" || { echo "ERROR: cannot source conda" >&2; exit 1; }
conda activate ai26 || { echo "ERROR: cannot activate ai26" >&2; exit 1; }
set -u
export PYTHONIOENCODING="utf-8"

cd "${SCRIPT_DIR}" || { echo "ERROR: cannot cd ${SCRIPT_DIR}" >&2; exit 1; }

echo "=== GPU check ===" | tee -a "${run_log}"
python - <<'PY' 2>&1 | tee -a "${run_log}"
import sys
import torch

print(f"torch {torch.__version__} (CUDA {torch.version.cuda})")
if not torch.cuda.is_available():
    print("ERROR: ai26 has no CUDA torch; install code/requirements-gpu.txt once, then resubmit", file=sys.stderr)
    raise SystemExit(2)
name = torch.cuda.get_device_name(0)
print(f"device: {name}")
if "A100" not in name:
    print(f"ERROR: expected A100, got {name}; do not run on RTX5090 nodes", file=sys.stderr)
    raise SystemExit(2)
PY
gpu_status=${PIPESTATUS[0]}
if [[ "${gpu_status}" -ne 0 ]]; then
  echo "ERROR: GPU check failed (${gpu_status})" >&2
  exit "${gpu_status}"
fi

run_step() {
  local label="$1"
  shift
  echo "=== RUN: ${label}: $*" | tee -a "${run_log}"
  python "$@" 2>&1 | tee -a "${run_log}"
  local status="${PIPESTATUS[0]}"
  if [[ "${status}" -ne 0 ]]; then
    echo "FAIL(${status}): ${label}" >&2
    return "${status}"
  fi
}

run_step "self-check-float32" extras.py self-check --device cuda || exit 1
run_step "self-check-float64" extras.py self-check --device cuda --dtype float64 || exit 1

fit_out="outputs_fit_T0008_${RUN_ID}"
eval_out="outputs_eval_T0008_${RUN_ID}"

run_step "fit" extras.py run --stage fit --device cuda --config config.json --output "${fit_out}" || exit 1
run_step "evaluate" extras.py evaluate --device cuda --experiment "${fit_out}" --output "${eval_out}" || exit 1
run_step "check-output" extras.py check-output --output "${eval_out}" || exit 1

echo "=== final output validation ===" | tee -a "${run_log}"
python - "${eval_out}" <<'PY' 2>&1 | tee -a "${run_log}"
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

out = Path(sys.argv[1])
labels = np.load(out / "clusters.npy", allow_pickle=False)
assert labels.shape == (56000,), labels.shape
assert labels.dtype == np.int32, labels.dtype
assert labels.min() >= 0
ids, counts = np.unique(labels, return_counts=True)
print("clusters:", labels.shape, labels.dtype, "actual_k", len(ids))
print("sizes min/max:", int(counts.min()), int(counts.max()))
print("sha256:", hashlib.sha256((out / "clusters.npy").read_bytes()).hexdigest())

manifest = json.loads((out / "manifest.json").read_text())
print("manifest:", manifest.get("status"), f'{manifest.get("completed_runs")}/{manifest.get("planned_runs")}',
      "successful_fits", manifest.get("successful_fits"), "selection", manifest.get("selection_status"))
assert manifest.get("status") == "completed", manifest.get("status")
assert manifest.get("completed_runs") == manifest.get("planned_runs") == 32
assert manifest.get("successful_fits") == 32
assert manifest.get("selection_status") == "selected"

selection = json.loads((out / "selection.json").read_text())
print("selected:", selection.get("method"), "K", selection.get("k"), selection.get("run_key"), "seed", selection.get("seed"))
print("ARTIFACT_DIR:", out.resolve())
PY
check_status=${PIPESTATUS[0]}
if [[ "${check_status}" -ne 0 ]]; then
  echo "FAIL: final output validation (${check_status})" >&2
  exit 1
fi

if [[ ${#failures[@]} -gt 0 ]]; then
  echo "FAILED: ${failures[*]}" >&2
  exit 1
fi
echo "ALL DONE"
echo "RUN_LOG: ${run_log}"
echo "FIT_DIR: ${SCRIPT_DIR}/${fit_out}"
echo "EVAL_DIR: ${SCRIPT_DIR}/${eval_out}"
