#!/usr/bin/env bash
#SBATCH --job-name=lab2-ablation2
#SBATCH --account=stu
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_default
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -uo pipefail

# Slurm executes a spool copy: resolve inputs via submission directory, not script location.
SCRIPT_DIR=""
for candidate in "${SLURM_SUBMIT_DIR:-}" "${SLURM_SUBMIT_DIR:-}/LABs/LAB2/code/q1_neural_network"; do
  if [[ -n "${candidate}" && -f "${candidate}/ablation_exp2.py" ]]; then
    SCRIPT_DIR="$(cd "${candidate}" && pwd)"
    break
  fi
done
if [[ -z "${SCRIPT_DIR}" ]]; then
  echo "ERROR: submit from code/q1_neural_network or the course repository root" >&2
  exit 1
fi
LAB_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUN_ID="$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-smoke_$$}"
run_root="${LOG_ROOT:-${LAB_ROOT}/logs/T0011}/${RUN_ID}"
mkdir -p "${run_root}" || exit 1
run_log="${run_root}/job.log"
if [[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_JOB_NAME:-}" ]]; then
  for stream in out err; do
    source_log="${SLURM_SUBMIT_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.${stream}"
    if [[ -f "${source_log}" ]]; then
      mv -- "${source_log}" "${run_root}/" || { echo "ERROR: cannot archive Slurm log" >&2; exit 1; }
    fi
  done
fi

set +u
source "${HOME}/miniconda3/etc/profile.d/conda.sh" || { echo "ERROR: cannot source conda" >&2; exit 1; }
conda activate ai26 || { echo "ERROR: cannot activate ai26" >&2; exit 1; }
set -u
export PYTHONIOENCODING=utf-8
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "${SCRIPT_DIR}" || exit 1
PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" - <<'PY' 2>&1 | tee -a "${run_log}"
import sys
import torch
print("python:", sys.executable)
print("torch:", torch.__version__, "CUDA:", torch.version.cuda)
assert torch.cuda.is_available(), "CUDA is required; no CPU fallback"
name = torch.cuda.get_device_name(0)
print("device:", name)
assert "A100" in name, "Expected A100"
PY
gpu_status=("${PIPESTATUS[@]}")
if [[ "${gpu_status[0]}" -ne 0 || "${gpu_status[1]}" -ne 0 ]]; then
  echo "ERROR: GPU check/logging failed" >&2
  exit 2
fi

run_step() {
  "${PYTHON_BIN}" "$@" 2>&1 | tee -a "${run_log}"
  local statuses=("${PIPESTATUS[@]}")
  if [[ "${statuses[0]}" -ne 0 || "${statuses[1]}" -ne 0 ]]; then
    echo "ERROR: step or logging failed: $*" >&2
    return 1
  fi
}

output="outputs_T0011_${RUN_ID}/ablation_exp2"
run_step ablation_exp2.py --run --config ablation_exp2_config.json --output "${output}" || exit 1
run_step ablation_exp2.py --check-output --output "${output}" || exit 1
echo "ALL DONE" | tee -a "${run_log}" || exit 1
echo "ARTIFACT_DIR: ${SCRIPT_DIR}/${output}" | tee -a "${run_log}" || exit 1
echo "RUN_LOG: ${run_log}"
