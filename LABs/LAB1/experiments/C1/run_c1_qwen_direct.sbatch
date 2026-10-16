#!/usr/bin/env bash
#SBATCH --job-name=lab1-qwen-direct
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_default
# Students 分区混部 RTX5090（anode01-15）与 A100（anode16-26）：
# 5090 节点上 ai26 的 torch cu130 触发 CUDA error 804，钉死已验证可用的 A100。
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00

# Submit from the repository root:
#   sbatch LABs/LAB1/experiments/C1/run_c1_qwen_direct.sbatch
# 可选环境变量：
#   RUN_TAG=名字     输出目录 results/slurm_qwen_direct/<名字>（默认 qwen_direct_v1；重投同名字=续跑）
#   C1_ARGS='...'   原样传给 python（如 C1_ARGS="--limit 8" 小样冒烟）
#   MODEL_DIR=路径   默认 ~/models/Qwen3-4B-Base
#
# C1：Qwen3-4B-Base 直接输出经纬度（630 条 world 原问题，batch=4 贪心 ≤64 token）。
# 幂等：JSONL 完整即 SKIP；缺行续写。日志集中存放：results/slurm_qwen_direct/<RUN_TAG>/
# 下含 c1.log、run_meta.json、{train,valid}.jsonl 与 .out/.err。

set -uo pipefail

# ---------------------------------------------------------------------------
# User-editable configuration
# ---------------------------------------------------------------------------

REPO_DIR="${HOME}/USTC-AI3002-26fall"
LAB_DIR="${REPO_DIR}/LABs/LAB1"
MODEL_DIR="${MODEL_DIR:-${HOME}/models/Qwen3-4B-Base}"
RUN_TAG="${RUN_TAG:-qwen_direct_v1}"
OUTPUT_ROOT="${LAB_DIR}/results/slurm_qwen_direct"

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

run_root="${OUTPUT_ROOT}/${RUN_TAG}"
if ! mkdir -p "${run_root}"; then
  echo "ERROR: failed to create output directory: ${run_root}" >&2
  exit 1
fi
c1_log="${run_root}/c1.log"

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
echo "Model: ${MODEL_DIR}"
echo "Run: ${run_root}"
echo "C1 args: ${C1_ARGS:-<none>}"
echo "Log: ${c1_log}"
echo

# GPU 环境自检：把节点/驱动/torch 信息写进日志；CUDA 不可用则立即失败
# （典型故障：作业落在 RTX5090 节点，torch 报 CUDA error 804）。
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
  echo "ERROR: no usable CUDA GPU; fix the ai26 environment or the node choice first" >&2
  exit 2
fi
echo

# shellcheck disable=SC2086  # C1_ARGS 需要按空格拆成多个参数
python -u experiments/C1/c1_qwen_direct.py \
  --model-dir "${MODEL_DIR}" \
  --train-csv datasets/world/train.csv \
  --valid-csv datasets/world/valid.csv \
  --out-dir "${run_root}" \
  --batch-size 4 \
  --max-new-tokens 64 ${C1_ARGS:-} 2>&1 | tee -a "${c1_log}"
c1_status="${PIPESTATUS[0]}"

if [[ "${c1_status}" -ne 0 ]]; then
  echo "ERROR: c1_qwen_direct.py exited with status ${c1_status}" >&2
  exit "${c1_status}"
fi

# 收尾校验：JSONL 行数 / 逐行 JSON 可解析 / 字段齐全 / 行序 / 范围一致性。
# 解析率低不算故障（协议允许）；仅结构性问题 exit 1。
echo
echo "==== validation ===="
RUN_ROOT="${run_root}" python - <<'PY'
import json
import os
from pathlib import Path

run_root = Path(os.environ["RUN_ROOT"])
meta = json.loads((run_root / "run_meta.json").read_text(encoding="utf-8"))
expected = meta["expected_rows"]
required = {
    "split", "row", "sentence", "country", "lat_true", "lon_true",
    "prompt", "raw_output", "tier", "pred_lat", "pred_lon",
    "lat_raw", "lon_raw", "in_range", "n_new_tokens", "finish", "notes",
}
problems = []
for split in ("train", "valid"):
    path = run_root / f"{split}.jsonl"
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != expected[split]:
        problems.append(f"{path.name}: {len(lines)} lines != expected {expected[split]}")
    for index, line in enumerate(lines):
        try:
            record = json.loads(line)
            if not required.issubset(record):
                raise ValueError(f"missing fields: {sorted(required - set(record))}")
            if record["row"] != index:
                raise ValueError(f"row order: {record['row']} != {index}")
            if record["in_range"]:
                lat, lon = record["pred_lat"], record["pred_lon"]
                if lat is None or lon is None or not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                    raise ValueError(f"in_range but invalid pred: {lat},{lon}")
        except Exception as exc:
            problems.append(f"{path.name}:{index}: {exc}")
            if len(problems) > 20:
                break
    tiers = meta.get("tiers", {}).get(split, {})
    total = sum(tiers.values())
    in_range = total - tiers.get("out_of_range", 0) - tiers.get("failed", 0)
    print(f"{split}: {len(lines)} rows; tiers={tiers}; valid(in_range)={in_range}")

if problems:
    print("Structural problems:")
    for problem in problems:
        print(f"  {problem}")
    print("Parse-rate deviations are NOT failures; only structural problems are.")
    raise SystemExit(1)

print("Structural check passed (parse rates reported above; low rates are allowed).")
PY
validation_status=$?

echo
if [[ "${validation_status}" -ne 0 ]]; then
  echo "ERROR: validation failed" >&2
  exit "${validation_status}"
fi

echo "Done: C1 generation complete. Results in ${run_root}."
echo "Next: commit + push this directory from the cluster, then run experiments/C1/c1_eval.py locally."
exit 0
