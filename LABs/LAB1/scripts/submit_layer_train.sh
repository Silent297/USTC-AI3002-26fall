#!/usr/bin/env bash
# 把分层训练拆成 5 个 Slurm 作业分块提交（每块远低于平台 4h 上限）。
#
# 用法（集群登录节点、仓库任意目录）：
#   bash LABs/LAB1/scripts/submit_layer_train.sh
#
# 断点续跑：某块超时/被杀/失败后，重投该块即可（已完成的层自动复用、不重训）：
#   RUN_TAG=truth_c TRAIN_ARGS='--tasks truth --layers 23,26,29' sbatch LABs/LAB1/scripts/run_layer_train.sbatch
# 或直接重跑本脚本（5 块全投；已完成的块秒级复算后退出，不浪费算力）。

set -euo pipefail

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: 找不到 sbatch，请在集群登录节点上运行本脚本" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_SCRIPT="${SCRIPT_DIR}/run_layer_train.sbatch"

if [[ ! -f "${SBATCH_SCRIPT}" ]]; then
  echo "ERROR: 找不到 ${SBATCH_SCRIPT}" >&2
  exit 1
fi

submit() {
  local tag="$1"
  local args="$2"
  echo "==> RUN_TAG=${tag}  TRAIN_ARGS='${args}'"
  RUN_TAG="${tag}" TRAIN_ARGS="${args}" sbatch "${SBATCH_SCRIPT}"
}

# world：13 层（z-score 标准化）+ extra（原始 L11 历史最优），快，约 1-2 分钟
submit world   "--tasks world"
# truth：原始特征 5000 轮，最慢节点上每层约 30 分钟量级，拆 4 块留足余量
submit truth_a "--tasks truth --layers 2,5,8,11"
submit truth_b "--tasks truth --layers 14,17,20"
submit truth_c "--tasks truth --layers 23,26,29"
submit truth_d "--tasks truth --layers 32,34,35"

echo
echo '已提交 5 个作业。查看状态：squeue -u $USER'
echo "结果目录：results/slurm_layers/{world,truth_a,truth_b,truth_c,truth_d}/"
echo "每块完成后可取回对应目录（summary.csv + weights/）；某块没跑完就重投该块。"
