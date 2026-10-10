#!/usr/bin/env bash
# 把分层训练拆成若干 Slurm 作业分块提交（每块远低于平台 4h 上限）。
#
# 用法（集群登录节点）：
#   bash LABs/LAB1/scripts/submit_layer_train.sh [truth|world|all]
#     truth  只提交 truth 4 块（world 已完成并入库，见 results/slurm_layers/world/）
#     world  只提交 world 1 块
#     all    提交全部 5 块（默认，向后兼容）
#
# 断点续跑：某块超时/被杀/失败后，重投该块即可（已完成的层自动复用、不重训）：
#   RUN_TAG=truth_c TRAIN_ARGS='--tasks truth --layers 23,26,29' sbatch LABs/LAB1/scripts/run_layer_train.sbatch
# 或直接重跑本脚本对应目标（已完成的块秒级复算后退出，不浪费算力）。

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

submit_truth_chunks() {
  # truth：原始特征 5000 轮；最慢观测节点上每层约 30 分钟量级，拆 4 块留足余量。
  # 块内已有权重的层（如 truth_a 的 layer 2）运行时自动复用，只复算指标。
  submit truth_a "--tasks truth --layers 2,5,8,11"
  submit truth_b "--tasks truth --layers 14,17,20"
  submit truth_c "--tasks truth --layers 23,26,29"
  submit truth_d "--tasks truth --layers 32,34,35"
}

case "${1:-all}" in
  truth)
    submit_truth_chunks
    echo
    echo "已提交 4 个 truth 作业（world 已完成，不重复提交）。"
    ;;
  world)
    # world：13 层（z-score 标准化）+ extra（原始 L11 历史最优），快，约 1-2 分钟
    submit world "--tasks world"
    echo
    echo "已提交 1 个 world 作业。"
    ;;
  all)
    submit world "--tasks world"
    submit_truth_chunks
    echo
    echo "已提交 5 个作业。"
    ;;
  *)
    echo "ERROR: 未知目标 '${1}'（可选：truth / world / all）" >&2
    exit 2
    ;;
esac

echo '查看状态：squeue -u $USER'
echo "结果目录：results/slurm_layers/{world,truth_a,truth_b,truth_c,truth_d}/"
echo "每块完成后可取回对应目录（results.json + summary.csv + weights/）；某块没跑完就重投该块。"
