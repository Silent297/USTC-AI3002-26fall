"""计算全批量梯度下降的学习率稳定界（只用 train split）。

对均方误差 loss = ‖Xw+b−y‖²/N，Hessian H = 2XᵀX/N，参数误差按 (I − lr·H) 传播，
迭代收敛（数值稳定）条件为 lr < 2/λmax(H) = N/‖X‖₂²（‖X‖₂² 即 λmax(XᵀX)）。
logistic 损失的 Hessian = Xᵀdiag(p(1−p))X/N ⪯ XᵀX/(4N)，稳定界放宽到
最坏情形上界 8N/‖X‖₂²（约 8 倍），这是 truth 任务原始特征可用的原因。

输出每层的 raw 与 z-score（μ/σ 取 train split，与 run_grid.py 口径一致）两种界。

运行：python experiments/stability_bound.py [--tasks world,truth] [--layers 11,23,35]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train23 import load_split

LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 34, 35)


def spectral_norm_sq(x: np.ndarray) -> float:
    # ‖X‖₂² = λmax(XᵀX)，用 SVD 的第一奇异值精确计算
    return float(np.linalg.norm(x, 2) ** 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default="world,truth")
    parser.add_argument("--layers", default=None, help="逗号分隔层号，默认 13 个已提取层")
    args = parser.parse_args()
    tasks = args.tasks.split(",")
    layers = [int(v) for v in args.layers.split(",")] if args.layers else list(LAYERS)

    header = (
        f"{'task':<6}{'layer':>5}{'N':>7} | {'raw ‖X‖₂²':>11}{'raw 界':>11} | "
        f"{'std ‖Xz‖₂²':>12}{'std 界':>11}"
    )
    print(header)
    for task in tasks:
        for layer in layers:
            x, _ = load_split(task, "train", layer)
            n = x.shape[0]
            raw = spectral_norm_sq(x)
            mu = x.mean(axis=0)
            sd = x.std(axis=0)
            sd = np.where(sd == 0, 1.0, sd)
            std = spectral_norm_sq((x - mu) / sd)
            print(
                f"{task:<6}{layer:>5}{n:>7} | {raw:>11.3e}{n / raw:>11.3e} | "
                f"{std:>12.3e}{n / std:>11.3e}"
            )


if __name__ == "__main__":
    main()
