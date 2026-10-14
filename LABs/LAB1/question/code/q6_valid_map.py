# -*- coding: utf-8 -*-
"""Q6：最优 world 模型（raw L11 线性探针）在验证集上的预测结果 → 地图可视化。

数据：datasets/world/valid.layer11.npy + valid.csv（130 条）
权重：results/slurm_layers/world/weights/world_layer11_extra0/weights.npz
      （standardize=false，已折回原始特征空间；valid MSE 150.43 为历史最优）
合规：地图只画海岸线（Natural Earth 110m），不画国界线。
输出：question/figures/q6_valid_map.{png,pdf}
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
FIG = ROOT / "question" / "figures"
FIG.mkdir(exist_ok=True)
W_PATH = ROOT / "results/slurm_layers/world/weights/world_layer11_extra0/weights.npz"

d = np.load(W_PATH)
w, b = d["weights"], d["bias"]                        # (2, 2560), (2,)
X = np.load(ROOT / "datasets/world/valid.layer11.npy").astype(np.float64)
df = pd.read_csv(ROOT / "datasets/world/valid.csv")
y = df[["latitude", "longitude"]].to_numpy(np.float64)  # 目标列序 (lat, lon)

pred = X @ w.T + b
mse = float(np.mean((pred - y) ** 2))
r2 = float(1.0 - np.sum((pred - y) ** 2) / np.sum((y - y.mean(0)) ** 2))
err = np.sqrt(np.sum((pred - y) ** 2, axis=1))        # 单点定位误差（度）
print(f"valid MSE = {mse:.4f}  (anchor 150.4277)")
print(f"R2 = {r2:.4f}, mean err = {err.mean():.2f} deg, max err = {err.max():.2f} deg")
q25, q50, q75 = np.percentile(err, [25, 50, 75])
print(f"err quartiles (25/50/75) = {q25:.2f} / {q50:.2f} / {q75:.2f} deg,"
      f"  >30 deg: {int((err > 30).sum())} pts")
print(f"per-coordinate MAE: lat {np.abs(pred[:, 0] - y[:, 0]).mean():.2f},"
      f" lon {np.abs(pred[:, 1] - y[:, 1]).mean():.2f} deg")
for i in np.argsort(err)[::-1][:10]:
    print(f"  worst: err={err[i]:6.2f}  truth=({y[i, 0]:7.2f},{y[i, 1]:7.2f})  "
          f"pred=({pred[i, 0]:7.2f},{pred[i, 1]:7.2f})  country={df['country'].iloc[i]}")

# ---- 地图：仅海岸线（合规红线：不画国界线）----
fig = plt.figure(figsize=(12.6, 7.0))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.set_global()
ax.coastlines(resolution="110m", linewidth=0.5, color="0.4")
gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="0.85", alpha=0.7)
gl.top_labels = gl.right_labels = False
gl.xlabel_style = {"size": 8}
gl.ylabel_style = {"size": 8}

ax.scatter(y[:, 1], y[:, 0], s=16, facecolor="none", edgecolor="0.35",
           linewidths=0.8, transform=ccrs.PlateCarree(), zorder=3,
           label="ground truth")
sc = ax.scatter(pred[:, 1], pred[:, 0], c=err, cmap="viridis", s=30,
                edgecolor="k", linewidths=0.3, transform=ccrs.PlateCarree(),
                zorder=4, label="predicted (color = positioning error)")
cbar = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
cbar.set_label("positioning error (deg)", fontsize=9)

ax.set_title(f"Q6: valid-set predictions of the world probe (raw L11, best cfg)\n"
             f"valid MSE = {mse:.1f} deg$^2$, $R^2$ = {r2:.3f}, n = {len(y)}",
             fontsize=11)
ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
fig.tight_layout()
fig.savefig(FIG / "q6_valid_map.png", dpi=150)
fig.savefig(FIG / "q6_valid_map.pdf")
print("saved:", FIG / "q6_valid_map.png")
