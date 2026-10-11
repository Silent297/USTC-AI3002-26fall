# -*- coding: utf-8 -*-
"""
Q1：复现课件《4-正则化》p42（Ridge/ℓ2）与 p49（Lasso/ℓ1）的系数路径。

数据：question/data/kc_house_data.csv（King County 房价，CC0），取课件同款 8 特征，y = price。
约定：① 特征先 z-score 标准化（课件 p20 批注 "all features normalized so coeff. on same scale"）；
      ② 目标 = min ||y-Xw||² + λ||w||²（课件 p46 的 RSS + λ 口径，等价于 sklearn 的 alpha）；
      ③ λ 轴 0~2e5 线性网格，与课件一致；λ=0 处为最小二乘解。
输出：question/figures/q1_{ridge,lasso,both}_paths.{png,pdf} 及控制台系数表。
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import Lasso, LinearRegression, Ridge

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

FEATURES = ["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
            "yr_built", "yr_renovated", "waterfront"]
COLORS = ["black", "red", "green", "blue", "cyan", "magenta", "goldenrod", "darkgray"]
LAM_MAX, N_LAM = 200_000.0, 201

df = pd.read_csv(ROOT / "data" / "kc_house_data.csv")
n0 = len(df)
df = df.dropna(subset=FEATURES)
print(f"samples: {len(df)} (dropped {n0 - len(df)} rows with missing bedrooms/bathrooms)")

X_raw = df[FEATURES].to_numpy(float)
y = df["price"].to_numpy(float)
X = (X_raw - X_raw.mean(0)) / X_raw.std(0)

lams = np.linspace(0.0, LAM_MAX, N_LAM)

# ---------- Ridge：SVD 闭式解（含截距等价于对 X,y 中心化） ----------
Xc, yc = X - X.mean(0), y - y.mean()
U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
Uty = U.T @ yc
W_ridge = np.array([Vt.T @ (s / (s ** 2 + lam) * Uty) for lam in lams])

for lam in (1e3, 5e4):
    d = np.abs(W_ridge[np.searchsorted(lams, lam)] - Ridge(alpha=lam).fit(X, y).coef_).max()
    print(f"ridge sanity a={lam:g}: max|diff| vs sklearn = {d:.2e}")

# ---------- Lasso：坐标下降 + warm start（自大 λ 向小 λ 扫） ----------
model = Lasso(fit_intercept=True, warm_start=True, max_iter=200_000, tol=1e-6)
W_lasso = []
for lam in lams[:0:-1]:  # 200000 -> 1000
    model.alpha = float(lam)
    model.fit(X, y)
    W_lasso.append(model.coef_.copy())
W_lasso = np.array(W_lasso)[::-1]  # 对齐 lams[1:]
ls_coef = LinearRegression().fit(X, y).coef_

W_lasso_full = np.vstack([ls_coef, W_lasso])  # λ=0 为最小二乘解

# ---------- 控制台表格 ----------
sel = [0, 20, 50, 100, 200]  # λ = 0, 2e4, 5e4, 1e5, 2e5
print("\nlambda =", [int(lams[i]) for i in sel])
print(f"{'feature':14s}" + "".join(f"{'ridge@'+str(int(lams[i])):>16s}" for i in sel)
      + "".join(f"{'lasso@'+str(int(lams[i])):>16s}" for i in sel))
for j, f in enumerate(FEATURES):
    print(f"{f:14s}" + "".join(f"{W_ridge[i, j]:16.0f}" for i in sel)
          + "".join(f"{W_lasso_full[i, j]:16.0f}" for i in sel))

print("\nlasso zero-entry (last grid lambda with nonzero coef; beyond it coef == 0):")
for j, f in enumerate(FEATURES):
    nz = lams[np.abs(W_lasso_full[:, j]) > 0]
    print(f"  {f:14s} " + (f"lambda* = {nz.max():.0f}" if len(nz) else "all zero"))

# ---------- 画图 ----------
def draw(ax, W, title):
    for j, f in enumerate(FEATURES):
        ax.plot(lams, W[:, j], color=COLORS[j], marker="o", markersize=3.5,
                markevery=5, mfc="white", lw=1.2, label=f)
    ax.axhline(0, color="k", lw=0.6, alpha=0.4)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"coefficients $\hat{w}_j$")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)

fig, ax = plt.subplots(figsize=(9.6, 6.4))
draw(ax, W_ridge, "Coefficient path - ridge (King County, z-scored)")
fig.tight_layout()
fig.savefig(FIG / "q1_ridge_path.png", dpi=150)
fig.savefig(FIG / "q1_ridge_path.pdf")
plt.close(fig)

fig, ax = plt.subplots(figsize=(9.6, 6.4))
draw(ax, W_lasso_full, "Coefficient path - lasso (King County, z-scored)")
fig.tight_layout()
fig.savefig(FIG / "q1_lasso_path.png", dpi=150)
fig.savefig(FIG / "q1_lasso_path.pdf")
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8))
draw(axes[0], W_ridge, "Ridge (L2)")
draw(axes[1], W_lasso_full, "Lasso (L1)")
fig.tight_layout()
fig.savefig(FIG / "q1_both_paths.png", dpi=150)
fig.savefig(FIG / "q1_both_paths.pdf")
plt.close(fig)

print("\nfigures saved to", FIG)
