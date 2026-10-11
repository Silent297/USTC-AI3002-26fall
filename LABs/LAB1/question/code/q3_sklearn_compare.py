# -*- coding: utf-8 -*-
"""
Q3：你的模型相比于 sklearn 的实现有拟合效果上的差异吗？（README 思考题 3，做完 world/truth 后再答）

对比设计（同数据、同划分、同目标函数）：
  A. world：层 11 隐表征 → (lat, lon) 线性回归
     - 标准化口径（分层训练配置）：无正则 / 纯 l2 / 纯 l1 / 完整配置（λ1=1e-3, λ2=1.0）
     - 原始口径（历史最优 λ2=0.1 + 发散演示 + 训练轮数影响）
  B. truth：层 20 隐表征 → 真假 逻辑回归（原始口径；无正则 / l2 / 最优配置 λ1=λ2=1e-3）

正则换算（ours: (1/N)·loss + λ1|w|₁ + λ2‖w‖²₂，sklearn 各自口径）：
  Ridge:        α  = N·λ2
  Lasso:        α  = λ1/2
  ElasticNet:   αρ = λ1/2,  α(1-ρ) = λ2
  Logistic(l2): C  = 1/(2Nλ2)

说明：最终指标对比用仓库 `model.py` 的真实实现；收敛曲线用与其更新式完全一致的仪器化副本
（model.py 的 fit 不记录 loss 历史，且不改动提交代码）。R² 之外同时给出权重一致性（cos、相对 L2）
与同一目标函数下的损失差，区分"优化器差异"与"目标函数差异"。

输出：figures/q3_convergence.png、q3_weight_parity.png、q3_valid_bars.png + 控制台数值表。
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression as SkLogReg, Ridge

HERE = Path(__file__).resolve().parent      # question/code
QROOT = HERE.parent                          # question
LAB1 = QROOT.parent                          # LABs/LAB1
FIG = QROOT / "figures"
sys.path.insert(0, str(LAB1))
from model import LinearRegressionModel, LogisticRegressionModel  # noqa: E402


def load(task, split, layer):
    X = np.load(LAB1 / "datasets" / task / f"{split}.layer{layer}.npy").astype(np.float64)
    df = pd.read_csv(LAB1 / "datasets" / task / f"{split}.csv")
    return X, df


# ============================== world ==============================
Xw_tr, dfw_tr = load("world", "train", 11)
Xw_va, dfw_va = load("world", "valid", 11)
ytr_w = dfw_tr[["latitude", "longitude"]].to_numpy(float)
yva_w = dfw_va[["latitude", "longitude"]].to_numpy(float)
Nw = len(ytr_w)
mu, sd = Xw_tr.mean(0), Xw_tr.std(0)
sd[sd == 0] = 1.0
Xz_tr, Xz_va = (Xw_tr - mu) / sd, (Xw_va - mu) / sd
print(f"world: train {Xw_tr.shape}, valid {Xw_va.shape}, N={Nw}")

world_rows = []      # (tag, mse_tr_ours, mse_va_ours, mse_va_sk, cos, rel, extra)
world_parity = {}    # tag -> (wo_concat, ws_concat)


def eval_world(tag, Xt, Xv, ours_kw, sk_factory, parity=False):
    t0 = time.perf_counter()
    W_o, b_o, W_s, b_s = [], [], [], []
    mse_tr = mse_va_o = mse_va_s = 0.0
    diverged = False
    for c in range(2):
        m = LinearRegressionModel(Xt.shape[1])
        m.fit(Xt, ytr_w[:, c], **ours_kw)
        diverged = diverged or m.diverged
        sk = sk_factory().fit(Xt, ytr_w[:, c])
        mse_tr += np.mean((m.predict(Xt) - ytr_w[:, c]) ** 2) / 2
        mse_va_o += np.mean((m.predict(Xv) - yva_w[:, c]) ** 2) / 2
        mse_va_s += np.mean((sk.predict(Xv) - yva_w[:, c]) ** 2) / 2
        W_o.append(m.weights)
        b_o.append(float(m.bias))
        W_s.append(sk.coef_)
        b_s.append(float(sk.intercept_))
    wo, ws = np.concatenate(W_o), np.concatenate(W_s)
    cos = float(wo @ ws / (np.linalg.norm(wo) * np.linalg.norm(ws) + 1e-12))
    rel = float(np.linalg.norm(wo - ws) / (np.linalg.norm(ws) + 1e-12))
    dt_o_s = time.perf_counter() - t0
    world_rows.append((tag, mse_tr, mse_va_o, mse_va_s, cos, rel, diverged))
    if parity:
        world_parity[tag] = (wo, ws)
    print(f"[world] {tag:34s} MSE_tr_ours {mse_tr:8.2f} | MSE_va ours {mse_va_o:8.2f} "
          f"sklearn {mse_va_s:8.2f} (Δ {mse_va_o - mse_va_s:+6.2f}) | cos {cos:.5f} "
          f"relL2 {rel:.3f} | diverged={diverged} | {dt_o_s:.0f}s(含sklearn)")
    return wo, ws


# ---- A1: 标准化，无正则 ----
eval_world("A1 z-score, no reg", Xz_tr, Xz_va,
           dict(lr=2e-3, epochs=3000, lambda1=0.0, lambda2=0.0),
           lambda: LinearRegression(), parity=True)
# ---- A2: 标准化，纯 l2（λ2=1.0 ≡ Ridge α=500） ----
eval_world("A2 z-score, l2=1.0 (Ridge500)", Xz_tr, Xz_va,
           dict(lr=2e-3, epochs=2000, lambda1=0.0, lambda2=1.0),
           lambda: Ridge(alpha=Nw * 1.0), parity=True)
# ---- A3: 标准化，纯 l1（λ1=1e-3 ≡ Lasso α=5e-4） ----
eval_world("A3 z-score, l1=1e-3 (Lasso5e-4)", Xz_tr, Xz_va,
           dict(lr=2e-3, epochs=2000, lambda1=1e-3, lambda2=0.0),
           lambda: Lasso(alpha=5e-4, max_iter=100_000, tol=1e-7))
# ---- A4: 标准化，完整训练配置（λ1=1e-3, λ2=1.0 ≡ ElasticNet α=1.0005, ρ=4.9975e-4） ----
eval_world("A4 z-score, full config (ENet)", Xz_tr, Xz_va,
           dict(lr=2e-3, epochs=2000, lambda1=1e-3, lambda2=1.0),
           lambda: ElasticNet(alpha=1.0005, l1_ratio=4.9975e-4, max_iter=100_000, tol=1e-7))
# ---- B1: 原始特征，无正则，长训练 ----
eval_world("B1 raw, no reg, ep20000", Xw_tr, Xw_va,
           dict(lr=5e-4, epochs=20000, lambda1=0.0, lambda2=0.0),
           lambda: LinearRegression())
# ---- B2: 原始特征，历史最优配置（λ1=1e-3, λ2=0.1 ≡ ElasticNet α=0.1005, ρ=4.975e-3） ----
eval_world("B2 raw, best cfg ep2000", Xw_tr, Xw_va,
           dict(lr=5e-4, epochs=2000, lambda1=1e-3, lambda2=0.1),
           lambda: ElasticNet(alpha=0.1005, l1_ratio=4.975e-3, max_iter=100_000, tol=1e-7),
           parity=True)
eval_world("B2b raw, best cfg ep20000", Xw_tr, Xw_va,
           dict(lr=5e-4, epochs=20000, lambda1=1e-3, lambda2=0.1),
           lambda: ElasticNet(alpha=0.1005, l1_ratio=4.975e-3, max_iter=100_000, tol=1e-7))
# ---- B3: 发散演示：lr=1e-3 超过 raw L11 稳定界（≈5.4e-4） ----
md = LinearRegressionModel(Xw_tr.shape[1])
md.fit(Xw_tr, ytr_w[:, 0], lr=1e-3, epochs=500, lambda1=0.0, lambda2=0.0)
print(f"[world] B3 raw, lr=1e-3 (above stability bound ~5.4e-4): diverged={md.diverged} "
      f"-> sklearn has no step-size issue (B1 exact LS)")

# ---- world 收敛曲线（仪器化副本）与 sklearn 最优值 ----
def gd_lin_hist(X, y, lr, epochs, l1, l2):
    n = len(y)
    w = np.zeros(X.shape[1])
    b = 0.0
    hist = np.empty(epochs)
    for e in range(epochs):
        r = X @ w + b - y
        hist[e] = np.sum(r ** 2) / n
        g = 2.0 / n * X.T @ r + l1 * np.sign(w) + 2.0 * l2 * w
        w = w - lr * g
        b = b - lr * (2.0 / n * np.sum(r))
    return hist


hist_w = gd_lin_hist(Xz_tr, ytr_w[:, 0], 2e-3, 3000, 0.0, 0.0)
sk_w = LinearRegression().fit(Xz_tr, ytr_w[:, 0])
opt_w = np.mean((sk_w.predict(Xz_tr) - ytr_w[:, 0]) ** 2)
print(f"[world] curve: ours final train MSE {hist_w[-1]:.4f} vs sklearn LS optimum {opt_w:.4f}")

# ============================== truth ==============================
Xt_tr, dft_tr = load("truth", "train", 20)
Xt_va, dft_va = load("truth", "valid", 20)
ytr_t = dft_tr["label"].to_numpy(float)
yva_t = dft_va["label"].to_numpy(float)
Nt = len(ytr_t)
print(f"\ntruth: train {Xt_tr.shape}, valid {Xt_va.shape}, N={Nt}")
C_MATCH = 1.0 / (2 * Nt * 1e-3)
print(f"truth: matched C = 1/(2*N*1e-3) = {C_MATCH:.4f}")

truth_rows = []
truth_parity = {}


def eval_truth(tag, ours_kw, sk_factory, parity=False):
    t0 = time.perf_counter()
    m = LogisticRegressionModel(Xt_tr.shape[1])
    m.fit(Xt_tr, ytr_t, **ours_kw)
    sk = sk_factory().fit(Xt_tr, ytr_t)
    p_tr_o = m.predict(Xt_tr) > 0.5
    p_va_o = m.predict(Xt_va) > 0.5
    p_tr_s = sk.predict(Xt_tr) > 0.5
    p_va_s = sk.predict(Xt_va) > 0.5
    acc_tr_o = np.mean(p_tr_o == ytr_t)
    acc_va_o = np.mean(p_va_o == yva_t)
    acc_tr_s = np.mean(p_tr_s == ytr_t)
    acc_va_s = np.mean(p_va_s == yva_t)
    ll_va_o = np.mean(np.logaddexp(0, Xt_va @ m.weights + m.bias) - yva_t * (Xt_va @ m.weights + m.bias))
    ll_va_s = np.mean(np.logaddexp(0, Xt_va @ sk.coef_ + sk.intercept_) -
                      yva_t * (Xt_va @ sk.coef_ + sk.intercept_))
    cos = float(m.weights @ sk.coef_ / (np.linalg.norm(m.weights) * np.linalg.norm(sk.coef_) + 1e-12))
    agree = np.mean(p_va_o == p_va_s)
    dt = time.perf_counter() - t0
    truth_rows.append((tag, acc_tr_o, acc_va_o, acc_tr_s, acc_va_s, ll_va_o, ll_va_s, cos, agree))
    if parity:
        truth_parity[tag] = (m.weights.copy(), sk.coef_.copy())
    print(f"[truth] {tag:30s} acc_tr ours {acc_tr_o:.4f} sk {acc_tr_s:.4f} | acc_va ours "
          f"{acc_va_o:.4f} sk {acc_va_s:.4f} | ll_va {ll_va_o:.4f}/{ll_va_s:.4f} | cos {cos:.5f} "
          f"agree {agree:.4f} | diverged={m.diverged} | {dt:.0f}s")
    return m


# T1: 无正则
md_t1 = LogisticRegressionModel(Xt_tr.shape[1])
md_t1.fit(Xt_tr, ytr_t, lr=5e-3, epochs=4000, lambda1=0.0, lambda2=0.0)
if md_t1.diverged:
    md_t1 = LogisticRegressionModel(Xt_tr.shape[1])
    md_t1.fit(Xt_tr, ytr_t, lr=1e-3, epochs=8000, lambda1=0.0, lambda2=0.0)
# 复用已经训练好的 T1（避免重复训练）：手动补一条记录
sk_none = SkLogReg(penalty=None, solver="lbfgs", max_iter=3000, tol=1e-6).fit(Xt_tr, ytr_t)
p_va_o = md_t1.predict(Xt_va) > 0.5
p_va_s = sk_none.predict(Xt_va) > 0.5
ll_va_o = np.mean(np.logaddexp(0, Xt_va @ md_t1.weights + md_t1.bias) -
                  yva_t * (Xt_va @ md_t1.weights + md_t1.bias))
ll_va_s = np.mean(np.logaddexp(0, Xt_va @ sk_none.coef_ + sk_none.intercept_) -
                  yva_t * (Xt_va @ sk_none.coef_ + sk_none.intercept_))
cos = float(md_t1.weights @ sk_none.coef_ / (np.linalg.norm(md_t1.weights) * np.linalg.norm(sk_none.coef_) + 1e-12))
agree = np.mean(p_va_o == p_va_s)
truth_rows.append(("T1 no reg (sk: penalty=None)", np.mean((md_t1.predict(Xt_tr) > 0.5) == ytr_t),
                   np.mean(p_va_o == yva_t), np.mean((sk_none.predict(Xt_tr) > 0.5) == ytr_t),
                   np.mean(p_va_s == yva_t), ll_va_o, ll_va_s, cos, agree))
truth_parity["T1 no reg"] = (md_t1.weights.copy(), sk_none.coef_.copy())
print(f"[truth] T1 no reg (sk: penalty=None)   acc_va ours {truth_rows[-1][2]:.4f} sk "
      f"{truth_rows[-1][4]:.4f} | ll_va {ll_va_o:.4f}/{ll_va_s:.4f} | cos {cos:.5f} "
      f"agree {agree:.4f} | diverged={md_t1.diverged}")

# T2: l2 匹配（λ2=1e-3 ≡ C=1/(2Nλ2)≈0.098）
eval_truth("T2 l2=1e-3 (sk C≈0.098)", dict(lr=5e-3, epochs=5000, lambda1=0.0, lambda2=1e-3),
           lambda: SkLogReg(penalty="l2", C=C_MATCH, solver="lbfgs", max_iter=3000, tol=1e-6),
           parity=True)
# T2b: 同配置更长训练（收敛趋势）
eval_truth("T2b l2=1e-3 ep10000", dict(lr=5e-3, epochs=10000, lambda1=0.0, lambda2=1e-3),
           lambda: SkLogReg(penalty="l2", C=C_MATCH, solver="lbfgs", max_iter=3000, tol=1e-6))
# T3: 最优配置（λ1=λ2=1e-3）；sklearn 侧只配 l2（无对应 l1 项），差异=目标函数本身差异
eval_truth("T3 best cfg l1=l2=1e-3", dict(lr=5e-3, epochs=5000, lambda1=1e-3, lambda2=1e-3),
           lambda: SkLogReg(penalty="l2", C=C_MATCH, solver="lbfgs", max_iter=3000, tol=1e-6))
# T4: sklearn 默认 C=1 参考
sk_c1 = SkLogReg(penalty="l2", C=1.0, solver="lbfgs", max_iter=3000, tol=1e-6).fit(Xt_tr, ytr_t)
print(f"[truth] T4 sklearn C=1 reference       acc_tr {np.mean((sk_c1.predict(Xt_tr) > 0.5) == ytr_t):.4f} "
      f"acc_va {np.mean((sk_c1.predict(Xt_va) > 0.5) == yva_t):.4f}")

# ---- truth 收敛曲线（仪器化副本）与 sklearn 最优 ----
def gd_log_hist(X, y, lr, epochs, l1, l2):
    n = len(y)
    w = np.zeros(X.shape[1])
    b = 0.0
    hist = np.empty(epochs)
    for e in range(epochs):
        z = X @ w + b
        hist[e] = np.mean(np.logaddexp(0, z) - y * z)
        r = 1.0 / (1.0 + np.exp(-z)) - y
        g = 1.0 / n * X.T @ r + l1 * np.sign(w) + 2.0 * l2 * w
        w = w - lr * g
        b = b - lr * (1.0 / n * np.sum(r))
    return hist


hist_t = gd_log_hist(Xt_tr, ytr_t, 5e-3, 5000, 0.0, 1e-3)
opt_t = np.mean(np.logaddexp(0, Xt_tr @ sk_none.coef_ + sk_none.intercept_) -
                ytr_t * (Xt_tr @ sk_none.coef_ + sk_none.intercept_))
sk_l2 = SkLogReg(penalty="l2", C=C_MATCH, solver="lbfgs", max_iter=3000, tol=1e-6).fit(Xt_tr, ytr_t)
opt_t_l2 = np.mean(np.logaddexp(0, Xt_tr @ sk_l2.coef_ + sk_l2.intercept_) -
                   ytr_t * (Xt_tr @ sk_l2.coef_ + sk_l2.intercept_))
print(f"[truth] curve: ours final train logloss {hist_t[-1]:.5f} vs sklearn l2 optimum "
      f"{opt_t_l2:.5f} (no-reg optimum {opt_t:.5f})")

# ============================== 图 ==============================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.2))
ax1.plot(np.arange(1, len(hist_w) + 1), hist_w, color="tab:red", lw=1.4,
         label="ours (GD, full batch)")
ax1.axhline(opt_w, color="k", ls="--", lw=1.2, label="sklearn LS (exact, SVD)")
ax1.set_yscale("log")
ax1.set_xlabel("epoch")
ax1.set_ylabel("train MSE  (world L11, z-scored, lat)")
ax1.set_title("world L11 (z-score, no reg): our GD reaches the LS optimum")
ax1.legend(fontsize=9)

ax2.plot(np.arange(1, len(hist_t) + 1), hist_t, color="tab:red", lw=1.4,
         label="ours (GD, full batch)")
ax2.axhline(opt_t_l2, color="k", ls="--", lw=1.2, label=f"sklearn l2 C={C_MATCH:.3f} optimum")
ax2.axhline(opt_t, color="gray", ls=":", lw=1.2, label="sklearn penalty=None optimum")
ax2.set_xlabel("epoch")
ax2.set_ylabel("train logloss  (truth L20, raw)")
ax2.set_title("truth L20 (raw, l2=1e-3): our GD approaches the lbfgs optimum")
ax2.legend(fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "q3_convergence.png", dpi=150)
fig.savefig(FIG / "q3_convergence.pdf")
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0))
panels = [("A1 z-score, no reg", "world z L11, no reg"),
          ("A2 z-score, l2=1.0 (Ridge500)", "world z L11, l2=1.0 (Ridge a=500)"),
          ("T2 l2=1e-3 (sk C≈0.098)", "truth L20, l2=1e-3 (sk C≈0.098)"),
          ]
pairs = [world_parity["A1 z-score, no reg"], world_parity["A2 z-score, l2=1.0 (Ridge500)"],
         truth_parity["T2 l2=1e-3 (sk C≈0.098)"]]
for ax, (tag, ttl), (wo, ws) in zip(axes, panels, pairs):
    ax.scatter(ws, wo, s=6, alpha=0.35, color="tab:blue")
    lim = max(np.abs(ws).max(), np.abs(wo).max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], color="k", lw=1, ls="--")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    cos = float(wo @ ws / (np.linalg.norm(wo) * np.linalg.norm(ws) + 1e-12))
    rel = float(np.linalg.norm(wo - ws) / (np.linalg.norm(ws) + 1e-12))
    ax.set_title(f"{ttl}\ncos = {cos:.4f}, relL2 = {rel:.3f}", fontsize=10)
    ax.set_xlabel("sklearn weight")
    ax.set_ylabel("our weight")
fig.suptitle("Q3: weight parity - our GD vs sklearn (same objective, mapped regularization)", y=1.0)
fig.tight_layout()
fig.savefig(FIG / "q3_weight_parity.png", dpi=150)
fig.savefig(FIG / "q3_weight_parity.pdf")
plt.close(fig)

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.6, 5.0))
tags = ["B1 raw, no reg, ep20000", "B2 raw, best cfg ep2000", "B2b raw, best cfg ep20000"]
lbl = ["ours\nno reg\nep20k", "ours best\ncfg ep2k", "ours best\ncfg ep20k"]
vals = [next(r[2] for r in world_rows if r[0] == t) for t in tags]
sk_vals = [next(r[3] for r in world_rows if r[0] == t) for t in tags]
xs = np.arange(3 + 1)
axL.bar(xs[:3] - 0.18, vals, width=0.34, color="tab:red", label="ours (GD)")
axL.bar(xs[:3] + 0.18, sk_vals, width=0.34, color="tab:blue", label="sklearn")
axL.set_xticks(np.arange(4))
axL.set_xticklabels(lbl + ["ours\nlr=1e-3\ndiverged"], fontsize=8.5)
axL.text(3, max(vals + sk_vals) * 1.05, "X", color="red", fontsize=22, ha="center", va="bottom",
         fontweight="bold")
axL.set_ylabel("valid MSE (lat/lon avg)")
axL.set_title("world raw L11: valid MSE\n(ours stability bound ~5.4e-4; sklearn exact LS)")
axL.legend(fontsize=9)

tags_t = ["T1 no reg (sk: penalty=None)", "T2 l2=1e-3 (sk C≈0.098)", "T3 best cfg l1=l2=1e-3"]
lbl_t = ["ours\nno reg", "ours\nl2=1e-3", "ours\nbest cfg"]
vals_t = [next(r[2] for r in truth_rows if r[0] == t) for t in tags_t]
sk_t = [next(r[4] for r in truth_rows if r[0] == t) for t in tags_t]
xs = np.arange(3)
axR.bar(xs - 0.18, vals_t, width=0.34, color="tab:red", label="ours (GD)")
axR.bar(xs + 0.18, sk_t, width=0.34, color="tab:blue", label="sklearn (lbfgs)")
axR.axhline(np.mean((sk_c1.predict(Xt_va) > 0.5) == yva_t), color="gray", ls=":", lw=1.4,
            label="sklearn C=1 (much stronger reg)")
axR.set_xticks(xs)
axR.set_xticklabels(lbl_t, fontsize=9)
axR.set_ylim(0.7, 0.95)
axR.set_ylabel("valid accuracy")
axR.set_title("truth raw L20: valid accuracy")
axR.legend(fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "q3_valid_bars.png", dpi=150)
fig.savefig(FIG / "q3_valid_bars.pdf")
plt.close(fig)

print("\nfigures saved to", FIG)
