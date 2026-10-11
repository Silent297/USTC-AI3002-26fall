# -*- coding: utf-8 -*-
"""
Q2：案例说明「把 ℓ2(Ridge) 的小系数阈值化为 0 做特征选择」为什么不好（课件 p43–p45）。

课件机制：
  p43/44: 人为阈值 τ 做选择 → 结果高度依赖 τ，不稳定；
  p45:    高度相关特征在 Ridge 下被同时收缩 → 阈值化会整组删除（"Nothing measuring bathrooms!"）。

实验：
  A. 真实数据 King County（9 特征 = Q1 的 8 列 + sqft_above，与 sqft_living r=0.876）：
     - Ridge 路径 + 阈值带 τ_band=1e4：选择集沿 λ 跳变 11 次（bedrooms/floors 离开后回归）；
       同一模型 τ 从 2e4→5e4→1e5 选择 5→2→0 个特征；
     - 对照 Lasso 零模式：单调嵌套、沿路径给出相关组代表（sqft_above 精确归零, sqft_living 保留）；
     - 共线性代价：加入 sqft_above 后 sqft_living 系数近似减半 → 保守阈值 τ_del=1e5 下
       "整组被删"（而单特征模型同 τ 会保留它）；
     - 预测代价：τ 扫动下重拟合 valid R² 抖动 vs Lasso 保留集；
     - Bootstrap（半样本 B=40）：阈值化选择的频率（不稳定）vs Lasso 非零频率（稳定）。
  B. 合成复制特征（x2 ≡ x1）：Ridge 最小范数解把权重对半切（各 ≈ w/2），任何 τ∈(w/2, w)
     都会删掉整组；Lasso 保留一个代表 (w, 0)，组信号不丢。

输出：figures/q2_*.{png,pdf} 及控制台数值表。
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path
from sklearn.linear_model import Lasso, LinearRegression, Ridge

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

FEATURES = ["bedrooms", "bathrooms", "sqft_living", "sqft_above", "sqft_lot",
            "floors", "yr_built", "yr_renovated", "waterfront"]
COLORS = ["black", "red", "green", "darkorange", "blue", "cyan", "magenta",
          "goldenrod", "darkgray"]
LAM_MAX, N_LAM = 200_000.0, 201
LAM_STORY = 20_000.0      # 中等正则强度（故事 λ）
TAU_BAND = 10_000.0       # 阈值带（不稳定性叙事：1e4 下沿 λ 出现 11 次跳变 + 离开后回归）
TAU_DEL = 100_000.0       # 保守阈值（"整组删除"叙事）
NZ_EPS = 1.0              # Lasso "非零"判定（美元/σ）

df = pd.read_csv(ROOT / "data" / "kc_house_data.csv")
n0 = len(df)
df = df.dropna(subset=FEATURES)
print(f"samples: {len(df)} (dropped {n0 - len(df)} rows)")
X_raw = df[FEATURES].to_numpy(float)
y = df["price"].to_numpy(float)
X = (X_raw - X_raw.mean(0)) / X_raw.std(0)
lams = np.linspace(0.0, LAM_MAX, N_LAM)
JL = FEATURES.index("sqft_living")
JA = FEATURES.index("sqft_above")
I20, I100 = 20, 100      # lams[20] = 2e4, lams[100] = 1e5


def svd_ridge(Xc, yc, lam):
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt.T @ (s / (s ** 2 + lam) * (U.T @ yc))


def kept(w, tau):
    return [f for f, wj in zip(FEATURES, w) if abs(wj) > tau]


# ---------- 0. 相关系数 ----------
C = np.corrcoef(X.T)
pairs = []
for a in range(len(FEATURES)):
    for b in range(a + 1, len(FEATURES)):
        pairs.append((abs(C[a, b]), FEATURES[a], FEATURES[b], C[a, b]))
pairs.sort(reverse=True)
print("\ntop correlated pairs:")
for _, f1, f2, c in pairs[:6]:
    print(f"  {f1:13s} {f2:13s} r = {c:+.3f}")

fig, ax = plt.subplots(figsize=(7.8, 6.8))
im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(FEATURES)))
ax.set_xticklabels(FEATURES, rotation=45, ha="right")
ax.set_yticks(range(len(FEATURES)))
ax.set_yticklabels(FEATURES)
for i in range(len(FEATURES)):
    for j in range(len(FEATURES)):
        ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", fontsize=7.5)
ax.set_title("King County (9 features): correlation matrix")
fig.colorbar(im, ax=ax, shrink=0.85)
fig.tight_layout()
fig.savefig(FIG / "q2_corr_matrix.png", dpi=150)
fig.savefig(FIG / "q2_corr_matrix.pdf")
plt.close(fig)

# ---------- 1. Ridge 路径（闭式解）----------
Xc, yc = X - X.mean(0), y - y.mean()
W_ridge = np.array([svd_ridge(Xc, yc, lam) for lam in lams])
d = np.abs(W_ridge[I20] - Ridge(alpha=LAM_STORY).fit(X, y).coef_).max()
print(f"\nridge sanity lambda=2e4: max|diff| vs sklearn = {d:.2e}")

print("\nLS (lambda=0) coefficients: " + "  ".join(
    f"{f[:9]}={W_ridge[0, j]:+8.0f}" for j, f in enumerate(FEATURES)))
print("\n[scan] ridge coefficients & kept-sets (tau grid):")
for i in (I20, 50, I100):
    w = W_ridge[i]
    print(f"  lambda={lams[i]:7.0f}: " + "  ".join(
        f"{f[:9]}={w[j]:+8.0f}" for j, f in enumerate(FEATURES)))
    for tau in (2e4, 5e4, 1e5):
        K = kept(w, tau)
        c = len([f for f in K if f != "sqft_above" or True])
        print(f"     tau={tau:6.0f}: kept({c}) = {{{', '.join(K)}}}")

# 沿 λ 的选择集变化（多个 τ）
trace_info = {}
for tau in (1e4, 2e4, 5e4, 1e5):
    Kmat = np.abs(W_ridge) > tau
    changes, reentries = 0, []
    prev = Kmat[0]
    out_before = {f: False for f in FEATURES}
    for i in range(1, N_LAM):
        cur = Kmat[i]
        if not np.array_equal(cur, prev):
            changes += 1
            for j, f in enumerate(FEATURES):
                if prev[j] and not cur[j]:
                    out_before[f] = True
                elif not prev[j] and cur[j] and out_before[f]:
                    reentries.append(f)
            prev = cur
    trace_info[tau] = (int(Kmat[0].sum()), int(Kmat[-1].sum()), changes,
                       sorted(set(reentries)))
    print(f"[trace] tau={tau:6.0f}: kept {Kmat[0].sum()} -> {Kmat[-1].sum()} along "
          f"lambda, set changes = {changes}, leave-then-return = {sorted(set(reentries))}")

K_ridge = np.abs(W_ridge) > TAU_BAND          # (N_LAM, 9)，作图用

# ---------- 2. Lasso 路径 ----------
model = Lasso(fit_intercept=True, warm_start=True, max_iter=200_000, tol=1e-6)
W = []
for lam in lams[:0:-1]:
    model.alpha = float(lam)
    model.fit(X, y)
    W.append(model.coef_.copy())
W_lasso = np.vstack([LinearRegression().fit(X, y).coef_, np.array(W)[::-1]])
K_lasso = np.abs(W_lasso) > NZ_EPS

print("\nlasso zero-entry (lambda*, beyond it coef == 0), sorted:")
for lam0, f in sorted((lams[np.abs(W_lasso[:, j]) > NZ_EPS].max(), FEATURES[j])
                      for j in range(len(FEATURES))):
    print(f"  {f:13s} lambda* = {lam0:7.0f}")

viol = 0
for j in range(len(FEATURES)):
    nz = K_lasso[:, j]
    first_zero = np.argmax(~nz) if (~nz).any() else N_LAM
    if nz[first_zero:].sum() != 0:
        viol += 1
print(f"[trace] lasso nonzero-set nested (monotone in lambda) violations: {viol}")
print(f"lasso @2e4 nonzero: {kept(W_lasso[I20], NZ_EPS)}")
print(f"lasso @1e5 nonzero: {kept(W_lasso[I100], NZ_EPS)}  "
      f"(sqft_living = {W_lasso[I100, JL]:+.0f}, sqft_above = {W_lasso[I100, JA]:+.0f})")

# ---------- 图 2：路径 + 阈值带 & 保留计数曲线 ----------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.8, 7.6), sharex=True,
                               gridspec_kw={"height_ratios": [3, 1.1]})
for j, f in enumerate(FEATURES):
    ax1.plot(lams, W_ridge[:, j], color=COLORS[j], marker="o", markersize=3.2,
             markevery=5, mfc="white", lw=1.2, label=f)
ax1.axhspan(-TAU_BAND, TAU_BAND, color="magenta", alpha=0.07)
for yline in (TAU_BAND, -TAU_BAND):
    ax1.axhline(yline, color="magenta", ls=":", lw=1.6)
ax1.axvline(LAM_STORY, color="gray", ls="--", lw=1)
ax1.text(0.985, 0.06, f"band |w|<{TAU_BAND:.0f} treated as 0",
         transform=ax1.transAxes, ha="right", va="bottom", color="magenta", fontsize=9)
ax1.axhline(0, color="k", lw=0.6, alpha=0.4)
ax1.set_ylabel(r"ridge coefficients $\hat{w}_j$")
_c, _r = trace_info[TAU_BAND][2], trace_info[TAU_BAND][3]
ax1.set_title(f"Q2: ridge keep-set flips (tau={TAU_BAND:.0f}) - {_c} set changes; "
              f"{', '.join(_r)} leave & return")
ax1.legend(loc="upper right", fontsize=8.5, framealpha=0.95)
ax2.step(lams, K_ridge.sum(1), where="mid", color="tab:red", lw=1.6,
         label=f"ridge: #{{|w|>{TAU_BAND:.0f}}} selected")
ax2.step(lams, K_lasso.sum(1), where="mid", color="tab:blue", lw=1.6,
         label="lasso: #nonzero (exact zeros)")
ax2.set_xlabel(r"$\lambda$")
ax2.set_ylabel("#features")
ax2.set_ylim(-0.4, len(FEATURES) + 0.6)
ax2.legend(loc="center right", fontsize=8.5)
fig.tight_layout()
fig.savefig(FIG / "q2_ridge_band_path.png", dpi=150)
fig.savefig(FIG / "q2_ridge_band_path.pdf")
plt.close(fig)

# ---------- 图 3：选择矩阵（ridge 阈值化 vs lasso 零模式）----------
xed = np.linspace(-LAM_MAX / (N_LAM - 1) / 2, LAM_MAX + LAM_MAX / (N_LAM - 1) / 2, N_LAM + 1)
yed = np.arange(len(FEATURES) + 1)
cmap2 = ListedColormap(["#f2f2f2", "#2b6cb0"])
fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.6))
for ax, K, ttl in ((axes[0], K_ridge, f"ridge: selected = |w| > tau ({TAU_BAND:.0f})"),
                   (axes[1], K_lasso, "lasso: selected = nonzero (exact zeros)")):
    ax.pcolormesh(xed, yed, K.T.astype(float), cmap=cmap2, shading="flat", vmin=0, vmax=1)
    ax.set_yticks(np.arange(len(FEATURES)) + 0.5)
    ax.set_yticklabels(FEATURES)
    ax.set_xlabel(r"$\lambda$")
    ax.set_title(ttl, fontsize=11)
axes[0].set_ylabel("flips along lambda (not nested)")
axes[1].set_ylabel("nested blocks (monotone)")
fig.suptitle("Q2: selection along the path - ridge + threshold vs lasso", y=1.0)
fig.tight_layout()
fig.savefig(FIG / "q2_selection_matrix.png", dpi=150)
fig.savefig(FIG / "q2_selection_matrix.pdf")
plt.close(fig)

# ---------- 3. 共线性案例：同一 τ 下"整组被删" ----------
X1 = (X[:, JL] - X[:, JL].mean()).reshape(-1, 1)
w_single = svd_ridge(X1, yc, LAM_STORY)[0]
w_pair = W_ridge[I20]
print(f"\n[collinear] lambda={LAM_STORY:.0f}, single vs collinear pair:")
print(f"  sqft_living alone                        : w = {w_single:+9.0f}")
print(f"  with sibling : sqft_living = {w_pair[JL]:+9.0f}, "
      f"sqft_above = {w_pair[JA]:+9.0f}, pair sum = {w_pair[JL] + w_pair[JA]:+9.0f}")
for tau in (TAU_BAND, 5e4, TAU_DEL):
    print(f"  tau={tau:6.0f}: single kept: {abs(w_single) > tau}; "
          f"pair kept: living {abs(w_pair[JL]) > tau}, above {abs(w_pair[JA]) > tau}; "
          f"full model kept({len(kept(w_pair, tau))}) = {{{', '.join(kept(w_pair, tau))}}}")

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.6, 5.4))
valsL = [w_single, w_pair[JL], w_pair[JA]]
colsL = ["green", "green", "darkorange"]
xs = np.arange(3)
axL.bar(xs, valsL, color=colsL, width=0.6)
axL.axhline(TAU_DEL, color="magenta", ls=":", lw=1.8)
axL.text(2.45, TAU_DEL * 1.03, f"tau = {TAU_DEL:.0f}", color="magenta",
         ha="right", fontsize=10)
axL.set_xticks(xs)
axL.set_xticklabels(["sqft_living\nalone", "sqft_living\nwith sibling",
                     "sqft_above\nwith sibling"], fontsize=9)
axL.set_ylabel("ridge coefficient  ($ / sigma)")
axL.set_title(f"Real data (lambda={LAM_STORY:.0f}): adding one collinear sibling\n"
              "halves each coefficient -> same tau drops the whole group")
for x, v in zip(xs, valsL):
    axL.text(x, v + 4000, f"{v:,.0f}", ha="center", fontsize=9)

# 合成复制特征（x2 = x1）
X2 = np.column_stack([X[:, JL], X[:, JL]])
w_dup = svd_ridge(X2 - X2.mean(0), yc, LAM_STORY)
w_l_dup = Lasso(alpha=LAM_STORY, max_iter=200_000, tol=1e-8).fit(X2, y).coef_
print(f"\n[duplicate] synthetic x2 = x1, lambda={LAM_STORY:.0f}:")
print(f"  ridge (w1, w2) = ({w_dup[0]:.0f}, {w_dup[1]:.0f})   ~ w/2 each")
print(f"  lasso (w1, w2) = ({w_l_dup[0]:.0f}, {w_l_dup[1]:.0f})   winner takes all")

valsR = [w_dup[0], w_dup[1], w_l_dup[0], w_l_dup[1]]
colsR = ["green", "darkorange", "green", "darkorange"]
xs2 = np.arange(4)
axR.bar(xs2, valsR, color=colsR, width=0.6)
axR.axhline(TAU_DEL, color="magenta", ls=":", lw=1.8)
axR.text(3.45, TAU_DEL * 1.03, f"tau = {TAU_DEL:.0f}", color="magenta",
         ha="right", fontsize=10)
axR.set_xticks(xs2)
axR.set_xticklabels(["ridge\nx1", "ridge\nx2 (=x1)", "lasso\nx1", "lasso\nx2 (=x1)"],
                    fontsize=9)
axR.set_title("Synthetic duplicates (x2 = x1): ridge splits the weight\n"
              "(both < tau), lasso keeps one representative")
axR.set_ylabel("coefficient  ($ / sigma)")
for x, v in zip(xs2, valsR):
    axR.text(x, v + 4000, f"{v:,.0f}", ha="center", fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "q2_collinear_case.png", dpi=150)
fig.savefig(FIG / "q2_collinear_case.pdf")
plt.close(fig)

# ---------- 图 4：课件式条形图 @ story lambda ----------
fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.2), sharey=True)
for ax, w, ttl in ((axes[0], W_ridge[I20], f"ridge @ lambda={LAM_STORY:.0f}"),
                   (axes[1], W_lasso[I20], f"lasso @ lambda={LAM_STORY:.0f}")):
    ax.bar(range(len(FEATURES)), w, color="tab:blue", width=0.55)
    ax.axhline(TAU_BAND, color="magenta", ls=":", lw=1.6)
    ax.axhline(-TAU_BAND, color="magenta", ls=":", lw=1.6)
    ax.axhspan(-TAU_BAND, TAU_BAND, color="magenta", alpha=0.06)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(FEATURES)))
    ax.set_xticklabels(FEATURES, rotation=40, ha="right", fontsize=9)
    ax.set_title(ttl, fontsize=11)
    ax.set_ylim(-1.3e5, 3.1e5)
axes[0].set_ylabel("coefficient  ($ / sigma)")
fig.suptitle(f"Q2 (slides p43 style): bars + threshold band tau={TAU_BAND:.0f} - "
             f"ridge selected {int(K_ridge[I20].sum())}, lasso selected {int(K_lasso[I20].sum())}",
             y=0.99)
fig.tight_layout()
fig.savefig(FIG / "q2_bars_story.png", dpi=150)
fig.savefig(FIG / "q2_bars_story.pdf")
plt.close(fig)

# ---------- 4. 预测代价：train/valid 拆分 ----------
rng = np.random.RandomState(0)
idx = rng.permutation(len(y))
ntr = int(len(y) * 0.7)
tr, va = idx[:ntr], idx[ntr:]
mu, sd = X_raw[tr].mean(0), X_raw[tr].std(0)
Xtr, Xva = (X_raw[tr] - mu) / sd, (X_raw[va] - mu) / sd
ytr, yva = y[tr], y[va]


def r2(pred):
    return 1 - ((yva - pred) ** 2).sum() / ((yva - yva.mean()) ** 2).sum()


def refit_r2(cols):
    if not cols:
        return r2(np.full(len(yva), ytr.mean()))
    lr = LinearRegression().fit(Xtr[:, cols], ytr)
    return r2(lr.predict(Xva[:, cols]))


print("\n[train/valid, n_train=%d, n_valid=%d] valid R^2 after OLS refit:" % (ntr, len(va)))
print(f"  all 9 features                    : {refit_r2(list(range(9))):.4f}")
w_rd = Ridge(alpha=LAM_STORY).fit(Xtr, ytr).coef_
for tau in (TAU_BAND, 5e4, TAU_DEL):
    sel = [j for j in range(len(FEATURES)) if abs(w_rd[j]) > tau]
    print(f"  ridge+tau={tau:6.0f} kept({len(sel)}) = {sorted(FEATURES[j] for j in sel)}"
          f" -> R2 = {refit_r2(sel):.4f}")
for lam in (LAM_STORY, 1e5):
    ls = Lasso(alpha=lam, max_iter=200_000, tol=1e-6).fit(Xtr, ytr)
    sel = [j for j in range(len(FEATURES)) if abs(ls.coef_[j]) > NZ_EPS]
    print(f"  lasso  lam={lam:6.0f} kept({len(sel)}) = {sorted(FEATURES[j] for j in sel)}"
          f" -> R2 = {refit_r2(sel):.4f}")
print(f"  sqft_living only                  : R2 = {refit_r2([JL]):.4f}")

# 合成复制特征的预测代价
X2tr = np.column_stack([Xtr[:, JL], Xtr[:, JL]])
X2va = np.column_stack([Xva[:, JL], Xva[:, JL]])
w_dup_tr = Ridge(alpha=LAM_STORY).fit(X2tr, ytr).coef_
keep_idx = [j for j in (0, 1) if abs(w_dup_tr[j]) > TAU_DEL]
if keep_idx:
    pred = LinearRegression().fit(X2tr[:, keep_idx], ytr).predict(X2va[:, keep_idx])
else:
    pred = np.full(len(yva), ytr.mean())
print(f"\n[synthetic dup, train] ridge coefs ({w_dup_tr[0]:.0f}, {w_dup_tr[1]:.0f}), "
      f"kept under tau={TAU_DEL:.0f}: {len(keep_idx)} -> valid R2 = {r2(pred):.4f}")

# ---------- 5. Bootstrap 稳定性 ----------
B = 40
rs = np.random.RandomState(1)
keepf = {t: {f: 0 for f in FEATURES} for t in (TAU_BAND, TAU_DEL)}
nzf = {lam: {f: 0 for f in FEATURES} for lam in (LAM_STORY, 1e5)}
pair_cnt = {t: [0, 0, 0] for t in (TAU_BAND, TAU_DEL)}
split_more = 0
pair_l = 0
for b in range(B):
    sub = rs.randint(0, len(X), len(X) // 2)
    Xb, yb = X[sub], y[sub]
    w = svd_ridge(Xb - Xb.mean(0), yb - yb.mean(), LAM_STORY)
    for t in (TAU_BAND, TAU_DEL):
        for j, f in enumerate(FEATURES):
            if abs(w[j]) > t:
                keepf[t][f] += 1
        cnt = int(abs(w[JL]) > t) + int(abs(w[JA]) > t)
        pair_cnt[t][cnt] += 1
    split_more += abs(w[JL]) > abs(w[JA])
    for lam in (LAM_STORY, 1e5):
        lm = Lasso(alpha=lam, max_iter=50_000, tol=1e-5).fit(Xb, yb)
        for j, f in enumerate(FEATURES):
            if abs(lm.coef_[j]) > NZ_EPS:
                nzf[lam][f] += 1
    lm1 = Lasso(alpha=1e5, max_iter=50_000, tol=1e-5).fit(Xb, yb)
    pair_l += (abs(lm1.coef_[JL]) > NZ_EPS) or (abs(lm1.coef_[JA]) > NZ_EPS)

print(f"\n[bootstrap B={B}, half-samples] selection frequency:")
hdr = f"  {'feature':13s} | {'ridge>1e4':>9s} {'ridge>1e5':>9s} | {'lasso@2e4':>9s} {'lasso@1e5':>9s}"
print(hdr)
for f in FEATURES:
    print(f"  {f:13s} | {keepf[TAU_BAND][f] / B:9.0%} {keepf[TAU_DEL][f] / B:9.0%} | "
          f"{nzf[LAM_STORY][f] / B:9.0%} {nzf[1e5][f] / B:9.0%}")
for t in (TAU_BAND, TAU_DEL):
    n0_, n1_, n2_ = pair_cnt[t]
    print(f"  ridge tau={t:6.0f}: pair {{living,above}} kept none/one/both = "
          f"{n0_ / B:.0%} / {n1_ / B:.0%} / {n2_ / B:.0%}")
print(f"  real pair: |living| > |above| in {split_more / B:.0%} of half-sample refits")
print(f"  lasso@1e5: pair represented (>=1 nonzero) in {pair_l / B:.0%} of runs")

print("\nfigures saved to", FIG)
