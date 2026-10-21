# -*- coding: utf-8 -*-
"""Q7：地理坐标表征 vs 国家识别 —— 国家均值基线 + 同国不同地点 + 未见国家。

模型：world raw L11 历史最优探针（results/slurm_layers/world/weights/world_layer11_extra0/
      weights.npz，standardize=false，推理不再 z-score）；pred = X·wᵀ + b，列序 (lat, lon)；
      MSE = 两坐标元素平均（与训练/前题管线一致），复算锚点 150.4277。

基线：
  oracle 国家均值 = train 中同国坐标均值（用了真值国家标签，模型没有这个优势）；
  未见国家（train 无此国）退化为全局均值兜底；全局均值基线另算。

分析集合：all 130 / seen 110 / multi(同国≥2条 valid) 73 / single 37 / unseen 20（19 国，Costa Rica 2 条）。
配对 bootstrap（10k）用于比较，报告不计置信区间。
同国内：以“该国 train 均值”为中心化，考察 Δpred 与 Δtrue 的相关（仅识别国家应为 0）。
未见国家：查表式基线无从解释，单独报告。

输出：question/figures/q7_{baseline_bars,within_country,unseen_map}.{png,pdf}
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
N_BOOT = 10000
rng = np.random.default_rng(0)

# ---------------- 数据与预测 ----------------
d = np.load(W_PATH)
w, b = d["weights"], d["bias"]
X = np.load(ROOT / "datasets/world/valid.layer11.npy").astype(np.float64)
va = pd.read_csv(ROOT / "datasets/world/valid.csv")
tr = pd.read_csv(ROOT / "datasets/world/train.csv")
y = va[["latitude", "longitude"]].to_numpy(np.float64)
pred = X @ w.T + b
mse = lambda p, t=y: float(np.mean((p - t) ** 2))
eucl = lambda p, t=y: np.sqrt(np.sum((p - t) ** 2, axis=1))
print(f"[sanity] valid MSE = {mse(pred):.4f} (anchor 150.4277)")

Xt = np.load(ROOT / "datasets/world/train.layer11.npy").astype(np.float64)
yt = tr[["latitude", "longitude"]].to_numpy(np.float64)
pt = Xt @ w.T + b
print(f"[sanity] train MSE = {mse(pt, yt):.4f}")

# ---------------- 基线 ----------------
tr_mean = tr[["latitude", "longitude"]].mean().to_numpy()
cm = tr.groupby("country")[["latitude", "longitude"]].mean()
seen = va["country"].isin(cm.index).to_numpy()
base = np.tile(tr_mean, (len(va), 1))
base[seen] = cm.loc[va.loc[seen, "country"].to_numpy()].to_numpy()
gbase = np.tile(tr_mean, (len(va), 1))
print(f"[baseline] train global mean = ({tr_mean[0]:.2f}, {tr_mean[1]:.2f})")
print(f"[baseline] global-mean MSE (all 130) = {mse(gbase):.2f}")

cva = va.groupby("country").size()
multi_countries = cva[cva >= 2].index
multi = va["country"].isin(multi_countries).to_numpy()
sets = {
    f"all (n={len(va)})": np.ones(len(va), bool),
    f"seen (n={seen.sum()})": seen,
    f"multi (n={(multi & seen).sum()})": multi & seen,
    f"single (n={(seen & ~multi).sum()})": seen & ~multi,
    f"unseen (n={(~seen).sum()})": ~seen,
}
multi_unseen = multi & ~seen
if multi_unseen.any():
    print("[note] multi-sample but unseen-in-train samples:", int(multi_unseen.sum()))

def boot_ci(vals, n=N_BOOT):
    idx = rng.integers(0, len(vals), (n, len(vals)))
    return np.percentile(vals[idx].mean(axis=1), [2.5, 97.5])

print("\n---- 分层基线对比（improvement = MSE_base - MSE_model > 0 表示模型更好）----")
rows = []
for name, s in sets.items():
    m_mse, b_mse = mse(pred[s], y[s]), mse(base[s], y[s])
    g_mse = mse(gbase[s], y[s])
    se_diff = (0.5 * ((pred - y) ** 2 - (base - y) ** 2).sum(1))[s]   # 元素平均口径
    lo, hi = boot_ci(se_diff)
    em, eb = eucl(pred)[s], eucl(base)[s]
    rows.append((name, int(s.sum()), m_mse, b_mse, g_mse, b_mse - m_mse, -hi, -lo,
                 np.median(em), np.median(eb)))
    print(f"{name:14s} model={m_mse:8.2f}  base={b_mse:8.2f}  global={g_mse:8.2f}  "
          f"imp={b_mse - m_mse:8.2f} [{-hi:7.2f},{-lo:7.2f}]  "
          f"medE model={np.median(em):5.2f} base={np.median(eb):5.2f}")

# 稳健性：multi 去掉南极洲（2 条，双方 MSE 均 ~3.4e3，主导集合均值）
sma = (multi & seen) & (va["country"] != "Antarctica").to_numpy()
print(f"[robust] multi w/o Antarctica (n={int(sma.sum())}): "
      f"model={mse(pred[sma], y[sma]):.2f} base={mse(base[sma], y[sma]):.2f}")

# 数据重叠侦察：valid 点到同国最近 train 点的距离（解释国家均值基线为何强）
tr_c = tr[["latitude", "longitude"]].to_numpy()
nn_d = np.array([
    np.sqrt((( (tr_c[(tr["country"] == va["country"].iloc[i]).to_numpy()] if
               (tr["country"] == va["country"].iloc[i]).any() else tr_c) - y[i]) ** 2).sum(1)).min()
    for i in range(len(va))])
print(f"[overlap] valid->nearest same-country train point: median {np.median(nn_d):.2f} deg, "
      f"d<1deg: {(nn_d < 1).sum()}, d<3deg: {(nn_d < 3).sum()}, of {len(nn_d)}")

u2 = va.loc[~seen].groupby("country").size()
print("[unseen] countries with >1 valid sample:", u2[u2 > 1].to_dict())

print("\n-- US / India 逐条（true vs pred）--")
for cc in ["United States", "India"]:
    s = (va["country"] == cc).to_numpy()
    for i in np.where(s)[0]:
        print(f"  {cc[:5]:5s} {va['sentence'].iloc[i][44:78]:34s} "
              f"true=({y[i, 0]:7.2f},{y[i, 1]:8.2f})  pred=({pred[i, 0]:7.2f},{pred[i, 1]:8.2f})")

print("\n-- unseen 逐条（按误差降序）--")
_un = np.where(~seen)[0]
for i in _un[np.argsort(eucl(pred)[_un])[::-1]]:
    print(f"  {va['country'].iloc[i]:22s} err={eucl(pred)[i]:6.2f}  "
          f"true=({y[i, 0]:7.2f},{y[i, 1]:8.2f})  pred=({pred[i, 0]:7.2f},{pred[i, 1]:8.2f})")

# ---------------- 同国中心化分析 ----------------
mset = multi & seen
mu = cm.loc[va.loc[mset, "country"].to_numpy()].to_numpy()
dt = y[mset] - mu
dp = pred[mset] - mu

def pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))

def spear(a, b):
    return pearson(pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy())

print(f"\n---- 同国中心化（multi∩seen, n={int(mset.sum())}）----")
for k, nm in enumerate(["lat", "lon"]):
    r, rs = pearson(dt[:, k], dp[:, k]), spear(dt[:, k], dp[:, k])
    slope = float(np.polyfit(dt[:, k], dp[:, k], 1)[0])
    idx = rng.integers(0, len(dt), (N_BOOT, len(dt)))
    rb = np.array([pearson(dt[i, k], dp[i, k]) for i in idx[:1000]])  # 1k 足够稳
    lo, hi = np.percentile(rb, [2.5, 97.5])
    print(f"  {nm}: pearson r={r:.3f} [{lo:.3f},{hi:.3f}]  spearman={rs:.3f}  slope={slope:.3f}")

same_c = []
for c in multi_countries:
    if c not in cm.index:
        continue
    s = (va["country"] == c).to_numpy()
    same_c.append((c, int(s.sum()), mse(pred[s], y[s]), mse(base[s], y[s]), s))
print("\n---- 逐国（multi∩seen，按 valid 条数降序）----")
for c, n, mm, bm, s in sorted(same_c, key=lambda r: -r[1]):
    print(f"  {c:22s} n={n}  model={mm:9.2f}  base={bm:9.2f}  imp={bm - mm:9.2f}")

# ---------------- 作图 ----------------
RED, BLUE, GRAY = "tab:red", "tab:blue", "0.45"

# 图 1：基线对比条形
fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.8, 4.5))
names = [r[0] for r in rows[:4]]
mm = [r[2] for r in rows[:4]]; bb = [r[3] for r in rows[:4]]
xs = np.arange(4)
axL.bar(xs - 0.18, mm, width=0.34, color=RED, label="our probe (raw L11)")
axL.bar(xs + 0.18, bb, width=0.34, color=BLUE, label="oracle country-mean baseline")
for x, (a, b2) in enumerate(zip(mm, bb)):
    axL.text(x - 0.18, a + 4, f"{a:.0f}", ha="center", fontsize=8.5)
    axL.text(x + 0.18, b2 + 4, f"{b2:.0f}", ha="center", fontsize=8.5)
    axL.text(x, min(a, b2) * 0.5, f"improvement\n{b2 - a:.0f}",
             ha="center", fontsize=8, color="0.25")
axL.set_xticks(xs)
axL.set_xticklabels([n.replace(" ", "\n", 1) for n in names], fontsize=9)
axL.set_ylabel("valid MSE (deg$^2$)")
axL.set_title("model vs oracle country-mean baseline\n(baseline uses the TRUE country labels)", fontsize=10)
axL.legend(fontsize=8.5)

r5 = rows[4]
axR.bar([0 - 0.18], [r5[2]], width=0.34, color=RED)
axR.bar([0 + 0.18], [r5[3]], width=0.34, color=GRAY, label="global-mean baseline")
axR.text(-0.18, r5[2] * 1.03, f"{r5[2]:.0f}", ha="center", fontsize=9)
axR.text(0.18, r5[3] * 1.03, f"{r5[3]:.0f}", ha="center", fontsize=9)
axR.text(0, r5[3] * 0.55, f"improvement\n{r5[5]:.0f}", ha="center",
         fontsize=8, color="0.25")
n_uns, n_uns_c = int((~seen).sum()), int(va.loc[~seen, "country"].nunique())
axR.set_xticks([0])
axR.set_xticklabels([f"unseen countries\n(n={n_uns})"], fontsize=9)
axR.set_title(f"{n_uns_c} valid countries absent from train\n(no country entry to look up)", fontsize=10)
axR.set_ylim(0, r5[3] * 1.25)
axR.legend(fontsize=8.5)
fig.tight_layout()
fig.savefig(FIG / "q7_baseline_bars.png", dpi=150)
fig.savefig(FIG / "q7_baseline_bars.pdf")

# 图 2：同国中心化散点 + US 局部放大
fig = plt.figure(figsize=(14.6, 4.4))
axes = [fig.add_subplot(1, 3, k + 1) for k in range(2)]
for k, (ax, nm) in enumerate(zip(axes, ["latitude", "longitude"])):
    ax.axhline(0, color="0.8", lw=0.6)
    ax.axvline(0, color="0.8", lw=0.6)
    lim = max(np.abs(dt[:, k]).max(), np.abs(dp[:, k]).max()) * 1.08
    ax.plot([-lim, lim], [-lim, lim], color="0.75", ls="--", lw=1, label="y = x")
    ax.scatter(dt[:, k], dp[:, k], s=26, color=RED, edgecolor="k", linewidths=0.3, zorder=3)
    r = pearson(dt[:, k], dp[:, k])
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel(f"true dev. from country mean ({nm}, deg)", fontsize=8.5)
    ax.set_ylabel(f"pred. dev. ({nm}, deg)", fontsize=8.5)
    ax.set_title(f"within-country variation, {nm}\npearson r = {r:.2f}  (n={int(mset.sum())})",
                 fontsize=10)
    ax.legend(fontsize=8)
axz = fig.add_subplot(1, 3, 3, projection=ccrs.PlateCarree())
us = (va["country"] == "United States").to_numpy()
us_m = (va.loc[mset, "country"] == "United States").to_numpy()
r_ul, r_un = (pearson(dt[us_m, k], dp[us_m, k]) for k in (0, 1))
coords = np.vstack([y[us], pred[us]])
axz.set_extent([coords[:, 1].min() - 6, coords[:, 1].max() + 6,
                coords[:, 0].min() - 4, coords[:, 0].max() + 4], ccrs.PlateCarree())
axz.coastlines(resolution="110m", linewidth=0.6, color="0.4")
gz = axz.gridlines(draw_labels=True, linewidth=0.25, color="0.85", alpha=0.6)
gz.top_labels = gz.right_labels = False
gz.xlabel_style = {"size": 7.5}
gz.ylabel_style = {"size": 7.5}
for i in np.where(us)[0]:
    axz.annotate("", xy=(pred[i, 1], pred[i, 0]), xytext=(y[i, 1], y[i, 0]),
                 arrowprops=dict(arrowstyle="->", color=RED, lw=1.1), zorder=4)
axz.scatter(y[us, 1], y[us, 0], s=30, facecolor="none", edgecolor="0.3", linewidths=1.0,
            transform=ccrs.PlateCarree(), zorder=3, label="ground truth")
axz.scatter(pred[us, 1], pred[us, 0], s=26, color=RED, edgecolor="k", linewidths=0.3,
            transform=ccrs.PlateCarree(), zorder=4, label="prediction")
axz.set_title(f"United States (valid, n={int(us.sum())}), "
              f"centered within country:\nr = {r_ul:.2f} (lat), {r_un:.2f} (lon)"
              f"  -> coarsely direction-correct", fontsize=9.5)
axz.legend(loc="lower left", fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "q7_within_country.png", dpi=150)
fig.savefig(FIG / "q7_within_country.pdf")

# 图 3：未见国家地图
un = ~seen
fig = plt.figure(figsize=(12.6, 6.6))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.set_global()
ax.coastlines(resolution="110m", linewidth=0.5, color="0.4")
gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="0.85", alpha=0.7)
gl.top_labels = gl.right_labels = False
gl.xlabel_style = {"size": 8}
gl.ylabel_style = {"size": 8}
for i in np.where(un)[0]:
    ax.annotate("", xy=(pred[i, 1], pred[i, 0]), xytext=(y[i, 1], y[i, 0]),
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.0, alpha=0.85),
                transform=ccrs.PlateCarree(), zorder=4)
ax.scatter(y[un, 1], y[un, 0], s=34, facecolor="none", edgecolor="0.25", linewidths=1.1,
           transform=ccrs.PlateCarree(), zorder=3, label="ground truth")
ax.scatter(pred[un, 1], pred[un, 0], s=28, color=RED, edgecolor="k", linewidths=0.3,
           transform=ccrs.PlateCarree(), zorder=4, label="prediction")
worst = np.where(un)[0][np.argsort(eucl(pred)[un])[::-1][:3]]
for i in worst:
    ax.text(pred[i, 1] + 3, pred[i, 0] - 6, f"{va['country'].iloc[i]} ({eucl(pred)[i]:.0f}$^\\circ$)",
            fontsize=8, color="0.15", transform=ccrs.PlateCarree(), zorder=5)
r5 = rows[4]
ax.set_title(f"Q7: valid samples from countries absent in training (n={int(un.sum())})\n"
             f"model MSE = {r5[2]:.0f} vs global-mean baseline {r5[3]:.0f} "
             f"(improvement +{r5[5]:.0f})",
             fontsize=11)
ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
fig.tight_layout()
fig.savefig(FIG / "q7_unseen_map.png", dpi=150)
fig.savefig(FIG / "q7_unseen_map.pdf")

print("\nsaved:", FIG / "q7_baseline_bars.png", "+ within_country + unseen_map (png/pdf)")
