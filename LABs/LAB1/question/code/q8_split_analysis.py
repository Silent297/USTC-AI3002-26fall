# -*- coding: utf-8 -*-
"""Q8：训练集与验证集的分布差异，以及我们的应对。

题面：仔细阅读训练集和验证集，模型的训练集和验证集存在哪些分布上的差异？你是如何应对的？

对象：truth（主）train 5088 / valid 300 句真假分类（句尾 token 表征）；world（对照）train 500 / valid 130。

[1] 分布差异（数字全部重算）：
    D1 噪声：相邻字母交换型错拼（typo）句率，truth train ~7% vs valid ~25%；
    D2 主体新颖度：valid 首实体（lead entity）未在 train 出现过的占比 ~77%，
       零假设 = 从 train 随机抽 300 条做同样统计（200 次重抽）~21% [15.6, 25.6]；
       稳健性 = "typo 落在主体内"的句子逐条核查（仅数行，修正后多数仍新颖）；
    D3 模板/长度：train 含 "One can"/"If one" 句式 ~3.0% vs valid ~0.3%；句长上限 487 vs 162；
    D4 world 对照：无错拼差异；20/130 来自 train 未见国家、126/130 地点字符串未见过
       （但多数距同国训练点 <3°，字符串新颖 != 空间新颖）。

[2] 探针分层评估：最终模型 truth L20 探针（锚点 valid acc 0.8967），
    在 valid 上按 typo x 主体见/未见 x 标签 分层，bootstrap 95% CI；
    typo 落在首实体内部 vs 句内其它位置的对比。

[3] 逐层 typo 敏感度：13 层探针在 干净/带 typo 子集 上的 acc（锚点对照训练报告值）。

[4] 应对实验：随机 3 折 CV vs 按主体分组 3 折 CV（仓库 model.py，L20 特征，
    lr=5e-3 / l1=l2=1e-3 / 5000 ep，两方案同行集同预算）——检验"随机划分是否高估泛化"。
    本数据实测 gap = -0.008（两种口径几乎无差），解读见 figures/q8_figure_notes.md §3。
    Q8_SKIP_CV=1 跳过（快速自检）；Q8_EPOCHS 覆盖轮数。

输出：question/figures/q8_{split_diff,stratified_acc,cv_gap}.{png,pdf}
"""
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
FIG = ROOT / "question" / "figures"
FIG.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT))

from spellchecker import SpellChecker  # noqa: E402

N_BOOT = 2000
rng = np.random.default_rng(0)
sp = SpellChecker(distance=1)

TTR = pd.read_csv(ROOT / "datasets/truth/train.csv")
TVA = pd.read_csv(ROOT / "datasets/truth/valid.csv")
WTR = pd.read_csv(ROOT / "datasets/world/train.csv")
WVA = pd.read_csv(ROOT / "datasets/world/valid.csv")

LAYERS = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 34, 35]
EXP_ACC = {2: 0.5667, 5: 0.5933, 8: 0.6567, 11: 0.7400, 14: 0.7900, 17: 0.8733,
           20: 0.8967, 23: 0.8900, 26: 0.8600, 29: 0.8600, 32: 0.8367,
           34: 0.7867, 35: 0.6867}

RED, BLUE, GRAY, ORANGE = "tab:red", "tab:blue", "0.45", "tab:orange"


def weights_path(layer):
    for sub in ["truth_a", "truth_b", "truth_c", "truth_d"]:
        p = ROOT / f"results/slurm_layers/{sub}/weights/truth_layer{layer}/weights.npz"
        if p.exists():
            return p
    raise FileNotFoundError(layer)


def sigmoid(z):
    return np.exp(-np.logaddexp(0.0, -z))


# ---------------- 工具：错拼检测 / 首实体 ----------------
def detect_typos(df):
    """相邻字母交换型错拼检测。返回 (句子级 flags, 类型表 w->sw, 每句命中 [(start,end,w,sw)])。"""
    flags = np.zeros(len(df), bool)
    types = {}
    hits = [[] for _ in range(len(df))]
    for j, s in enumerate(df["sentence"]):
        for mt in re.finditer(r"[A-Za-z']+", s):
            w = mt.group(0).lower()
            if len(w) < 4 or not w.isalpha() or w in sp:
                continue
            for i in range(len(w) - 1):
                sw = w[:i] + w[i + 1] + w[i] + w[i + 2:]
                if sw != w and sw in sp:
                    types[w] = sw
                    flags[j] = True
                    hits[j].append((mt.start(), mt.end(), w, sw))
                    break
    return flags, types, hits


def fix_sentence(s, types):
    out, last = [], 0
    for mt in re.finditer(r"[A-Za-z']+", s):
        lw = mt.group(0).lower()
        if len(lw) >= 4 and lw.isalpha() and lw in types:
            rep = types[lw]
            if mt.group(0)[0].isupper():
                rep = rep[0].upper() + rep[1:]
            out.append(s[last:mt.start()]); out.append(rep); last = mt.end()
    out.append(s[last:])
    return "".join(out)


STOP = {"The", "A", "An", "Many", "One", "You", "It", "If", "There", "People", "Some", "In",
        "When", "Because", "After", "This", "That", "They", "Their", "These", "Those", "However",
        "Most", "More", "Both", "Each", "All", "No", "Not", "While", "During", "Although",
        "Before", "Since", "Through", "Using", "Use", "Founded", "Two", "Three", "Four", "Five"}
ENT_RE = re.compile(r"^((?:[A-Z][A-Za-z'\-\.]+)(?: [A-Z][A-Za-z'\-\.]+)*)")


def lead_ent(s):
    m = ENT_RE.match(s)
    if not m:
        return None
    e = m.group(1)
    return None if e in STOP else e


def novelty_frac(sub_sentences, pool_sentences):
    pool = "\n".join(pool_sentences)
    cnt = {}
    for s in sub_sentences:
        e = lead_ent(s)
        if e is not None:
            cnt[e] = cnt.get(e, 0) + 1
    tot = sum(cnt.values())
    nu = sum(c for e, c in cnt.items() if e not in pool)
    return tot, nu, nu / tot


def acc_ci(correct, n=N_BOOT):
    v = correct.astype(np.float64)
    idx = rng.integers(0, len(v), (n, len(v)))
    m = v[idx].mean(axis=1)
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def diff_ci(c1, c2, n=N_BOOT):
    i1 = rng.integers(0, len(c1), (n, len(c1)))
    i2 = rng.integers(0, len(c2), (n, len(c2)))
    d = c1[i1].astype(np.float64).mean(1) - c2[i2].astype(np.float64).mean(1)
    return float(c1.mean() - c2.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


# ================= [1] 分布差异 =================
print("=" * 78)
print("[1] train/valid distribution differences (truth primary, world contrast)")
f_tr, ty_tr, hits_tr = detect_typos(TTR)
f_va, ty_va, hits_va = detect_typos(TVA)
fw_tr, tyw_tr, _ = detect_typos(WTR)
fw_va, tyw_va, _ = detect_typos(WVA)

lab_tr = TTR["label"].to_numpy()
lab_va = TVA["label"].to_numpy()
print(f"  label balance: train {lab_tr.mean():.3f} positive / valid {lab_va.mean():.3f}")
for nm, df, fl, ty in [("truth train", TTR, f_tr, ty_tr), ("truth valid", TVA, f_va, ty_va)]:
    lab = df["label"].to_numpy()
    print(f"  [{nm}] typo sentences {fl.sum()}/{len(df)} ({100 * fl.mean():.1f}%) | "
          f"label=1 {100 * fl[lab == 1].mean():.1f}% | label=0 {100 * fl[lab == 0].mean():.1f}% | "
          f"distinct typo word types {len(ty)}")
print("  truth train typo types (first 12):",
      ", ".join(f"{w}->{s}" for w, s in list(ty_tr.items())[:12]))
print("  truth valid typo types (first 12):",
      ", ".join(f"{w}->{s}" for w, s in list(ty_va.items())[:12]))
print(f"  world typo sentences: train {fw_tr.sum()}/{len(WTR)} ({100 * fw_tr.mean():.1f}%) | "
      f"valid {fw_va.sum()}/{len(WVA)} ({100 * fw_va.mean():.1f}%)")
for i in np.where(fw_tr)[0][:2]:
    print(f"    (world train flag: {WTR['sentence'].iloc[i]})")
for i in np.where(f_va)[0][:5]:
    print(f"    (truth valid example: {TVA['sentence'].iloc[i]})")

# D2 主体新颖度
tr_sents = TTR["sentence"].tolist()
va_sents = TVA["sentence"].tolist()
n_id_va, n_new_va, frac_va = novelty_frac(va_sents, tr_sents)
n_id_tr = sum(1 for s in tr_sents if lead_ent(s) is not None)
print(f"\n  [D2] valid lead-entity novelty: {n_id_va}/{len(TVA)} identifiable, "
      f"unseen-in-train {n_new_va} -> {100 * frac_va:.1f}%")
print(f"       (train identifiable lead entities: {n_id_tr}/{len(TTR)})")
N_REP = 200
null_fracs = np.empty(N_REP)
for r in range(N_REP):
    idx = rng.permutation(len(TTR))
    *_, null_fracs[r] = novelty_frac([tr_sents[i] for i in idx[:300]],
                                     [tr_sents[i] for i in idx[300:]])
nl, nh = np.percentile(null_fracs, [2.5, 97.5])
verdict = "FAR outside" if frac_va > nh else "within"
print(f"       null (random 300-of-train, {N_REP} reps): mean {100 * null_fracs.mean():.1f}% "
      f"[{100 * nl:.1f}, {100 * nh:.1f}]  -> valid is {verdict} the null")
tr_pool_raw = "\n".join(tr_sents)
n_flip = 0
print("       in-subject typo rows (only these can mask a known subject):")
for i in np.where(f_va)[0]:
    m = ENT_RE.match(va_sents[i])
    if m is None or not any(h[0] < m.end(1) for h in hits_va[i]):
        continue
    e = lead_ent(va_sents[i])
    if e is None:
        continue
    ec = fix_sentence(e, ty_va)
    if e in tr_pool_raw:
        rec = "raw-in-train (detector FP on a proper noun)"
    elif ec in tr_pool_raw:
        rec = f"correction-recoverable ('{ec}' in train)"
        n_flip += 1
    else:
        rec = "still novel"
    print(f"         {e!r} -> {ec!r}: {rec}")
frac_va_adj = (n_new_va - n_flip) / n_id_va
print(f"       adjusted for subject-typo recovery: unseen {n_new_va} -> {n_new_va - n_flip} "
      f"({100 * frac_va_adj:.1f}%)")
top_ents = pd.Series([lead_ent(s) for s in va_sents]).dropna()
top_new = top_ents[~top_ents.isin(set(lead_ent(s) for s in tr_sents if lead_ent(s)))]
print("       valid top entities unseen in train:", top_new.value_counts().head(8).to_dict())

# D3 模板/长度
for tmpl in ["One can", "If one"]:
    print(f"  [D3] template '{tmpl}': train {TTR.sentence.str.contains(tmpl).sum()}/{len(TTR)} "
          f"({100 * TTR.sentence.str.contains(tmpl).mean():.1f}%) vs valid "
          f"{TVA.sentence.str.contains(tmpl).sum()}/{len(TVA)}")
Ltr = TTR.sentence.str.len().to_numpy()
Lva = TVA.sentence.str.len().to_numpy()
print(f"       length: train mean {Ltr.mean():.1f} p95 {np.percentile(Ltr, 95):.0f} max {Ltr.max()} | "
      f"valid mean {Lva.mean():.1f} p95 {np.percentile(Lva, 95):.0f} max {Lva.max()}")

# D4 world 对照
wp_re = re.compile(r"^What's the longitude and latitude of (.+?), (.+)\?$")
tr_places = {wp_re.match(s).group(1) for s in WTR.sentence if wp_re.match(s)}
va_places = [wp_re.match(s).group(1) if wp_re.match(s) else None for s in WVA.sentence]
print(f"\n  [D4] world template match: train {sum(1 for s in WTR.sentence if wp_re.match(s))}/{len(WTR)} "
      f"| valid {sum(1 for p in va_places if p)}/{len(WVA)}")
tr_countries = set(WTR["country"])
new_ctry = (~WVA["country"].isin(tr_countries)).to_numpy()
new_place = np.array([p is not None and p not in tr_places for p in va_places])
# 到同国最近训练点的距离（q7 口径）
tr_coords = WTR[["latitude", "longitude"]].to_numpy()
grp = {c: g[["latitude", "longitude"]].to_numpy()
       for c, g in WTR.groupby("country")}
va_coords = WVA[["latitude", "longitude"]].to_numpy()
nn_d = np.empty(len(WVA))
for i in range(len(WVA)):
    pool = grp.get(WVA["country"].iloc[i], tr_coords)
    nn_d[i] = np.sqrt(((pool - va_coords[i]) ** 2).sum(1)).min()
newfar = new_place & (nn_d >= 3)
print(f"       new country: {new_ctry.sum()}/{len(WVA)} | new place string: {new_place.sum()}/{len(WVA)}"
      f" | new string AND >=3 deg from same-country train point: {newfar.sum()}")

# ================= [2] L20 探针分层评估 =================
print("\n" + "=" * 78)
print("[2] L20 probe stratified accuracy on valid (anchor 0.8967)")
d20 = np.load(ROOT / "results/slurm_layers/truth_b/weights/truth_layer20/weights.npz")
w20, b20 = d20["weights"], d20["bias"]
Xv20 = np.load(ROOT / "datasets/truth/valid.layer20.npy").astype(np.float64)
pred20 = (sigmoid(Xv20 @ w20 + b20) > 0.5).astype(int)
correct = pred20 == lab_va
print(f"  overall valid acc = {correct.mean():.4f} (269/300 = 0.8967 expected)")
assert abs(correct.mean() - 0.8967) < 5e-4

# 主体见/未见
ents_va = [lead_ent(s) for s in va_sents]
tr_pool = "\n".join(tr_sents)
ent_seen = np.array([e is not None and e in tr_pool for e in ents_va])
ent_na = np.array([e is None for e in ents_va])
strata = [
    ("all", np.ones(len(TVA), bool), BLUE),
    ("clean", ~f_va, BLUE),
    ("typo", f_va, RED),
    ("subject seen", ent_seen, BLUE),
    ("subject unseen", ~ent_seen & ~ent_na, RED),
    ("subject n/a", ent_na, ORANGE),
]
print("  -- main strata --")
strata_res = {}
for nm, m, c in strata:
    a, lo, hi = acc_ci(correct[m])
    strata_res[nm] = (int(m.sum()), a, lo, hi)
    print(f"  {nm:15s} n={int(m.sum()):3d}  acc={a:.3f} [{lo:.3f},{hi:.3f}]")
print("  -- label split (typo strata) --")
for lab in (1, 0):
    m = (lab_va == lab)
    ac, lo, hi = acc_ci(correct[m & ~f_va])
    at, lo2, hi2 = acc_ci(correct[m & f_va])
    print(f"  label={lab}: clean n={int((m & ~f_va).sum())} acc={ac:.3f} | "
          f"typo n={int((m & f_va).sum())} acc={at:.3f}")
print("  -- crosses: typo x subject --")
cross_res = []
for sn, sm, col in [("seen", ent_seen, BLUE), ("unseen", ~ent_seen & ~ent_na, RED)]:
    for tn, tm, _ in [("clean", ~f_va, BLUE), ("typo", f_va, RED)]:
        m = tm & sm
        if m.sum() == 0:
            continue
        a, lo, hi = acc_ci(correct[m])
        cross_res.append((f"{tn}/{sn}", int(m.sum()), a, lo, hi, col))
        print(f"  {tn:5s} x {sn:6s} n={int(m.sum()):3d}  acc={a:.3f} [{lo:.3f},{hi:.3f}]")

d, lo, hi = diff_ci(correct[~f_va], correct[f_va])
print(f"  diff clean-typo = {d:+.3f} [{lo:+.3f},{hi:+.3f}]")
d2, lo2, hi2 = diff_ci(correct[ent_seen], correct[~ent_seen & ~ent_na])
print(f"  diff seen-unseen = {d2:+.3f} [{lo2:+.3f},{hi2:+.3f}]")
print(f"  typo subset composition: seen {int((f_va & ent_seen).sum())}, "
      f"unseen {int((f_va & ~ent_seen & ~ent_na).sum())}, n/a {int((f_va & ent_na).sum())} | "
      f"clean: seen {int((~f_va & ent_seen).sum())}, unseen {int((~f_va & ~ent_seen & ~ent_na).sum())}, "
      f"n/a {int((~f_va & ent_na).sum())}")
fam = np.array([("Dave Grohl" in s) or ("Dwyane Wade" in s) for s in va_sents])
print(f"  sensitivity: typo acc excluding Grohl/Wade rows = "
      f"{correct[f_va & ~fam].mean():.3f} (n={int((f_va & ~fam).sum())})")

# typo 落点：首实体内部 vs 句内其它位置
in_ent, out_ent = [], []
for i in np.where(f_va)[0]:
    m = ENT_RE.match(va_sents[i])
    if m is None:
        continue
    (in_ent if any(h[0] < m.end(1) for h in hits_va[i]) else out_ent).append(i)
print(f"  -- typo location --")
print(f"  in lead entity:    n={len(in_ent):3d}  acc={correct[in_ent].mean():.3f}" if in_ent else "")
print(f"  elsewhere in sent: n={len(out_ent):3d}  acc={correct[out_ent].mean():.3f}" if out_ent else "")
for i in in_ent[:6]:
    print(f"    [in-ent] {lab_va[i]} {va_sents[i]}")
for i in out_ent[:3]:
    print(f"    [other ] {lab_va[i]} {va_sents[i]}")

# ================= [3] 逐层 typo 敏感度 =================
print("\n" + "=" * 78)
print("[3] per-layer typo sensitivity (13 layers)")
per_layer = {}
for L in LAYERS:
    d = np.load(weights_path(L))
    Xv = np.load(ROOT / f"datasets/truth/valid.layer{L}.npy").astype(np.float64)
    ok = (sigmoid(Xv @ d["weights"] + d["bias"]) > 0.5).astype(int) == lab_va
    a_all = ok.mean()
    a_c, c_lo, c_hi = acc_ci(ok[~f_va], n=800)
    a_t, t_lo, t_hi = acc_ci(ok[f_va], n=800)
    per_layer[L] = (a_all, a_c, a_t, c_lo, c_hi, t_lo, t_hi)
    flag = "OK" if abs(a_all - EXP_ACC[L]) < 1e-4 else f"MISMATCH expected {EXP_ACC[L]}"
    print(f"  L{L:2d}: all {a_all:.4f} [{flag}] | clean {a_c:.3f} | typo {a_t:.3f} | "
          f"gap {a_c - a_t:+.3f}")

# ================= [4] 随机 vs 分组 CV =================
skip_cv = os.environ.get("Q8_SKIP_CV", "") not in ("", "0")
cv_res = None
if not skip_cv:
    print("\n" + "=" * 78)
    from sklearn.model_selection import GroupKFold, KFold
    from model import LogisticRegressionModel

    Xtr20 = np.load(ROOT / "datasets/truth/train.layer20.npy").astype(np.float64)
    ytr = lab_tr.astype(np.float64)
    ents_tr = [lead_ent(s) for s in tr_sents]
    ident = np.array([e is not None for e in ents_tr])
    X_id, y_id = Xtr20[ident], ytr[ident]
    g_id = np.array([e for e in ents_tr if e is not None], dtype=object)
    EP = int(os.environ.get("Q8_EPOCHS", "5000"))
    print(f"[4] 3-fold CV, rows={int(ident.sum())}/{len(ident)} identifiable, epochs={EP}, "
          f"lr=5e-3 l1=l2=1e-3 (repo model.py)")

    def run_scheme(kind):
        folds = (list(KFold(3, shuffle=True, random_state=0).split(X_id)) if kind == "random"
                 else list(GroupKFold(3).split(X_id, y_id, g_id)))
        out = []
        for k, (it, ie) in enumerate(folds):
            t0 = time.time()
            m = LogisticRegressionModel(X_id.shape[1])
            m.fit(X_id[it], y_id[it], lr=5e-3, epochs=EP, lambda1=1e-3, lambda2=1e-3)
            ok_te = ((m.predict(X_id[ie]) > 0.5).astype(int) == y_id[ie])
            ok_tr = ((m.predict(X_id[it]) > 0.5).astype(int) == y_id[it])
            gset = set(g_id[it])
            un = float(np.mean([g not in gset for g in g_id[ie]]))
            rec = dict(k=k, n_tr=len(it), n_te=len(ie), train_acc=float(ok_tr.mean()),
                       test_acc=float(ok_te.mean()), unseen=un, secs=time.time() - t0,
                       diverged=bool(m.diverged))
            out.append(rec)
            print(f"  [{kind:7s}] fold{k}: n_tr={len(it)} n_te={len(ie)} "
                  f"train_acc={rec['train_acc']:.3f} test_acc={rec['test_acc']:.3f} "
                  f"test-subj-unseen={100 * un:.0f}% ({rec['secs']:.0f}s, diverge={rec['diverged']})")
        accs = [r["test_acc"] for r in out]
        print(f"  [{kind:7s}] mean test acc = {np.mean(accs):.4f} +- {np.std(accs):.4f}")
        return out

    cv_random = run_scheme("random")
    cv_grouped = run_scheme("grouped")
    cv_res = (cv_random, cv_grouped)
    print(f"  gap random - grouped = {np.mean([r['test_acc'] for r in cv_random]) - np.mean([r['test_acc'] for r in cv_grouped]):+.4f}")

# ================= 图 1：分布差异 =================
fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.4))

ax = axes[0, 0]
vals = [100 * f_tr.mean(), 100 * f_va.mean(), 100 * fw_tr.mean(), 100 * fw_va.mean()]
ns = [(f_tr.sum(), len(TTR)), (f_va.sum(), len(TVA)), (fw_tr.sum(), len(WTR)), (fw_va.sum(), len(WVA))]
cols = [GRAY, RED, GRAY, RED]
ax.bar(range(4), vals, color=cols, width=0.62)
for x, (v, n) in enumerate(zip(vals, ns)):
    ax.text(x, v + 0.7, f"{v:.1f}%\n{n[0]}/{n[1]}", ha="center", fontsize=9)
ax.set_xticks(range(4))
ax.set_xticklabels(["truth\ntrain", "truth\nvalid", "world\ntrain", "world\nvalid"], fontsize=9)
ax.set_ylabel("% sentences with a transposition typo")
ax.set_ylim(0, max(vals) * 1.28)
ax.set_title("(a) misspelling noise: truth valid ~3.5x train", fontsize=10)

ax = axes[0, 1]
ax.hist(100 * null_fracs, bins=18, color=BLUE, alpha=0.75,
        label=f"null: random 300 of train\n({N_REP} reps)")
ax.axvline(100 * frac_va, color=RED, lw=2.2, label=f"truth valid: {100 * frac_va:.1f}%")
if abs(frac_va_adj - frac_va) > 1e-9:
    ax.axvline(100 * frac_va_adj, color=RED, lw=1.4, ls=":",
               label=f"valid, subject-typo adjusted: {100 * frac_va_adj:.1f}%")
ax.axvline(100 * nh, color="0.35", lw=1.1, ls="--", label=f"null 97.5% = {100 * nh:.1f}%")
ax.set_xlabel("% samples with lead entity unseen in train")
ax.set_ylabel("reps")
ax.set_title("(b) subject novelty far beyond the null", fontsize=10)
ax.legend(fontsize=8)

ax = axes[1, 0]
for df, c, nm in [(TTR, GRAY, "train"), (TVA, RED, "valid")]:
    L = np.sort(df.sentence.str.len().to_numpy())
    ax.plot(L, np.arange(1, len(L) + 1) / len(L), color=c, lw=1.6, label=f"{nm} (max {L.max()})")
for tmpl, c in [("One can", BLUE), ("If one", ORANGE)]:
    k = int(TTR.sentence.str.contains(tmpl).sum())
    kv = int(TVA.sentence.str.contains(tmpl).sum())
    ax.plot([], [], color=c, lw=1.4, label=f"'{tmpl}': train {k} vs valid {kv}")
ax.set_xlabel("sentence length (characters)")
ax.set_ylabel("ECDF")
ax.set_xlim(0, 500)
ax.set_title("(c) length ECDF + template skew", fontsize=10)
ax.legend(fontsize=8)

ax = axes[1, 1]
v4 = [int(new_ctry.sum()), int(new_place.sum()), int(newfar.sum())]
ax.bar(range(3), v4, color=[RED, BLUE, ORANGE], width=0.58)
for x, v in enumerate(v4):
    ax.text(x, v + 2, str(v), ha="center", fontsize=10)
ax.set_xticks(range(3))
ax.set_xticklabels(["new country\n(absent in train)", "new place string\n(exact match)",
                    "new string &\n>3$^\\circ$ from any same-\ncountry train point"], fontsize=8.5)
ax.set_ylabel("valid samples (of 130)")
ax.set_ylim(0, 145)
ax.set_title("(d) world: string novelty, not spatial", fontsize=10)

fig.suptitle("Q8: train vs valid distribution differences", fontsize=12.5)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(FIG / "q8_split_diff.png", dpi=150)
fig.savefig(FIG / "q8_split_diff.pdf")

# ================= 图 2：探针分层 acc + 逐层敏感度 =================
fig = plt.figure(figsize=(15.8, 5.0))
ax = fig.add_subplot(1, 3, 1)
main = ["all", "clean", "typo", "subject seen", "subject unseen", "subject n/a"]
xs = np.arange(len(main))
acs = [strata_res[m][1] for m in main]
los = [strata_res[m][2] for m in main]
his = [strata_res[m][3] for m in main]
cMap = {m: c for m, _, c in strata}
ax.bar(xs, acs, color=[cMap[m] for m in main], width=0.66)
ax.errorbar(xs, acs, yerr=[np.array(acs) - np.array(los), np.array(his) - np.array(acs)],
            fmt="none", ecolor="k", capsize=3, lw=1)
for x, m in zip(xs, main):
    n, a = strata_res[m][0], strata_res[m][1]
    ax.text(x, min(a + 0.03, 1.045), f"{a:.3f}", ha="center", fontsize=8.5)
    ax.text(x, 1.065, f"n={n}", ha="center", fontsize=8, color="0.25")
ax.axhline(0.8967, color="0.3", ls=":", lw=1.1, label="overall 0.8967")
ax.set_xticks(xs)
ax.set_xticklabels([m.replace(" ", "\n", 1) for m in main], fontsize=8)
ax.set_ylim(0.6, 1.1)
ax.set_ylabel("valid accuracy (L20 probe)")
ax.set_title("(a) stratified accuracy (bootstrap 95% CI)", fontsize=10)
ax.legend(fontsize=7.5, loc="lower right")

ax = fig.add_subplot(1, 3, 2)
xs = np.arange(len(cross_res))
ax.bar(xs, [r[2] for r in cross_res], color=[r[5] for r in cross_res], width=0.62)
ax.errorbar(xs, [r[2] for r in cross_res],
            yerr=[[r[2] - r[3] for r in cross_res], [r[4] - r[2] for r in cross_res]],
            fmt="none", ecolor="k", capsize=3, lw=1)
for x, r in zip(xs, cross_res):
    ax.text(x, min(r[2] + 0.03, 1.045), f"{r[2]:.3f}", ha="center", fontsize=8.5)
    ax.text(x, 1.065, f"n={r[1]}", ha="center", fontsize=8, color="0.25")
ax.axhline(0.8967, color="0.3", ls=":", lw=1.1, label="overall 0.8967")
ax.set_xticks(xs)
ax.set_xticklabels([r[0].replace("/", "\n") for r in cross_res], fontsize=8)
ax.set_ylim(0.6, 1.1)
ax.set_ylabel("valid accuracy")
ax.set_title("(b) typo x subject cross", fontsize=10)
ax.legend(fontsize=7.5, loc="lower right")

ax = fig.add_subplot(1, 3, 3)
xi = np.arange(len(LAYERS))
a_c = np.array([per_layer[L][1] for L in LAYERS])
a_t = np.array([per_layer[L][2] for L in LAYERS])
ax.fill_between(xi, [per_layer[L][3] for L in LAYERS], [per_layer[L][4] for L in LAYERS],
                color=BLUE, alpha=0.18)
ax.fill_between(xi, [per_layer[L][5] for L in LAYERS], [per_layer[L][6] for L in LAYERS],
                color=RED, alpha=0.18)
ax.plot(xi, a_c, "-o", color=BLUE, lw=1.8, ms=4.5, label=f"clean (n={int((~f_va).sum())})")
ax.plot(xi, a_t, "-s", color=RED, lw=1.8, ms=4.5, label=f"typo (n={int(f_va.sum())})")
ax.plot(xi, [per_layer[L][0] for L in LAYERS], "--", color="0.45", lw=1.4, label="all valid")
ax.set_xticks(xi)
ax.set_xticklabels(LAYERS, fontsize=8)
ax.set_xlabel("layer")
ax.set_ylabel("valid accuracy")
ax.set_ylim(0.4, 1.02)
ax.legend(fontsize=8)
ax.set_title("(c) typo sensitivity across layers", fontsize=10)

fig.suptitle("Q8: probe behavior on the OOD valid set", fontsize=12.5)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(FIG / "q8_stratified_acc.png", dpi=150)
fig.savefig(FIG / "q8_stratified_acc.pdf")

# ================= 图 3：CV 对照 =================
if cv_res is not None:
    cv_random, cv_grouped = cv_res
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    xs = np.arange(3)
    r_acc = [r["test_acc"] for r in cv_random]
    g_acc = [r["test_acc"] for r in cv_grouped]
    ax.bar(xs - 0.185, r_acc, width=0.35, color=BLUE, label="random 3-fold (rows shuffled)")
    ax.bar(xs + 0.185, g_acc, width=0.35, color=RED, label="subject-grouped 3-fold")
    for x, (a, b) in enumerate(zip(r_acc, g_acc)):
        ax.text(x - 0.185, a + 0.006, f"{a:.3f}", ha="center", fontsize=8.5)
        ax.text(x + 0.185, b + 0.006, f"{b:.3f}", ha="center", fontsize=8.5)
    mr, mg = np.mean(r_acc), np.mean(g_acc)
    ax.axhline(mr, color=BLUE, ls="--", lw=1.1)
    ax.axhline(mg, color=RED, ls="--", lw=1.1)
    ax.axhline(0.8967, color="0.25", ls=":", lw=1.4)
    ax.text(2.42, 0.8975, "actual valid 0.8967\n(OOD + typos)", fontsize=8, color="0.25", ha="right")
    un_r = np.mean([r["unseen"] for r in cv_random])
    ax.text(-0.42, 0.955, f"mean: random {mr:.3f} vs grouped {mg:.3f}  (gap {mr - mg:+.3f})\n"
                          f"test subjects unseen in train fold: random {100 * un_r:.0f}% vs grouped 100%",
            fontsize=8.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"fold {k + 1}" for k in xs])
    ax.set_ylabel("held-out accuracy")
    ax.set_ylim(0.8, 1.0)
    ax.legend(fontsize=8.5, loc="lower left")
    ax.set_title(f"Q8: 3-fold CV on train (L20 features, repo model.py, "
                 f"lr=5e-3/{int(os.environ.get('Q8_EPOCHS', '5000'))} ep, n={int(ident.sum())})",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "q8_cv_gap.png", dpi=150)
    fig.savefig(FIG / "q8_cv_gap.pdf")

print("\nsaved:", FIG / "q8_split_diff.png", "+ q8_stratified_acc.png"
      + (" + q8_cv_gap.png" if cv_res is not None else " (cv skipped)"))
