# -*- coding: utf-8 -*-
"""B1：CREAK 逐句对拍（truth 数据集来源与扰动类型刻画）。

来源：《可做实验清单》B1（guide3 §3.1，同学参考报告，非官方）；2026-10-16 指派执行。

对象：本地 truth {train 5088, valid 300} 句 vs CREAK 公开数据
（Onoe et al. 2021, https://github.com/yasumasaonoe/creak,
 4 个文件 train/dev/test_without_labels/contrast_set，JSONL）。

回答四个问题：
  Q1 本地句子能否逐句追溯到 CREAK 原文？来自哪个文件？
  Q2 对不上原文的句子差在哪（编辑类型分布）？
  Q3 本地标签与 CREAK 标签是否一致（匹配后再核对，不参与检索）？
  Q4 与 Q8 错拼检测器交叉：valid 的高错拼率里多少是课程注入、多少是 CREAK 自带？

匹配协议（不使用标签检索）：
  direct 层 = whitespace 归一化（" ".join(s.split())）后逐字相等；
  near  层 = rapidfuzz fuzz.ratio >= 90 的最优候选（候选文本按归一化去重；
             margin = top1 - top2，并列记 0）；
  none  层 = 剩余未匹配（本数据实测为 0，即 100% 可追溯）；
  编辑分类以 candidate -> local 方向的单 token 替换为单位（比较时剥去首尾标点），
  分为 adjacent_transposition / agreement / typo_other / case_only / other。

输出：experiments/B1/b1_metrics.json + figures/b1_{match_tiers,source_attribution,
perturbation_dist,q8_cross}.{png,pdf}
"""
import json
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from hashlib import md5
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein
from spellchecker import SpellChecker

ROOT = Path(__file__).resolve().parent.parent.parent
B1 = Path(__file__).resolve().parent
CREAK_REPO = B1 / "data" / "creak"
CREAK_DIR = CREAK_REPO / "data" / "creak"
FIG = B1 / "figures"
FIG.mkdir(exist_ok=True)

CREAK_FILES = ["train", "dev", "test_without_labels", "contrast_set"]
CREAK_LINES = {"train": 10176, "dev": 1371, "test_without_labels": 1371, "contrast_set": 500}
NEAR_ACCEPT = 90.0
TRUTH_ROWS = {"train": 5088, "valid": 300}
LAB = {1: "true", 0: "false"}

BLUE, ORANGE, RED, GRAY = "tab:blue", "tab:orange", "tab:red", "0.45"
sp = SpellChecker(distance=1)

ANCHORS = []


def check(name, actual, expected):
    ok = actual == expected
    ANCHORS.append({"name": name, "expected": expected, "actual": actual, "pass": bool(ok)})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {actual} (expected {expected})")
    return ok


def norm(s):
    return " ".join(s.split())


# ------ 与 question/code/q8_split_analysis.py 的同名函数保持一致（交叉核对用） -*-
def detect_typos(sentences):
    """相邻字母交换型错拼检测。返回 (句子级 flags, 类型表 w->sw, 每句命中 [(start,end,w,sw)])。"""
    flags = np.zeros(len(sentences), bool)
    types = {}
    hits = [[] for _ in range(len(sentences))]
    for j, s in enumerate(sentences):
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
            out.append(s[last:mt.start()])
            out.append(rep)
            last = mt.end()
    out.append(s[last:])
    return "".join(out)


# ---------------------------------- 编辑分类 ----------------------------------
AGREEMENT = {("is", "are"), ("are", "is"), ("was", "were"), ("were", "was"),
             ("has", "have"), ("have", "has"), ("does", "do"), ("do", "does")}


def is_adjacent_swap(x, y):
    if len(x) != len(y) or len(x) < 2:
        return False
    d = [i for i in range(len(x)) if x[i] != y[i]]
    if len(d) != 2 or d[1] != d[0] + 1:
        return False
    i = d[0]
    return x[i] == y[i + 1] and x[i + 1] == y[i]


def classify_op(x, y):
    """x = CREAK 原文 token，y = 本地 token（小写并剥去首尾标点后比较，保留原文例）。"""
    xl, yl = x.lower().strip(".,!?;:\"'()[]"), y.lower().strip(".,!?;:\"'()[]")
    if xl == yl:
        return "case_only", None
    if (xl, yl) in AGREEMENT:
        return "agreement", "->".join(sorted((xl, yl)))
    if is_adjacent_swap(xl, yl):
        return "adjacent_transposition", None
    if Levenshtein.distance(xl, yl) <= 2 and abs(len(xl) - len(yl)) <= 1:
        return "typo_other", None
    return "other", None


def token_ops(cand, local):
    """tok 级比较：等长则返回 [(i, cand_tok, local_tok)]，否则 None（结构差异）。"""
    tc, tl = cand.split(" "), local.split(" ")
    if len(tc) != len(tl):
        return None
    return [(i, tc[i], tl[i]) for i in range(len(tc)) if tc[i] != tl[i]]


def family_of(cats):
    if cats is None:
        return "structural_other"
    s = set(cats)
    if not s:
        return "identical"
    typo = bool(s & {"adjacent_transposition", "typo_other"})
    agr = "agreement" in s
    if not s <= {"adjacent_transposition", "typo_other", "agreement", "case_only"}:
        return "other"
    if typo and agr:
        return "typo+agreement"
    if typo:
        return "typo_only"
    if agr:
        return "agreement_only"
    return "other"


# ---------------------------------- 主流程 ----------------------------------
def main():
    t0 = time.time()
    print("== 1/6 载入 CREAK ==")
    creak_rows, creak_meta = {}, {}
    for f in CREAK_FILES:
        p = CREAK_DIR / f"{f}.json"
        lines = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        creak_rows[f] = [json.loads(ln) for ln in lines]
        creak_meta[f] = {"lines": len(lines), "md5": md5(p.read_bytes()).hexdigest()}
        print(f"  {f}.json: {len(lines)} 行 md5={creak_meta[f]['md5'][:10]}")
        check(f"creak/{f} 行数", len(lines), CREAK_LINES[f])
    try:
        rev = subprocess.run(["git", "-C", str(CREAK_REPO), "rev-parse", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(CREAK_REPO), "status", "--porcelain"],
                               capture_output=True, text=True).stdout.strip()
    except Exception:
        rev, dirty = "unknown", ""
    print(f"  revision = {rev} (dirty={'no' if not dirty else 'yes'})")

    # 候选文本索引（按归一化去重；files 保留全部来源）
    texts = {}
    for f in CREAK_FILES:
        for r in creak_rows[f]:
            t = norm(r["sentence"])
            d = texts.setdefault(t, {"files": [], "labels": [], "ex_ids": []})
            if f not in d["files"]:
                d["files"].append(f)
            d["labels"].append(r.get("label"))
            d["ex_ids"].append(r["ex_id"])
    unique_texts = list(texts)
    print(f"  CREAK 归一化后唯一句子: {len(unique_texts)} / 原始 {sum(CREAK_LINES.values())}")
    combo_key = lambda files: "+".join(sorted(set(files)))

    print("== 2/6 载入本地 truth 并对拍 ==")
    dfs = {"train": pd.read_csv(ROOT / "datasets/truth/train.csv"),
           "valid": pd.read_csv(ROOT / "datasets/truth/valid.csv")}
    records = {}          # split -> list of per-sentence dict
    near_scores = {"train": [], "valid": []}
    near_margins = {"train": [], "valid": []}
    for split in ["train", "valid"]:
        df = dfs[split]
        assert set(df["label"].unique()) <= {0, 1}
        check(f"truth/{split} 行数", len(df), TRUTH_ROWS[split])
        check(f"truth/{split} 归一化重复句", len(df) - len(set(map(norm, df["sentence"]))), 0)
        rows = []
        for i, (sent, lab) in enumerate(zip(df["sentence"], df["label"])):
            t = norm(sent)
            local_lab = LAB[int(lab)]
            if t in texts:
                d = texts[t]
                avail = {l for l in d["labels"] if l}
                rows.append({"row": i, "tier": "direct", "score": 100.0, "margin": None,
                             "cand": t, "files": list(d["files"]), "ex_ids": list(d["ex_ids"]),
                             "label_ok": (local_lab in avail) if avail else None, "ops": None,
                             "family": "identical", "fixed_ok": None, "hits": []})
                continue
            hits = process.extract(t, unique_texts, scorer=fuzz.ratio, limit=2)
            cand, score = hits[0][0], float(hits[0][1])
            margin = float(score - hits[1][1]) if len(hits) > 1 else None
            tier = "near" if score >= NEAR_ACCEPT else "none"
            d = texts.get(cand, {"files": [], "labels": [], "ex_ids": []})
            avail = {l for l in d["labels"] if l}
            ops = token_ops(cand, t) if tier == "near" else None
            rows.append({"row": i, "tier": tier, "score": score, "margin": margin,
                         "cand": cand, "files": list(d["files"]), "ex_ids": list(d["ex_ids"]),
                         "label_ok": ((local_lab in avail) if avail else None) if tier == "near" else None,
                         "ops": [list(o) for o in ops] if ops else ops,
                         "family": family_of([classify_op(x, y)[0] for _, x, y in ops]) if ops else (
                             "structural_other" if tier == "near" else None),
                         "fixed_ok": None, "hits": []})
            if tier == "near":
                near_scores[split].append(score)
                near_margins[split].append(margin if margin is not None else float("nan"))
        records[split] = rows
        tiers = Counter(r["tier"] for r in rows)
        print(f"  {split}: direct={tiers['direct']} near={tiers['near']} none={tiers['none']}")
        check(f"{split} direct 数", tiers["direct"], {"train": 4380, "valid": 190}[split])
        check(f"{split} near 数", tiers["near"], {"train": 708, "valid": 110}[split])
        check(f"{split} none 数", tiers["none"], 0)

    # 方向核对：候选 -> 本地
    vr = dict(Counter(combo_key(r["files"]) for r in records["valid"] if r["tier"] == "direct"))
    check("valid direct 来源组合", vr, {"dev": 166, "contrast_set": 21,
                                        "contrast_set+dev": 2,
                                        "contrast_set+test_without_labels": 1})

    print("== 3/6 标签一致性 ==")
    label_check = {}
    for split in ["train", "valid"]:
        matched = [r for r in records[split] if r["tier"] in ("direct", "near")]
        conflicts = [r for r in matched if r["label_ok"] is False]
        uncheck = [r for r in matched if r["label_ok"] is None]
        label_check[split] = {"matched": len(matched), "conflicts": len(conflicts),
                              "uncheckable": len(uncheck),
                              "examples": [{"row": r["row"], "cand": r["cand"]}
                                           for r in conflicts[:5]]}
        check(f"{split} 标签冲突数", len(conflicts), 0)
        print(f"    uncheckable (候选无标签, e.g. test 文件): {len(uncheck)}")

    print("== 4/6 编辑类型分布 ==")
    ops_cls, ops_ex, ops_other, agr_pairs = {}, {}, {}, {}
    families = {}
    for split in ["train", "valid"]:
        cc, ex, ap = Counter(), Counter(), Counter()
        for r in records[split]:
            if r["tier"] != "near" or not r["ops"]:
                continue
            for _, x, y in r["ops"]:
                cat, pair = classify_op(x, y)
                cc[cat] += 1
                ex[(cat, x, y)] += 1
                if cat == "agreement":
                    ap[pair] += 1
        ops_cls[split], agr_pairs[split] = dict(cc), dict(ap)
        ops_ex[split] = [{"cat": k[0], "cand_tok": k[1], "local_tok": k[2], "n": v}
                         for k, v in sorted(ex.items(), key=lambda kv: -kv[1])[:15]]
        ops_other[split] = [{"cat": k[0], "cand_tok": k[1], "local_tok": k[2], "n": v}
                            for k, v in sorted(ex.items(), key=lambda kv: -kv[1])
                            if k[0] in ("other", "case_only")]
        families[split] = dict(Counter(r["family"] for r in records[split]
                                       if r["tier"] == "near"))
        tot = sum(cc.values())
        print(f"  {split}: ops={tot} {dict(cc)}")
        print(f"    agreement pairs: {dict(ap)}")
        print(f"    families: {families[split]}")
    tot_ops = sum(sum(v.values()) for v in ops_cls.values())
    check("near 对总编辑数", tot_ops, 846)
    if any(ops_other.values()):
        print(f"    [注意] 非两族编辑示例: {ops_other}")

    print("== 5/6 Q8 交叉 + 修复核验 ==")
    q8_cross, recovery, q8_examples = {}, {}, {}
    for split in ["train", "valid"]:
        sents = dfs[split]["sentence"].tolist()
        flags, types, hits_all = detect_typos(sents)
        print(f"  {split}: Q8 typo flags = {int(flags.sum())}/{len(flags)} = "
              f"{flags.mean():.4f}（Q8 报告值 7.0% / 24.7%）")
        check(f"{split} Q8 typo flags", int(flags.sum()), {"train": 354, "valid": 74}[split])
        rec_near_tot = rec_near_fix = 0
        cc = Counter()
        ex_v, ex_multi = [], []
        for r in records[split]:
            if r["tier"] != "near":
                continue
            fixed = fix_sentence(norm(dfs[split]["sentence"].iloc[r["row"]]), types)
            r["fixed_ok"] = bool(fixed == r["cand"])
            rec_near_tot += 1
            rec_near_fix += r["fixed_ok"]
        typo_fam = [r for r in records[split] if r["tier"] == "near" and r["family"] == "typo_only"]
        typo_fam_fix = sum(r["fixed_ok"] for r in typo_fam)
        recovery[split] = {"near": rec_near_tot, "near_fix_equal_cand": rec_near_fix,
                           "typo_only_family": len(typo_fam), "typo_only_fixed": typo_fam_fix}
        print(f"    修复核验: near={rec_near_tot} fix==cand {rec_near_fix}; "
              f"typo_only {len(typo_fam)} 中修复 {typo_fam_fix}")
        for j in range(len(sents)):
            if not flags[j]:
                continue
            r = records[split][j]
            flag_words = {h[2] for h in hits_all[j]}
            if r["tier"] == "direct":
                cc["verbatim"] += 1
                if len(ex_v) < 5:
                    w = hits_all[j][0][2] if hits_all[j] else None
                    ex_v.append({"row": j, "sentence": sents[j][:100], "typo_word": w})
            elif r["tier"] == "near":
                touched = any(any(fw in y.lower().strip(".,!?;:\"'()[]") for fw in flag_words)
                              for _, x, y in (r["ops"] or []))
                if not touched:
                    cc["creak_own_other_edits"] += 1
                elif r["fixed_ok"]:
                    cc["injected_recoverable"] += 1
                else:
                    cc["injected_other"] += 1
                    if len(ex_multi) < 8:
                        ex_multi.append({"row": j, "family": r["family"],
                                         "cand": r["cand"], "fixed_ok": r["fixed_ok"]})
            else:
                cc["unmatched"] += 1
        q8_cross[split] = dict(cc)
        q8_cross[split]["flagged"] = int(flags.sum())
        q8_examples[split] = {"verbatim": ex_v, "injected_other": ex_multi}
        print(f"    QC: near 中 typo 未被编辑触及(creak 自带): {cc.get('creak_own_other_edits', 0)}, "
              f"注入可修复 {cc.get('injected_recoverable', 0)}, 注入其他 {cc.get('injected_other', 0)}")
        check(f"{split} typo-flagged & verbatim", cc["verbatim"], {"train": 40, "valid": 6}[split])
        print(f"    cross: {dict(cc)}")

    print("== 6/6 写出 metrics 与图 ==")
    def pct(x, n):
        return round(100.0 * x / n, 2)

    def fnum(x):
        x = float(x)
        return None if not np.isfinite(x) else x

    metrics = {
        "task": "B1 CREAK sentence-level provenance matching",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": {
            "normalization": "\" \".join(s.split())",
            "direct": "normalized exact equality",
            "near": f"rapidfuzz fuzz.ratio >= {NEAR_ACCEPT}, top-2 over deduped candidate texts",
            "margin": "top1 - top2 score (0 = tie)",
            "label_used_for_retrieval": False,
            "ops_direction": "candidate(CREAK) -> local",
        },
        "creak": {"repo": "https://github.com/yasumasaonoe/creak", "revision": rev,
                  "dirty": bool(dirty), "files": creak_meta,
                  "unique_normalized_texts": len(unique_texts)},
        "truth_csv_md5": {s: md5((ROOT / f"datasets/truth/{s}.csv").read_bytes()).hexdigest()
                          for s in ["train", "valid"]},
        "tiers": {s: dict(Counter(r["tier"] for r in records[s])) for s in records},
        "tier_pct": {s: {t: pct(v, TRUTH_ROWS[s]) for t, v in
                         Counter(r["tier"] for r in records[s]).items()} for s in records},
        "sources_matched": {s: {tier: dict(Counter(
            combo_key(r["files"]) for r in records[s] if r["tier"] == tier))
            for tier in ("direct", "near")} for s in records},
        "near_detail": {s: {"n": len(near_scores[s]),
                            "score_min": fnum(np.min(near_scores[s])),
                            "score_median": fnum(np.median(near_scores[s])),
                            "score_max": fnum(np.max(near_scores[s])),
                            "margin_median": fnum(np.nanmedian(near_margins[s])),
                            "margin_min": fnum(np.nanmin(near_margins[s])),
                            "margin_zero": int(np.sum(np.array(near_margins[s]) == 0))}
                        for s in records},
        "label_check": label_check,
        "perturbation": {"ops_by_category": ops_cls, "agreement_pairs": agr_pairs,
                         "families": families, "examples": ops_ex,
                         "other_ops": ops_other},
        "recovery": recovery,
        "q8_cross": {s: dict(q8_cross[s]) for s in q8_cross},
        "q8_examples": q8_examples,
        "anchors": ANCHORS,
        "per_pair": {s: [{k: r[k] for k in ("row", "tier", "score", "margin", "cand",
                                            "files", "family", "label_ok", "fixed_ok", "ops")}
                         for r in records[s] if r["tier"] == "near"] for s in records},
    }
    (B1 / "b1_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    print("  已写 b1_metrics.json")

    draw_figures(metrics, near_scores, ops_cls, agr_pairs, families, q8_cross)
    n_fail = sum(1 for a in ANCHORS if not a["pass"])
    print(f"== 完成 {time.time() - t0:.1f}s, anchors {len(ANCHORS) - n_fail}/{len(ANCHORS)} PASS ==")
    if n_fail:
        sys.exit(1)


# ---------------------------------- 图 ----------------------------------
def draw_figures(metrics, near_scores, ops_cls, agr_pairs, families, q8_cross):
    n = TRUTH_ROWS

    # fig1 匹配层级 + near 分数 ECDF
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6))
    ax = axes[0]
    ypos = np.arange(2)
    dsc = np.array([metrics["tiers"]["train"]["direct"], metrics["tiers"]["valid"]["direct"]], float)
    nsc = np.array([metrics["tiers"]["train"]["near"], metrics["tiers"]["valid"]["near"]], float)
    tots = np.array([n["train"], n["valid"]], float)
    dp, npp = 100 * dsc / tots, 100 * nsc / tots
    ax.barh(ypos, dp, color=BLUE, label="direct (exact, normalized)")
    ax.barh(ypos, npp, left=dp, color=ORANGE, label="near (fuzz.ratio$\\geq$90)")
    for y, d, nr, dpp, nppv in zip(ypos, dsc, nsc, dp, npp):
        ax.text(dpp / 2, y, f"{int(d)} ({dpp:.1f}%)", ha="center", va="center",
                fontsize=9, color="white")
        ax.text(dpp + nppv / 2, y, f"{int(nr)} ({nppv:.1f}%)", ha="center", va="center",
                fontsize=9, color="black")
    ax.set_yticks(ypos, ["truth\ntrain", "truth\nvalid"])
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of split")
    ax.set_title("(a) provenance tiers: 100% traceable to CREAK", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax = axes[1]
    for s, c in [("train", BLUE), ("valid", ORANGE)]:
        v = np.sort(np.array(near_scores[s]))
        ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=c,
                label=f"{s} (n={len(v)})")
        ax.axvline(np.median(v), color=c, ls=":", lw=1)
    ax.set_xlabel("best-candidate fuzz.ratio")
    ax.set_ylabel("ECDF")
    ax.set_xlim(89, 100.8)
    txt = "\n".join(f"{s}: min {min(near_scores[s]):.1f} / med {np.median(near_scores[s]):.1f}"
                    for s in ["train", "valid"])
    ax.text(0.02, 0.55, txt, transform=ax.transAxes, fontsize=8.5,
            bbox=dict(fc="white", ec="0.8"))
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("(b) near-tier score distribution (all $\\geq$ 90)", fontsize=10)
    fig.suptitle("B1: local truth sentences vs CREAK (pyramid: exact / near)", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG / "b1_match_tiers.png", dpi=150)
    fig.savefig(FIG / "b1_match_tiers.pdf")
    plt.close(fig)

    # fig2 来源归属
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6))
    for ax, s in zip(axes, ["train", "valid"]):
        src = metrics["sources_matched"][s]
        combos = sorted(set(src["direct"]) | set(src["near"]),
                        key=lambda k: -(src["direct"].get(k, 0) + src["near"].get(k, 0)))
        d = np.array([src["direct"].get(k, 0) for k in combos])
        nr = np.array([src["near"].get(k, 0) for k in combos])
        x = np.arange(len(combos))
        ax.bar(x, d, color=BLUE, label="direct")
        ax.bar(x, nr, bottom=d, color=ORANGE, label="near")
        for xi, (dv, nv) in enumerate(zip(d, nr)):
            if dv >= 25:
                ax.text(xi, dv / 2, str(dv), ha="center", va="center", fontsize=9, color="white")
            if nv >= 25:
                ax.text(xi, dv + nv / 2, str(nv), ha="center", va="center", fontsize=9)
            ax.text(xi, dv + nv + 0.025 * d.max(), f"{dv+nv} ({dv}+{nv})", ha="center",
                    fontsize=8.5, color="0.25")
        ax.set_xticks(x, [k.replace("+", "+\n") for k in combos], fontsize=8.5)
        ax.set_ylabel(f"{s} sentences")
        ax.set_ylim(0, (d + nr).max() * 1.18)
        ax.set_title(f"({'ab'[s=='valid']}) {s}: matched by CREAK file", fontsize=10)
        ax.legend(fontsize=8)
    axes[1].annotate("valid sources: dev / contrast_set / test\n"
                     "— no overlap with CREAK train.json",
                     xy=(0.40, 0.60), xycoords="axes fraction", fontsize=9, color=RED)
    fig.suptitle("B1: source attribution (CREAK file combos; top label = total (direct+near))",
                 fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG / "b1_source_attribution.png", dpi=150)
    fig.savefig(FIG / "b1_source_attribution.pdf")
    plt.close(fig)

    # fig3 扰动分布
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8))
    ax = axes[0]
    pair_map = {"agreement": "agreement"}
    cnt = {s: Counter() for s in ["train", "valid"]}
    for s in cnt:
        for cat, v in ops_cls[s].items():
            if cat == "agreement":
                for pk, pv in agr_pairs[s].items():
                    a, b = pk.split("->")
                    cnt[s][f"agr:{'+'.join(sorted((a, b)))}"] += pv
            else:
                cnt[s][cat] += v
    keys = sorted(set(cnt["train"]) | set(cnt["valid"]),
                  key=lambda k: -(cnt["train"].get(k, 0) + cnt["valid"].get(k, 0)))
    lbl = {"adjacent_transposition": "adjacent transposition",
           "typo_other": "typo (other)",
           "case_only": "case only",
           "other": "other / structural"}
    for k in keys:
        if k.startswith("agr:"):
            a, b = k[4:].split("+")
            lbl[k] = f"agreement {a}$\\leftrightarrow${b}"
    y = np.arange(len(keys))
    tr = np.array([cnt["train"].get(k, 0) for k in keys])
    va = np.array([cnt["valid"].get(k, 0) for k in keys])
    ax.barh(y - 0.19, tr, height=0.36, color=BLUE, label="train")
    ax.barh(y + 0.19, va, height=0.36, color=ORANGE, label="valid")
    for yi, (tv, vv) in enumerate(zip(tr, va)):
        if tv:
            ax.text(tv + 2, yi - 0.19, str(tv), va="center", fontsize=8.5)
        if vv:
            ax.text(vv + 2, yi + 0.19, str(vv), va="center", fontsize=8.5)
    ax.set_yticks(y, [lbl[k] for k in keys], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("edit operations in near pairs")
    ax.set_title("(a) op-level: 2 families only", fontsize=10)
    ax.legend(fontsize=8)
    ax = axes[1]
    fams = ["typo_only", "agreement_only", "typo+agreement", "other", "structural_other"]
    ypos = np.arange(2)
    left = np.zeros(2)
    colors = {fams[0]: ORANGE, fams[1]: BLUE, fams[2]: "tab:purple", fams[3]: RED,
              fams[4]: "black"}
    for f in fams:
        v = np.array([families[s].get(f, 0) for s in ["train", "valid"]])
        if v.sum() == 0:
            continue
        ax.barh(ypos, v, left=left, color=colors[f], label=f)
        for yi in range(2):
            if v[yi]:
                ax.text(left[yi] + v[yi] / 2, yi, str(v[yi]), ha="center", va="center",
                        fontsize=8.5, color="white")
        left += v
    ax.set_yticks(ypos, ["train\nnear pairs", "valid\nnear pairs"])
    ax.set_xlabel("near-tier pairs")
    ax.set_title("(b) sentence-level families", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("B1: what the course edited (candidate $\\rightarrow$ local)", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG / "b1_perturbation_dist.png", dpi=150)
    fig.savefig(FIG / "b1_perturbation_dist.pdf")
    plt.close(fig)

    # fig4 Q8 交叉
    fig, ax = plt.subplots(figsize=(9.8, 4.6))
    cats = ["verbatim", "creak_own_other_edits", "injected_recoverable", "injected_other",
            "unmatched"]
    clbl = {"verbatim": "CREAK-own (verbatim copy)",
            "creak_own_other_edits": "CREAK-own (edited elsewhere in sentence)",
            "injected_recoverable": "injected: fixable transposition",
            "injected_other": "injected: other",
            "unmatched": "unmatched"}
    ccol = {"verbatim": GRAY, "creak_own_other_edits": "0.75",
            "injected_recoverable": ORANGE, "injected_other": RED, "unmatched": "black"}
    ypos = np.arange(2)
    left = np.zeros(2)
    for c in cats:
        v = np.array([q8_cross[s].get(c, 0) for s in ["train", "valid"]])
        if v.sum() == 0:
            continue
        ax.barh(ypos, v, left=left, color=ccol[c], label=clbl[c])
        for yi in range(2):
            if v[yi] >= 8:
                ax.text(left[yi] + v[yi] / 2, yi, str(v[yi]), ha="center", va="center",
                        fontsize=9, color="white")
        left += v
    for yi, s in enumerate(["train", "valid"]):
        qc = q8_cross[s]
        brk = " + ".join(f"{qc.get(c, 0)} {clbl[c].split(' (')[0].lower()}"
                         for c in cats if qc.get(c, 0) > 0 and qc.get(c, 0) < 8)
        cap = f"of {qc['flagged']} flagged"
        if brk:
            cap += f"\n(= {brk})"
        ax.text(left[yi] + 5, yi, cap, va="center", fontsize=8, color="0.3")
    ax.set_yticks(ypos, ["truth\ntrain", "truth\nvalid"])
    ax.set_xlim(0, left.max() * 1.34)
    ax.set_xlabel("# sentences flagged by Q8 transposition detector")
    ax.set_title("B1 $\\times$ Q8: flagged 'typos' split into CREAK-own vs course-injected",
                 fontsize=10.5)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "b1_q8_cross.png", dpi=150)
    fig.savefig(FIG / "b1_q8_cross.pdf")
    plt.close(fig)
    print("  已出 4 图 (png+pdf)")


if __name__ == "__main__":
    main()
