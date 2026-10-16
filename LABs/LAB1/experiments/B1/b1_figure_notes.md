# B1 逐图解释说明

> 来源：《可做实验清单》B1（指南3 §3.1，**同学参考报告，非官方**）；2026-10-16 用户指派执行。
> 对象：本地 truth train 5088 / valid 300 句（真假分类）与 **CREAK** 公开数据（Onoe et al. 2021, NeurIPS D&B，`github.com/yasumasaonoe/creak`，本地 clone rev `5e892b28`；4 个 JSONL 文件 train 10176 / dev 1371 / test_without_labels 1371 / contrast_set 500）。
> 匹配协议（**不使用标签检索**）：direct = whitespace 归一化（`" ".join(s.split())`）后逐字相等；near = rapidfuzz `fuzz.ratio >= 90` 的最优候选（候选按归一化去重，margin = top1−top2）；none = 剩余（实测为 0）。标签一致性在匹配**之后**单独核对。编辑分类以 candidate→local 方向单 token 替换为单位（比较时剥掉 token 首尾标点）。
> 全部读数由 `experiments/B1/b1_creak_match.py` 打印、一键复现（22 项内置锚点全 PASS，~12–30 s）；原始数字在 `b1_metrics.json`（含 818 个 near 对逐条明细）。
> 维护约定：每完成一批绘图，在新目录下建 `<题号>_figure_notes.md`，格式同本文与 q2/q3/q6/q7/q8 各 notes。

## 总览

| 图 | 一句话 | 回答的内容 |
|---|---|---|
| `b1_match_tiers` | 层级金字塔：direct 86.1%/63.3% + near（≥90）13.9%/36.7% = **100% 可追溯**，none=0 | "多少句子能对上 CREAK 原文" |
| `b1_source_attribution` | train 全部来自 CREAK train.json；valid 全部来自 dev/contrast/test（**与 train.json 零重叠**） | "来自哪个文件" |
| `b1_perturbation_dist` | 846 处编辑只有两族：相邻字母交换错拼 415 + 主谓一致翻转 431，零例外 | "对不上的差在哪" |
| `b1_q8_cross` | Q8 检出的"错拼"拆解：train 87%/valid 88% 为课程注入（可修复），其余为 CREAK 自带或修复器误伤 | "错拼噪声的来源拆分" |

一句话主线：**truth 语料 100% 由 CREAK 派生**——train 逐句来自 CREAK train.json（4380 条逐字 + 708 条单处编辑），valid 逐句来自 CREAK dev/contrast/test（190 + 110）；5388 个标签与 CREAK **零冲突**（继承无损）。课程对语料做的"扰动工程"只有两族、共 846 处：相邻字母交换型错拼（415）与主谓一致翻转（431，is↔are / has↔have / was↔were / do↔does）——**没有**实体替换、内容词改写或标点/大小写编辑。这把 Q8 的"valid 是设计过的轻量 OOD 挑战集"从统计推断升级为**来源级证据**：valid 连"新主体"都多半是 CREAK dev 自带（拼写扰动把 `Dwyane Wade`、`Paula Deen` 等名字误标为 typo 的 40+6 条，实为 CREAK 原文照抄）。

---

## 1. `b1_match_tiers.*` — 匹配层级与近邻分数

**画的是什么**：(a) train/valid 的层级堆叠条（按 split 归一化百分比）：direct（归一化逐字相等）vs near（fuzz.ratio ≥ 90 的最优候选）；(b) near 层 708+110 条的 best-candidate 分数 ECDF（点线 = 中位）。

**关键读数**：
- **层级**：train 4380 direct（86.1%）+ 708 near（13.9%）；valid 190 direct（63.3%）+ 110 near（36.7%）；**none 两侧均为 0**。
- **guide3 对拍**：direct 率 train 4380/5088 = **86.08%**、valid 190/300 = **63.33%**，与同学参考报告 §3.1 逐位一致（其两轮匹配口径下的同款数字）。
- **近邻分数**（708/110 条）：train min 90.20 / med 97.47 / max 99.35；valid min 92.96 / med 97.01 / max 98.75。
- **余量**（top1−top2，去重候选）：train med **38.28** / min 1.39；valid med **35.21** / min 0.84；**margin=0 的并列 0 条**——near 对几乎无歧义，不会被"多个 CREAK 近邻"污染。

**证明了什么**：
1. 追源是**全覆盖**的：5388/5388 = 100%，没有"既不在 CREAK 又说不清来历"的句子。direct+near 的上限意味着语料**整体**（而非抽样）由 CREAK 构造而来——这是后续一切结论的地基。
2. near 层余量中位 35+ 表示这些"对不上"全部是**单点小编辑**，不存在"同主题但不同句子"的伪匹配风险（min 0.84 的两条可人工复核，仍在 90 阈值之上、与候选仅差一个 token）。

**报告怎么用**：若在报告中引用 B1（可选，如 Q8 的补充证据）：这是"语料来源"主图。
图注候选（中文）：*"本地 truth 语料与 CREAK 的逐句对拍。(a) 匹配层级（按 split 归一化）：direct = 归一化逐字相等，near = rapidfuzz 最优候选 ratio ≥ 90；train 86.1%+13.9%、valid 63.3%+36.7%，两侧均无未匹配。(b) near 层的候选分数 ECDF（train n=708 / valid n=110，全部 ≥ 90，中位 ≈ 97）。"*
图注候选（English）：*"Sentence-level provenance matching of our truth corpus against CREAK. (a) Tiers per split (normalized): direct (exact after whitespace normalization) and near (best candidate fuzz.ratio ≥ 90); 86.1%+13.9% (train) and 63.3%+36.7% (valid), with zero unmatched. (b) Score ECDF of near-tier pairs (all ≥ 90, median ≈ 97)."*

---

## 2. `b1_source_attribution.*` — 来源归属

**画的是什么**：(a) train、(b) valid 的匹配句按 CREAK 文件组合分解（direct/near 堆叠；顶标 = 总数（direct+near））。

**关键读数**：
- **train**：全部 5088 条来自**同一个**文件 `train.json`（4380+708）——100% 单一来源。
- **valid**：dev 269（166+103）、contrast_set 27（21+6）、contrast_set+test_without_labels 2（1+1）、contrast_set+dev 2（2+0）——**与 CREAK 的 train.json 零重叠**（跨四个文件组合计数里都不含 train）。
- 即：课程把 CREAK 的 **train 文件**当我们的训练集来源，把 **dev（+少量 contrast/test）**当 valid 来源——复刻了 CREAK 自己的"训练/留存"划分，同时再叠加两类扰动（§3）。

**证明了什么**：
1. train/valid 的**文件级**来源干净可分——valid 的"新主体多"（Q8 §1b：77.4%）直接继承自 CREAK dev 与 train 的实体不同，不是课程额外造的新实体句。
2. "valid 全部来自 dev/contrast/test" 是 Q8"valid 是设计过的挑战集"的最直接证据之一：**来源级**（文件不同）+ **编辑级**（§3 扰动）+ 统计级（Q8 分布差异）三层证据闭合。
3. 也解释了 valid 侧 contrast_set 的存在（contrast 句是 CREAK 的"对照句"——多为对同实体错误常识的否定版本）——它只占 valid 的 ~9%。

**报告怎么用**：Q8 补充证据（可选）；也可用于说明"我们的 valid 不是随机留出，而是同源异构"。
图注候选（中文）：*"匹配句的 CREAK 文件来源。train 全部来自 CREAK train.json；valid 全部来自 dev/contrast_set/test（与 train.json 零重叠），体现课程对 CREAK 原始划分的复刻。"*
图注候选（English）：*"CREAK-file attribution of matched sentences. All train sentences trace to CREAK train.json; all valid sentences trace to dev/contrast_set/test — zero overlap with train.json, mirroring CREAK's own split."*

---

## 3. `b1_perturbation_dist.*` — 课程扰动类型分布

**画的是什么**：(a) near 对中编辑操作的类型计数（train/valid 分组条）；(b) 句子级家族堆叠条（typo_only / agreement_only）。

**关键读数**：
- **总量 846 处编辑、818 个 near 对**，无一处落在两族之外（初始分类唯一一条 `is.`→`are.` 是句末标点粘连，剥标点后归入 agreement；**零 case-only、零 other**）。
- **族 1——相邻字母交换错拼（415 处）**：train 318 + valid 97。典型：`mkaes→makes`、`tehir→their`、`conutry→country`、`Fihgters→Fighters`、`Nrivana→Nirvana`、`Connectiuct→Connecticut`、`baskebtall`、`craeer`（截取见 b1_metrics `perturbation.examples`）。
- **族 2——主谓一致翻转（431 处）**：train 390 + valid 41。方向双向：`is→are`（train 146，反方向 86）、`has→have`（56+12 vs 29）、`was→were`（51+8 vs 8+2）、`do→does`（11 vs 4）。
- **句子级**：train 318 typo_only + 390 agreement_only；valid 69 + 41。**28 个 valid 对一次带两处错拼**（如 row11 `Fihgters`+`Nrivana`、row17 `baskebtall`+`craeer`），对应 valid 97 处错拼承载在 69 句上；**没有** typo+agreement 混合句。

**证明了什么**：
1. 课程的"加噪"是**两种精心选择的语义保值扰动**：字符级乱序（错拼）考验 token 级鲁棒性；主谓一致翻转造出**语法错误但语义不变**的句子（"E multiplied by 2 **are** equal …"），考验句法/语义层对表层形态的归一化——两类都**不改标签**（§4 零标签冲突佐证）。这解释了 Q8 的模型行为：深层（≥L17）探针对 typo 不敏感（语法/语义特征稳定）、浅层（L5–L11）敏感（词面特征被改）。
2. 没有实体替换/内容词改写 ⇒ valid 的"难"不来自知识面变化，而来自**表层扰动 + 主体新颖**（后者的来源在 §2 已定：文件级换源）。
3. 分布上 valid 比 train 扰动更密（错拼句率 3.5×、若干句双错拼），与 Q8 §1 的量化互证。

**报告怎么用**：回答"valid 到底怎么造的"（Q8 用）；也支撑选层结论（深层抗扰动的又一条理由）。
图注候选（中文）：*"课程对 CREAK 原文的两类扰动（candidate→local，共 846 处，无第三类）。(a) 操作级计数：相邻字母交换错拼 vs 主谓一致翻转（is↔are / has↔have / was↔were / do↔does）；(b) 句子级家族（valid 有 28 句同带两处错拼）。"*
图注候选（English）：*"The two — and only two — perturbation families applied to CREAK originals (candidate→local; 846 edits total). (a) Op-level counts: adjacent-transposition typos vs subject–verb agreement flips; (b) sentence-level families (28 valid pairs carry two typos)."*

---

## 4. `b1_q8_cross.*` — Q8 错拼检测 × CREAK 来源拆解

**画的是什么**：Q8 的相邻交换错拼检测器（pyspellchecker 字典）在本地语料上检出的 354（train）/ 74（valid）条"带 typo 句"，按与 CREAK 的关系拆解：CREAK 原文照抄（verbatim）/ CREAK 自带但句内另有编辑（creak_own_edited）/ 课程注入且字典修复可还原（injected_recoverable）/ 课程注入但修复不还原（injected_other）。

**关键读数**：
- **train 354**：injected_recoverable **307（86.7%）** + verbatim **40（11.3%）** + creak_own_edited 3 + injected_other 4。
- **valid 74**：injected_recoverable **65（87.8%）** + verbatim **6（8.1%）** + injected_other 3 + creak_own_edited 0。
- **verbatim 例**：`Pharoah of Egypt…`（`pharoah`）、`Niels Bohr…`（`niels`）、`Ewan McGregor…`（`ewan`）、`Paula Deen…`（`deen→eden`）、`Dwyane Wade…`（`dwyane→dwayne` ×4）——**检测器的假阳性全部是 CREAK 自带的名词**（真实人名/专名被字典误判为交换错拼），与 Q8 notes 记录一致（40+6 条逐条对上）。
- **修复核验**：`fix_sentence`（对全语料建错的 w→swap 同表）在 typo_only 家族上修复成功率 **train 307/318 = 96.5%、valid 65/69 = 94.2%**（修复后与 CREAK 候选**逐字相等**——强证据：这些句子=CREAK 原文+单一字符交换）。
- **injected_other 的 7 条**（train 4 + valid 3）不是"没注入"，而是**字典修复器把它们改坏了**：`MOBA→MOAB`（row 914）、`Loni→Loin`（row 1153）、`taeks→teaks`（row 3813，正确的 `takes` 没被选中）、`Elon→Leon`（row 4020）——修复器对**实体名过度纠正**，一个句子里多处 flag 时"修好一个、改坏一个"。
- **creak_own_edited 例**：row 3202 `…restuarant are called…`（`restuarant` 是 CREAK 自带错拼，课程只翻了 `is→are`）；row 3485 `Akon have…`（`Akon` 被误判，课程只翻 `has→have`）；row 4115 `Sangre de Cristo…prats…`（`Sangre` 误判；课程注入的 `parts→prats` 反而**没被检出**——`prats` 本身是字典词）。

**证明了什么**：
1. **Q8 的"拼写噪声"现在可完全归因**：valid 24.7% 的检出率中 ~88% 是课程注入（且基本都能用"还原交换"精确修回 CREAK 原文）、~8% 是 CREAK 自带名词假阳性——Q8 notes 里"检测器有句子级假阳性"的判断由 B1 定量坐实（train 40/354 = 11.3%）。
2. 修复链 `local → (还原1处交换) → CREAK 原文` 的高成功率（94–97%）说明注入方式**单一确定**——课程实现就是"随机选一个词做一次相邻交换"。
3. **反面警示**（写报告时避免踩坑）：任何"字典纠错预处理"都会误伤专名（MOBA/Elon/Loni/Deen/Dwyane 全家）——我们的特征提取与训练**未做任何文本清洗**是对的（Q8 §4 同款结论）。

**报告怎么用**：Q8 的增强证据（可选）；回答"验证集错拼多，你怎么办"时可引用"扰动=单次交换、语义保值、标签不发散"。
图注候选（中文）：*"Q8 错拼检测结果与 CREAK 来源的交叉拆解。训练集 354 条检出中 307 条为课程注入（字典还原后与 CREAK 原文逐字一致）、40 条为 CREAK 原文自带名词假阳性；valid 74 条中 65 条为注入、6 条为自带。"*
图注候选（English）：*"Q8 typo detections cross-classified against CREAK provenance. Of 354 flagged train sentences, 307 are course-injected (dictionary de-transposition exactly recovers the CREAK original) and 40 are CREAK-own proper-noun false positives; for valid, 65/74 injected and 6/74 CREAK-own."*

---

## 5. 交叉核对、复现与注意事项

- **锚点自检**（脚本内置 22 项全 PASS）：CREAK 4 文件行数；truth 行数/无重复；direct 4380/190；near 708/110；none 0；valid direct 来源组合 {dev 166, contrast_set 21, contrast_set+dev 2, contrast_set+test 1}；标签冲突 0/0；总编辑数 846；Q8 flags 354/74；verbatim 40/6。
- **与 Q8 文档互证**：typo 句率 7.0%/24.7%、40+6 条 verbatim、"detector 假阳性"三处数字与 `q8_figure_notes.md` 完全一致（同一检测器实现，本脚本在 `detect_typos/fix_sentence` 处注明与 `question/code/q8_split_analysis.py` 保持一致）。
- **与 guide3（同学报告）互证**：direct 率 86.08%/63.33% 逐位一致；报告未做 near 层，B1 在此之上把覆盖率推到 100%。
- **注意事项**：
  1. near 阈值 90 是约定（min 分别为 90.2/93.0，margin>0.8，无并列）；报告引用时 near 应写"单点编辑（ratio≥90）"，不等同 direct。
  2. 错拼检测是**下界**（只认"交换后成词"）；如 `prats`（本身是字典词）检不出。
  3. `fix_sentence` 是诊断工具（还原交换以对拍），**不是**建议的清洗/预处理手段（§4 反面警示）。
  4. 引用 CREAK 请标注原论文：Onoe, Zhang, Durrett — *CREAK: A Dataset for Commonsense Reasoning over Entity Knowledge*, NeurIPS 2021 Datasets & Benchmarks；数据不入 git（`experiments/B1/data/` 已 ignore），复现锚在 `b1_metrics.json` 的 revision + 4 文件 md5。
- **复现**：`cd LABs/LAB1 && python -X utf8 experiments/B1/b1_creak_match.py`（依赖 rapidfuzz ≥3.14、pyspellchecker；~12–30 s；重跑幂等）。数据获取：`git clone https://github.com/yasumasaonoe/creak experiments/B1/data/creak && git -C experiments/B1/data/creak checkout 5e892b28`（md5 见 metrics，必须一致）。
