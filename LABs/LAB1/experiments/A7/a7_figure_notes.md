# A7 逐图解释说明（5 折 CV 协议补做）

> 实验：`experiments/A7/` —— 拓展实验 A 系列（附加探究，非课程必交内容）。问题：**我们基于单次 train/valid 划分选出的层，在规范的 K 折 CV 下还成立吗？CV 与 valid 的层排序有多一致？**（对应《可做实验清单》A7 / 指南1 §4.2；参考报告非官方）
> 协议：CV **只在 train 上做**，5 折 × 3 次重复（seed 0/1/2）；truth 按标签分层，world 无类别标签、按**纬度五分位**分层（保证每折地理覆盖）；**fold 内**用训练折统计量 z-score（无泄漏）；估计器与 A1 单层协议完全一致——world 闭式对偶 ridge 固定 $\lambda_2=1.0$，truth lbfgs 固定 $C=1/(2N_{\text{fold}}\lambda_2)$、$\lambda_2=10^{-3}$。每任务 13 层 × 15 折 = **195 次拟合**（合计 390 次）；主读数 = 3 次重复的 **OOF** 指标均值，误差棒 = 15 折 std。
> 参照列：train / valid 直接取 **A1 单层**（同协议）的 `a1_results.json`；valid **从不参与任何选择**。锚点：world L2 valid **157.6186**、truth L20 valid **0.8667**（脚本内置断言，逐位一致）；truth 195 次拟合全部收敛（`n_iter` ≤ 284）。
> 产物与复现：`python experiments/A7/a7_cv_protocol.py`（**344 s**，2026-10-17）；`a7_results.json`（逐层 + 折级 + 秩矩阵）、`a7_summary.csv`、`a7_run.log`、`a7_tails_audit.log`（尾部统计）。全部读数由脚本打印，一键复现。
> 维护约定：每完成一批绘图，在实验目录下新建 `aX_figure_notes.md`，格式同本文与 `question/figures/q8_figure_notes.md`。

## 总览

| 图 | 一句话 | 回答的内容 |
|---|---|---|
| `a7_cv_curves` | 13 层 CV OOF（±折间 std）vs train/valid 三条曲线 | "CV 选的层和 valid 一样吗；绝对水平差多少" |
| `a7_cv_vs_valid` | 逐层 CV↔valid 散点（world MSE / truth acc / truth logloss 方向反转） | "秩一致性有多高" |
| `a7_fold_stability` | 13 层 × 15 折排名热图 + 每折冠军 | "折间选层稳不稳" |
| `a7_world_error_tails` | world L2/L8 的 CV vs valid 逐样本误差 ECDF（平方误差 + 大圆距离） | "为什么 CV 的均值 MSE 远高于 valid，中位却一致" |

一句话主线：**truth 的选层在 CV 下完全复现**（13 层排序 $\rho=0.968$，CV 与 valid 都选 **L20**，选层 regret = 0；15 折中 L20 赢 11 次）——valid 上选 L20（0.8667；历史 GD 协议 0.8967）不是 300 条抽样的运气。**world 的头部是一个统计等价的平台**（CV↔valid $\rho=0.703$；CV 最优 L8、valid 最优 L2，但 CV 选择在 valid 上只差 **+1.29 MSE**；单折冠军极不稳定）。同时，A7 解释了两种"绝对水平差"：**world 的 CV 均值 MSE 远高于 valid（~350–445 vs 157–241），但中位误差几乎一致**（L2：CV 47.1°/988 km vs valid 45.1°/978 km）——差距来自地理离群点尾部（CV OOF 前 5/500 占 SSE 40.6%，最大平方误差 2.2×10⁴）；valid 是"新名字、近位置"的近邻划分（Q7：中位 2.03°）。**truth 则相反，CV 在多数层比 valid 低 0.4–6.5 pp**（每折只有 80% 训练量；Q8 已证 valid 的噪声/新主体对 L20 无显著伤害）。**train 指标不能用于选层**（world $\rho=-0.247$，ns；truth 的 train acc 自 L17 起饱和 1.0）。

---

## 1. `a7_cv_curves.*` — 13 层 CV vs train/valid 曲线

**画的是什么**：左 world（MSE，log y）、右 truth（accuracy）。x = 13 层；灰点线 = A1 train 的 **in-sample** 指标；蓝实线 ± 误差棒 = **5 折 CV OOF**（3 重复均值，误差棒 = 15 折 std）；橙虚线 = A1 **valid**（OOD/加噪口径）；浅色竖带 = CV / valid 各自最优层（蓝/橙）。

**关键读数**（2026-10-17，13 层 × 15 折）：
- **world**：CV 最优 **L8 = 351.0 ± 150.6**（valid 在同层 158.9）；valid 最优 **L2 = 157.6**（CV 在同层 378.9 ± 177.4）。CV 头部 L2/L5/L8/L11 = 350.96–378.9，**层间差（≤28）远小于折间 std（150–201）**；L14–L35 = 392.9–444.8，与 valid 的"中层差、L32 回升"同向。**valid 选 L2 在 CV 下的 regret 仅 +27.9（折噪声内）；CV 选 L8 在 valid 上的 regret 仅 +1.29 MSE**——两个冠军实际不可分辨。
- **truth**：CV 曲线与 valid 同形。CV 最优 **L20 = 0.8508 ± 0.0124**（valid 0.8667）；L23 0.8458（0.8500）、L17 0.8293（0.8333）；早层 CV 明显低于 valid（L11 0.6517 vs 0.7167，−6.5 pp），深层差 0.4–2.0 pp（L26/L29 甚至 +0.4/+0.7 pp）。**CV 与 valid 的冠军都是 L20**。
- **train 参照**：world in-sample MSE = 83.1–125.5（正则不弱），仍比 CV 低 **2.8–5.4×**；truth in-sample acc 从 L2 0.859 升到 L17+ 的 **0.998–1.000（饱和）**——train 指标在深层层完全无分辨力。

**证明了什么**：
1. **truth 的选层稳健**：CV 与 valid 同峰同形，L20 的选择不依赖某一次 300 条的 valid 划分（Q8/layer 报告中 L17–L23 为统计平台，A7 在 CV 上独立复现同一平台）。
2. **world 的"冠军"不可当真**：L2/L5/L8/L11 的 CV 差 < 折间噪声，valid 的 L2 冠军（157.62）与 L8（158.91）只差 1.3 MSE——报告应写成"早层平台"，而不是"L2 最优"。这与分层训练报告的 ±13–19 SE 结论一致。
3. **绝对水平不能跨口径直接相减**：CV 是随机折 + 80% 训练量，valid 是任务设计的 OOD 划分；truth 的 CV<valid、world 的 CV>valid 方向相反（下一节与 §4 解释）。

**报告怎么用**：方法论小节/附录的核心图——"我们的选层不是只知道 valid 结果"；同时给出"层间差异 vs 折间噪声"的量级对比，支撑"不追单个冠军、按平台+CI 报告"的写法。
图注候选（中文）：*"A7：13 层探针在 5 折 CV（train 内部，3 次重复）与单次 train/valid 划分下的对照。左 world（MSE，对数轴）：CV 最优 L8 与 valid 最优 L2 同处早层平台（层间差 ≪ 折间标准差），CV 选择的 valid regret 仅 +1.3 MSE；右 truth（准确率）：CV 与 valid 曲线同形、同峰 L20（CV 0.851 vs valid 0.867）。灰点线为 A1 train in-sample 指标。"*
图注候选（English）：*"A7: 13-layer probes under 5-fold CV (train only, 3 repeats) vs the single train/valid split. Left, world (MSE, log scale): CV-best L8 and valid-best L2 lie on the same early-layer plateau (between-layer spread ≪ fold std); the valid regret of CV selection is only +1.3 MSE. Right, truth (accuracy): CV and valid curves share the same shape and peak at L20 (CV 0.851 vs valid 0.867). Grey dotted: A1 in-sample train metrics."*

---

## 2. `a7_cv_vs_valid.*` — 逐层 CV↔valid 秩一致性

**画的是什么**：三个散点，每个点 = 一层，标注层号与 Spearman $\rho$。(a) world：CV OOF MSE vs valid MSE；(b) truth：CV OOF acc vs valid acc；(c) truth：CV OOF **logloss** vs valid acc（注意 logloss 越小越好 → 负相关是**预期方向**，不是错配）。

**关键读数**：
- **world**：$\rho = 0.703$（$p=0.007$）——中高一致；头部四层（L2/L5/L8/L11）挤在一起，具体名次由噪声决定。补充：CV 中位距离 vs valid 中位距离 $\rho=0.852$（比 MSE 更一致）；CV MSE vs CV 中位距离 $\rho=0.731$（同一协议内两种指标也不完全同序 → 均值 MSE 受尾部影响）。
- **truth**：$\rho = 0.968$（$p<10^{-6}$）——几乎完美同序；CV acc 与 valid acc 在浅层差 2–6.5 pp，深层多数 <2 pp（L35 为 2.0 pp），但**排序不变**。
- **truth 指标方向**：CV logloss vs valid acc $\rho=-0.916$；CV acc vs valid logloss $\rho=-0.896$。负号只是"lower vs higher"的方向反转；按 CV logloss 选层仍选 **L20**（`selection_logloss`：regret = 0）——本数据上不存在真正的"指标错配"。
- **train vs valid**：world $\rho=-0.247$（$p=0.415$，不显著，甚至反向）；truth $\rho=0.935$，但 train acc 自 L17 起饱和在 1.0，平台内的排序没有信息量。

**证明了什么**：
1. **truth：valid 的层排序可被 CV 复现**（0.968）——L20 的选层结论稳健；`L17–L23` 平台也在 CV 上成立。
2. **world：排序一致性中等偏高，但"头部同名次"是平坦区内的随机抖动**——不宜引用"L2 比 L8 好 1.3 MSE"这种结论；应引用"早层平台 L2/L5/L8/L11"。
3. **train 指标选层是错的**（world 反相关、truth 饱和）；选层必须用 held-out 口径。
4. **logloss 的负相关不是警告而是方向**——只要按"越小越好"解释，选层与 accuracy 一致。

**报告怎么用**：一张 ρ 图同时支撑"选层稳健（truth）"与"头部不可分辨（world）"；可放在"方法论/稳健性"小节，或作为 Q4 选层的补充证据。
图注候选（中文）：*"A7：13 层在 CV 与 valid 下的排序一致性。左 world：CV MSE vs valid MSE，Spearman ρ=0.70（头部四层挤在平坦区）；中 truth：CV acc vs valid acc，ρ=0.97（几乎同序、同峰 L20）；右 truth：CV logloss vs valid acc，ρ=−0.92——负号只是'越小越好'的方向反转，按 logloss 选层仍选 L20。"*
图注候选（English）：*"A7: layer-ranking agreement between CV and valid across 13 layers. Left, world: CV MSE vs valid MSE, Spearman ρ=0.70 (the top four layers form a flat plateau). Middle, truth: CV acc vs valid acc, ρ=0.97 (same ordering, same peak L20). Right, truth: CV logloss vs valid acc, ρ=−0.92 — the sign is just the lower-is-better direction; logloss-based selection still picks L20."*

---

## 3. `a7_fold_stability.*` — 折级排名稳定性

**画的是什么**：上下两个 13 层 × 15 折热图（列 = 3 次重复 × 5 折；白线分隔重复）；颜色 = 该折内的层排名（1 = 最好，黄）；黑圈 = 该折冠军。上 world（按折 MSE 排名）、下 truth（按折 acc 排名）。

**关键读数**：
- **truth**：**L20 在 15 折中夺冠 11 次，L23 夺冠 4 次**；L17–L23 平台在所有折都占据前 4；浅层（L2–L11）在所有折稳定垫底——热图呈连续条带，排名几乎不随折变化。折间 acc std = 0.8–1.6 pp。
- **world**：冠军 **L8 7/15、L2 3/15**，其余 L5/L11/L20/L32/L34 各 1 次；热图呈"棋盘"状，中上层名次随折大幅跳动。折间 MSE std = **147–201**，而层间均值整体跨度只有 94（351–445）——**单折 n=100 的排名基本是噪声**。
- 折冠军的离群点来源：随机折会抽到孤立地点（太平洋岛屿、南极洲），单点平方误差可达 $10^4$ 量级（§4）。

**证明了什么**：
1. **truth：选层稳定**，L20 是多数折的赢家；L20 vs L23 的差异（0.5 pp）小于折间噪声（1.2–1.5 pp），与 Q8/分层报告的"L17–L23 统计平台"一致。
2. **world：折级选层不可靠**——不是"L2 或 L8"哪个更好的问题，而是**均值 MSE 在 n=100 的折上被 1–2 个离群点主导**；任何用单次划分（包括 valid n=130）选出的"冠军"都应视为平台内的抽样结果。
3. 方法学结论：**报告层选择时应给平台 + CI，而不是单个 argmin**；这与 Q8 的 bootstrap 纪律、分层报告的 ±SE 说明是同一条方法论。

**报告怎么用**：讨论"选层不确定性"的直接证据；若报告写了"world L2 最优"，用此图说明应改为"L2/L5/L8/L11 平台"。
图注候选（中文）：*"A7：折级层排名稳定性。上 world：每折冠军在 L8（7/15）、L2（3/15）等层间跳动，折间 MSE 标准差 147–201，远大于层间差；下 truth：L20 在 15 折中赢 11 次、L23 赢 4 次，排名条带稳定。"*
图注候选（English）：*"A7: fold-level layer-ranking stability. Top, world: the per-fold winner jumps among L8 (7/15), L2 (3/15), etc.; fold-to-fold MSE std is 147–201, far larger than between-layer differences. Bottom, truth: L20 wins 11 of 15 folds and L23 the other 4; the rank bands are stable."*

---

## 4. `a7_world_error_tails.*` — 为什么 CV 的均值 MSE 远高于 valid

**画的是什么**：world 的 valid 最优层 L2 与 CV 最优层 L8 的**逐样本误差 ECDF**。左 = 平方误差（deg²，log x）；右 = 大圆距离（km，log x）；实线 = CV OOF（第 0 次重复，R0），虚线 = valid（全 train 拟合）。图例给出均值 MSE、中位平方误差、中位距离。

**关键读数**（R0 OOF；表/曲线图为 3 重复均值，故个别数字略有差异）：
- **L2（valid 最优）**：CV MSE **368.5** vs valid **157.6**；**中位平方误差 47.1 vs 45.1**、**中位距离 988 vs 978 km**——典型误差几乎相同。
- **L8（CV 最优）**：CV MSE 361.1 vs valid 158.9；中位平方误差 60.5 vs 45.6、中位距离 1074 vs 944 km（CV 差 ~14%）。
- **尾部**：CV OOF 前 5/500 条占 SSE **40.6%**（L8 42.0%）；valid 前 5/130 条占 **53.0%**（L8 47.6%）。去掉前 5 条后 valid MSE **157.6 → 77.0**（L8 158.9 → 86.6）。最大单点平方误差：CV **22,235**（Rarotonga，Cook Islands）/ **33,493**（L8）vs valid **3,930**（Deception Island）——CV 折会抽到更多孤立地点。
- **原因**：world valid 是"**新地名字符串、但地理近邻**"的划分（Q7：valid→同国最近 train 点中位 2.03°，89/130 <3°）；随机 CV 折不保证近邻，孤立地点（太平洋岛屿、南极洲）会带来 $10^4$ 量级的平方误差。

**证明了什么**：
1. **CV 与 valid 的绝对 MSE 差距不是实现/协议 bug**，而是"划分口径 + 均值指标的尾部敏感性"：中位误差一致，均值被少数点拉开。world valid 的 157.6 **低估了真正孤立地点的难度**；随机 CV（~350–445）是更严苛的尾部测试。
2. **指标必须捆中位距离/距离分布**：只看 mean MSE 会把"划分差异"误读成"模型差异"——与 C1（Tomanivi 单条占 96.5%）、Q6（南极洲 2 条）的教训完全一致。
3. 因此 world 的选层应看**秩一致性**（$\rho=0.70$）与**平台**，不能看绝对 MSE 差；truth 的 acc 有界、没有这个问题（$n=5088$ 的 OOF 也足够稳）。

**报告怎么用**：讨论"指标脆弱性"的一页图；也解释了为什么 A7 的 world 曲线（§1）三条线数量级不同却不能直接相减。若报告有"world 天花板"讨论，可与 Q7 的 oracle 基线、C1 的生成式距离并列。
图注候选（中文）：*"A7：world 的 CV vs valid 逐样本误差分布（L2 = valid 最优，L8 = CV 最优）。左：平方误差 ECDF（对数轴），典型误差接近（中位 47 vs 45 deg²），但 CV 的尾部重得多（最大单点 2.2×10⁴ vs 3.9×10³）；右：大圆距离 ECDF，L2 的中位距离几乎相同（988 vs 978 km）。均值 MSE 的差距由少数地理离群点主导。"*
图注候选（English）：*"A7: per-sample error distributions for world (L2 = valid best, L8 = CV best). Left: squared-error ECDF (log scale); typical errors agree (median 47 vs 45 deg²) but the CV tail is much heavier (max 2.2×10⁴ vs 3.9×10³). Right: great-circle distance ECDF; L2 medians are nearly identical (988 vs 978 km). The mean-MSE gap is driven by a few geographic outliers."*

---

## 附 A：逐层数字速查

**world（valid MSE；CV = 3 重复 OOF 均值 ± 15 折 std；中位距离单位 km）**

| 层 | CV MSE | valid MSE | CV 中位 km | valid 中位 km |
|---|---|---|---|---|
| L2 | 378.9 ± 177.4 | **157.6** | 943.5 | 977.5 |
| L5 | 377.1 ± 175.6 | 162.8 | 1042.2 | 942.0 |
| L8 | **351.0 ± 150.6** | 158.9 | 1064.0 | 943.7 |
| L11 | 377.7 ± 156.4 | 167.2 | 1057.1 | 923.1 |
| L14 | 404.3 ± 147.6 | 213.7 | 1278.8 | 1077.4 |
| L17 | 444.8 ± 157.0 | 227.0 | 1364.0 | 1208.4 |
| L20 | 419.8 ± 155.2 | 241.6 | 1391.6 | 1188.2 |
| L23 | 392.9 ± 164.5 | 241.0 | 1338.9 | 1416.5 |
| L26 | 423.2 ± 173.8 | 226.4 | 1337.8 | 1322.6 |
| L29 | 426.1 ± 194.9 | 210.3 | 1235.4 | 1188.0 |
| L32 | 413.3 ± 201.5 | 180.9 | 1097.8 | 1120.8 |
| L34 | 401.3 ± 194.8 | 196.9 | 1118.0 | 1081.0 |
| L35 | 404.1 ± 189.6 | 205.3 | 1180.6 | 1171.8 |

**truth（accuracy；CV = 3 重复 OOF 均值 ± 15 折 std）**

| 层 | CV acc | valid acc | CV logloss |
|---|---|---|---|
| L2 | 0.5666 ± 0.0142 | 0.5900 | 1.1546 |
| L5 | 0.5844 ± 0.0151 | 0.6233 | 1.2151 |
| L8 | 0.6216 ± 0.0083 | 0.6800 | 1.3627 |
| L11 | 0.6517 ± 0.0132 | 0.7167 | 1.4175 |
| L14 | 0.7666 ± 0.0143 | 0.7800 | 0.9171 |
| L17 | 0.8293 ± 0.0111 | 0.8333 | 0.6555 |
| L20 | **0.8508 ± 0.0124** | **0.8667** | **0.5381** |
| L23 | 0.8458 ± 0.0145 | 0.8500 | 0.5449 |
| L26 | 0.8210 ± 0.0145 | 0.8167 | 0.6401 |
| L29 | 0.8067 ± 0.0129 | 0.8000 | 0.6780 |
| L32 | 0.7984 ± 0.0129 | 0.8167 | 0.7245 |
| L34 | 0.7871 ± 0.0159 | 0.7867 | 0.7453 |
| L35 | 0.7864 ± 0.0134 | 0.8067 | 0.7418 |

**选层与一致性**：world CV best **L8**（valid 158.9）/ valid best **L2**（157.6）；valid regret of CV selection **+1.29 MSE**；CV regret of valid selection +27.9。truth CV best = valid best = **L20**，双向 regret = 0。秩相关：world CV↔valid **0.703**（$p=0.007$）、中位距离 **0.852**、train↔valid **−0.247**（ns）；truth CV↔valid **0.968**、train↔valid 0.935（train 饱和）、CV logloss↔valid acc **−0.916**。

## 附 B：口径与注意事项

1. **CV 每折只用 80% 训练数据**（truth 4070/5088、world 400/500），与 valid 的绝对差混合了"训练量"和"分布"两种因素，不能把差值解释为 OOD 代价。truth 的 CV 比 valid 低 0.4–6.5 pp（浅中层最大），部分正是训练量效应；world 的 CV 均值 MSE 高则是尾部效应（§4）。
2. **world 的"分层"是纬度五分位**（无类别标签可用；158 个国家中最小样本数 1，无法按国家分层）。若需"未见国家"口径，应另做 GroupKFold（Q8 已对 truth 做过主体分组 CV：0.869 vs 0.877，差 −0.8 pp）。
3. **3 次重复的 OOF 均值是主读数**；误差棒用 15 折 std（比 3 个 OOF 值的 std 更保守）。逐样本尾部图用 R0，故其 MSE 与表略异（L2 368.5 vs 378.9）。
4. **truth 的 195 次拟合全部收敛**（lbfgs `n_iter` ≤ 284，远低于 2000 上限），无发散/无警告。
5. **logloss 负相关是方向反转**，不是"指标错配"：按"越低越好"解释后，logloss 与 accuracy 的选层结果一致（均为 L20）。
6. **world 的 valid 是近邻划分**（Q7：中位 2.03°），不是真正的地理外推；它的 157.6 不能当作"未见地点"的泛化误差。真正的地理 OOD 评估应看 Q7 的 unseen 20 条或另建分组 CV。

## 附 C：相关数字的出处

- 生成脚本：`experiments/A7/a7_cv_protocol.py`（一条命令复现全量 CV + 4 图；`--tasks world` 可只跑 world 冒烟，但会覆盖 JSON，正式重跑请不带参数）。
- 结果：`experiments/A7/a7_results.json`（逐层记录、折级指标、13×15 秩矩阵、折冠军、Spearman、选层 regret）、`a7_summary.csv`（逐层 train/CV/valid 与排名）。
- 日志：`a7_run.log`（全量 344 s）、`a7_tails_audit.log`（§4 的尾部统计：top-5 SSE 占比、去尾 MSE、最大单点误差）。
- 参照：`experiments/A1/a1_results.json`（同协议 train/valid 单层指标；world L2 157.6186、truth L20 0.8667 锚点）。
- 数据：`datasets/{world,truth}/{train,valid}.layer*.npy`、`datasets/{world,truth}/{train,valid}.csv`。
- 图：`experiments/A7/figures/a7_{cv_curves,cv_vs_valid,fold_stability,world_error_tails}.{png,pdf}`（dpi=150 + PDF，2026-10-17 生成）。
