# C3 逐图解释说明

> 实验：GeoNames 数据扩增（《可做实验清单》C3 / 指南2 §1）。**附加实验（非课程提交内容）**。问题：在 world 500 条之外重采样上万 GeoNames 地点、重新提 Qwen 特征并训练探针，能否提升原 world 任务与"未见国家"泛化？
> 资产：代码 `experiments/C3/`；原始 GeoNames dump `experiments/C3/data/raw/`（gitignored）；划分 `experiments/C3/data/splits/`；特征（集群生成，gitignored）`experiments/C3/features/`；训练产物 `results/slurm_c3/c3_v1/`；评估脚本 `experiments/C3/c3_eval.py`（`C:\Users\35150\miniconda3\envs\ai26\python.exe -X utf8 experiments/C3/c3_eval.py` 一键复现）；图 `experiments/C3/figures/`；指标 `experiments/C3/c3_analysis.json`。
> 模型/提取：Qwen3-4B-Base rev `906bfd4b`，36 blocks、hidden 2560、bf16；13 层 `[2,5,8,11,14,17,20,23,26,29,32,34,35]`，block 输出（final RMSNorm 之前），国家名最后一个 subtoken 读点，float16 存储；prompt 与 world 完全一致（城邦省略国家后缀）。
> 探针口径：**std13** = train 统计量 z-score + lr 0.002 + 2000 epochs + λ1=0.001 + λ2=1.0；与 `model.py`/`run_grid.evaluate` 及 C2 主网格一致。距离 = haversine（R=6371）；MSE=(Δlat²+Δlon²)/2（课程原始尺度；周期经度口径另存 JSON）。
> 主口径：**L11** 为课程预注册层；L2 是 orig_valid 上事后最优、L32 是 full `test_unseen` 上事后最优，二者只作描述性（见 §6）。配对 Δ = 训练集 A − 训练集 B（**负 = A 更好**）；item bootstrap 10k 次、seed `20261017`；另给 country-cluster bootstrap（按国家整簇重采样）作为更保守的敏感性区间。

## 总览

| 图 | 一句话 | 回答的内容 |
|---|---|---|
| `c3_layer_curves` | 5 个训练集 × 13 层 × 4 个评估集的 MSE 曲线 | 增广在哪些层/哪些评估集上有用；noleak 曲线是否重合 |
| `c3_transfer_bars` | 固定 L2 / L11 / L32 的分组柱状图 | 三个训练集在 world、GeoNames test、strict unseen 上的总账 |
| `c3_paired_effects` | 配对 ΔMSE 森林图（item + country-cluster CI） | 增广效应是否跨 0；在 seen/unseen 分层与 strict unseen 上各是多少 |
| `c3_strata_unseen` | world valid 分层 + strict unseen 逐国误差 | 增益来自哪些国家；是否每个国家都变好 |

一句话主线：**GeoNames 增广不是"免费午餐"**——它把 world valid 的 MSE 从 157.6 降到 139.2（L2，−11.7%）但在课程 L11 上反而从 167.0 升到 172.9（+3.5%），总体配对 CI 跨 0；增益几乎全部来自 world train 未覆盖的国家（L11 的 12 条 unseen→C3 子集 −66.5%），seen 国家则变差（+16.7%，CI 跨 0）。在严格剔除"world train 已出现国家"后的 8 国 260 条上，union 在 L11 比 GeoNames-only **低 21.7 MSE**（country-cluster CI [−41.9,−7.8]），是唯一在国家级 bootstrap 下也稳健的增广收益；但逐国方向高度异质（哥斯达黎加/巴勒斯坦/台湾变好，丹麦/黎巴嫩/也门反而 orig/new 更好）。

---

## 0. 设计与验证（先看这节再读图）

**数据构建**（`c3_build_dataset.py`；详见 `data/meta.json`、`data/leak_audit.json`）：

| 项 | 值 |
|---|---|
| 来源 | GeoNames `cities15000.txt`（34,166 populated places / 244 country codes，md5 `93552399…`）+ `countryInfo.txt`（252 国，md5 `8b8c1506…`） |
| 采样 | 12,000 条；235 个国家；30 个国家（每国 8–500 城）**整国留出**；每个入样国家至少 1 城 |
| 划分 | train 8,382（193 国）/ valid 1,050（117 国）/ test 1,050（110 国）/ test_unseen 1,518（30 国）；seed 20261004 |
| prompt | `What's the longitude and latitude of {place}, {country}?`；城邦（place==country）省略 ", {country}"；最长 98 token |
| 标签 | GeoNames 经纬度；不做标签清洗/回绕 |
| 泄漏审计 | train 中与原始 world valid **同名同国** 14 条（另与 world train 68 条）；noleak 训练集剔除这 14 条；valid/test/unseen 分别有 2/1/8 条与 world valid 同名同国 |

**五个训练集**：`orig`（world 500）、`new`（GeoNames 8,382）、`new_noleak`（8,368）、`union`（8,882）、`union_noleak`（8,868）。**四个评估集**：`orig_valid`（130）、`new_valid`（1,050）、`new_test`（1,050）、`new_unseen`（1,518）。

**严格未见子集（本实验新增，关键）**：
- `test_unseen` 的 30 国中，**22 国其实出现在 world train**（world train 158 国）；因此对 `orig` 来说 full unseen **不是**未见国家，`orig` 在该集上的好成绩（L11 56.5）主要来自这 22 国的训练样本。
- `strict8` = 8 个在 world train 完全没有国家字符串的国家：Costa Rica、Denmark、Hungary、Lebanon、Niger、Palestine、Taiwan、Yemen（260 条）。
- `hard6` = 再去掉 Niger/Palestine（212 条，6 国）：world train 有 `Chad / Nigeria / Niger / Cameroon`、`Israel/Palestine` 这类复合字符串，可能包含这两国的城市；hard6 是对"真正没有国家级先验"的敏感性口径。
- 8 国在 **C3 train 中也是 0 条**（整国留出），所以 strict8/hard6 对 `new`/`union` 同样干净。

**提取自检**：重提 `datasets/world` 的 train/valid 26 个 (split, layer) 组合，与历史特征逐层比较：mean cosine ≥ **0.99993**；L11/L23/L35 有 float16 级差异（min cos 0.999897/0.999811/0.999684，max_abs 0.5/1/10，来自 bf16 归约顺序），阈值 0.999 通过。共 52 个特征文件。

**训练自检**：65 个 run 全部有限（无发散、无 lr 降半）；`orig` 的 13 层 MSE 与 `results/slurm_layers/world/summary.csv` 的 std13 历史值 **13/13 逐位一致**（差 <1e-4；L2 157.5709、L11 167.0413、L35 204.9075），说明提取与训练管线无错配。65 个权重文件、65 条记录通过 sbatch 收尾校验。

**坐标数学**：MSE 用 `coord_sq_error`（原始经度差，不先回绕）；周期 MSE 在 JSON 中另存。本批预测没有大面积跨 ±180°，除 `orig` 在 new_valid/new_test 的少数条目（周期 78.98/142.86 vs 原始 87.80/147.01）外，原始与周期值接近。

---

## 1. `c3_layer_curves.*` — 训练集 × 层 × 评估集

**画的是什么**：2×2 四个评估集；每个面板 5 条线（orig / new / union 实线，new_noleak / union_noleak 同色虚线）；横轴 13 个 decoder block，纵轴原始 MSE（对数），点线 = L11，浅点线 = L2；world valid 面板另有国家均值 oracle 113.8 的虚线。

**关键读数**（MSE；完整曲线见 `c3_analysis.json` 的 `layer_curves`）：

| 评估集 | 训练集 | L2 | L11 | L32 | 该评估集最优层 |
|---|---|---|---|---|---|
| world valid | orig | **157.57** | 167.04 | 180.31 | L2 |
| world valid | new | 186.65 | 225.29 | 268.03 | L2 |
| world valid | union | **139.21** | 172.92 | 212.04 | L2 |
| Geo valid | orig | 121.20 | 143.32 | 160.57 | L2（但 L11 是 new/union 的最优） |
| Geo valid | new | 105.84 | **87.80** | 94.89 | L11 |
| Geo valid | union | 97.04 | **81.42** | 90.48 | L11 |
| Geo test | orig | 163.25 | **147.01** | 190.21 | L11 |
| Geo test | new | 84.34 | **64.02** | 73.96 | L11 |
| Geo test | union | 84.49 | **62.34** | 71.55 | L11 |
| Geo unseen（full） | orig | 61.21 | 56.46 | **53.89** | L32 |
| Geo unseen（full） | new | 263.29 | 148.49 | **86.23** | L32 |
| Geo unseen（full） | union | 138.47 | 81.90 | **65.60** | L32 |

- **world valid 上三条线都是"浅层最好、L20 附近最差"的 U 形**；union 在 L2 比 orig 低 18.4 MSE，但在 L11 高 5.9、L32 高 31.7。`new` 单独训练在 world valid 上全程最差（L11 225.3），说明 8,382 条 GeoNames 不能替代 500 条原分布样本。
- **Geo valid/test 上 new 与 union 几乎重合**，L11 最优（union 81.4/62.3）；orig 在 Geo test 上 L11 147.0，明显更差。增广让"同一个 probe"在新分布上可用，但 `new` 单独已经足够；加 orig 的收益只有 −1.7 MSE（见 §3）。
- **full unseen 面板最容易误读**：orig 在 L11 只有 56.5，看似"小数据泛化最好"；实际 30 国中 22 国在 world train 出现过，orig 有直接国家样本。该面板不能回答"未见国家"问题，必须看 §4 的 strict8/hard6。
- **noleak 曲线与完整曲线几乎重合**（new_noleak/new、union_noleak/union），说明 14 条同名同国泄漏样本对结论没有影响；差异见 §3 的 leak 行（均 <1 MSE）。

图注候选（中文）：*"C3 五个训练集在四个评估集上的坐标 MSE 随解码层变化（Qwen3-4B-Base，std13 探针，对数纵轴）。world valid 上 union 仅在 L2 低于 orig；GeoNames test 上 new/union 在 L11 达到 62–64；full unseen 上 orig 的优势主要来自 world train 已覆盖的 22/30 国家，不能当作未见国家泛化。"*
图注候选（English）：*"C3 coordinate MSE vs decoder layer for five training sets on four evaluation sets (Qwen3-4B-Base, std13 probe, log scale). On world valid, union beats orig only at L2; on GeoNames test, new/union reach 62–64 at L11; the full unseen panel favors orig because 22 of its 30 countries occur in the original world training set, so it is not a clean unseen-country benchmark."*

---

## 2. `c3_transfer_bars.*` — 固定层的总账

**画的是什么**：三格分别固定 L2（orig_valid 事后最优）、L11（课程层）、L32（full unseen 事后最优）；每格 5 个评估集 × 3 个训练集（orig/new/union），纵轴 MSE（对数），柱顶标数值。

**关键读数**（MSE；L2 / L11 / L32）：

| 评估集 | orig | new | union |
|---|---|---|---|
| world valid 130 | 157.6 / 167.0 / 180.3 | 186.6 / 225.3 / 268.0 | **139.2** / 172.9 / 212.0 |
| Geo valid 1,050 | 121.2 / 143.3 / 160.6 | 105.8 / 87.8 / 94.9 | 97.0 / **81.4** / 90.5 |
| Geo test 1,050 | 163.3 / 147.0 / 190.2 | 84.3 / 64.0 / 74.0 | 84.5 / **62.3** / 71.6 |
| Geo unseen 1,518 | 61.2 / 56.5 / **53.9** | 263.3 / 148.5 / 86.2 | 138.5 / 81.9 / **65.6** |
| strict unseen 260 | 224.9 / 93.8 / 80.3 | 88.5 / 97.4 / 64.0 | 91.4 / **75.7** / **57.4** |

- **L11（课程主口径）**：union vs orig 在 world valid 上 **+5.9（更差）**；在 Geo valid/test 上 **−61.9/−84.7**（相对 orig）；在 strict unseen 上 **−18.1**。new vs orig 在 Geo test 上 **−83.0**，但在 world valid 上 **+58.2**。三条线的"谁更好"完全取决于评估集。
- **L2（事后最优）**：union 的 world valid 139.2 是三个训练集里最好的（orig 157.6），但仍高于国家均值 oracle **113.8**；strict unseen 上 union 91.4 ≈ new 88.5，优势消失。
- **L32（事后最优）**：union 在 full unseen 65.6、strict unseen 57.4 都最低；但 L32 在 Geo test 上已明显退化（union 71.6 vs L11 62.3），不是通用层。
- **L2 是非预注册层**：它是在同一 130 条 orig_valid 上挑出来的，因此 "union 139.2 vs orig 157.6" 是乐观比较；L11 的 +5.9 才是课程口径的无偏对照（但仍无 world test 标签）。报告必须并列两个层，不能只报 L2。

图注候选（中文）：*"固定层的训练集对比（MSE，对数纵轴）。L11：union 在 GeoNames valid/test 上比 orig 低 62/85 MSE，但在 world valid 上高 5.9；L2：union 把 world valid 从 157.6 降到 139.2（事后选层，仍高于国家均值 113.8）；L32：union 在 unseen 上最低，但 Geo test 已退化。"*
图注候选（English）：*"Fixed-layer comparison (MSE, log scale). At L11, union is 62/85 MSE below orig on GeoNames valid/test but 5.9 above orig on world valid. At L2 (post-hoc), union lowers world valid from 157.6 to 139.2, still above the 113.8 country-mean oracle. At L32, union is lowest on unseen but has degraded on Geo test."*

---

## 3. `c3_paired_effects.*` — 配对效应与置信区间

**画的是什么**：左面板 = 原任务（world valid n=130）；右面板 = GeoNames 任务与 strict unseen。横轴 = 配对 ΔMSE（A−B；**负 = A 更好**），红点 = 均值，细线 = item bootstrap 95% CI，灰色粗线 = country-cluster bootstrap 95% CI（按国家整簇重采样；国家数少时更保守）。纵轴标签给出 A−B、评估集/分层与层号。

**关键读数**（ΔMSE；item CI / country-cluster CI；负 = 前者更好）：

| 对比 | ΔMSE | item 95% CI | cluster 95% CI | Δ距离（均值） |
|---|---:|---|---|---:|
| union−orig，world valid all，L2 | **−18.4** | [−56.5, +15.5] | [−54.9, +13.5] | −45.7 km |
| union−orig，world valid all，L11 | **+5.9** | [−49.0, +61.7] | [−29.5, +43.1] | +32.7 km |
| union−orig，seen 110，L2 | +15.2 | [−11.1, +43.0] | [−5.8, +41.9] | +103.7 km |
| union−orig，seen 110，L11 | +27.4 | [−33.9, +91.5] | [−6.0, +67.1] | +137.4 km |
| union−orig，unseen→C3 12，L2 | **−287.3** | [−565.9, −90.8] | [−547.7, −88.2] | −1217.2 km |
| union−orig，unseen→C3 12，L11 | **−143.1** | [−303.4, −25.1] | [−303.2, −25.8] | −813.1 km |
| union−orig，unseen both 8，L2 | **−76.1** | [−144.5, −20.2] | [−146.6, −20.9] | −342.2 km |
| union−orig，unseen both 8，L11 | −66.6 | [−209.0, +36.2] | [−210.4, +36.7] | −137.8 km |
| new−orig，world valid all，L11 | +58.2 | [−12.9, +150.4] | [−6.2, +136.4] | +209.2 km |
| union−new，world valid all，L11 | **−52.4** | [−97.2, −21.1] | [−112.2, −12.3] | −176.5 km |
| union−new，Geo test，L11 | −1.7 | [−4.7, +1.4] | [−6.1, +1.7] | −2.2 km |
| union−new，Geo valid，L11 | −6.4 | [−18.2, −0.0] | [−21.4, −0.0] | +2.7 km |
| union−new，full unseen，L11 | **−66.6** | [−72.7, −60.7] | [−128.8, −27.1] | −355.3 km |
| new−orig，full unseen，L11 | **+92.0** | [+81.0, +103.4] | [+20.3, +199.5] | +519.0 km |
| union−new，strict8，L11 | **−21.7** | [−25.1, −18.5] | [−41.9, −7.8] | −143.8 km |
| union−new，strict8，L32 | −6.6 | [−9.1, −4.2] | [−17.1, +0.3] | −28.7 km |
| union−orig，strict8，L11 | −18.1 | [−30.4, −6.4] | [−88.4, +25.6] | −169.3 km |
| union−orig，strict8，L32 | −22.9 | [−33.1, −12.8] | [−58.8, +23.8] | −238.7 km |
| union−new，hard6，L11 | **−22.7** | [−26.5, −18.9] | [−47.6, −4.5] | −125.2 km |
| union−orig，hard6，L11 | −17.5 | [−32.3, −3.7] | [−114.0, +33.0] | −138.9 km |
| leak：union_noleak−union，world valid，L11 | +0.55 | [+0.02, +1.17] | [−0.0, +1.3] | +4.5 km |
| leak：new_noleak−new，Geo test，L11 | +0.04 | [−0.0, +0.1] | [−0.0, +0.1] | +0.1 km |

**怎么读**：

1. **原任务总体是平的**：union−orig 在 L2 −18.4、L11 +5.9，两个层的 CI 都跨 0；不能据此宣称"增广提升 world 任务"。
2. **分层后方向非常清楚**：在 110 条 seen 国家上 union 更差（L11 +27.4，CI 跨 0，但方向稳定）；在 12 条 unseen→C3 上 union 大幅更好（L11 −143.1，item 与 cluster CI 都不跨 0）；在 8 条 unseen both 上 L2 −76.1（显著）、L11 −66.6（CI 跨 0）。**增广的收益是"覆盖新国家"，代价是"稀释已见国家"**，总量相互抵消。
3. **`new` 单独训练会伤害原任务**（L11 +58.2，CI 跨 0 但 cluster 下界 −6.2），而 union 把 orig 加回来后在原任务上相对 new 显著更好（L11 −52.4，CI 不跨 0）。这支持"混合训练"而非"用新数据替换旧数据"。
4. **新任务上 union ≈ new**：Geo test −1.7（CI 跨 0）；Geo valid −6.4（CI 上界 −0.0，边缘）。加 500 条原数据不伤害新分布。
5. **full unseen 的 −66.6 是"22 国重叠"造成的假优势**：更干净的 strict8 上 union−new 只有 −21.7（L11），但仍显著（cluster CI [−41.9,−7.8]）；hard6 −22.7（cluster [−47.6,−4.5]）。**这是本实验最稳健的增广收益**：在 6–8 个完全未见的国家上，union 比 GeoNames-only 低 ~21–23 MSE（约 −20%）。
6. **vs orig 的 strict 收益在国家级不稳健**：L11 −18.1 / L32 −22.9，item CI 都不跨 0，但 cluster CI 都跨 0（8 国太少）；不能写成"增广显著优于原模型"。
7. **leak 控制接近 0**：union_noleak 比 union 只差 +0.55 MSE（L11，CI [+0.02,+1.17]），new_noleak 比 new 差 +0.04；不是"背 valid 答案"。注意 L11 的 +0.55 在 item bootstrap 下"显著"但实际幅度可忽略（0.3%），不要当成效应。
8. **item CI 与 cluster CI 的差别是本次实验的重要方法论点**：strict8 只有 8 个国家，同国样本高度相关；item bootstrap 给出 [−25.1,−18.5] 的"很窄"区间，cluster bootstrap 给出 [−41.9,−7.8] 的诚实区间。报告 strict unseen 时必须用 cluster CI（或逐国表），否则会高估精度。

图注候选（中文）：*"C3 配对效应（负 = 前者更好；细线 item bootstrap，粗灰线 country-cluster bootstrap）。world valid 上 union−orig 在 L2/L11 分别为 −18.4/+5.9，CI 均跨 0；收益集中在 unseen→C3 的 12 条（L11 −143.1）。strict8 上 union−new 在 L11 为 −21.7（cluster CI [−41.9,−7.8]），是唯一在国家级也稳健的增广收益；union−orig 的 cluster CI 仍跨 0。"*
图注候选（English）：*"C3 paired effects (negative = first set better; thin = item bootstrap, thick grey = country-cluster bootstrap). On world valid, union−orig is −18.4 at L2 and +5.9 at L11, both CIs crossing zero; the gain concentrates on the 12 unseen→C3 items (−143.1 at L11). On strict8, union−new is −21.7 at L11 (cluster CI [−41.9,−7.8]), the only augmentation gain that survives country-level resampling; union−orig still has a cluster CI crossing zero."*

---

## 4. `c3_strata_unseen.*` — 分层与逐国异质性

**画的是什么**：左 = world valid 在 L11 的三个分层（seen 110 / unseen→C3 12 / unseen both 8）× 三个训练集；中/右 = strict8 的 8 个国家在 L11/L32 的逐国 MSE（对数纵轴，每条线一个国家，横轴三个训练集）。

**world valid 分层读数（MSE，L11；见 `strata`）**：

| 分层 | n | orig | new | union |
|---|---:|---:|---:|---:|
| seen（world train 有该国家字符串） | 110 | 163.8 | 253.0 | 191.2 |
| unseen→C3（world 无、C3 train 有） | 12 | 215.2 | 65.3 | **72.1** |
| unseen both（两者都无） | 8 | 138.7 | 83.9 | **72.2** |

- **seen 国家：union 比 orig 差 +27.4（L11）；new 更差 +89.2**。8,382 条新样本把决策边界拉向 GeoNames 分布，稀释了原 500 条的国家级先验。
- **unseen→C3：union 72.1，接近 new 65.3，远好于 orig 215.2**。这是"增广补覆盖"的直接证据；12 条样本少，但效应大且 CI 不跨 0（§3）。
- **unseen both：union 72.2 最好**，说明即使某国在两边都没有样本，增广仍能通过"附近国家/全球布局"改善；但 n=8，CI 在 L11 跨 0（L2 显著）。

**strict8 逐国读数**（MSE；`orig / new / union`）：

| 国家 | n | L11 | L32 |
|---|---:|---|---|
| Costa Rica | 28 | 87.8 / 94.2 / **77.7** | **176.3** / 304.2 / 276.4 |
| Denmark | 41 | 31.3 / **7.2** / 8.2 | 16.3 / 17.1 / 16.9 |
| Hungary | 82 | **73.6** / 137.5 / 120.9 | 59.9 / **6.0** / 7.2 |
| Lebanon | 10 | 327.2 / **7.7** / 23.2 | 143.3 / **22.9** / 33.1 |
| Niger | 24 | **27.0** / 54.6 / 42.6 | 110.7 / 86.3 / **62.4** |
| Palestine | 24 | 68.0 / 33.4 / **10.7** | 143.8 / 29.6 / **27.2** |
| Taiwan | 34 | 254.2 / 199.0 / **123.1** | **59.6** / 93.7 / 74.6 |
| Yemen | 17 | **24.4** / 126.3 / 91.1 | 47.0 / **43.6** / 52.3 |

- **逐国方向高度异质**：L11 上 union 在 Costa Rica/Palestine/Taiwan 最好，但 Denmark/Lebanon 是 new 最好、Hungary/Niger/Yemen 是 orig 最好；L32 上 union 只在 Niger/Palestine 最好，Costa Rica/Taiwan 反而是 orig 最好。**"strict unseen 总体 union 最低"不是"每个国家都变好"**，而是少数高误差国家（Palestine/Taiwan/Niger）的尾部改善拉低了均值。
- 与距离分布一致：strict8 L11 的 MSE 优势主要是**尾部**——P90 距离 union 1612 km < new 1913 < orig 2219，但中位距离 union 1133 > orig 1101、< new 1277；L32 中位 union 539 ≈ new 516、远好于 orig 980，P90 union 1851 < new 2058。**不能只用 MSE 说"定位更准"**；要同时报中位/P90/hit100（见 JSON）。
- hit100（≤100 km）：strict8 L11 orig 0.8% / new 4.6% / union 1.9%；L32 orig 1.5% / new 3.5% / union 2.3%。union 的 MSE 优势不是"更多点落进 100 km"。

图注候选（中文）：*"左：world valid 三层国家×三个训练集（L11）。union 在 seen 国家变差、在 unseen→C3 与 unseen-both 变好。右：strict unseen 8 国的逐国 MSE（L11/L32；对数轴）。总体增益来自少数高误差国家，逐国方向高度异质。"*
图注候选（English）：*"Left: world-valid country strata by training set at L11; union hurts seen countries but helps unseen→C3 and unseen-both. Right: per-country MSE within the strict unseen set at L11/L32 (log scale). The aggregate gain comes from a few high-error countries; per-country directions are heterogeneous."*

---

## 5. 结论（可直接进报告 C3 小节）

1. **增广对原 world 任务没有稳定收益**：world valid 上 union vs orig 在 L2 **−18.4 MSE（−11.7%，CI 跨 0）**、L11 **+5.9（+3.5%，CI 跨 0）**；L2 是事后选层，L11 是课程口径。与指南2"扩增未证明稳定收益"的定性结论一致，但我们的分层解释了原因：**收益在 unseen 国家、损失在 seen 国家**。
2. **收益是覆盖而不是整体提升**：world valid 的 12 条 unseen→C3 上 union 比 orig 低 **143.1 MSE（−66.5%，CI [−303.4,−25.1]）**；seen 110 条上反而高 **27.4（+16.7%，CI 跨 0）**。`new` 单独训练在 seen 上差 +89.2、在原任务总体差 +58.2——**不能拿新数据替换原数据，只能混合**。
3. **新分布上 union ≈ new**：Geo test L11 **62.34 vs 64.02（Δ −1.7，CI 跨 0）**；Geo valid **81.42 vs 87.80（Δ −6.4，CI 上界 ≈0）**。加 500 条原数据不伤害 GeoNames 任务。
4. **严格未见国家上，增广相对 GeoNames-only 有稳健收益**：strict8 L11 **75.66 vs 97.35（−21.7，cluster CI [−41.9,−7.8]）**；hard6 **86.76 vs 109.43（−22.7，cluster CI [−47.6,−4.5]）**。但相对 orig 的 −18.1/−22.9 在 8 国 cluster bootstrap 下不显著；且逐国方向异质（§4）。
5. **`test_unseen` 的"未见"必须重新定义**：30 国中 22 国在 world train 出现，因此 orig 在 full unseen 上 L11 56.5 的优势不是"小数据泛化强"，而是"这 22 国它见过"。strict8/hard6 是本实验对指南协议的关键修正。
6. **增广没有突破 world 天花板**：union 最好的 world valid（L2）139.2 仍高于国家均值 oracle **113.8**；L11 172.9 更差。增广缩小了"未见国家"的差距，但没有改变"world 任务整体低于国家均值基线"的事实。
7. **方法学**：85 次？不，65 次训练全部有限、13/13 锚点逐位一致、no-leak 对照差 <1 MSE；strict unseen 必须用 country-cluster CI，item CI 会因同国相关而过度自信。

---

## 6. 局限与报告注意事项

- **层选择**：L11 是课程预注册层；L2（world valid 最优）和 L32（full unseen 最优）都是**事后在评估集上挑的**，不可当作无偏估计。对新任务，L11 是在 `new_valid` 上选的，`new_test` 是诚实留出；对原任务，world 没有带标签 test，L2 的 −18.4 只能写成"valid 上的描述性结果"。
- **strict unseen 国家太少**：8 国 / 260 条（hard6 6 国 / 212 条）；country-cluster CI 很宽，逐国差异大。不要用"8 国"外推到"所有未见国家"。
- **复合国家字符串**：world train 有 `Chad / Nigeria / Niger / Cameroon`、`Israel/Palestine` 等，Niger/Palestine 可能已有部分城市样本；hard6 是敏感性口径。若严格按城市级去重，还需要解析 world train 的地点名，本实验未做到。
- **seen/unseen 分层是字符串精确匹配**：world valid 中 `Estonia / Russia`、`Germany / Switzerland / Austria` 等复合标签被归入 unseen both，尽管其组成部分可能在 train 出现；12/8 条子集很小，CI 很宽。
- **单模型、单 prompt、单 seed**：Qwen3-4B-Base rev 906bfd4b；数据划分 seed 20261004；没有跨模型/跨模板复现。 GeoNames 标签本身有噪声（尤其小城市），未做坐标清洗。
- **与指南数字不可直接比**：指南2 的新 test MSE 18.65 用的是 40k steps + momentum 0.9、不同的划分/选型协议；本实验统一 std13（2,000 epochs 全批量 GD）以与课程/C2 对齐，绝对 MSE 更高（62.3）不代表特征更差。引用指南时注明"同学参考报告（非官方）"。
- **原始 MSE 的尾部敏感性**：world/Geo 的 MSE 都由少数大误差点主导；本实验同时报告了中位/P90/hit100 与周期 MSE，不要只报均值 MSE。C1 的 Tomanivi 现象在 C3 没有单条 96% 那么极端，但同样存在。
- **不做提交**：C3 只用于报告的附加探究；课程提交仍是单层 2560 维原 world 探针（见报告 §提交说明）。不要把 union 权重当课程答案提交。
- **可复现命令**：评估 `python -X utf8 experiments/C3/c3_eval.py`；训练需集群 A100 + `experiments/C3/run_c3.sbatch`（原始 dump、特征均 gitignored，重跑需按模块 docstring 下载 GeoNames 并重新提特征）。
