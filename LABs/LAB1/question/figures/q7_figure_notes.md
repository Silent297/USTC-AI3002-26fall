# Q7 逐图解释说明

> 题目：从训练集和验证集的表现，是否能够强烈说明模型内部存在地理坐标的表征，而不是仅仅识别了输入中的国家？可参考原始论文 [3]（Gurnee & Tegmark, *Language Models Represent Space and Time*, ICLR 2024）的部分结论和 Q6 的可视化结果；也可与按训练集国家分组的平均坐标基线比较，考察同一国家内不同地点的预测。
> 本文档记录 3 张 `q7_*` 图：**画的是什么 / 关键读数 / 证明了什么 / 报告怎么用**，供写 LaTeX 报告时直接取材。
> 读数口径：模型 = world raw L11 历史最优探针（同 Q6：`results/slurm_layers/world/weights/world_layer11_extra0/weights.npz`，standardize=false，推理时**不再 z-score**）；pred = X·wᵀ + b，目标列序 (latitude, longitude)；MSE = 两坐标平方误差平均（与训练管线 `train23.load_split` 一致）。基线 = **oracle 国家均值**：每国取**训练集**该国坐标均值，用**真实国家标签**查找（oracle 对基线有利——baseline 多出"真国家标签"信息），训练集未见国家回落全局均值 (15.67, 12.59)，其 MSE = **3311.27**（与 Q6 报告的元素平均口径锚点一致）。improvement = MSE_base − MSE_model（>0 表示模型更好），全部 CI 为 paired bootstrap（10k 重抽，seed 0，元素平均口径）。全部读数由 `code/q7_country_baseline.py` 打印、一键复现。
> 分层集合：all 130 / **seen** 110（训练集中出现过的国家）/ **multi** 73（seen 中同国 ≥2 条 valid）/ **single** 37（seen 中独苗）/ **unseen** 20（19 个训练集未出现的国家；Costa Rica 2 条）。valid 30 国家中除 Costa Rica 外未见国家均只有 1 条。
> 维护约定：每完成一批绘图，在 `figures/` 下新建一份 `<题号>_figure_notes.md`，格式同本文与 `q2/q3/q6_figure_notes.md`。

## 总览

| 图 | 一句话 | 回答的内容 |
|---|---|---|
| `q7_baseline_bars` | 模型 vs oracle 国家均值基线的分层 MSE 对比（左：seen 四集合；右：unseen vs 全局均值） | 模型表现是否只是"识别国家 + 查该国平均坐标" |
| `q7_within_country` | 同国中心化散点（lat / lon）+ US 地图放大（n=8，真值→预测箭头） | 同一国家内部不同地点能否被区分 |
| `q7_unseen_map` | 训练集未见国家的 20 条样本全球地图（真值→预测箭头，标注最差 3 国） | "查表不可能"时模型还剩多少定位能力 |

一句话主线：**"仅识别国家"不足以解释全部表现**——未见国家样本上模型大幅胜过全局均值基线（improvement +1024 [514, 1654]，查表不可能），同国内**纬度**差异有弱但显著的跟踪（r=0.201 [0.021, 0.389]）；**但精细的国内定位很弱**——经度国内相关≈0（r=0.046，ns），seen 国家上 oracle 国家均值反而反超模型（−35.1 [−60.9, −7.1]），且验证集地点与训练集高度重叠（89/130 在 3° 内）使"记住常见地点"也能取得好成绩。综合证据强度为**中等**，与论文 §4.1 对同类 block-holdout 实验的自评 "*weak evidence*" 一致；"模型生成时确实使用该表征"的强证据（论文 §5 神经元/干预）超出本实验范围。

---

## 1. `q7_baseline_bars.*` — 分层基线对比

**画的是什么**：左图 = 四个 seen 集合上 model（红）vs oracle 国家均值基线（蓝）的 valid MSE 条形，条间标注 improvement [95% CI]；右图 = unseen 20 条上 model vs 全局均值基线（灰）。注意左侧为"对基线有利"的比较（oracle 国家标签 + 训练集记忆）。

**关键读数**（valid MSE；medE = 单点欧氏误差中位数）：

| 集合 | 模型 | oracle 国家均值 | improvement [95% CI] | medE 模型/基线 |
|---|---|---|---|---|
| all (n=130) | 150.43 | 278.24 | **+127.81 [27.24, 253.03]** | 9.52 / 5.91 |
| seen (n=110) | 148.89 | **113.77** | **−35.12 [−60.90, −7.08]** | 8.88 / 4.23 |
| multi (n=73) | 204.97 | **159.58** | **−45.39 [−82.22, −6.94]** | 9.84 / 6.36 |
| single (n=37) | 38.24 | 23.39 | −14.85 [−39.79, 14.07]（ns） | 5.83 / 2.60 |
| unseen (n=20) | 158.90 | 1182.83（global） | **+1023.93 [513.94, 1654.14]** | 12.69 / 35.85 |

- 稳健性：multi 去掉南极洲（2 条，双方 MSE 均 ≈3.4e3，主导集合均值）后 n=71：model **115.87** vs base **65.24**——方向不变，oracle 仍更好。
- **数据重叠侦察**：valid 样本到"同国最近训练点"的距离中位数仅 **2.03°**，40/130 < 1°、89/130 < 3°——验证集大量地点与训练集重合或邻近（同一城市/近郊），oracle 均值基线相当于"查表 + 近邻插值"，其强度有相当部分来自数据重叠。
- 极端个例：Jamaica base MSE = **0.01**（valid 地点与训练地点几乎同点，"查表"近乎精确命中）；Guatemala、South Africa、Australia、China 同模式（base 0.55–97.5）。反向例：United States model 78.15 vs base 121.92（+43.77）、Canada 399.68 vs 481.53（+81.85）——模型在部分国家反超 oracle；Antarctica 双方都差（3.37e3 vs 3.51e3）。

**证明了什么**：
1. **全局结论由 unseen 驱动**：查不到表的 20 条样本上模型 +1024；一旦限定"训练见过的国家"，oracle（多出真实国家标签 + 训练记忆）反超 35.1。两个方向共同说明：**模型确实携带"超出国家身份"的信息（否则 unseen 上不可能大胜），但"国家身份 + 记住常见地点"是它在 seen 国家上的主要得分来源**。
2. single（独苗国家）上模型与基线差距不显著（CI 跨 0）且双方 MSE 都很低（38.24 / 23.39，medE 5.8°/2.6°）——独苗样本多为"国家≈地点"的小国，识别国家即完成任务；这正说明该子集可用于对照"纯国家信息"能走多远。
3. Q6 的"大洲级精度"在此得到量化基线佐证：oracle 基线在全球尺度上就相当于"知道国家质心"，模型 all-130 上胜之 +127.8，主要来自它能处理未见国家。

**报告怎么用**：Q7 正文第一层证据（量化表）；讨论"seen 上 oracle 反超"与"unseen 大胜"并存的张力，并给出数据重叠解释。
图注候选（中文）：*"模型（world raw L11 探针）与 oracle 国家均值基线的 valid MSE 对比。左：训练集出现过的国家按样本数分层；右：19 个训练集未见国家（20 条）。improvement = 基线 MSE − 模型 MSE（>0 模型更好），误差棒为 paired bootstrap 95% CI。"*
图注候选（English）：*"Valid MSE of the probe vs an oracle country-mean baseline (per-country train mean, looked up with true country labels). Left: countries seen in training, stratified by sample count; right: the 19 countries absent from training (20 samples). Improvement = baseline MSE − model MSE (positive: model better); whiskers are paired-bootstrap 95% CIs."*

---

## 2. `q7_within_country.*` — 同国中心化分析 + US 放大

**画的是什么**：左/中 = 73 条 multi∩seen 样本的散点，坐标 = 真值/预测**相对训练集该国均值**的偏差（度），虚线为 y=x；右 = United States（n=8）局部地图，灰空心圆 = 真值、红点 = 预测、箭头 = 真值→预测。

**关键读数**：
- **纬度**：pearson r = **0.201 [0.021, 0.389]**（弱显著），spearman 0.244，slope **0.115**。
- **经度**：pearson r = **0.046 [−0.152, 0.214]**（不显著），spearman −0.055，slope **0.027**。
- US 放大（n=8）：中心化 r = **0.69 (lat) / 0.60 (lon)**，方向整体正确（如 (47.70, −87.50) 苏必利尔湖 → (45.30, −91.54) 西北向；但 (29.95, −90.07) 新奥尔良 → (37.97, −98.97) 偏北错向）。
- 两坐标 slope 都 ≪1（0.115 / 0.027）：**即使方向对，幅度也强烈收缩**——模型倾向于把同国地点拉向"国家中心"，与 Q6"大洲级精度"一致。

**证明了什么**：
1. **"仅识别国家"假设的判据**：若模型只知道国家，则同国内中心化预测应恒为常数（散点全落在 y=0 上、斜率为 0）。实测纬度斜率 0.115、相关弱显著 ⇒ 国家内部信息**存在但弱**；经度则几乎无国内分辨力。
2. **纬度/经度不对称再次出现**（与 Q6 的 MAE 4.25° vs 10.85°、top-10 误差几乎全由经度贡献同构）：模型敏感的是"气候带式"纬度线索，经度需要具体地名知识。
3. 单国噪声提示：US（n=8）方向正确的印象不能外推——n=8 的相关 CI 很宽；样本量第二的 India（n=7）预测聚集在 21–24°N 附近、方向感弱（逐条读数见脚本输出）。
4. 该图与左图联合读：这里测的是**相对位置**（论文 §4.1 术语 "relative position correct"），经度上"相对对、绝对差"的结构与论文 held-out 实验结论一致。

**报告怎么用**：Q7 正文第二层证据（同国内分辨力）；与 Q6 图的不对称结论相互印证；US 放大图作"同一国家、不同地点、粗略正确"的直观例证（须注明 n=8、噪声大）。
图注候选（中文）：*"同一国家内部的定位分辨力。左/中：73 条多样本国家样本相对该国训练均值的偏差（虚线 y=x；纬度 r=0.20 弱显著，经度 r=0.05 不显著）。右：美国 8 条样本的真值（灰圈）→预测（红点）箭头；中心化 r=0.69/0.60（lat/lon），方向大体正确但幅度收缩。"*
图注候选（English）：*"Within-country resolution. Left/middle: per-sample deviations from the country's train mean (dashed: y=x; lat r=0.20 weakly significant, lon r=0.05 n.s., n=73). Right: the 8 US samples, ground truth (gray circles) → prediction (red dots); centered r=0.69/0.60 (lat/lon) — coarsely direction-correct with strong shrinkage."*

---

## 3. `q7_unseen_map.*` — 未见国家样本地图

**画的是什么**：全球 PlateCarree 地图（cartopy，110m 海岸线；**合规：不画国界线**）。20 条 unseen 样本：灰空心圆 = 真值、红点 = 预测、箭头 = 真值→预测；标注误差最大 3 国。

**关键读数**（单点欧氏误差，度）：
- 最差 3：**Mauritius 38.03**（(−20.16, 57.50) → (−21.51, **19.49**)，经度掉到非洲大陆）；**Estonia / Russia 31.60**；**Finland 24.53**。
- 其余多在 10–25°：Liberia 23.75、Andorra 21.32、Chad 16.93、Bahrain 15.83、Costa Rica 19.49 / 11.88（2 条均落在中美南侧）。
- 最好：**Denmark 0.79**（55.68, 12.57 → 56.40, 12.25）；Rwanda 6.11、Slovakia 7.83、Mauritania 8.23。
- 结构性观察：欧洲样本多条预测落在欧洲邻区（Netherlands/Luxembourg/Hungary 等 10–13°）；全部 20 条误差都远小于全局均值基线的 1182.83。

**证明了什么**：
1. **"超出国家身份"的最直接证据**：这些国家在训练集中完全未出现，**按国家查表在原理上不可能**；模型仍能把样本放到正确的大洲/区域（欧洲→欧洲、中美洲→中美洲）——对应论文 §4.1 的结论结构："relative position correct, absolute position often wrong"。
2. 失败模式与 Q6 一致：误差集中在**经度/小岛屿/洋面**（Mauritius 落非洲大陆、Estonia/Russia 东偏 ~31°），再次对应"训练覆盖决定精度"。
3. 与右图 baseline 条的联合读法：+1024 的巨大 improvement 说明该比较的基线（全局均值）很弱；它证明的是"模型远不止于一个全局平均"，而非"模型精确"——精确度仍受 Q6/图 2 的中等误差约束。

**报告怎么用**：Q7 正文第三层证据（泛化到未见国家）；与论文 §4.1 直接对照。
图注候选（中文）：*"19 个训练集未见国家的 20 条验证样本：灰空心圆 = 真值、红点 = 预测、箭头 = 真值→预测。模型 MSE=159 vs 全局均值基线 1183（improvement +1024，paired bootstrap 95% CI [514, 1654]）。仅绘海岸线，无国界线。"*
图注候选（English）：*"20 valid samples from 19 countries absent from training. Gray circles: ground truth; red dots: predictions; arrows: truth→prediction. Model MSE 159 vs global-mean baseline 1183 (improvement +1024, paired-bootstrap 95% CI [514, 1654]). Coastlines only; no national borders."*

---

## 附 A：Q7 结论与论文 [3] 的对应

| 论文 [3]（Gurnee & Tegmark 2024） | 本实验对应 |
|---|---|
| §3.1 world 数据集 LLaMA-70B R²=0.911（60% 深度）；我们的 Q6 为 R²=0.955（Qwen3-4B L11） | Q6 地图 + 本文三图共用同一探针 |
| §4.1 **block holdout**：扣掉某国训练数据后，模型对该国地点仍"相对位置对、绝对位置偏"（held-out 国家近国误差 0.071→0.164 vs 随机 0.5）；作者自评 "*weak evidence*"、"does not fully rule out the underlying binary features hypothesis" | **Q7 主体**：unseen 20 条 +1024 大胜全局基线，但 seen 上 oracle 反超、同国内经度无信号——同样只能给"中等强度"证据，与论文自评一致 |
| §4.2 PCA top-k：Spearman 随 k 增长远快于 R²（"特征在，但 probes 需要更多参数转换坐标系"） | 可类比：Q7 的"方向对、幅度收缩"（slope 0.115/0.027）同样是"坐标系→人类坐标"的转译损耗 |
| §5 **space/time neurons + 干预**：论文称个体神经元 + 消融/干预实验是 "very strong evidence"（模型确实使用该表征） | **超出本实验范围**：无模型权重、无法做激活干预；只能在报告中引用 |
| §7 world model 猜想 | 讨论段素材 |
| README §1.2 提醒："可解码不代表 LLM 在生成过程中一定使用了该信息" | Q7 结论的强度上限必须声明这一点 |

**Q7 分层答案（一句话各层）**：
1. **不是"仅仅识别国家"**：unseen 上查表不可能却 +1024（强证据）；同国纬度 r=0.20 弱显著（弱-中证据）；all-130 上模型胜 oracle +127.8（但由 unseen 驱动，需注明）。
2. **限度**：seen 上 oracle 反超 −35.1（CI 不含 0）；经度国内相关≈0；valid↔train 地点重叠大（89/130 <3°），"记住常见地点"贡献可观。
3. **强度定性**：**中等**。强于"纯国家分类器"，弱于论文 §5 级证据（神经元 + 干预）；与论文 §4.1 的 "weak evidence" 自评、以及本课 README "可解码 ≠ 使用" 的警示一致。

## 附 B：相关数字的出处

- 生成脚本：`code/q7_country_baseline.py`（一条命令复现全部读数：分层基线表 + CI、稳健性、重叠侦察、US/India 与 unseen 逐条、同国中心化统计、逐国表）。
- 权重：`results/slurm_layers/world/weights/world_layer11_extra0/weights.npz`；复算锚点 valid MSE=150.4277（train MSE 116.9279）。
- 数据：`datasets/world/{train,valid}.layer11.npy` + `{train,valid}.csv`（真值 lat/lon、country）；train 500 条 / 158 国，valid 130 条 / 78 国。
- 图：`figures/q7_{baseline_bars,within_country,unseen_map}.png`（dpi=150）+ 同名 `.pdf`，2026-10-14 生成（cartopy 0.26.0，Natural Earth 110m 海岸线）。
