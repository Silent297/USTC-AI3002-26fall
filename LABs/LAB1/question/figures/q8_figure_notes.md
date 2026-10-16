# Q8 逐图解释说明

> 题目：仔细阅读训练集和验证集，模型的训练集和验证集存在哪些分布上的差异？你是如何应对的？
> 本文档记录 3 张 `q8_*` 图：**画的是什么 / 关键读数 / 证明了什么 / 报告怎么用**，供写 LaTeX 报告时直接取材。第 3 张（`q8_cv_gap`）为"应对"实验（2026-10-16 CV 完成，见 §3）。
> 读数口径：truth（主）train 5088 / valid 300 句真假分类（句尾 token 层表征）；world（对照）train 500 / valid 130（国名 token 层表征）。探针 = truth **L20**（`results/slurm_layers/truth_b/weights/truth_layer20/weights.npz`，raw 特征，λ1=λ2=1e-3，lr=5e-3，5000 ep），valid acc 锚点 **0.8967**（= 分层训练报告值，亦为 Q3 T3 锚点）；pred = sigmoid(X·w+b) > 0.5。
> 错拼（typo）检测 = pyspellchecker 字典 + **相邻字母交换**（transposition）判定；仅覆盖这一种噪声类型（其它类型未计入 ⇒ 噪声率是下界），且对专有名词有少量假阳性（`deen→eden`(Paula Deen)、`dwyane→dwayne`(真实拼法即 Dwyane)、world 侧 `centre/rila/iber/abbe/addo` 全是地名假阳性，详见 §1）。
> 首实体（lead entity）= 句首连续大写词组（剔除 The/A/One/If 等停用开头；无法识别的主体记 "n/a"），用于"主体新颖度"统计。
> 全部 CI = bootstrap 95%（准确率 2000 次重抽；逐层 800 次；零假设 200 次重抽）；全部读数由 `code/q8_split_analysis.py` 打印、一键复现。
> 维护约定：每完成一批绘图，在 `figures/` 下新建 `<题号>_figure_notes.md`，格式同本文与 `q2/q3/q6/q7_figure_notes.md`。

## 总览

| 图 | 一句话 | 回答的内容 |
|---|---|---|
| `q8_split_diff` | 四个分布差异：错拼率（truth 3.5×）、主体新颖度（77.4% vs 零假设 20.8%）、模板/长度、world 的字符串新颖 vs 空间新颖 | "差在哪" |
| `q8_stratified_acc` | L20 探针在 valid 上的分层 acc（typo × 主体见/未见 × 标签）+ 13 层 typo 敏感度曲线 | "差异对模型的实际影响多大" |
| `q8_cv_gap` | 随机 3 折 CV vs 按主体分组 3 折 CV（仓库 model.py，L20 特征） | "怎么应对：两种划分口径实测几乎无差（0.869 vs 0.877），探针不依赖主体记忆——valid 的 0.8967 不是注水" |

一句话主线：**truth 的 valid 是"设计过的 OOD 挑战集"**——错拼率约为 train 的 3.5 倍（24.7% vs 7.0%，两侧标签内均衡），且 77.4% 的样本主体不在 train 中出现（零假设 20.8% [15.6, 25.6]，远远出界），train 侧特有模板（"One can"/"If one"，3.0%）在 valid 几乎消失；**但最终 L20 探针对这些差异是稳健的**：干净 vs 带错拼、主体见 vs 未见的两组差异都不显著（点估计甚至略偏"难组"）；错拼的危害只体现在**浅层**（L5–L11 干净-错拼差 +5~7pp），说明深层表征比浅层词面特征更抗噪。**应对**的核心是评估方法论：分层 + bootstrap CI 读数、不做"在 valid 上反复调参"的选择、"同分布口径"用 train 内部划分对照（§3：随机 vs 分组 CV 几乎无差 ⇒ valid 成绩不是靠主体记忆堆出来的）。

---

## 1. `q8_split_diff.*` — 四个分布差异

**画的是什么**：(a) 相邻字母交换型错拼句率条形（truth train/valid + world 对照）；(b) "首实体未在 train 出现"占比：valid 实测 vs 零假设（从 train 随机抽 300 条同样统计，200 次重抽）的直方图；(c) 句长 ECDF（train vs valid）+ "One can"/"If one" 模板计数；(d) world：新国家 / 新地点字符串 / 新字符串且距同国训练点 ≥3° 的条数。

**关键读数**：
- **(a) 错拼率**：truth train **354/5088 = 7.0%**（label=1 7.1% / label=0 6.8%），valid **74/300 = 24.7%**（25.3% / 24.0%）——**3.5×**，且对标签均衡（不是"假句才加噪"）；错误类型为相邻字母交换的"形近错"（`mkaes→makes`、`fihgters→fighters`、`nrivana→nirvana`、`connectiuct→connecticut`、`baskebtall→basketball`）。world 两侧 0.8% vs 0.8%，且全部为专有名词假阳性（`centre`、`rila`、`iber`、`abbe`、`addo`）⇒ **错拼噪声是 truth 特有、且只在 valid 侧加重**。
- **(b) 主体新颖度**：valid 212/300 条主体可识别，其中 **164 条（77.4%）主体未在 train 出现**；零假设 200 次重抽 mean **20.8%**、97.5% 分位 **25.6%**——**远超**。稳健性：typo 落在主体内部的仅 8 条（逐条核查见脚本输出：`Paula Deen→Paula Eden`、`Dwyane Wade→Dwayne Wade`×7，修正后**全部仍为"train 未见"**）⇒ 77.4% 不受错拼污染。valid 高频新主体：Dave Grohl ×8、Dwyane Wade ×7、The Offspring ×6、Popeye ×6、Arnold Schwarzenegger ×5、Robert De Niro ×5。
- **(c) 模板/长度**：`"One can"` train 129（2.5%）vs valid 2；`"If one"` train 24 vs valid 0 —— train 特有的"情态否定"句式在 valid 几乎消失（train 合计 ~3.0% vs valid ~0.7%）；句长 train mean 61.0 / max **487** vs valid mean 54.2 / max **162**（valid 长尾被截断）。
- **(d) world**：20/130 条来自 train 未见国家；126/130 条地点字符串 train 未出现（4 条字面重合），但其中大多数仍距"同国最近训练点"< 3°（全部 130 条里 89 条 <3°）——**字符串新颖 ≠ 空间新颖**；新字符串且真正远离（≥3°）只有 **41** 条。

**证明了什么**：
1. train/valid 的划分**不是同分布随机划分**：噪声（仅 truth、且 valid 侧 3.5×）与主体新颖度（77.4% vs 零假设 20.8%）两个维度都显著偏离随机划分的零假设——valid 是"轻量 OOD 挑战集"（新主体 + 加噪），这也是"valid 指标普遍低于 train 内部指标"的结构性原因。
2. truth 与 world 的差异**维度不同**：world 无噪声差异，差异在**实体/字符串层面**（新国家/新地名），且字符串新颖大多只是"换了个说法"，真正的空间新颖（≥3°）只有 41/130——两题 valid 的"难"来源不同：truth 难在**噪声 + 新主体**，world 难在**地理外推**（Q6/Q7 已量化：中位误差 9.5°、unseen 国家 20 条）。
3. truth 的噪声是**单向的**（只加在 valid）：如果训练时按"清洗错拼"来预处理，train（7% 噪声）与 valid（24.7% 噪声）的口径反而更不一致；正确做法是保持原样让模型见到噪声（我们的特征提取也未做任何文本清洗，见 §4 应对）。

**报告怎么用**：Q8"分布差异"部分的量化主图（a/b/c/d 四个读数直接引用）；与 world 的对照用来强调"两个数据集各自不同的 OOD 设计"。
图注候选（中文）：*"训练集/验证集的四个分布差异。(a) 相邻字母交换型错拼句率（truth valid 24.7% vs train 7.0%；world 两侧 0.8% 且全为专有名词假阳性）；(b) valid 主体新颖度 77.4%，vs 零假设（train 随机抽 300，200 次重抽）均值 20.8%、97.5% 分位 25.6%；(c) 句长 ECDF 与 train 特有模板计数；(d) world 验证集在新国家/新字符串/真正空间远离（≥3°）三个口径下的条数。"*
图注候选（English）：*"Four train-vs-valid distribution gaps. (a) Adjacent-transposition misspelling rate (truth valid 24.7% vs train 7.0%; world 0.8% both, all proper-noun false positives); (b) subject novelty: 77.4% of valid samples have a lead entity unseen in training, versus a random-split null of 20.8% [15.6, 25.6] (200 reps); (c) sentence-length ECDF and train-only templates; (d) world: new countries / new place strings / genuinely distant (>3°) cases."*

---

## 2. `q8_stratified_acc.*` — 探针在差异上的分行为

**画的是什么**：(a) L20 探针在 valid 上的分层准确率（all / clean / typo / 主体 seen / unseen / n-a 六组，误差棒 = bootstrap 95% CI，虚线 = 总体 0.8967）；(b) typo × 主体 的四格交叉；(c) 13 层探针在**干净**与**带错拼**子集上的准确率曲线（带 CI 带）+ 全体曲线（灰虚线）。

**关键读数**（L20 探针，anchor 复算 0.8967 逐位一致）：
- 分层（a）：all **0.897** [0.860, 0.930]；clean **0.885** [0.841, 0.925]（n=226）；typo **0.932** [0.865, 0.986]（n=74）；主体 seen **0.917** [0.833, 0.979]（n=48）；unseen **0.884** [0.835, 0.933]（n=164）；n/a **0.909**（n=88）。diff clean−typo = **−0.047 [−0.115, +0.024]**；diff seen−unseen = **+0.033 [−0.063, +0.119]**——**两组差异都不显著**。
- 交叉（b）：clean×seen 0.900（n=40）、typo×seen **1.000**（n=8）、clean×unseen 0.864（n=118）、typo×unseen 0.935（n=46）——四个格子点估计单调不降顺序为 clean×unseen < clean×seen < typo×unseen < typo×seen，即**对新主体与错拼各自都未见明显惩罚**。
- 标签拆解：label=1：clean 0.911 vs typo 0.895（平）；label=0：clean 0.860 vs typo **0.972**（n=36，点估计大涨但样本小，CI 宽）。
- 组成与时点核查：typo 子集组成（seen 8 / unseen 46 / n/a 20）与 clean（40 / 118 / 68）接近；**排除 Dave Grohl / Dwyane Wade 两个重复家族后 typo 组 acc 仍 0.935（n=62）**（非重复句假象）；错拼落在**主体内部**的 8 条 acc 0.875，落在**句内其它位置**的 58 条 acc 0.966（提示"改主体名"比"改普通词"更伤，但 n 小）。
- 逐层（c）：全体 acc 单调爬升到 L20 峰值 0.897 后回落（与分层训练报告完全一致，13 层全对拍 OK）；**干净−typo 差随层反转**：L5 +0.052 / L8 +0.064 / L11 +0.067（浅层怕错拼，显著方向稳定），L17 −0.061 / L20 −0.047 / L29 −0.042（深层点估计反向），过渡层 L14 +0.026、L26 +0.011。

**证明了什么**：
1. **"差异真实存在，但最终模型对差异稳健"**：虽然 valid 的错拼率达 train 的 3.5 倍、77.4% 主体是新的，L20 探针在这两类子集上的 acc 与干净/见过子集的差异在 95% 水平上不可分辨（CI 跨 0，点估计甚至偏高）——这解释了为什么 truth 的 valid acc（0.8967）能逼近 train 内部泛化水平，而不是被 OOD 拉垮。
2. **浅层怕错拼、深层不怕**：L5–L11 干净−typo 差 5–7pp（这些层接近词面/局部特征，字符级扰动直接改变激活），而 L17 之后注意力/语义级表征把错拼"归一化"掉了——这为"选深层做探针"提供了又一条独立理由（Q4/Q9 的选层结论在噪声鲁棒性维度上一致）。
3. **不过度解读小差异**：单格最大"奇观"（label=0 的 typo 组 0.972、typo×seen 1.000）n 都只有 8–36，CI 极宽；报告中只应作为"未见惩罚"的证据，不作为"错拼反而有用"的结论。

**报告怎么用**：Q8"如何应对/影响评估"主体证据之一——先量化差异（§1），再用分层 acc 证明"对我们的最终模型影响不显著、且选 L20 在鲁棒性上也有理由"。
图注候选（中文）：*"L20 探针在验证集上的分层准确率。（a）总体 / 干净 / 带错拼 / 主体见 / 主体未见 / 无法识别主体 六组（误差棒为 bootstrap 95% CI；虚线 = 总体 0.8967）；（b）错拼 × 主体 四格交叉；（c）13 层探针在干净与带错拼子集上的准确率（阴影为 95% CI），浅层（L5–L11）对错拼敏感（差 +5~7pp），深层（≥L17）不敏感。"*
图注候选（English）：*"Stratified accuracy of the L20 probe on the valid set. (a) six strata with bootstrap 95% CIs (dotted line: overall 0.8967); (b) typo × subject cross; (c) per-layer accuracy on clean vs typo subsets: shallow layers (L5–L11) lose 5–7 pp to misspellings, deep layers (≥L17) do not."*

---

## 3. `q8_cv_gap.*` — 随机划分 vs 主体分组 CV（应对实验）

**画的是什么**：train 内部 3 折 CV 的 held-out 准确率（L20 特征、仓库 `model.py` 同配置 lr=5e-3 / 5000 轮 / $\lambda_1=\lambda_2=10^{-3}$、n=3785 条可识别主体行）：蓝 = 随机划分（KFold），红 = 按主体分组划分（GroupKFold，同主体不跨折）；红/蓝虚线 = 两种划分的 3 折均值，灰点线 = 实际 valid 0.8967（OOD + 错拼口径）。

**关键读数**（2026-10-16 完成，全批量 GD 9 次拟合）：
- 随机：0.867 / 0.858 / 0.883，均值 **0.869**（折间 std ≈ 0.010）；分组：0.871 / 0.880 / 0.881，均值 **0.877**（std ≈ 0.004）；**gap = 随机 − 分组 = −0.008**（量级在折间噪声内）。
- 测试折"主体未见"占比：随机 **38%** vs 分组 **100%**（分组折的测试主体全部不在训练折出现）。
- 无发散（9 次拟合 all finite）。

**证明了什么**（诚实结论：假设未被证实，但有信息量）：
1. **"随机划分高估泛化"在本任务上并未发生**：即使测试折 100% 主体未见，分组 CV 的 held-out acc（0.877）也不低于随机划分（0.869）——与分层结果（valid seen vs unseen 差 ns）独立吻合。合起来说明：**L20 探针的准确率不依赖"记住主体"**，valid 的 0.8967 没有被主体新颖性注水；也说明 77.4% 主体新颖度这个"差异"对最终模型是无害差异，而不是需要修复的缺陷。
2. **注意口径**：CV 每折训练量 n_tr≈2523 < 最终模型 5088，CV 测试折又来自 train 分布，所以**不能把 CV 均值与实际 valid 0.8967 直接相减**（图上三条线的相对位置只作参考）——该图回答的是"两种划分口径差多少"，不是"valid 比同分布高/低多少"。
3. 方法层面仍成立：用分组 CV 做"同分布口径"的对照设计本身是对的（若真有主体记忆效应，这张图就是量化它的工具）；本数据上结论是"没有该效应"，报告按此如实写。

**报告怎么用**：Q8"如何应对"的量化证据——(a) 随机 vs 分组几乎重合 ⇒ valid 成绩不是主体记忆堆出来的；(b) 与 §2 分层结果互证（两条独立路径同结论）。图注候选（中文）：*"train 内部 3 折 CV（L20、同配置）：随机划分 vs 按主体分组划分（测试折主体未见比例 38% vs 100%）的 held-out 准确率——均值 0.869 vs 0.877（差 −0.8pp 在折间噪声内）；灰点线为实际 valid 0.8967。两种划分口径几乎无差，说明 L20 探针不依赖主体记忆。"*

---

## 附 A：Q8 答案要点（"差异 + 应对"清单）

**差异（已证实，见 §1）**：
1. 噪声：truth valid 错拼句率 24.7%，约为 train（7.0%）的 3.5 倍，标签内均衡；world 无此差异（0.8% vs 0.8% 全为假阳性）。
2. 主体新颖：77.4% 的 valid 样本主体未在 train 出现（零假设 20.8% [15.6, 25.6]）；world 对应地有 20/130 未见国家、126/130 新字符串（多数空间上邻近，真正远的有 41 条）。
3. 模板/长度：train 特有的 "One can / If one" 句式（3.0%）在 valid 基本消失；valid 句长被截断（max 162 vs 487）。

**应对（做法与理由）**：
1. **评估读数方式**：一切子集比较用 bootstrap CI（n=300 的单点差异 ≈ ±3.5pp，30 条子集 ≈ ±10pp），不把 CI 跨 0 的差异当选层/选参依据；分层（typo / 主体 / 标签）报告而不是只看总 acc。
2. **不向 valid 泄漏**：超参与层数选择依赖扫描/训练报告 + CI/配对检验口径，训练用固定预算（不按 valid 早停），避免"在 300 条 OOD 样本上反复挑选"。
3. **文本不做清洗**：特征提取对 raw 文本（含错拼）直接前向；若"修错拼"会同时改变 train/valid 分布口径，且模型部署时也要面对噪声——保持原样最优（实验上：L20 对错拼不敏感，浅层敏感 ⇒ 选深层即天然的抗噪手段）。
4. **口径声明**：valid 指标是"OOD + 加噪"的严格口径，报告中注明其保守性；同分布口径用 train 内部划分（见 §3 分组 CV 的量化）。
5. **稳健性交叉验证**：主体内 typo 逐条核查（8 条）、重复家族剔除敏感性（Grohl/Wade 剔除后仍 0.935）、逐层敏感度曲线——三类检查均不改变结论。

## 附 B：相关数字的出处

- 生成脚本：`code/q8_split_analysis.py`（一条命令复现：四组分布统计 + 零假设 200 次重抽 + L20 分层 acc/CI + 13 层敏感度 + CV；`Q8_SKIP_CV=1` 可跳过 CV）。
- 权重：13 层 truth 探针 `results/slurm_layers/truth_{a..d}/weights/truth_layer*/weights.npz`（L20 = `truth_b/weights/truth_layer20/`，valid acc 锚点 0.8967，13 层全对拍分层训练报告值 OK）。
- 数据：`datasets/truth/{train,valid}.{csv,layer*.npy}`、`datasets/world/{train,valid}.csv`（对照）。
- 工具：pyspellchecker（ai26 环境）；错拼类型计数 train 318 / valid 95 种。
- 图：`figures/q8_{split_diff,stratified_acc,cv_gap}.png`（dpi=150）+ 同名 `.pdf`，2026-10-16 生成。
