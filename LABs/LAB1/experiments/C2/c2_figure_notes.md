# C2 逐图解释说明

> 实验：Prompt 变体特征提取 + 线性探针（world train→valid，500/130 条）。**附加实验（非课程提交内容）**；题目来源 `guide/指南综述与可做实验清单.md` 创新点 6。目的：与 Q7 反向呼应——Q7 问"探针是否依赖国家名先验"，C2 直接切掉国家信息、指令或换问法，看坐标可读性还剩多少。
> 资产：代码 `experiments/C2/`；特征（集群生成，gitignored）`experiments/C2/features/`；训练产物 `results/slurm_c2/c2_v1/`；评估脚本 `experiments/C2/c2_eval.py`（`C:\Users\35150\miniconda3\envs\ai26\python.exe -X utf8 experiments/C2/c2_eval.py --metrics results/slurm_c2/c2_v1/c2_metrics.json` 一键复现）。图在 `experiments/C2/figures/`，指标在 `experiments/C2/c2_analysis.json`。
> 模型：Qwen3-4B-Base rev `906bfd4b`，13 层 [2,5,8,11,14,17,20,23,26,29,32,34,35]，block 输出（final RMSNorm 之前），float16 存储。
> 探针口径：与 `model.py`/`run_grid.evaluate` 完全一致。**std13** = z-score（train 统计量）+ lr 0.002 + 2000 轮 + λ1=0.001、λ2=1.0，与 world 主网格同配置；**raw11** = 原始特征 + lr 5e-4 + λ2=0.1，与 world 历史最优 extra run 同配置。距离 = haversine（R=6371）；MSE=(Δlat²+Δlon²)/2。
> 主口径：**std13-L11**（5 个变体齐全）；raw11 仅作历史配置辅助对拍（`nameonly` 在该配置下发散，见 §0）。配对 Δ = 变体 − 参照（正 = 变体更差），bootstrap 10k 次、seed `20261017`；分层 = 国家是否在 train 中出现过（Q7 口径，seen 110 / unseen 20）。

## 总览

| 图 | 一句话 | 回答的内容 |
|---|---|---|
| `c2_layer_curves` | 5 变体 × 13 层的 MSE / 中位距离曲线（std13） | 可读性随层深如何变化、各变体差距在哪层最大 |
| `c2_variant_bars` | std13-L11 的 MSE、median/P90、距离 CDF | 变体之间的总账（5 个变体齐全） |
| `c2_paired_effects` | 4 组配对效应的森林图（Δ 距离 + 预测漂移） | 切掉/改掉某一因素后，误差与答案各移动多少，CI 是否跨 0 |

一句话主线：**国家名是坐标可读性的主要信息来源，但不是唯一来源**——在同样的句末 "?" 读点上删掉国家名，std13-L11 的 valid MSE 从 539 翻倍到 1069、中位距离 2280→3115 km（配对 +1004.9 km，CI [623.5, 1391.3]）；**仅给"地名, 国家"不给指令，可读性基本保留**（MSE 225 vs 167、中位 1100 vs 909 km，配对仅 +232.6 km，CI [76.8, 385.4]）；**换问法影响很小**（−199.2 km，CI [−398.0, +0.3]）；**读点位置的影响最大**（国家 token vs 句末 "?"：+1511 km，CI [1238.9, 1783.6]）。国家名效应集中在 train 见过的国家（seen +1127 km vs unseen +335 km，后者 CI 跨 0），与 Q7 的"国家先验"解释一致。

---

## 0. 设计与验证（先看这节再读图）

**五个变体**（同一批 500/130 个地点，行序与 `datasets/world` 对齐）：

| 变体 | prompt | 读点 token |
|---|---|---|
| `origcountry` | 原句（world 历史协议） | 国家名最后一个 subtoken |
| `origlast` | 原句 | 句末 "?" |
| `nocountry` | 删去句尾 ", {country}" | 句末 "?" |
| `nameonly` | `{place}, {country}`，无指令 | 国家名最后一个 subtoken（与 origcountry 同一 token） |
| `rephrase` | `Could you tell me the latitude and longitude of {place}, {country}?` | 句末 "?" |

配对设计因此是干净的：`origcountry→nameonly` 只改指令、读点相同；`origlast→nocountry` 只删国家名、读点相同；`origlast→rephrase` 只改问法、读点相同；`origcountry→origlast` 同句、只改读点。630 条中有 10 条（1.6%）是"隐式国家"（城邦/地名本身以国家名结尾，如 "Vatican City"），这些行 `nocountry` 与原句相同、`nameonly` 就是地点短语本身；不影响配对方向，但会轻微稀释"删国家名"的效应。

**锚点与提取自检**（origcountry 应与 `datasets/world` 逐位一致）：
- 13 层中有 10 层与历史特征**逐位相同**（max_abs=0）；L11/L23/L35 有 float16 级数值差（min cosine 0.999897/0.999811/0.999684，max_abs 0.5/1/10），来自 bf16 前向的归约顺序差异，不是管线错配。
- 探针 MSE 对拍：13/14 与 `results/slurm_layers/world/summary.csv` **逐位一致**（差 <1e-4）；仅 L35 差 **+0.7667**（204.1408 vs 204.9075，超 0.5 容差），raw11 差 −0.1018（150.3259 vs 150.4277）。因此集群作业在锚点检查处退出非零，但全部权重/指标已在检查前落盘，结果可用；L35 的偏差按"特征 float16 数值差"记录，不当作管线错误。
- **`nameonly` 的 raw11 探针发散**：lr 从 5e-4 连续减半到 6.25e-5 共 4 次，均在 epoch 9–19 触发 loss 爆炸（初始 loss ~1.0e3/5.4e3，爆炸到 1e9–1e10）。原始特征在该 prompt 下条件数很差，属该配置的数值问题；std13-L11 正常（MSE 224.6）。因此 raw11 只用于辅助对照，主结论一律以 std13 为准。

**数据规模**：train 500、valid 130；70 次训练中 69 次有限。原始/周期 MSE 在本实验中全部相等（探针预测没有跨 ±180° 的经度跳变），故图中只画原始 MSE。

---

## 1. `c2_layer_curves.*` — 5 变体 × 13 层的可读性曲线

**画的是什么**：左图 = valid 原始 MSE vs 解码层（线性纵轴）；右图 = valid 到标签的中位大圆距离 vs 层。5 条线对应 5 个 prompt 变体，虚线标出 L11（头条层）。

**关键读数**（std13，完整表见 `c2_analysis.json`；MSE / 中位 km）：

| 层 | origcountry | origlast | nocountry | nameonly | rephrase |
|---|---|---|---|---|---|
| 2 | 158 / 974 | 421 / 1707 | 2598 / 5138 | 160 / 1061 | 420 / 1722 |
| 5 | 163 / 947 | 418 / 1846 | 1824 / 3749 | 185 / 1129 | 449 / 1799 |
| 11 | 167 / 909 | 539 / 2280 | 1069 / 3115 | 225 / 1100 | 493 / 2163 |
| 20 | 242 / 1167 | 725 / 2602 | 1267 / 2759 | 350 / 1659 | 782 / 2679 |
| 32 | 180 / 1132 | 534 / 1871 | 689 / 1954 | 243 / 1294 | 680 / 2323 |
| 35 | 204 / 1130 | 567 / 1953 | 823 / 2244 | 314 / 1291 | 791 / 2225 |

- **国家 token 曲线最平、最好**（158–242 MSE）；`nameonly` 曲线紧随其后，浅层几乎重合：L2 160 vs 158，L11 225 vs 167，L20 350 vs 242。**指令的收益随层深增大**（L2 +2、L11 +57、L20 +108 MSE），说明浅层表征本身已足够读出坐标，深层表征在缺少问句框架时反而更容易漂移。
- **删国家名（绿线）在最浅层崩得最厉害**：L2 2598 vs `origlast` 421（6.2×）；随层深收窄，L32 689 vs 534（+29%）、L35 823 vs 567。即便最好的 `nocountry` 层（L32, 689）也远差于 `origlast` 的最好层（L5, 418）和 `origcountry`（158）——国家名删掉后，深层的部分补偿不足以补回信息缺口。
- **换问法（紫线）与 `origlast` 基本重合**，仅在 L11（493 vs 539）和 L26/L29（922/986 vs 692/737）有有限差异；方向不稳定，说明问法不是主要因素。
- **各变体的最优层**：origcountry L2（157.6）、origlast L5（417.9）、nocountry L32（689.2）、nameonly L2（159.8）、rephrase L2（419.9）。L11 不是任何变体的最优点，但它是历史主口径/与其他实验对齐的层，故头条固定 L11。

**证明了什么**：坐标信息在浅层就已强烈可读；删国家名造成的损失在浅层最大、深层部分恢复，支持"国家名提供了显式的国家→坐标先验，地名/上下文只能部分替代"的解释。`nameonly` 与 `origcountry` 在 L2 的几乎重合说明：**"有坐标知识"不依赖于指令措辞，指令主要帮助深层读出稳定**。

图注候选（中文）：*"world valid 上 5 种 prompt 变体的坐标线性探针误差随解码层变化（Qwen3-4B-Base，z-score 特征，std13 网格）。国家 token 读点（原句/仅地名）最稳；删去国家名后浅层误差升高约 6 倍、深层部分恢复；换问法曲线与原句句末读点基本重合。"*
图注候选（English）：*"Coordinate-probe error vs decoder layer for five prompt variants on world valid (Qwen3-4B-Base, z-scored features, std13 grid). Reading at the country token (original / name-only) is the most stable; removing the country name inflates shallow-layer error ~6× with partial recovery at depth; rephrasing tracks the original last-token readout."*

---

## 2. `c2_variant_bars.*` — std13-L11 的变体总账

**画的是什么**：三格，均为 std13-L11、world valid n=130。左 = 原始 MSE（对数纵轴，柱顶标数值）；中 = 中位与 P90 大圆距离；右 = 距离经验 CDF（虚线 = 1000 km 参考）。

**关键读数**：

| 变体 | MSE | 中位 km | 均值 km | P90 km | ≤1000 km 占比 |
|---|---|---|---|---|---|
| origcountry | 167.4 | 909 | 1203 | 2471 | **55.4%** |
| origlast | 539.1 | 2280 | 2715 | 5530 | 11.5% |
| nocountry | 1069.4 | 3115 | 3719 | 7544 | 9.2% |
| nameonly | 224.6 | 1100 | 1436 | 3009 | **45.4%** |
| rephrase | 492.6 | 2163 | 2515 | 4540 | 13.8% |

- CDF 的其余分位：origcountry 在 250/500/2000/4000 km 分别为 6.9%/19.2%/85.4%/97.7%；nameonly 3.1%/10.0%/80.0%/97.7%；origlast 0%/3.8%/37.7%/80.8%；nocountry 0.8%/2.3%/30.0%/68.5%；rephrase 0%/1.5%/46.9%/80.8%。
- **参照基线**：用 train 平均坐标（15.67°N, 12.59°E）预测所有 valid，MSE 3311、中位距离 7300 km。即便 `nocountry`（1069 / 3115 km）也比均值基线好 3.1 倍，**"删掉国家名"不等于"没有地理知识"**。
- CDF 横轴截到 8000 km；`nocountry` 有 10/130 条（7.7%）在轴外（最大 16353 km），因此其曲线在右端未到 1.0，读数时不要误当"已收敛"。
- 图 2 与图 1 的口径差异：图 2 固定 L11、完整展示 5 个变体；raw11 的 `nameonly` 缺失（§0），故不放进主图。raw11 的数值（origcountry 150.3 / origlast 1901.9 / nocountry 1883.7 / rephrase 2003.2，中位 944.8/4114.4/4014.0/4428.2 km）只作历史配置辅助对照。

**证明了什么**：L11 上，**"只给地名+国家、不给指令"几乎保留了原句的可读性**（MSE 只高 34%、≤1000 km 占比 45.4% vs 55.4%）；**删掉国家名则使误差翻倍、≤1000 km 占比降到 9.2%**。换问法的结果介于两者之间、且优于原句句末读点，说明句末 "?" 读点本身不是最优，问法内容不是主导因素。

图注候选（中文）：*"std13-L11 线性探针在 world valid（n=130）上的变体对比：原始 MSE、median/P90 距离与距离 CDF。仅给地名不给指令基本保留可读性（MSE 225 vs 167）；删去国家名使 MSE 翻倍（1069）且 ≤1000 km 占比从 55% 降到 9%。"*
图注候选（English）：*"Variant comparison with the std13-L11 linear probe on world valid (n=130): raw MSE, median/P90 distance, and distance CDF. Dropping the instruction while keeping the place name largely preserves decodability (MSE 225 vs 167); removing the country name doubles MSE (1069) and cuts the ≤1000 km fraction from 55% to 9%."*

---

## 3. `c2_paired_effects.*` — 配对效应（误差变化 + 答案漂移）

**画的是什么**：左 = 逐条距离差 Δ = 变体 − 参照的均值（正 = 变体更差，红点 + bootstrap 95% CI）；右 = 两个 prompt 条件下预测点之间的大圆距离（"prompt 把答案移动了多少"，蓝点 + CI）。纵轴按 std13 四组、raw11 三组排列（raw11 无 `nameonly`）。

**关键读数**（std13 在前，n=130）：

| 配对 | Δ 均值 km | CI95 | 变体更近 | 预测漂移 km | CI95 |
|---|---|---|---|---|---|
| instruction removed（origcountry→nameonly） | **+232.6** | [+76.8, +385.4] | 48/130 | 905.2 | [773.9, 1055.9] |
| country name removed（origlast→nocountry） | **+1004.9** | [+623.5, +1391.3] | 42/130 | 2915.6 | [2591.8, 3249.8] |
| question rephrased（origlast→rephrase） | **−199.2** | [−398.0, +0.3] | 76/130 | 1319.8 | [1173.5, 1472.7] |
| token position（origcountry→origlast） | **+1511.4** | [+1238.9, +1783.6] | 20/130 | 2531.3 | [2235.2, 2840.6] |

raw11 辅助对照：country removed **−149.0** [−420.7, +125.4]（n.s.）；question rephrased **+141.7** [−13.4, +295.5]（n.s.）；token position **+4094.3** [+3515.4, +4699.8]。

**分层（std13，按国家是否在 train 出现）**：

| 配对 | seen（n=110） | unseen（n=20） |
|---|---|---|
| instruction removed | +297 [+151, +448] | −122 [−716, +383] |
| country name removed | **+1127 [+725, +1555]** | +335 [−640, +1243] |
| question rephrased | −233 [−450, −1] | −12 [−381, +376] |
| token position | +1577 [+1275, +1881] | +1152 [+608, +1788] |

**怎么读**：
1. **删国家名的效应真实且大**：+1005 km、CI 不跨 0；同时预测点平均移动 2916 km——不是"标签附近抖动"，而是答案被整体搬走。效应集中在 seen 国家（+1127 km，CI 不跨 0）；unseen 国家仅 +335 km 且 CI 跨 0（n=20，CI 很宽）。这与 Q7 的"国家先验"解释一致：probe 在训练中见过该国家时，国家名能直接激活一个坐标先验；没见过时，国家名的可恢复信息少得多（但地名本身仍有信息，见 §2 的基线对比）。
2. **指令的作用小而显著**：+233 km（CI 不跨 0），预测漂移 905 km；同样集中在 seen 国家（+297），unseen 为 −122（CI 跨 0）。仅给"地名, 国家"就能把 ≤1000 km 占比保持在 45%，说明指令不是坐标信息的主要载体。
3. **换问法几乎没有系统影响**：−199 km，CI 上界 +0.3，刚碰到 0；seen 国家 −233（CI [−450,−1]），unseen −12。方向是"换问法略好"，但幅度远小于删国家名；不应据此宣称某种问法更优（单一模板、单一模型）。
4. **读点位置是最大的方法学因素**：国家 token 比句末 "?" 好 +1511 km（std13）/ +4094 km（raw11），预测漂移 2531/5054 km。world 历史协议选国家 token 是合理的；任何跨变体比较都必须固定读点，否则会被读点差异淹没。
5. **raw11 为什么和 std13 方向不一致（国家名删除 n.s.）**：raw11 的 "?" 读点本身已接近不可读（origlast/nocountry MSE ~1900、中位 ~4000 km，接近均值基线的 3311/7300），存在地板效应；std13 的标准化把同一读点的可用信号释放出来，才看见国家名效应。raw11 只适合与历史 150.43 对拍，不适合做变体间的精细比较。

**证明了什么**：C2 直接回答了 Q7 的反向问题——**国家名确实是一个主要的信息来源（删掉后误差翻倍、答案整体漂移），但模型/表征并不只靠它**：仅地名、无指令仍能把中位误差保持在 1100 km（均值基线 7300 km），且浅层几乎不受影响。国家名效应在见过的国家上显著、在未见国家上不显著，与"国家→坐标先验"的机制一致。

图注候选（中文）：*"prompt 变体的配对效应（world valid n=130，bootstrap 95% CI）。左：到标签的平均距离变化（正 = 变体更差）；右：两种 prompt 下预测点之间的平均大圆距离。删国家名 +1005 km（CI [624, 1391]）且答案平均移动 2916 km；去指令仅 +233 km；换问法 −199 km（CI 上界 +0.3）；读点从国家 token 换到句末 "?" +1511 km。"*
图注候选（English）：*"Paired prompt-variant effects on world valid (n=130; bootstrap 95% CI). Left: mean change in distance to truth (positive = variant worse); right: mean great-circle distance between predictions under the two prompts. Removing the country name costs +1005 km (CI [624, 1391]) and moves answers by 2916 km; removing the instruction costs only +233 km; rephrasing −199 km (CI upper bound +0.3); switching the readout from the country token to the final '?' costs +1511 km."*

---

## 4. 结论（可直接进报告 C2 小节）

1. **国家名是主要信息来源，但不是唯一来源**：同一读点删掉国家名，std13-L11 MSE 539→1069、中位 2280→3115 km（+1005 km，CI [624,1391]）；但无国家名仍比均值基线好 3.1 倍，地名与上下文保留了相当信息。
2. **指令的作用是"锦上添花"**：仅给"地名, 国家"时 MSE 225（原句 167）、中位 1100 km（原句 909），≤1000 km 占比 45.4% vs 55.4%；浅层（L2）几乎无差，深层差距才显现。
3. **换问法影响很小**（−199 km，CI 刚碰 0），不是主要混杂因素；**读点位置影响最大**（国家 token vs 句末 "?"：+1511 km），跨变体比较必须固定读点。
4. **与 Q7 闭环**：国家名效应集中在 train 见过的国家（+1127 km）而非未见国家（+335 km，CI 跨 0），支持"国家→坐标先验"机制；未见国家仍有地名信息可读。

## 5. 局限与报告注意事项

- 单一模型（Qwen3-4B-Base rev 906bfd4b）、每个变体单一模板；结论不外推到其他模型/模板。
- valid 只有 130 条，unseen 国家仅 20 条，分层 CI 很宽；所有配对 CI 未做多重比较校正。
- `country` 列存在少量不规范标签（如 "Yamoussoukro, Côte d'Ivoire"、"Guinea / Côte d'Ivoire / Liberia"），seen/unseen 按字符串精确匹配划分；另有 10 条隐式国家行，`nocountry` 与原句相同。
- `nameonly` 的 raw11 发散（4 次降 lr 仍爆炸），raw11 的变体比较不完整；主结论只依赖 std13。
- L35 锚点差 +0.77、raw11 锚点差 −0.10：特征是 float16、集群 bf16 前向与历史提取存在数值差（cos ≥0.99968），不是管线错配；报告引用历史 150.43 时应注明复现值为 150.33。
- 本实验测的是**线性探针的可读性**，不是模型生成行为；"国家名被使用"指表征中可线性读出的信息，不等同于因果机制。
