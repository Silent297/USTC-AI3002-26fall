# C1 逐图解释说明

> 实验：Qwen3-4B-Base 直接输出经纬度（world 630 题），对照我们的 L11 线性探针。**附加实验（非课程提交内容）**；题目来源 `guide/计划：C1 方案定稿…md`。
> 运行资产：`results/slurm_qwen_direct/qwen_direct_v1/`（集群 A100-80GB，Qwen3-4B-Base rev `906bfd4b`（本地克隆，`run_meta.revision_note`），贪心解码、max_new_tokens=64、bf16；run_meta.json 记录 prompt 模板 / 生成参数 / csv md5 / 环境）。
> 评估脚本：`experiments/C1/c1_eval.py`（`C:\Users\35150\miniconda3\envs\ai26\python.exe -X utf8 experiments/C1/c1_eval.py` 一键复现；含探针锚点自检，不符即中止）。图在 `experiments/C1/figures/`，指标在 `results/slurm_qwen_direct/qwen_direct_v1/c1_metrics.json`。
> 参考线口径：`guide/2026AIML_LAB1_指南3.pdf` §4 同学参考报告——**非官方**，只做对照（字典 `REFERENCE`，图中黑菱形）。
> 探针锚点（每次评估前置自检，本地已通过）：world raw L11，valid MSE **150.4277** / 中位 **953.8 km**、train MSE **116.9279** / 中位 **662.4 km**，两 split 均为 |Δlon|>180 计 0 条。
> 口径：有效回答 = tier ∈ {json, embedded, regex} 且 in_range（无效不进 MSE 分母）；MSE = 有效条数上 (Δlat²+Δlon²)/2 均值；周期经度 MSE 为经度差回绕后的附加口径；距离 = haversine 大圆（R=6371）；hit100 分母 = 全部输入。地图合规：仅 110m 海岸线，无国界线。
> 维护约定：每完成一批绘图，在 `figures/` 下新建 `<题号>_figure_notes.md`，格式同本文与 `b1/q2/q3/q6/q7/q8_figure_notes.md`。

## 总览

| 图 | 一句话 | 回答的内容 |
|---|---|---|
| `c1_distance_cdf` | 大圆距离 ECDF（4 条线：直接/探针 × train/valid，对数横轴） | "直接输出比探针好多少" |
| `c1_valid_map` | valid 130 条全球散点：空心=标签、实心=预测（颜色=距离），top-5 误差标注 | "错在哪、错多远" |
| `c1_metric_bars` | 三格：解析分层 / MSE（原始+周期，叠加参考线与探针星标）/ 距离中位+P90 | "总账 + 与参考报告对拍" |

一句话主线：**Qwen3-4B-Base 零样本直接输出坐标，全面优于我们的线性探针**——valid 距离中位 **43.8 km vs 953.8 km**（约 22×），逐条配对 119/130 直接更近（均值差 +982.5 km，CI95 [835.0, 1138.9]）；解析稳健（严格 99.8%/99.2%）；**原始 MSE 不是公平指标**：valid 原始 MSE 505.76 的 **96.5% 来自单条 Mount Tomanivi（斐济，跨±180°日界线）**，周期口径下 17.64；train 原始 MSE 159.64 的 ~92% 来自 5 条极地/海外属地极端样本。总体读数与同学参考报告几乎逐项一致（valid hit100 两队完全相同 63.85%；Tomanivi 预测值逐位相同：lon −178.25、Δlon_raw −356.265）——同一模型 revision + 贪心解码的可复现性佐证。

---

## 1. `c1_distance_cdf.*` — 距离 ECDF（直接 vs 探针）

**画的是什么**：横轴 = 到标签的大圆距离（对数，1–30000 km），纵轴 = 经验 CDF；4 条线：Qwen 直接 valid（红实线，n=130，中位 43.8 km）、直接 train（粉实线，n=499，中位 15.5 km）、探针 valid（蓝虚线，n=130，中位 953.8 km）、探针 train（浅蓝虚线，n=500，中位 662.4 km）。

**关键读数**（全部来自 c1_metrics.json）：
- **直接输出曲线全局左移**：每条百分位上直接都优于探针。中位：valid **43.8 vs 953.8 km**（22×）、train **15.5 vs 662.4 km**（43×）。均值：valid 159.1 vs 1141.7 km；P90：valid 416.0 vs 2391.0 km。
- **逐条配对（valid 有效子集 n=130）**：直接更近 **119/130**（探针仅 11 条更近）；距离差（探针−直接）均值 **+982.5 km，CI95 [835.0, 1138.9]**（10k 重抽，不跨 0）。
- 直观读图：直接 valid 曲线在 ~500 km 处已达 CDF≈0.95，而探针 valid 要到 ~2400 km 才到同一水平——探针水平是"大洲级"，直接输出是"城市级"。
- train 与 valid 的差距：直接输出 train 曲线略左于 valid（15.5 vs 43.8 km 中位），差距远小于探针的 train/valid 落差（662 vs 954 km），说明直接输出的泛化对 world 的 train/valid 划分不那么敏感。

**证明了什么**：
1. "**线性读出误差 ≠ 模型没有地理知识**"（C1 计划 §2 原话）：同一批 2560 维表征，换个读出方式（生成式直接输出）误差降一个量级多——探针弱是**读出侧瓶颈**，不是模型侧瓶颈。与 Q6（经度弱）、Q7（超越国家身份）形成闭环。
2. 统计显著性：逐条配对差 CI 不跨 0，且方向压倒性（119:11），不是个别样本拉出来的。
3. 注意声明（与参考报告 §4.4 一致）：生成式直接输出 ≠ 线性头，两者口径不同，只能**并列报告**，不能据此宣称"最优线性模型不如 Qwen"。

**报告怎么用**：C1 的主图（附加实验，报告可选附录/延伸讨论）。若引用：先给 ECDF 主图，再给配对统计（119/130、+982.5 km CI）作显著性与个位数检验。

图注候选（中文）：*"到标签的大圆距离经验 CDF（world 630 题；Qwen3-4B-Base 零样本直接输出 vs 我们的 L11 线性探针）。直接输出全部百分位占优：valid 中位 43.8 km vs 953.8 km（约 22 倍）；逐条配对 119/130 直接更近，均值差 +982.5 km（bootstrap 95% CI [835.0, 1138.9]）。"*
图注候选（English）：*"Empirical CDF of great-circle distance to ground truth (world 630 items; Qwen3-4B-Base zero-shot direct output vs our L11 linear probe). Direct output dominates at every percentile: valid median 43.8 km vs 953.8 km (~22×); paired per-item comparison favours direct output in 119/130, mean gap +982.5 km (bootstrap 95% CI [835.0, 1138.9])."*

---

## 2. `c1_valid_map.*` — valid 全球散点与 top-5 误差

**画的是什么**：PlateCarree 全球图（仅 110m 海岸线）。空心灰圈 = 标签位置；实心点 = Qwen 直接预测在**标签位置**着色（viridis，LogNorm 1–1000 km，距离越近越深）；5 个最大平方误差样本带白底标注框（实体名 + 距离）。

**关键读数**：
- 130/130 valid 全部有效并上图；绝大多数点深色（< 100 km），黄点（> 300 km）稀疏。
- **top-5（valid 的 96.5%+2.5%+0.4%+0.1%+0.05% 平方误差）**：
  - **Mount Tomanivi, Fiji**（row 73）：真值 (178.015, −17.611)，预测 (−178.250, −17.917)——**跨日界线的符号错误**，原始 Δlon = −356.265°、距离仅 **397.0 km**；单条占 valid 原始平方误差 **96.52%**（我队）——与参考报告的 96.55% 一致，"Tomanivi 现象"复现（评估自检 `checks.valid_top1_tomanivi_pattern = true`，预测值与参考报告逐位相同：lon −178.25）。
  - Lake Vostok, Antarctica：1287 km（预测纬度对、经度偏 57.5°）。
  - Saint Louis Cathedral, USA：2247 km（预测到了纽约附近 45.5N/−73.6——**同城同名教堂先验**，把新奥尔良的教堂放去了纽约）。
  - Timbuktu, Mali：1334 km（经度偏 −12.5°，落到了毛里塔尼亚/塞内加尔方向）。
  - Charyn Canyon, Kazakhstan：691 km（纬度偏 +4.2°，类似"国家中心"吸附）。
- 地图上也能直读出：欧洲/东亚/北美东部点密集且深色（模型知识强区）；非洲内陆、南极、太平洋岛屿偏黄。

**证明了什么**：
1. **误差结构是"少数极端 + 绝大多数很近"**，不是均匀偏大：中位 43.8 km 但均值 159.1 km（均值被日界线和南极拉高）——这正是为什么下面 §3 说 MSE 必须配周期口径一起看。
2. Tomanivi 一条 = 96.5% 原始 MSE——**单条日界线样本就能主导 MSE 指标**（周期经度 MSE 下 Fiji 只有 3.7° 误差）。这为"MSE 是易碎指标、要报告距离分布 + 周期口径"提供了活教材（Q6/Q7 已有同款讨论）。
3. Saint Louis Cathedral（新奥尔良→纽约）是**同名校先验**的典型案例；Charyn Canyon 偏到"哈萨克中心"附近是"国家中心吸附"。二者都是"知识不足时回退到常见先验"的错误模式，可作为定性例子。

**报告怎么用**：C1 的定性主图；Tomanivi 的 96.5% 数字是"为什么 MSE 会骗人"的旗舰论据（若报告有"指标选择"讨论段落，直接引用）。地图合规已核（仅 110m 海岸线，无国界）。

图注候选（中文）：*"valid 130 条 Qwen 直接输出的预测位置。空心圈 = 标签，实心点 = 预测（按大圆距离对数着色），标注 5 个最大误差样本。原始 MSE 的 96.5% 来自单条 Mount Tomanivi（斐济，跨±180° 日界线，实际仅偏 397 km）。"*
图注候选（English）：*"Predicted locations for the 130 world-valid items (Qwen3-4B-Base direct output). Hollow circles = ground truth; dots = predictions coloured by great-circle distance (log scale); the five largest errors are annotated. A single dateline-wrapped sample (Mount Tomanivi, Fiji) accounts for 96.5% of the raw MSE despite being only 397 km off."*

---

## 3. `c1_metric_bars.*` — 三格总账

**画的是什么**：左 = 解析分层（每 split 100% 堆叠条：json / embedded / regex / out_of_range / failed）；中 = 原始 MSE vs 周期 MSE 条形（对数纵轴；黑菱形 = 同学参考报告（非官方）；红星 = 我们探针锚点）；右 = 距离中位 / P90 条形（同参考线与探针星标）。

**关键读数**：

解析（左格）：
- train：json **499/500 = 99.8%**（1 条 out_of_range）；valid：json 129 + regex 1 = **130/130 = 100% 宽松**（严格 99.2%）。
- 参考报告严格率 99.00%/98.46%——我们略高（细节：拒绝 1 条 out_of_range 后 train 严格 99.8%；参考的 495/500 与 128/130 可能还剔了别的解析边缘样本）。
- 两条非 json 边角案例（值得写进细节段）：train row 217（Phnom Penh）模型输出 `{"latitude": 115.2364, "longitude": 11.5833}`——**纬度位写了 115.24（越界）且疑似经纬互换**，被 out_of_range 拒；valid row 40（Yaoundé）输出 `3°20'30"N` **度分秒格式**，regex 层只取首数（3.0/11.0）——丢了分秒（该条误差 ≈ 1° 量级）。⇒ "解析层"本身值得报告（宽松 100% 但严格 99.2%），DMS 输出是唯一收紧点。

MSE（中格，对数轴）：
- train：原始 **159.64**（参考 161.07）≈ 周期 **108.54**（参考 109.56）——两口径都贴近参考线。
- valid：原始 **505.76**（参考 513.51，差 −1.5%），周期 **17.64**（参考 17.77，差 −0.7%）——**原始口径的差异几乎全来自 Tomanivi 那条**（我们与参考都各含它）。
- 探针星标对照（红）：探针 train MSE 116.93、valid 150.43——**原始 MSE 下探针"看起来更好"**（valid 150 < 506）！这正是中格要传达的反差：MSE 是平方、被少量极端样本主导（探针恰好没有日界线缠绕样本——其锚点 |Δlon|>180 为 0 条）；距离分布（右格）给出相反且更符合实际的结论。
- combined：原始 231.18 / 周期 89.75（参考 233.48 / 90.70），有效 629/630。

距离（右格，对数轴）：
- valid 中位 **43.8**（参考 35.8）/ P90 **416.0**（参考 439.1）；train 中位 **15.5**（参考 15.2）/ P90 **342.0**（参考 332.0）；combined 中位 **20.9**（参考 19.8）。
- 探针星标（红）远高于直接条：valid 中位 953.8、train 662.4——右格与中格并看 = "**MSE 谁大谁小取决于一条样本，距离中位差 20+ 倍不取决于任何一条**"。

**证明了什么**：
1. 总体读数与同学参考报告**逐项一致到"几乎相同"**（valid hit100 一模一样 63.85%；均值 159.1 vs 159.0；Tomanivi 预测逐位相同）——我们的重实现是可靠的；小差异（valid 中位 43.8 vs 35.8）来自解析接受度的差别（我们宽松口径多收 2 条中距离样本）。
2. "选哪个指标决定讲什么故事"：**原始 MSE 下探针(150) 好于直接(506)，周期 MSE / 距离下直接压倒性好**（Med 43.8 vs 953.8）。报告结论必须捆口径报告：原始 MSE + 周期 MSE + 距离中位/P90 三件套。
3. 解析层稳健（宽松 100%、严格 99.2%），"追加 JSON 指令"的 prompt 设计有效；两条边角案例（经纬互换、DMS 格式）说明模型输出格式仍非 100% 稳定。

**报告怎么用**：C1 的定量总账图，也是"指标选择讨论"（若有）的最佳论据。可只引中格+右格的并列反差，一句话："同一批预测，原始 MSE 说探针更好，距离分布说直接输出好 20 倍——因为 MSE 的 96.5% 来自一条日界线样本。"

图注候选（中文）：*"C1 指标总账。(a) 解析分层：train 99.8% 严格、valid 100% 宽松可解析；(b) 原始/周期经度 MSE 与参考报告（黑菱形、非官方）及我方探针（红星）：valid 原始 MSE 505.8 的 96.5% 来自单条日界线样本，周期口径 17.6；(c) 距离中位/P90：直接输出 valid 中位 43.8 km，探针 953.8 km。"*
图注候选（English）：*"C1 scoreboard. (a) parse tiers: 99.8% strict on train, 100% loose on valid; (b) raw vs periodic-lon MSE with the classmate report (black diamonds, unofficial) and our probe (red stars): 96.5% of the raw valid MSE 505.8 comes from one dateline sample; periodic-lon 17.6; (c) median/P90 distance: direct output 43.8 km vs probe 953.8 km (valid median)."*

---

## 4. 对拍与注意事项

- **与参考报告一致性**：valid hit100 完全一致（0.6385）；combined 均值 192.7 vs 193.1；Tomanivi 预测值逐位相同（lon −178.25、Δlon_raw −356.265）⇒ 同模型 revision + 贪心解码下结果可复现。差异点：我们宽松口径 valid 收满 130 条（参考 128），train 收 499（参考 495），故中位/P90 略有差异（43.8 vs 35.8 / 416.0 vs 439.1），**属于解析接受度差异，不是模型行为差异**。
- **探针锚点自检**：每次运行先复算 world raw L11 的 MSE/中位（±1e-3 / ±0.5 km），本轮逐位通过（train 116.9279/662.4、valid 150.4277/953.8），保证"探针对照线"和 Q6/Q7 报告口径同源。
- **MSE 易碎性**：raw valid MSE 里 Tomanivi 占 96.52%（参考 96.55%）；train raw MSE 亦被 5 条主导（Antarctica 4 条 + Mount Otemanu/法属波利尼西亚 1 条：Vinson Massif Δlon 249°、Concordia −181.7°、Otemanu 预测到法国本土 15529 km）。**只报原始 MSE 会得出与常识相反的结论**。
- **train 侧 top-5 的定性看点**：Antarctica 站名（Vinson/Concordia/Rothera/Erebus）是"知识稀疏区系统性偏移"；Mount Otemanu（Bora Bora，country=France）被定位到**法国本土**——"国家标签 → 本土坐标"的语义先验压过具体记忆（与 world 数据集的 country 字段设计呼应：country=France 而地点在法属波利尼西亚，这类"属地陷阱"值得在报告定性段一提）。
- **regex 层语义**：度分秒输出只取首数（不换算分秒），这是有意的宽松接受口径；该行为 valid 唯一 regex 条（Yaoundé，误差 ~100 km 级）。若做敏感性分析：严格口径（仅 json）valid n=129、train n=499 时读数变化在 CI 噪声内。
- **C1 是附加实验**：不在课程提交要求内（正式提交的对照物是课程给定 probe 口径），报告如需引用应注明"附加实验（非提交要求）"，并保留参考报告"非官方"标注。
- **复现**：`C:\Users\35150\miniconda3\envs\ai26\python.exe -X utf8 experiments/C1/c1_eval.py`（本地 CPU，约 20 s；含锚点自检→指标→3 图）。输入 = `results/slurm_qwen_direct/qwen_direct_v1/{train,valid}.jsonl + run_meta.json`（集群产物，已随 git 取回）。
