# A3 逐图解释说明（自训 MLP 对照）

> 实验：`experiments/A3/`（torch CPU）—— 拓展实验 A 系列之一（附加探究，非课程必交内容）。在同一 z-score 单层特征上自训三个架构：**linear 对照（同配方）**、MLP-64、MLP-128-64（ReLU），比较"非线性 + 现代训练配方"能否超过闭式线性探针。
> 协议：AdamW(lr=1e-3, wd=1e-4)、batch 128(world)/256(truth)、patience 30、max 400/200 epoch；**早停用 train 内部 10% 切分（不看 valid）**，恢复最优内部 epoch 权重后再报 valid；3 seeds（0/1/2）。world L11/L2，truth L20。
> 锚点：闭式 ridge z L11 167.17 / L2 157.62；truth 线性 LR 0.8667、GD raw 0.8967；参考报告（非官方）：MLP-128-64 152.70。
> 读数：`a3_run.log`、`a3_results.json`（逐 seed best_epoch / epochs_run / train / valid / 耗时）；`python experiments/A3/a3_mlp_torch.py`（500 s，2026-10-16）。

## 总览

| 图 | 一句话 | 回答的内容 |
|---|---|---|
| `a3_model_bars` | 三面板 mean±std 条形（3 seeds），vs 锚点线 | "自训网络 vs 闭式线性，谁赢" |
| `a3_curves` | 内部 val 损失曲线（3 seeds × 3 架构，log 轴）+ 早停点（×） | "训练动力学：为什么 linear 对照差、truth 为什么早早停" |

**关键读数**（valid，mean ± std / 3 seeds；括号 = 最好单 seed）：
| 面板 | linear（同配方对照） | MLP-64 | MLP-128-64 | 闭式锚点 |
|---|---|---|---|---|
| world L11 | 475.97 ± 17.04 | 189.86 ± 12.17 | **175.94 ± 17.09**（159.43） | **167.17**（Ridge） |
| world L2 | 504.14 ± 21.15 | 205.88 ± 24.92 | **193.59 ± 42.57**（144.25） | **157.62**（Ridge） |
| truth L20 | 0.8844 ± 0.0175 | **0.8900 ± 0.0109** | 0.8822 ± 0.0110 | 0.8667（LR）/ 0.8967（GD raw） |

参数规模（world）：linear 5,122 ／ MLP-64 164,034 ／ MLP-128-64 336,194（vs 训练样本 500——强过参数化，早停是主要正则）。

**证明了什么**：
1. **没有任何自训网络在均值意义上超过闭式线性探针**（world 最好架构 mean 175.94 vs Ridge 167.17；L2 193.59 vs 157.62）——"线性读出头足够"的正面证据：500 样本 / 2560 维下，额外非线性容量换来的是方差而非精度。个体 seed 可达 159.43（L11）/ 144.25（L2），低于 Ridge，但跨 seed std 12–43 说明不稳定，报告中只应引用均值±std。
2. **同配方内 MLP ≫ linear 对照**（175.9 vs 476）——但对照的差是**优化/预算问题**而非模型问题：其 train MSE≈486（几乎没拟合上），曲线 150 轮仍在缓降被 patience 截断；闭式 ridge 167 的存在说明"同一模型换个求解/正则口径立即最优"。**训练配方与正则口径的影响力 ≫ 非线性本身**（与 Q3 结论互通、也可与 A2 核方法不如 ridge 互引）。
3. **truth 三架构挤在 0.88–0.89（±0.01–0.02）**：与 A1 拼接线性 0.8833、A2 SVC-RBF 0.8767 同一水平带，非线性无决定性增益；均低于历史 GD raw 0.8967（协议不同，仅作参照）。
4. 与 A1 联读：world 上"自训 MLP 的均值（175.9）"也够不到"拼接线性（128.8）"——**表征组合的收益大于模型家族/容量的收益**。

**曲线（`a3_curves.*`）关键读数**：world 面板 linear 停在高位平台（≈450–500，欠拟合证据）、MLP 收敛到 160–390 后走平；truth 面板三架构都在 epoch **2–5** 触底（内部 logloss ≈0.29）后回升——信号"一学就饱和"，early stop 点极早（best_ep 2–5），这也是 truth 上三个架构成绩接近的机制原因。

**报告怎么用**：A3 是"线性上限"论证的**双面证据**——(a) 正面：现代配方自训网络在均值上都不能超过闭式线性探针；(b) caution：朴素 AdamW 训练的线性对照差一个数量级，说明结论对训练/正则口径高度敏感（呼应 Q3）。
图注候选（中文）：*"自训网络 vs 闭式线性探针（同一 z-score 特征、3 seeds、train 内部早停）。world 上 MLP-128-64 均值 175.9±17.1，仍高于闭式 ridge 167.2；同配方的 linear-AdamW 对照 476.0（欠拟合）；truth 上三架构挤在 0.88–0.89。右图：内部验证损失曲线——truth 全部在 epoch 2–5 触底（x = 早停点）。"*
图注候选（English）：*"Self-trained nets vs closed-form linear probes (same z-scored features, 3 seeds, train-internal early stopping). On world, MLP-128-64 averages 175.9±17.1, still above ridge 167.2; the same-recipe linear AdamW control reaches only 476.0 (underfit). On truth all three architectures sit at 0.88–0.89. Right: internal-val curves — truth bottoms out at epoch 2–5 (x marks the early-stop point)."*

## 附：跨实验统一排行榜（A1–A3 汇总，报告可直接引用）

**world（valid MSE，越低越好）**
| 名次 | 数值 | 来源（协议：z-score + 固定正则） |
|---|---|---|
| 1 | **128.82** | A1 拼接线性 **L5+L32** |
| 2 | 140.17 | A1 三元组 L2+L11+L35 |
| 3 | 140.70 | A2 KNN k=3（L11） |
| 4 | 150.43 | 历史提交探针（raw L11，协议不同） |
| 5 | 157.62 | A1/A2 单层 Ridge（L2） |
| 6 | 175.94±17.09 | A3 MLP-128-64（L11）均值 |
| — | 125.27* / 152.70* | 参考报告 KNN / MLP-128-64（非官方，valid 选择协议） |

**truth（valid acc，越高越好）**：0.8967 历史 GD raw L20（raw 协议）｜ 0.8900 A3 MLP-64 ／ 0.8844 A3 linear-AdamW ｜ 0.8833 A1 拼接 L17+L20 ｜ 0.8767 A2 SVC-RBF（≈0.87* 参考报告）｜ 0.8667 单层线性锚。

**一句话**：两个任务上"**换表征（层/拼接）+ 对齐正则口径**"的收益 ≥ "换更复杂模型族"——线性读出（尤其多层拼接）已足够强。

图：`figures/a3_{model_bars,curves}.{png,pdf}`（dpi=150 + PDF）。
