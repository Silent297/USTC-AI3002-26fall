# A2 逐图解释说明（sklearn 模型家族对照）

> 实验：`experiments/A2/` —— 拓展实验 A 系列之一（附加探究，非课程必交内容）。固定**单层**特征（z-score，train 统计），在同一协议下比较 8 类 sklearn 模型家族：线性（OLS / Ridge）、KNN、核方法（SVR-RBF / KernelRidge-RBF / SVC-RBF）、树集成（RandomForest / ExtraTrees / HistGradientBoosting）。超参由 **train 内部 RepeatedKFold(5×2；truth 为 5×1) CV** 选择后全量重训（valid 只用于报告）。
> 协议：world L11（提交层）/ L2（std 单层最优），truth L20（峰值层）。SVR 目标按输出 z-score（train 统计）再逆变换（SVR 尺度敏感）；网格按本地基准耗时裁剪（truth：SVC 49.3 s/次、LinearSVC 42.1 s/次、HistGB 50 s/次 → 少配置）。
> 锚点（同图为参照线）：A1 协议内线性 ridge 167.17（L11）/ 157.62（L2）、truth LR 0.8667；历史：raw L11 提交 150.43、GD raw L20 0.8967；参考报告（非官方，valid 选择协议）：KNN 125.27 / MLP-128-64 152.70 / RBF-SVC 0.87。
> 读数：`a2_run.log`、`a2_results.json`（每模型 cv / train / valid / 耗时 / 选中超参）；`python experiments/A2/a2_sklearn_family.py`（5175 s，2026-10-16）。

## 1. `a2_model_bars.*` — 三个面板的模型家族条形

**画的是什么**：左 world L11、中 world L2、右 truth L20；横条 = 各模型 valid 指标（条形右侧标注选中超参），虚线 = A1 线性锚、点红线 = 历史锚、绿点划线 = 参考报告数字（*，非官方）。

**关键读数**（valid）：
| 面板 | 排名（好→差） |
|---|---|
| world L11 | **KNN k=3 140.70** ｜ Ridge α=500 167.17 ｜ KernelRidge 253.67 ｜ ExtraTrees 286.44 ｜ SVR 296.17 ｜ OLS 311.56 ｜ HistGB 332.19 ｜ RF 388.84 |
| world L2 | **Ridge 157.62** ｜ KernelRidge 168.23 ｜ ExtraTrees 194.97 ｜ SVR 206.92 ｜ KNN k=5 211.79 ｜ HistGB 232.36 ｜ RF 273.14 ｜ OLS 718.26 |
| truth L20 | **SVC-RBF 0.8767** ｜ LinearSVC 0.8633 ｜ LogisticRegression 0.8600 ｜ HistGB 0.8600 ｜ RF / ET 0.8500 ｜ KNN 0.7767 |

（Ridge α=500 即 λ2=1.0×N，与 A1 主口径同解；LR 选中 C=1、SVC 选中 C=10/γ=3/D。）

**证明了什么**：
1. **world L11 上唯一显著超过闭式 ridge 的家族是 KNN**（140.70，−26.5 MSE），与参考报告 KNN（125.27*）方向一致（协议/网格不同，量级同带）；但 KNN 优势**依赖层**：L2 上掉到 211.79 → 邻域度量在中间层最对齐地理结构。
2. **树/提升在高维稠密 z 特征上全面偏弱**（RF 273–389、HistGB 232–332）：轴平行分裂对 2560 维稠密特征几乎无效。
3. **核方法（SVR/KR）不敌闭式 ridge**：同为 RBF 类，KernelRidge 253.67/168.23 明显差于 Ridge 167.17/157.62——gap 主要来自优化/尺度口径（SVR 另带 ε 管），不是表达能力。
4. **truth 全家族挤在 0.85–0.877**：最强 SVC-RBF 0.8767 ≈ 参考报告 0.87（互相印证）；与线性锚 0.8667 差 ≤1pp，KNN 掉队（0.7767）——分类任务上"换模型家族"不是关键变量。
5. 与 A1 联读：**world 全局最优仍是 A1 的拼接线性 128.82**——单层特征上赢不过 KNN 的线性读出，换多层表征后反超；"线性 vs 非线性"不是主要矛盾。

**报告怎么用**：回答"线性够不够"的模型侧证据：① KNN(L11) 例外且与参考报告一致；② 树全面弱；③ truth 全家族一档。引用时务必注明 * 为非官方协议数字。
图注候选（中文）：*"同一 z-score 单层特征上 8 类 sklearn 家族的验证指标（超参由 train 内部 CV 选择）。world L11 上 KNN(k=3)=140.7 超过闭式 ridge 167.2，但换到 L2 时 KNN 反被 ridge 反超（211.8 vs 157.6）；truth 上全家族挤在 0.85–0.877（SVC-RBF 最高 0.8767）；树集成在两任务上均最弱。虚线为 A1 协议内线性锚，红线为历史锚点，* 为参考报告数字。"*
图注候选（English）：*"Validation metrics of 8 sklearn families on the same z-scored single layer (hyperparameters via train-internal CV). KNN(k=3)=140.7 beats closed-form ridge 167.2 on world L11 but loses on L2 (211.8 vs 157.6); on truth all families collapse into 0.85–0.877 (SVC-RBF best, 0.8767); tree ensembles are weakest throughout."*

**注意与留档**：首跑因 `HistGradientBoosting` 不支持多输出回归（world y 为 (n,2)）中断、修复（`MultiOutputRegressor` 包一层）后完整复跑；崩溃日志留档 `a2_firstrun_crashed.log`。

## 附：A2 数字速查

- world L11 家族排名见上表；world L2 Ridge 157.62 = A1 单层锚（口径闭环）。
- truth：SVC-RBF 0.8767（≈guide* 0.87）、LinearSVC 0.8633、LR/HistGB 0.86、RF/ET 0.85、KNN 0.7767。
- 图：`figures/a2_model_bars.{png,pdf}`（dpi=150 + PDF）。
