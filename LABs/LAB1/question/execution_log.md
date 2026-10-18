# LAB1 思考题执行日志（Question Execution Log）

> 目录约定：`data/` 数据集 · `code/` 脚本 · `figures/` 图表
> 本文件记录 9 道思考题的数据、代码、实验过程与结论，随进度更新。

## 1. 数据集来源表

| 编号 | 数据集 | 用于 | 规模 | 本仓库副本 | 原始来源 | 许可 | 备注 |
|---|---|---|---|---|---|---|---|
| D1 | King County 房屋销售（kc_house_data） | Q1 系数路径复现；Q2 特征选择案例（主数据） | 21613 行 × 21 列 | `data/kc_house_data.csv`（md5 `b782e80f040246c885d674bb16304979`） | Kaggle: [harlfoxem/housesalesprediction "House Sales in King County, USA"](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction) | CC0: Public Domain | 已与 OpenML `house_sales` v1/v2 全表对拍，见 §1.2 |
| D2 | California Housing | 备选 / 稳健性检查 | 20640 行 × 8 特征 | 运行时代取（暂未使用） | `sklearn.datasets.fetch_california_housing` ← StatLib（Pace & Barry, 1997） | 公开数据 | 需要时再拉取 |
| D3 | 合成数据（人工构造高相关特征） | Q2 教材式对照实验 | 自定 | `code/` 内生成 | 自建 | — | 按需 |

### 1.1 与课件图的对应

- 课件 p20 / p42 / p48 图的 8 个特征（`bedrooms, bathrooms, sqft_living, sqft_lot, floors, yr_built, yr_renovated, waterfront`）与 D1 的列名完全一致 ⇒ 判定课件系数路径图的数据来源为 **King County 房屋销售数据**；Q1 复现将取相同 8 列子集（y = price）。
- 课件 p20 手写批注 "all features normalized so coeff. on same scale" ⇒ 复现前必须**先标准化特征**，否则各特征系数不在同一尺度、路径不可比；λ=0 处即最小二乘解 ŵ_LS。
- 课件对应页：p42 = ridge（ℓ2）路径；p48 = 同图 + 批注 "everything has ŵⱼ ≠ 0, but all ŵⱼ small for large λ"（岭回归不产生精确 0）；p49 = lasso（ℓ1）路径（系数随 λ 增大精确归零）；p43–45 = 阈值化 ridge 系数的问题（Q2 依据）。

### 1.2 获取途径与对拍记录（2026-10-11）

- Kaggle 原始页面（下载需登录）：https://www.kaggle.com/datasets/harlfoxem/housesalesprediction
- 本仓库副本下载自 GitHub 镜像（IBM 课程同款 `housing.csv`）：
  `https://raw.githubusercontent.com/Willie-Conway/House-Sales-Price-Prediction-King-County-USA-Project/main/House-Sales-Price-Prediction/housing.csv`
- OpenML 官方镜像（作者本人上传，可编程获取；推荐复现时用）：
  - `sklearn.datasets.fetch_openml(data_id=42092, as_frame=True)`
  - ARFF v2: https://openml.org/data/v1/download/21578898/house_sales.arff
  - ARFF v1: https://openml.org/data/v1/download/21553045/house_sales.arff
- **对拍结论**：镜像副本与 OpenML v1/v2 按 `id` 对齐后，除 `bedrooms`/`bathrooms` 两列的 23 个单元格（bedrooms 13 个、bathrooms 10 个；镜像为**空值**，OpenML 被填成 **0**）外，其余 20 列逐位一致。
- **采用口径**：保留空值（镜像口径），实验中剔除含缺失的 16 行（共 23 个缺失单元格，占 0.07%），在报告/代码中注明。
- 副本清洗：仅去除镜像文件自带的 pandas 行号索引列，未做任何数值改动。

## 2. 进度总览

| 题 | 主题 | 状态 | 数据 | 代码 | 图表 | 结论要点 |
|---|---|---|---|---|---|---|
| Q1 | 复现 ℓ1/ℓ2 系数路径 + 解释数据来源 | 实验完成 ✅ / 待成文 | D1 | `code/q1_coefficient_paths.py` | `figures/q1_{ridge,lasso,both}_paths.{png,pdf}` | 与课件 p42/p49 趋势、量级一致，见 §3 |
| Q2 | ℓ2 阈值化做特征选择不好的案例 | 实验完成 ✅ / 待成文 | D1（9 列）+ D3 | `code/q2_ridge_thresholding.py` | `figures/q2_*.{png,pdf}`（5 张） | τ/λ 扫描选择 5→2→0；共线平摊→整组删（R² −0.0003）；lasso 嵌套保留代表 |
| Q3 | 与 sklearn 实现的拟合差异 | 实验完成 ✅ / 待成文 | world L11 + truth L20 | `code/q3_sklearn_compare.py` | `figures/q3_{convergence,weight_parity,valid_bars}.{png,pdf}` | 目标函数对齐后同解（cos=1.00000）；表面差异=优化器/收敛预算/换算口径，见 §3 |
| Q4 | world 各层效果对比与选层 | 素材齐 ✅ | datasets/world | results/slurm_layers/* | world_layers.png | raw L11 最优（150.43） |
| Q5 | world 正则化选择 | 素材齐 ✅ | 同上 | 扫描报告 §8 + 稳定界说明 | world_grid.png | 标准化 + λ2 稳定界分析 |
| Q6 | valid 预测可视化到地图 | 实验完成 ✅ / 待成文 | datasets/world valid | `code/q6_valid_map.py` | `figures/q6_valid_map.{png,pdf}` | 复算 MSE 150.43 = 训练锚点；大洲级定位（中位 9.5°）、经度误差 ≫ 纬度；仅海岸线无国界线，见 §3 |
| Q7 | 坐标表征 vs 国家识别 | 实验完成 ✅ / 待成文 | datasets/world train+valid | `code/q7_country_baseline.py` | `figures/q7_{baseline_bars,within_country,unseen_map}.{png,pdf}` | unseen 国家模型 +1024 大胜全局基线（查表不可能）；seen 上 oracle 国家均值反超 −35；同国 lat r=0.20 弱显著 / lon ns；中等强度证据，见 §3 |
| Q8 | truth 训练/验证集分布差异 | 实验完成 ✅ / 待成文 | datasets/truth (+world 对照) | `code/q8_split_analysis.py` | `figures/q8_{split_diff,stratified_acc,cv_gap}.{png,pdf}` | valid=轻量 OOD（typo 3.5×、77.4% 新主体）；L20 对差异稳健；CV 随机 vs 分组无差（−0.8pp），见 §3 |
| Q9 | truth 选层 + 正则化 | 素材齐 ✅ | datasets/truth | results/slurm_layers/* | truth_layers.png | L20 峰值（0.8967） |

Q4/Q5/Q9 素材位置：`results/slurm_layers/LAB1 分层训练结果分析报告.md`、`results/slurm_grid/20261009_115559_88653/LAB1 超参数扫描实验报告.md`、`experiments/稳定界说明.md`。

附加实验（不进必交）：**B1** CREAK 逐句对拍已完成（见 §3「附录 B1」，`experiments/B1/`）；**C1**（Qwen 直接输出坐标，`experiments/C1/`，已完成，见 §3「附录 C1」）；**C2**（prompt 变体特征 + 探针，`experiments/C2/`，见 §3「附录 C2」）；**C3**（GeoNames 数据扩增，`experiments/C3/`，见 §3「附录 C3」）；**A1–A3**（线性读出上限包）、**A6**（L1 稠密性）、**A7**（5 折 CV 协议补做）、**A8**（优化器对照）、**A9**（world 地理分布）与 **A10**（球面损失 vs MSE 控制组）见 `experiments/` 各自目录与 §3 对应附录。

## 3. 逐题执行记录

### Q1 系数路径（ℓ1、ℓ2）

- **题面**：重现课件（第 42、48 页）中 ℓ1、ℓ2 正则化的"系数路径"，并解释数据来源。
- **课件依据**：p20 定义 + 标准化批注；p42 ridge 路径；p49 lasso 路径（本版课件页码：见 §1.1）。
- **数据**：D1，取 8 列 + price，剔除 16 个含缺失行（共 23 个缺失单元格），特征标准化。
- **方案（草案）**：λ 网格（含 0）→ 逐个拟合 Ridge/Lasso（sklearn 为主，可用 2.1 节自实现代码交叉验证）→ 画 ŵⱼ(λ)；λ=0 标出 LS 解；对照课件结论（ℓ2 只缩不归零 / ℓ1 精确归零）解释数据来源（Kaggle King County，CC0）。
- **记录**：
  - 2026-10-11：数据集落盘并完成来源对拍（§1.2）。
  - 2026-10-11：**系数路径复现完成**（`code/q1_coefficient_paths.py`，21597 行，特征 z-score，λ 网格 0–2e5 共 201 点；Ridge 用 SVD 闭式解并与 sklearn 对拍 max|Δ|≈1e-9；Lasso 用坐标下降 + warm start）。产物：`figures/q1_ridge_path.*`、`q1_lasso_path.*`、`q1_both_paths.*`。
  - 关键数值（λ=0 最小二乘解，与课件量级一致）：sqft_living 268160、waterfront 63307、bathrooms 49137、floors 28399、yr_renovated 3397、sqft_lot −12378、bedrooms −56868、yr_built −92382。
  - 行为对照：**Ridge** 全部系数只缩不归零；bedrooms（−5.7e4）与 bathrooms（+4.9e4）在 λ≲2e4 先升后降（相关特征的补偿效应），与课件黑/红曲线形态一致。**Lasso** 归零顺序（λ*）：sqft_lot 9k → yr_renovated 13k → bedrooms 21k → floors 23k → bathrooms 26k → yr_built 45k → waterfront 79k；**sqft_living 至 2e5 仍非零**（57.7k）——即"只剩一个特征"，对应课件 p49 "only features in our model"。
  - 结论：课件系数路径图的数据来源判定为 **King County 8 特征 + z-score 标准化 + RSS+λ 口径**，本次复现趋势与量级吻合（个别系数数值有小差，源于数据版本/预处理细节）。

### Q2 ℓ2 阈值化特征选择的反例

- **题面**：用案例说明 ℓ2 正则化在回归中将很小的系数设为 0 进行特征选择是不好的（相比 ℓ1，课件第 44 页）。
- **课件依据**：p43/44（人为阈值 → 结果高度依赖阈值，不稳定）；p45（相关特征在 ridge 下同时收缩 → 阈值化整组删除，"Nothing measuring bathrooms!"）。
- **数据**：D1 取 9 列 = Q1 的 8 列 + `sqft_above`（与 `sqft_living` r=0.876 组成共线对；次级对 bathrooms~sqft_living r=0.756）；合成复制特征用 D3（脚本内生成，x2 ≡ x1）。
- **代码**：`code/q2_ridge_thresholding.py`（Ridge 用 SVD 闭式解并与 sklearn 对拍 4e-10；Lasso 坐标下降 + warm start；含 bootstrap 稳定性与 train/valid 预测代价）。
- **产出图**：`figures/q2_corr_matrix.*`（相关矩阵）、`q2_ridge_band_path.*`（路径+阈值带+保留计数）、`q2_selection_matrix.*`（ridge 阈值化 vs lasso 零模式选择矩阵）、`q2_collinear_case.*`（共线减半 + 复制特征对照）、`q2_bars_story.*`（课件 p43 式条形图+阈值带）。
- **逐图说明**：`figures/q2_figure_notes.md` —— 5 张图的"是什么/精确读数/证明了什么/报告用法"逐图整理（读数按脚本同口径重算核对，2026-10-13）。
- **记录**（2026-10-11）：
  - **① 选择高度依赖阈值 τ（p43/44）**：同一 ridge 拟合（λ=2e4），τ 从 2e4 → 5e4 → 1e5，选中特征数 **5 → 2 → 0**（τ=1e5 时"全删"，sqft_living 也落选）。沿 λ 看（τ=1e4）：选择集跳变 **11 次**，且 **bedrooms、floors 离开后又回归**（共线补偿使系数幅度先减后增），而 lasso 的零模式沿 λ 严格嵌套（违例 0 次）。bootstrap（40 次半样本重抽，τ=1e4）中 `yr_renovated` 仅 92% 保持选中；τ=1e5 下 40 次全部"选空"。
  - **② 共线组被"平摊"后整组删除（p45）**：`sqft_living` 单特征 ridge(λ=2e4) 系数 **133,791**；加入兄弟特征 `sqft_above` 后变为 **86,578 + 59,449 = 146,027**（近似对半切，和≈单特征值）。保守阈值 τ=1e5 下：单特征模型保留（133,791 > 1e5），9 特征模型两个成员都被删（86,578 / 59,449 < 1e5）→ 整个 living-area 组从"必选"变"全删"。
  - **③ 合成复制特征（D3）**：x2 ≡ x1 时 ridge 最小范数解精确对半切 (88,067, 88,067)，任何 τ∈(w/2, w) 都会删掉整组；lasso 保留一个代表 **(237,689, 0)**，组信号完整。
  - **④ 对比 lasso（p46–49）**：零入场 λ*：sqft_lot 10k → yr_renovated 13k → **sqft_above 17k** → bedrooms 21k → floors 23k → bathrooms 26k → yr_built 45k → waterfront 79k；sqft_living 至 2e5 非零。lasso 自动把共线组压成一个代表（sqft_above 精确归零、sqft_living 保留）；λ=1e5 时 lasso 只留 sqft_living（=157,689）仍代表整组，bootstrap 40/40 稳定。
  - **⑤ 预测代价**（train/valid 7:3，按选中特征重拟合 OLS 的 valid R²）：全 9 特征 0.5972；ridge+τ=1e4（7 特征）0.5806；τ=5e4（2 特征）0.5032；**τ=1e5（0 特征，退化为常数模型）−0.0003**；lasso@2e4（7 特征）0.5950；lasso@1e5（仅 sqft_living）0.5033。合成复制特征阈值化删整组后 R² −0.0003（同一信号单独就有 0.5033 的 R²）。
- **解释原因（为什么 ℓ2 阈值化不好、ℓ1 好）**：
  1. **ridge 的解没有稀疏性**：ŵ(λ)=(XᵀX+λI)⁻¹Xᵀy 恒非零、随 λ 连续收缩；"系数小"只说明收缩程度大（λ 大时全体都小，课件 p48），不等于"特征不重要"——阈值化把"收缩量"误当"重要性"。
  2. **共线时 |ŵⱼ| 不是特征重要性的可识别度量**：共线方向的权重可在组内任意重新分配而不改变拟合（预测只依赖组内系数之和），ridge 的最小范数准则把权重平摊（复制 k 份 → 每个 ≈ w/k）。于是加入一个共线兄弟就能把一份重要信号的每个系数压到阈值之下 → 阈值化把整个组删除（课件 p45 的 #bathrooms/#showers；本数据 sqft_living/sqft_above）。
  3. **阈值 τ 是人为的第二个超参数**：选择集随 τ、λ、数据扰动跳变（本例沿 λ 跳 11 次、含非单调回归；bootstrap 下边界特征进进出出），没有任何目标函数可供优化 τ；"先选后拟合"两阶段也失去原估计的名义统计保证。
  4. **ℓ1 的零来自优化本身**：KKT/软阈值给出精确 0；零集沿 λ 单调嵌套、仅由单一超参数 λ 控制；对相关组通常保留一个代表（本数据：sqft_above 在 λ*=17k 归零、sqft_living 一直保留），组信号不丢——这些都是阈值化 ridge 不具备的。

### Q3 与 sklearn 的拟合差异

- **题面**：你的模型相比于 sklearn 的实现有拟合效果上的差异吗？（提示：做完 world/truth 两题后回答。）
- **思路**：先把正则化换算对齐（否则比较的是两个不同的目标函数），再逐行对比；最终指标用仓库 `model.py` 的真实实现，收敛曲线用与其更新式完全一致的仪器化副本（不改动提交代码）。
- **换算（ours 目标 = (1/N)·loss + λ1‖w‖₁ + λ2‖w‖²₂）**：Ridge α=N·λ2；Lasso α=λ1/2；ElasticNet αρ=λ1/2、α(1−ρ)=λ2；逻辑回归 l2 C=1/(2Nλ2)（truth N=5088、λ2=1e-3 → **C≈0.098**；注意 sklearn 默认 C=1 是另一个正则强度）。
- **数据**：world 层 11（A 组 z-score / B 组原始特征，N=500）；truth 层 20（原始特征，N=5088）。
- **代码**：`code/q3_sklearn_compare.py`；**图**：`figures/q3_convergence.*`（收敛曲线）、`q3_weight_parity.*`（权重一致散点）、`q3_valid_bars.*`（valid 指标条形）。
- **逐图说明**：`figures/q3_figure_notes.md` —— 3 张图的读数、结论与报告用法逐一整理（数字取自 run 2 日志核对，2026-10-13）。
- **结果——world**（valid MSE = lat/lon 均值；cos/relL2 = 两坐标权重拼接后与 sklearn 的相似度）：

| 行 | 设置（ours 口径） | sklearn 等价 | ours valid | sk valid | cos | relL2 |
|---|---|---|---|---|---|---|
| A1 | z-score，无正则，ep20000 | LinearRegression（精确 LS） | 306.55 | 311.56 | 0.99929 | 0.039 |
| A2 | z-score，λ2=1.0 | Ridge α=N·λ2=500 | 167.06 | 167.17 | 1.00000 | 0.002 |
| A3 | z-score，λ1=1e-3 | Lasso α=5e-4 * | 238.36 | 598.47 * | 0.594 | 0.814 |
| A4 | z-score，λ1=1e-3 + λ2=1.0 | ElasticNet α=1.0005、ρ=5.0e-4 | 167.04 | 167.16 | 1.00000 | 0.002 |
| B1 | raw，无正则，ep20000 | LinearRegression（精确 LS） | 223.58 | 395.02 | 0.871 | 0.573 |
| B2 | raw，历史最优（λ2=0.1），ep2000 | ElasticNet α=0.1005、ρ=5.0e-3 | 150.43 | 165.34 | 0.897 | 0.475 |
| B2b | raw，历史最优，ep20000 | 同上 | 162.36 | 165.34 | 0.997 | 0.082 |

  　\* A3 的 sklearn Lasso 未收敛到 tol=1e-7（dual gap 6.3 ≫ tol；坐标下降在病态问题上的固有困难），该行 sklearn 数值仅供参考。
- **B3 发散演示**：raw L11、无正则、lr=1e-3（超过稳定界 ≈5.4e-4）→ 第 8 轮 loss 爆炸自动停（1.14e10 vs 初始 2.4e3）；sklearn（SVD 精确解）无步长问题。
- **结果——truth**（valid：准确率 / 对数损失；预测一致率 = 两者 valid 标签一致比例）：

| 行 | 设置 | ours acc | sk acc | ours ll | sk ll | cos | 一致率 |
|---|---|---|---|---|---|---|---|
| T1 | 无正则，ep4000 | 0.8867 | 0.8533 | 0.255 | 27.77 | 0.772 | 0.900 |
| T2 | λ2=1e-3（sk C≈0.098），ep5000 | 0.8867 | 0.8600 | 0.257 | 0.383 | 0.802 | 0.927 |
| T2b | 同 T2，ep10000 | 0.8733 | 0.8600 | 0.269 | 0.383 | 0.883 | 0.947 |
| T3 | 最优配置 λ1=λ2=1e-3，ep5000 | 0.8967 | 0.8600 † | 0.259 | 0.383 | 0.712 | 0.917 |

  　† T3 的 sklearn 侧只配了 l2（没有对应 l1 项），差异主要来自目标函数本身不同；T4 参考：sklearn 默认 C=1（更强正则）→ train 准确率 1.0000、valid 0.8500。
- **收敛曲线**：world z 无正则 GD 20000 轮从 ~1e3 降到 **0.0047**（sklearn 精确解 train MSE=0，但 p≫n 病态下 GD 逼近慢）；truth l2 GD 5000 轮 train logloss 0.181 vs lbfgs 最优 0.0656（无正则"最优"在可分方向上趋 0/无穷远）。
- **一致性锚点**：B2 valid MSE **150.43** = world raw L11 历史最优；T3 valid acc **0.8967** = truth L20 训练报告值 —— 对比装置与分层训练管线一致。
- **结论（回答题面）**：**没有本质差异**。同一目标函数 + 充分收敛时两者同解（A2/A4 cos=1.00000、relL2=0.002；B2b cos 0.997）。观察到的拟合差异全部可归为三类"非模型"因素：
  1. **优化器机制**：固定步长全批量 GD 受稳定界约束（B3：lr=1e-3 第 8 轮即爆炸；sklearn 精确求解无此问题）；坐标下降在病态稀疏问题上同样难收敛（A3 注）。
  2. **收敛预算 ≈ 隐式早停正则**：有限轮 GD 停在更小范数的点、valid 反而更好（A1 306.55 < 311.56；B1 223.58 < 395.02；T2 ours 0.8867 > 0.8600）；延长训练向 sklearn 解靠拢（T2→T2b cos 0.80→0.88）但 valid 可能回落（B2b 162.4 vs B2 150.4）。
  3. **正则口径**：不换算直接比（如 sklearn 默认 C=1 vs 匹配 C≈0.098）会把"超参不同"误读成"实现差异"。
  一句话：差异是**优化差异**（步长/求解器/预算/早停），不是**模型或实现差异**。

### Q6 验证集预测可视化（地图）

- **题面**：给出你的模型在验证集上的预测结果，并可视化在地图上（推荐不含国界线）。
- **模型与数据**：world raw L11 历史最优探针（`results/slurm_layers/world/weights/world_layer11_extra0/weights.npz`，standardize=false，权重已折回原始特征空间，推理时不再 z-score）；`datasets/world/valid.layer11.npy`（130×2560）+ `valid.csv` 真值 (lat, lon)。推理为纯矩阵乘，无需集群（权重当年本地 18.3s 训练）。
- **代码**：`code/q6_valid_map.py`（复算 MSE 自检 + cartopy PlateCarree 全球地图，110m 海岸线，**无国界线**——合规红线）。
- **产出图**：`figures/q6_valid_map.{png,pdf}`；**逐图说明**：`figures/q6_figure_notes.md`。
- **记录**（2026-10-13）：
  - 复算锚点：valid MSE = **150.4277**，与 `results.json` 历史最优记录 **完全一致**（口径闭环：权重折回、列序 (lat,lon)、dtype 均正确）；R² = **0.9545**。
  - 单点定位误差 √(Δlat²+Δlon²)：平均 **12.44°**、中位 **9.52°**（四分位 4.96 / 9.52 / 15.34°），>30° 仅 7/130。
  - 误差结构：**纬度 MAE 4.25° vs 经度 MAE 10.85°**（约 2.5 倍）——模型对气候带式纬度线索敏感、具体经度定位弱；最准样本 <1.1°（Denmark 0.79°）；最差为南极洲 2 条（89.59° / 73.79°，预测均落到 ~28–34°E 的南大洋；valid 南极样本共 2 条，全部进入 top-2；valid 覆盖 78 国）。
  - 结论：模型具备**大洲级**定位能力（中位误差约千公里量级），高纬/孤立岛屿样本失败，与训练样本覆盖直接相关（Q7/Q8 伏笔）。图合规：仅海岸线、无国界线。

### Q7 坐标表征 vs 国家识别

- **题面**：从训练集和验证集的表现，是否能够强烈说明模型内部存在地理坐标的表征，而不是仅仅识别了输入中的国家？（可参考论文 [3]，可与按训练集国家分组的平均坐标基线比较，考察同一国家内不同地点。）
- **设计（三层证据 + 双基线）**：① **oracle 国家均值基线**——每国取训练集该国坐标均值，用真实国家标签查找（对基线有利；未见国家回落全局均值 3311.27）；② **同国中心化**——真值/预测相对该国训练均值的偏差，检验"仅国家识别"预测的"常数预测"假设；③ **unseen 国家**——训练集未出现（查表不可能）；另测数据重叠（valid→同国最近 train 点距离）。集合分层：all 130 / seen 110 / multi 73 / single 37 / unseen 20（19 国）。CI 为 paired bootstrap（10k，seed 0）。
- **代码**：`code/q7_country_baseline.py`；**图**：`figures/q7_{baseline_bars,within_country,unseen_map}.{png,pdf}`；**逐图说明**：`figures/q7_figure_notes.md`（含论文 [3] §4.1/§5 对应表）。
- **记录**（2026-10-14）：
  - 复算锚点：valid MSE = **150.4277**（= 训练锚点，口径闭环）；train MSE 116.9279；全局均值基线 3311.27（= Q6 锚点）。
  - **分层基线**（improvement = base − model，>0 模型更好）：all 130：150.43 vs 278.24，**+127.81 [27.24, 253.03]**；seen 110：148.89 vs **113.77**，**−35.12 [−60.90, −7.08]**（oracle 反超）；multi 73：−45.39 [−82.22, −6.94]；single 37：−14.85 [−39.79, 14.07]（ns）；unseen 20：158.90 vs 全局 1182.83，**+1023.93 [513.94, 1654.14]**。稳健性：multi 去南极洲（n=71）115.87 vs 65.24，方向不变。
  - **数据重叠侦察**（解释 oracle 为何强）：valid→同国最近 train 点距离中位 **2.03°**，40/130 <1°、89/130 <3°——大量验证地点与训练地点重合/邻近；极端例 Jamaica base MSE=0.01。
  - **同国中心化（n=73）**：lat r=**0.201 [0.021, 0.389]**（弱显著），slope 0.115；lon r=**0.046 [−0.152, 0.214]**（ns），slope 0.027——方向对但幅度强烈收缩。US（n=8）放大：中心化 r=0.69/0.60（lat/lon），方向大体正确。
  - **unseen 逐条**：最差 Mauritius 38.03（经度掉到非洲大陆）、Estonia/Russia 31.60、Finland 24.53；最好 Denmark **0.79**；预测整体落对区域（欧洲→欧洲、中美→中美）。
  - **结论（分层回答题面）**：Ⅰ "不仅识别国家"——unseen 上查表不可能却 +1024（强）、同国 lat 弱显著（弱-中）、all-130 胜 oracle（由 unseen 驱动）；Ⅱ 限度——seen 上 oracle 反超（−35，CI 不含 0）、经度国内无信号、valid↔train 重叠大。**综合为"中等强度"表征证据**，与论文 §4.1 自评 "weak evidence"/"relative position correct, absolute wrong" 一致；"生成时使用"的强证据（论文 §5 神经元+干预）超出本实验范围（无权重），只能引用；并须注明 README §1.2"可解码 ≠ 使用"。

### Q8 truth 训练/验证集分布差异与应对

- **题面**：仔细阅读训练集和验证集，模型的训练集和验证集存在哪些分布上的差异？你是如何应对的？
- **设计（四组分布统计 + 分层稳健性 + CV 对照）**：① 错拼检测（pyspellchecker 字典 + 相邻字母交换判定）；② 主体新颖度（句首连续大写词组识别首实体，valid 实测 vs 从 train 随机抽 300 的零假设 200 次重抽）；③ 模板/句长；④ world 对照（新国家/新地点字符串/真正空间远离）；⑤ L20 探针在 valid 上的分层 acc（typo/主体×标签，bootstrap CI）；⑥ 13 层 typo 敏感度；⑦ train 内部 3 折 CV 随机 vs 按主体分组。
- **范围说明**：本页未做 CREAK 逐句对拍（当时外部数据集未获取）；2026-10-16 已由 **B1** 补齐（`experiments/B1/`，见下文「附录 B1」）：truth 语料 100% 来自 CREAK，Q8 检出的错拼绝大部分为课程注入。
- **代码**：`code/q8_split_analysis.py`；**图**：`figures/q8_{split_diff,stratified_acc,cv_gap}.{png,pdf}`；**逐图说明**：`figures/q8_figure_notes.md`（含 §1-3 读数与"报告怎么用"）。
- **记录**（2026-10-16）：
  - **差异①拼写噪声**：相邻字母换位型错拼句率 truth **valid 24.7%（74/300） vs train 7.0%（354/5088）= 3.5×**，标签内均衡（25.3/24.0 与 7.1/6.8）；world 两侧 0.8%/0.8% 且全为地名假阳性（centre/rila/iber 等）⇒ 错拼是 truth 特有的、只在 valid 侧加重。错误类型形近交换（mkaes、fihgters、connectiuct）。
  - **差异②主体新颖度**：valid 212 条可识别主体中 **164 条（77.4%）未在 train 出现**；零假设（train 随机抽 300，200 次重抽）mean 20.8%、97.5% 分位 25.6%——远超。头部新主体 Dave Grohl×8、Dwyane Wade×7、The Offspring×6。稳健性：typo 落在主体内部的仅 8 条、修正后仍全部"未见"⇒ 不受错拼污染。
  - **差异③模板/长度**：train 特有 "One can"（129，2.5%）/"If one"（24）在 valid 几乎消失（2/0）；句长 train mean 61.0/max 487 vs valid 54.2/162。
  - **差异④world 对照**：无噪声差异；valid 20/130 未见国家、126/130 地点字符串未见，但多数空间邻近（89/130 距同国最近 train 点 <3°），真正远离（≥3°）仅 **41** 条——字符串新颖 ≠ 空间新颖。
  - **分层稳健性（L20，锚点 0.8967 逐位复算一致）**：clean 0.885 [0.841,0.925] vs typo 0.932 [0.865,0.986]（diff −0.047，CI 含 0）；seen 0.917 vs unseen 0.884（diff +0.033 ns）。**逐层**：浅层怕错拼（L5/L8/L11 干净−typo 差 +5.2/+6.4/+6.7pp），中深层反向（L17 −6.1pp）——深层表征抗噪。
  - **应对实验（CV）**：train 内部 3 折（n=3785 可识别主体行、同配置 lr=5e-3/5000ep）：随机均值 0.869（0.867/0.858/0.883） vs 分组均值 0.877（0.871/0.880/0.881），**gap −0.8pp 在折间噪声内**；分组折测试主体 100% 未见。⇒ **"随机划分高估"假设未被证实**：L20 不依赖主体记忆，valid 的 0.8967 无主体记忆注水；与分层结果两条独立路径同结论（如实报告负结果）。
  - **结论**：truth 的 valid 是"轻量 OOD 挑战集"（加噪 + 主体轮换 + 长度截断），world 的难在空间外推——两个数据集 OOD 设计维度不同；我们的应对 = 不向 valid 泄漏（固定预算、不按 valid 早停）+ 分层 CI 读数 + 文本不清洗（保持口径一致，深层选层天然抗噪）+ 同分布口径用 train 内部 CV 对照。

### 附录 B1 CREAK 逐句对拍（可选实验，Q8 增强证据）

- **来源**：《可做实验清单》B1（指南3 §3.1，同学参考报告非官方）；2026-10-16 指派执行。属**附加探究**（不进课程必交内容）；若报告篇幅允许，可作为 Q8 的"来源级证据"引用。
- **设计**：本地 truth 5388 句对 CREAK 公开数据（Onoe et al. 2021, NeurIPS Datasets & Benchmarks；`github.com/yasumasaonoe/creak`，本地 clone rev `5e892b28`，train/dev/test_without_labels/contrast_set = 10176/1371/1371/500 行）三层匹配：direct（whitespace 归一化逐字相等；**不使用标签检索**）→ near（rapidfuzz fuzz.ratio ≥ 90 最优候选，候选按归一化去重、margin=top1−top2）→ none；标签一致性匹配**之后**单独核对；编辑按 candidate→local 单 token 替换分类（剥 token 首尾标点）。
- **代码**：`experiments/B1/b1_creak_match.py`（22 项锚点内置自检，~12–30 s）；**图**：`experiments/B1/figures/b1_{match_tiers,source_attribution,perturbation_dist,q8_cross}.{png,pdf}`；**逐图说明**：`experiments/B1/b1_figure_notes.md`；明细数字：`experiments/B1/b1_metrics.json`（818 个 near 对逐条 + 全部统计；CREAK 数据不入 git，`data/` 已 ignore，revision+md5 记在 metrics）。
- **记录**（2026-10-16 完成）：
  - **匹配 100% 覆盖**：train 4380 direct（86.1%）+ 708 near（13.9%）；valid 190（63.3%）+ 110（36.7%）；**none 两侧 = 0**。direct 率与 guide3 §3.1 逐位一致（86.08% / 63.33%）；near 分数中位 97.5/97.0、margin 中位 38.3/35.2、无并列 ⇒ 全部可追溯且无歧义。
  - **来源归属**：train **全部**来自 CREAK train.json；valid 全部来自 dev（269）/contrast_set（27）/test（2）——**与 train.json 零重叠**（课程复刻了 CREAK 自己的训练/留存划分）。
  - **标签**：5388 条匹配后核对，与 CREAK 标签 **0 冲突**（继承无损）。
  - **扰动分类**：846 处编辑 = 相邻字母交换错拼 **415**（train 318 / valid 97；如 mkaes、tehir、conutry、Fihgters、Nrivana）+ 主谓一致翻转 **431**（is↔are 252、has↔have 99、was↔were 69、do↔does 11；train 390 / valid 41），**无第三类**（无实体替换/内容词改写/标点编辑）；28 个 valid 对含双错拼。⇒ 课程扰动 = 两类**语义保值**编辑（词面级），与"深层探针抗噪、浅层怕噪"（Q8 §3）机理一致。
  - **Q8 交叉**：Q8 检出的 typo 句 train 354 = **307 课程注入**（字典还原交换后与 CREAK 原文逐字一致，修复率 96.5%）+ 40 CREAK 自带（专名假阳性：Pharoah/Niels/Ewan/Paula Deen/Dwyane Wade）+ 7 其他；valid 74 = 65 注入（94.2%）+ 6 自带 + 3。⇒ Q8 的 valid 错拼率 24.7% 基本全部是课程注入；另发现字典修复器会误伤实体名（MOBA→MOAB、Elon→Leon、Loni→Loin）——"不做文本清洗"的又一警示。
  - **结论**：课程语料 = CREAK 原文 + 两类扰动、valid 用 CREAK 保留文件换源——Q8"valid 是设计过的轻量 OOD 挑战集"获得**来源级证据**（统计推断 → 逐句证据闭环）。

### 附录 C1 Qwen 直接输出坐标 vs 线性探针（附加实验）

- **来源**：`guide/计划：C1 方案定稿…md`（参考报告为`指南3.pdf` §4，**非官方**）；属**附加实验**（不进课程必交内容），对照我们自己的最优探针（world raw L11）。
- **设计**：Qwen3-4B-Base（本地克隆 rev `906bfd4b`）对 world 630 题（train 500 + valid 130）**零样本直接生成**经纬度——prompt 在原题面后追加 JSON 指令，贪心解码、max_new_tokens=64、bf16，集群 A100-80GB（学生分区钉卡）；本地 CPU 评估：解析分层（json/embedded/regex/out_of_range，in_range 过滤）→ MSE（原始 + 周期经度）+ haversine 距离分布 + hit100 → 与探针逐条配对 bootstrap → 单条平方误差 top-5 → Tomanivi 现象检查。探针锚点自检（valid 150.4277/953.8 km、train 116.9279/662.4 km，±1e-3）不符即中止。
- **代码**：`experiments/C1/{c1_lib.py, c1_qwen_direct.py, c1_eval.py, run_c1_qwen_direct.sbatch}`；运行资产 `results/slurm_qwen_direct/qwen_direct_v1/`（jsonl + run_meta + c1_metrics.json）；**图**：`experiments/C1/figures/c1_{distance_cdf,valid_map,metric_bars}.{png,pdf}`；**逐图说明**：`experiments/C1/c1_figure_notes.md`（含与参考报告对拍表与注意事项）。
- **记录**（2026-10-16 完成）：
  - **解析稳健**：train 严格 499/500 = 99.8%（唯一失败 = Phnom Penh，输出纬度 115.24 越界、疑似经纬互换）；valid 宽松 130/130 = 100%（129 json + 1 regex = Yaoundé 输出度分秒、regex 只取首数）。
  - **距离**：valid 中位 **43.8 km**（探针 953.8，≈22×）、均值 159.1（探针 1141.7）、P90 416.0（探针 2391.0）、hit100 **63.85%**（与参考报告完全相同）；train 中位 15.5 km、hit100 68.8%。逐条配对（n=130）：**直接更近 119/130**，距离差（探针−直接）**+982.5 km，CI95 [835.0, 1138.9]**。
  - **MSE 易碎性**：valid 原始 MSE 505.76 的 **96.52%** 来自单条 Mount Tomanivi（斐济，预测 lon −178.25 落在±180° 另一侧，dlon_raw −356.265°，实际仅偏 397.0 km；预测值与参考报告逐位相同）→ 周期口径 17.64；train 原始 159.64（~92% 来自 5 条极地/海外属地）→ 周期 108.54；combined 231.18/89.75。**原始 MSE 下探针(150.4) 反而"更小"**——只报原始 MSE 会把结论讲反。
  - **与参考报告（非官方）一致性**：valid hit100 完全相同（0.6385）、均值 159.1 vs 159.0、Tomanivi 预测逐位相同 ⇒ 同 revision + 贪心解码可复现；中位 43.8 vs 35.8 等小差异源于解析接受度（我们宽松收满 130/499 vs 参考 128/495）。
  - **结论**："线性读出误差 ≠ 模型没有地理知识"（与 Q6/Q7 闭环）；附加实验口径，仅并列报告、不宣称"探针 < 生成式"；指标须捆周期 MSE + 距离分布一起报。

### 附录 C2 Prompt 变体特征提取与探针（附加实验）

- **来源**：《可做实验清单》创新点 6（换 prompt 抽特征）；属**附加实验**（不进课程必交内容），与 Q7 反向呼应——直接切掉国家名/指令，看坐标可读性还剩多少。
- **设计**：同一批 world 500/130 条，5 个变体：原句国家 token（world 历史协议）、原句句末 token、删国家名（句末 token）、仅"地名, 国家"无指令（国家 token）、换问法（句末 token）。配对设计固定读点：原句→仅地名（只去指令，同一 token）、原句末→去国家名（只删国家名，同一 token）、原句末→换问法（只改问法）、原句国家 token→原句末 token（只换读点）。Qwen3-4B-Base rev `906bfd4b`，13 层、float16；探针与课程协议一致（std13 = z-score/lr 0.002/2000 轮/λ2=1.0；raw11 = 原始/lr 5e-4/λ2=0.1）。
- **代码/资产**：`experiments/C2/{c2_extract_variants.py, c2_train_probes.py, c2_eval.py, c2_probe_config.json, run_c2.sbatch}`；变体 CSV `experiments/C2/data/`；特征（集群，gitignored）`experiments/C2/features/`；训练产物 `results/slurm_c2/c2_v1/`（c2_metrics.json 含逐条 valid 预测）；图 `experiments/C2/figures/c2_{layer_curves,variant_bars,paired_effects}.{png,pdf}`；逐图说明 `experiments/C2/c2_figure_notes.md`。
- **记录**（2026-10-17 完成；集群 A100，70 次训练中 69 次有限）：
  - **锚点**：origcountry 的 14 个历史 MSE 中 13 个逐位一致；L35 差 +0.77（204.1408 vs 204.9075，特征 float16 数值差，cos 0.999684）、raw11 差 −0.10（150.3259 vs 150.4277）。`nameonly` 的 raw11 在 4 次降 lr（5e-4→6.25e-5）后仍发散，故 raw11 只作辅助，主口径用 std13。
  - **主结果（std13-L11, n=130）**：origcountry MSE **167.4** / 中位 **909 km** / ≤1000 km 55.4%；origlast 539.1 / 2280 / 11.5%；nocountry **1069.4 / 3115 / 9.2%**；nameonly **224.6 / 1100 / 45.4%**；rephrase 492.6 / 2163 / 13.8%。均值基线（train 平均坐标）MSE 3311 / 中位 7300 km。
  - **配对（bootstrap 95% CI）**：删国家名 **+1004.9 km [623.5, 1391.3]**、预测漂移 2915.6 km；去指令 **+232.6 [76.8, 385.4]**、漂移 905.2；换问法 **−199.2 [−398.0, +0.3]**、漂移 1319.8；换读点（国家 token→句末 "?"）**+1511.4 [1238.9, 1783.6]**、漂移 2531.3。raw11 辅助：删国家名 −149.0 [−420.7, +125.4]（n.s.，句末读点已接近地板）、换问法 +141.7 [−13.4, +295.5]、换读点 +4094.3 [3515.4, 4699.8]。
  - **分层（国家是否在 train 出现）**：删国家名 seen +1127 [725, 1555] vs unseen +335 [−640, 1243]；去指令 seen +297 [151, 448] vs unseen −122 [−716, 383]；换问法 seen −233 [−450, −1]、unseen −12；换读点 seen +1577、unseen +1152。国家名效应集中于见过的国家，与 Q7 的"国家→坐标先验"一致；未见国家仍有地名信息。
  - **结论**：国家名是主要信息来源但非唯一（无国家名仍比均值基线好 3.1×）；指令是锦上添花（浅层几乎无差、深层差距增大）；换问法影响小；读点位置影响最大（跨变体比较必须固定读点）；与 Q7 反向闭环（删国家名的损失集中在 seen 国家）。

### 附录 A1–A3 线性读出上限包（多层拼接 / sklearn 家族 / 自训 MLP，附加实验）

- **来源**：《可做实验清单》A1/A2/A3（参考报告为`指南3.pdf` §2，**非官方**）；2026-10-16 指派执行。属**附加探究**（不进课程必交内容），回答"线性读出是否已接近任务上限"。
- **设计**：统一闭式协议（world 主口径 = 双形式 ridge 固定 λ2=1.0；truth = lbfgs 固定 C）对照三族：① **A1 多层拼接**——13 层特征各自 train z-score 后拼接，97 种组合（13 单层 + 78 对 + 6 三元，world 与 truth 双任务）；② **A2 sklearn 家族**——world：KNN/Ridge/KernelRidge/ET/SVR/OLS/HistGB/RF，truth：LR/LinearSVC/SVC-RBF/HistGB/RF/ET/KNN；③ **A3 自训 MLP**——PyTorch AdamW（lr=1e-3、wd=1e-4、patience 30、3 seeds），linear/mlp64/mlp128_64 三架构，train 内部 10% 早停（不触 valid）。
- **代码**：`experiments/A1/a1_multilayer_concat.py`、`experiments/A2/a2_sklearn_family.py`、`experiments/A3/a3_mlp_torch.py`（本地跑，合计 ~2.4 h）；**图**：`experiments/A{1,2,3}/figures/`（a1_concat_heatmaps / a1_best_combos / a1_corr_matrices / a2_model_bars / a3_model_bars / a3_curves，各 png+pdf）；**逐图说明**：`experiments/A{1,2,3}/a{1,2,3}_figure_notes.md`。
- **记录**（2026-10-16 完成）：
  - **A1 多层拼接**：world 最优 **L5+L32 → 128.82**（中位 800.4 km；CV 选参口径 139.56、λ2=3），比最好单层 L2（157.62）低 18%，也低于历史提交 150.43；78 对中 21 对超最好单层、头部对全部含 L32；truth 最优对 L17+L20 = 0.8833（vs 单层 L20 0.8667，6/78 超越）。锚点复算逐位一致（L11 167.1707 / L2 157.6186）。
  - **A2 sklearn 家族**：world 仅 **L11 KNN k=3 → 140.70** 超过闭式 Ridge（L2 上 Ridge 157.62 即最优）；核/树方法全面偏弱（SVR 296.2/206.9、RF 388.8/273.1、HistGB 332.2/232.4；HistGB 需 MultiOutputRegressor 包装，首跑崩溃留档 `experiments/A2/a2_firstrun_crashed.log`）；truth 全家族挤在 0.85–0.877（**SVC-RBF 0.8767** ≈ 参考报告 0.87，LinearSVC 0.8633）。
  - **A3 自训 MLP**：均值均未超闭式锚——world L11 mlp128_64 **175.94±17.09**、L2 193.59±42.57；单 seed 可及/超过（L2 144.25、L11 159.43）但折间 std 12–43 很大；同配方 linear-AdamW 对照 476–504（欠拟合，呼应 Q3：结论对训练/正则口径敏感）；truth mlp64 **0.8900±0.0109** 为 z 协议最高（略超 A3 linear 0.8844 的 AdamW 口径）。
  - **跨实验统一排行榜**（world valid MSE）：**128.82 拼接 L5+L32 < 140.17 三元组 < 140.70 KNN < 150.43 历史提交 < 157.62 单层 Ridge(L2) < 175.94 MLP 均值**；truth：0.8900 MLP-64 ／ 0.8833 拼接 L17+L20 ／ 0.8767 SVC-RBF ／ 0.8667 单层锚（GD raw 0.8967 为历史协议）。
  - **结论**：**"换表征（选层/多层拼接）+ 对齐正则口径"的收益 ≥ "换更复杂的模型族"**——线性读出（尤其多层拼接）已接近该特征下的可达上限；一切对比以统一闭式口径为准，A2 的 valid 网格选择与 A3 的随机种子波动均如实标注口径与方差。

### 附录 A7 5 折 CV 协议补做（world/truth 全 13 层，附加实验）

- **来源**：《可做实验清单》A7（指南1 §4.2，参考报告为`指南3.pdf`，**非官方**）；2026-10-17 指派执行。属**附加探究**（不进课程必交内容），用途是**方法论严谨性**：在 train 内部做交叉验证，检验"用 valid 选层"的稳健性，并补上指南综述标注缺失的 CV↔valid 一致性图。
- **设计**：只在 train 上做 **5 折 × 3 重复**（seeds 0/1/2）：truth 按标签 `StratifiedKFold`；world 按纬度五分位分层（158 国、最小样本 1，无法按国家分层）；每折用训练折自身统计量 z-score（无泄漏）。估计器与 A1 单层完全一致：world 闭式 dual ridge 固定 λ2=1.0、truth lbfgs 固定 C=1/(2n_tr·λ2)、λ2=1e-3。指标：world 元素均值 MSE + 中位 haversine；truth acc + logloss；OOF 取 3 次重复的均值、误差条 = 15 折 std。对照列取 `experiments/A1/a1_results.json` 的同协议 train/valid（锚点 world L2 valid 157.6186、truth L20 0.8667 在脚本内断言）。13 层 × 15 折 × 2 任务 = 390 次拟合，本地 344.3 s。
- **代码**：`experiments/A7/a7_cv_protocol.py`（含锚点断言）；**记录**：`a7_results.json`（逐层记录 + 各折指标 + 13×15 秩矩阵 + 折冠军 + 相关性 + selection 块）、`a7_summary.csv`、`a7_run.log`、`a7_tails_audit.log`；**图**：`experiments/A7/figures/a7_{cv_curves,cv_vs_valid,fold_stability,world_error_tails}.{png,pdf}`；**逐图说明**：`experiments/A7/a7_figure_notes.md`（含逐层表与 6 条 caveats）。
- **记录**（2026-10-17 完成）：
  - **truth：选层可复现**。CV↔valid 秩相关 **ρ=0.968**（13 层）；CV 最优 = valid 最优 = **L20**——CV 0.8508±0.0124 vs valid 0.8667，**selection regret = 0**；15 折冠军 L20 **11/15**、L23 4/15。CV logloss↔valid acc ρ=−0.916（仅方向相反，按 logloss 选层仍落 L20）。train↔valid ρ=0.935，但 train acc 自 L17 起饱和在 0.998–1.000，**不能用于选层**。CV 略低于 valid（每折只用 80% 数据；Q8 已证 typo/新主体不伤 L20），深层差 0.4–2.0 pp。
  - **world：头部是平台，不是单峰**。CV↔valid ρ=**0.703**（p=0.007）显著但远低于 truth；CV 最优 **L8** 351.0±150.6（该层 valid 158.9），valid 最优 **L2** 157.6（该层 CV 378.9）。但 CV 选 L8 的 valid regret 仅 **+1.29 MSE**（≈0.8%，与折间噪声同级）；15 折冠军 L8 7/15、L2 3/15、其余 5 层各 1——**头部层统计上不可分辨**，报告应写"L2/L8 平台"而非"L2 冠军"。CV↔valid 的中位距离 ρ=0.852（p=2.2e-4）、CV 内均值 MSE↔中位 km ρ=0.731——mean/median 两种误差排序同向，结论不依赖单一指标。
  - **world CV 均值 MSE 高于 valid 是尾部效应，不是"CV 更差"**。L2：CV OOF MSE 368.5（重复 0）vs valid 157.6，但典型误差一致（中位 SE **47.1 vs 45.1**、中位距离 **988 vs 978 km**）；差异全在尾部——CV top-5/500 平方误差占 SSE **40.6%**（valid top-5/130 占 53.0%），最大 SE 22,235 vs 3,930，valid 去掉 top-5 后 MSE 仅 **77.0**；L8 同构（361.1 vs 158.9；中位 60.5/1074 vs 45.6/944；max SE 33,493 vs 4,332）。成因：Q7 已证 valid 是"新地名、近位置"的近邻划分（中位 2.03° 到同国 train 点），随机折会抽到太平洋孤岛/南极等孤立点。**world 的均值 MSE 对划分方式高度敏感，跨划分比较须并列中位数/尾部。**
  - **train 指标不能选层**：world train↔valid ρ=−0.247（ns），且 in-sample MSE 83.1–125.5 比 CV 低 2.8–5.4×；truth train acc 饱和（见上）。
  - **结论**：truth 的"L20 最优"经 CV 完全复现（同选、零 regret、11/15 折冠军），报告可放心把 valid 选层作为主结论；world 的 valid 选层在 CV 下**排序不稳但代价极小**（+1.29 MSE），应按"L2/L8 平台 + 折间 CI"报告，并把"均值 MSE 的划分敏感性（离群点尾部）"写进 limitation；选层一律不用 train 指标。

### 附录 A10 球面损失 θ² vs MSE 控制组（world L11，附加实验）

- **来源**：《可做实验清单》A10（指南3 §5，参考报告**非官方**）；2026-10-17 指派执行。属**附加探究**（不进课程必交内容）。问题：把 world L11 探针的坐标 MSE 换成球面中心角平方 $\theta^2$，地理距离能否改善？**周期等价解与坐标越界会把课程口径 MSE 扭曲到什么程度？** 指南要求在 valid 上选超参；我们的改动是**超参与早停只在 train 内部 5 折 × 3 seed CV 上选，valid 只做最终报告**。
- **设计**：模型 $p=90(Az+a)$（$A\in\mathbb{R}^{2\times2560}$），特征按训练折 z-score。三组损失：① MSE 控制 $\frac1{2N}\sum\|p-y\|^2+\lambda\|A\|_F^2$；② 球面 $\frac1N\sum\theta_\varepsilon^2+\lambda\|A\|_F^2$，$\theta_\varepsilon=\mathrm{atan2}(\sqrt{\|u\times\hat u\|^2+\varepsilon},u\cdot\hat u)$、$\varepsilon=10^{-12}$；③ hybrid $\theta^2+\beta\cdot\mathrm{MSE}_{\rm rad}$。网格 lr $\{10^{-3},3\times10^{-3}\}$ × $\lambda\{0,10^{-3},10^{-2}\}$；hybrid 固定球面选中 lr/λ 扫 $\beta\in\{0.01,0.1,0.2,0.3,0.5,1.0\}$。选择/早停：train-only 5 折 × 3 seed OOF（MSE 组按 OOF 原始 MSE；球面/hybrid 按 OOF 平均大圆距离），patience 150、≤1200 轮；最终全 train 重训，轮数 = 15 次 best epoch 中位数，3 seeds 报均值±std。数值自检：球面梯度对拍指南解析式 max abs err **3.85e-19**；ridge 锚点 valid MSE **167.1707** / 中位 **923.1206 km**（脚本内断言）。
- **代码/资产**：`experiments/A10/a10_spherical_loss.py`（`--hybrid` 全量、`--smoke`、`--replot`；本地 CPU **108.9 s**，torch 2.14.1+cpu / 4 threads）；结果 `a10_results.json`（protocol/anchors/cv/selected/ridge + 逐折记录）、`a10_predictions.npz`、`a10_weights.npz`、`a10_run.log`；图 `experiments/A10/figures/a10_{training_curves,metric_bars,hybrid_sweep,distance_ecdf,valid_map}.{png,pdf}`；逐图说明 `experiments/A10/a10_figure_notes.md`（含 10 条反射样本表与 7 条 caveats）。
- **记录**（2026-10-17 完成）：
  - **球面组距离大幅改善**：valid 平均 **1395.7±13.9 km vs MSE 控制 1977.3±119.2（−29.4%）**、中位 **1157.0±29.5 vs 1718.4±117.1（−32.7%）**；OOF 同向（1542.7 vs 1987.4，−22.4%），是分布级改善（ECDF 大部分分位优于 MSE 控制），不是少数样本的均值效应。
  - **周期等价解确实出现，raw MSE 必须与分支诊断同报**：valid **10/130（7.7%）**、OOF **34/500（6.8%）** 输出落在反向纬度分支（原始 lat 低至 −184°）；课程口径 raw MSE **1801.45±11.74 vs 357.40±23.82（5.0×）**，但分支规范化 **370.46 vs 357.40（+3.7%）**、周期 MSE **289.93（−18.8%）**——"球面 MSE 更差"主要是坐标表示问题，不是 3D 位置。10 条中 9 条是澳洲—大洋洲点（真实经度 130–178°E），规范化后 SE 仅 7–295 deg²；第 10 条南极半岛（idx 101）规范化后仍 SE=22,982（真 3D 错误）。排除 10 条反射，球面规范化 MSE **205.9**（seed 0）< MSE 控制 383.0。
  - **hybrid β=0.5 消掉反射**：β 由 0.3→0.5 是相变（反射 2.6%→0、raw MSE 770→485、平均距离 1775→1844 km）；β=0.5 最终 valid **239.71±2.63 MSE / 1187.5±28.3 中位 / 1460.9±10.6 平均 / 0 反射**，为三训练组综合最优；但 **OOF MSE 484.66 vs MSE 控制 472.46（+2.6%）**，valid 的 −33% 未复现，距离优势 OOF 成立（−7.2%）⇒ MSE 改进只能表述为"valid 上的观察"。
  - **球面目标非凸/分支不稳**：同 lr=3e-3 下 MSE 控制正常（OOF 1987 km、0.2% 反射），球面组塌到 **43.5% 反射、OOF 3728.6 km、250 条纬度越界**（有限值坏分支解，非数值 inf/nan）⇒ 球面需更小 lr（1e-3）并监控反射率。λ 不可分辨：球面 lr=1e-3 下 λ=0/1e-3/1e-2 的 OOF 平均距离 1545.1/1544.9/1542.7 km（差 ≈0.7× seed std 3.4）；MSE 控制 lr=3e-3 下 OOF MSE 476.5/476.2/472.5（std 13–18）。
  - **ridge 仍是整体最优**：valid 167.17 MSE / 923.1 中位 / 1203.1 平均，优于所有梯度训练组。A10 只回答"梯度训练探针内部换损失"，不能写成"球面损失优于 ridge"。**ensemble 仅作诊断**（球面 unit-vector 平均天然消分支，规范化 MSE 834.5 反比单 seed 均值 370.5 差）；正文用 seed 均值±std、地图/ECDF 用 seed 0。
  - **口径说明**：指南参考数（valid 选参口径）不可逐位比较；我们所有选择在 train-only OOF，valid 只用一次，且最终用全部 500 条重训（CV 折仅 400 条），故 valid 指标系统性优于 OOF。
- **结论**：球面 $\theta^2$ 在距离指标上是真实的分布级改善，但会产生 ~7% 周期等价解并把课程 raw MSE 放大 5×，必须同时报告分支规范化/周期 MSE 与越界计数；hybrid β=0.5 是零反射的工程折中（valid 240 MSE / 1187 km），但其 MSE 优势未被 OOF 复现；闭式 ridge（167 / 923）仍是整体上限参照。

### 附录 C3 GeoNames 数据扩增（附加实验）

- **来源**：《可做实验清单》C3 / 指南2 §1（参考报告**非官方**）；2026-10-17 指派执行。属**附加探究**（不进课程必交内容）。问题：在 world 500 条之外重采样上万 GeoNames 地点、重提 Qwen 特征并训练探针，能否提升原 world 任务与"未见国家"泛化？
- **数据/资产**：GeoNames `cities15000`（34,166 城 / 244 国，md5 `93552399…`）+ `countryInfo`（252 国，md5 `8b8c1506…`）；12,000 条 → train 8,382 / valid 1,050 / test 1,050 / test_unseen 1,518（30 国整国留出，每国 8–500 城），seed 20261004；prompt 与 world 完全一致（城邦省略国家后缀），国家名最后 subtoken 读点，13 层 float16。泄漏审计：train 与 world valid 同名同国 14 条（与 world train 68 条），`new_noleak`/`union_noleak` 剔除；5 训练集 orig/new/new_noleak/union/union_noleak × 4 评估集。**严格未见子集**：`test_unseen` 30 国中 22 国出现在 world train，故另设 `strict8`（8 国 260 条）/`hard6`（去 Niger/Palestine，6 国 212 条）；这两个子集对 orig/new/union 都干净。
- **代码/资产**：`experiments/C3/`（`c3_build_dataset.py`、`c3_extract.py`、`c3_train_probes.py`、`c3_eval.py`、`run_c3.sbatch`；划分 `data/splits/`；原始 dump 与特征 gitignored）；训练产物 `results/slurm_c3/c3_v1/`（65/65 run 有限）；评估 `experiments/C3/c3_analysis.json`；图 4 张 ×2 格式；逐图说明 `experiments/C3/c3_figure_notes.md`。
- **记录**（2026-10-17 完成）：
  - **提取/训练自检**：重提 world 26 个 (split, layer) 组合 mean cosine ≥0.99993；`orig` 13 层 MSE 与历史 std13 summary **13/13 逐位一致**（L2 157.571 / L11 167.041 / L35 204.907），无管线错配。
  - **原任务总体是平的**：world valid（130）union−orig L2 **−18.4 MSE**（item CI [−56.5,+15.5]）、L11 **+5.9**（item CI [−49.0,+61.7]），均跨 0。L2 是事后选层（同一 130 条），union 139.2 仍高于国家均值 oracle 113.8；L11 课程口径 union 172.9 vs orig 167.0。
  - **分层解释"平"**：seen 110 条 L11 union 比 orig 差 **+27.4**（CI 跨 0）；unseen→C3 12 条 **−143.1**（item CI [−303.4,−25.1]，cluster [−303.2,−25.8]）；unseen both 8 条 L2 −76.1（CI 不跨 0）。⇒ 增广收益=覆盖新国家，代价=稀释已见国家，总量抵消。
  - **新任务**：Geo test L11 new 64.0 / union 62.3 / orig 147.0；union−new **−1.7**（CI 跨 0）⇒ 500 条原数据不伤害新分布，new 单独已足够。Geo valid union 81.4 vs new 87.8（−6.4，CI 上界 −0.0）。
  - **strict unseen 是最稳健收益**：L11 orig 93.8 / new 97.4 / union 75.7；union−new **−21.7**（item [−25.1,−18.5]，country-cluster [−41.9,−7.8]），hard6 −22.7（cluster [−47.6,−4.5]）；但 union−orig 的 cluster CI 跨 0（8 国太少），不能写成"显著优于原模型"。L32 union 57.4 vs new 64.0，cluster CI 跨 0。
  - **full unseen 是陷阱**：orig L11 56.5 看似最好，实则 22/30 国已在 world train；该面板不能回答未见国家泛化。
  - **逐国高度异质**：strict8 上 union 在 Costa Rica/Palestine 等变好，Denmark/Hungary/Lebanon/Niger/Yemen 上 orig/new 更好 ⇒ 报告须给逐国表，不能只报均值。
  - **泄漏控制可忽略**：noleak−full |ΔMSE| <1（L11 +0.55，0.3%），不是背答案。
- **结论**：GeoNames 增广不是免费午餐——在原 world 分布上总体无显著收益（L11 +5.9），收益集中于 world 未覆盖国家（unseen→C3 −143.1；strict8 上 union 比 GeoNames-only 低 21.7 MSE，且 cluster CI 不跨 0），代价是 seen 国家略变差；混合训练（union）优于用新数据替换旧数据（new 在原任务 L11 +58.2）。报告应同时给 L2（事后）与 L11（预注册）、item 与 country-cluster CI，并把 full unseen 的 22/30 重叠写进 limitation。

### 附录 A6 L1 稠密性定量（world L11 + truth L20/L23，附加实验）

- **来源**：《可做实验清单》A6（指南1 §4.3.2；指南称"λ1=0.01 时仍有 1860 个非零系数、表征稠密、L1 全程有害"）；2026-10-17 指派执行。属**附加探究**。问题：线性读出是否稠密？L1 能在多少性能损失下稀疏化？"L1 全程有害"在本课程协议下是否成立？
- **设计**：三个探针（world L11 线性 λ2=1；truth L20/L23 logistic λ2=1e-3），特征 train z-score，λ1 取 14 点几何网格（1e-5 → ≥λ1_max）。用 **FISTA + 软阈值**（proximal，沿 λ1 递减 warm start）求解，而非 `model.py` 的 sign 次梯度——后者在零点附近不会产生精确零。λ1_max 解析求（w=0 最优的最小 λ1）。**valid 只报告、不参与选择**。非零按行（world 一行 = 一对经纬权重，任一非零即计）与元素两种口径；另报有效维度 $(\sum|w|)^2/\sum w^2$。目标函数与 `model.py` 一致（线性按单目标 1/N 的 Frobenius 和，logistic 为 1/N 交叉熵），λ2 固定以隔离 L1 效应。
- **代码/资产**：`experiments/A6/a6_l1_density.py`（`--recompute-metrics` / `--replot`）；`a6_results.json`（3 任务 × 14 点 + 权重）、`a6_l1_density.csv`、`a6_weights.npz`、`a6_run.log`（本地 CPU **366.4 s**，42/42 `converged=True`）；图 `figures/a6_{density_curves,metric_vs_l1,coef_structure}.{png,pdf}`；逐图说明 `a6_figure_notes.md`。
- **记录**（2026-10-17 完成）：
  - **λ1_max / Gram λmax**：world **91.4091** / 139,309.9；L20 **0.225972** / 726,381；L23 **0.187965** / 945,495。
  - **"稠密"成立**：world λ1=0.01（0.011% λ1_max）仍有 **2559/2560 行（99.96%）**、5119/5120 元素非零，只压掉 1 行。50% 行归零需 world λ1≈**1.179**（1.29% λ1_max）、L20 **7.71e-4**（0.341%）、L23 **7.31e-4**（0.389%）；10% 行需 world **9.787**（10.7%）、L20 **8.87e-3**（3.93%）、L23 **9.45e-3**（5.03%）。⇒ 表征稠密但可稀疏化，truth 比 world 更易稀疏。
  - **world 性能**：valid MSE 锚点 167.1707；λ1=0.03 **166.715**、0.1 **165.896（最低；2466 行 / 4008 元素，中位 904.2 km，平均 1205.0 km）**、0.3 166.819、1 **183.486**、3 242.641、10 496.872、30 1356.617、50 2082.783、λ1_max **3311.269**（截距-only，中位 7299.7 km）。改善幅度仅 −0.76%，是描述性浅收益，未做配对检验。
  - **truth L20**：锚点 acc 0.8667 / logloss 0.4835；λ1=0.001（1147 维）0.8700/0.3247；0.003（609 维）0.8667/**0.2747**；**0.01（217 维）0.8900/0.2948**；0.03（72 维）0.8667；0.05（45 维）0.8533；0.1（15 维）0.8300；0.15（6 维）0.7667；λ1_max 处 1 维 → acc 0.5/log2。**L23**：锚点 0.85/0.4865；0.001（1111 维）0.8633/0.3275；**0.003（562 维）0.8567/0.3150**；0.01（241 维）0.8400；0.03（87 维）0.8433；0.05（55 维）0.8033；0.1（20 维）0.7867；0.15（11 维）0.7767。⇒ 用 **8–24% 的维度**即可保持或略超锚点 acc，logloss 改善远大于 acc（0.4835→0.2747，−43%）。
  - **有效维度**：world 1635.8（λ1→0）→1254.1（0.1）→472.0（1）→59（30）；L20 1626.9→107.5（0.01）→22.7（0.05）；L23 1595.3→126.2→35.2。名义 2560 维的有效参与度仅 ~1600，L1 可把它压到几百而不伤泛化。
  - **求解器对拍**：world λ1=0.01 时 sign 次梯度 **5120/5120 元素非零**（5109 个 >1e-6，train MSE 92.929），proximal **2559 行 / 5119 元素**（train MSE 92.086）⇒ 密度统计必须用近端/坐标下降。
  - **边界伪影**：λ1=λ1_max 时 truth 权重 ~1e-16、p≈0.5，浮点噪声会让 `p>=0.5` 产生虚假 acc；已用 `np.round(p,12)` 修正为 0.5 / log2（不影响非边界点）。
- **结论**：world/truth 的线性读出都是**稠密但可稀疏化**：world 在 λ1≈0.1（0.1% λ1_max）保留 96% 的行、valid MSE 浅降 0.76%；truth 在 λ1≈0.003–0.01 保留 217–609 维（8–24%），acc 持平或 +2.33pp、logloss −43%。指南"L1 全程有害 / 1860 非零"依赖其归一化与求解器，不能原样引用；可写进报告的结论是"温和 L1 有窄的正则化窗口，强 L1 迅速崩溃，可用区间 world λ1≲0.3 / truth λ1≲0.03"。

### 附录 A9 world 训练/验证集地理分布（附加实验）

- **来源**：《可做实验清单》A9（指南2 §2）；2026-10-17 指派执行。属**附加探究**。问题：world train/valid 覆盖全球多少、集中在哪里？valid 是否只是 train 的近邻子集？
- **设计**：读取 `datasets/world/{train,valid}.csv`（500/130 条），统计国家标签计数、Gini/HHI/top-k 份额；haversine（R=6371 km）最近邻（train→train 去掉自身、valid→train）；10°×10° 网格（18×36=648 格）占用率，面积权重 cos(中纬)；纬带密度按 cos(lat) 归一；Lorenz 曲线。国家标签含 13 个多国组合（如 "Germany / Switzerland / Austria"），按原标签统计，不是严格国别。纯描述性，无训练/选择。
- **代码/资产**：`experiments/A9/a9_geo_distribution.py`；`a9_results.json`；图 `figures/a9_{maps,country_counts,coverage}.{png,pdf}`（cartopy 110m 海岸线，**无国界线**）；逐图说明 `a9_figure_notes.md`。
- **记录**（2026-10-17 完成）：
  - **国家覆盖**：train **158** 个标签 / valid **78** 个；两者共有 **59**；valid-only **19 个标签 / 20 条**，train-only **99 个 / 187 条**。单样本标签 train **73** / valid **55**；≥5 条 train **26** / valid **4**。
  - **集中度**：国家计数 Gini train **0.520** / valid **0.325**；HHI **0.0208 / 0.0214**（有效国家数 ~48 / ~47）；top10 份额 **34.8% / 36.2%**；top1 **9.0%（US 45）/6.15%（US 8）**。Top train：US 45、Australia 20、China 20、Brazil 17、India 15、Mexico 15、Canada 13；Russia（10）、Japan（8）在 valid 为 0。
  - **valid 是近邻子集**：valid→train 最近邻中位 **187.7 km**、平均 235.5、P90 536.5、max **974.1 km**；**100% < 1000 km**、86.9% < 500 km、30.8% < 100 km。train→train 中位 136.0 km，仅略小于 valid→train ⇒ valid 不是"更难的远距离外推"，而是与 train 同分布的内插集（呼应 Q7：同国最近邻中位 ~226 km）。
  - **空间覆盖**：10° 网格 train 占 **148/648（29.76% 陆地面积权重）**、valid **82/648（16.66%）**；占用格 Gini **0.438/0.274**、有效格数 **84.2/58.3**、top5 份额 12.4%/**18.5%**。北半球占比 **73.4% / 70.0%**；train 密度峰值 45–55°N（113.1 样本/cos 单位，80 条）与 35–45°N（105.0，86 条），valid 峰值 45–55°N（29.7，21 条）与 15–25°N（29.0，28 条）。
  - **地图读法**：valid 点几乎全部落在 train 覆盖区内，欧洲/东亚/北美/澳洲最密；撒哈拉以南非洲、南美内陆、西伯利亚、北极圈基本空白。valid-only 的 19 个标签多为小国/组合标签，样本量 1–2 条。
- **结论**：world 划分是**稀疏、北半球、国家高度集中**的采样：~30% 的陆地面积权重承载 train，valid 更窄（16.7%）；valid 与 train 的近邻中位仅 188 km、全部 <1000 km。因此 world valid MSE 主要度量**同分布内插**，不能解释为地理外推能力；Q7/Q8 中"unseen 国家"的结论必须另设划分（如 C3 的 strict8）才成立。

### 附录 A8 优化器对照：GD vs Momentum vs Adam（附加实验）

- **来源**：《可做实验清单》A8（指南3 §4）；2026-10-17 指派执行。属**附加探究**。问题：在同一目标、同一初始化、同一学习率、同一步数下，全批量 GD、heavy-ball momentum（β=0.9）与 Adam 的收敛速度、稳定边界、最终指标差多少？课程 `model.py` 用纯 GD，本实验回答"换优化器会不会改变选层/正则的结论"。
- **设计**：world L11（线性，λ2=1.0）与 truth L20（logistic，λ2=1e-3）；特征 train z-score，λ1=0（见 A6）。目标与 `model.py` 一致：world 按单目标 1/N 的 Frobenius 和（两坐标分别对应一次单目标拟合），truth 为 1/N 交叉熵；表中 MSE 仍为**元素平均**口径，可直接对拍 A1 锚点 167.1707。全批量 torch float64、4 threads。学习率写成 $f=\mathrm{lr}/\mathrm{lr}_{crit}$，$\mathrm{lr}_{crit}=2/L$：线性 $L=2\lambda_{\max}([X,1])/N+2\lambda_2$，logistic $L=0.25\lambda_{\max}([X,1])/N+2\lambda_2$。稳定性扫描用 $w,b\sim\mathcal N(0,10^{-3})$（seed 1234）激发所有方向、600 步不早停；头对头用 `model.py` 的 $w=b=0$、$f=0.9$、≤8000 步、允许早停。参考最优：world dual ridge、truth L-BFGS-B（319 次迭代，$\|\nabla\|_\infty=3.2\times10^{-9}$）。**valid 只报告，不参与选择**。
- **代码/资产**：`experiments/A8/a8_optimizers.py`（`--replot` 只重画、`--smoke` 4 点冒烟）；`a8_results.json`（protocol/reference/sweep/init_probe/head + 逐步 history）、`a8_run.log`（本地 CPU **529.6 s**）；图 `figures/a8_{convergence,stability,final_metrics}.{png,pdf}`；逐图说明 `a8_figure_notes.md`。
- **记录**（2026-10-17 完成）：
  - **临界学习率**：world $\lambda_{\max}=139{,}309.94$、$L=559.2397$、$\mathrm{lr}_{crit}=3.5763\times10^{-3}$；truth $\lambda_{\max}=726{,}381.04$、$L=35.6929$、$\mathrm{lr}_{crit}=5.60336\times10^{-2}$。world 参考 valid **MSE 167.17072 / 中位 923.12059 km / 平均 1203.0653 km**（与 A1 逐位一致）；truth 参考 valid **acc 0.863333 / logloss 0.484373**。
  - **头对头（$f=0.9$）**：world 到 $10^{-2}/10^{-3}/10^{-4}$ 的步数——momentum **40/70/90**（501 步早停）、GD 150/320/500（2335 步早停）、Adam 3030/5270/6550（跑满 8000，gap 8.8e-7）；三者 valid MSE 均 167.17（Adam 167.14），中位 923.12 / 923.12 / **926.89** km。truth——Adam **80/120/260**、momentum 350/770/1370、GD 3500/7850/未到 $10^{-4}$；momentum/Adam 收敛到 acc 0.8633 / logloss 0.4844，GD 8000 步仍有 gap 9.3e-4，valid acc 0.8667 / logloss 0.4638 是**早停效应**，不是更优解。
  - **稳定边界**：world 精确复现理论——GD 在 $f=0.9$ 稳定（gap 2.45e-5）、$f=1.0$ 边缘（1.44e-2）、$f=1.1$ 起发散（DIV@51）；momentum 到 $f=1.5$ 全 0、$f=1.9$ 边缘（5.19）、$f=2.2$ 起发散（DIV@13）；Adam 到 $f=3$ 未发散（gap 0.171→0.020，单调下降）。⇒ GD 边界 $f\approx1$、momentum $f\approx1+\beta=1.9$，是 $L$ 与 heavy-ball 理论的干净对拍。
  - **logistic 的 $0.25$ 界偏保守**：GD/momentum 到 $f=3$ 都不发散且 gap 随 $f$ 单调下降（GD 0.252→0.027；momentum 0.088→1.62e-5）；**Adam 非单调**——$f=0.7$ 的 gap 3.57e-8，但 $f=0.9/1.3/2.2/3.0$ 分别 0.291/2.28/0.215/1.17（有界震荡，不是 inf 爆炸）。⇒ 不能把 $2/L$ 当 logistic 的硬发散阈值，也不能把"未发散"当"收敛"。
  - **初始化探针**：world 在 $f=1.5$ 时零初始化与 $\sigma=10^{-3}$ 随机初始化都在**第 15 步发散**（零点梯度 $X^\top y$ 沿顶部特征向量有分量）；truth 在 $f=1.5$ 两种初始化都稳定（gap 0.0624）。⇒ world 的边界不是零初始化造成的假象。
  - **口径修复**：首次冒烟发现 world 两输出时 `torch.mean` 对 $N\times2$ 元素平均 = $1/(2N)$，而 `model.py` 是单目标 $1/N$（向量化为 Frobenius 和），导致数据项减半、有效 $\lambda_2$ 翻倍、$\mathrm{lr}_{crit}$ 失配（GD 在 $f=1.5$ 竟不发散）。已改为 `mean(sum(..., dim=1))`，上面所有数字均来自修正版；报告 MSE 仍用元素平均以对齐 A1。
- **结论**：凸问题上优化器**只影响速度、不影响最终解**——world 三组收敛后 valid MSE 都是 167.17，truth 两组与参考差 <1e-6。速度强烈依赖任务几何：world（病态二次型）momentum 最快（到 $10^{-4}$ 90 步 vs GD 500、Adam 6550），truth（logistic）Adam 最快（260 步 vs momentum 1370、GD >8000）。稳定边界在 world 上可被 $2/L$ 与 $2(1+\beta)/L$ 精确预测；logistic 的 $0.25$ 界偏保守，Adam 存在非单调有界震荡。报告应写"按任务几何选优化器"，不能写"某个优化器全面更优"；truth GD 的 86.67% 必须标注为早停效应。

### Q4 – Q9

见 §2 总览；做到某题时在此追加小节（目标 / 数据 / 代码 / 结果 / 结论）。

## 4. 变更日志

- 2026-10-11：创建 `question/{data,code}`；D1 落盘 + 来源对拍（§1.2）；建立本文件。
- 2026-10-11：Q1 系数路径复现完成（Ridge/Lasso 图与数值表，见 §3）。
- 2026-10-11：Q2 完成：阈值化 ridge 反例（5 图 + bootstrap 稳定性 + 预测代价），§3 含原因解释。
- 2026-10-11：Q3 完成：与 sklearn 全口径对比（world 7 行 + truth 4 行 + 3 图；含正则换算与发散演示），结论：对齐目标函数后同解、差异归于优化器/收敛预算/换算口径，见 §3。
- 2026-10-13：新增 Q2 逐图解释说明 `figures/q2_figure_notes.md`（约定：此后每批图都配 `<题号>_figure_notes.md`，格式同）。
- 2026-10-13：新增 Q3 逐图解释说明 `figures/q3_figure_notes.md`（收敛/权重一致/valid 条形 3 图）。
- 2026-10-13：Q6 完成：最优 world 探针（raw L11）验证集预测地图（`figures/q6_valid_map.{png,pdf}`，cartopy 110m 海岸线、无国界线；本地推理无需集群）；复算 MSE=150.4277 与训练锚点一致；逐图说明 `figures/q6_figure_notes.md`。
- 2026-10-14：Q7 完成：坐标表征 vs 国家识别（三层证据 + oracle 国家均值/全局均值双基线 + paired bootstrap CI）；unseen 20 条 +1024 大胜全局基线、seen 上 oracle 反超 −35、同国 lat r=0.20 弱显著——结论定性为"中等强度证据"；3 图 + 逐图说明 `figures/q7_figure_notes.md`（含论文 [3] §4.1/§5 对应）。
- 2026-10-16：Q8 完成：分布差异四维统计（typo 3.5×、主体新颖 77.4% vs 零假设 20.8%、模板/长度、world 对照）+ L20 分层稳健性（差异不显著、浅层怕错拼）+ CV 对照（随机 vs 分组 −0.8pp，负结果如实报告）；3 图 + 逐图说明 `figures/q8_figure_notes.md`。
- 2026-10-16：**B1 完成**（CREAK 逐句对拍，可选实验）：truth 100% 可追溯（direct 86.08%/63.33% 与 guide3 一致，near 层补全到 100%）、valid 来源 = dev/contrast/test 与 train.json 零重叠、标签 0 冲突、扰动仅两族（错拼 415 + 主谓翻转 431 处，无第三类）；Q8 错拼检出来源还原（注入 307/354 与 65/74、自带专名假阳性 40/6）。产物 `experiments/B1/`（脚本 22 锚点 + 4 图 + `b1_figure_notes.md` + `b1_metrics.json`）；Q8 小节"范围说明"同步更新。
- 2026-10-16：**C1 完成**（附加实验：Qwen3-4B-Base 直接输出坐标 vs L11 探针）：集群 A100 跑 630 题（严格解析 99.8%/99.2%、宽松 100%）、valid 距离中位 43.8 km（探针 953.8 的 22×）、逐条 119/130 直接更近（+982.5 km CI [835, 1139]）；原始 MSE 被单条日界线样本主导（Tomanivi 96.5%，周期口径 17.6）；与参考报告（非官方）逐项一致（hit100 相同）。产物 `experiments/C1/`（评估脚本 + 3 图 + `c1_figure_notes.md`）与 `results/slurm_qwen_direct/qwen_direct_v1/`。
- 2026-10-16：**A1–A3 完成**（线性读出上限包，附加实验）：A1 多层拼接 world **L5+L32 128.82**（−18% vs 最好单层，低于历史提交 150.43）/ truth L17+L20 0.8833；A2 sklearn 家族仅 world-L11 KNN（140.70）超闭式 ridge，truth 全家族 0.85–0.877（SVC-RBF 0.8767 ≈ 参考报告 0.87）；A3 自训 MLP 均值均未超闭式锚（单 seed 可及但方差大，linear-AdamW 对照欠拟合）。⇒ 表征 + 正则对齐 > 模型族，线性读出已近上限。产物 `experiments/A{1,2,3}/`（3 脚本 + 3 结果 JSON + 6 图 + 3 逐图说明）。
- 2026-10-17：**A7 完成**（5 折 CV 协议补做，附加实验）：train-only 5 折 × 3 重复、估计器同 A1。truth CV↔valid 秩相关 **0.968**、CV 与 valid 同选 **L20**（regret 0，15 折中 L20 赢 11）⇒ valid 选层可复现；world ρ=0.703、CV 最优 L8 vs valid 最优 L2 但头部不可分辨（valid regret 仅 **+1.29 MSE**）⇒ 应按 L2/L8 平台报告；world CV 均值 MSE 高于 valid 系离群点尾部（典型误差一致：L2 中位 SE 47.1/45.1、中位距离 988/978 km；top-5 占 SSE 40.6%）⇒ 均值 MSE 对划分敏感；train 指标不能选层（world ρ=−0.25 ns、truth train acc 饱和）。产物 `experiments/A7/`（脚本 + 结果 JSON + CSV + 4 图 + 留档日志 + 逐图说明）。
- 2026-10-17：**C2 完成**（prompt 变体特征 + 探针，附加实验）：5 变体 × 13 层 std13（raw11 辅助，nameonly raw11 发散）。std13-L11：删国家名 MSE 539→1069、中位 2280→3115 km（+1005 km CI [624, 1391]）、去指令仅 +233 km、换问法 −199 km（CI 上界 +0.3）、读点国家 token→句末 "?" +1511 km；国家名效应集中在 seen 国家（+1127 vs unseen +335，CI 跨 0）⇒ 与 Q7 反向闭环。锚点 13/14 逐位一致（L35 +0.77 为 float16 数值差，raw11 −0.10）。产物 `experiments/C2/`（脚本 + 5 变体 CSV + 3 图 + `c2_figure_notes.md`）与 `results/slurm_c2/c2_v1/`。
- 2026-10-17：**A10 完成**（球面损失 θ² vs MSE 控制组，附加实验）：world L11 探针，train-only 5 折 × 3 seed 选参/早停（valid 只用一次）。球面组 valid 平均距离 **1396 vs MSE 控制 1977 km（−29.4%）**、中位 1157 vs 1718（−32.7%）；但 **10/130（7.7%）** 输出落反向纬度分支（OOF 34/500），raw MSE **1801 vs 357**（5.0×），分支规范化后 **370 vs 357**、周期 MSE 290 ⇒ 恶化主要来自坐标表示而非 3D 位置；9/10 反射集中在澳洲—大洋洲。hybrid β=0.5 零反射（valid 240 MSE / 1187 km 中位），但 OOF MSE 485 vs 472（+2.6%）未复现 valid 的 −33%；球面 lr=3e-3 塌到 43.5% 反射（非凸证据）；λ 不可分辨；ridge 167/923 仍全场最优。产物 `experiments/A10/`（脚本 + 结果 JSON + 预测/权重 NPZ + 5 图 + `a10_figure_notes.md`）。
- 2026-10-17：**C3 完成**（GeoNames 数据扩增，附加实验）：12,000 城 / 30 国整国留出；13 层重提 + 5 训练集 × 4 评估集。原 world valid 总体不显著（union−orig L2 −18.4 / L11 +5.9，CI 跨 0）；收益集中于 unseen（L11 −143.1，CI 不跨 0）；strict8 上 union 比 new 低 21.7 MSE（cluster CI [−41.9,−7.8]）但逐国异质；new 单独训练伤原任务（L11 +58.2）⇒ 混合优于替换；full unseen 的 22/30 国重叠是陷阱。产物 `experiments/C3/`（4 图 + `c3_figure_notes.md` + `c3_analysis.json` + 评估脚本）与 `results/slurm_c3/c3_v1/`（65 runs）。
- 2026-10-17：**A9 完成**（world 地理分布，附加实验）：train/valid = 500/130；国家标签 **158/78**（共有 59、valid-only 19 个/20 条、train-only 99 个/187 条）；Gini **0.520/0.325**、top10 34.8%/36.2%；valid→train 最近邻中位 **187.7 km**、P90 536.5、**max 974.1 km（100% <1000 km）**，train→train 中位 136.0 km ⇒ valid 是同分布内插集而非地理外推；10° 网格 train 占 **29.76%** 面积权重、valid **16.66%**；北半球 73.4%/70.0%。产物 `experiments/A9/`（脚本 + 结果 JSON + 3 图 + `a9_figure_notes.md`）。
- 2026-10-17：**A6 完成**（L1 稠密性定量，附加实验）：FISTA+软阈值（λ1 递减 warm start，λ1_max 解析）；world L11 / truth L20 / L23。world λ1=0.01（0.011% λ1_max）仍 **2559/2560 行非零**；50% 行归零需 λ1≈**1.179**（1.29% λ1_max），10% 需 **9.787**（10.7%）；truth 更易稀疏（50% 只需 0.34–0.39% λ1_max）。world valid MSE 锚 167.1707，λ1=0.1 时 **165.896（−0.76%，2466 行）**，λ1≥1 迅速崩溃；truth L20 在 λ1=0.01（217 维）valid acc **0.89** / logloss **0.2948**（锚 0.8667/0.4835），L23 在 λ1=0.003（562 维）logloss 0.3150。⇒ "稠密但可稀疏化；温和 L1 有窄窗口，强 L1 有害"，指南"L1 全程有害/1860 非零"依赖归一化与求解器，不能原样引用；sign 次梯度在 λ1=0.01 仍 5120/5120 元素非零（密度统计必须用近端法）。产物 `experiments/A6/`（脚本 + 结果 JSON/CSV/权重 NPZ + 3 图 + `a6_figure_notes.md`）。
- 2026-10-17：**A8 完成**（优化器对照，附加实验）：world L11 线性 + truth L20 logistic，同目标/同 lr（$f=0.9$）/同初始化/≤8000 步。world 到 $10^{-4}$：momentum **90** 步、GD 500、Adam 6550；truth：Adam **260**、momentum 1370、GD 8000 未到 ⇒ 速度强烈任务依赖。稳定边界：world GD $f\approx1$、momentum $f\approx1+\beta=1.9$（精确复现 $2/L$ 理论）；truth 的 $0.25$ 界偏保守（GD/momentum 到 $f=3$ 不发散），Adam 在 $f=0.9/1.3/2.2/3.0$ 有界震荡（gap 0.29/2.28/0.22/1.17 vs 0.7 处 3.6e-8）。三组收敛后 valid 指标相同（world MSE 167.17、truth 0.8633/0.4844）⇒ 优化器只影响速度、不影响凸问题的解；truth GD 的 86.67% 是 8000 步未收敛（gap 9.3e-4）的早停效应。修正 world 两输出的损失归一化（`torch.mean` 的 $1/(2N)$ → `model.py` 的 $1/N$ Frobenius 和）。产物 `experiments/A8/`（脚本 + 结果 JSON + 日志 + 3 图 + `a8_figure_notes.md`）。
