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
| Q8 | truth 训练/验证集分布差异 | 未做 | datasets/truth | — | — | 两份现有报告均未覆盖 |
| Q9 | truth 选层 + 正则化 | 素材齐 ✅ | datasets/truth | results/slurm_layers/* | truth_layers.png | L20 峰值（0.8967） |

Q4/Q5/Q9 素材位置：`results/slurm_layers/LAB1 分层训练结果分析报告.md`、`results/slurm_grid/20261009_115559_88653/LAB1 超参数扫描实验报告.md`、`experiments/稳定界说明.md`。

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
