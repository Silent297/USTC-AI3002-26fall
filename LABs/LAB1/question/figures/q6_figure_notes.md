# Q6 逐图解释说明

> 题目：给出你的模型在验证集上的预测结果，并可视化在地图上（推荐不含国界线）。
> 本文档记录 1 张 `q6_*` 图：**画的是什么 / 关键读数 / 证明了什么 / 报告怎么用**，供写 LaTeX 报告时直接取材。
> 读数口径：模型 = world raw L11 历史最优探针（`results/slurm_layers/world/weights/world_layer11_extra0/weights.npz`，standardize=false —— 权重已折回原始特征空间，推理时**不再 z-score**）；pred = X·wᵀ + b，目标列序 (latitude, longitude)；MSE = 两坐标平方误差平均（与训练管线 `train23.load_split` 一致）；图上误差着色 = 单点欧氏定位误差 √(Δlat²+Δlon²)（度）。全部读数由 `code/q6_valid_map.py` 打印、可一键复现。
> 维护约定：每完成一批绘图，在 `figures/` 下新建一份 `<题号>_figure_notes.md`，格式同本文与 `q2_figure_notes.md`、`q3_figure_notes.md`。

## 总览

| 图 | 一句话 | 回答的内容 |
|---|---|---|
| `q6_valid_map` | 130 个验证样本：灰空心圆 = 真值、彩点 = 预测（误差着色），叠加 110m 海岸线 | 最优模型在全球尺度定位到什么精度、误差集中在哪 |

一句话主线：**模型学到了"大洲级"定位（中位误差 ~9.5° ≈ 千公里量级），且纬度远比经度准**；最大失败集中在高纬（南极洲）与孤立岛屿。

---

## 1. `q6_valid_map.*` — 验证集预测地图

**画的是什么**：全球 PlateCarree 地图（cartopy，110m 海岸线；**合规：不画国界线**）。灰空心圆 = 130 个验证样本的真值位置 (lon, lat)；彩色实心点 = 模型预测位置，颜色 = 单点定位误差 √(Δlat²+Δlon²)（viridis 色条 0–90°）；淡色经纬网格辅助读数。

**关键读数**：
- **复算锚点**：valid MSE = **150.4277**，与 `results.json` 历史最优记录 **逐位一致**（150.42772354798632）——权重折回、列序 (lat,lon)、dtype 三处口径全部正确。
- R² = **0.9545**（经纬度联合）。valid 共 130 条、覆盖 78 个国家。
- 单点定位误差：平均 **12.44°**，四分位（25/50/75）= **4.96 / 9.52 / 15.34°**；>30° 的仅 **7/130**。
- **纬度远比经度准**：per-coordinate MAE = **纬度 4.25° vs 经度 10.85°**（约 2.5 倍）。
- 最准样本：Denmark **0.79°**、Kenya 1.01°、Honduras 1.05°（欧洲、东非、中美等覆盖密集区）。
- 最差样本 top-3：
  1. **Antarctica 89.59°**：(−62.96, −60.64) → (−71.41, **28.55**)，两坐标同时失败；
  2. **Antarctica 73.79°**：(−77.50, 106.00) → (−62.12, **33.83**)，与上一条落到相近的错误经度（~28–34°E 的南大洋）；
  3. **Canada 40.74°**：(49.58, −57.81) → (49.35, −98.54)：**纬度几乎完美（Δ0.23°），经度差 40.7°**（纽芬兰 → 加拿大中部）。
- top-10 最差的国家构成：南极洲 ×2、加拿大 ×2、毛里求斯、牙买加、爱沙尼亚/俄罗斯、澳大利亚、危地马拉、中国。

**证明了什么**：
1. **口径闭环**：本地推理复算与训练记录完全一致——图为"可信可视化"，为 Q7/Q8 的讨论提供基准。
2. **模型学到了什么**：R² = 0.955 说明线性探针确实从 L11 表示中读出了"句子 → 地理位置"的回归信号；但中位误差 9.52°（约千公里）说明精度是**大洲级**而非国家/城市级——与 world 训练目标（线性回归坐标）的能力边界一致。
3. **误差结构——方向不对称**：top-10 中 5 条纬度误差 <5°、误差几乎全部由经度贡献（Jamaica Δlat 0.01° / Δlon 33.1°；Canada-纽芬兰 Δlat 0.23° / Δlon 40.7°；Mauritius Δlat 1.35° / Δlon 38.0°）。结合整体 MAE（4.25° vs 10.85°）：**模型对"气候带式"的纬度线索敏感（tropical/desert/snow 等词面信息），对需要具体地名知识的经度定位明显更弱**。例外：南极洲 2 条（valid 中南极样本共 2 条，全部进了 top-2）与澳大利亚 1 条为两坐标同时偏离。
4. **样本覆盖决定精度**：覆盖密集区（欧洲/东非/中美）误差 <5°；高频难例集中在样本极端稀少区域（南极洲、加拿大北部、孤立岛屿）。
5. 绘图细节：一个预测点经度 180.04° 超出 [−180, 180]，按环绕惯例显示于地图左缘，不影响任何统计量。

**报告怎么用**：Q6 正文插图；正文引 150.43 / R² 0.955 / 中位 9.5° / 纬度-经度不对称；讨论段可用南极洲失败案例佐证"训练分布覆盖"对定位精度的影响（Q7/Q8 伏笔）。
图注候选（中文）：*"world 最优模型（raw L11 线性探针）对 130 个验证样本的定位：灰空心圆 = 真值，彩点 = 预测（颜色 = 单点定位误差 √(Δlat²+Δlon²)，单位度）。valid MSE = 150.4 deg²（R² = 0.955）；中位误差 9.5°、纬度误差显著小于经度；高纬（南极洲）与孤立岛屿样本误差最大。仅绘海岸线（110m），无国界线。"*
图注候选（English）：*"Geo-localization of the 130 valid sentences by the best world probe (raw layer 11). Hollow gray circles: ground truth; colored dots: predictions, shaded by the per-point positioning error √(Δlat²+Δlon²) in degrees. Valid MSE = 150.4 deg² (R² = 0.955); errors concentrate at high latitudes and isolated islands. Coastlines (110m) only; no national borders drawn."*

---

## 附：相关数字的出处

- 生成脚本：`code/q6_valid_map.py`（一条命令复现全部读数：打印 MSE / R² / 误差分位数 / per-coordinate MAE / top-10 最差样本及国家）。
- 权重：`results/slurm_layers/world/weights/world_layer11_extra0/weights.npz`；锚点出处：`results/slurm_layers/world/results.json`（metric 150.42772354798632，extra L11 配置）。
- 数据：`datasets/world/valid.layer11.npy`（130×2560）+ `datasets/world/valid.csv`（真值 lat/lon、country）。
- 图：`figures/q6_valid_map.png`（dpi=150）+ `q6_valid_map.pdf`，2026-10-13 生成（cartopy 0.26.0，Natural Earth 110m 海岸线）。
