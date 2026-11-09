# Lab2 实验步骤

> 课程：人工智能与机器学习基础（2026 秋）
> 内容：第一题 神经网络拟合与梯度传播；第二题 无监督聚类
> 原则：先保证代码正确 → 小规模验证 → 正式运行 → 分析成文；每阶段有明确验收门。

## 0. 协作方式

- 作业代码由本人完成；助手负责方案设计、接口核验、结果审查与报告结构建议，不代写实现。
- 每完成一个阶段，提交运行结果或日志，再进入下一阶段。
- 所有正式产物（输出目录、日志、源码、环境记录）本地保留至验收结束，不随提交上传。

## 1. 环境基线

| 项 | 本机 | 说明 |
|---|---|---|
| Conda 环境 | `ai26`，Python 3.12.14 | 可用 |
| PyTorch | 2.14.1+cpu | 仅 CPU |
| CUDA | 不可用 | Q1 本地跑；Q2 正式跑上集群 |
| 依赖 | numpy / scipy / scikit-learn / matplotlib / threadpoolctl 齐全 | pytest 缺失，需补装 |
| CPU | 32 核 | 本地只做小规模验证与计时 |

注意事项：

- 不要在本机执行 `pip install -r code/requirements.txt`，该文件把 torch 固定为 2.11.0，会替换现有 2.14.1+cpu。只补装 `pytest>=8`。
- 集群侧使用 `code/requirements-gpu.txt`（torch 2.11.0+cu128），GPU 自检必须在计算节点内执行。
- 报告中记录：设备（CPU/GPU 型号）、Python/PyTorch 版本、线程数、精度。

本地补装命令（Git Bash）：

```bash
~/miniconda3/envs/ai26/python.exe -m pip install "pytest>=8"
```

## 2. 阶段总览

| 阶段 | 内容 | 预计耗时 | 关键验收门 |
|---|---|---|---|
| 0 | 环境与数据基线 | 0.5 h | pytest 可启动、数据哈希通过 |
| 1 | Q1 代码补全与自检 | 2–3 h | pytest 通过、参数量正确、loss 可降 |
| 2 | Q1 两组正式实验 | 0.5 h + 检查 | 4 组 MSE 有限、曲线下降、图完整 |
| 3 | Q1 图表与 6 个分析题 | 2–3 h | 每题有读数与图支撑 |
| 4 | Q2 代码补全与分项自检 | 3–4 h | algorithms/metrics 自检、float32/64、pytest |
| 5 | Q2 小规模验证与计时 | 0.5–1 h | 玩具数据全链路跑通、耗时外推 |
| 6 | Q2 正式实验（集群） | 2–4 h | manifest 完成、check-output 通过 |
| 7 | Q2 指标对比与 T1/T2 | 2 h | 指标表、选择依据、现象分析 |
| 8 | 报告与提交打包 | 3–4 h | 结构正确、默认实现可复现 |

依赖关系：0 → 1 → 2 → 3；0 → 4 → 5 → 6 → 7；3、7 完成后进入 8。

---

## 阶段 0：环境与数据基线

**动作**

1. 本地 ai26 环境补装 pytest，确认 torch 未被替换。
2. 记录 Python/torch/设备/线程数。
3. 核验第二题数据哈希：`cd code/q2_clustering && sha256sum -c data/SHA256SUMS.txt`。
4. 核对第一题数据：训练 1024 点，网格 65×65=4225 点，float32。
5. 集群侧确认 `ai26` 环境与 GPU 依赖，提交一个小作业做 CUDA 自检。

**验收门**

- [ ] `pytest --version` 可用，`torch.__version__` 仍为 2.14.1+cpu
- [ ] 数据哈希全部 OK
- [ ] 环境表写入报告草稿

**产出**：环境记录、数据校验结果。

---

## 阶段 1：Q1 代码补全与自检

**动作**

1. `code/q1_neural_network/models.py`
   - TODO 1：构造 `hidden`（`nn.ModuleList`，带偏置 Linear）、`activation`（ReLU/Sigmoid）、`output`（→1）。
   - TODO 2：`forward` 依次过隐藏层+激活，最后线性输出。
2. `code/q1_neural_network/train.py`
   - 在循环内实现一次完整更新：清零梯度 → 前向 → MSE → 反向 → 参数更新；使用全部 `x`、`y`。
   - 保留已有的记录逻辑与最后一次 `break`。

**接口约束（已核实初值文件）**

- state_dict 键名：`hidden.0.weight/bias` … `hidden.N.weight/bias`、`output.weight/bias`。
- 参数量：
  - expression：A=65、B=65、C=257、D=65
  - gradient：A/B/C/D 均为 7521
- 前向输出形状 `(B, 1)`，dtype 为 float32。

**验收门**

- [ ] `python -m pytest -q` 三个结构用例通过
- [ ] 四组初值均可 `load_state_dict(..., strict=True)`
- [ ] 参数量与上表一致
- [ ] 固定小批量上执行一次更新后 loss 下降、参数发生变化
- [ ] 无 NaN/Inf

---

## 阶段 2：Q1 正式实验

**动作**

```bash
cd code/q1_neural_network
python main.py --task expression --device cpu
python main.py --task gradient --device cpu
```

- 实验 A（expression）：Adam，lr=0.01，4 组
  - A `[16]` 3000 步；B `[16]` 9000 步；C `[64]` 3000 步；D `[8,4]` 3000 步
- 实验 B（gradient）：SGD，1000 步，4 组
  - A 2×32 Sigmoid lr=0.05；B 8×32 Sigmoid lr=0.05；C 8×32 ReLU lr=0.05；D 8×32 Sigmoid lr=0.1

**产物**：每组指标、训练记录、预测、权重，以及 `fitting.png`、`training.png`、`gradients.png`。

**验收门**

- [ ] 两组实验各 4 组结果齐全
- [ ] MSE 均为有限值，训练/网格 MSE 曲线整体下降
- [ ] 梯度实验含每层激活前梯度 RMS 与相对输出端比例
- [ ] 记录每组最终 train/grid MSE 与耗时

**排错规则**

| 现象 | 优先检查 |
|---|---|
| loss 不下降 | 梯度清零、`optimizer.step()`、优化器类型 |
| loss 为 NaN/发散 | 学习率、初始化、是否重复反向 |
| 图缺失 | `diagnostics` 开关、输出目录是否为空 |
| 形状报错 | `hidden` 层顺序与 `output` 输入维度 |

---

## 阶段 3：Q1 结果整理与 6 个分析题

**实验 A**

- (a) 比较 A、D 的 MSE 剖面图，判断哪组拟合更好；说明 MSE 绝对值小是否足以证明拟合良好。
- (b) 解释双隐藏层为何更适合本题；用 ReLU 与线性运算表示目标函数（提示：`|t| = ReLU(t) + ReLU(-t)`）。
- (c) 比较 B、C 相对 A 的提升及增加的成本；判断本题中哪种改动更有效及原因。

**实验 B**

- (a) 比较 A、B 的梯度指标，判断哪组存在更明显的梯度消失；讨论只看 loss 能否察觉。
- (b) 比较 B、C，说明换 ReLU 后梯度与拟合的变化及原因。
- (c) 比较 B、D，判断提高学习率是否有效；说明学习率改变的是更新速度还是梯度本身；同初值时两组梯度是否相同。

**产出**

- 4 组结果表（参数量、步数、学习率、train MSE、grid MSE）
- 剖面/误差面/曲线/梯度图
- 每题一段结论，引用具体数值与图号
- 图注文件：说明“是什么 / 精确读数 / 结论 / 报告用法”

**验收门**

- [ ] 6 个小题全部作答，均有数据或图支撑
- [ ] 区分“实际拟合差异”与“MSE 数值大小”
- [ ] 图表编号与正文引用一致

---

## 阶段 4：Q2 代码补全与分项自检

**动作**

1. 先补五个算法函数：`init_centers_kmeanspp`、`assign_clusters`、`update_centers`、`gmm_estep`、`gmm_mstep`。
2. 再补三个评价函数：`silhouette_score`、`calinski_harabasz_score`、`davies_bouldin_score`。
3. 分项自检：

```bash
cd code/q2_clustering
python extras.py self-check --part algorithms --device cpu
python extras.py self-check --part metrics --device cpu
python extras.py self-check --device cpu
python extras.py self-check --device cpu --dtype float64
python -m pytest -q
```

**实现约束**

- 全部使用 PyTorch Tensor，禁止 NumPy 参与计算；不得调用现成聚类求解器/初始化器/评价函数。
- 新张量继承 `X.dtype` 与 `X.device`；不修改输入。
- 算法返回的 labels 为 `torch.int32` 或 `torch.int64`。
- 随机抽样传入 `generator=rng`。
- 指标在原始特征空间计算，即使拟合时开启标准化。

**验收门**

- [ ] `--part algorithms`、`--part metrics`、完整自检均通过
- [ ] float32 与 float64 均通过
- [ ] 六个测试文件通过
- [ ] toy 数据上与 sklearn 独立对拍三项指标（仅核验，不写入提交代码）
- [ ] 代码审查：无 NumPy 计算、无输入修改、类型/设备正确、边界行为符合 docstring

---

## 阶段 5：Q2 小规模验证与计时

**动作**

1. 玩具数据全链路（按 `FRAMEWORK_V0.md` 生成 `toy_features.npz` 与 `config_dev.json`）：

```bash
python extras.py run --device cpu --development \
  --input toy_features.npz --config config_dev.json --output outputs_dev
python extras.py check-output --development --input toy_features.npz --output outputs_dev
```

2. 在真实数据上做计时试点：`candidate_k=[10,15]`、`seeds=[17,42]`、`n_init=1`，单独输出目录 `outputs_pilot`。
   - 共 8 次拟合（2 K × 2 seed × 1 init × 2 方法）。
   - 测得总耗时 T，则正式规模约 96 次拟合（每方法 8 K × 2 seed × 3 init），总耗时约 12T。

**验收门**

- [ ] 玩具数据拟合→评价→选择→导出全链路成功
- [ ] `check-output --development` 通过
- [ ] 记录单次 KMeans/GMM 耗时与显存/内存占用
- [ ] 根据外推结果确定作业时长与 GPU 型号

---

## 阶段 6：Q2 正式实验（集群）

**动作**

1. 集群准备：ai26 环境、`requirements-gpu.txt`、计算节点内 CUDA 自检。
2. 推荐分阶段运行，便于换指标回放：

```bash
python extras.py self-check --device cuda
python extras.py run --stage fit --device cuda --config config.json --output outputs_fit
python extras.py evaluate --device cuda --experiment outputs_fit --output outputs_evaluated
python extras.py check-output --output outputs_evaluated
```

**正式网格**

| 项 | 基线（正式输出） | 可选消融（只进报告） |
|---|---|---|
| K | 5/8/10/12/15/20/25/30 | 缩减为 3 个 |
| seeds | 17, 42 | 17, 42 |
| n_init | 3 | 2 |
| standardize | false | true |
| 主指标 | silhouette（同时报 CH/DB） | 同左 |
| 精度/设备 | float32 / GPU | 同左 |

**判定规则**

1. 同 K、同 seed、同预处理下比较 KMeans 与 GMM，不直接比较惯性数值与对数似然数值。
2. 只有全部计划种子成功、主指标有效、实际非空簇数合规的组才参与选择；不得只平均成功种子。
3. 主指标差在 `1e-6` 内：优先较小 K，再优先 KMeans。
4. 组内选代表运行：KMeans 取惯性最小，GMM 取平均对数似然最大；并列时取较小 seed。
5. 单次重启失败可继续；整组失败则剔除该组；全部失败则修正后用新输出目录重跑。
6. 候选达到迭代上限但结果有限时仍可参与比较，报告中结合 `fit.json` 轨迹说明收敛情况。

**验收门**

- [ ] `manifest.json` 显示完成
- [ ] `check-output` 结果为 `passed: true`
- [ ] `clusters.npy` 形状 `(56000,)`、非负整数、实际非空簇数 5–30
- [ ] 选中组的所有种子均成功，选择记录与 `selection.json` 一致
- [ ] 记录运行日志、设备、耗时

---

## 阶段 7：Q2 指标对比与 T1/T2

**汇总表**

- 每个方法 × K：Silhouette、CH、DB 的跨种子均值与标准差
- 每组簇大小分布、实际非空簇数、收敛迭代数、训练目标值、耗时
- 三指标是否指向同一方法/K；不一致时如实记录

**T1：为什么选择这个聚类方案**

- 最终方法、K、主指标及数值
- 与其他候选（另一方法/相邻 K）的必要比较
- 选择依据：无监督指标 + 跨种子稳定性 + 簇结构 + 运行预算

**T2：为什么会出现这个实验现象**

从以下候选中选一个：

- KMeans 与 GMM 的分区差异
- 随机种子导致的指标波动
- 三项指标排序不一致
- 标准化对指标/分区的影响
- 空簇或小簇现象

分析：现象描述 → 算法原理层面的可能原因 → 解释的适用范围与局限。

**验收门**

- [ ] 指标表完整，标注均值/标准差/异常
- [ ] T1/T2 均引用自己的实验数据
- [ ] 明确拟合空间、评价空间、抽样范围与随机种子

---

## 阶段 8：报告与提交打包

**报告结构建议**

1. 环境与设备：CPU/GPU 型号、Python/PyTorch、精度、线程
2. 第一题：实验 A/B 结果表与图、6 个分析题
3. 第二题：方法实测比较、指标汇总、最终选择依据、T1/T2
4. 附录：运行命令、日志摘要、`check-output` 结果、必要的实现说明

**提交目录（严格按实验文档）**

```text
submit/
├── report.pdf
└── code/
    ├── q1_neural_network/
    │   ├── models.py
    │   └── train.py
    └── q2_clustering/
        ├── submission.py
        └── data/
            └── clusters.npy
```

**最终验收门**

- [ ] `submission.py` 与生成最终 `clusters.npy` 的实现完全一致
- [ ] 用默认实现重新运行并通过 `check-output`
- [ ] 报告覆盖两题的结果、分析与必要实验记录
- [ ] 本地保留完整输出目录、实际运行代码与日志至验收结束
- [ ] 不提交 `features.npz` 及其它原始输出

---

## 待确认事项

1. Q2 是否采用「正式基线 + standardize 消融」，还是只跑正式基线以节省时间？
2. Q2 正式作业使用哪块 GPU（A100 / RTX5090），申请多长时限？
3. 报告是否需要在阶段 3 后先出一版第一题草稿，还是最后统一撰写？

## 进度勾选

- [ ] 阶段 0 环境与数据基线
- [ ] 阶段 1 Q1 代码补全与自检
- [ ] 阶段 2 Q1 两组正式实验
- [ ] 阶段 3 Q1 图表与分析
- [ ] 阶段 4 Q2 代码补全与自检
- [ ] 阶段 5 Q2 小规模验证与计时
- [ ] 阶段 6 Q2 正式实验
- [ ] 阶段 7 Q2 指标对比与 T1/T2
- [ ] 阶段 8 报告与提交打包
