# Lab2 实验执行计划

> 修订：2026-11-13，T0003，by codex。
> 依据：[实验正文](../lab2.pdf)第 2–8 页及附录 A、[README](../README.md)、[Q2 框架](../code/q2_clustering/FRAMEWORK_V0.md)和实际源码。
> 工作规范：[WORKSPACE.md](../WORKSPACE.md)。本计划已被 Git 跟踪，继续在原位置维护，不因 logs/ 被忽略而移动。
> 原则：本人实现，AI 解释与检查；先代码验证，再小数据链路，最后全量拟合与分析。下面的命令是后续执行步骤，不表示本次已运行。

## 0. 协作方式与当前起点

### 0.1 每个阶段怎样交付

1. 做会留下产出的工作前，先读看板和断点、检查 Git，认领实际任务号；本计划的“阶段编号”不是预先分配的台账任务号。
2. 人编写 TODO；AI 可以做代码检查、验证和结果审查，不代写实现。每阶段按下文验收门检查，通过后请人验收再推进。
3. 交付源码、配置或分析时，写 docs/tasks/Txxxx.md，记录命令、结果、来源输出目录、关键文件 SHA-256、失败情况和人工验收清单。
4. 只提交本任务文件，commit 后登记 awaiting_review；用户明确通过后才 done。原始输出、日志、环境记录本地保留至课程验收结束。
5. 课程要求、当前源码、实测结果优先于计划里的预估。修订计划不能冒充实验已完成。

### 0.2 2026-11-13 已核对的事实

| 项目 | 当前事实 | 下一动作 |
|---|---|---|
| 本机环境 | ai26：Python 3.12.14、torch 2.11.0+cpu、pytest 9.2.0；CUDA=False，32 个逻辑 CPU | 不再按旧计划补装 pytest；新终端仍要确认解释器 |
| Q1 实现 | models.py、train.py 已有实现，当前公开测试 3 passed | 不重新代填；保留必要的来源核对 |
| Q1 输出 | outputs/expression/、outputs/gradient/ 各有完整的 6 个文件 | 优先复核并整理报告，不盲目重跑 |
| Q1 数值核对 | 八组权重均能加载；以现有权重重新前向计算，预测及 train/grid MSE 与记录一致；记录步数和首尾梯度层数符合设置 | 尚不能据此证明历史训练源码与原命令；补来源证据，不足则新目录重跑 |
| Q2 实现 | submission.py 八个函数仍抛 NotImplementedError | 从三个 KMeans 函数开始，由本人实现 |
| Q2 框架测试 | 当前 6 个测试文件：48 passed、49 subtests passed | 只说明框架回归通过，绝不代表 TODO 正确 |
| Q2 数据 | features.npz 的 SHA-256 与发布清单一致 | 不切分正式拟合数据，不改变行序 |
| Q2 正式输出 | 当前未发现 outputs* 目录 | 不存在可提交的最终 clusters.npy |
| 正式报告/集群 | 本次未核验报告完成度、集群账户或 GPU 配额 | 不标记完成；需要实际执行阶段 3、6–8 |

Q1 两项 results.json 均记录 CPU / torch 2.11.0+cpu。其内部总耗时分别约 6.62 s 和 4.00 s，仅供定位历史结果，不是未来耗时承诺。当前 main.py 固定 1 个计算线程，不要把环境默认线程数写成本次实验线程数。

当前八组网格 MSE（直接来自既有结果，本次只重算核对、未重训）：

| 实验 | A | B | C | D |
|---|---:|---:|---:|---:|
| expression | 0.00702260 | 0.00702569 | 0.00276959 | 7.62932e-7 |
| gradient | 0.000260915 | 0.429819 | 0.00123965 | 0.429819 |

**最近的实际行动顺序：** 补齐已有 Q1 的命令/源码来源 → 阶段 3 写 Q1 结果与分析 → 阶段 4 实现 Q2。无需把已存在的 Q1 当成完全没开始。

## 1. 命令约定、环境与文件位置

本地统一使用 PowerShell，避免混用 Bash 的反斜杠续行、here-doc 或 Linux 路径。首次从 LAB2 根目录执行以下准备块；重新开终端时重建变量：

```powershell
$Lab = (Resolve-Path .).Path
if (-not (Test-Path (Join-Path $Lab "lab2.pdf"))) { throw "请先进入 LAB2 根目录" }
$Py = Join-Path $HOME "miniconda3/envs/ai26/python.exe"
if (-not (Test-Path $Py)) { throw "找不到 ai26 解释器，请先核对 Conda 安装位置" }
$Q1 = Join-Path $Lab "code/q1_neural_network"
$Q2 = Join-Path $Lab "code/q2_clustering"
$TaskId = Read-Host "输入已认领的任务号（例如 T0004，不是阶段编号）"
if ($TaskId -notmatch '^T\d{4,}$') { throw "任务号格式不正确" }
$Run = Get-Date -Format "yyyyMMdd_HHmmss"
$LogDir = Join-Path $Lab "logs/$TaskId/$Run"
New-Item -ItemType Directory -Force $LogDir | Out-Null
$env:PYTHONUTF8 = "1"
function Invoke-LabPython {
    param([string]$Log, [string[]]$PyArgs)
    & $Py @PyArgs 2>&1 | Tee-Object -FilePath (Join-Path $LogDir $Log)
    if ($LASTEXITCODE -ne 0) { throw "命令失败，停止本阶段并检查 $Log" }
}
```

- 后文每个代码块都会说明工作目录；日志自动进入本次 $LogDir。每次重试先更换 $Run，避免覆盖上次日志和非空输出目录。
- 数据/初值保留 code/ 下原位置；新配置放 Q2 目录；小数据文件放 _tmp/；原始输出放题目 outputs*；报告源稿与精选图放 docs/reports/。
- $Run 只是运行标识，不能代替任务号。实际运行命令、解释器、配置及对应任务号写入任务说明。
- 当前本机版本已满足依赖读取。不要为了“和旧笔记一致”升级或降级 torch；安装前先检查，只有缺包才安装，并记录变动。
- 集群另有 Linux 命令，见阶段 6；不能把本机解释器路径照抄到集群。

## 2. 阶段总览与最低交付

| 阶段 | 负责人/输入 | 产物 | 进入下一阶段的门槛 |
|---|---|---|---|
| 0 环境与数据基线 | 人执行命令，AI核对 | 环境、哈希、测试日志 | 解释器正确，数据匹配 |
| 1 Q1 实现复核 | 人已有代码，AI检查 | 接口/更新检查记录 | 八组初值和前向行为正确 |
| 2 Q1 正式结果 | 人提供历史来源或运行 | 两项共八组完整输出 | 参数、步数、图表、数值可核对 |
| 3 Q1 分析 | 人写结论，AI审查 | Q1 源稿、两张表、四张必需图、六小题 | 观察和解释分开，引用数据 |
| 4 Q2 实现 | 人逐个完成八个函数 | submission.py、自检记录 | 算法和指标自检均通过 |
| 5 Q2 开发链路 | 小型无标签合成数据 | toy 数据/配置/完整输出 | development 检查通过 |
| 6 Q2 全量比较 | 全部 56000 行 | 最低合规方案，按预算扩展 | 两方法、同 K/seed、最终输出核验通过 |
| 7 Q2 分析 | 全量结果及跨种子统计 | 对比表、最终选择、T1/T2 | 选择由无监督证据支持 |
| 8 报告和打包 | 两题结果源稿、最终实现 | report.pdf 和五个提交文件 | 来源一致、结构准确、本人能解释 |

依赖：0 → 1 → 2 → 3；0 → 4 → 5 → 6 → 7；3 和 7 → 8。阶段 0 是自加的风险检查，不是课程独立评分项，可并入阶段 1/5；不必机械等待“半小时”。

最低要求和可选扩展分开：Q1 必须八组；Q2 必须两方法、至少两个 K（5–30）、至少两个种子、每次全量拟合。更多 K、更多重启、标准化、ARI 和 PCA 都不是替代必做流程的捷径。

## 阶段 0：环境与数据基线

**工作目录：LAB2；通常数分钟，不安装已存在的包。**

```powershell
Set-Location $Lab
Invoke-LabPython -Log "environment.log" -PyArgs @("-c", "import sys,os,torch,numpy,scipy,sklearn,matplotlib,pytest; print(sys.executable); print(sys.version); print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'cpu',os.cpu_count(),'threads',torch.get_num_threads()); print('numpy',numpy.__version__,'scipy',scipy.__version__,'sklearn',sklearn.__version__,'matplotlib',matplotlib.__version__,'pytest',pytest.__version__)")
$Expected = ((Get-Content "code/q2_clustering/data/SHA256SUMS.txt" -Raw).Trim() -split '\s+')[0]
$Actual = (Get-FileHash "code/q2_clustering/data/features.npz" -Algorithm SHA256).Hash.ToLower()
if ($Actual -ne $Expected.ToLower()) { throw "正式特征数据哈希不匹配，停止实验" }
"features.npz SHA256=$Actual" | Tee-Object -FilePath (Join-Path $LogDir "data_hash.log")
```

发布哈希应为 a07578f6e1ab9d17667b987b25b1b9bc1f06a5f1f678a127e31dc78bc5ff0b38。PowerShell 直接按空白解析，可避免原清单 CRLF 对 Bash 校验的影响；保留了旧计划提醒“先检查数据”的意图。

**通过门：**
- [ ] 日志中的解释器确实是 ai26；版本、设备和依赖导入成功。
- [ ] 数据哈希一致；Q1 固定数据为 train_x (1024,2)、train_y (1024,1)、grid_x (4225,2)、grid_y (4225,1)，float32。
- [ ] 将环境摘要带入后续报告，而非强制新开一项环境报告任务。

**失败处理：** 缺 pytest 时才执行 `& $Py -m pip install "pytest>=8"`；其他缺包按实际补齐。数据哈希不符先恢复课程原始文件，不绕过校验、不重新生成“相似数据”。沙箱临时目录 PermissionError 先按权限流程处理后原命令重跑，不视为算法失败。

## 阶段 1：Q1 代码复核，不重新代填

**工作目录：$Q1；输入为本人已有 models.py/train.py。**

```powershell
Set-Location $Q1
Invoke-LabPython -Log "q1_pytest.log" -PyArgs @("-m", "pytest", "-q", "-p", "no:cacheprovider")
```

参数量和配置以 main.py 及 assets/initial_states.json 为准：

| 实验/组 | 隐藏层 | 激活 | 参数量 | 更新次数 | 学习率/优化器 |
|---|---|---|---:|---:|---|
| expression A | [16] | ReLU | 65 | 3000 | 0.01 / Adam |
| expression B | [16] | ReLU | 65 | 9000 | 0.01 / Adam |
| expression C | [64] | ReLU | 257 | 3000 | 0.01 / Adam |
| expression D | [8,4] | ReLU | 65 | 3000 | 0.01 / Adam |
| gradient A | [32,32] | Sigmoid | **1185** | 1000 | 0.05 / SGD |
| gradient B | 8×32 | Sigmoid | 7521 | 1000 | 0.05 / SGD |
| gradient C | 8×32 | ReLU | 7521 | 1000 | 0.05 / SGD |
| gradient D | 8×32 | Sigmoid | 7521 | 1000 | 0.1 / SGD |

**检查单：**
- [ ] hidden 为带偏置 Linear 的 ModuleList，output 为线性输出；前向形状 (B,1)。
- [ ] 全八组课程初值严格加载；公开测试只抽三组，不能替代全八组检查。
- [ ] 一次更新使用全部 x/y、平均 MSE、正确清梯度/反传/更新；steps=0 不更新，steps=n 恰好更新 n 次。
- [ ] 用一个简单线性小例子核对 SGD 更新或训练可降；这是调试检查，不替代课程数据的完整训练。
- [ ] ReLU/Sigmoid 与空隐藏层行为正确，参数和诊断为有限值。

**失败处理：** 先定位本人代码的形状、层次、梯度问题，不修改课程初值或公共接口来迎合错误实现。旧计划“所有 gradient 组都是 7521”已纠正；不把“任意 batch 一步损失必须下降”当通用规则。

## 阶段 2：确认或重建 Q1 正式结果

### 2.1 优先利用已有产物

现有目录为 $Q1/outputs/expression 与 $Q1/outputs/gradient。本次已核对权重、预测、MSE 和记录结构，也查看了 expression/fitting.png 和 gradient/gradients.png。

后续应补记：原始运行命令、实际运行的源码/配置来源、运行环境及文件哈希。若历史命令或训练源码不能确认，不伪造历史记录；可以在新目录重跑八组，使用新日志作为正式来源。仅补图缺失时，使用 `python -m utils.plot 输出目录`，无需重训，但会重写该目录中的图像，应先记录原图来源。

### 2.2 确需重跑时的命令

**工作目录：$Q1；沿用本次新 $Run，不覆盖 outputs/。**

```powershell
Set-Location $Q1
$Q1Expression = "outputs_$Run/expression"
$Q1Gradient = "outputs_$Run/gradient"
Invoke-LabPython -Log "q1_expression.log" -PyArgs @("main.py", "--task", "expression", "--device", "cpu", "--output", $Q1Expression)
Invoke-LabPython -Log "q1_gradient.log" -PyArgs @("main.py", "--task", "gradient", "--device", "cpu", "--output", $Q1Gradient)
```

不能在非空输出目录强行重跑。两组 batch_size 均为 1024；expression A/B 共用初值，gradient B/C/D 共用 Linear 初值，不改这些控制变量。

### 2.3 文件与验收门

| 文件 | expression | gradient |
|---|---|---|
| results.json / summary.csv | 四组指标、配置和历史 | 四组指标、配置和历史 |
| predictions.npz / models.pt | 网格预测及最终权重 | 网格预测及最终权重 |
| training.png | 四组训练曲线 | 四组训练曲线 |
| fitting.png / gradients.png | x2=0 目标与预测截面 | 初始/最终逐层相对梯度 |

- [ ] 两项各四组，配置、参数量和末次记录步数与阶段 1 表一致；train/grid MSE 均有限。
- [ ] 历史包含 step=0 和最终 step，每 100 次更新记录；不误把日志条数当更新次数。
- [ ] 梯度诊断首尾均有 A 的 2 层、B/C/D 的 8 层；RMS/relative_to_output 的零值或未定义值按实际记录解释。
- [ ] 从保存权重重算预测/MSE，与保存数据在浮点容差内一致；CSV 与 JSON 对得上。
- [ ] 查看图像坐标、图例及重合曲线，不能只检查“文件存在”。
- [ ] 记录两项总耗时；框架没有逐组耗时字段，不凭空填每组耗时。

**重要：** 不要求每组损失都显著下降或每一步单调下降。梯度消失本就是要观察训练困难，B/D 基本重合不意味着实验失败。网格 MSE 在训练过程并未逐点记录，不把训练曲线称为“网格 MSE 曲线”。

## 阶段 3：Q1 图表与六个分析小题

**输入：阶段 2 确定的唯一正式来源；产物：docs/reports/q1.md 和 figures/ 下精选图。**

1. 写两张四组结果表：widths、parameters、steps、lr、train_mse、grid_mse；数据从 results.json/summary.csv 转录，不根据图片猜小数。
2. 选择四张必需图：expression 的 training/fitting、gradient 的 training/gradients；复制到报告图目录并在图注记来源目录。
3. 先用简短文字描述观察，再按下表作答，不预填“预期结论”。
4. 每个小题至少引用一个具体数值、图或结构推导；本人能口头解释。

| 小题 | 必须处理的证据/问题 |
|---|---|
| A(a) | A/D 网格 MSE 和 x2=0 截面；低平均 MSE 是否掩盖峰高、峰间谷底或背景错误 |
| A(b) | 本人尝试用 ReLU 与含偏置的线性运算表示单峰再组合双峰，解释层次；不只说“深度更强” |
| A(c) | B/A 的更新预算、C/A 的参数量与误差改善；分清训练误差和网格误差，不能预设多训练必然更好 |
| B(a) | A/B 的首尾梯度分布与误差；损失下降慢为何不能单独证明梯度消失 |
| B(b) | B/C 同 Linear 初值下，激活改变后的梯度及拟合变化；避免宣称 ReLU 保证无梯度问题 |
| B(c) | B/D 的实际误差与初始梯度；区分梯度和更新幅度，判断本次加学习率是否有效 |

**通过门：** 六问完整、两表四图齐全、数值和图注一致，解释与观察分开，不用训练不佳作为删掉某组的理由。先完成 Q1 草稿再推进 Q2，避免最后一起赶报告。

## 阶段 4：Q2 逐函数实现与分项自检

**工作目录：$Q2；只有本人修改 submission.py，保持八个函数签名及返回契约。**

| 顺序 | 函数 | 自查重点（不是实现答案） |
|---|---|---|
| 1 | init_centers_kmeanspp | 同设备 rng，可重复；零距离退化时选未选行；不构造 N×N |
| 2 | assign_clusters | 返回整数 labels 与平方距离；并列小编号；防消去负值，避免 N×K×D |
| 3 | update_centers | 非空簇均值，空簇保留旧中心，不修改输入 |
| 4 | gmm_estep | 对角协方差、对数空间归一化、责任度每行和为1、平均观测对数似然 |
| 5 | gmm_mstep | 权重/均值/对角方差；每维加正 reg_covar；N_k<=1e-12 时按契约报错 |
| 6 | silhouette_score | 欧氏距离而非平方距离；同簇排除自身、其他簇平均距离；单点簇/零分母约定 |
| 7 | calinski_harabasz_score | 实际非空簇数，零 W 返回1.0 |
| 8 | davies_bouldin_score | 平均欧氏距离；atol=1e-8/rtol=0 的整体近零约定；仅排除中心距离恰为0的簇对 |

**执行顺序：** 1–3 做完先独立小例子验证 KMeans；1–5 做完跑 algorithms；6–8 做完跑 metrics；最后完整自检。只实现 KMeans 时 algorithms 仍可能在 GMM 处失败，不把它误判为 KMeans 一定错误。

```powershell
Set-Location $Q2
Invoke-LabPython -Log "q2_algorithms.log" -PyArgs @("extras.py", "self-check", "--part", "algorithms", "--device", "cpu")
Invoke-LabPython -Log "q2_metrics.log" -PyArgs @("extras.py", "self-check", "--part", "metrics", "--device", "cpu")
Invoke-LabPython -Log "q2_all_float32.log" -PyArgs @("extras.py", "self-check", "--device", "cpu")
Invoke-LabPython -Log "q2_all_float64.log" -PyArgs @("extras.py", "self-check", "--device", "cpu", "--dtype", "float64")
Invoke-LabPython -Log "q2_regression.log" -PyArgs @("-m", "pytest", "-q", "-p", "no:cacheprovider", "-k", "not test_unfinished_submission_saves_status_without_final_output")
```

**测试边界：** 当前完整 pytest 中有一项明确假定 submission.py 尚未实现，断言 not_implemented。补全后这项前提失效，上面只排除这一项；记录为“假定未实现的框架测试不适用”，不改课程测试来伪造通过，不排除其他失败项。当前 48 passed/49 subtests 是补全前基线，不是学生算法验收。新自检和阶段 5 链路才是补全后的必要证据。

**通过门：**
- [ ] algorithms、metrics、完整 float32/64 自检成功；记录实际检查条数与命令，不写固定的旧文件数量当测试结果。
- [ ] 继承 X.dtype/X.device；不修改输入；新 labels 为整数；评价函数允许不连续簇号。
- [ ] 不把完整数据转 NumPy 运算，不调用 sklearn 聚类/初始化/指标代替 TODO；独立对拍时可用 sklearn，不进入提交实现。
- [ ] 小例子覆盖重复点、空簇、单点簇、非连续标签和退化情况；不把 N×N×D 或 N×K×D 中间数组带到全量运行。

**失败处理：** 按首个失败函数对照 docstring、形状和精度排查。NotImplementedError 是待实现，不是调学习率或配置能绕过的问题。CPU 过后若正式用 CUDA，还要在实际 GPU 节点跑相同自检。

## 阶段 5：Q2 小数据全链路

**前置：阶段 4 通过；工作目录：$Q2。** 只生成匿名特征 X，不写真实标签。

下面是输入/配置准备，不是八个 TODO 的实现：

```powershell
Set-Location $Q2
@'
import json, sys
from pathlib import Path
import numpy as np
run = sys.argv[1]
folder = Path("../../_tmp") / run
folder.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(17)
X = np.concatenate([rng.normal(-3, .3, (60, 2)), rng.normal(3, .3, (60, 2))]).astype(np.float32)
np.savez(folder / "toy_features.npz", X=X)
cfg = json.loads(Path("config.json").read_text())
cfg.update(candidate_k=[2, 3], seeds=[17, 42], n_init=1, kmeans_max_iter=30,
           gmm_max_iter=30, silhouette_sample_size=120, selection_metric="silhouette")
Path("config_dev.json").write_text(json.dumps(cfg, indent=2) + "\n")
'@ | & $Py - $Run
if ($LASTEXITCODE -ne 0) { throw "小数据准备失败" }
$Toy = Join-Path $Lab "_tmp/$Run/toy_features.npz"
$DevOut = "outputs_dev_$Run"
Invoke-LabPython -Log "q2_toy_run.log" -PyArgs @("extras.py", "run", "--device", "cpu", "--development", "--input", $Toy, "--config", "config_dev.json", "--output", $DevOut)
Invoke-LabPython -Log "q2_toy_check.log" -PyArgs @("extras.py", "check-output", "--development", "--input", $Toy, "--output", $DevOut)
```

**通过门：** 两种方法 × 两个 K × 两个种子，8 个候选；候选状态、责任度/方差和三项指标可解释；最终选择与导出存在；check-output 为 passed:true；同一个 toy 输入用于运行和核验。

**失败处理：** 若 no_eligible_group，查看每组 exclusion_reasons 和各指标 status/reason；若算法报错，保留 manifest 和日志，修复后换新 $Run。development 结果不能直接提交，不可以 120 行开发成功替代 56000 行正式拟合。

## 阶段 6：Q2 全量对比，先合规再扩展

### 6.1 默认配置与规模

保留原 config.json，不覆盖输出目录中的 config.json/fit_config.json。默认先新建 config_minimum.json，使用完整正式数据，**不带 --development、不切分样本**：

| 参数 | 最小合规/计时方案 | 预算充分的扩展方案 |
|---|---|---|
| candidate_k | [10,15] | 原默认 [5,8,10,12,15,20,25,30] |
| seeds | [17,42] | 同左 |
| n_init | 1 | 3 |
| 拟合 | 全部 56000×512 | 同左 |
| kmeans_max_iter / tol | 200 / 1e-4 | 同左 |
| gmm_max_iter / tol | 100 / 1e-4 | 同左 |
| reg_covar / standardize | 1e-4 / false | 同左 |
| dtype / threads | float32 / 4 | 同左 |
| 主指标 | silhouette | 同左，改指标需单独回放 |
| Silhouette 抽样 | 3000，无放回，seed=20261011 | 同左 |
| 候选数 / 重启总次数 | 8 / 8 | 32 / 96 |

K=[10,15] 是预先固定的预算起点，不代表已知类别数或最优 K。最低方案本身符合最低数量要求；若成功且证据充分，可作为最终候选来源，不强制为扩展而重跑。n_init=3 不是题面硬性要求。不得承诺隐藏标签 ACC；99.5% 是教师侧 performance 满分门槛，不是本地可计算或保证的结果。

**工作目录：$Q2，准备最低配置：**

```powershell
Set-Location $Q2
@'
import json
from pathlib import Path
from utils.data import validate_config
cfg = json.loads(Path("config.json").read_text())
cfg.update(candidate_k=[10, 15], seeds=[17, 42], n_init=1, selection_metric="silhouette")
cfg = validate_config(cfg)
Path("config_minimum.json").write_text(json.dumps(cfg, indent=2) + "\n")
print("planned candidates:", 2 * len(cfg["candidate_k"]) * len(cfg["seeds"]))
'@ | & $Py -
if ($LASTEXITCODE -ne 0) { throw "最低配置准备失败" }
```

### 6.2 CPU 可执行路径

只有阶段 4、5 已通过才运行。Q2 可以 CPU，GPU 是资源选项，不是课程强制要求。若准备在 GPU 上正式运行，可跳到 6.3，不先跑一遍昂贵的 CPU 全量计时。

```powershell
Set-Location $Q2
$FitOut = "outputs_fit_min_$Run"
$FinalOut = "outputs_eval_min_$Run"
Invoke-LabPython -Log "q2_full_fit.log" -PyArgs @("extras.py", "run", "--stage", "fit", "--device", "cpu", "--config", "config_minimum.json", "--output", $FitOut)
Invoke-LabPython -Log "q2_full_evaluate.log" -PyArgs @("extras.py", "evaluate", "--device", "cpu", "--experiment", $FitOut, "--output", $FinalOut)
Invoke-LabPython -Log "q2_full_check.log" -PyArgs @("extras.py", "check-output", "--output", $FinalOut)
```

fit 只保存候选，没有最终 clusters.npy，不在 fit 输出上运行最终检查器。evaluate 生成完整正式输出，可直接交付，无需再跑 main.py 重复拟合。保存结果所用的 submission.py、实际配置和源码 commit；框架不替你保存完整源码快照。

### 6.3 107/GPU 路径与作业预算

1. 先由本人确认登录方式、远程 LAB2 路径、Conda 初始化脚本和可用账户/分区/GPU；这些本次未验证，不填写虚构路径或宣称已配置。
2. 参照 README 的 107 模板准备 sbatch。保留计算节点执行、1 GPU、4 CPU、16G 内存的起点；4 小时只是申请上限，不是预计耗时。不要在登录节点跑拟合或 CUDA 依赖安装。
3. CPU 和 CUDA 依赖分开；安装需要时在计算节点按 requirements-gpu.txt 完成并记录版本。
4. 在实际 GPU 节点运行下面三步，自检不通过则不进入全量拟合。示例已进入远程 LAB2 的 code/q2_clustering，并激活正确环境。

```bash
set -euo pipefail
python -c 'import torch; print(torch.__version__); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
python extras.py self-check --device cuda
python extras.py self-check --device cuda --dtype float64
RUN="$(date +%Y%m%d_%H%M%S)"
python extras.py run --stage fit --device cuda --config config_minimum.json --output "outputs_fit_min_$RUN"
python extras.py evaluate --device cuda --experiment "outputs_fit_min_$RUN" --output "outputs_eval_min_$RUN"
python extras.py check-output --output "outputs_eval_min_$RUN"
```

配置 config_minimum.json 需在上传前生成，或在远端按 6.1 生成。README 的 sbatch 头和工作目录应使用本人核实过的值；Slurm 日志和 job ID 随任务记录。GPU 不可用会报错，不会悄悄退回 CPU。GPU fit 的回放必须在同设备类型及精度下做，不能搬回 CPU 回放；check-output 可在完整文件下载后本地运行。

### 6.4 是否扩展，先看证据再花预算

- 收集 runs.json 逐候选 elapsed_seconds、候选 fit.json 的重启日志，以及 results.json 的阶段总 elapsed_seconds；拟合、评价、排队时间分开。框架没有 fit_seconds 字段。只在同设备、同实现下用实际耗时粗估，不直接拿 CPU 耗时预测 GPU。
- 8 次重启到 96 次约是 12 倍“次数”，不是 12 倍墙钟时间保证；更大 K、GMM 迭代、指标计算和 I/O 都会改变成本。
- 若跨种子差异大，先固定 K=[10,15] 将 n_init 提到3；若需要更广 K 对比且预算允许，再采用完整八 K、n_init=3。新增配置另存为 config_formal.json，重新拟合到新目录，不修改旧产物。
- 默认先完成未标准化方案。standardize=true 仅在主流程交付后另开可选消融任务，不能用它替代两方法必做比较。
- 若已定稿 submission.py 后未再改算法，可回放指标；若算法/拟合配置改变，正式产物必须重新拟合，不能把旧标签与新源码拼在一起。

### 6.5 正式输出核验门与失败分支

- [ ] manifest 的计划/完成数量匹配配置，逐候选成功情况和所有指标 status 已审查；优先争取 completed，但不能把退出码0等同“全部成功”。
- [ ] 若为 partial，列出失败候选/指标与原因，不能静默删种子后求平均。若导致某方法或必要 K/seed 比较缺失，修复并重新运行，不能声称最低对比已经完整。
- [ ] 选中组全部计划种子成功、主指标有效，实际非空簇数5–30；可能不等于声明 K，应记录实际值。
- [ ] selection.json、runs.json、groups.json、export_manifest.json 相互一致，最后 check-output 显示 passed:true。
- [ ] 最终 clusters.npy 为 (56000,) 非负整数，与原始行序对应；记录其 SHA-256。
- [ ] 已保留完整输出目录、实际源码、配置、数据哈希和日志，不只留下 clusters.npy。

遇到 not_implemented 先补函数；framework_error/NaN/Inf 先查接口、稳定性和内存；no_eligible_group 先读 exclusion_reasons。达到迭代上限但数值有限不自动判失败，结合 fit.json 分析收敛。程序异常或作业超时后保留日志和产物，使用新目录重新拟合；框架没有通用的断点续训入口，不直接在半成品目录追加。只评价函数有问题且原 fit 完整时，可修正指标后在新目录 evaluate，不必重拟合。

## 阶段 7：Q2 对比、最终选择与 T1/T2

**输入：完整正式输出；产物：docs/reports/q2.md。**

1. 从 runs.json 建“方法×K×seed”明细表；从 groups.json 建“方法×K”均值/样本标准差表，包含失败、排除原因，不手工只选好看的种子。
2. 表中至少列主指标均值/波动、其他诊断指标、实际簇数、最小簇规模、耗时和异常。若定义“极小簇”，先写清阈值，例如小于全量样本的0.1%，并注明是诊断口径不是课程硬阈值。
3. 按框架规则复核最终方法/K/代表 seed：主指标跨种子均值；差在1e-6内优先小 K 再 KMeans；组内按本方法训练目标，目标差在1e-7*max(1,abs(最佳目标))内选小 seed。
4. 写 T1：为何选最终方案，至少比较另一方法和另一个 K，说明主指标的几何偏好与局限。
5. 写 T2：从本人真实结果选一个现象，按“现象与读数 → 算法原理 → 可能原因 → 适用边界”组织；无改善或稳定也可分析。

**统一评价口径：** 拟合全56000行；Silhouette 在固定3000行子集内部计算距离，不在函数内再抽样；CH/DB 全量原始空间。standardize 只改变拟合空间。记录 silhouette_seed 和子集覆盖簇数；样本标准差不是置信区间。惯性与 GMM 平均对数似然不可直接比较大小；无监督指标、稳定性和 check-output 均不等于隐藏标签 ACC。

若决定用 CH/DB 做正式主指标，不手动复制某候选文件冒充正式导出。复制原 fit 输出的 config.json 为新评价配置，只改 selection_metric 等允许字段，然后在**原设备/原精度/原数据**上回放，输出到新的 outputs_eval_ch_<标识> 等目录。允许改的字段仅 selection_metric、silhouette_sample_size、silhouette_seed、threads；不允许改 K、seed、n_init、standardize、dtype 或算法后继续沿用旧 fit。

**通过门：** 本人能解释最终选择、两方法假设、随机性、三个指标及其局限；T1/T2 均引用本人的表/图。PCA/ARI 可选；ARI 比较两个预测分区不是读取真实标签，二维 PCA 若展示须报告保留方差比例，不能替代原空间评价。

## 阶段 8：统一报告、提交打包与口头验收

### 8.1 报告

维护 docs/reports/report.md 或其他可编辑源稿，最后导出 PDF 并查看每一页。报告围绕结果与分析，避免重复题面、粘贴源码或整段终端日志，不人为规定题面没有的页数上限。

- 环境、实际设备与线程、依赖版本、精度、运行命令和关键配置。
- Q1 两表四图及六个分析小题，图注与正式来源一致。
- Q2 两方法实测对比、配置/种子与异常、最终选择、T1/T2、check-output 核验结果。
- 不声称本地知道教师隐藏 ACC；性能满分门槛不影响其他已满足项的记录。
- 在导出 PDF 前本人核对结论、公式、图表引用，并能回答关键代码和结果问题。

### 8.2 提交目录，按 PDF 第8页的树形缩进

```text
submit/
  report.pdf
  code/
    q1_neural_network/
      models.py
      train.py
    q2_clustering/
      submission.py
    data/
      clusters.npy
```

**特别注意：data/ 与 q1_neural_network/、q2_clustering/ 同级。最终标签路径是 submit/code/data/clusters.npy，不是 submit/code/q2_clustering/data/clusters.npy。** 本次已查看 PDF 第8页渲染图确认，修正了旧计划的错误缩进。

在最终正式目录已确定后，按需手动或用 Copy-Item 把三个实现、report.pdf 和该目录的 clusters.npy 放入上述位置；不复制 features.npz、模型权重或整套运行目录。打包前确认目标路径，禁止为“重打包”递归删除未检查的目录。

**最终验收门：**
- [ ] 恰好五个交付文件，目录层级与上面一致，PDF 能打开且覆盖两题。
- [ ] 三个代码文件与最终实验使用的实现一致；正式运行未用临时替代实现。
- [ ] 对完整 Q2 正式输出运行 check-output，通过后再复制最终标签；复制后用 SHA-256 确认与来源一致，不能只对单独提交文件使用来源检查器。
- [ ] Q1 两项共八组的原始指标/历史/预测/权重/梯度与图像仍保留；Q2 完整输出、实际代码/配置/日志仍保留。
- [ ] 本人能解释 MLP/更新、梯度证据、KMeans++/EM、指标、选择规则与至少一个实验现象。
- [ ] 工作任务 commit 和人工验收完成；课程提交由本人执行，不把 Git push 或任务验收当成课程已提交。

## 当前行动清单与修订记录

以下只勾选本次确实完成的核对，不替人宣称实验阶段已验收：
- [x] 重新阅读实验正文、README 和实际框架；修正提交树形层级。2026-11-13 by codex。
- [x] 确认本机 Python/torch/pytest、Q2 数据哈希，Q1 公开测试3项通过。2026-11-13 by codex。
- [x] 以已有 Q1 八组权重重算前向、预测和 MSE，核对历史步数与梯度层数；本次未重训。2026-11-13 by codex。
- [x] Q2 补全前框架基线：48 passed、49 subtests passed；不代表八个 TODO 正确。2026-11-13 by codex。
- [ ] 阶段0：后续正式工作留存环境、命令和校验日志。
- [ ] 阶段1–2：确定 Q1 正式来源；缺历史源码/命令证据时在新目录重跑。
- [ ] 阶段3：完成 Q1 源稿、两表四图、六个小题。
- [ ] 阶段4：本人实现 Q2 八函数，按顺序自检。
- [ ] 阶段5：小数据链路与 development 输出核验通过。
- [ ] 阶段6：最低全量方案完成；按证据与预算决定扩展。
- [ ] 阶段7：Q2 指标表、最终选择、T1/T2。
- [ ] 阶段8：统一 PDF、五文件打包、本人验收与课程提交。

| 日期/任务 | 变化 | 证据/边界 |
|---|---|---|
| 原计划已有修改 | 保留“阶段0为自加风险检查”及哈希/CRLF风险的意图，改用本机 PowerShell 可执行校验 | 不撤销用户既有修改，不新增环境安装任务 |
| 2026-11-13 / T0003 | 从“重新开始八阶段”改为当前状态驱动；纠正环境版本、gradient A参数量、6测试文件的实际范围、Q1图表/耗时字段、提交数据层级 | 本次只核对已有结果与框架，不实施 Q2、不进行正式训练 |
| 2026-11-13 / T0003 | 增加 PowerShell 命令、日志约定、开发数据/配置准备、最低全量路径、GPU前置、失败与回放规则 | 集群权限、实际可用 GPU、截止日期未获确认；不虚构已安排的资源或时长 |
