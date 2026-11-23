# P2：Q2 十种子与邻近 K 的稳健性实验

任务：T0016，2026-11-23，codex。本文保留提交准备时的设计与自检记录。后续Job190299已完成60候选，T0017核验与分析见[P2正式结果](p2_robustness_results.md)；人工验收状态另见任务说明。

## 问题与固定设计

原 Q2 每方法/K 只有两个外部种子，不能有力支持“不依赖初始化”。本实验增加到十个种子，检验初始化稳定性及 K=10 对 K=8、12 的邻域优势。

| 项目 | 设置 |
|---|---|
| 方法 | 现有学生实现的 KMeans、对角 GMM；不改课程 TODO 或框架 |
| K | [8,10,12]，只检验该邻域 |
| 外部种子 | [17,42,123,2026,20261123,7,11,29,101,314]，事前固定，不筛掉较差种子 |
| 重启 | 每候选 n_init=3，60 候选、180 次重启；每候选仍按原训练目标选最佳重启 |
| 数据 | 原始 56000×512、float32、行序不变、不标准化、不使用真实标签 |
| KMeans | max_iter=200，tol=0.0001 |
| 对角 GMM | max_iter=100，tol=0.0001，reg_covar=0.0001 |
| 指标 | 同一固定 3000 行子集上 Silhouette（无放回，seed=20261011）；全量原始空间 CH/DB |
| 设备 | 学校 Slurm，Students/stu/qos_stu_default，1 张 A100，4 CPU、16G、30 分钟；无 CPU 正式回退 |
| 比较口径 | 两方法用同一外部种子列表；不声称它们所有随机数消耗或迭代路径相同 |

数据 SHA-256：`a07578f6e1ab9d17667b987b25b1b9bc1f06a5f1f678a127e31dc78bc5ff0b38`。运行前校验，改变输入会拒绝执行。

这是独立的新实验，全部候选重新拟合；即使包括原有种子17/42，也保留本次来源而不拼接旧产物。旧八 K、两种子的 Job177703 及最终 `clusters.npy` 不被替换。P2 目录内框架仍会导出一个按预设规则选择的实验标签，是否用于正式提交另行审查。

## 执行文件与产物

- [p2_config.json](../../code/q2_clustering/p2_config.json)：独立完整配置，不修改默认 `config.json`。
- [p2_robustness.py](../../code/q2_clustering/p2_robustness.py)：`run` 调用现有 `main.run`，`analyze` 只读已存指标和标签，`smoke` 仅运行90行二维合成数据。
- [run_p2_T0016.sbatch](../../code/q2_clustering/run_p2_T0016.sbatch)：激活 ai26、检查 CUDA/A100、运行及分析；Python 或 tee 失败都会非零退出。
- `.gitattributes`：固定此 sbatch 为 LF。

正式运行目录（时间戳/作业号由实际运行生成）：

```text
code/q2_clustering/outputs_T0016_<stamp>_<jobid>/
  p2/
    p2_manifest.json       # 作业号、配置、环境、数据与源码身份、运行状态
    source_snapshot/       # 12份运行源码/配置/作业脚本
    experiment/            # 原框架全部模型、标签、重启历史、指标与选择
    export_check.json      # 框架导出来源检查；有有效选择时生成
    artifact_hashes.json   # 全部上述文件SHA-256（不递归包含自身）
  p2_analysis/
    summary.json / summary.md
    candidates.csv
    restarts.csv
    within_method_ari.csv
    cross_method_ari.csv
    per_seed_k.csv
    paired_method_metrics.csv
logs/T0016/<stamp>_<jobid>/  # job.log及Slurm .out/.err
```

输出必须是新目录，无覆盖、断点续跑或自动跳过旧候选；失败保留现场，新提交使用新目录。若完整遍历后部分候选失败，分析保留其失败状态，缺失值不填零；不将只统计成功者的均值当作全十种子结果。若作业被中断、文件损坏或缺少计划候选，检查会拒绝给出完整稳健性总结，保留原始日志供诊断。

`analyze` 检查所有候选的文件哈希、运行/拟合记录、标签形状与簇计数、配置、固定评价子集和全部计划种子；分析前后校验源目录未变。它可以在本机CPU进行，因为只计算标签ARI，不重新拟合、不读取特征做指标回放。哈希用于核对产物完整性，不是独立证明指标数学正确的签名。

## 预定统计与解释

1. **全部候选与失败情况**：60行候选表、180行重启表；分别统计候选失败、所选重启未收敛、各内部重启失败/达到上限，以及实际簇数、最小/最大簇和完整簇大小。
2. **六个方法/K组**：每项指标十种子的均值、样本标准差（ddof=1）、中位数和范围；标明有效个数与是否完整，不隐藏较差种子。
3. **组内稳定性**：每组10种子的45对ARI，共270对，记录每对及最小值/中位数/范围。它们共享拟合运行，不是270次独立实验；内部三重启也不是三份外部种子。
4. **K 邻域比较**：每方法、每种子记录 K=8/10/12 的 Silhouette；分数距该种子最佳值不超过1e-6视为并列，另按原框架较小K规则给出选择和频次。保留并列列表，频次并非独立数据集上的正确率。
5. **跨方法比较**：同K同种子的30对ARI、三指标的 KMeans−GMM 差值。Silhouette/CH越大越好，DB越小越好。惯性与对数似然不相互比较。

若K=10在多数或全部种子中领先邻域，且分区稳定，可以加强“在该数据、配置和三重启预算下，K=10的邻域选择较稳健”的结论；不能扩展成十种子验证了原全部八K。若KMeans/GMM都稳定且指标接近，应保留近似并列，不把稳定性写成隐藏类别准确率证明。耗时只按框架记录留存，不作为排除预热、同步和执行顺序影响的严格效率基准。

## 本地验证（实际已执行）

环境：Windows、Python3.12.14、torch2.11.0+cpu；没有CUDA或正式拟合。

| 检查 | 实际结果 |
|---|---|
| Python AST、固定配置、LF | 通过；60候选/180重启，sbatch无CR字节 |
| `python p2_robustness.py smoke --output outputs_T0016_local_smoke_20261123_02` | 退出0；90×2 toy，2方法×2K×2种子=8候选、16重启；全部指标有效，导出来源核验通过 |
| 标签编号置换及不同分区 ARI | 置换ARI=1，不同分区ARI<0；通过 |
| 损坏标签文件 | 校验拒绝，未输出伪完整分析 |
| 删除候选记录并重封哈希 | 仍因缺少计划候选被拒绝 |
| `bash -n run_p2_T0016.sbatch` | 退出0 |
| 缺提交目录 / 缺Conda | 均退出1 |
| CPU torch 下GPU检查 / 模拟tee失败 | 均退出2，没有进入正式运行 |
| 直接正式入口，无Slurm / 模拟作业号但无CUDA | 均在创建输出和加载正式数据前拒绝 |

首轮toy核心流水线通过后，Windows沙箱限制了 `tempfile` 新目录访问；将损坏样例保留在正常创建的独立冒烟目录后，第二轮全项通过。日志失败测试首次通过PATH模拟未生效，改为明确的Bash函数后验证退出2。上述是本地测试修正，不是正式聚类结果。

本地冒烟产物：`code/q2_clustering/outputs_T0016_local_smoke_20261123_02/`；失败路径证据：`_tmp/T0016/preflight_02/checks.json` 及同目录日志。本地日志与toy产物均忽略，不纳入正式结果。

## 提交与取回

本地由用户执行 `git push`，再在集群执行：

```bash
ssh ustc
cd ~/USTC-AI3002-26fall
git status --short
git pull --ff-only
cd LABs/LAB2/code/q2_clustering
sbatch run_p2_T0016.sbatch
squeue -u pb25030753
```

若集群有未提交改动，先检查并协调再pull，不覆盖。作业通过 `SLURM_SUBMIT_DIR` 定位仓库，不依赖Slurm spool脚本位置。登录节点不运行训练或CUDA检查。作业号与实际路径以 sbatch 输出和日志为准。

取回时在本地 LAB2 目录执行（将占位符替换成实际值）：

```powershell
scp -r 'ustc:~/USTC-AI3002-26fall/LABs/LAB2/code/q2_clustering/outputs_T0016_<stamp>_<jobid>' code/q2_clustering/
scp -r 'ustc:~/USTC-AI3002-26fall/LABs/LAB2/logs/T0016/<stamp>_<jobid>' logs/T0016/
```

取回整个运行目录（同时含p2与p2_analysis），不要只取最终标签。随后可在本地重建一份新的只读分析：

```powershell
cd code/q2_clustering
python p2_robustness.py analyze --source outputs_T0016_<stamp>_<jobid>/p2 --output outputs_T0016_<stamp>_<jobid>/p2_analysis_local
```

正式作业是否成功、耗时、GPU环境、60候选指标与稳定性都需在取回后核验；本文件不预填结论。
