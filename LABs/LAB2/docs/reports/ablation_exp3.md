# 实验 3：深度与激活对梯度传播的影响

比较2/4/8个隐藏层与Sigmoid/ReLU在三个固定种子下的梯度传播和拟合表现。每个深度、种子先保存一份CPU Linear初值，再让两种激活严格加载它；使用课程gradient数据、SGD、lr=0.05、全批量训练1000步。

**当前为T0013提交准备：本地toy与无GPU检查通过，尚未提交Slurm，没有正式实验3结果。** 不修改实验1/2和统一报告的待验收基线。

## 预设设计与测量

| 项目 | 设置 |
|---|---|
| 深度 / 宽度 | 2、4、8隐藏层，各32单元；参数量1185、3297、7521 |
| 激活 / 种子 | Sigmoid、ReLU；种子17、42、123 |
| 配对初值 | 9份保存的CPU默认Linear初值；每份供两种激活加载，逐张量内容哈希完全相同 |
| 训练 | 18条轨迹，单次连续1000步SGD、lr=0.05，无动量/权重衰减；调用原train.py |
| 数据 | 课程gradient.npz，1024点全批量；4225网格只评价；不重采样或选最好种子 |
| 诊断 | 原训练点前128行，在0和1000步各测一次，固定索引另存 |
| 精度 / 设备 | Slurm A100、float32、单线程、确定性算法、TF32关闭 |
| 快照 / 历史 | 36份首尾模型快照；每100步训练MSE，共每轨迹11条历史 |

测量复用utils/diagnostics.py中的课程口径：各隐藏层激活前z的梯度RMS、相对输出RMS的比值ρ、局部激活导数均值。额外记录激活前/后均值与标准差、每层权重Frobenius范数、输出层权重范数，以及ReLU在本诊断批次上始终为零的单元比例。后者只表示这128点上的不活跃，不能直接说在整个输入域上永久死亡。

同时记录训练/网格MSE、最大绝对误差、轨迹与全流程耗时。误差指标采用float64归约，训练与梯度诊断仍为float32，不把它写成双精度训练。计时含首尾诊断和保存开销，不能作为纯SGD算力指标。

若输出梯度RMS=0，ρ保持null；真实零梯度也保持0。对数图只绘制正值，不把0/null替换为极小正数；原值在JSON保留，图中标出零/未定义数量。float32平方归约下的零值也不自动等同数学精确零。

## 代码与本地检查

独立入口为code/q1_neural_network/ablation_exp3.py，配置为ablation_exp3_config.json。不修改models.py、train.py、utils/diagnostics.py；复用已有训练与诊断，新增部分只组织配对、记录和核验。数据SHA-256固定为b3b3d27308694a343dfd1fc5b0376929b9a25f5cdd15f1e54bcd8e319e02977f。

本地toy（每次替换新标识）：

```powershell
& "$HOME/miniconda3/envs/ai26/python.exe" code/q1_neural_network/ablation_exp3.py --smoke --output _tmp/T0013/toy_<新标识>
```

实际已验证四个合成点、两种激活各两步SGD：初值配对相同，终点与直接调用原train一致；附加诊断不会改变参数或原.grad；零损失时输出RMS=0、各层ρ=null。未读取正式数据做本地训练。

sbatch的bash-n与LF检查通过；缺提交目录/缺Conda退出1，CPU torch的CUDA门槛退出2，正式入口本机拒绝且不创建输出目录。正式18轨迹、CUDA诊断与36快照的完整核验尚未执行。

## 集群提交

沿用[ustc-slurm-job-workflow](<C:/Users/35150/Desktop/Agentic Coding/skills/ustc-slurm-job-workflow/SKILL.md>)：交互式集群命令由用户手动执行；本地不自动push或SSH代办。脚本run_ablation_exp3_T0013.sbatch申请stu账户、Students分区、qos_stu_default、A100一张、4CPU、16G与30分钟。时长是申请上限，不是预估。

本地提交后由本人执行：

```powershell
git push
```

集群侧在确认工作树可安全更新后：

```bash
ssh ustc
cd ~/USTC-AI3002-26fall
git status --short
git pull --ff-only
cd LABs/LAB2/code/q1_neural_network
sbatch run_ablation_exp3_T0013.sbatch
squeue -u pb25030753
```

脚本先激活~/miniconda3的ai26，打印并断言CUDA/A100；用SLURM_SUBMIT_DIR定位，不从spool目录推算仓库。日志进入LABs/LAB2/logs/T0013/<时间戳_JobID>/，含job.log与归档.out/.err；请一并取回，避免只有manifest而缺原作业日志。

输出为code/q1_neural_network/outputs_T0013_<时间戳_JobID>/ablation_exp3/。驱动保存全部失败记录，只有18条成功且检查器通过才打印ALL DONE与ARTIFACT_DIR。程序与tee日志任一失败都返回非零。失败/超时保留目录，重投换新运行标识；不覆盖或断点续训，不删除坏种子。

## 输出布局与核验

```text
ablation_exp3/
  config.json
  manifest.json
  source_snapshot/
  diagnostic_indices.npy
  paired_initial/              # 9份共享Linear初值
  depth2_sigmoid_seed17/       # 其余17条轨迹同类
    step_0.pt
    step_1000.pt
    stages.json               # 首尾指标、逐层诊断与统计
    history.json
    predictions.npz
  summary.json                # 36条首尾记录
  gradient_profiles.png
  depth_comparison.png
  verification.json
```

原始输出与权重不进Git。集群会自动--check-output；取回后本地也可CPU轻量核验：

```powershell
& "$HOME/miniconda3/envs/ai26/python.exe" code/q1_neural_network/ablation_exp3.py --check-output --output code/q1_neural_network/outputs_T0013_<运行标识>/ablation_exp3
```

该入口会重新绘图与写verification，因此对原始产物的完整副本运行；先保留收到的verification、图及日志。检查内容包括源码快照/配置/数据哈希、18轨迹/36快照、128固定索引、历史步数、配对初值逐字节内容相同、模型结构与参数量、预测和网格MSE、诊断层数、ρ与绝对RMS的一致性。校验不是训练重现，也不是来源身份认证。

## 结果取回后的分析顺序

1. 分深度和种子展示两种激活的初始/最终逐层梯度剖面；首层ρ随深度的散点不得筛种子，0/null须单列。
2. 结合绝对RMS、输出端RMS、局部导数、激活分布和权重范数解释相对梯度，防止只看分母变化就称梯度增强。
3. 对照18条最终train/grid MSE及训练曲线，区分梯度传播证据与训练结果。相同Linear权重、不同激活也可能有不同初始预测与损失，不应期待初始梯度相同。
4. 讨论哪些解释被支持、哪些仍不足：增加深度还改变参数量和权重乘积；Sigmoid导数≤0.25本身不能单独证明整体梯度必然衰减，ReLU也不保证所有路径有效。

跨深度比较不是等参数量实验；每层32单元保持固定是本实验口径。三种子为描述性重复，不报告普遍成功率或置信区间。全部结论等真实产物后填写，不预设Sigmoid必然失败或ReLU必然成功。
