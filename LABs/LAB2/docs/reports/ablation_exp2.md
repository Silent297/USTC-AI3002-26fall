# 实验 2：等参数量、多初始化与配对训练预算

本实验比较浅层[16]与两隐藏层[8,4]在五个初始化下的训练表现，两结构均65参数。每条轨迹连续训练9000步，在第0、3000、9000步保存模型和同一个Adam的状态，以区分初始化差异、训练预算和结构因素。

**当前状态：T0011 已完成本地驱动与 Slurm 提交准备；未提交集群、没有正式GPU结果。** 本地只运行四点两步toy冒烟，不将其作为本实验结论。原三项设计见[拓展方案](ablation_plan.md)，实验1与统一报告的验收基线保持不变。

## 预先固定的设置

| 项目 | 设置 |
|---|---|
| 结构 | shallow=[16]，deep=[8,4]，ReLU，线性输出，各65参数 |
| 种子 | 17、42、123、2026、20261123，全部展示，不选最好一次 |
| 初始化 | 每条轨迹重设种子，使用PyTorch Linear默认初始化；在CPU构造后迁移CUDA，不载入课程/手工初值 |
| 数据 | 课程expression.npz，固定1024点全批量训练、4225网格评价 |
| 优化 | 同一Adam连续9000次更新，lr=0.01，float32平均MSE；不中断重建优化器 |
| 快照 / 历史 | 0/3000/9000；每100步记录训练MSE，均指已完成更新次数 |
| 设备 | Slurm A100，float32，torch单线程；禁用TF32，启用确定性算法 |
| 规模 | 10条轨迹；30个快照含10个初始状态，20条非零预算结果 |
| 指标 | train/grid MSE、网格最大绝对误差、峰区(f≥0.5)MSE、背景(f=0)MSE及累计耗时 |

配置归档于code/q1_neural_network/ablation_exp2_config.json。新增驱动为ablation_exp2.py，不改课程models.py或train.py；由于现有train每次调用都会重建优化器，驱动独立组织一次连续更新循环，并用toy与已有train进行对拍。

不同结构下“同种子”不等于“相同初值”；只有同一条轨迹的3000/9000步是严格配对的预算比较。评价不反向传播、不按网格选种子或改变预算。网格误差使用float64差值归约，训练仍用float32损失；不能把归约精度写成双精度训练。

## 本地已验证的范围

- CPU四点、两步、单种子浅层toy：新增驱动与现有train两步Adam终点参数逐元素一致；0/1/2快照和Adam状态计数连续。
- 正式入口要求SLURM_JOB_ID与可用CUDA/A100，没有本机CPU回退；本地直接调用正式入口被拒绝，未创建输出目录。
- sbatch通过bash -n与LF检查；缺提交目录、缺Conda均退出1；使用真实CPU torch检查CUDA门槛退出2，未进入正式训练。Conda失败路径的测试使用临时HOME，CPU门槛测试使用不激活环境的临时Conda函数及明确的ai26解释器。
- 尚未在计算节点验证确定性算法兼容性、CUDA训练速度、收敛、全部快照和正式输出校验；这些必须以真实作业为准。

本地toy命令（每次换新目录，不能指向正式输出）：

```powershell
& "$HOME/miniconda3/envs/ai26/python.exe" code/q1_neural_network/ablation_exp2.py --smoke --output _tmp/T0011/toy_<新标识>
```

## 集群执行步骤

按[Slurm技能](<C:/Users/35150/Desktop/Agentic Coding/skills/ustc-slurm-job-workflow/SKILL.md>)，集群交互命令由本人执行。本地不自动push，不通过SSH代办。脚本run_ablation_exp2_T0011.sbatch请求stu账户、Students分区、qos_stu_default、A100一张、4CPU、16G和30分钟；30分钟是申请上限，不是耗时预估。

本地确认任务commit后，由本人推送：

```powershell
git push
```

登录集群并执行（已有远程修改时先协调，不直接覆盖）：

```bash
ssh ustc
cd ~/USTC-AI3002-26fall
git status --short
# 工作树可安全更新后：
git pull --ff-only
cd LABs/LAB2/code/q1_neural_network
sbatch run_ablation_exp2_T0011.sbatch
squeue -u pb25030753
```

脚本显式激活~/miniconda3的ai26、打印CUDA/设备、断言A100。Slurm复制脚本到spool后仍用SLURM_SUBMIT_DIR定位源码；正式驱动逐条记录成功/失败，失败不筛掉；最终只有10条全部成功且核验通过才打印ALL DONE。tee的Python和日志退出码均检查，避免把日志管道成功当训练成功。

日志位于LABs/LAB2/logs/T0011/<时间戳_JobID>/，包括job.log和归档的.out/.err。成功时打印ARTIFACT_DIR，目录形式为：

```text
code/q1_neural_network/outputs_T0011_<时间戳_JobID>/ablation_exp2/
  config.json
  manifest.json
  source_snapshot/
  shallow_seed17/       # 其余结构/种子同类
    step_0.pt
    step_3000.pt
    step_9000.pt
    checkpoints.json
    history.json
    predictions.npz
  summary.csv
  statistics.json
  paired_budgets.png
  training.png
  verification.json
```

异常/超时保留半成品与日志，重投使用新目录；本版本不支持断点续训或跳过旧候选。不要删除坏种子后声称10条已完成。

## 取回与分析

作业退出squeue不等于成功，应查看job.log末尾、manifest状态及verification.json。将sbatch返回的JobID、ARTIFACT_DIR和日志末尾反馈后再分析。取回整个ablation_exp2目录及对应日志，不能只取汇总表；产物走scp，不进Git。

例如将下列<运行标识>替换为日志中的真实目录名，本地先创建同名输出父目录，再取回：

```powershell
New-Item -ItemType Directory -Force code/q1_neural_network/outputs_T0011_<运行标识>
scp -r 'ustc:~/USTC-AI3002-26fall/LABs/LAB2/code/q1_neural_network/outputs_T0011_<运行标识>/ablation_exp2' code/q1_neural_network/outputs_T0011_<运行标识>/
```

本地可用--check-output轻量复核，不调用GPU或训练：

```powershell
& "$HOME/miniconda3/envs/ai26/python.exe" code/q1_neural_network/ablation_exp2.py --check-output --output code/q1_neural_network/outputs_T0011_<运行标识>/ablation_exp2
```

检查器复核来源快照哈希、配置、数据哈希、10条/30快照、历史步数、预测有限性、MSE/最大误差、模型结构和Adam计数，生成汇总与图。verification.json记录完整文件哈希；重新绘图可能使图文件哈希变化，应先保留集群原verification.json与原图来源，再记录本地核验版本。当前校验不是数字签名，仍需保留原作业记录。

正式结果取回后按以下顺序审阅：

1. 每结构五个种子在3000和9000步的网格MSE散点、中位数和范围；不把两个结构共享种子编号误当共享参数。
2. 同轨迹3000→9000的配对变化，结合train/grid MSE、最大误差与峰区/背景读数判断预算效果。
3. 查看所有失败、反弹、异常种子，并报告耗时口径。累计耗时包含评价及此前快照I/O，不是纯优化器时间。
4. 若深层多数种子更好，支持本配置下的稳健优势；若浅层某些种子改善，说明原平台可能有优化因素；若深层也失败，区分存在表达与可学到性。

不按预期结论筛选结果，不把十条轨迹推广为所有宽度、初始化、优化器的普遍结论，也不把新增实验源稿自动并入待验收统一报告。
