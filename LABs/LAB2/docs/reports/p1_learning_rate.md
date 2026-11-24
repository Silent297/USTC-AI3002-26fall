# P1：小学习率导致慢收敛的配对实验

2026-11-24，T0018，codex。**运行准备与本地toy自检完成；正式Slurm尚未提交，没有正式实验结论。**

## 要回答的问题

实验B(a)中的MSE下降缓慢，单凭曲线不能诊断梯度消失。本实验使用没有隐藏层的线性模型，排除多层反传链的衰减，构造“梯度仍明显、但学习率小导致每步更新小”的对照。目标是验证这一诊断边界，不是替原八层Sigmoid实验重新判定原因。

## 固定设计

| 因素 | 设置 |
|---|---|
| 模型 | `nn.Linear(2,1)`，含偏置共3参数，无隐藏层/激活函数 |
| 数据 | 课程固定gradient数据：1024训练点、4225网格点；目标 `x1 + 0.5*x2` |
| 损失与优化 | 全批量平均MSE；SGD，无动量、无权重衰减、无调度 |
| 学习率 | 0.05 与 0.00001 |
| 种子 | 17、42、123；CPU默认Linear初始化，每种子两学习率共享同一参数副本 |
| 预算 | 每轨迹1000次更新，共6轨迹；固定种子与学习率，不根据结果替换 |
| 精度 | float32；1计算线程、确定性算法、禁用TF32 |
| 正式环境 | Slurm，stu/Students/qos_stu_default，A100×1、4CPU、16G、30分钟；激活ai26 |

课程数据SHA-256：`b3b3d27308694a343dfd1fc5b0376929b9a25f5cdd15f1e54bcd8e319e02977f`。正式入口在创建输出/加载数据前检查Slurm与A100，再核对数据哈希、形状和精度；无CPU回退。

全部实验代码独立放在 [p1_learning_rate.py](../../code/q1_neural_network/p1_learning_rate.py)，配置见 [p1_config.json](../../code/q1_neural_network/p1_config.json)，作业见 [run_p1_T0018.sbatch](../../code/q1_neural_network/run_p1_T0018.sbatch)。没有改写学生的models.py/train.py或课程初值。

## 测量语义

每条轨迹记录step=0…1000，共1001行：

- 当前参数 `(w1,w2,bias)`、训练/网格MSE。
- 当前训练MSE对三参数的梯度向量及整体L2范数；这是参数梯度，不是原MLP的隐藏层梯度RMS或归一化rho，两者不能混为同一指标。
- `expected_update_norm = lr * gradient_norm`，以及执行该步后真实参数差的L2范数。
- step=1000只测量，不再更新，实际更新栏为空；前1000行各对应一次SGD更新。
- 保存每条轨迹首尾state_dict和网格预测；保存输入数组、源码/配置快照、作业号、Git版本、环境及全部产物SHA-256。

首步两学习率的参数和梯度应完全一致；理想SGD的更新范数比例为5000。实际float32参数减法有舍入，实际比例可以略偏离5000。接近最优时可能出现非零小梯度却无法改变float32参数，记录实际差值以识别这种精度下限，不强制实际更新永远等于数学乘积。

参考值用float64最小二乘求解并记录训练/网格残差、设计矩阵秩与Hessian特征值，同时测量已知参数 `[1,0.5,0]` 的残差。课程float32目标的舍入可能使最小残差非严格零，不能把它误判为模型表达不足。

## 结果检查与解释

`check` 是只读CPU代数审计，不继续训练、不调用GPU：核对完整轨迹和哈希、两学习率配对、每步训练/网格误差、独立公式 `2*A.T@(A*theta-y)/N` 的梯度、SGD参数递推与实际更新、首尾权重和预测、最小二乘参考。新检查报告写在独立目录，原输出前后哈希不变。

梯度检查rtol=1e-4、atol=2e-6；MSE检查rtol=1e-4、atol=1e-10；参数递推rtol/atol=2e-7；首尾预测rtol=1e-4、atol=5e-7。跨Windows/Linux的最小二乘/SVD参考允许rtol=1e-10、atol=1e-12，避免要求不同BLAS逐位相同。它验证float32轨迹与独立代数的一致性，不证明CUDA每个算子的逐位一致。

正式取回后应逐种子报告：

1. 首步相同梯度下，两学习率实际更新量的差别。
2. 全程损失、梯度范数、实际更新量的曲线；早期相同步数比较，不能要求已经收敛的正常学习率终点梯度仍与慢速组相近。
3. 1000步首尾训练/网格MSE、最终参数到参考解的距离、训练误差的相对下降。
4. 正常学习率能否接近参考残差，小学习率是否仍有明显梯度但下降缓慢；即使结果不符合预期也完整保留，不把自检通过等同于假设得到支持。

若观察符合设计，支持“慢收敛可以由小学习率造成，不能只凭MSE曲线诊断梯度消失”。它不能证明原八层Sigmoid仅因学习率过小，也不能否定原梯度实测；不将Sigmoid饱和当成与梯度消失独立的原因。

## 本地实际验证

环境：Windows、Python3.12.14、torch2.11.0+cpu。只用4个合成点、一个种子、两学习率、各2步，不是正式1024点六轨迹。

| 检查 | 实际结果 |
|---|---|
| `python p1_learning_rate.py smoke --output outputs_T0018_smoke_20261124_03` | 退出0；两学习率首参/首梯度完全一致；每条3行、2次更新 |
| 对拍现有课程SGD训练 | 两种学习率终点参数均逐元素相同 |
| 全历史独立代数审计、首尾权重/预测、参考解 | 通过；梯度最大绝对差6.09e-8；原文件哈希未变 |
| 损坏history文件 | 检查器按哈希不一致拒绝；随后恢复原toy文件 |
| `bash -n run_p1_T0018.sbatch`、LF | 退出0，无CR字节 |
| 缺提交目录 / 缺Conda | 均退出1 |
| CPU环境CUDA检查 / 模拟tee失败 | 均退出2，没有进入正式训练 |
| 正式入口无Slurm / 仅模拟作业号但无CUDA | 均在创建输出前拒绝 |

toy产物：`code/q1_neural_network/outputs_T0018_smoke_20261124_03/`；环境失败检查：`_tmp/T0018/preflight_01/checks.json`。这些仅本地保留、忽略，不计作P1正式完成。

## 提交、取回与复核

本地由用户 `git push`。集群由用户执行（先确认没有未提交改动冲突）：

```bash
ssh ustc
cd ~/USTC-AI3002-26fall
git status --short
git pull --ff-only
cd LABs/LAB2/code/q1_neural_network
sbatch run_p1_T0018.sbatch
squeue -u pb25030753
```

脚本按SLURM_SUBMIT_DIR定位，不从spool脚本位置反推。每次新建时间戳/JobID目录，不覆盖或断点续跑：

```text
code/q1_neural_network/outputs_T0018_<stamp>_<jobid>/
  p1/                       # data/config/reference/manifest、五份源码快照、哈希
    normal_seed17/          # 首尾.pt、history.csv、首尾网格预测
    small_seed17/           # 其余seed42/123同理，共六目录
  p1_check/verification.json # 含六轨迹验证及三配对首尾汇总
logs/T0018/<stamp>_<jobid>/   # job.log和Slurm .out/.err
```

取回时在本地LAB2执行（替换真实stamp/jobid，不把占位符原样执行）：

```powershell
New-Item -ItemType Directory -Force logs/T0018
scp -r 'ustc:~/USTC-AI3002-26fall/LABs/LAB2/code/q1_neural_network/outputs_T0018_<stamp>_<jobid>' code/q1_neural_network/
scp -r 'ustc:~/USTC-AI3002-26fall/LABs/LAB2/logs/T0018/<stamp>_<jobid>' logs/T0018/
```

完整运行目录和日志一并取回；不要只取verification.json。随后可在本地创建一份新的只读检查：

```powershell
cd code/q1_neural_network
python p1_learning_rate.py check --source outputs_T0018_<stamp>_<jobid>/p1 --output outputs_T0018_<stamp>_<jobid>/p1_check_local
```

当前未提交作业，真实A100环境、耗时、六轨迹收敛与反例是否成立均待正式运行确认。统一报告/PDF与原实验输出保持现状。
