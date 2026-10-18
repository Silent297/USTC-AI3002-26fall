# A8 逐图解释说明（优化器对照：GD vs Momentum vs Adam）

> 实验：`experiments/A8/` —— 拓展实验 A 系列（附加探究，非课程必交内容）。问题：在**同一目标、同一初始化、同一学习率、同一步数预算**下，全批量 GD、heavy-ball momentum（β=0.9）与 Adam 的收敛速度、稳定边界和最终指标差多少？课程 `model.py` 用的是纯 GD；本实验回答"换优化器会不会改变结论"。
> 来源：《可做实验清单》A8（指南3 §4；参考报告非官方）。
> 协议：两个任务——world L11（线性，λ2=1.0）与 truth L20（logistic，λ2=1e-3），特征按 train 统计量 z-score，λ1=0（L1 见 A6）。目标函数与 `model.py` 一致：world 按单目标 1/N 的 Frobenius 和（两个坐标分别对应 `model.py` 的一次单目标拟合），truth 为 1/N 交叉熵；表中 MSE 仍是**元素平均**口径，因此与 A1 ridge 锚点 167.1707 可直接对拍。全批量 torch float64，4 threads。学习率统一写成 $f=\mathrm{lr}/\mathrm{lr}_{crit}$，$\mathrm{lr}_{crit}=2/L$，$L$ 为光滑目标的 Lipschitz 上界：线性 $L=2\lambda_{\max}([X,1])/N+2\lambda_2$，logistic $L=0.25\lambda_{\max}([X,1])/N+2\lambda_2$（$p(1-p)\le0.25$ 的保守界）。参考最优：world 用闭式 dual ridge，truth 用 L-BFGS-B（`gtol=1e-10`，319 次迭代，$\|\nabla\|_\infty=3.2\times10^{-9}$）。
> 两种初始化：**稳定性扫描**用 $w,b\sim\mathcal N(0,10^{-3})$（seed 1234），以激发所有特征方向；**头对头比较**用 `model.py` 的 $w=b=0$。稳定性扫描每点 600 步、不早停；头对头每任务最多 8000 步、$f=0.9$、允许早停。valid 只报告，不参与选择。
> 产物与复现：`python experiments/A8/a8_optimizers.py`（本地 CPU **529.6 s**）；`--replot` 只重画图，`--smoke` 为 4 点快速冒烟；`a8_results.json`（protocol/reference/sweep/init_probe/head + 逐步 history）、`a8_run.log`；图在 `figures/`。
> 维护约定：每完成一批绘图，在实验目录下新建 `aX_figure_notes.md`，格式同本文与 `experiments/A7/a7_figure_notes.md`。

## 总览

| 图 | 一句话 | 回答的内容 |
|---|---|---|
| `a8_convergence` | 三个优化器在 $f=0.9$ 的 best-so-far 损失间隙（细线为瞬时值） | "同一 lr 下谁先到 $10^{-2}/10^{-3}/10^{-4}$" |
| `a8_stability` | 600 步后的间隙随 $f$ 的变化，× 为发散 | "各自的稳定边界在哪里，是否等于理论临界值" |
| `a8_final_metrics` | 三个优化器收敛后的 valid 指标（点）vs 参考最优（虚线） | "优化器是否改变最终解与泛化指标" |

**一句话主线**：优化器选择**只影响速度、不改变凸问题的最终解**——三者在收敛后都到达参考最优（world valid MSE 167.17、truth valid acc 86.33%/logloss 0.4844）。但速度的任务依赖性极强：在**病态二次型 world** 上 momentum 最快（到 $10^{-4}$ 用 90 步，GD 500 步、Adam 6550 步），Adam 反而最慢；在**logistic truth** 上 Adam 最快（260 步到 $10^{-4}$，momentum 1370 步、GD 8000 步仍未到 $10^{-4}$）。稳定边界方面，world 上 GD 在 $f\approx1$ 处精确发散（理论值），momentum 在 $f\approx1.9$ 附近（理论 $1+\beta=1.9$），Adam 在 $f\le3$ 全程有限但收敛很慢；truth 的 $0.25$ 界是保守的，GD/momentum 到 $f=3$ 都不发散，而 Adam 在某些 $f$（0.9/1.3/2.2/3.0）出现**非单调的损失震荡**。

---

## 0. 全量读数（先看表）

### 0.1 临界学习率与参考最优

| task | $\lambda_{\max}([X,1])$ | $L$ | $\mathrm{lr}_{crit}=2/L$ | $f_0$（零输出损失） | 参考最优损失 |
|---|---|---|---|---|---|
| world L11 | 139,309.94 | 559.2397 | **3.5763e-3** | 7620.934 | 398.0165（dual ridge） |
| truth L20 | 726,381.04 | 35.6929 | **5.60336e-2** | 0.693147 | 0.0660047（L-BFGS-B） |

> world 参考最优指标：train MSE 91.2115 / valid MSE **167.17072** / 中位 **923.12059 km** / 平均 1203.0653 km（与 A1 锚点逐位一致）。truth 参考最优：train acc 1.0 / logloss 0.033827；valid acc **0.863333** / logloss **0.484373**。

### 0.2 头对头（$f=0.9$，零初始化，最多 8000 步）

| task | optimizer | 实际步数 | 早停 | 最终 gap | 到 $10^{-2}$ | 到 $10^{-3}$ | 到 $10^{-4}$ | valid 指标 |
|---|---|---|---|---|---|---|---|---|
| world | GD | 2335 | 是 | 4.47e-15 | 150 | 320 | 500 | MSE 167.17072 / 中位 923.12077 / 平均 1203.0653 |
| world | Momentum | **501** | 是 | **0** | **40** | **70** | **90** | MSE 167.17072 / 中位 923.12059 / 平均 1203.0653 |
| world | Adam | 8000 | 否 | 8.79e-7 | 3030 | 5270 | 6550 | MSE **167.14163** / 中位 926.894 / 平均 1202.681 |
| truth | GD | 8000 | 否 | 9.30e-4 | 3500 | 7850 | — | acc **0.866667** / logloss **0.463776** |
| truth | Momentum | 8000 | 否 | **2.20e-10** | 350 | 770 | 1370 | acc 0.863333 / logloss 0.484369 |
| truth | Adam | 8000 | 否 | 1.08e-8 | **80** | **120** | **260** | acc 0.863333 / logloss 0.484374 |

> 读法：world 上 momentum 到 $10^{-4}$ 只用 90 步（GD 500、Adam 6550）；truth 上顺序完全反过来（Adam 260、momentum 1370、GD 8000 仍未到）。所有收敛到最优的解给出相同 valid 指标（world 167.1707 / truth 0.8633 / 0.4844），差异只在第 4–5 位有效数字。truth GD 的 acc 86.67% 与 logloss 0.4638 **不是更好的解**，而是 8000 步仍未收敛（gap 9.3e-4）带来的隐式早停效应。

### 0.3 稳定性扫描（600 步后间隙；`DIV@n` = 第 n 步发散；× = 图上标在 10）

| task | optimizer | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 | 1.0 | 1.1 | 1.3 | 1.5 | 1.9 | 2.2 | 3.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| world | GD | 0.0276 | 4.49e-3 | 7.84e-4 | 1.38e-4 | 2.45e-5 | 1.44e-2 | DIV@51 | DIV@21 | DIV@15 | DIV@10 | DIV@9 | DIV@7 |
| world | Momentum | 6.28e-6 | 3.93e-17 | 0 | 0 | 7.87e-18 | 0 | 0 | 0 | 0 | **5.19** | DIV@13 | DIV@8 |
| world | Adam | 0.171 | 0.0708 | 0.0577 | 0.0508 | 0.0458 | 0.0438 | 0.0419 | 0.0386 | 0.0357 | 0.0307 | 0.0274 | 0.0202 |
| truth | GD | 0.252 | 0.174 | 0.136 | 0.112 | 0.0951 | 0.0880 | 0.0817 | 0.0711 | 0.0624 | 0.0490 | 0.0414 | 0.0272 |
| truth | Momentum | 0.0880 | 0.0270 | 0.0107 | 4.74e-3 | 2.27e-3 | 1.60e-3 | 1.15e-3 | 6.20e-4 | 3.52e-4 | 1.31e-4 | 6.90e-5 | 1.62e-5 |
| truth | Adam | 4.52e-4 | 4.76e-6 | 8.09e-6 | 3.57e-8 | **0.291** | 6.14e-4 | 6.26e-5 | **2.28** | 3.67e-4 | 6.37e-4 | **0.215** | **1.17** |

> 读法：world 是教科书式的二次型——GD 的稳定边界精确落在 $f=1$（0.9 稳定、1.0 边缘振荡、1.1 发散），momentum 的边界在 $f\approx1.9$（$2(1+\beta)/L$ 归一化后正好 $1+\beta$），Adam 在 $f\le3$ 不发散但 600 步只把 gap 压到 $0.02$–$0.17$。truth 的 $0.25$ 界过保守：GD 和 momentum 到 $f=3$ 都不发散且 gap 随 $f$ 单调下降；Adam 的 gap 随 $f$ 强烈非单调（0.7 处 $3.6\times10^{-8}$，0.9 处 0.291，1.3 处 2.28，3.0 处 1.17），说明"有限损失"不等于"稳定收敛"。

### 0.4 初始化探针（GD，$f=1.5$）

| task | init | 结果 |
|---|---|---|
| world | $w=b=0$ | **DIV@15** |
| world | $w,b\sim\mathcal N(0,10^{-3})$ | **DIV@15** |
| truth | $w=b=0$ | 稳定，gap 0.06242 |
| truth | $w,b\sim\mathcal N(0,10^{-3})$ | 稳定，gap 0.06240 |

> 读法：world 的发散边界不是"零初始化恰好躲开顶部特征方向"造成的假象——两种初始化都在第 15 步炸掉，因为零点的梯度 $X^\top y$ 沿顶部特征向量本身就有分量。truth 在 $f=1.5$ 两种初始化都稳定，再次说明 logistic 的 $f=1$ 只是保守上界。

---

## 1. `a8_convergence`：同 lr 下的收敛速度

**是什么**：左 world、右 truth，横轴步数（log），纵轴 best-so-far 归一化损失间隙 $(f-f^*)/(f_0-f^*)$（log）；粗线为截至该步的最小间隙，细淡线为瞬时间隙。虚线标 $10^{-2}$ 与 $10^{-4}$。三条曲线同一 $f=0.9$、同一零初始化、同样最多 8000 步。

**精确读数**（到各阈值的步数，来自 `a8_results.json` 的 `steps_to_1e-*`）：
- world：momentum **40 / 70 / 90** 步到 $10^{-2}/10^{-3}/10^{-4}$；GD 150 / 320 / 500；Adam 3030 / 5270 / 6550。momentum 在 501 步早停（损失已到机器精度 0），GD 在 2335 步早停，Adam 跑满 8000 步（最终 8.8e-7）。
- truth：Adam **80 / 120 / 260**；momentum 350 / 770 / 1370；GD 3500 / 7850 / 未到 $10^{-4}$（8000 步时 9.3e-4）。
- 瞬时细线：truth 的 Adam 在 8000 步内多次冲到 $10^{-2}$ 以上再落回，best-so-far 因而呈台阶；其余曲线基本单调。

**结论**：
1. **world 二次型（条件数约 280）**：momentum 利用历史梯度沿长轴加速，比 GD 快 5.6×（到 $10^{-4}$ 90 vs 500 步）；Adam 的逐坐标归一化在高度相关的特征上失效（有效步长被各坐标的梯度尺度差异拉平），反而最慢（6550 步）。
2. **truth logistic**：Adam 最快（260 步），momentum 次之（1370），GD 最慢（8000 步未到 $10^{-4}$）。分类损失的曲率在最优附近迅速变小，Adam 的归一化步长在平坦方向仍能推进；GD 步长被全局 Lipschitz 界限制，无法利用局部平坦性。
3. **同一 lr 不代表同一有效步长**：$f$ 只按 GD 的全局曲率归一化，Adam 的更新幅度近似为 lr（与梯度尺度无关），所以在 world 上"过小"、在 truth 上"刚好"。这正是"优化器没有全局优劣"的直接证据。

**报告用法**：放在"优化器/训练策略"小节，配一句"momentum 适合病态二次型，Adam 适合分类/平坦方向；本课程 world 用 momentum 可把 8000 步预算缩到 ~100 步，truth 用 Adam 可缩到 ~300 步"。不要写成"Adam 比 GD 好/差"。

**图注候选**：
- 中文：A8 同学习率（$f=0.9$）下的收敛曲线。粗线为 best-so-far 归一化损失间隙，细线为瞬时值；world 上 momentum 最快、Adam 最慢，truth 上顺序反转（Adam 最快、GD 最慢）。
- English: A8 convergence at the same normalized learning rate ($f=0.9$). Thick lines: best-so-far normalized loss gap; thin lines: instantaneous gap. Momentum wins on the ill-conditioned quadratic (world), while Adam wins on the logistic task (truth).

---

## 2. `a8_stability`：稳定边界 vs 归一化学习率

**是什么**：左 world、右 truth，横轴 $f=\mathrm{lr}/\mathrm{lr}_{crit}$，纵轴 600 步后的归一化间隙（log）；曲线上的 × 表示该点已发散（画在 10）。黑色竖向虚线 $f=1$ 是 GD 的理论临界值，灰色点线 $f=1.9$ 是 momentum 的理论临界值 $1+\beta$。扫描用 $\sigma=10^{-3}$ 随机初始化以激发所有方向。

**精确读数**：
- world：GD 在 0.9 处 gap 2.45e-5，1.0 处 1.44e-2（边缘振荡），1.1 起全部 `DIV@51/21/15/10/9/7`；momentum 到 1.5 全为 0，1.9 处 5.19（边缘），2.2/3.0 `DIV@13/8`；Adam 从 0.1 的 0.171 单调降到 3.0 的 0.0202，全程未发散。
- truth：GD 从 0.252（$f=0.1$）单调降到 0.0272（$f=3$），全程未发散；momentum 从 0.0880 降到 1.62e-5，全程未发散；Adam 非单调：4.52e-4 → 4.76e-6 → 8.09e-6 → **3.57e-8**（0.7）→ **0.291**（0.9）→ 6.14e-4 → 6.26e-5 → **2.28**（1.3）→ 3.67e-4 → 6.37e-4 → **0.215**（2.2）→ **1.17**（3.0）。

**结论**：
1. **world 精确复现理论**：GD 边界 $f\approx1$、momentum 边界 $f\approx1.9=1+\beta$。这两个数字可以直接作为"$L$ 的计算与 $2/L$ 归一化是对的"的数值证据。
2. **truth 的 $f=1$ 不是真实边界**：$0.25$ 的 $p(1-p)$ 上界在最优点附近极松（真实曲率远小于 $0.25\lambda_{\max}$），所以 GD/momentum 到 $f=3$ 仍稳定且越快越好。**不要**把 truth 面板的 $f>1$ 读成"越过临界值"。
3. **Adam 的"稳定"是弱意义**：它不发散到 inf，但在 truth 上 $f=0.9/1.3/2.2/3.0$ 的 600 步损失比 $f=0.7$ 差 7–9 个数量级，属于"有界但不收敛/震荡"。需要看 best-so-far 曲线而不是单点损失。
4. **world 上 Adam 越大的 $f$ 反而越好**（gap 单调下降），因为它的有效步长仍远小于曲率允许的步长；这与"Adam 在二次型上慢"是同一件事的两面。

**报告用法**：放在"学习率稳定界"或"优化器"小节。世界面板可作为 `稳定界说明.md` 的实证补充；truth 面板用于提醒"logistic 的 Lipschitz 界是保守的，不能把 $2/L$ 当作硬性发散阈值"。

**图注候选**：
- 中文：A8 稳定边界。world 上 GD 在 $f\approx1$、momentum 在 $f\approx1.9$ 发散，与 $2/L$ 和 $2(1+\beta)/L$ 的理论值一致；truth 的 $0.25$ 界偏保守，GD/momentum 到 $f=3$ 仍稳定，Adam 则出现非单调震荡。
- English: A8 stability regions. On world, GD diverges at $f\approx1$ and momentum at $f\approx1.9$, matching $2/L$ and $2(1+\beta)/L$; on truth the $0.25$ bound is conservative, so GD/momentum stay finite through $f=3$, while Adam oscillates non-monotonically.

---

## 3. `a8_final_metrics`：最终 valid 指标 vs 参考最优

**是什么**：2×2 点图。上排 world 的 valid MSE 与中位距离，下排 truth 的 valid acc 与 logloss；每个点是该优化器在 $f=0.9$、8000 步预算下的最终 valid 指标，虚线是参考最优（world dual ridge / truth L-BFGS-B）。纵轴按数据范围收紧，以便看清第 3–5 位有效数字的差异。

**精确读数**：
- world MSE：GD 167.17072、momentum 167.17072、Adam **167.14163**；参考 167.17072。中位距离：GD 923.12077、momentum 923.12059、Adam **926.894**；参考 923.12059。平均距离：GD 1203.0653、momentum 1203.0653、Adam 1202.681。
- truth acc：GD **0.866667**、momentum 0.863333、Adam 0.863333；参考 0.863333。logloss：GD **0.463776**、momentum 0.484369、Adam 0.484374；参考 0.484373。

**结论**：
1. **收敛后解相同**：GD/momentum 在 world 上与参考逐位一致（MSE 差 <1e-9），truth 的 momentum/Adam 与参考差 <1e-6。凸问题的优化器只改变路径，不改变最优解。
2. **Adam 的 world MSE 略低、但中位距离略差**：167.1416 vs 167.1707（−0.017%）、中位 926.9 vs 923.1 km（+0.4%）。这不是"Adam 找到更好的解"，而是它 8000 步后仍有 8.8e-7 的损失间隙，参数向量与参考略有差异；两个指标的微小反向变化在 n=130 上不构成任何结论。
3. **truth GD 的"更好"是早停**：acc 86.67%（+1 条）与 logloss 0.4638（−4.3%）来自 8000 步未收敛（gap 9.3e-4）的隐式正则，而不是 GD 的最终解更优。若继续训练，它会收敛到参考的 0.8633 / 0.4844。报告里必须写明这一点，否则会误读为"GD 泛化更好"。
4. **优化器选择对本课程的最终指标没有实质影响**：三组在 valid 上的差异远小于 A1/A7 中的层间差异和 seed 波动。真正影响结果的是**特征层、正则强度、损失函数**（A6/A7/A10），不是优化器。

**报告用法**：放在"优化器"小节的收尾，配一句"在凸的线性/逻辑回归读出上，GD、momentum、Adam 收敛后给出相同解；优化器只影响达到解的步数，不影响选层/正则的结论"。同时用它说明 truth GD 的 86.67% 不应被当作更优模型。

**图注候选**：
- 中文：A8 最终 valid 指标（点）与参考最优（虚线）。三者收敛后落在参考附近；world 上 Adam 的 MSE 略低但中位距离略差，truth 上 GD 的较高 acc 是 8000 步未收敛的早停效应。
- English: A8 final valid metrics (dots) vs the reference optimum (dashed). All optimizers land on the reference once converged; Adam's slightly lower world MSE comes with a slightly worse median distance, and GD's higher truth accuracy is an early-stopping artefact (gap $9.3\times10^{-4}$ after 8000 steps).

---

## 4. 结论、caveats 与报告写法

### 4.1 可以直接写进报告的结论

1. **优化器只影响速度，不影响凸问题的解**：world 三组收敛后 valid MSE 都是 167.17（Adam 167.14 是未完全收敛的微差）、truth 两组与参考差 <1e-6；没有"某个优化器泛化更好"的证据。
2. **速度强烈依赖任务几何**：world（病态二次型）momentum 最快（到 $10^{-4}$ 90 步 vs GD 500、Adam 6550）；truth（logistic）Adam 最快（260 步 vs momentum 1370、GD >8000）。
3. **稳定边界可被 $2/L$ 精确预测（线性）**：GD $f\approx1$、momentum $f\approx1+\beta=1.9$；这是 $L=2\lambda_{\max}/N+2\lambda_2$ 与 heavy-ball 理论的一次干净对拍。
4. **logistic 的 $0.25$ 界是保守的**：GD/momentum 到 $f=3$ 仍稳定；不要把 $2/L$ 当成 logistic 的硬发散阈值。
5. **Adam 的稳定性是弱意义**：不发散 ≠ 收敛；truth 上 $f=0.9/1.3/2.2/3.0$ 的 600 步损失比 $f=0.7$ 差 7–9 个数量级，需看 best-so-far。
6. **零初始化不会掩盖 world 的发散边界**：$f=1.5$ 时零初始化与 $\sigma=10^{-3}$ 随机初始化都在第 15 步发散（零点梯度沿顶部特征向量有分量）。

### 4.2 caveats（报告/答辩要主动说）

- **同 lr 不是各自最优 lr**：头对头固定 $f=0.9$，这是"同一名义学习率"的公平比较，不是"每个优化器调到自己最优"的比赛。Adam 的自然 lr 尺度与 GD 不同；若给 Adam 单独调 lr，world 上的差距会缩小。结论只能写成"同 lr 下"。
- **world 目标口径**：`model.py` 的 `LinearRegressionModel` 是单目标 1/N；world 两个坐标按两次单目标拟合处理，向量化后是 $\frac1N\|Y-XW-b\|_F^2$。若误用 `torch.mean` 对 $N\times2$ 个元素取平均，数据项会变成 $1/(2N)$，有效 $\lambda_2$ 翻倍，$\mathrm{lr}_{crit}$ 也会错——本实验首次冒烟就踩了这个坑（GD 在 $f=1.5$ 竟不发散），修正后 world 的边界才与理论对齐。报告中的 MSE 仍是元素平均，与 A1 锚点一致。
- **logistic 的 $L$ 不是紧的**：$0.25\lambda_{\max}$ 只在 $p=0.5$ 处取到；最优点附近 $p(1-p)\ll0.25$，所以 $f>1$ 不代表发散。truth 面板的竖线只是"保守界参考线"。
- **Adam 的发散判据**：脚本把"损失非有限或超过 $f_0$ 的 $10^6$ 倍"记为发散；Adam 在 truth 的震荡没有触发该判据，因此图上显示为有限 gap。这是有意的（它不是数值爆炸），但报告中应称"有界震荡"而非"稳定"。
- **没有 seed 误差棒**：全批量、确定性目标，优化器轨迹只受初始化影响；稳定性扫描固定 seed 1234，头对头用零初始化。参考解是闭式/L-BFGS，不含随机性。表中的微小差异不是统计显著性的证据。
- **truth GD 的 86.67%**：比参考多 1 条（300 条中的 1 条），logloss 低 4.3%，但训练损失 gap 仍有 9.3e-4；这是早停效应，不能写成"GD 泛化更好"。
- **步数预算**：truth 的 GD 在 8000 步仍未到 $10^{-4}$；若把预算加到 2–3 万步，三者的最终指标会更接近，但"Adam 最快"的排序不会变（到 $10^{-2}$ 的 80 步 vs GD 3500 步）。

### 4.3 与指南/课程的关系

- 课程 `train1.py`/`train23.py` 用纯 GD；本实验说明：**在本课程的两个任务上，换 momentum/Adam 不会改变选层与正则的结论，只会改变达到解的步数**。如果报告要讨论"为什么用 GD"，可以引用 world 上 momentum 的 5.6× 加速与 truth 上 Adam 的 30× 加速（到 $10^{-3}$：120 vs 3500 步），但不要建议替换课程代码——课程要求的是手写 GD 的实现与理解。
- 指南 A8 的参考报告若给出"Adam 全面更好"或"GD 最稳"的单边结论，与本实验的任务依赖性不一致；我们的数据支持"按任务几何选择/至少不要假设单一优化器全面占优"。

### 4.4 图表清单

| 文件 | 内容 | 建议放置 |
|---|---|---|
| `figures/a8_convergence.{png,pdf}` | best-so-far + 瞬时损失间隙 vs 步数 | 优化器小节主图 |
| `figures/a8_stability.{png,pdf}` | 600 步 gap vs $f$，发散标记 | 学习率/稳定性小节 |
| `figures/a8_final_metrics.{png,pdf}` | 最终 valid 指标 vs 参考最优 | 优化器小节收尾 |
