# FAIML HW1

> **姓名：** 常志强  
> **学号：** PB25030753

---
## 第一题

### (a) 
$$
y - Z = (f_*(x_0) + \epsilon) - Z = \epsilon + (f_*(x_0) - Z)
$$

$$
(y - Z)^2 = \epsilon^2 + 2\epsilon(f_*(x_0) - Z) + (f_*(x_0) - Z)^2
$$

由于 $\epsilon$ 与 $S$ 独立，且 $\mathbb{E}[\epsilon]=0$，$\mathbb{E}[\epsilon^2]=\sigma^2$，可以将期望拆解：
$$
\begin{aligned}
\mathbb{E}_{S,\epsilon}[(y - Z)^2] &= \mathbb{E}_{S,\epsilon}[\epsilon^2] + 2\mathbb{E}_{S,\epsilon}[\epsilon(f_*(x_0) - Z)] + \mathbb{E}_{S,\epsilon}[(f_*(x_0) - Z)^2] \\
&= \mathbb{E}_\epsilon[\epsilon^2] + 2\mathbb{E}_\epsilon[\epsilon] \cdot \mathbb{E}_S[f_*(x_0) - Z] + \mathbb{E}_S[(f_*(x_0) - Z)^2] \\
&= \sigma^2 + 0 + \mathbb{E}_S[(f_*(x_0) - Z)^2]
\end{aligned}
$$

$$
\begin{aligned}
\mathbb{E}_S[(f_*(x_0) - Z)^2] &= \mathbb{E}_S[(f_*(x_0) - m + m - Z)^2] \\
&= \mathbb{E}_S[(f_*(x_0) - m)^2 + 2(f_*(x_0) - m)(m - Z) + (m - Z)^2] \\
&= (f_*(x_0) - m)^2 + 2(f_*(x_0) - m)\mathbb{E}_S[m - Z] + \mathbb{E}_S[(m - Z)^2]
\end{aligned}
$$

因为 $(f_*(x_0) - m)^2$ 对 $S$ 是常数，可以直接提出。

因为 $\mathbb{E}_S[m - Z] = m - \mathbb{E}_S[Z] = 0$，所以第二项为 $0$。

最后一项 $\mathbb{E}_S[(m - Z)^2]$ 正好是 $Z$ 的方差 $\text{Var}_S(Z)$。

因此：
$$
\mathbb{E}_S[(f_*(x_0) - Z)^2] = (m - f_*(x_0))^2 + \text{Var}_S(Z)
$$


代回原式，得
$$
\mathbb{E}_{S,\epsilon}[(y - Z)^2] = \sigma^2 + (m - f_*(x_0))^2 + \text{Var}_S(Z)
$$

### (b)

**1. 若模型的平均预测等于真实条件均值，则期望测试误差一定为零。**
*   判断：错误
*   理由： 当 $m = f_*(x_0)$ 时，偏差平方项 $(m - f_*(x_0))^2 = 0$。但期望测试误差还包含测试噪声 $\sigma^2$ 和模型预测方差 $\text{Var}_S(Z)$，除非这两项同时为零，否则期望测试误差不为零。

**2. 在测试噪声不变时，若平方偏差的增加小于预测方差的减少，则期望测试误差下降。**
*   判断：正确
*   理由： 期望测试误差由三项相加组成，当测试噪声 $\sigma^2$ 保持不变时，偏差平方的增加小于方差的减少，则满足 $\Delta \text{Bias}^2 + \Delta \text{Var} < 0$，期望测试误差就会下降。

**3. 模型的训练误差越低，其泛化表现越好。**
*   判断：错误
*   理由： 训练误差低并不必然意味着泛化表现好。模型可能过度拟合了训练数据中的噪声，导致训练误差极低，但预测方差 $\text{Var}_S(Z)$ 变大，从而使得泛化误差变大。

## 第二题

### (a) 单输出损失与可加分解

第 \(j\) 个输出的均方误差为
\[
L_j(w_j)
=
\frac1N\sum_{i=1}^N
\bigl(w_j\cdot \phi(x_i)-y_{ij}\bigr)^2.
\]
总平方损失为
\[
L(W)
=
\frac1N\sum_{i=1}^N\sum_{j=1}^n
\bigl(w_j\cdot \phi(x_i)-y_{ij}\bigr)^2
=
\sum_{j=1}^n L_j(w_j).
\]
即
\[
L(W)=L_1(w_1)+L_2(w_2)+\cdots+L_n(w_n).
\]

只改变 \(w_k\)、保持其他 \(w_j\ (j\ne k)\) 不变时，只有第 \(k\) 个损失项 \(L_k(w_k)\) 可能变化；所有 \(j\ne k\) 的 \(L_j(w_j)\) 完全不含 \(w_k\)，因此不变。

理由：模型中第 \(j\) 个输出为
\[
f_j(x)=w_j\cdot \phi(x),
\]
它只依赖于第 \(j\) 列参数 \(w_j\)。所以改变 \(w_k\) 只影响第 \(k\) 个输出的预测及对应损失。

### (b)

可以。

各输出的最优参数：
\[
w_j^*\in \arg\min_{w_j} L_j(w_j),
\qquad j=1,\dots,n,
\]
组合成
\[
W^*=[w_1^*,w_2^*,\dots,w_n^*].
\]
则 \(W^*\) 是联合损失 \(L(W)\) 的最优解。

**证明：**

独立最优推导联合最优：

对任意 \(W=[w_1,\dots,w_n]\)，有
\[
L(W)
=
\sum_{j=1}^n L_j(w_j)
\ge
\sum_{j=1}^n L_j(w_j^*)
=
L(W^*).
\]
因此独立最优解的组合 \(W^*\) 是联合最优解。

联合最优推导独立最优：

设
\[
\bar W=[\bar w_1,\dots,\bar w_n]
\]
是联合损失 \(L(W)\) 的最优解。固定其他列不动，只把第 \(j\) 列换成任意 \(v\in\mathbb R^d\)，得到
\[
W'=[\bar w_1,\dots,\bar w_{j-1},v,\bar w_{j+1},\dots,\bar w_n].
\]
则
\[
L(W')
=
L_j(v)+\sum_{k\ne j}L_k(\bar w_k).
\]
由 \(\bar W\) 联合最优，
\[
L(W')\ge L(\bar W)
=
L_j(\bar w_j)+\sum_{k\ne j}L_k(\bar w_k).
\]
消去相同项得
\[
L_j(v)\ge L_j(\bar w_j)
\]
对任意 \(v\) 成立。因此
\[
\bar w_j\in \arg\min_{w_j} L_j(w_j).
\]
所以联合最优解的每一列参数也分别是最优的。

因此分别最小化各\(L_j\)后组合，与联合最小化L等价。

这里不需要假设不同输出在统计上独立。

原因：总损失在代数上可以分解为各输出损失之和，且各输出参数不耦合，所以不需要统计独立性。

### (c) 

不能直接组合两个单输出问题的最优解。

当 \(n=2\) 且要求
\[
w_1=w_2=w
\]
时，总损失变为
\[
L_{\rm shared}(w)
=
L_1(w)+L_2(w).
\]
设两个单输出问题的最优解分别为
\[
w_1^*=\arg\min_w L_1(w),
\qquad
w_2^*=\arg\min_w L_2(w),
\]
且它们唯一且不同：
\[
w_1^*\ne w_2^*.
\]

无约束时，各自的最小值为
\[
L_1(w_1^*),\qquad L_2(w_2^*).
\]
若希望在共享参数下同时达到这两个最小值，就需要存在 \(w\) 使得
\[
L_1(w)=L_1(w_1^*),
\qquad
L_2(w)=L_2(w_2^*).
\]
由于最优解唯一，这要求
\[
w=w_1^*
\quad\text{且}\quad
w=w_2^*,
\]
即
\[
w_1^*=w_2^*,
\]
与假设矛盾。因此不存在一个共享的 \(w\) 能同时达到两个单输出问题各自的无约束最小值。

在 (b) 中，等价性依赖于题干中的“各输出的参数向量 \(w_j\) 可以独立选择”，即损失可分解为
\[
L(W)=\sum_{j=1}^n L_j(w_j),
\]
且各变量互不耦合。加入 \(w_1=w_2\) 后，参数发生耦合，不能再对两列独立最小化，因此 (b) 的独立优化条件不再满足。

## 第三题

### (a)

\[
\Phi(x) = [1, x_1, x_2]^T
\]

\[
\Phi =
\begin{bmatrix}
1 & 1 & 0 \\
1 & 0 & 1 \\
1 & 1 & 1 \\
1 & -1 & 0 \\
1 & 0 & -1 \\
1 & -1 & -1
\end{bmatrix},
\quad
y =
\begin{bmatrix}
2 \\
0 \\
3 \\
-4 \\
-2 \\
-5
\end{bmatrix}
\]

\[
\Phi \in \mathbb{R}^{6\times 3},\qquad y\in\mathbb{R}^{6\times 1}
\]

如果 \(\exists w_A\in\mathbb{R}^3\)，使得

\[
f_w(x_i)=y_i,\quad i=1,\dots,6
\]

对六条样本成立，则

\[
\Phi w_A=y
\]

有解，解得

\[
w_A=[-1,3,1]^T
\]

代回验证，
\[
\Phi w_A=
\begin{pmatrix}
2\\0\\3\\-4\\-2\\-5
\end{pmatrix}=y.
\]


### (b)
设

\[
w=(w_0,w_1,w_2)^T
\]

目标函数为

\[
J_\lambda(w)
=
\frac16\|\Phi w-y\|^2
+
\lambda\|Pw\|_2^2
\]

展开得

\[
\begin{aligned}
J_\lambda(w)
={}&\frac16\Big[
(w_0+w_1-2)^2
+(w_0+w_2)^2
+(w_0+w_1+w_2-3)^2\\
&\quad
+(w_0-w_1+4)^2
+(w_0-w_2+2)^2
+(w_0-w_1-w_2+5)^2
\Big]\\
&+\lambda(w_1^2+w_2^2)
\end{aligned}
\]

整理得

\[
\begin{aligned}
J_\lambda(w)
={}&
\frac16(6w_0^2+4w_1^2+4w_2^2+4w_1w_2)\\
&-\frac13(-6w_0+14w_1+10w_2)
+\frac{29}{3}
+\lambda(w_1^2+w_2^2)
\end{aligned}
\]

梯度为

\[
\nabla J_\lambda(w)
=
\frac13\Phi^T(\Phi w-y)
+
2\lambda Pw
\]

展开得

\[
\nabla J_\lambda(w)
=
\begin{pmatrix}
2w_0+2\\[6pt]
\dfrac{4w_1+2w_2-14}{3}+2\lambda w_1\\[6pt]
\dfrac{2w_1+4w_2-10}{3}+2\lambda w_2
\end{pmatrix}
\]

### (c)
残差为

\[
r^{(0)}=\Phi w^{(0)}-y
=
(-1,-1,-3,3,3,5)^T
\]

梯度为

\[
\nabla J_\lambda(w^{(0)})
=
\left(2,-\frac{10}{3},-\frac{14}{3}\right)^T
\]


更新公式为

\[
\begin{aligned}
w^{(1)}
&=
w^{(0)}-\eta\nabla J_\lambda(w^{(0)})\\
&=
(0,1,-1)^T
-
\frac14
\left(2,-\frac{10}{3},-\frac{14}{3}\right)^T\\
&=
\left(-\frac12,\frac{11}{6},\frac16\right)^T
\end{aligned}
\]

目标函数值为

\[
\begin{aligned}
J_\lambda(w^{(0)})
&=
\frac16\|r^{(0)}\|^2
+
\lambda\|Pw^{(0)}\|_2^2\\
&=
\frac16\times 54
+
\frac13\times 2\\
&=
\frac{29}{3}
\end{aligned}
\]

以及

\[
\begin{aligned}
J_\lambda(w^{(1)})
&=
\frac16\|\Phi w^{(1)}-y\|^2
+
\lambda\|Pw^{(1)}\|_2^2\\
&=
\frac{367}{108}
\end{aligned}
\]

因此

\[
J_\lambda(w^{(1)})<J_\lambda(w^{(0)})
\]

说明这次更新使目标函数下降。

### (d)

对于 \(w_A\)，由于

\[
\Phi w_A=y
\]

所以

\[
L(w_A)=0
\]

\[
\|Pw_A\|_2^2=9+1=10
\]

在 3 个验证样本上：

\[
\begin{aligned}
MSE
&=\frac13\sum_{i=1}^3(f_{w_A}(x_i)-y_i)^2
&=\frac13\left(1^2+0^2+(-2)^2\right)
&=\frac53
\end{aligned}
\]


对于 \(w_B\)，有

\[
r=\Phi w_B-y
=
(-1,0,-1,1,0,1)^T
\]


损失为

\[
L(w_B)
=
\frac16(1+1+1+1)
=
\frac23
\]


\[
\|Pw_B\|_2^2=w_{B1}^2+w_{B2}^2
\]

验证 MSE 为

\[
\begin{aligned}
MSE
&=
\frac13\sum_{i=1}^3(w_B(x_i)-y_i)^2\\
&=
\frac13\left((-2)^2+(-1)^2+(-1)^2\right)\\
&=
2
\end{aligned}
\]

根据验证结果，应选择 \(w_A\)，因为 \(w_A\) 在验证集上表现更好。


在计算验证 MSE 时，不应加入正则惩罚项，因为验证集评估的是泛化误差，而正则项是模型复杂度的控制，两者不应混在一起。

### (e)

由全局最优性可知：

\[
J_{\lambda_1}(w^{(1)})\le J_{\lambda_1}(w^{(2)})
\]

以及

\[
J_{\lambda_2}(w^{(2)})\le J_{\lambda_2}(w^{(1)})
\]

展开可得：

\[
L(w_1)+\lambda_1\|Pw_1\|^2
\le
L(w_2)+\lambda_1\|Pw_2\|^2
\]

\[
L(w_2)+\lambda_2\|Pw_2\|^2
\le
L(w_1)+\lambda_2\|Pw_1\|^2
\]

移项得

\[
\lambda_1\left(\|Pw_1\|^2-\|Pw_2\|^2\right)
\le
L(w_2)-L(w_1)
\tag{①}
\]

\[
\lambda_2\left(\|Pw_2\|^2-\|Pw_1\|^2\right)
\le
L(w_1)-L(w_2)
\]

两边相加得

\[
(\lambda_1-\lambda_2)
\left(
\|Pw_1\|^2-\|Pw_2\|^2
\right)
\le 0
\]


将

\[
\|Pw^{(2)}\|^2\le \|Pw^{(1)}\|^2
\]

代入 ① 中，可得

\[
L(w^{(2)})\ge L(w^{(1)})
\]


增大 \(\lambda\) 不一定降低 MSE。
- \(\lambda\) 过小时，模型容易过拟合；
- \(\lambda\) 过大时，模型容易欠拟合；
- MSE 在这两种情况下都可能较大。

### (f)
设 \(L_0(w)\) 为原始损失。使用普通梯度下降：

\[
\begin{aligned}
w^{(t+1)}
=
w^{(t)}-\eta\nabla L(w^{(t)})-2\eta\lambda P w^{(t)}
=
(I-2\eta\lambda P)w^{(t)}-\eta\nabla L(w^{(t)}).
\end{aligned}
\]

使用权重衰减更新：

\[
w^{(t+1)}
=
D_c w^{(t)}
-\eta\nabla L_0(w^{(t)})
\]

两者等价时

\[
c
=
1-2\eta\lambda
=
1-2\times\frac14\times\frac13
=
\frac56
\]

\(D_c\) 的第一个对角元素为 \(1\)，因为第一个权重 \(w_0\) 为截距。若对截距进行衰减，会生成不必要的整体平移偏差。

这与正则化矩阵

\[
P=\operatorname{diag}(0,1,1)
\]

的第一个对角元素为 \(0\) 相呼应。

因此

\[
D_c=\operatorname{diag}(1,c,c)
\]

只衰减 \(w_1\) 和 \(w_2\)，不衰减截距 \(w_0\)。

## 第四题

## (a)
在数据集上：

\[
L_A(t)=\sigma^6(t)\,(1-\sigma(t))^2
\]

\[
\begin{aligned}
J_A(t)
&=-\frac18\ln L_A(t)\\
&=-\frac34\ln\sigma(t)-\frac14\ln(1-\sigma(t))
\end{aligned}
\]

\[
L_B(t)=\sigma^8(t)
\]

\[
J_B(t)=-\frac18\ln L_B(t)=-\ln\sigma(t)=\ln(1+e^{-t})
\]

一般地，

\[
J(t)=-\frac18\ln L(t)
\]

由于 \(\ln x\) 在 \((0,+\infty)\) 上单调递增，因此

\[
\min L_D(t)
\iff
\max J_D(t),
\qquad
J_D(t)=-\frac1n\ln L_D(t)
\]

所以最大化似然等价于最小化负对数似然损失。

### (b)

令

\[
s=\sigma(t)
\]

则

\[
J_A(t)
=
-\frac34\ln s-\frac14\ln(1-s)
=
f(s)
\]

求导得

\[
\frac{\partial f}{\partial s}
=
-\frac34\cdot\frac1s
+
\frac14\cdot\frac1{1-s}
=
\frac{4s-3}{4s(1-s)}
\]

令

\[
\frac{\partial f}{\partial s}=0
\]

得

\[
s=\frac34
\]

又因为

\[
f''(s)=\frac{3}{4s^2}+\frac{1}{4(1-s)^2}>0
\]

所以 \(f(s)\) 为严格凸函数，在 \(s=\frac34\) 处取得唯一最小值：

\[
f\left(\frac34\right)
=
2\ln 2-\frac34\ln 3
\]

此时 \(J_A(t)\) 最小，最优参数 \(t\) 满足

\[
\sigma(t)=\frac34
\]

即

\[
\frac{1}{1+e^{-t}}=\frac34
\quad\Rightarrow\quad
e^{-t}=\frac13
\quad\Rightarrow\quad
t=\ln 3
\]

唯一性证明：

\[
J_A'(t)=\sigma(t)-\frac34
\]

\[
J_A''(t)=\sigma(t)(1-\sigma(t))>0
\]

因此 \(J_A(t)\) 在 \(\mathbb{R}\) 上为严格凸函数，仅有唯一驻点，故最优解唯一。


对于数据集 B：

\[
J_B(t)=\ln(1+e^{-t})
\]

当

\[
t\to+\infty
\]

时，

\[
J_B(t)\to0
\]

且

\[
1+e^{-t}>1
\quad\Rightarrow\quad
J_B(t)>0
\]

因为

\[
\inf_t J_B(t)=0
\]

所以不存在有限参数 \(t\) 使其达到下确界 \(0\)。

原因：

- 数据集 A 在每个 \(x\) 处标签有重叠，错误率非零，因此最优解为有限值；
- 数据集 B 完全线性可分，无限增大 \(t\) 会使所有真实标签概率趋于 1，损失趋于 0，无法在有限值达到最优。

### (c)

放大前决策边界：

\[
P(y=1\mid x)=\sigma(tx)=\frac12
\]

\[
\iff tx=0
\iff x=0
\]

放大后决策边界：

\[
P(y=1\mid x)=\sigma(ctx)=\frac12
\]

\[
\iff ctx=0
\iff x=0
\]

因此决策边界不变。

对于数据 B：

- 分类错误率：由于决策边界不变，分类错误率也不变，保持为 0；
- 真实标签预测概率：放大前为 \(\sigma(t)\)，放大后为 \(\sigma(ct)\)。若 \(c>1,\ t>0\)，则

\[
\sigma(ct)>\sigma(t)
\]

即预测概率增大；

- 平均负对数似然：

\[
J_B(t)=-\ln\sigma(t),
\qquad
J_B(ct)=-\ln\sigma(ct)
\]

因为

\[
\sigma(ct)>\sigma(t)
\iff
J_B(ct)<J_B(t)
\]

所以平均负对数似然下降。

差异原因：

分类错误率只关心预测值是否越过阈值，已经为 0 后无法继续降低；而概率和负对数似然关心置信度，放大权重会让模型更加“自信”，使真实标签概率更接近 1，因此负对数似然继续下降。

### (d)

1. 错误。

在数据集 B 上，任意 \(t>0\) 都有分类错误率为 0，但 \(J_B(t)\) 的下界为 0，有限 \(t\) 达不到。因此分类错误率为 0 不等同于负对数似然已经最小。

2. 错误。

训练样本上的真实标签概率接近 1，只说明模型在训练集上越来越自信，可能是过拟合。对于测试数据，若模型过度自信或测试分布与训练分布存在差异，测试平均负对数似然反而可能变差。训练损失下降不能保证测试损失下降。


### 加入 L2 正则项为什么能抑制权重无限增大

在数据集 \(D_B\) 上，无正则化时

\[
J_B(t)=\ln(1+e^{-t})
\]

当 \(t\to+\infty\) 时损失趋近于 0，因此模型会倾向于让 \(t\) 无限增大。

加入正则项后，目标变为

\[
J_B(t)+\lambda t^2
=
\ln(1+e^{-t})+\lambda t^2,
\qquad
\lambda>0
\]

当 \(t\) 很大时，数据损失已经很小且趋于常数 0，但二次惩罚 \(\lambda t^2\) 会迅速增大并最终主导目标函数。因此最优的 \(t\) 被限制在有限范围内，不会无限增大，从而抑制权重过大和过度自信。