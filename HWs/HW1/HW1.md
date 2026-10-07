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

由于 $\epsilon$ 与 $S$ 独立，且 $\mathbb{E}[\epsilon]=0$，$\mathbb{E}[\epsilon^2]=\sigma^2$，我们可以将期望拆解：
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

*   因为 $(f_*(x_0) - m)^2$ 对 $S$ 是常数，可以直接提出。
*   因为 $\mathbb{E}_S[m - Z] = m - \mathbb{E}_S[Z] = 0$，所以第二项为 $0$。
*   最后一项 $\mathbb{E}_S[(m - Z)^2]$ 正好是 $Z$ 的方差 $\text{Var}_S(Z)$。

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
*   **判断：错误**
*   **理由：** 当 $m = f_*(x_0)$ 时，偏差平方项 $(m - f_*(x_0))^2 = 0$。但期望测试误差还包含测试噪声 $\sigma^2$ 和模型预测方差 $\text{Var}_S(Z)$，除非这两项同时为零，否则期望测试误差不为零。

**2. 在测试噪声不变时，若平方偏差的增加小于预测方差的减少，则期望测试误差下降。**
*   **判断：正确**
*   **理由：** 期望测试误差由三项相加组成，当测试噪声 $\sigma^2$ 保持不变时，偏差平方的增加小于方差的减少，则满足 $\Delta \text{Bias}^2 + \Delta \text{Var} < 0$，期望测试误差就会下降。

**3. 模型的训练误差越低，其泛化表现越好。**
*   **判断：错误**
*   **理由：** 训练误差低并不必然意味着泛化表现好。模型可能过度拟合了训练数据中的噪声，导致训练误差极低，但预测方差 $\text{Var}_S(Z)$ 变大，从而使得泛化（期望测试）误差变大。

## 第二题

### (a)
第j个输出的均方误差 $L_j(w_j) = \frac{1}{N}\sum_{i=1}^N (w_j \cdot \phi(x_i) - y_{ij})^2$ 。

只改变 $w_k$ 时，只有第k个输出的均方误差$L_k(w_k)$会发生变化。

理由：在 $L_j(w_j)$ 的表达式中，只包含 $w_j$ 而不含 $w_k$（当 $j \neq k$ 时），因此各输出维度的损失项在数学形式上并不耦合，改变其中一个不会影响其他值。
