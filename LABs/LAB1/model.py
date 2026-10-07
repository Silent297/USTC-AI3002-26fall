import numpy as np


class LinearRegressionModel:
    """用单个线性回归模型预测一个连续目标。"""

    def __init__(self, n_features: int):
        # 一个连续目标只需要一个权重向量和一个标量偏置。
        self.weights = np.zeros(n_features, dtype=np.float64)
        self.bias = np.float64(0.0)

    def fit(self, features: np.ndarray, targets: np.ndarray) -> None:
        # 训练模型；features 为 (N, D)，targets 为 (N,)。
        def cal_loss(Phi:np.ndarray, y:np.ndarray ,w:np.ndarray, b:np.float64, lambda1:np.float64, lambda2:np.float64) -> np.float64:
            residual = Phi @ w + b - y
            return np.sum(residual ** 2) / len(y) + lambda1 * np.sum(np.abs(w)) + lambda2 * np.dot(w,w)
        def cal_grad(Phi:np.ndarray, y:np.ndarray ,w:np.ndarray, b:np.float64, lambda1:np.float64, lambda2:np.float64) -> tuple[np.ndarray, np.float64]:
            residual = Phi @ w + b - y
            N = len(y)
            return 2.0 / N * Phi.T @ residual + lambda1 * np.sign(w) + 2.0 * lambda2 * w  , np.float64(2.0)/ N * np.sum(residual)
        
        epochs = 10000
        lr = 0.001
        lambda1 = np.float64(0.01)
        lambda2 = np.float64(0.01)
        w = np.zeros(features.shape[1])
        b = np.float64(0.0)
        pre_loss = np.inf

        for epoch in range(epochs):
            grad_w, grad_b = cal_grad(features, targets, w, b, lambda1, lambda2)
            if not np.all(np.isfinite(grad_w) and np.isfinite(grad_b)):
                print(f"nan/inf occuered in epoch {epoch}, training has stopped")
                break
            w = w - lr * grad_w
            b = b - lr * grad_b
            if not np.all(np.isfinite(w) and np.isfinite(b)):
                print(f"nan/inf occuered in epoch {epoch}, training has stopped")
                break
            loss = cal_loss(features, targets, w, b, lambda1, lambda2)
            if np.isclose(loss, pre_loss, rtol=1e-6, atol=1e-10):
                print(f"Training converged at epoch {epoch}，loss={loss:.8f}")
                break
            pre_loss = loss

        self.weights = w
        self.bias = b


    def predict(self, features: np.ndarray) -> np.ndarray:
        # 提示：使用矩阵乘法和偏置，返回形状为 (N,) 的预测值。
        return features @ self.weights + self.bias



class LogisticRegressionModel:
    """用逻辑回归从隐藏表征预测句子真假。"""

    def __init__(self, n_features: int):
        # 提示：单个二分类任务只需要一个权重向量和一个标量偏置。
        self.weights = np.zeros(n_features, dtype=np.float64)
        self.bias = np.float64(0.0)

    def fit(self, features: np.ndarray, targets: np.ndarray) -> None:
        # 训练模型；features 为 (N, 2560)，targets 为 (N,) 的 0/1 标签。
        raise NotImplementedError

    def predict(self, features: np.ndarray) -> np.ndarray:
        # 提示：返回形状为 (N,) 的真类概率，不要在此处做 0.5 阈值化。"""
        raise NotImplementedError