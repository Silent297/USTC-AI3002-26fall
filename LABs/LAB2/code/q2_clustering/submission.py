"""Lab2 第二题：补全五个算法函数和三个评价函数。

所有函数都使用 PyTorch Tensor，运行设备由全局 ``device`` 控制；
``device`` 由框架设置为 ``torch.device("cpu")`` 或 ``torch.device("cuda:0")``。
不要在函数中写 CPU/GPU 两套分支，也不要把张量转为 NumPy 做计算。
不能调用现成聚类求解器、初始化器或评价函数；scikit-learn 仅用于独立核验。
浮点类型由配置 dtype 或 --dtype 选择，默认 float32，也支持 float64。
新浮点张量应继承 X.dtype 和 X.device；算法返回的 labels 使用 torch.int32 或 torch.int64，
评价函数接收的 labels 为 torch.int64。平均对数似然与评价指标可返回有限 Python float
或同设备零维 Tensor。rng 为框架已设置种子的同设备 torch.Generator，随机抽样时传入
generator=rng。不要修改输入。框架负责转换、迭代、重启、评价抽样、汇总和保存文件。
评价函数只接收特征与预测分区，不接收真实标签。
"""
import torch
import math
from utils.runtime import device


# KMeans：初始化、分配与中心更新。
def init_centers_kmeanspp(X, n_clusters, rng):
    """TODO 1：返回 (K,D) 初始中心。"""
    N, D = X.shape
    centers = X.new_empty((n_clusters, D))
    min_sq = X.new_full((N,), float("inf"))
    selected = torch.zeros(N, dtype=torch.bool, device=X.device)

    first = int(torch.randint(0, N, (1,), generator=rng, device=X.device).item())
    centers[0] = X[first]
    selected[first] = True
    min_sq = torch.minimum(min_sq, (X - X[first]).square().sum(dim=1))

    for k in range(1, n_clusters):
        if bool(min_sq.sum() > 0):    
            index = int(torch.multinomial(min_sq, 1, generator=rng).item())
        else:    
            remain = torch.nonzero(~selected, as_tuple=False).flatten()    
            position = int(torch.randint(0, remain.numel(), (1,),generator=rng, device=X.device).item())  
            index = int(remain[position].item())
        centers[k] = X[index]
        selected[index] = True
        min_sq = torch.minimum(min_sq, (X - X[index]).square().sum(dim=1))

    return centers


def assign_clusters(X, centers):
    """TODO 2：返回 labels:(N,) 整数、min_sq_dist:(N,)。"""
    x_sq = X.square().sum(dim=1)
    mu_sq = centers.square().sum(dim=1)
    dist = x_sq.unsqueeze(1) + mu_sq.unsqueeze(0) - 2 * (X @ centers.t())
    dist = dist.clamp_min(0)
    min_sq_dist, labels = torch.min(dist, dim=1)
    return labels, min_sq_dist


def update_centers(X, labels, old_centers):
    """TODO 3：返回更新后的 (K,D) 中心，不修改输入。"""
    K, D = old_centers.shape
    sums = X.new_zeros((K, D))
    index = labels.long()
    sums.index_add_(dim=0, index=index, source=X)
    counts = torch.bincount(index, minlength=K)
    _counts = counts.clamp_min(1).to(X.dtype).unsqueeze(1)
    means = sums / _counts
    is_new_centers = (counts > 0).unsqueeze(1)
    new_centers = torch.where(is_new_centers, means, old_centers)
    return new_centers


# GMM：对角协方差下的 E 步与 M 步。
def gmm_estep(X, weights, means, variances):
    """TODO 4：对角协方差 GMM 的 E 步。"""
    log_weights = torch.log(weights)
    var_reciprocal = 1.0 / variances
    mean_term = (means.square() * var_reciprocal).sum(dim=1)
    x_sq = X.square() @ var_reciprocal.t()
    cross = X @ (means * var_reciprocal).t()        
    quad = x_sq - 2.0 * cross + mean_term

    D = X.shape[1]
    log_var = torch.log(variances).sum(dim=1)
    log_joint = log_weights - 0.5 * (D* math.log(2 * math.pi) + log_var + quad)
    lse = torch.logsumexp(log_joint, dim=1)
    resp = torch.exp(log_joint - lse.unsqueeze(1))
    mean_log_likelihood = lse.mean()

    return resp, mean_log_likelihood


def gmm_mstep(X, resp, reg_covar):
    """TODO 5：返回 weights:(K,), means:(K,D), variances:(K,D)。"""
    Nk = resp.sum(dim=0)
    if bool((Nk <= 1e-12).any()):
        raise ValueError("gmm_mstep: component with zero effective count")
    weights = Nk / Nk.sum()
    means = (resp.t() @ X) / Nk.unsqueeze(1)
    variances = (resp.t() @ X.square()) / Nk.unsqueeze(1) - means.square()
    variances = variances.clamp_min(0) + reg_covar

    return weights, means, variances


# 评价：框架提供原始特征与预测分区，以下函数负责计算指标。
def silhouette_score(X, labels):
    """TODO E1：返回平均 Silhouette，使用欧氏距离（不是平方距离）。"""
    dist = torch.cdist(X, X)
    unique, inverse = torch.unique(labels, return_inverse=True)
    K = unique.numel()
    one_hot = torch.nn.functional.one_hot(inverse, K).to(X.dtype)
    counts = one_hot.sum(dim=0)      
    sum_dist = dist @ one_hot         
    mean_dist = sum_dist / counts
    
    own = mean_dist.gather(1, inverse.unsqueeze(1)).squeeze(1)
    n_i = counts[inverse]
    a = own * n_i / (n_i - 1).clamp_min(1)
    others = mean_dist.masked_fill(one_hot.bool(), float("inf"))
    b = others.min(dim=1).values
    max_ab = torch.maximum(a, b)
    s = torch.where(max_ab > 0, (b-a)/max_ab, torch.zeros_like(a))
    s = torch.where(n_i > 1, s, torch.zeros_like(s))
    return s.mean()


def calinski_harabasz_score(X, labels):
    """TODO E2：返回 CH，框架传入全量原始特征。"""
    unique, inverse = torch.unique(labels, return_inverse=True)
    K = unique.numel()
    one_hot = torch.nn.functional.one_hot(inverse, K).to(X.dtype)
    counts = one_hot.sum(dim=0)
    means = (one_hot.t() @ X) / counts.unsqueeze(1)
    all_means = X.mean(dim=0)
    B = ((means - all_means).square().sum(dim=1) * counts).sum()
    x_sq = (one_hot.t() @ X.square()).sum(dim=1)
    w = x_sq - counts * means.square().sum(dim=1)
    W = w.clamp_min(0).sum()
    if float(W) == 0:
        return 1.0
    CH = (B /(K - 1)) / (W / (X.shape[0]-K))
    return CH



def davies_bouldin_score(X, labels):
    """TODO E3：返回 DB，框架传入全量原始特征。"""
    unique, inverse = torch.unique(labels, return_inverse=True)
    K = unique.numel()
    one_hot = torch.nn.functional.one_hot(inverse, K).to(X.dtype)
    counts = one_hot.sum(dim=0)
    means = (one_hot.t() @ X) / counts.unsqueeze(1)
    mean_dist = torch.cdist(X, means)
    S = (mean_dist * one_hot).sum(dim=0) / counts
    M = torch.cdist(means, means)
    if bool((S <= 1e-8).all()):
        return 0.0
    if bool((M <= 1e-8).all()):
        return 0.0
    M_mask = M.masked_fill(M == 0, float("inf"))
    R = ((S.unsqueeze(1) + S.unsqueeze(0)) / M_mask).max(dim=1).values
    DB = R.mean()
    return DB
