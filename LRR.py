# 导入必要的库
import numpy as np  # 用于数值计算
from scipy.linalg import solve_sylvester  # 用于求解Sylvester方程
import time
from numpy.linalg import norm, svd

def nuclear_norm_prox_batch(X, tau, batch_size=100):
    """
    分批计算核范数近似
    """
    n_samples = X.shape[1]
    n_batches = (n_samples + batch_size - 1) // batch_size 
    result = np.zeros_like(X)
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)  #确保了最后一批的处理不会超出数据范围
        # 对当前批次进行SVD
        U, S, Vt = svd(X[:, start_idx:end_idx], full_matrices=False)
        # 软阈值处理
        S = np.maximum(S - tau, 0)
        # 重构并存储结果
        result[:, start_idx:end_idx] = U @ np.diag(S) @ Vt
    
    return result

def compute_lrr(X, D, lambda_=0.001, beta=0.0005, min_iter = 30, max_iter=100, epsilon = 2.7e-7, batch_size=100):
    """
    使用LADMAP算法求解低秩表示问题，使用批处理方式
    
    参数:
    X: 数据矩阵，shape=(n_samples, n_features)
    D: 字典矩阵，shape=(n_features, n_dict)
    lambda_: E的正则化参数 0.12
    beta: J的正则化参数 0.12
    max_iter: 最大迭代次数
    min_iter: 最小迭代次数
    epsilon: 收敛阈值
    batch_size: 批处理大小
    
    返回:
    E: 残差矩阵
    reconstruction_errors: 每个样本的重构误差
    """
    # 转置X使其维度与D@A匹配
    X = X.T  # 现在X的shape为(n_features, n_samples)
    n_samples = X.shape[1]
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    # 初始化变量
    A = np.zeros((D.shape[1], X.shape[1]))  # 系数矩阵维度为(n_dict, n_samples)
    J = np.zeros_like(A)  # 辅助变量
    E = np.zeros_like(X)  # 残差矩阵，shape与X.T相同
    Y1 = np.zeros_like(X)  # 拉格朗日乘子1
    Y2 = np.zeros_like(A)  # 拉格朗日乘子2
    S = np.zeros_like(A)  # 辅助变量S
    
    # 初始化参数
    mu = 0.14  # 初始惩罚参数
    mu_max = 1.5e4  # 最大惩罚参数
    rho = 1.04  # 惩罚参数增长率
    eta1 = norm(D, 2) ** 2  # 近似Hessian参数
    
    print(f"Processing data with shape {X.shape} using dictionary of shape {D.shape}")
    print(f"Coefficient matrix A shape: {A.shape}")
    print(f"Number of batches: {n_batches} (batch_size={batch_size})")
    
    # LADMAP迭代
    for k in range(max_iter):
        max_error = 0
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)  #确保了最后一批的处理不会超出数据范围
            
            # 获取当前批次的数据
            X_batch = X[:, start_idx:end_idx]
            A_batch = A[:, start_idx:end_idx]
            J_batch = J[:, start_idx:end_idx]
            E_batch = E[:, start_idx:end_idx]
            Y1_batch = Y1[:, start_idx:end_idx]
            Y2_batch = Y2[:, start_idx:end_idx]
            S_batch = S[:, start_idx:end_idx]
            
            # 1. 固定J和E，更新A
            # 计算公式中的梯度项 [-D^T(X-DA-E+Y1/μ) + (S-J+Y2/μ)]/η1
            DT_term = -D.T @ (X_batch - D @ A_batch - E_batch + Y1_batch/mu)
            Y_term = S_batch - J_batch + Y2_batch/mu
            grad = (DT_term + Y_term) / eta1
            
            # 计算临时变量
            U = A_batch - grad

            # 使用批处理计算核范数近似算子求解最小化问题
            A_new_batch = nuclear_norm_prox_batch(U, 1/(eta1*mu), batch_size=min(batch_size, 1000))
            
            # 2. 固定A和E，更新J；目标函数：min β||J||_1 + (μ/2)||A-J+Y2/μ||_F^2
            S_batch = A_new_batch # 更新辅助变量S
            temp = S_batch + Y2_batch/mu
            J_new_batch = np.maximum(0, np.abs(temp) - beta/mu) * np.sign(temp)
            
            # 3. 固定A和J，更新E；目标函数：min λ||E||_2,1 + (μ/2)||X-DA-E+Y1/μ||_F^2
            temp = X_batch - D @ A_new_batch + Y1_batch/mu
            E_new_batch = np.maximum(0, np.abs(temp) - lambda_/mu) * np.sign(temp)
            
            # 4. 更新拉格朗日乘子
            Y1[:, start_idx:end_idx] = Y1_batch + mu * (X_batch - D @ A_new_batch - E_new_batch)
            Y2[:, start_idx:end_idx] = Y2_batch + mu * (A_new_batch - J_new_batch)
            
            # 更新变量
            A[:, start_idx:end_idx] = A_new_batch
            J[:, start_idx:end_idx] = J_new_batch
            E[:, start_idx:end_idx] = E_new_batch
            S[:, start_idx:end_idx] = S_batch
            
            # 更新mu,计算当前批次的误差
            batch_error = mu * max(
                np.sqrt(eta1) * norm(A_new_batch - A_batch, 'fro'),
                norm(J_new_batch - J_batch, 'fro'),
                norm(E_new_batch - E_batch, 'fro')
            ) / norm(X_batch, 'fro')
            
            max_error = max(max_error, batch_error)
            
        # 更新mu
        if k >= min_iter:  # 只有达到最小迭代次数后才考虑调整rho
            if  max_error <= epsilon:
                rho = 1.15  # 快速增长
            elif max_error > epsilon:
                rho = 1.1   # 缓慢增长
              
        else:
            rho = 1.04     # 前20次迭代保持较小的增长率
            
        mu = min(mu_max, rho * mu)

        # 每10次迭代打印一次信息
        if (k + 1) % 10 == 0:
            print(f"Iteration {k+1}, mu={mu:.2e}, error={max_error:.2e}")
    
    # 转置E回原始维度并计算重构误差
    E = E.T
    reconstruction_errors = np.sum(E**2, axis=1)
    
    # 输出统计信息
    print("\nLRR重构误差统计:")
    print(f"最小值: {np.min(reconstruction_errors):.4f}")
    print(f"最大值: {np.max(reconstruction_errors):.4f}")
    print(f"均值: {np.mean(reconstruction_errors):.4f}")
    print(f"标准差: {np.std(reconstruction_errors):.4f}")
    
    return E, reconstruction_errors 