# 导入必要的库
import numpy as np  # 用于数值计算
from scipy.stats import chi2  # 用于卡方分布计算（虽然在当前代码中未使用）

def construct_background_dict(data, threshold_percentile=5, reg_factor=1e-4):
    """
    使用RX算法构建背景字典
    
    参数:
    data: 输入数据,形状为 (n_samples, n_features),每行是一个样本(像素),每列是一个特征(波段)
    threshold_percentile: 用于选择背景像素的阈值百分位数,默认5%
    reg_factor: 正则化因子,防止协方差矩阵奇异
    
    返回:
    background_dict: 背景字典矩阵,包含被识别为背景的像素,shape=(n_features, n_dict)
    """
    # 打印输入数据信息
    print(f"Input data shape: {data.shape}")
    
    # 计算协方差矩阵
    # 使用转置是因为np.cov期望每行是一个变量(波段),每列是一个观测(像素)
    covariance = np.cov(data.T)  # 计算特征(波段)之间的协方差矩阵,表示特征(波段)之间的相关性
    
    # 添加正则化项
    covariance += reg_factor * np.eye(covariance.shape[0])

    # 打印协方差矩阵信息
    print(f"Covariance matrix shape: {covariance.shape}")
    
    # 计算数据均值 每个特征（波段）的平均值
    mean = np.mean(data, axis=0)  # 沿样本方向计算均值
    
    # 计算每个像素的RX检测值（马氏距离）
    rx_scores = np.zeros(len(data))  # 初始化RX得分数组
    for i in range(len(data)):
        # 计算当前样本与均值的差
        diff = data[i] - mean
        # 计算马氏距离：(x-μ)^T * Σ^(-1) * (x-μ)
        rx_scores[i] = np.dot(np.dot(diff, np.linalg.inv(covariance)), diff.T)
    
    # 打印RX得分统计信息
    print(f"RX scores: min={np.min(rx_scores):.4f}, max={np.max(rx_scores):.4f}, mean={np.mean(rx_scores):.4f}")
    
    # 根据阈值选择背景像素
    # 选择RX得分低于阈值的像素作为背景
    threshold = np.percentile(rx_scores, threshold_percentile)  # 计算阈值
    background_indices = rx_scores < threshold  # 获取背景像素的索引
    
    # 构建背景字典（选择背景像素构成字典）并转置
    background_dict = data[background_indices].T  # 转置以得到(n_features, n_dict)的形状
    print(f"Selected background dictionary shape: {background_dict.shape}")
    
    return background_dict 