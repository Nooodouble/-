# 导入必要的库
import numpy as np
import torch
from matplotlib import pyplot as plt

# 导入自定义模块
from autoencoder import LowRankAutoEncoder
from RX import construct_background_dict
from LRR import compute_lrr
from common_func import preprocess_data, plot_roc, calculate_auc

class AnomalyDetector:
    """
    基于多表示融合的高光谱图像异常检测器
    将自编码器重构误差和LRR残差进行加权融合
    """
    def __init__(self, input_dim, hidden_dim=64, lambda1=0.65, lambda2=0.06, device=None):
        """
        初始化检测器
        
        参数:
        input_dim: 输入数据维度（光谱带数）
        hidden_dim: 自编码器隐层维度，默认64
        lambda1: 低秩约束参数，默认0.65
        lambda2: 正则化参数，默认0.06
        device: 运行设备（GPU/CPU）
        """
        # 设置运行设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device if isinstance(device, torch.device) else torch.device(device)
            
        print(f"Using device: {self.device}")
            
        try:
            # 初始化自编码器
            self.autoencoder = LowRankAutoEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                lambda1=lambda1,
                lambda2=lambda2
            )
            # 将模型移动到指定设备
            self.autoencoder = self.autoencoder.to(self.device)
            print("Successfully initialized autoencoder")
            
        except Exception as e:
            print(f"Error initializing autoencoder: {str(e)}")
            print(f"Input dimension: {input_dim}")
            print(f"Hidden dimension: {hidden_dim}")
            print(f"Device: {self.device}")
            raise
        
        # 存储中间结果
        self.features = None
        self.reconstruction_error = None
        self.residual_error = None
        
    def fit(self, data, batch_size=64, epochs=248, learning_rate=0.0006):
        """
        训练模型
        
        参数:
        data: 输入数据，形状为(n_samples, n_features)
        batch_size: 批次大小
        epochs: 训练轮数
        learning_rate: 学习率
        """
        
        processed_data = data
        
        # 确保数据类型正确
        if not isinstance(processed_data, np.ndarray):
            processed_data = np.array(processed_data)
            
        # 打印调试信息
        print(f"Processed data shape: {processed_data.shape}")
        print(f"Processed data type: {processed_data.dtype}")
        
        try:
            # 训练自编码器并获取压缩特征
            self.features, self.reconstruction_error = self.autoencoder.train_model(
                processed_data, batch_size, epochs, learning_rate
            )
            print(f"Compressed features shape: {self.features.shape}")
            
            # 在压缩特征空间构建背景字典 进入RX.py
            background_dict = construct_background_dict(self.features, threshold_percentile=5)
            print(f"Background dictionary shape: {background_dict.shape}")
            
            # 在压缩特征空间计算LRR残差 进入LRR.py
            E, self.residual_error = compute_lrr(self.features, background_dict)
            
        except RuntimeError as e:
            print(f"Error occurred during model training: {str(e)}")
            print(f"Device: {self.device}")
            print(f"Data shape: {processed_data.shape}")
            print(f"Data type: {processed_data.dtype}")
            raise
        
        return self
    
    def weighted_fusion(self, reconstruction_error, residual_error, weight):
        """
        对重构误差和残差进行加权融合
        
        参数:
        reconstruction_error: 自编码器重构误差
        residual_error: LRR残差
        weight: 融合权重 (weight * reconstruction_error + (1-weight) * residual_error)
        
        返回:
        fusion_result: 融合后的检测结果
        """
        # 确保权重在[0,1]范围内
        weight = np.clip(weight, 0, 1)
        
        # 检查输入数据的有效性
        if np.all(residual_error == 0):
            print(f"警告：残差全为0，请检查LRR算法的实现")
            return reconstruction_error
            
        # 加权融合
        fusion_result = weight * reconstruction_error + (1 - weight) * residual_error
        
        return fusion_result

    def detect(self, weights=None):
        """
        执行异常检测
        
        参数:
        weights: 融合权重列表，默认为[0, 0.1, ..., 1.0]
        
        返回:
        all_results: 所有权重下的检测结果字典
        """
        if weights is None:
            weights = np.arange(0, 1.1, 0.1)
            
        if self.reconstruction_error is None or self.residual_error is None:
            raise ValueError("请先调用fit方法训练模型")
            
        # 打印原始误差的统计信息
        print("\n原始误差统计信息:")
        print("重构误差 - 均值: {:.4f}, 标准差: {:.4f}, 最小值: {:.4f}, 最大值: {:.4f}".format(
            np.mean(self.reconstruction_error),
            np.std(self.reconstruction_error),
            np.min(self.reconstruction_error),
            np.max(self.reconstruction_error)
        ))
        print("残差 - 均值: {:.4f}, 标准差: {:.4f}, 最小值: {:.4f}, 最大值: {:.4f}".format(
            np.mean(self.residual_error),
            np.std(self.residual_error),
            np.min(self.residual_error),
            np.max(self.residual_error)
        ))
        
        # 对原始误差进行标准化（使用z-score标准化）
        def standardize(x):
            mean = np.mean(x)
            std = np.std(x)
            #return (x - mean) / (std + 1e-10)
            if std < 1e-10:  # 避免除以接近0的标准差
                print("警告：标准差接近0，数据可能存在问题")
                return np.zeros_like(x)
            return (x - mean) / std
        
        # 分别对两种误差进行标准化
        recon_error_std = standardize(self.reconstruction_error)
        residual_error_std = standardize(self.residual_error)
        
        # 打印标准化后的统计信息
        print("\n标准化后的统计信息:")
        print("重构误差 - 均值: {:.4f}, 标准差: {:.4f}, 最小值: {:.4f}, 最大值: {:.4f}".format(
            np.mean(recon_error_std),
            np.std(recon_error_std),
            np.min(recon_error_std),
            np.max(recon_error_std)
        ))
        print("残差 - 均值: {:.4f}, 标准差: {:.4f}, 最小值: {:.4f}, 最大值: {:.4f}".format(
            np.mean(residual_error_std),
            np.std(residual_error_std),
            np.min(residual_error_std),
            np.max(residual_error_std)
        ))
        
        all_results = {}
        
        # 对所有权重进行融合，直接使用标准化后的分数
        for w in weights:
            # 加权融合，不进行最终归一化
            fusion_score = self.weighted_fusion(recon_error_std, residual_error_std, w)
            
            # 存储融合结果
            all_results[w] = fusion_score
            
            # 打印每个权重下的融合分数统计信息
            print(f"\n权重 {w:.1f} 的融合分数统计:")
            print("均值: {:.4f}, 标准差: {:.4f}, 最小值: {:.4f}, 最大值: {:.4f}".format(
                np.mean(fusion_score),
                np.std(fusion_score),
                np.min(fusion_score),
                np.max(fusion_score)
            ))
        
        return all_results

    def evaluate(self, ground_truth):
        """
        评估检测结果
        
        参数:
        ground_truth: 真实标签
        
        返回:
        best_detection: 最佳检测结果
        best_weight: 最佳权重
        best_auc: 最佳AUC值
        weights: 所有权重值列表
        auc_scores: 所有权重对应的AUC值列表
        all_results: 所有权重的检测结果字典
        """
        # 获取所有权重的检测结果
        all_results = self.detect()
        
        best_auc = 0
        best_weight = 0
        best_detection = None
        weights = []
        auc_scores = []
        
        print("\n异常检测评估结果:")
        print("异常样本数量:", np.sum(ground_truth))
        print("总样本数量:", len(ground_truth))
        
        # 评估每个权重的结果
        for weight, detection_score in all_results.items():
            # 计算异常样本和正常样本的平均分数
            anomaly_scores = detection_score[ground_truth == 1]
            normal_scores = detection_score[ground_truth == 0]
            
            print(f"\n权重 {weight:.1f} 的检测统计:")
            print("异常样本 - 均值: {:.4f}, 标准差: {:.4f}".format(
                np.mean(anomaly_scores), np.std(anomaly_scores)))
            print("正常样本 - 均值: {:.4f}, 标准差: {:.4f}".format(
                np.mean(normal_scores), np.std(normal_scores)))
            
            # 计算AUC
            auc_score = calculate_auc(ground_truth.flatten(), detection_score.flatten())
            print("AUC分数:", auc_score)
            
            # 存储结果
            weights.append(weight)
            auc_scores.append(auc_score)
            
            # 更新最佳结果
            if auc_score > best_auc:
                best_auc = auc_score
                best_weight = weight
                best_detection = detection_score
        
        print("\n最佳检测结果:")
        print(f"最佳权重: {best_weight:.1f}")
        print(f"最佳AUC: {best_auc:.4f}")
        
        return best_detection, best_weight, best_auc, weights, auc_scores, all_results
