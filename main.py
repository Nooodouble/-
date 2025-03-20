# 导入必要的库
import numpy as np  # 用于数值计算
import scipy.io as sio  # 用于读取.mat文件
import torch  # PyTorch深度学习框架
from matplotlib import pyplot as plt  # 用于绘图
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import mahalanobis

# 导入自定义模块
from common_func import preprocess_data, plot_roc, calculate_auc  # 导入数据预处理和评估函数
from autoencoder import LowRankAutoEncoder  # 导入低秩约束自编码器模型
from RX import construct_background_dict  # 导入RX算法函数
from LRR import compute_lrr  # 导入LRR算法函数
from anomaly_detector import AnomalyDetector  # 导入异常检测类

def main():
    try:
        """
        # 提取高光谱数据立方体，通常shape为(H*W, C)，其中：
        # - H和W是图像的高度和宽度
        # - C是光谱通道数
        """
        # 加载高光谱数据
        data_path = "data/plane.mat" # 指定高光谱图像数据集的路径
        hsi_data = sio.loadmat(data_path)   # 使用scipy.io加载.mat格式的数据文件
        
        # 加载地面真值数据
        gt_path = "data/plane_gt.mat"  # 指定地面真值数据的路径
        gt_data = sio.loadmat(gt_path)
        
        # 首先查看.mat文件中的所有键名
        print("Available keys in the HSI data file:", hsi_data.keys())
        print("Available keys in the ground truth file:", gt_data.keys())
        
        # 加载数据
        X = hsi_data['data']
        ground_truth = gt_data['map']  # 直接使用地面真值数据
        print("Data shape:", X.shape)
        print("Ground truth shape:", ground_truth.shape)
        
        # 如果数据是3D的(H,W,C)，需要重塑为2D(H*W,C)并进行预处理
        if len(X.shape) == 3:
            height, width, bands = X.shape
            X_2d = preprocess_data(X)  # 使用common_func.py中的预处理函数
        else:
            X_2d = preprocess_data(X)  # 使用common_func.py中的预处理函数
            
        # 将地面真值重塑为1D数组
        ground_truth = ground_truth.flatten()
        
        print("Processed data shape:", X_2d.shape)
        print("Ground truth shape:", ground_truth.shape)
        print("Number of anomalies:", np.sum(ground_truth))
        
        # 设置设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(torch.cuda.is_available())
        print(f"Using device: {device}")
        
        # 创建检测器并训练
        detector = AnomalyDetector(
            input_dim=X_2d.shape[1],
            hidden_dim=64,
            lambda1=0.65,  # 低秩约束参数
            lambda2=0.06,  # 正则化参数
            device=device  # 明确指定设备
        )
        
        try:
            detector.fit(X_2d)
        except Exception as e:
            print(f"Error during training: {str(e)}")
            print(f"Input data shape: {X_2d.shape}")
            print(f"Device: {device}")
            raise
        
        # 评估结果
        best_detection, best_weight, best_auc, weights, auc_scores, all_results = detector.evaluate(ground_truth)
        
        # 创建一个包含两个子图的图形
        plt.figure(figsize=(15, 5))
        
        # 第一个子图：ROC曲线
        plt.subplot(121)
        for weight, detection_score in all_results.items():
            fpr, tpr, auc_score = plot_roc(ground_truth, detection_score)
            plt.plot(fpr, tpr, label=f'Weight={weight:.1f}, AUC={auc_score:.3f}')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Different Fusion Weights')
        plt.legend()
        plt.grid(True)
        
        # 第二个子图：AUC随权重变化的曲线
        plt.subplot(122)
        plt.plot(weights, auc_scores, '-o')
        plt.xlabel('Weight')
        plt.ylabel('AUC Score')
        plt.title('AUC Score vs Weight')
        plt.grid(True)
        
        plt.tight_layout()
        # 保存ROC曲线图
        plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
        print("ROC curves saved as 'roc_curves.png'")
        plt.close()
        
        print(f'Best AUC: {best_auc:.3f} at weight = {best_weight:.1f}')
        
        # 创建新图形：检测结果可视化
        plt.figure(figsize=(12, 4))
        
        plt.subplot(131)
        plt.imshow(X.reshape(height, width, bands)[:,:,30]) # 显示一个波段
        plt.title('Original Image (Band 30)')
        plt.axis('off')
        
        plt.subplot(132)
        plt.imshow(ground_truth.reshape(height, width))
        plt.title('Ground Truth')
        plt.axis('off')
        
        plt.subplot(133)
        plt.imshow(best_detection.reshape(height, width))
        plt.title(f'Detection Result\n(w={best_weight:.1f}, AUC={best_auc:.3f})')
        plt.axis('off')
        
        plt.tight_layout()
        # 保存检测结果图
        plt.savefig('detection_results.png', dpi=300, bbox_inches='tight')
        print("Detection results saved as 'detection_results.png'")
        plt.close()
        
        print("Program completed!")
        
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
        plt.close('all')
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        plt.close('all')

# 程序入口点
if __name__ == "__main__":
    main() 