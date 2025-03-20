import numpy as np
from sklearn.metrics import roc_curve, auc

def preprocess_data(data):
    """
    数据预处理函数
    - 数据标准化
    - 形状调整
    - 数据类型转换
    
    参数:
    data: 输入的高光谱图像数据,可以是3D(高度×宽度×波段)或2D(像素×波段)
    
    返回:
    data_normalized: 标准化后的2D数据矩阵
    """
    # 将数据reshape为2D矩阵 (pixels, bands)
    if len(data.shape) == 3:
        height, width, bands = data.shape
        data_2d = data.reshape(-1, bands)  # 将3D数据展平为2D
    else:
        data_2d = data  # 如果已经是2D则直接使用
    
    # 标准化
    data_mean = np.mean(data_2d, axis=0)    # 计算每个波段的均值
    data_std = np.std(data_2d, axis=0)     # 计算每个波段的标准差
    data_normalized = (data_2d - data_mean) / (data_std + 1e-10)    # 标准化，加入小量避免除零
    
    # 选择要打印的波段索引（Python索引从0开始，所以要减1）
    bands_to_print = [0, 1, 2, 92, 93, 94, 186, 187, 188]

    # 取出对应波段的均值和标准差
    selected_means = data_mean[bands_to_print]
    selected_stds = data_std[bands_to_print]

    # 打印结果
    print(f"Selected bands: {bands_to_print}")
    print(f"Data mean for selected bands: {selected_means}")
    print(f"Data std for selected bands: {selected_stds}")
    # 打印调试信息
    
    print(f"Processed data shape: {data_normalized.shape}")
    
    return data_normalized

def plot_roc(y_true, y_score):
    """
    计算并返回ROC曲线的数据
    
    参数:
    y_true: 真实标签 (0表示正常，1表示异常)
    y_score: 异常检测得分
    
    返回:
    fpr: 假阳性率
    tpr: 真阳性率
    auc_score: ROC曲线下面积，即AUC值
    """
    # 计算ROC曲线
    fpr, tpr, _ = roc_curve(y_true, y_score)
    # 计算AUC
    auc_score = auc(fpr, tpr)
    
    return fpr, tpr, auc_score

def calculate_auc(y_true, y_score):
    """
    仅计算AUC值
    
    参数:
    y_true: 真实标签
    y_score: 异常检测得分
    
    返回:
    auc_score: ROC曲线下面积，即AUC值
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return auc(fpr, tpr) 