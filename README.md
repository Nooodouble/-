# 基于多表示融合的高光谱图像异常检测

## 项目简介
本项目提出了一种基于多表示融合的高光谱图像异常检测方法。通过融合自编码器的重构误差和LRR的残差，实现了高效准确的异常目标检测。

## 环境配置

### 确保已经创建并且激活虚拟环境

### 依赖安装
```bash
pip install -r requirements.txt
```

## 项目结构
├── main.py # 主程序
├── autoencoder.py # 自编码器模型
├── LRR.py # 低秩表示算法
├── RX.py # RX检测器
├── common_func.py # 通用函数
├── anomaly_detector.py # 异常检测器
├── requirements.txt # 依赖项
└── data/ # 数据目录
├── plane.mat # 高光谱数据
└── plane_gt.mat # 地面真值

## 快速开始

### 1. 数据准备
- 将高光谱数据`plane.mat`放入`data`目录
- 将地面真值`plane_gt.mat`放入`data`目录

### 2. 运行程序
```bash
python main.py
```

## 输出文件
- ROC曲线图: `roc_curves.png`
- 检测结果可视化: `detection_results.png`