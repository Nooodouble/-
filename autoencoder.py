import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import time
import torch.nn.init as init

class LowRankAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, lambda1=0.65, lambda2=0.06):
        """
        初始化低秩约束的自编码器模型
        
        参数:
        input_dim: 输入数据维度
        hidden_dim: 压缩后的特征维度
        lambda1: 低秩约束项的权重
        lambda2: 参数正则化项的权重
        """
        super(LowRankAutoEncoder, self).__init__()
        
        # 确保所有维度都是整数
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.lambda1 = lambda1  # 低秩约束系数
        self.lambda2 = lambda2  # 参数正则化系数
        
        # 计算中间层维度
        dim_diff = self.input_dim - self.hidden_dim  # 计算输入维度和隐藏层维度的差
        step = dim_diff / 4  # 每层的维度减少步长
        
        # 从输入层到隐藏层的维度逐步递减
        self.mid_dim3 = int(self.input_dim - step)     # 第一中间层
        self.mid_dim2 = int(self.input_dim - 2 * step) # 第二中间层
        self.mid_dim1 = int(self.input_dim - 3 * step) # 第三中间层
        
        print(f"\n网络结构信息:")
        print(f"Input dimension: {self.input_dim}")
        print(f"Network structure: {self.input_dim} -> {self.mid_dim3} -> {self.mid_dim2} -> {self.mid_dim1} -> {self.hidden_dim}")
        
        try:
            # 定义编码器结构
            self.encoder = nn.Sequential(
                nn.Linear(self.input_dim, self.mid_dim3),  
                nn.ReLU(),
                nn.Linear(self.mid_dim3, self.mid_dim2),  
                nn.ReLU(),
                nn.Linear(self.mid_dim2, self.mid_dim1),  
                nn.ReLU(),
                nn.Linear(self.mid_dim1, self.hidden_dim) 
            )
            
            # 定义解码器结构
            self.decoder = nn.Sequential(
                nn.Linear(self.hidden_dim, self.mid_dim1), 
                nn.ReLU(),
                nn.Linear(self.mid_dim1, self.mid_dim2),  
                nn.ReLU(),
                nn.Linear(self.mid_dim2, self.mid_dim3),   
                nn.ReLU(),
                nn.Linear(self.mid_dim3, self.input_dim)  
            )
            
            # 使用He初始化
            for layer in self.encoder:
                if isinstance(layer, nn.Linear):
                    init.kaiming_normal_(layer.weight, nonlinearity='relu')
            
            for layer in self.decoder:
                if isinstance(layer, nn.Linear):
                    init.kaiming_normal_(layer.weight, nonlinearity='relu')
            
            print("Successfully initialized network layers")
            
        except Exception as e:
            print(f"\n网络初始化错误:")
            print(f"Error type: {type(e)}")
            print(f"Error message: {str(e)}")
            print(f"Current PyTorch version: {torch.__version__}")
            raise
    
    def forward(self, x):
        """前向传播函数"""
        encoded = self.encoder(x)  # 编码过程
        decoded = self.decoder(encoded)  # 解码过程
        return encoded, decoded
    
    def nuclear_norm(self, Z):
        """
        计算矩阵的核范数（奇异值之和）
        用于替代rank函数作为低秩约束
        """
        # 计算奇异值
        singular_values = torch.linalg.svdvals(Z)
        # 返回奇异值之和
        return torch.sum(singular_values)
    
    def parameter_norm(self):
        """
        计算所有网络参数的L2范数
        """
        total_norm = 0
        for param in self.parameters():
            if param.requires_grad:
                total_norm += torch.norm(param, p=2) ** 2
        return total_norm
    
    def compute_loss(self, x, encoded, decoded, batch_size):
        """
        计算总损失
        包括重构误差、低秩约束和参数正则化
        """
        # 重构误差项 
        reconstruction_loss = torch.sum((decoded - x) ** 2, dim=1)  # 每个样本的重构误差
        reconstruction_loss_mean = torch.mean(reconstruction_loss)  # 用于优化的平均损失
        
        # 低秩约束项（使用核范数）
        low_rank_loss = self.lambda1 * self.nuclear_norm(encoded)
        
        # 参数正则化项
        regularization_loss = self.lambda2 * self.parameter_norm()
        
        # 总损失 (用于优化的损失使用平均重构误差)
        total_loss = 0.5 * (reconstruction_loss_mean + low_rank_loss + regularization_loss)
        
        return total_loss, reconstruction_loss_mean, low_rank_loss, regularization_loss
    
    def train_model(self, data, batch_size=64, epochs=248, learning_rate=0.0006):
        """
        无监督训练自编码器
        """
        # 获取模型所在设备
        device = next(self.parameters()).device
        print(f"Training on device: {device}")
        
        # 创建检查点目录
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 将输入数据转换为PyTorch数据集
        data_tensor = torch.FloatTensor(data)
        if device.type == 'cuda':
            data_tensor = data_tensor.cuda()
        dataset = TensorDataset(data_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # 使用Adam优化器
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        # 早停相关变量
        early_stop_threshold = 0.0005
        patience = 30
        loss_history = []
        best_loss = float('inf')
        no_improve_count = 0
        
        # 训练开始时间
        start_time = time.time()
        last_checkpoint_time = start_time
        checkpoint_interval = 600  # 每10分钟保存一次检查点
        
        # 训练循环
        for epoch in range(epochs):
            epoch_start_time = time.time()
            total_loss = 0
            recon_loss = 0
            rank_loss = 0
            reg_loss = 0
            
            # 每个epoch开始时清理GPU内存
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
            for batch_idx, batch in enumerate(dataloader):
                # 获取批次数据
                x = batch[0]
                
                # 前向传播
                encoded, decoded = self(x)
                
                # 计算损失
                loss, recon, rank, reg = self.compute_loss(x, encoded, decoded, batch_size)
                
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # 累计损失
                total_loss += loss.item()
                recon_loss += recon.item()
                rank_loss += rank.item()
                reg_loss += reg.item()
            
            # 计算平均损失
            avg_total = total_loss / len(dataloader)
            avg_recon = recon_loss / len(dataloader)
            avg_rank = rank_loss / len(dataloader)
            avg_reg = reg_loss / len(dataloader)
            
            # 每轮结束打印详细信息
            epoch_time = time.time() - epoch_start_time
            print(f'\nEpoch [{epoch+1}/{epochs}] 完成')
            print(f'用时: {epoch_time:.2f}秒')
            print(f'总损失: {avg_total:.6f}')
            print(f'重构损失: {avg_recon:.6f}')
            print(f'低秩损失: {avg_rank:.6f}')
            print(f'正则化损失: {avg_reg:.6f}')
            
            # 早停检查
            loss_history.append(avg_total)
            if avg_total < best_loss:
                best_loss = avg_total
                no_improve_count = 0
                # 保存最佳模型
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                }, os.path.join(checkpoint_dir, 'best_model.pt'))
            else:
                no_improve_count += 1
            
            if no_improve_count >= patience:
                print(f'\n连续{patience}轮未改善，停止训练')
                break
        
        # 训练结束，计算特征和重构误差
        self.eval()
        with torch.no_grad():
            data_tensor = torch.FloatTensor(data)
            if device.type == 'cuda':
                data_tensor = data_tensor.cuda()
            encoded, decoded = self(data_tensor)
            reconstruction_error = torch.sum((decoded - data_tensor) ** 2, dim=1).cpu().numpy()
        
        # 清理GPU内存
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        return encoded.cpu().numpy(), reconstruction_error
        