# utils/training.py

import torch
import torch.nn as nn
from tqdm import tqdm  # 引入tqdm来显示进度条，提升用户体验

def train_model(model, train_loader, optimizer, criterion, device):
    """
    模型单轮训练函数。

    参数:
    - model (nn.Module): 要训练的模型。
    - train_loader (DataLoader): 训练数据的DataLoader。
    - optimizer (torch.optim.Optimizer): 优化器。
    - criterion (nn.Module): 损失函数。
    - device (torch.device): 'cuda' 或 'cpu'。

    返回:
    - float: 当前epoch的平均训练损失。
    """
    model.train()  # 设置模型为训练模式
    total_loss = 0.0
    
    # 使用tqdm包装dataloader，可以显示一个漂亮的进度条
    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    
    for features, labels in progress_bar:
        # 1. 将数据移动到指定设备
        features = features.to(device)
        labels = labels.long().to(device) 
        # 2. 前向传播
        outputs = model(features)
        
        # 3. 计算损失
        loss = criterion(outputs, labels)
        
        # 4. 反向传播和优化
        optimizer.zero_grad()  # 清空梯度
        loss.backward()        # 计算梯度
        optimizer.step()       # 更新权重
        
        # 5. 累加损失
        total_loss += loss.item()
        
        # (可选) 在进度条上显示当前批次的损失
        progress_bar.set_postfix(loss=loss.item())

    # 计算并返回平均损失
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def eval_model(model, val_loader, criterion, device):
    """
    模型评估函数，针对分类任务增加了准确率计算。

    参数:
    - model (nn.Module): 要评估的模型。
    - val_loader (DataLoader): 验证数据的DataLoader。
    - criterion (nn.Module): 损失函数 (例如 nn.CrossEntropyLoss)。
    - device (torch.device): 'cuda' 或 'cpu'。

    返回:
    - float: 当前epoch的平均验证损失。
    - float: 数据集上的总准确率。
    """
    model.eval()  # 设置模型为评估模式
    total_loss = 0.0
    all_predictions = []
    all_labels = []

    # 在评估模式下，不计算梯度，以节省计算资源和内存
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Evaluating", leave=False)
        for features, labels in progress_bar:
            # 1. 将数据移动到指定设备
            features = features.to(device)
            # 对于CrossEntropyLoss，标签应该是LongTensor，形状为[B]
            labels = labels.long().to(device)

            # 2. 前向传播，获取原始logits
            outputs = model(features)
            
            # 3. 计算损失
            # outputs 形状: [B, C], labels 形状: [B]
            loss = criterion(outputs, labels)
            
            # 4. 累加损失
            total_loss += loss.item()
            
            # 5. 计算预测结果
            # torch.max(outputs, 1) 返回每一行的 (最大值, 最大值索引)
            # 我们只需要索引，即预测的类别
            outputs = torch.nn.Softmax(dim=1)(outputs)  # 应用Softmax以获得概率分布
            _, predicted_classes = torch.max(outputs, 1)
            
            # 6. 收集所有批次的预测和真实标签，用于最后统一计算准确率
            all_predictions.extend(predicted_classes.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # --- 计算最终指标 ---
    
    # 计算平均损失
    avg_loss = total_loss / len(val_loader)
    
    # 方法一：使用 scikit-learn 计算准确率 (推荐，代码简洁)
    
    # 方法二：手动计算准确率
    correct_predictions = (torch.tensor(all_predictions) == torch.tensor(all_labels)).sum().item()
    total_samples = len(all_labels)
    accuracy = correct_predictions / total_samples
    
    return avg_loss, accuracy