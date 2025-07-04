import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import os
import time
from tqdm import tqdm # 用于显示进度条

# --- 从自定义模块导入 ---
# 确保这些模块的路径正确
from Loader.SeriesLoaderAlignment import PairedTimeSeriesDataset
from Model.FitsAlignment import PairedFFTTransformer # 导入我们新创建的模型
from Model.TransformerAlign import PairedTransformer # 如果你有其他模型也可以导入
from utils.Setseed import set_seed
# 假设你的config文件也已准备好
from config import config, model_config, dataset_config

# ==============================================================================
#  重新实现 train_model 和 eval_model 以处理双输入
# ==============================================================================

def train_model_paired(model, data_loader, optimizer, criterion, device):
    """
    针对双输入模型的训练函数。
    """
    model.train()  # 设置模型为训练模式
    total_loss = 0
    
    # 使用tqdm显示进度
    progress_bar = tqdm(data_loader, desc="Training", leave=False)
    
    for features_A, features_B, labels in progress_bar:
        # 将数据移动到指定设备
        features_A = features_A.to(device)
        features_B = features_B.to(device)
        labels = labels.to(device)
        
        # 1. 前向传播
        outputs = model(features_A, features_B)
        
        # 2. 计算损失
        loss = criterion(outputs, labels)
        
        # 3. 反向传播与优化
        optimizer.zero_grad()  # 清空之前的梯度
        loss.backward()        # 计算梯度
        optimizer.step()       # 更新权重
        
        total_loss += loss.item()
        
        # 更新进度条上的损失显示
        progress_bar.set_postfix(loss=f'{loss.item():.4f}')
        
    avg_loss = total_loss / len(data_loader)
    return avg_loss

def eval_model_paired(model, data_loader, criterion, device):
    """
    针对双输入模型的评估函数。
    """
    model.eval()  # 设置模型为评估模式
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    progress_bar = tqdm(data_loader, desc="Evaluating", leave=False)
    
    with torch.no_grad():  # 在评估阶段不计算梯度
        for features_A, features_B, labels in progress_bar:
            features_A = features_A.to(device)
            features_B = features_B.to(device)
            labels = labels.to(device)
            
            # 前向传播
            outputs = model(features_A, features_B)
            
            # 计算损失
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions / total_samples
    
    return avg_loss, accuracy

# ==============================================================================
#  主函数
# ==============================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    set_seed(config["seed"])  # 设置全局随机种子

    # --- 数据准备与划分 ---
    print("\n--- Loading and Splitting Paired Dataset ---")
    
    # 确保 config["model"] 是为这个脚本设置的，例如 "PairedFFTTransformer"
    # 直接使用 PairedTimeSeriesDataset
    # 从 dataset_config 中获取配对数据集的特定参数
    paired_config = dataset_config["Model"]
    full_dataset = PairedTimeSeriesDataset(
        data_dir=paired_config["data_dir"],
        file_pattern_A=paired_config["file_pattern_A"],
        file_pattern_B=paired_config["file_pattern_B"],
        window_size_sec=paired_config["window_size_sec"],
        total_duration_sec=paired_config["total_duration_sec"],
        stride_sec=paired_config["stride_sec"],
        delete_col=paired_config["delete_col"],
        constant_col=paired_config["constant_col"],
        target_col=paired_config["target_col"],
        Normalization=paired_config["Normalization"]
    )

    # 计算划分大小
    val_size = int(dataset_config["Model"]["validation_split"] * len(full_dataset))
    train_size = len(full_dataset) - val_size
    
    # 随机划分数据集
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"\nTotal samples: {len(full_dataset)}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # 创建独立的DataLoader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True
    )

    # --- 模型, 损失函数与优化器准备 ---
    print("\n--- Initializing Model, Criterion, and Optimizer ---")
    
    # 从数据集中动态获取模型初始化所需的参数
    feature_num_A, feature_num_B = full_dataset.get_feature_numbers()
    seq_len_A = full_dataset.window_size_A_pts
    seq_len_B = full_dataset.window_size_B_pts
    
    # 从 model_config 中获取 PairedFFTTransformer 的特定参数
    model_params = model_config["PairedFFTTransformer"]

    if config["model"] == "PairedFFTTransformer":
        print("Using PairedFFTTransformer model")
        model = PairedFFTTransformer(
            feature_num_A=feature_num_A,
            feature_num_B=feature_num_B,
            seq_len_A=seq_len_A,
            seq_len_B=seq_len_B,
            class_num=model_params["output_size"],
            d_model=model_params["d_model"],
            nhead=model_params["nhead"],
            num_encoder_layers=model_params["num_encoder_layers"],
            dim_feedforward=model_params["dim_feedforward"],
            dropout=model_params["dropout"],
            mlp_hidden_dim=model_params["mlp_hidden_dim"]
        ).to(device)
    elif config["model"] == "PairedTransformer":
        print("Using PairedTransformer model")
        model = PairedTransformer(
            feature_num_A=feature_num_A,
            feature_num_B=feature_num_B,
            seq_len_A=seq_len_A,
            seq_len_B=seq_len_B,
            class_num=model_params["output_size"],
            d_model=model_params["d_model"],
            nhead=model_params["nhead"],
            num_encoder_layers=model_params["num_encoder_layers"],
            dim_feedforward=model_params["dim_feedforward"],
            dropout=model_params["dropout"],
            mlp_hidden_dim=model_params["mlp_hidden_dim"]
        ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=config["milestone"], gamma=config["gamma"]
    )
    
    # --- 训练与验证循环 ---
    print("\n--- Starting Training Loop ---")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    
    best_val_loss = float('inf')
    
    for epoch in range(config["num_epochs"]):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        start_time = time.time()
        
        # 调用为双输入模型定制的训练函数
        train_loss = train_model_paired(model, train_loader, optimizer, criterion, device)
        
        # 调用为双输入模型定制的评估函数
        val_loss, val_acc = eval_model_paired(model, val_loader, criterion, device)
        
        # 更新学习率
        scheduler.step()
        
        epoch_duration = time.time() - start_time
        print(f"Epoch duration: {epoch_duration:.2f}s")
        print(f"  -> Average Training Loss: {train_loss:.4f}")
        print(f"  -> Average Validation Loss: {val_loss:.4f}")
        print(f"  -> Validation Accuracy: {val_acc:.4f}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # 确保保存路径存在
            torch.save(model.state_dict(), config["model"]+config["model_save_path"])
            print(f"  -> Best model saved with validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()