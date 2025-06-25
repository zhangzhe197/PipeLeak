import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import random
import pdb
# --- 从自定义模块导入 ---
# 假设你的类和函数都在这些路径下
from Loader.SeriesLoader import TimeSeriesDataset
from Model.LstmModel import LSTMForecastModel
from Model.Transformer import TransformerClassificationModel
from utils.Training import train_model, eval_model
from config import config, model_config 

def plot_test_samples(test_loader, dataset_instance, num_samples_per_label=3, output_path="feature_plots.png"):
    """
    绘制测试集中不同标签类型的特征随时间变化的图形。

    Args:
        test_loader (DataLoader): 测试数据的DataLoader。
        dataset_instance (TimeSeriesDataset): 用于获取特征信息的原始数据集实例。
        num_samples_per_label (int): 每种标签类型要显示的样本数量。
        output_path (str): 保存图像的文件路径。
    """
    print(f"\nPlotting {num_samples_per_label} samples per label from the test set...")
    
    samples_by_label = defaultdict(list)
    # 获取特征名称（如果TimeSeriesDataset提供的话，否则使用默认名称）
    # 假设TimeSeriesDataset有一个方法可以返回特征列名
    feature_names = dataset_instance.get_column_names()

    # 1. 收集测试集中的样本，按标签分类
    # 注意：这里我们只从DataLoader中获取一定数量的样本，以避免内存溢出
    # 如果测试集很大，你可能需要更智能的抽样策略
    max_samples_to_collect = 1000 # 收集多一些，确保能抽到
    
    for batch_idx, (data, labels) in enumerate(test_loader):
        for i in range(data.shape[0]):
            sample_data = data[i].cpu().numpy() # (window_size, input_size)
            sample_label = labels[i].item()
            samples_by_label[sample_label].append(sample_data)
            
            # 简单判断是否已经收集到足够多的样本
            collected_count = sum(len(v) for v in samples_by_label.values())
            if collected_count >= max_samples_to_collect:
                break
        if collected_count >= max_samples_to_collect:
            break
            
    if not samples_by_label:
        print("No samples collected. Test set might be empty or data loading failed.")
        return

    unique_labels_found = sorted(samples_by_label.keys())
    print(f"Unique labels found in test set: {unique_labels_found}")

    # 2. 绘制图形
    num_rows = len(unique_labels_found)
    num_cols = num_samples_per_label
    
    # 动态调整图形大小
    fig_width = 5 * num_cols
    fig_height = 4 * num_rows
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height), squeeze=False)
    
    for row_idx, label in enumerate(unique_labels_found):
        label_samples = samples_by_label[label]
        
        # 从该标签的样本中随机选择指定数量的样本
        selected_samples = random.sample(label_samples, min(len(label_samples), num_samples_per_label))
        
        for col_idx, sample_data in enumerate(selected_samples):
            ax = axes[row_idx, col_idx]
            # 遍历每个特征并绘制
            for feature_idx in range(sample_data.shape[1]):
                ax.plot(sample_data[:, feature_idx], label=feature_names[feature_idx])
            
            ax.set_title(f"Label: {label}, Sample {col_idx+1}")
            ax.set_xlabel("Time Step (within window)")
            ax.set_ylabel("Normalized Value")
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # 只在最右侧的图例显示一次
            if col_idx == num_cols - 1:
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
                
    plt.tight_layout(rect=[0, 0, 0.95, 1]) # 调整布局以适应图例
    plt.suptitle("Feature Evolution Over Time for Different Labels (Test Set Samples)", y=1.02, fontsize=16)
    plt.savefig(output_path, bbox_inches='tight')
    plt.show()
    print(f"Plots saved to {output_path}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 3. 数据准备与划分 ---
    print("Loading and splitting dataset...")
    full_dataset = TimeSeriesDataset(
        data_dir=config["data_dir"],
        file_pattern=config["file_pattern"],
        window_size=config["window_size"],
        stride=config["stride"],
        target_col=config["target_col"],
        Normalization="All" # 默认启用归一化
    )
    
    # 计算划分大小：训练集、验证集、测试集
    # 假设 config["validation_split"] 是非训练集的总比例
    # 我们将非训练集再二分为验证集和测试集（例如，各占一半）
    non_train_size = int(config["validation_split"] * len(full_dataset))
    train_size = len(full_dataset) - non_train_size
    
    # 第一次随机划分：训练集 vs. (验证集+测试集)
    train_dataset, val_test_dataset = random_split(full_dataset, [train_size, non_train_size])
    
    # 第二次随机划分：(验证集+测试集) -> 验证集 vs. 测试集
    # 这里我们假设将 non_train_size 均匀分成验证集和测试集
    val_size = int(0.5 * len(val_test_dataset)) # 验证集占非训练集的一半
    test_size = len(val_test_dataset) - val_size # 测试集占非训练集的另一半
    
    val_dataset, test_dataset = random_split(val_test_dataset, [val_size, test_size])
    
    print(f"Total samples: {len(full_dataset)}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}") # 新增测试集信息

    # 创建独立的DataLoader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config["batch_size"],
        shuffle=True, # 训练集需要打乱
        num_workers=config["num_workers"],
        pin_memory=True # 如果使用GPU，可以加速数据传输
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config["batch_size"],
        shuffle=False, # 验证集不需要打乱
        num_workers=config["num_workers"],
        pin_memory=True
    )

    test_loader = DataLoader( # 新增测试集DataLoader
        dataset=test_dataset,
        batch_size=config["batch_size"],
        shuffle=False, # 测试集不需要打乱
        num_workers=config["num_workers"],
        pin_memory=True
    )

    # --- 4. 模型, 损失函数与优化器准备 (保持不变，但为了绘图可以不运行训练和评估) ---
    input_size = full_dataset.get_feature_number() # 动态获取特征数量
    if config["model"] == "LSTM":
        model = LSTMForecastModel(
            input_size=input_size,
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            output_size=config["output_size"],
            dropout=config["dropout"]
        ).to(device)
    
    if config["model"] == "Transformer":
        model = TransformerClassificationModel(
            input_size=input_size,
            d_model=model_config["Transformer"]["d_model"],
            nhead=model_config["Transformer"]["nhead"],
            num_encoder_layers=model_config["Transformer"]["num_encoder_layers"],
            dim_feedforward=model_config["Transformer"]["dim_feedforward"],
            output_size=model_config["Transformer"]["output_size"],
            dropout=model_config["Transformer"]["dropout"],
            max_len=config["window_size"]  # 确保最大长度与窗口大小一致
        ).to(device)
    
    model.load_state_dict(torch.load("best_classification_model_LSTM.pth", map_location=device))
    # 对于单标签多分类任务，CrossEntropyLoss是标准选择
    # 注意：如果您只是为了绘图，下面这行可以注释掉，因为它需要一个预训练模型文件
    # try:
    #     model.load_state_dict(torch.load("best_classification_model.pth"))
    #     print("Model loaded successfully from best_classification_model.pth")
    # except FileNotFoundError:
    #     print("Warning: Model checkpoint not found. Starting with untrained model.")
    # except Exception as e:
    #     print(f"Error loading model: {e}. Starting with untrained model.")

    # --- 5. 执行绘图 ---
    plot_test_samples(test_loader, full_dataset, num_samples_per_label=3, output_path="test_set_feature_plots.png")

    # --- 6. 训练与评估 (可选，如果你想继续训练模型的话) ---
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    # 
    # for epoch in range(config["num_epochs"]):
    #     train_loss, train_accuracy = train_model(model, train_loader, criterion, optimizer, device)
    #     val_loss, val_accuracy = eval_model(model, val_loader, criterion, device)
    #     print(f"Epoch {epoch+1}/{config['num_epochs']}: "
    #           f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f} | "
    #           f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
    # 
    # print("\nTraining complete!")
    # 
    # # 在测试集上进行最终评估（如果模型已训练）
    # final_test_loss, final_test_accuracy = eval_model(model, test_loader, criterion, device)
    # print(f"Final Test Loss: {final_test_loss:.4f}, Final Test Accuracy: {final_test_accuracy:.4f}")

if __name__ == "__main__":
    main()