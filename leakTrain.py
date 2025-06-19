import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import os

# --- 从自定义模块导入 ---
# 假设你的类和函数都在这些路径下
from Loader.SeriesLoader import TimeSeriesDataset
from Model.LstmModel import LSTMForecastModel
from utils.Training import train_model, eval_model

def main():
    # --- 1. 配置参数 ---
    config = {
        "data_dir": "/home/zhangzhe/data/leak/processed_data",
        "file_pattern": 'leak_exp_*.csv',
        "window_size": int(2560 * 0.1),
        "stride": int(2560 * 0.1),  # 窗口大小的50%作为步长
        "target_col": 'LeakType',
        "batch_size": 64,
        "num_workers": 8,  # 根据你的CPU核心数调整
        "validation_split": 0.2, # 使用20%的数据作为验证集
        
        "hidden_size": 256,
        "num_layers": 4,
        "output_size": 4,   # 类别数量，对应 LeakType 的不同种类
        "dropout": 0.2,
        
        "num_epochs": 50,   # 增加训练轮数以观察效果
        "learning_rate": 0.001,
        "model_save_path": "best_classification_model.pth"
    }

    # --- 2. 设备准备 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 3. 数据准备与划分 ---
    print("Loading and splitting dataset...")
    full_dataset = TimeSeriesDataset(
        data_dir=config["data_dir"],
        file_pattern=config["file_pattern"],
        window_size=config["window_size"],
        stride=config["stride"],
        target_col=config["target_col"]
    )
    
    # 计算划分大小
    val_size = int(config["validation_split"] * len(full_dataset))
    train_size = len(full_dataset) - val_size
    
    # 随机划分数据集
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Total samples: {len(full_dataset)}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

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

    # --- 4. 模型, 损失函数与优化器准备 ---
    input_size = full_dataset.get_feature_number() # 动态获取特征数量

    model = LSTMForecastModel(
        input_size=input_size,
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        output_size=config["output_size"],
        dropout=config["dropout"]
    ).to(device)

    # 对于单标签多分类任务，CrossEntropyLoss是标准选择
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    # --- 5. 训练与验证循环 ---
    print("\n--- Starting Training Loop ---")
    best_val_loss = float('inf') # 初始化最佳验证损失为无穷大

    for epoch in range(config["num_epochs"]):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        
        # 调用训练函数
        train_loss = train_model(model, train_loader, optimizer, criterion, device)
        
        # 调用评估函数
        # 注意：eval_model 需要返回 val_loss, predictions, labels
        val_loss, acc  = eval_model(model, val_loader, criterion, device)
        
        print(f"  -> Average Training Loss: {train_loss:.4f}")
        print(f"  -> Average Validation Loss: {val_loss:.4f}")
        print(f"  -> Validation Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()