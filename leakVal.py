import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import os, pdb
# --- 从自定义模块导入 ---
# 假设你的类和函数都在这些路径下
from Loader.SeriesLoader import TimeSeriesDataset
from Model.LstmModel import LSTMForecastModel
from Model.Transformer import TransformerClassificationModel
from Model.Fits import FFTClassify
from torch.optim.lr_scheduler import MultiStepLR
from utils.Setseed import set_seed
from utils.Training import train_model, eval_model
from config import config, model_config 
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    set_seed(config["seed"])  # 设置全局随机种子
    # --- 3. 数据准备与划分 ---
    print("Loading and splitting dataset...")
    full_dataset = TimeSeriesDataset(
        data_dir=config["data_dir"],
        file_pattern=config["file_pattern"],
        window_size=config["window_size"],
        stride=config["stride"],
        target_col=config["target_col"],
        Normalization=config.get("Normalization", True) ,
        delete_col=config["delete_col"] , # 删除不需要的列
        constant_col=config["constant_col"] # 常量列不参与归一化
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
    if config["model"] == "LSTM":
        model = LSTMForecastModel(
            input_size=input_size,
            hidden_size=model_config["LSTM"]["hidden_size"],
            num_layers=model_config["LSTM"]["num_layers"],
            output_size=model_config["LSTM"]["output_size"],
            dropout=model_config["LSTM"]["dropout"]
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
    if config["model"] == "FFT":
        model = FFTClassify(
            seq_len=config["window_size"],
            classNum=model_config["FFT"]["output_size"],
            feature_num=input_size,
            minFreq=int(config["window_size"] // 2),  # 最小频率

        ).to(device)
    # 对于单标签多分类任务，CrossEntropyLoss是标准选择
    criterion = nn.CrossEntropyLoss()
    model.load_state_dict(torch.load("FFTbest_classification_model.pth", map_location=device))
    print("Model loaded successfully.")
    eval_stat = eval_model(
        model=model,
        val_loader=val_loader,
        criterion=criterion,
        device=device,
    )
    print(f"Validation Loss: {eval_stat[0]:.4f}, Accuracy: {eval_stat[1]:.4f}")

main()