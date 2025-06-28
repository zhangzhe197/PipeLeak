import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
from utils.predictior import ModelInference

# --- 从自定义模块导入 ---
# 确保这些模块的路径正确
from Loader.SeriesLoader import TimeSeriesDataset
from Model.LstmModel import LSTMForecastModel
from Model.Transformer import TransformerClassificationModel
from Model.Fits import FFTClassify
from utils.Setseed import set_seed
from config import config, model_config

def get_predictions(model, data_loader, device):
    """
    在给定的数据集上运行模型并收集所有预测和真实标签。
    """
    model.eval()  # 设置模型为评估模式
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():  # 在此模式下，不计算梯度，节省内存和计算
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            # 获取预测类别（概率最高的那个）
            _, predicted = torch.max(outputs.data, 1)
            
            # 将批次的预测和标签收集到列表中
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return np.array(all_labels), np.array(all_predictions)

def plot_confusion_matrix(y_true, y_pred, class_names, filename='confusion_matrix.png'):
    """
    计算并绘制混淆矩阵。
    """
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 绘制
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # 保存图像
    plt.savefig(filename)
    print(f"Confusion matrix saved to {filename}")
    
    # 显示图像
    plt.show()

def main():
    # --- 1. 配置和准备 ---
    MODEL_PATH = "FFTbest_classification_model.pth"  # 指定要加载的模型文件
    OUTPUT_FILENAME = "confusion_matrix_FFT.png"    # 输出图像的文件名
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # **重要**: 使用与训练时相同的随机种子，确保数据集划分一致
    set_seed(config["seed"])
    
    # --- 2. 数据准备 ---
    # 这部分代码与你的主脚本完全相同，以确保得到相同的验证集
    print("Loading and splitting dataset...")
    full_dataset = TimeSeriesDataset(
        data_dir=config["data_dir"],
        file_pattern=config["file_pattern"],
        window_size=config["window_size"],
        stride=config["stride"],
        target_col=config["target_col"],
        Normalization=config.get("Normalization", True),
        delete_col=config["delete_col"],
        constant_col=config["constant_col"]
    )
    
    val_size = int(config["validation_split"] * len(full_dataset))
    train_size = len(full_dataset) - val_size
    
    # 使用相同的种子进行划分
    _, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Using validation set with {len(val_dataset)} samples.")

    # 创建验证集的DataLoader
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,  # 验证集不需要打乱
        num_workers=config["num_workers"],
        pin_memory=True
    )

    # --- 3. 模型加载 ---
    input_size = full_dataset.get_feature_number()
    model = None # 初始化model变量
    
    # 根据config动态选择和实例化模型
    if config["model"] == "LSTM":
        model = LSTMForecastModel(
            input_size=input_size,
            hidden_size=model_config["LSTM"]["hidden_size"],
            num_layers=model_config["LSTM"]["num_layers"],
            output_size=model_config["LSTM"]["output_size"],
            dropout=model_config["LSTM"]["dropout"]
        ).to(device)
    
    elif config["model"] == "Transformer":
        model = TransformerClassificationModel(
            input_size=input_size,
            d_model=model_config["Transformer"]["d_model"],
            nhead=model_config["Transformer"]["nhead"],
            num_encoder_layers=model_config["Transformer"]["num_encoder_layers"],
            dim_feedforward=model_config["Transformer"]["dim_feedforward"],
            output_size=model_config["Transformer"]["output_size"],
            dropout=model_config["Transformer"]["dropout"],
            max_len=config["window_size"]
        ).to(device)
        
    elif config["model"] == "FFT":
        model = FFTClassify(
            seq_len=config["window_size"],
            classNum=model_config["FFT"]["output_size"],
            feature_num=input_size,
            minFreq=int(config["window_size"] // 2)
        ).to(device)

    if model is None:
        raise ValueError(f"Model type '{config['model']}' is not recognized.")

    # 加载训练好的模型权重
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
        
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"Model '{config['model']}' loaded successfully from {MODEL_PATH}.")

    # --- 4. 生成预测并绘制混淆矩阵 ---
    print("Generating predictions on the validation set...")
    true_labels, pred_labels = get_predictions(model, val_loader, device)
    
    # 获取类别数量和名称
    # 假设类别是从0到N-1的整数
    num_classes = model_config[config["model"]]["output_size"]
    class_names = [str(i) for i in range(num_classes)]
    # 如果你有具体的类别名称，可以替换这里，例如:
    # class_names = ['Class A', 'Class B', 'Class C'] 
    class_names = ["CC", "GL", "LC", "OL", "NL"]
    print("Plotting confusion matrix...")
    plot_confusion_matrix(true_labels, pred_labels, class_names, filename=OUTPUT_FILENAME)

if __name__ == '__main__':
    main()