import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import os

# --- 从自定义模块导入 ---
from Loader.SeriesLoader import TimeSeriesDataset
from utils.predictior import ModelInference  # <--- 核心改动：导入 ModelInference
from config import config, model_config

def get_predictions(model, data_loader, device):
    """
    在给定的数据集上运行模型并收集所有预测和真实标签。
    (此函数无需改动)
    """
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            print(f"\rProcessing batch {i+1}/{len(data_loader)}...", end="")
    
    print("\nPrediction complete.")
    return np.array(all_labels), np.array(all_predictions)

def plot_confusion_matrix(y_true, y_pred, class_names, filename='confusion_matrix.png'):
    """
    计算并绘制混淆矩阵。
    (此函数无需改动)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    plt.savefig(filename)
    print(f"Confusion matrix saved to {filename}")
    plt.close()

def main():
    # --- 1. 配置和准备 ---
    MODEL_PATH = "FFTbest_classification_model.pth"
    CLASS_NAMES = ["CC", "GL", "LC", "OL", "NL"]
    OUTPUT_FILENAME = "confusion_matrix_full_dataset.png"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- 2. 模型加载 (使用 ModelInference 接口) ---
    print("Initializing model using ModelInference interface...")
    try:
        # 实例化 predictor，它会自动处理模型的选择、实例化和权重加载
        predictor = ModelInference(
            model_path=MODEL_PATH,
            config=config,
            model_config=model_config,
            class_names=CLASS_NAMES
        )
        print(f"Model '{config['model']}' loaded successfully via ModelInference.")
    except Exception as e:
        print(f"Failed to load model using ModelInference: {e}")
        return # 如果加载失败，则退出

    # --- 3. 数据准备 ---
    print("\nLoading full dataset for evaluation...")
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
    
    print(f"Full dataset loaded with {len(full_dataset)} samples.")

    full_loader = DataLoader(
        dataset=full_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True
    )

    # --- 4. 生成预测、计算准确率并绘制混淆矩阵 ---
    print("\nGenerating predictions on the full dataset...")
    # 从 predictor 实例中获取已加载的 PyTorch 模型，并将其传递给预测函数
    # 这是关键步骤，连接了 ModelInference 和批量评估逻辑
    true_labels, pred_labels = get_predictions(predictor.model, full_loader, device)
    
    # 计算总体准确率
    accuracy = accuracy_score(true_labels, pred_labels)
    print("\n" + "="*30)
    print(f"Overall Accuracy on the Full Dataset: {accuracy * 100:.2f}%")
    print("="*30 + "\n")
    
    # 绘制并保存混淆矩阵
    print("Plotting confusion matrix...")
    plot_confusion_matrix(true_labels, pred_labels, CLASS_NAMES, filename=OUTPUT_FILENAME)
    
    print("\nEvaluation finished.")

if __name__ == '__main__':
    main()