import torch
from torch.utils.data import DataLoader, random_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# --- 从自定义模块导入 ---
from Loader.SeriesLoader import TimeSeriesDataset
from utils.predictior import ModelInference # 导入我们刚刚创建的推理接口
from config import config, model_config,dataset_config
from utils.Setseed import set_seed

def main():
    # --- 1. 配置与准备 ---
    MODEL_PATH = "FFTbest_classification_model.pth"   # 你的预训练模型文件
    CLASS_NAMES = ["CC", "GL", "LC", "OL", "NL"]      # 类别名称，顺序必须正确
    OUTPUT_FILENAME = "prediction_visualization.png" # 输出图像的文件名
    
    # 确保你的 config.py 文件中有 'all_columns' 字段，这对于 ModelInference 至关重要
    if 'all_columns' not in config:
        raise ValueError("请在 config.py 中添加 'all_columns' 列表，以确保特征顺序正确。")
        
    # --- 2. 初始化模型推理接口 ---
    # 这会加载模型并准备好所有必要的预处理参数
    try:
        predictor = ModelInference(
            model_path=MODEL_PATH,
            data_config=dataset_config["Data"],
            config=config,
            model_config=model_config,
            class_names=CLASS_NAMES
        )
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请确保模型文件存在，并且你已经运行了训练脚本。")
        return
        
    # --- 3. 数据准备与划分 (与训练时完全一致) ---
    print("\nLoading and splitting dataset to get the validation set...")
    set_seed(config["seed"])  # 使用相同的种子确保划分一致！
    
    full_dataset = TimeSeriesDataset(
        data_dir=dataset_config["Data"]["data_dir"],
        file_pattern=dataset_config["Data"]["file_pattern"],
        window_size=dataset_config["Data"]["window_size"],
        stride=dataset_config["Data"]["stride"],
        target_col=dataset_config["Data"]["target_col"],
        Normalization="Sample", # 与你的脚本一致
        delete_col=dataset_config["Data"]["delete_col"],
        constant_col=dataset_config["Data"]["constant_col"]
    )
    
    # 完全复制你的划分逻辑
    non_train_size = int(dataset_config["Data"]["validation_split"] * len(full_dataset))
    train_size = len(full_dataset) - non_train_size
    train_dataset, val_test_dataset = random_split(full_dataset, [train_size, non_train_size])
    val_size = int(0.5 * len(val_test_dataset))
    test_size = len(val_test_dataset) - val_size
    val_dataset, _ = random_split(val_test_dataset, [val_size, test_size])
    
    # 创建验证集的DataLoader
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config["batch_size"],
        shuffle=True, # 打乱以便能快速找到不同类别的样本
        num_workers=config["num_workers"]
    )

    # --- 4. 从验证集中为每个类别找到一个样本 ---
    print("\nSearching for one sample from each class in the validation set...")
    found_samples_info = {}
    feature_names = full_dataset.get_column_names()

    for data, labels in val_loader:
        for i in range(data.shape[0]):
            sample_data_tensor = data[i]
            sample_label_index = labels[i].item()
            
            if sample_label_index not in found_samples_info:
                print(f"  > Found sample for class: {CLASS_NAMES[sample_label_index]} (Index: {sample_label_index})")
                
                # 将 Pytorch Tensor 转换为 Pandas DataFrame
                sample_df = pd.DataFrame(
                    sample_data_tensor.cpu().numpy(),
                    columns=feature_names
                )
                
                found_samples_info[sample_label_index] = {
                    "df": sample_df,
                    "true_label": CLASS_NAMES[sample_label_index]
                }
            
            # 如果已经找到了所有类别的样本，就提前退出
            if len(found_samples_info) == len(CLASS_NAMES):
                break
        if len(found_samples_info) == len(CLASS_NAMES):
            break

    if len(found_samples_info) < len(CLASS_NAMES):
        print(f"\nWarning: Could only find samples for {len(found_samples_info)} out of {len(CLASS_NAMES)} classes in the validation set.")

    # --- 5. 对收集到的样本进行推理 ---
    print("\nRunning inference on the collected samples...")
    for label_index, info in found_samples_info.items():
        df_to_predict = info['df']
        prediction = predictor.predict(df_to_predict)
        info['predicted_label'] = prediction
        print(f"  > Sample (True: {info['true_label']}) -> Predicted: {prediction}")

    # --- 6. 绘图 ---
    print("\nGenerating plots...")
    sorted_labels = sorted(found_samples_info.keys())
    num_plots = len(sorted_labels)
    
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 4 * num_plots), squeeze=False)
    axes = axes.flatten()

    for i, label_index in enumerate(sorted_labels):
        info = found_samples_info[label_index]
        df_to_plot = info['df']
        true_label = info['true_label']
        predicted_label = info['predicted_label']
        ax = axes[i]
        
        # 绘制特征
        df_to_plot.plot(ax=ax, legend=False)
        
        # 设置标题，并根据预测是否正确使用不同颜色
        is_correct = (true_label == predicted_label)
        title_color = 'green' if is_correct else 'red'
        title = f"True: {true_label}  |  Predicted: {predicted_label}"
        
        ax.set_title(title, fontsize=14, color=title_color, weight='bold')
        ax.set_xlabel("Time Step (within window)")
        ax.set_ylabel("Normalized Value")
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # 将图例放在图外
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')

    fig.suptitle("Model Inference on Samples from Validation Set", fontsize=18, y=1.0)
    plt.tight_layout(rect=[0, 0, 0.9, 0.98]) # 调整布局以适应图例和总标题
    plt.savefig(OUTPUT_FILENAME, bbox_inches='tight')
    plt.show()

    print(f"\nPlot saved successfully to {OUTPUT_FILENAME}")

if __name__ == '__main__':
    main()