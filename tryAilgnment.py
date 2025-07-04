import glob
import os
import pdb
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import random

# --- 从项目中导入必要的模块 ---
from config import config, model_config, dataset_config
from Loader.SeriesLoader import TimeSeriesDataset
from utils.predictior import ModelInference
MODEL_PATH = "FFTbest_classification_model.pth"
CLASS_NAMES = ["CC", "GL", "LC", "OL", "NL"]
NUM_PIPELINES = 3 # 我们将模拟的管道数量
INITIAL_TIMESTEP = 200 # 初始显示的时间步
WINDOW_SIZE = int(config["window_size"] * config["freq"])  # 窗口大小转换为采样点数
from Loader.SeriesLoaderAlignment import PairedTimeSeriesDataset
import torch
import pdb
def load_model_and_data():
    """
    加载预训练模型和样本数据并转换为DataFrame格式。
    使用Streamlit缓存避免重复加载。
    """

    # --- 1. 初始化模型推理接口 ---
    try:
        predictor = ModelInference(
            model_path=MODEL_PATH,
            config=config,
            data_config=dataset_config["Data"],
            model_config=model_config,
            class_names=CLASS_NAMES
        )
    except Exception as e:
        print(f"加载模型失败: {e}")
        return None, None, None

    # --- 2. 读取原始CSV文件并转换为DataFrame ---
    try:
        # 获取所有数据文件路径
        data_dir = dataset_config["Data"]["data_dir"]
        file_pattern = dataset_config["Data"]["file_pattern"]
        file_paths = sorted(glob.glob(os.path.join(data_dir, file_pattern)))
        
        if not file_paths:
            print(f"在 '{data_dir}' 中未找到匹配 '{file_pattern}' 的文件")
            return predictor, None, None

        # 读取第一个文件获取特征列信息
        temp_df = pd.read_csv(file_paths[0])
        target_col = dataset_config["Data"]["target_col"]
        delete_col = dataset_config["Data"]["delete_col"]
        
        # 确定特征列（与SeriesLoader保持一致）
        feature_cols_names = [col for col in temp_df.columns 
                            if col != target_col and col not in delete_col]
        
        # 读取所有文件并存储为DataFrame列表
        all_dataframes = []
        for file_path in file_paths:
            df = pd.read_csv(file_path)
            # 仅保留特征列和目标列
            df = df[feature_cols_names + [target_col]]
            all_dataframes.append(df)
        pdb.set_trace()  # 调试断点，检查加载的内容

    except Exception as e:
        print(f"读取数据文件失败: {e}")
        return predictor, None, None

    # --- 3. 随机选择样本作为模拟管道数据 ---
    try:
        # 确保有足够的样本
        if len(all_dataframes) < NUM_PIPELINES:
            print(f"样本数量不足，使用全部 {len(all_dataframes)} 个样本")
            selected_indices = list(range(len(all_dataframes)))
        else:
            selected_indices = random.sample(range(len(all_dataframes)), NUM_PIPELINES)

        # 选择样本并确保每个DataFrame有足够行数
        pipeline_data = []
        for idx in selected_indices:
            df = all_dataframes[idx]
            if len(df) < WINDOW_SIZE:
                print(f"样本 {idx} 的行数 {len(df)} 小于窗口大小，跳过该样本")
                continue
            pipeline_data.append(df)

        # 如果筛选后样本不足，使用全部可用样本
        if len(pipeline_data) == 0:
            print("所有数据文件均小于窗口大小，无法生成有效样本")
            return predictor, None, feature_cols_names

    except Exception as e:
        print(f"样本选择失败: {e}")
        return predictor, None, feature_cols_names

    return predictor, pipeline_data, feature_cols_names

a , b ,c = load_model_and_data()
