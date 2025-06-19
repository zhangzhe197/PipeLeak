import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
import glob

class TimeSeriesDataset(Dataset):
    """
    优化版的数据集，针对多分类任务进行了调整。
    - 在初始化时将所有数据转换为PyTorch Tensor。
    - 确保 __getitem__ 返回适合 CrossEntropyLoss 的标签。
    """
    def __init__(self, data_dir, file_pattern, window_size, stride=1, target_col='LeakType'):
        self.window_size = window_size
        self.stride = stride
        
        file_paths = sorted(glob.glob(os.path.join(data_dir, file_pattern)))
        
        if not file_paths:
            raise FileNotFoundError(f"在目录 '{data_dir}' 中未找到匹配 '{file_pattern}' 的文件。")
            
        self.data_tensors = []
        
        print("正在加载数据并转换为Tensor...")
        
        # 确定特征列和目标列的整数索引
        temp_df = pd.read_csv(file_paths[0])
        if target_col not in temp_df.columns:
            raise ValueError(f"目标列 '{target_col}' 在文件 {file_paths[0]} 中不存在。")
        
        if isinstance(target_col, str):
            self.target_idx = temp_df.columns.get_loc(target_col)
        else:
            self.target_idx = target_col
        
        self.feature_cols_names = [col for col in temp_df.columns if col != target_col]
        self.feature_indices = [temp_df.columns.get_loc(col) for col in self.feature_cols_names]

        # 遍历文件，读取并立即转换为Tensor
        for file_path in file_paths:
            df = pd.read_csv(file_path)
        
            tensor_data = torch.tensor(df.values, dtype=torch.float32)
            self.data_tensors.append(tensor_data)
        
        # 创建索引映射表
        self.index_map = []
        print("正在创建样本索引...")
        for file_idx, tensor in enumerate(self.data_tensors):
            # 确保有足够的长度来创建一个窗口
            if len(tensor) > self.window_size:
                max_start_idx = len(tensor) - self.window_size
                for start_idx in range(0, max_start_idx, self.stride):
                    self.index_map.append((file_idx, start_idx))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        # 1. 查找元数据
        file_idx, start_idx = self.index_map[idx]
        
        # 2. 从Tensor列表中获取对应的Tensor
        data_tensor = self.data_tensors[file_idx]
        
        # 3. 切片窗口和标签
        end_idx = start_idx + self.window_size
        
        # 特征已经是 float32 类型
        features = data_tensor[start_idx:end_idx, self.feature_indices]
        
        # 从原始的 float32 Tensor 中切片出标签
        label_tensor = data_tensor[end_idx, self.target_idx]
        
        # --- 关键修改点 2 ---
        # 这是修复所有问题的核心。
        # 1. 不再使用 torch.tensor() 包装。
        # 2. 将标签的类型从 float32 转换为 long。
        # 3. label_tensor 已经是0D标量，不需要 squeeze()。
        label = label_tensor.long()
        
        return features, label
    
    def get_feature_number(self):
        """
        返回特征数量。
        """
        return len(self.feature_indices)