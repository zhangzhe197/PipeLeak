import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np # 用于计算均值和标准差
import os
import glob, pdb

class TimeSeriesDataset(Dataset):
    """
    优化版的数据集，针对多分类任务进行了调整。
    - 在初始化时将所有数据转换为PyTorch Tensor。
    - 确保 __getitem__ 返回适合 CrossEntropyLoss 的标签。
    - 添加了Z-score数据归一化。
    支持样本级别的归一化处理, 全部数据的归一化处理, 和不做归一化处理
    """
    def __init__(self, data_dir, file_pattern, window_size, stride=1, delete_col = [],constant_col = [], target_col='LeakType', Normalization="None"):
#    初始化数据集。, 并指定归一化处理的方式 "None "、"Sample" 或 "All"。
        self.window_size = window_size
        self.stride = stride
        self.target_col = target_col
        self.Normalization = Normalization
        file_paths = sorted(glob.glob(os.path.join(data_dir, file_pattern)))
        
        if not file_paths:
            raise FileNotFoundError(f"在目录 '{data_dir}' 中未找到匹配 '{file_pattern}' 的文件。")
            
        self.data_tensors = []
        self.feature_indices = [] # 存储特征列的索引
        self.constant_cols = constant_col
        print("正在初始化数据集并计算归一化参数...")
        
        # 1. 确定特征列和目标列的整数索引
        # 读取第一个文件以确定列结构
        temp_df = pd.read_csv(file_paths[0])
        if self.target_col not in temp_df.columns:
            raise ValueError(f"目标列 '{self.target_col}' 在文件 {file_paths[0]} 中不存在。")
        
        self.target_idx = temp_df.columns.get_loc(self.target_col)
        self.feature_cols_names = [col for col in temp_df.columns if (col != self.target_col and col not in delete_col) ]
        self.feature_indices = [temp_df.columns.get_loc(col) for col in self.feature_cols_names]

        print(self.feature_cols_names, "特征列已确定。")
        print(self.feature_indices, "特征索引已确定。")
        # 2. 收集所有特征数据用于计算均值和标准差
        # 这一步会遍历所有文件，将特征数据临时存储起来，用于全局统计量的计算。
        # 对于非常大的数据集，这可能仍然占用大量内存。
        # 更优化的方法是增量计算均值和方差，但对于多数情况，这种方法足够且简单。
        if self.Normalization == "All":
            all_features_data_np = []
            for file_path in file_paths:
                df = pd.read_csv(file_path)
                # 确保只选取特征列，并转换为 numpy 数组
                all_features_data_np.append(df[self.feature_cols_names].values) 
            
            # 将所有特征数据合并为一个大的 NumPy 数组
            # np.concatenate 是为了把所有文件的特征数据堆叠起来，方便统一计算统计量
            combined_features = np.concatenate(all_features_data_np, axis=0)
            
            # 3. 计算均值和标准差
            self.mean = torch.tensor(np.mean(combined_features, axis=0), dtype=torch.float32)
            self.std = torch.tensor(np.std(combined_features, axis=0), dtype=torch.float32)
            
            # 4. 处理标准差为零的情况（例如，某个特征是常数）
            # 添加一个小的 epsilon 来避免除以零
            epsilon = 1e-8
            self.std = torch.where(self.std == 0, torch.tensor(epsilon, dtype=torch.float32), self.std)
            
            print(f"计算得到的特征均值: {self.mean}")
            print(f"计算得到的特征标准差: {self.std}")
            print("数据归一化参数已计算。")

        # 5. 加载数据并转换为Tensor（与之前相同）
        print("正在加载数据并转换为Tensor...")
        for file_path in file_paths:
            df = pd.read_csv(file_path)
            tensor_data = torch.tensor(df.values, dtype=torch.float32)
            self.data_tensors.append(tensor_data)

        # 6. 创建索引映射表（与之前相同）
        self.index_map = []
        print("正在创建样本索引...")
        for file_idx, tensor in enumerate(self.data_tensors):
            # 确保有足够的长度来创建一个窗口
            # 修正：如果label是窗口末尾的元素，那么需要保证window_size-1是合法索引
            # 如果label是窗口后的一个元素，则需要保证len(tensor) > end_idx
            # 这里我们假设 LeakType 是对当前窗口的分类，所以标签取窗口的最后一个点
            if len(tensor) >= self.window_size: # 至少要能形成一个完整的窗口
                # 最后一个可以开始的索引，确保 end_idx - 1 是有效的
                max_start_idx = len(tensor) - self.window_size + 1 
                for start_idx in range(0, max_start_idx, self.stride):
                    self.index_map.append((file_idx, start_idx))
            else:
                print(f"Warning: 文件 {file_paths[file_idx]} (长度 {len(tensor)}) 小于窗口大小 {self.window_size}, 将被跳过。")


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
        # 4. 应用归一化
        if self.Normalization == "All":
            # 使用预先计算的均值和标准差进行 Z-score 标准化
            features = (features - self.mean) / self.std
        elif self.Normalization == "Sample":
            # 对每个样本进行 Z-score 标准化
            sample_mean = features.mean(dim=0)
            sample_std = features.std(dim=0) + 1e-8 # 避免除以零
            features = (features - sample_mean) / sample_std
        # 如果 Normalization 是 "None
        # 从原始的 float32 Tensor 中切片出标签
        # 假设 LeakType 标签是对应窗口末尾的类型
        label_tensor = data_tensor[end_idx - 1, self.target_idx] 
        
        # 将标签的类型从 float32 转换为 long，因为 CrossEntropyLoss 需要 Long 类型
        label = label_tensor.long()
        
        return features, label
    
    def get_feature_number(self):
        """
        返回特征数量。
        """
        return len(self.feature_indices)
    
    def get_column_names(self):
        """
        返回特征列的名称列表。
        """
        return self.feature_cols_names