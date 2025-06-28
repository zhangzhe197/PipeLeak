import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np # 用于计算均值和标准差
import os
import glob

class TimeSeriesDataset(Dataset):
    """
    优化版的数据集，针对多分类任务进行了调整。
    - 在初始化时将所有数据转换为PyTorch Tensor。
    - 确保 __getitem__ 返回适合 CrossEntropyLoss 的标签。
    - 添加了Z-score数据归一化。
    - 支持样本级别的归一化处理, 全部数据的归一化处理, 和不做归一化处理。
    - **新增**: 在样本级别归一化时，可以指定常量列(constant_col)，这些列将不参与归一化计算。
    """
    def __init__(self, data_dir, file_pattern, window_size, stride=1, delete_col = [], constant_col = [], target_col='LeakType', Normalization="None"):
        """
        初始化数据集。
        :param data_dir: 包含CSV文件的数据目录。
        :param file_pattern: 用于匹配文件的模式，例如 "*.csv"。
        :param window_size: 时间序列窗口的长度。
        :param stride: 滑动窗口的步长。
        :param delete_col: 需要从特征中删除的列名列表。
        :param constant_col: 在样本归一化中不参与计算的常量特征列名列表。
        :param target_col: 目标（标签）列的名称。
        :param Normalization: 归一化处理的方式，可选 "None"、"Sample" 或 "All"。
        """
        self.window_size = window_size
        self.stride = stride
        self.target_col = target_col
        self.Normalization = Normalization
        file_paths = sorted(glob.glob(os.path.join(data_dir, file_pattern)))
        
        if not file_paths:
            raise FileNotFoundError(f"在目录 '{data_dir}' 中未找到匹配 '{file_pattern}' 的文件。")
            
        self.data_tensors = []
        self.constant_cols = constant_col
        print("正在初始化数据集并计算归一化参数...")
        
        # 1. 确定特征列和目标列的整数索引
        temp_df = pd.read_csv(file_paths[0])
        if self.target_col not in temp_df.columns:
            raise ValueError(f"目标列 '{self.target_col}' 在文件 {file_paths[0]} 中不存在。")
        
        self.target_idx = temp_df.columns.get_loc(self.target_col)
        self.feature_cols_names = [col for col in temp_df.columns if (col != self.target_col and col not in delete_col)]
        self.feature_indices = [temp_df.columns.get_loc(col) for col in self.feature_cols_names]

        # ==================== 新增逻辑：为样本归一化准备索引 ====================
        # 这个逻辑计算了在最终的 features 张量中，哪些列是可变的，哪些是常量的。
        # 这样在 __getitem__ 中就可以直接使用这些索引来选择性地进行归一化。
        self.variable_feature_indices_in_features = []
        self.constant_feature_indices_in_features = []
        for i, col_name in enumerate(self.feature_cols_names):
            if col_name in self.constant_cols:
                self.constant_feature_indices_in_features.append(i)
            else:
                self.variable_feature_indices_in_features.append(i)
        # =================================================================

        print(f"特征列已确定: {self.feature_cols_names}")
        print(f"特征列在原数据中的索引: {self.feature_indices}")
        if self.Normalization == "Sample":
             print(f"可变特征在特征张量中的索引: {self.variable_feature_indices_in_features}")
             print(f"常量特征在特征张量中的索引: {self.constant_feature_indices_in_features}")


        # 2. 如果是全局归一化，则收集所有特征数据用于计算均值和标准差
        if self.Normalization == "All":
            all_features_data_np = []
            print("正在为全局归一化收集数据...")
            for file_path in file_paths:
                df = pd.read_csv(file_path)
                all_features_data_np.append(df[self.feature_cols_names].values) 
            
            combined_features = np.concatenate(all_features_data_np, axis=0)
            
            # 3. 计算均值和标准差
            self.mean = torch.tensor(np.mean(combined_features, axis=0), dtype=torch.float32)
            self.std = torch.tensor(np.std(combined_features, axis=0), dtype=torch.float32)
            
            # 4. 处理标准差为零的情况
            epsilon = 1e-8
            self.std = torch.where(self.std == 0, torch.tensor(epsilon, dtype=torch.float32), self.std)
            
            print(f"计算得到的特征均值: {self.mean}")
            print(f"计算得到的特征标准差: {self.std}")
            print("数据归一化参数已计算。")

        # 5. 加载数据并转换为Tensor
        print("正在加载数据并转换为Tensor...")
        for file_path in file_paths:
            df = pd.read_csv(file_path)
            tensor_data = torch.tensor(df.values, dtype=torch.float32)
            self.data_tensors.append(tensor_data)

        # 6. 创建索引映射表
        self.index_map = []
        print("正在创建样本索引...")
        for file_idx, tensor in enumerate(self.data_tensors):
            if len(tensor) >= self.window_size:
                max_start_idx = len(tensor) - self.window_size + 1 
                for start_idx in range(0, max_start_idx, self.stride):
                    self.index_map.append((file_idx, start_idx))
            else:
                print(f"警告: 文件 {file_paths[file_idx]} (长度 {len(tensor)}) 小于窗口大小 {self.window_size}, 将被跳过。")


    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        # 1. 查找元数据
        file_idx, start_idx = self.index_map[idx]
        
        # 2. 从Tensor列表中获取对应的Tensor
        data_tensor = self.data_tensors[file_idx]
        
        # 3. 切片窗口和标签
        end_idx = start_idx + self.window_size
        features = data_tensor[start_idx:end_idx, self.feature_indices]
        
        # 4. 应用归一化
        if self.Normalization == "All":
            # 使用预先计算的均值和标准差进行 Z-score 标准化
            features = (features - self.mean) / self.std
            
        elif self.Normalization == "Sample":
            # ==================== 修改后的样本归一化逻辑 ====================
            # 确保有可变特征列时才进行操作
            if self.variable_feature_indices_in_features:
                # 仅选择可变特征列进行计算
                variable_features = features[:, self.variable_feature_indices_in_features]
                
                # 计算这些可变列的均值和标准差
                sample_mean = variable_features.mean(dim=0)
                sample_std = variable_features.std(dim=0) + 1e-8 # 避免除以零
                
                # 仅对可变特征列进行归一化
                normalized_variable_features = (variable_features - sample_mean) / sample_std
                
                # 将归一化后的数据放回原特征张量的对应位置，常量列保持不变
                features[:, self.variable_feature_indices_in_features] = normalized_variable_features
            # 如果没有可变特征列（即所有特征都是常量列），则不执行任何操作。
            # =============================================================

        # 5. 提取标签
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