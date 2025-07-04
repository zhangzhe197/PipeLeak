import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
import glob
from collections import defaultdict
import pdb
class PairedTimeSeriesDataset(Dataset):
    """
    一个用于处理配对时间序列的数据集类。
    - 从一个目录中读取两组具有不同采样频率的文件（例如，传感器A和传感器B的数据）。
    - 接收以秒为单位的窗口和步长，并自动计算以数据点为单位的窗口大小。
    - 在 __getitem__ 中返回两个特征张量和一个共享标签。
    - 保留了 Z-score 归一化（全局、样本级或无）、列删除和常量列处理等功能。
    """

    def __init__(self, data_dir: str, 
                 file_pattern_A: str, file_pattern_B: str,
                 window_size_sec: float, 
                 total_duration_sec: float = 30.0,
                 stride_sec: float = 1.0, 
                 delete_col: list = [], 
                 constant_col: list = [], 
                 target_col: str = 'LeakType', 
                 Normalization: str = "None"):
        """
        初始化数据集。

        :param data_dir: 包含CSV文件的数据目录。
        :param file_pattern_A: 用于匹配第一类文件的模式（例如 "*_sensorA.csv"）。
        :param file_pattern_B: 用于匹配第二类文件的模式（例如 "*_sensorB.csv"）。
        :param window_size_sec: 时间窗口的长度（单位：秒）。
        :param total_duration_sec: 每个CSV文件代表的总时长（单位：秒）。
        :param stride_sec: 滑动窗口的步长（单位：秒）。
        :param delete_col: 需要从特征中删除的列名列表。
        :param constant_col: 在样本归一化中不参与计算的常量特征列名列表。
        :param target_col: 目标（标签）列的名称。
        :param Normalization: 归一化处理的方式，可选 "None"、"Sample" 或 "All"。
        """
        print("--- 初始化 PairedTimeSeriesDataset ---")
        self.target_col = target_col
        self.Normalization = Normalization

        # 1. 文件配对
        self.file_pairs = self._pair_files(data_dir, file_pattern_A, file_pattern_B)
        if not self.file_pairs:
            raise FileNotFoundError("未能找到任何配对的文件。请检查 data_dir 和 file_patterns。")

        # 2. 计算频率和窗口/步长（以数据点为单位）
        self._calculate_timing_parameters(window_size_sec, stride_sec, total_duration_sec)

        # 3. 获取特征元数据（列名、索引等）
        first_pair_key = next(iter(self.file_pairs))
        df_A = pd.read_csv(self.file_pairs[first_pair_key]['A'])
        df_B = pd.read_csv(self.file_pairs[first_pair_key]['B'])
        
        print("\n--- 设置元数据 (Type A) ---")
        self.metadata_A = self._get_feature_metadata(df_A, delete_col, constant_col, "A")
        
        print("\n--- 设置元数据 (Type B) ---")
        self.metadata_B = self._get_feature_metadata(df_B, delete_col, constant_col, "B")

        # 4. 如果是全局归一化，计算均值和标准差
        if self.Normalization == "All":
            self._calculate_global_normalization_params()
        
        # 5. 加载所有数据到内存
        print("\n--- 正在加载所有数据到Tensor ---")
        self.data_tensors = defaultdict(dict)
        for key, paths in self.file_pairs.items():
            df_A = pd.read_csv(paths['A'])
            df_B = pd.read_csv(paths['B'])
            self.data_tensors[key]['A'] = torch.tensor(df_A.values, dtype=torch.float32)
            self.data_tensors[key]['B'] = torch.tensor(df_B.values, dtype=torch.float32)
        print("所有数据已加载。")

        # 6. 创建索引映射表
        self._create_index_map()
        print(f"\n--- 初始化完成。共找到 {len(self)} 个有效样本。---\n")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        # 1. 查找元数据
        key, start_idx_A, start_idx_B = self.index_map[idx]
        
        # 2. 从Tensor列表中获取对应的Tensor
        tensor_A = self.data_tensors[key]['A']
        tensor_B = self.data_tensors[key]['B']
        
        # 3. 切片窗口
        end_idx_A = start_idx_A + self.window_size_A_pts
        end_idx_B = start_idx_B + self.window_size_B_pts
        
        features_A = tensor_A[start_idx_A:end_idx_A, self.metadata_A['feature_indices']]
        features_B = tensor_B[start_idx_B:end_idx_B, self.metadata_B['feature_indices']]

        # 4. 应用归一化
        features_A = self._apply_normalization(features_A, 'A')
        features_B = self._apply_normalization(features_B, 'B')

        # 5. 提取标签 (假设两份数据的标签是一致的，我们从A中提取)
        label_tensor = tensor_A[end_idx_A - 1, self.metadata_A['target_idx']]
        label = label_tensor.long()
        
        # 6. 返回两个特征张量和一个标签
        return features_A, features_B, label

    # --- Helper Methods ---
    
    def _pair_files(self, data_dir, pattern_A, pattern_B):
        """根据文件名模式配对文件。"""
        paths_A = glob.glob(os.path.join(data_dir, pattern_A))
        paths_B = glob.glob(os.path.join(data_dir, pattern_B))
        
        # 创建一个字典，方便快速查找
        paths_B_dict = {os.path.basename(p): p for p in paths_B}
        file_pairs = {}
        # 通过替换模式中的通配符来找到配对文件
        base_pattern_A = pattern_A.replace("*", "")
        base_pattern_B = pattern_B.replace("*", "")

        for path_A in paths_A:
            filename_A = os.path.basename(path_A)
            # 提取共同的文件名部分
            number = filename_A.split(".")[0].split("_")[-1]  # 假设数字在文件名的最后部分
            expected_filename_B = pattern_B.replace("*", number)
            if expected_filename_B in paths_B_dict:
                file_pairs[number] = {'A': path_A, 'B': paths_B_dict[expected_filename_B]}
            else:
                print(f"警告: 找到文件 {filename_A}，但未找到其配对文件 {expected_filename_B}。")
                
        print(f"找到 {len(file_pairs)} 对有效文件。")
        return file_pairs

    def _calculate_timing_parameters(self, window_size_sec, stride_sec, total_duration_sec):
        """计算采样频率和以数据点为单位的窗口/步长。"""
        first_pair_key = next(iter(self.file_pairs))
        df_A = pd.read_csv(self.file_pairs[first_pair_key]['A'])
        df_B = pd.read_csv(self.file_pairs[first_pair_key]['B'])
        
        # 计算频率 (点/秒)
        self.freq_A = len(df_A) / total_duration_sec
        self.freq_B = len(df_B) / total_duration_sec
        
        # 计算窗口大小和步长 (以数据点为单位)
        self.window_size_A_pts = int(window_size_sec * self.freq_A)
        self.window_size_B_pts = int(window_size_sec * self.freq_B)
        self.stride_A_pts = max(1, int(stride_sec * self.freq_A)) # 步长至少为1
        
        print("\n--- 时间参数计算 ---")
        print(f"频率 (Type A): {self.freq_A:.2f} Hz")
        print(f"频率 (Type B): {self.freq_B:.2f} Hz")
        print(f"窗口大小 (Type A): {self.window_size_A_pts} 点 (基于 {window_size_sec} 秒)")
        print(f"窗口大小 (Type B): {self.window_size_B_pts} 点 (基于 {window_size_sec} 秒)")
        print(f"步长 (Type A): {self.stride_A_pts} 点 (基于 {stride_sec} 秒)")

        if self.window_size_A_pts == 0 or self.window_size_B_pts == 0:
            raise ValueError("计算出的窗口大小为0。请检查 window_size_sec 或数据文件长度。")


    def _get_feature_metadata(self, df, delete_col, constant_col, data_type):
        """提取特征列、索引等元数据。"""
        metadata = {}
        if self.target_col not in df.columns and  data_type == "A":
            raise ValueError(f"目标列 '{self.target_col}' 在文件中不存在。")
        if data_type == "A":   
            metadata['target_idx'] = df.columns.get_loc(self.target_col)
        metadata['feature_cols_names'] = [c for c in df.columns if (c != self.target_col and c not in delete_col)]
        metadata['feature_indices'] = [df.columns.get_loc(c) for c in metadata['feature_cols_names']]

        var_indices, const_indices = [], []
        for i, col_name in enumerate(metadata['feature_cols_names']):
            if col_name in constant_col:
                const_indices.append(i)
            else:
                var_indices.append(i)
        
        metadata['variable_indices'] = var_indices
        metadata['constant_indices'] = const_indices

        print(f"特征列 (Type {data_type}): {metadata['feature_cols_names']}")
        if self.Normalization == "Sample":
            print(f"  - 可变特征索引: {metadata['variable_indices']}")
            print(f"  - 常量特征索引: {metadata['constant_indices']}")
            
        return metadata

    def _calculate_global_normalization_params(self):
        """为全局归一化计算均值和标准差。"""
        print("\n--- 正在为全局归一化收集数据 ---")
        features_A_list, features_B_list = [], []
        for key, paths in self.file_pairs.items():
            df_A = pd.read_csv(paths['A'])
            df_B = pd.read_csv(paths['B'])
            features_A_list.append(df_A[self.metadata_A['feature_cols_names']].values)
            features_B_list.append(df_B[self.metadata_B['feature_cols_names']].values)

        combined_A = np.concatenate(features_A_list, axis=0)
        combined_B = np.concatenate(features_B_list, axis=0)

        epsilon = 1e-8
        self.mean_A = torch.tensor(np.mean(combined_A, axis=0), dtype=torch.float32)
        self.std_A = torch.tensor(np.std(combined_A, axis=0), dtype=torch.float32)
        self.std_A = torch.where(self.std_A == 0, torch.tensor(epsilon), self.std_A)
        
        self.mean_B = torch.tensor(np.mean(combined_B, axis=0), dtype=torch.float32)
        self.std_B = torch.tensor(np.std(combined_B, axis=0), dtype=torch.float32)
        self.std_B = torch.where(self.std_B == 0, torch.tensor(epsilon), self.std_B)

        print("全局归一化参数已计算。")

    def _create_index_map(self):
        """创建从整数索引到 (文件key, 起始点A, 起始点B) 的映射。"""
        print("\n--- 正在创建样本索引 ---")
        self.index_map = []
        for key in self.file_pairs.keys():
            tensor_A = self.data_tensors[key]['A']
            tensor_B = self.data_tensors[key]['B']
            
            if len(tensor_A) < self.window_size_A_pts or len(tensor_B) < self.window_size_B_pts:
                print(f"警告: 文件对 {key} 的长度不足，将被跳过。")
                continue

            max_start_idx_A = len(tensor_A) - self.window_size_A_pts + 1
            for start_idx_A in range(0, max_start_idx_A, self.stride_A_pts):
                # 根据A的起始点，按时间比例计算B的起始点
                start_time_sec = start_idx_A / self.freq_A
                start_idx_B = int(start_time_sec * self.freq_B)
                
                # 确保B的窗口不会越界
                if start_idx_B + self.window_size_B_pts <= len(tensor_B):
                    self.index_map.append((key, start_idx_A, start_idx_B))

    def _apply_normalization(self, features, data_type):
        """对给定的特征张量应用归一化。"""
        if self.Normalization == "None":
            return features
            
        metadata = self.metadata_A if data_type == 'A' else self.metadata_B
        
        if self.Normalization == "All":
            mean = self.mean_A if data_type == 'A' else self.mean_B
            std = self.std_A if data_type == 'A' else self.std_B
            return (features - mean) / std
            
        elif self.Normalization == "Sample":
            if metadata['variable_indices']:
                variable_features = features[:, metadata['variable_indices']]
                sample_mean = variable_features.mean(dim=0)
                sample_std = variable_features.std(dim=0) + 1e-8
                normalized_variable = (variable_features - sample_mean) / sample_std
                features[:, metadata['variable_indices']] = normalized_variable
            return features

    # --- Public Getter Methods ---
    
    def get_feature_numbers(self):
        """返回两类特征的数量 (A, B)。"""
        return len(self.metadata_A['feature_indices']), len(self.metadata_B['feature_indices'])
    
    def get_column_names(self):
        """返回两类特征的列名列表 ({'A': [...], 'B': [...]})。"""
        return {
            'A': self.metadata_A['feature_cols_names'],
            'B': self.metadata_B['feature_cols_names']
        }