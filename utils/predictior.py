import torch
import pandas as pd
import numpy as np
import os

# --- 从你的项目中导入必要的模块 ---
# 确保这些模块的路径相对于你运行此脚本的位置是正确的
from Model.LstmModel import LSTMForecastModel
from Model.Transformer import TransformerClassificationModel
from Model.Fits import FFTClassify

class ModelInference:
    """
    一个模型推理接口层，用于对单个DataFrame进行预测。
    该接口固定使用与训练时一致的样本级别归一化方法。
    """
    def __init__(self, model_path, config,data_config, model_config, class_names=None):
        """
        初始化推理接口。

        Args:
            model_path (str): 预训练模型文件的路径 (e.g., 'FFTbest_classification_model.pth').
            config (dict): 主配置文件，包含 window_size, delete_col, constant_col 等。
            model_config (dict): 模型特定的配置文件。
            class_names (list, optional): 类别名称列表，顺序与标签索引一致。
                                          如果提供，predict方法将返回类别名称而非索引。
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件未找到: {model_path}")

        # --- 1. 设置配置和设备 ---
        self.config = config
        self.data_config = data_config  # 用于数据处理的配置
        self.model_config = model_config
        self.class_names = class_names
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"推理接口正在使用设备: {self.device}")

        # --- 2. 预计算数据处理参数 (模仿 TimeSeriesDataset 的 __init__) ---
        # 这些参数在整个生命周期内都是固定的，提前计算可以提高效率
        self.window_size = int(self.config["freq"] * self.config["window_size"])  # 转换为采样点数
        
        # 假设特征列可以从配置中推断或需要一个样本文件来确定
        # 为了简单起见，我们假设可以通过 config['delete_col'] 和 config['target_col']
        # 以及一个已知的完整列列表来确定特征列。
        # 这里我们用一种更通用的方式，直接从config中读取或硬编码
        # 注意：这里的 feature_cols_names 顺序必须与训练时完全一致！
        # 如果无法从config确定，最好的方法是保存训练时的列名。
        # 假设 config 中有 `all_columns` 字段
        if 'all_columns' not in self.data_config:
             raise ValueError("Config必须包含一个'all_columns'列表，用于确定特征顺序。")
        
        all_cols = self.data_config['all_columns']
        delete_cols = self.data_config.get('delete_col', [])
        self.feature_cols_names = [col for col in all_cols if col not in delete_cols]
        print(f"使用的特征列: {self.feature_cols_names}")
        
        # 计算在最终特征张量中，可变列和常量列的索引
        self.variable_feature_indices = []
        self.constant_feature_indices = []
        constant_cols = set(self.data_config.get('constant_col', []))
        for i, col_name in enumerate(self.feature_cols_names):
            if col_name in constant_cols:
                self.constant_feature_indices.append(i)
            else:
                self.variable_feature_indices.append(i)
        
        print(f"可变特征索引: {self.variable_feature_indices}")
        print(f"常量特征索引: {self.constant_feature_indices}")
        
        # --- 3. 加载模型 ---
        input_size = len(self.feature_cols_names)
        model_type = self.config['model']
        
        if model_type == "LSTM":
            self.model = LSTMForecastModel(input_size=input_size, **self.model_config["LSTM"])
        elif model_type == "Transformer":
            self.model = TransformerClassificationModel(input_size=input_size, max_len=self.window_size, **self.model_config["Transformer"])
        elif model_type == "FFT":
            self.model = FFTClassify(seq_len=self.window_size, feature_num=input_size, classNum = self.model_config["FFT"]["output_size"],minFreq=int(config["window_size"] // 2))
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()  # !! 关键：将模型设置为评估模式 !!
        print(f"模型 '{model_type}' 已成功加载并设置为评估模式。")

    def _preprocess(self, df: pd.DataFrame) -> torch.Tensor:
        """
        对输入的DataFrame进行预处理，使其符合模型输入要求。
        """
        # --- 1. 验证输入 ---
        if not isinstance(df, pd.DataFrame):
            raise TypeError("输入必须是 pandas DataFrame。")
        if len(df) < self.window_size:
            raise ValueError(f"输入DataFrame的行数 ({len(df)}) 必须大于或等于窗口大小 ({self.window_size})。")
        
        missing_cols = set(self.feature_cols_names) - set(df.columns)
        if missing_cols:
            raise ValueError(f"输入DataFrame缺少以下必需的列: {missing_cols}")

        # --- 2. 准备数据 ---
        # 仅选择特征列，并确保顺序正确
        df_features = df[self.feature_cols_names]
        
        # 从DataFrame的末尾截取一个窗口
        df_window = df_features.tail(self.window_size)
        
        # 转换为Tensor
        features = torch.tensor(df_window.values, dtype=torch.float32)

        # --- 3. 应用样本归一化 (逻辑与TimeSeriesDataset完全一致) ---
        if self.variable_feature_indices:
            variable_features = features[:, self.variable_feature_indices]
            
            sample_mean = variable_features.mean(dim=0)
            sample_std = variable_features.std(dim=0) + 1e-8 # 避免除以零
            
            normalized_variable_features = (variable_features - sample_mean) / sample_std
            
            features[:, self.variable_feature_indices] = normalized_variable_features
        
        # --- 4. 调整形状以匹配模型输入 (添加batch维度) ---
        # 形状从 [window_size, num_features] -> [1, window_size, num_features]
        return features.unsqueeze(0).to(self.device)

    def predict(self, df: pd.DataFrame):
        """
        对输入的DataFrame进行预测。

        Args:
            df (pd.DataFrame): 包含时间序列数据的DataFrame，至少需要有 'window_size' 行。

        Returns:
            - (str): 如果初始化时提供了 class_names，则返回预测的类别名称。
            - (int): 如果未提供 class_names，则返回预测的类别索引。
        """
        # 使用 no_grad 来关闭梯度计算，加速推理并节省内存
        with torch.no_grad():
            # 1. 预处理数据
            input_tensor = self._preprocess(df)
            # 2. 模型推理
            outputs = self.model(input_tensor)
        #    outputs = torch.nn.Softmax(dim=1)(outputs)  # 应用 softmax 获取概率分布
            # 3. 获取预测结果
            # outputs 的形状是 [1, num_classes]，我们取概率最大的那个
            _, predicted_index_tensor = torch.max(outputs.data, 1)
            predicted_index = predicted_index_tensor.item()

            # 4. 返回用户友好的结果
            if self.class_names:
                if 0 <= predicted_index < len(self.class_names):
                    return self.class_names[predicted_index]
                else:
                    return f"Unknown Index: {predicted_index}"
            else:
                return predicted_index
