import torch
import torch.nn as nn

class LSTMForecastModel(nn.Module):
    """
    一个经典的LSTM模型，用于时序预测。

    模型结构:
    1. LSTM层: 捕捉时间序列中的长期依赖关系。
    2. Dropout层 (可选): 防止过拟合。
    3. 全连接层 (Linear): 将LSTM的输出映射到最终的预测值。
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        """
        初始化模型。

        参数:
        - input_size (int): 输入特征的数量 (num_features)。
        - hidden_size (int): LSTM隐藏层的大小。这个值越大，模型的容量越大，但风险也越高。
        - num_layers (int): LSTM堆叠的层数。多层LSTM可以学习更复杂的模式。
        - output_size (int): 输出的大小。对于单步预测，这个值是1。
        - dropout (float): Dropout的比率，用于正则化。
        """
        super(LSTMForecastModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.softmax = nn.Softmax(dim=1)  # Softmax层用于多分类任务
        # 定义LSTM层
        # batch_first=True 让输入的张量形状为 (batch, seq_len, feature_size)，这与DataLoader的输出一致
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0 # Dropout只在多层LSTM之间生效
        )
        
        # 定义Dropout层（用于LSTM输出和全连接层之间）
        self.dropout = nn.Dropout(dropout)
        
        # 定义全连接层，将LSTM的输出映射到目标维度
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        定义模型的前向传播。

        参数:
        - x (torch.Tensor): 输入的批次数据，形状为 (batch_size, window_size, input_size)。
        """
        # 1. 初始化LSTM的隐藏状态 (h_0) 和细胞状态 (c_0)
        # 形状: (num_layers, batch_size, hidden_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # 2. 将输入和隐藏状态传递给LSTM层
        # lstm_out 包含了所有时间步的输出
        # _ 是最后一个时间步的隐藏状态和细胞状态
        # lstm_out 形状: (batch_size, window_size, hidden_size)
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # 3. 我们只关心序列中最后一个时间步的输出，因为它包含了整个序列的信息
        # lstm_out[:, -1, :] 的形状: (batch_size, hidden_size)
        last_time_step_out = lstm_out[:, -1, :]
        
        # 4. (可选) 应用Dropout
        last_time_step_out = self.dropout(last_time_step_out)
        
        # 5. 将最后一个时间步的输出通过全连接层得到最终预测值
        # out 形状: (batch_size, output_size)
        out = self.fc(last_time_step_out)
        # 6. 应用Softmax激活函数，得到每个类别的概率分布
        return out