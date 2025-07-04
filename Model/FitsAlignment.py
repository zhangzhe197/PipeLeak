import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- 1. 定义一个独立的、可复用的Transformer编码器模块 ---
class TransformerEncoder(nn.Module):
    """一个通用的Transformer编码器模块，用于特征提取。"""
    def __init__(self, input_size, d_model, nhead, num_encoder_layers, dim_feedforward, dropout=0.1, max_len=5000):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        
        # 输入层：将输入特征维度映射到d_model
        self.input_projection = nn.Linear(input_size, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True  # 确保输入形状是 (batch, seq, feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # 初始化权重
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.input_projection.weight.data.uniform_(-initrange, initrange)
        self.input_projection.bias.data.zero_()

    def forward(self, src):
        # src shape: (batch_size, seq_len, input_size)
        
        # 1. 投影到 d_model
        src = self.input_projection(src) * math.sqrt(self.d_model)
        
        # 2. 添加位置编码
        src = self.pos_encoder(src)
        
        # 3. 通过Transformer编码器
        # TransformerEncoderLayer默认输出与输入形状相同
        output = self.transformer_encoder(src) # shape: (batch_size, seq_len, d_model)
        
        # 4. 特征池化：取序列维度的平均值得到一个固定大小的特征向量
        pooled_output = output.mean(dim=1) # shape: (batch_size, d_model)
        
        return pooled_output

class PositionalEncoding(nn.Module):
    """位置编码模块，为序列中的每个位置添加唯一的位置信息。"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.transpose(0, 1)) # shape: (1, max_len, d_model)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# --- 2. 定义主模型：PairedFFTTransformer ---
class PairedFFTTransformer(nn.Module):
    """
    处理配对时间序列的模型。
    对每个输入序列分别进行FFT，通过独立的Transformer编码器，
    融合特征后，通过MLP进行分类。
    """
    def __init__(self, 
                 # 输入数据参数
                 feature_num_A: int, feature_num_B: int,
                 seq_len_A: int, seq_len_B: int,
                 class_num: int,
                 
                 # Transformer参数 (两个分支共享)
                 d_model: int = 64, 
                 nhead: int = 8, 
                 num_encoder_layers: int = 3, 
                 dim_feedforward: int = 256, 
                 dropout: float = 0.1,
                 
                 # MLP分类头参数
                 mlp_hidden_dim: int = 128):
        
        super(PairedFFTTransformer, self).__init__()
        
        # FFT后的频率分量数量
        freq_bins_A = seq_len_A // 2 + 1
        freq_bins_B = seq_len_B // 2 + 1

        # Transformer的输入维度是FFT后实部和虚部拼接的结果，所以是 feature_num * 2
        transformer_input_size_A = feature_num_A * 2
        transformer_input_size_B = feature_num_B * 2

        print("--- PairedFFTTransformer 初始化 ---")
        print(f"分支 A: 特征数={feature_num_A}, 序列长度={seq_len_A}, FFT后频率点={freq_bins_A}")
        print(f"分支 B: 特征数={feature_num_B}, 序列长度={seq_len_B}, FFT后频率点={freq_bins_B}")
        
        # --- 创建两个独立的Transformer编码器分支 ---
        self.transformer_A = TransformerEncoder(
            input_size=transformer_input_size_A,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=freq_bins_A  # 最大长度与FFT后的频率分量数量一致
        )
        
        self.transformer_B = TransformerEncoder(
            input_size=transformer_input_size_B,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=freq_bins_B  # 最大长度与FFT后的频率分量数量一致
        )
        
        # --- MLP 分类头 ---
        # 输入维度是 d_model，因为两个Transformer的输出特征向量相加后维度不变
        self.classifier = nn.Sequential(
            nn.Linear(d_model, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, class_num)
        )
        print(f"分类器输入维度: {d_model}, 输出类别数: {class_num}")
        print("--- 初始化完成 ---\n")

    def forward(self, x1, x2):
        # x1 形状: (batch_size, seq_len_A, feature_num_A)
        # x2 形状: (batch_size, seq_len_B, feature_num_B)
        
        # --- 分支 A 处理 ---
        # 1. FFT
        fft_x1 = torch.fft.rfft(x1, dim=1) # (batch, freq_bins_A, feature_num_A)
        # 2. 拼接实部和虚部
        fft_features_x1 = torch.cat((fft_x1.real, fft_x1.imag), dim=2) # (batch, freq_bins_A, feature_num_A * 2)
        # 3. 通过Transformer A
        features_A = self.transformer_A(fft_features_x1) # (batch, d_model)

        # --- 分支 B 处理 ---
        # 1. FFT
        fft_x2 = torch.fft.rfft(x2, dim=1) # (batch, freq_bins_B, feature_num_B)
        # 2. 拼接实部和虚部
        fft_features_x2 = torch.cat((fft_x2.real, fft_x2.imag), dim=2) # (batch, freq_bins_B, feature_num_B * 2)
        # 3. 通过Transformer B
        features_B = self.transformer_B(fft_features_x2) # (batch, d_model)
        
        # --- 特征融合 ---
        # 元素级相加
        fused_features = features_A + features_B
        
        # --- 分类 ---
        output = self.classifier(fused_features) # (batch, class_num)
        
        return output