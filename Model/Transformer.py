# Model/TransformerModel.py

import torch
import torch.nn as nn
import math
import pdb
class PositionalEncoding(nn.Module):
    """
    实现Transformer的位置编码。
    PositionalEncoding adds information about the relative or absolute position
    of the tokens in the sequence.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1) # shape: (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)) # shape: (d_model/2)
        
        # 核心修改：pe 的维度顺序调整为 (1, max_len, d_model)
        # 这样在 forward 中与 (batch_size, seq_len, d_model) 相加时，
        # batch 维度为1可以被广播，而 seq_len 维度直接匹配。
        pe = torch.zeros(1, max_len, d_model) # <--- 修改这里
        
        pe[0, :, 0::2] = torch.sin(position * div_term) # <--- 修改这里，从第0个batch维度，所有seq_len
        pe[0, :, 1::2] = torch.cos(position * div_term) # <--- 修改这里
        
        self.register_buffer('pe', pe) 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape (batch_size, seq_len, embedding_dim)
        """
        # Add positional encoding to the input embeddings
        # x: (batch_size, seq_len, d_model)
        # pe: (max_len, 1, d_model)
        x = x + self.pe[:, :x.size(1)] # <--- 修改这里，对第0维(batch维)不切片
        return self.dropout(x)

class TransformerClassificationModel(nn.Module):
    """
    用于时间序列分类的Transformer模型。
    它由一个输入线性投影、位置编码、一个Transformer编码器堆栈和一个分类头组成。
    """
    def __init__(self, input_size: int, d_model: int, nhead: int, 
                 num_encoder_layers: int, dim_feedforward: int, 
                 output_size: int, dropout: float = 0.1, max_len: int = 256):
        super().__init__()
        
        # 确保 d_model 是 nhead 的倍数
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) 必须是 nhead ({nhead}) 的倍数")

        self.model_type = 'Transformer'
        self.d_model = d_model
        
        # 1. 输入投影层：将原始特征维度映射到 Transformer 的 d_model 维度
        self.input_projection = nn.Linear(input_size, d_model)
        
        # 2. 位置编码
        # max_len 应该至少等于你的 window_size
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len)
        
        # 3. Transformer 编码器层
        # batch_first=True 表示输入张量的形状是 (batch_size, sequence_length, features)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True # IMPORTANT: ensure batch dimension is first
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        # 4. 分类头
        # 经过Transformer编码后，我们通常会对序列维度进行池化（例如平均池化）
        # 将 d_model 维度的向量映射到 output_size (类别数)
        self.classification_head = nn.Linear(d_model, output_size)
        self.softmax = nn.Softmax(dim=1)
        self._init_weights() # 初始化权重

    def _init_weights(self):
        """
        标准Transformer模型的权重初始化。
        """
        initrange = 0.1
        self.input_projection.weight.data.uniform_(-initrange, initrange)
        self.classification_head.bias.data.zero_()
        self.classification_head.weight.data.uniform_(-initrange, initrange)
        # TransformerEncoderLayer 内部已经有默认的初始化

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src: Tensor, shape (batch_size, seq_len, input_size)
                 seq_len corresponds to window_size
                 input_size corresponds to the number of features
        """
        # 1. 输入投影
        src = self.input_projection(src) * math.sqrt(self.d_model) # 缩放因子很重要
        
        # 2. 添加位置编码
        src = self.positional_encoding(src)
        
        # 3. 通过Transformer编码器
        # 对于时间序列，我们通常不需要 src_mask 或 src_key_padding_mask
        # 如果有变长序列和padding，则需要 src_key_padding_mask
        output = self.transformer_encoder(src) # output shape: (batch_size, seq_len, d_model)
        
        # 4. 池化：将序列维度降维到单个向量
        # 常见方法：全局平均池化 (GAP)
        # 也可以使用 max pooling 或者添加一个 [CLS] token
        pooled_output = output.mean(dim=1) # shape: (batch_size, d_model)
        
        # 5. 分类
        logits = self.classification_head(pooled_output) # shape: (batch_size, output_size)
        return logits