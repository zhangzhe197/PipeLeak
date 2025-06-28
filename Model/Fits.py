import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
from Model.Transformer import TransformerClassificationModel
class FFTClassify(nn.Module):

    def __init__(self, seq_len, classNum, feature_num, minFreq, mid_layer_dim = 512, individual=False):
        super(FFTClassify, self).__init__()
        self.seq_len = seq_len
        self.classNum = classNum
        self.channels = feature_num
        self.minFreq = minFreq # minFreq现在是高通滤波的下限

        # 计算FFT后的总频率分量数量
        self.total_freq_bins =  mid_layer_dim // 2 + 1

        # 高通滤波后，实际保留的高频分量数量
        # 从 minFreq 索引开始，到 total_freq_bins-1 结束
        self.num_high_freq_bins = self.total_freq_bins - self.minFreq

        if self.num_high_freq_bins <= 0:
            raise ValueError(f"minFreq ({minFreq}) is too high for seq_len ({seq_len}). No high frequency bins remaining.")
        flattened_input_dim_mlp = self.num_high_freq_bins * 2 
        
        self.transformer = TransformerClassificationModel(
            input_size=2 * self.channels,
            d_model=64,
            nhead=8,
            num_encoder_layers=3,
            dim_feedforward=256,
            output_size=self.classNum ,
            dropout=0.1,
            max_len=seq_len // 2 + 1  # 确保最大长度与FFT后的频率分量数量一致
        )



    def forward(self, x):
        # x 形状: (batch_size, seq_len, channels)

        # RIN (Optional, but good practice for time series)
        # 考虑你的数据是否需要归一化，如果已经归一化，可以移除
        # x_mean = torch.mean(x, dim=1, keepdim=True)
        # x = x - x_mean
        # x_var = torch.var(x, dim=1, keepdim=True) + 1e-5
        # x = x / torch.sqrt(x_var)

        # 1. FFT
        full_specx = torch.fft.rfft(x, dim=1) # 形状: (batch_size, total_freq_bins, channels)
        real_part = full_specx.real # (batch_size, total_freq_bins, channels)
        imag_part = full_specx.imag # (batch_size, total_freq_bins, channels)

        # 拼接实部和虚部
        # combined_features 形状: (batch_size, total_freq_bins, channels * 2)
        combined_features = torch.cat((real_part, imag_part), dim=2) 
        return self.transformer(combined_features)
        
      