import torch
import torch.nn as nn

class CNNClassify(nn.Module):
    """
    用于时间序列分类的CNN模型，使用二维卷积 nn.Conv2d。
    它将输入时间序列视为 (batch_size, in_channels, 1, input_length) 的图像。

    Args:
        in_channels (int): 输入时间序列的特征通道数 (例如，对于多变量时间序列)。
        input_length (int): 输入时间序列的长度。
        num_classes (int): 分类的类别数量。
    """
    def __init__(self, in_channels: int, input_length: int, num_classes: int):
        super(CNNClassify, self).__init__()
        
        self.in_channels = 256
        self.input_length = input_length
        self.num_classes = num_classes
        
        # 定义二维卷积层和池化层
        # 注意：Conv2d 的 kernel_size 和 padding 都是 (height, width)
        # 我们这里把高度设为1，宽度设为3，以模拟1D卷积
        self.features = nn.Sequential(
            # Conv2d: (in_channels, 1, input_length) -> (512, 1, input_length)
            # 这里的 in_channels 必须与传入模型的数据通道数匹配！
            nn.Conv2d(in_channels=self.in_channels, out_channels=512, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            # MaxPool2d: (512, 1, input_length) -> (512, 1, input_length // 2)
            # 池化也是在宽度（时间序列长度）维度上进行
            nn.MaxPool2d(kernel_size=(1, 2)), 
            
            # Conv2d: (512, 1, input_length // 2) -> (512, 1, input_length // 2)
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            # MaxPool2d: (512, 1, input_length // 2) -> (512, 1, input_length // 4)
            nn.MaxPool2d(kernel_size=(1, 2))
        )
        
        # 动态计算全连接层fc1的输入特征数
        with torch.no_grad():
            # 创建一个假的输入张量来通过卷积层，以确定展平后的维度
            # 注意这里：形状从 (batch_size, in_channels, input_length) 变为
            # (batch_size, in_channels, 1, input_length)
            dummy_input = torch.randn(1, self.in_channels, 1, self.input_length)
            dummy_output = self.features(dummy_input)
            
            self.num_features_before_fc = dummy_output.numel() // dummy_output.shape[0]
            
            if self.num_features_before_fc == 0:
                raise ValueError(
                    f"Calculated 0 features for FC layer. Input length {input_length} "
                    f"might be too short after pooling. (Current processed length: {dummy_output.shape[-1]})"
                    f"Consider increasing input_length or reducing pooling."
                )

        # 定义全连接层
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_classes)
        )
        
    def forward(self, x):
        """
        Args:
            x: Tensor, shape (batch_size, in_channels, input_length)
                在输入到卷积层之前，会添加一个虚拟维度，使其变为 (batch_size, in_channels, 1, input_length)。
        """
        # 添加一个虚拟的第3维 (高度)，使其变为 4D 张量，以适应 Conv2d
        x = x.unsqueeze(2) # 形状变为 (batch_size, in_channels, 1, input_length)
        
        # 经过卷积层和池化层提取特征
        x = self.features(x)
        
        # 展平特征，准备输入全连接层
        x = x.view(x.size(0), -1)  # (batch_size, total_features_before_fc)
        
        # 经过全连接分类层
        x = self.classifier(x)  # 输出 logits
        
        return x
