import torch
import torch.nn as nn

class CNNClassify(nn.Module ):
    """
    用于时间序列分类的CNN模型。
    它由多个卷积层、池化层和全连接层组成。
    """
    def __init__(self, input_length: int, num_classes: int ):
        super(CNNClassify, self).__init__()
        
        self.input_length = input_length
        self.num_classes = num_classes
        
        # 定义卷积层
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        
        # 定义池化层
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # 定义全连接层
        self.fc1 = nn.Linear(32 * (input_length // 2), 128)  # 池化后长度减半
        self.fc2 = nn.Linear(128, num_classes)
        
        # 激活函数
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        Args:
            x: Tensor, shape (batch_size, 1, input_length)
        """
        # 卷积 -> ReLU -> 池化
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        
        # 展平
        x = x.view(x.size(0), -1)  # (batch_size, features)
        
        # 全连接层 -> ReLU -> 输出层
        x = self.relu(self.fc1(x))
        x = self.fc2(x)  # 输出 logits
        
        return x
