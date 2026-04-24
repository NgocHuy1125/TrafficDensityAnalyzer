import torch
import torch.nn as nn
from torchvision import models

class ResNetWith4Channels(nn.Module):
    # Khởi tạo mô hình với số lớp đầu ra (num_classes) mặc định là 3
    def __init__(self, num_classes=3):
        super().__init__()
        # Sử dụng mô hình ResNet-50 đã được huấn luyện trước
        self.resnet = models.resnet50(pretrained=True)
          # Thay đổi lớp convolution đầu tiên để chấp nhận đầu vào có 4 kênh (ví dụ: RGBA)
        self.resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Thay đổi lớp fully connected (fc) cuối cùng để có số lớp đầu ra là num_classes
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)
