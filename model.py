import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import platform

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, padding=1)  # 28x28 -> 28x28
        self.bn1 = nn.BatchNorm2d(4)
        self.pool = nn.MaxPool2d(2, 2)    # 28x28 -> 14x14
        self.conv2 = nn.Conv2d(4, 8, 3, padding=1)  # 14x14 -> 14x14
        self.bn2 = nn.BatchNorm2d(8)
        self.fc1 = nn.Linear(8 * 7 * 7, 32)
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 8 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def save_model(self, path="model"):
        suffix = f"_{platform.system()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_local"
        full_path = f"{path}{suffix}.pth"
        torch.save(self.state_dict(), full_path)
        return full_path 