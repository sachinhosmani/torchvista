import torch
import torch.nn as nn


import torch
import torch.nn as nn
from torchvista import trace_model

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class DeepCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1 = ConvBlock(3, 32)
        self.stage2 = ConvBlock(32, 64)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(64, 10)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        return self.classifier(x)

model = DeepCNN()
example_input = torch.randn(1, 3, 64, 64)

forced_module_tracing_depth = 4

code_contents = """\
import torch
import torch.nn as nn
from torchvista import trace_model

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class DeepCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1 = ConvBlock(3, 32)
        self.stage2 = ConvBlock(32, 64)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(64, 10)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        return self.classifier(x)

model = DeepCNN()
example_input = torch.randn(1, 3, 64, 64)

trace_model(model, example_input, forced_module_tracing_depth=4)

"""
