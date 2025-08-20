import torch
import torch.nn as nn

class UNetPP(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv00 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv10 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv01 = nn.Conv2d(16 + 32, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.final = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        x00 = torch.relu(self.conv00(x))
        x10 = torch.relu(self.conv10(self.pool(x00)))
        x10_up = self.up(x10)
        x01 = torch.relu(self.conv01(torch.cat([x00, x10_up], dim=1)))
        return self.final(x01)

model = UNetPP()
example_input = torch.randn(1, 1, 64, 64)

code_contents = """\
import torch
import torch.nn as nn
from torchvista import trace_model

class UNetPP(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv00 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv10 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv01 = nn.Conv2d(16 + 32, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.final = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        x00 = torch.relu(self.conv00(x))
        x10 = torch.relu(self.conv10(self.pool(x00)))
        x10_up = self.up(x10)
        x01 = torch.relu(self.conv01(torch.cat([x00, x10_up], dim=1)))
        return self.final(x01)

model = UNetPP()
example_input = torch.randn(1, 1, 64, 64)

trace_model(model, example_input)

"""
