import torch
import torch.nn as nn

class MiniInception(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch1 = nn.Conv2d(3, 8, 1)
        self.branch3 = nn.Conv2d(3, 8, 3, padding=1)
        self.branch5 = nn.Conv2d(3, 8, 5, padding=2)
        self.pool = nn.Conv2d(3, 8, 1)
        self.final = nn.Conv2d(32, 10, 1)

    def forward(self, x):
        b1 = torch.relu(self.branch1(x))
        b3 = torch.relu(self.branch3(x))
        b5 = torch.relu(self.branch5(x))
        bp = torch.relu(self.pool(torch.max_pool2d(x, 3, stride=1, padding=1)))
        return self.final(torch.cat([b1, b3, b5, bp], dim=1))

model = MiniInception()
example_input = torch.randn(1, 3, 32, 32)

code_contents = """\
import torch
import torch.nn as nn
from torchvista import trace_model

class MiniInception(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch1 = nn.Conv2d(3, 8, 1)
        self.branch3 = nn.Conv2d(3, 8, 3, padding=1)
        self.branch5 = nn.Conv2d(3, 8, 5, padding=2)
        self.pool = nn.Conv2d(3, 8, 1)
        self.final = nn.Conv2d(32, 10, 1)

    def forward(self, x):
        b1 = torch.relu(self.branch1(x))
        b3 = torch.relu(self.branch3(x))
        b5 = torch.relu(self.branch5(x))
        bp = torch.relu(self.pool(torch.max_pool2d(x, 3, stride=1, padding=1)))
        return self.final(torch.cat([b1, b3, b5, bp], dim=1))

model = MiniInception()
example_input = torch.randn(1, 3, 32, 32)

trace_model(model, example_input)

"""
