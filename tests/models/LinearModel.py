import torch
import torch.nn as nn

class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)

model = LinearModel()

example_input = torch.randn(2, 10)

code_contents = """\
import torch
import torch.nn as nn
from torchvista import trace_model

class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)

model = LinearModel()
example_input = torch.randn(2, 10)

trace_model(model, example_input)

"""
