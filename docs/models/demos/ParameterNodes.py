import torch
import torch.nn as nn

class SubModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(64, 64)
        self.scale = nn.Parameter(torch.ones(64))

    def forward(self, x):
        return self.linear(x) * self.scale

class ModelWithParameters(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(64))
        self.sub = SubModule()

    def forward(self, x):
        x = x + self.bias
        x = x + torch.ones(64)
        x = torch.relu(x)
        x = self.sub(x)
        return x

model = ModelWithParameters()
example_input = torch.randn(1, 64)

code_contents = """\
import torch
import torch.nn as nn
from torchvista import trace_model

class SubModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(64, 64)
        self.scale = nn.Parameter(torch.ones(64))

    def forward(self, x):
        return self.linear(x) * self.scale

class ModelWithParameters(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(64))
        self.sub = SubModule()

    def forward(self, x):
        x = x + self.bias
        x = x + torch.ones(64)
        x = torch.relu(x)
        x = self.sub(x)
        return x

model = ModelWithParameters()
example_input = torch.randn(1, 64)

trace_model(model, example_input)

"""
