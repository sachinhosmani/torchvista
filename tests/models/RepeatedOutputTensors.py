import torch
import torch.nn as nn


class Container0:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Container:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.container0 = Container0(x, y)

class CM(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4, 4)

    def forward(self, x):
        y = self.linear1(x)
        return {0: self.linear1(self.linear1(x) + 3), 1: {0: self.linear1(x) + torch.ones(2,4)}, 2: Container(x - 2, y - 2)}

  
class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4, 4)
        self.cm = CM()

    def forward(self, x):
        y = self.cm(x)[0]
        return Container(x, y)

model = CustomModel()

example_input = torch.randn(2, 4)

code_contents = """\
import torch
import torch.nn as nn
from torchvista import trace_model

class Container0:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Container:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.container0 = Container0(x, y)

class CM(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4, 4)

    def forward(self, x):
        y = self.linear1(x)
        return {0: self.linear1(self.linear1(x) + 3), 1: {0: self.linear1(x) + torch.ones(2,4)}, 2: Container(x - 2, y - 2)}

  
class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4, 4)
        self.cm = CM()

    def forward(self, x):
        y = self.cm(x)[0]
        return Container(x, y)

model = CustomModel()

example_input = torch.randn(2, 4)

trace_model(model, example_input)

"""
