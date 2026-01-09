import torch
import torch.nn as nn

class CompressionTest1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(64, 64),
            nn.Linear(64, 64),
            nn.Linear(64, 64),
            nn.Linear(64, 64),
        )

    def forward(self, x):
        return self.layers(x)

model = CompressionTest1()

example_input = torch.randn(2, 64)

show_compressed_view = True

code_contents = """\
import torch
import torch.nn as nn
from torchvista import trace_model

class CompressionTest1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(64, 64),
            nn.Linear(64, 64),
            nn.Linear(64, 64),
            nn.Linear(64, 64),
        )

    def forward(self, x):
        return self.layers(x)

model = CompressionTest1()
example_input = torch.randn(2, 64)

trace_model(model, example_input, show_compressed_view=True)

"""
