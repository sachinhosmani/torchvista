import torch
import torch.nn as nn

class CompressionTest5(nn.Module):
    def __init__(self):
        super().__init__()
        block = nn.Sequential(*[nn.Linear(64, 64) for _ in range(10)])
        self.layers = nn.ModuleList([block] * 10)

    def forward(self, x):
        for seq in self.layers:
            x = seq(x)
        return x

model = CompressionTest5()
example_input = torch.randn(2, 64)

show_compressed_view = True

code_contents = """\
import torch
import torch.nn as nn
from torchvista import trace_model

class CompressionTest5(nn.Module):
    def __init__(self):
        super().__init__()
        block = nn.Sequential(*[nn.Linear(64, 64) for _ in range(10)])
        self.layers = nn.ModuleList([block] * 10)

    def forward(self, x):
        for seq in self.layers:
            x = seq(x)
        return x

model = CompressionTest5()
example_input = torch.randn(2, 64)

trace_model(model, example_input, show_compressed_view=True)

"""
