import torch
import torch.nn as nn

class BlockWithParameter(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(64, 64)
        self.scale = nn.Parameter(torch.ones(64))

    def forward(self, x):
        return self.linear(x) * self.scale

class CompressionTest13(nn.Module):
    """Compression test with nn.Parameter nodes inside repeated blocks."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            BlockWithParameter(),
            BlockWithParameter(),
            BlockWithParameter(),
            BlockWithParameter(),
        )

    def forward(self, x):
        return self.layers(x)

model = CompressionTest13()
example_input = torch.randn(2, 64)

show_compressed_view = True

code_contents = """\
import torch
import torch.nn as nn
from torchvista import trace_model

class BlockWithParameter(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(64, 64)
        self.scale = nn.Parameter(torch.ones(64))

    def forward(self, x):
        return self.linear(x) * self.scale

class CompressionTest13(nn.Module):
    \"\"\"Compression test with nn.Parameter nodes inside repeated blocks.\"\"\"
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            BlockWithParameter(),
            BlockWithParameter(),
            BlockWithParameter(),
            BlockWithParameter(),
        )

    def forward(self, x):
        return self.layers(x)

model = CompressionTest13()
example_input = torch.randn(2, 64)

trace_model(model, example_input, show_compressed_view=True)

"""
