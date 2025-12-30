import torch
import torch.nn as nn

class CompressionTest8(nn.Module):
    """
    4 linear layers with non-matching dimensions - no compression should be detected.
    Each layer has different input/output dimensions, so they cannot be grouped.
    """
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(64, 128),
            nn.Linear(128, 256),
            nn.Linear(256, 512),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        return self.layers(x)

model = CompressionTest8()

example_input = torch.randn(2, 64)

show_compressed_view = True

code_contents = """\
import torch
import torch.nn as nn
from torchvista import trace_model

class CompressionTest8(nn.Module):
    \"\"\"
    4 linear layers with non-matching dimensions - no compression should be detected.
    Each layer has different input/output dimensions, so they cannot be grouped.
    \"\"\"
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(64, 128),
            nn.Linear(128, 256),
            nn.Linear(256, 512),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        return self.layers(x)

model = CompressionTest8()
example_input = torch.randn(2, 64)

trace_model(model, example_input, show_compressed_view=True)

"""
