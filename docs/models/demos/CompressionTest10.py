import torch
import torch.nn as nn

class CompressionTest10(nn.Module):
    """
    Parallel branches with repeating linear layers (2 in each chain).
    Input goes to both a Sequential and a ModuleList with matching dims,
    then outputs are combined.
    """
    def __init__(self):
        super().__init__()
        self.branch_sequential = nn.Sequential(
            nn.Linear(64, 64),
            nn.Linear(64, 64),
        )
        self.branch_modulelist = nn.ModuleList([
            nn.Linear(64, 64),
            nn.Linear(64, 64),
        ])

    def forward(self, x):
        out1 = self.branch_sequential(x)
        out2 = x
        for layer in self.branch_modulelist:
            out2 = layer(out2)
        return out1 + out2

model = CompressionTest10()

example_input = torch.randn(2, 64)

show_compressed_view = True

code_contents = """\
import torch
import torch.nn as nn
from torchvista import trace_model

class CompressionTest10(nn.Module):
    \"\"\"
    Parallel branches with repeating linear layers (2 in each chain).
    Input goes to both a Sequential and a ModuleList with matching dims,
    then outputs are combined.
    \"\"\"
    def __init__(self):
        super().__init__()
        self.branch_sequential = nn.Sequential(
            nn.Linear(64, 64),
            nn.Linear(64, 64),
        )
        self.branch_modulelist = nn.ModuleList([
            nn.Linear(64, 64),
            nn.Linear(64, 64),
        ])

    def forward(self, x):
        out1 = self.branch_sequential(x)
        out2 = x
        for layer in self.branch_modulelist:
            out2 = layer(out2)
        return out1 + out2

model = CompressionTest10()
example_input = torch.randn(2, 64)

trace_model(model, example_input, show_compressed_view=True)

"""
