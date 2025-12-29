import torch
import torch.nn as nn

class TransposePropertiesModel(nn.Module):
    """
    Demonstrates the three tensor transpose properties: .T, .mT, and .H
    - .T: Simple transpose (swaps all dimensions for 2D, reverses all dims for nD)
    - .mT: Matrix transpose (transposes last two dimensions, works on batches)
    - .H: Hermitian transpose (conjugate transpose for complex tensors)
    """
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8, dtype=torch.cfloat)

    def forward(self, x):
        # .T - simple 2D transpose
        y1 = self.linear(x)
        out1 = y1 @ y1.T

        # .mT - batch matrix transpose (transposes last 2 dims)
        y2 = self.linear(x)
        out2 = y2 @ y2.mT

        # .H - Hermitian (conjugate) transpose for complex tensors
        y3 = self.linear(x)
        out3 = y3 @ y3.H

        return out1, out2.mean(dim=0), out3

model = TransposePropertiesModel()

example_input = (
    torch.randn(4, 8, dtype=torch.cfloat),
)

code_contents = """\
import torch
import torch.nn as nn
from torchvista import trace_model

class TransposePropertiesModel(nn.Module):
    \"\"\"
    Demonstrates the three tensor transpose properties: .T, .mT, and .H
    - .T: Simple transpose (swaps all dimensions for 2D, reverses all dims for nD)
    - .mT: Matrix transpose (transposes last two dimensions, works on batches)
    - .H: Hermitian transpose (conjugate transpose for complex tensors)
    \"\"\"
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(8, 8)
        self.linear2 = nn.Linear(8, 8)
        self.linear3 = nn.Linear(8, 8, dtype=torch.cfloat)

    def forward(self, x):
        # .T - simple 2D transpose
        y1 = self.linear1(x)
        out1 = y1 @ y1.T

        # .mT - batch matrix transpose (transposes last 2 dims)
        y2 = self.linear2(x)
        out2 = y2 @ y2.mT

        # .H - Hermitian (conjugate) transpose for complex tensors
        y3 = self.linear3(x)
        out3 = y3 @ y3.H

        return out1, out2.mean(dim=0), out3

model = TransposePropertiesModel()

example_input = (
    torch.randn(4, 8),              # 2D input for .T
    torch.randn(2, 4, 8),           # 3D batch input for .mT
    torch.randn(4, 8, dtype=torch.cfloat),  # complex input for .H
)

trace_model(model, example_input)

"""
