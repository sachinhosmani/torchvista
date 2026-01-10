title = "Compressed View for Repeated Structures"

intro = """
When your model has many repeated layers (like a stack of identical blocks), the visualization can become very long and repetitive.

TorchVista can detect these repeated patterns and compress them into a single "repeated" node. But this is possible only when the repeated structures are within `nn.ModuleList` or `nn.Sequential` as a chain.

Set `show_compressed_view=True` to enable this feature.
"""

conclusion = """
You've now completed the TorchVista tutorial series! Check out the demos page for more examples.
"""

code_contents = """\
import torch
import torch.nn as nn
from torchvista import trace_model

class DeepModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 10 identical Sequential blocks, each with 10 Linear layers
        block = nn.Sequential(*[nn.Linear(64, 64) for _ in range(10)])
        self.layers = nn.ModuleList([block] * 10)

    def forward(self, x):
        for seq in self.layers:
            x = seq(x)
        return x

model = DeepModel()
example_input = torch.randn(2, 64)

# Compress repeated structures into a single representation
trace_model(
    model,
    example_input,
    ###############################
    show_compressed_view=True  # <-- compresses repeated layers
    ###############################
)
"""

# --- Execution code (not displayed) ---
import torch
import torch.nn as nn

class DeepModel(nn.Module):
    def __init__(self):
        super().__init__()
        block = nn.Sequential(*[nn.Linear(64, 64) for _ in range(10)])
        self.layers = nn.ModuleList([block] * 10)

    def forward(self, x):
        for seq in self.layers:
            x = seq(x)
        return x

model = DeepModel()
example_input = torch.randn(2, 64)

show_compressed_view = True
