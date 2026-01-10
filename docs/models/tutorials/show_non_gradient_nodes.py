title = "Hiding Non-Gradient Nodes"

intro = """
By default, TorchVista shows all nodes in the computation graph, including constant tensors and scalars that don't require gradients.
These non-gradient nodes can clutter the visualization when you only care about the learnable parts of your model.

You can hide these nodes by setting `show_non_gradient_nodes=False` when calling `trace_model`.
"""

conclusion = """
You've learned how to hide non-gradient nodes to create a cleaner visualization focused on the learnable parts of your model.

This is useful when you want to see only the tensors that flow through your model's trainable parameters.

In the next tutorial, we'll learn how to keep tensor inputs and outputs in dicts to give meaningful names to tensor nodes.
"""

code_contents = """\
import torch
import torch.nn as nn
from torchvista import trace_model

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.conv(x))

class SimpleResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1)
        )
        self.block1 = ResidualBlock(16)
        self.block2 = ResidualBlock(16)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, 10)

    def forward(self, inputs):
        x = inputs['image']
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        return self.fc(x)

model = SimpleResNet()
example_input = {'image': torch.randn(1, 3, 32, 32)}

# Hide constant tensors and scalars that don't require gradients
trace_model(
    model,
    example_input,
    forced_module_tracing_depth=3,
    collapse_modules_after_depth=3,
    ##############################
    show_non_gradient_nodes=False  # <-- hides non-gradient nodes
    ##############################
)
"""

# --- Execution code (not displayed) ---
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.conv(x))

class SimpleResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1)
        )
        self.block1 = ResidualBlock(16)
        self.block2 = ResidualBlock(16)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, 10)

    def forward(self, inputs):
        x = inputs['image']
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        return self.fc(x)

model = SimpleResNet()

example_input = {
    'image': torch.randn(1, 3, 32, 32)
}

forced_module_tracing_depth = 3
collapse_modules_after_depth = 3
show_non_gradient_nodes = False
