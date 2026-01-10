title = "Exploring Nested Modules"

intro = """
Here we look at a slightly more complex model with nested modules.
The code to trace remains the same, but this time, try clicking the "+" button on the `Sequential` modules to see what lies inside.
"""

conclusion = """
Now you've seen how to expand and collapse nested modules in the visualization.

But you might have noticed that inbuilt modules that you didn't define yourself, such as `Conv2d`, `BatchNorm2d` cannot be expanded.
This is because TorchVista only traces modules defined in your code by default, to avoid cluttering the visualization with low-level details.

Next we will learn how to overcome this by forcing the tracing depth.
"""

code_contents = """\
import torch
import torch.nn as nn
from torchvista import trace_model

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # This Sequential can be expanded in the visualization
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
        self.stem = nn.Conv2d(3, 16, 3, padding=1)
        # These nested modules can be expanded too
        self.block1 = ResidualBlock(16)
        self.block2 = ResidualBlock(16)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        return self.fc(x)

model = SimpleResNet()
example_input = torch.randn(1, 3, 32, 32)

trace_model(model, example_input)
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
        self.stem = nn.Conv2d(3, 16, 3, padding=1)
        self.block1 = ResidualBlock(16)
        self.block2 = ResidualBlock(16)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        return self.fc(x)

model = SimpleResNet()
example_input = torch.randn(1, 3, 32, 32)
