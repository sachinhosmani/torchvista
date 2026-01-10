title = "Tracing depth control"

intro = """
In the last tutorial, we saw how to explore nested modules in the visualization, but some inbuilt modules like `Conv2d` and `BatchNorm2d` could not be expanded.

Now we show the use of `forced_module_tracing_depth` to trace into any modules within the specified nesting depth.
Simply set `forced_module_tracing_depth` to the desired depth when calling `trace_model`. Now you can click the "+" button on modules like `Conv2d` in this model.
"""

conclusion = """
You've now seen how to use `forced_module_tracing_depth` to trace into inbuilt PyTorch modules.

This is useful when you want to see the internals of inbuilt modules which are not traced by default.

But now let us say you want to expand the graph fully, which requires you to click the "+" button on every module which can be tedious.
In the next tutorial, we will learn how to automatically expand all modules to a certain depth when initially displayed.
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

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        return self.fc(x)

model = SimpleResNet()
example_input = torch.randn(1, 3, 32, 32)

# Trace up to depth 3 to see inside Conv2d, BatchNorm2d, etc.
trace_model(
    model,
    example_input,
    forced_module_tracing_depth=3
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

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        return self.fc(x)

model = SimpleResNet()
example_input = torch.randn(1, 3, 32, 32)

forced_module_tracing_depth = 3
