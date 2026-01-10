title = "Control initial visual expansion depth?"

intro = """
In the last tutorial you learned how to trace modules to the desired nesting depth, but if you wanted to expand and view everything it would be tedious to click the "+" button on every module recursively.

This is because TorchVista by default collapses all modules beyond the depth of `1` to keep the visualization clean, and lets you click "+" to expand them as needed.

Now we show the use of `collapse_modules_after_depth` to control this behaviour. Just set `collapse_modules_after_depth` to the desired depth when calling `trace_model`, and only modules beyond that depth will be collapsed initially.

So in our case we will increase it to 3 to match the `forced_module_tracing_depth`, so that all modules up to depth 3 are expanded when the visualization is first displayed.
"""

conclusion = """
Remember that `forced_module_tracing_depth` controls how deep modules are traced, while `collapse_modules_after_depth` controls how deep modules initially appear expanded in the visualization.

This visualization has many raw tensor and scalar nodes which don't require gradients. In the next tutorial, we'll learn how to hide these non-gradient nodes to simplify the visualization.
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

# Expand modules up to depth 3 when first displayed
trace_model(
    model,
    example_input,
    forced_module_tracing_depth=3,
    ##############################
    collapse_modules_after_depth=3  # <-- modules beyond this depth start collapsed
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

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        return self.fc(x)

model = SimpleResNet()
example_input = torch.randn(1, 3, 32, 32)

forced_module_tracing_depth = 3
collapse_modules_after_depth = 3
