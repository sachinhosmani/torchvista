title = "Basic Usage"

intro = """
Let us trace a simple module which adds the outputs of two linear layers.

When you run this code, you get an intractive visualization of the model's forward pass. You can pan and zoom the graph to explore it.
Click on the "i" Show info button to see the parameters and attributes of each module or operation.
"""

conclusion = """
You've learned how to visualize the forward pass of a simple PyTorch model using `trace_model`.

Next let us look at a more complex model with nesting.
"""

code_contents = """\
import torch
import torch.nn as nn
from torchvista import trace_model

class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 5)
        self.linear2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear1(x) + self.linear2(x)

model = LinearModel()
example_input = torch.randn(2, 10)

# Visualize the forward pass
trace_model(model, example_input)
"""

# --- Execution code (not displayed) ---
import torch
import torch.nn as nn

class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 5)
        self.linear2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear1(x) + self.linear2(x)

model = LinearModel()
example_input = torch.randn(2, 10)
