import torch
import torch.nn as nn
from torchvista import trace_model

class TensorInjector(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Get the shape of the input
        B, C, H, W = x.shape

        # Create a new tensor filled with zeros, same shape as x
        new_tensor = torch.zeros((B, C, H, W), device=x.device, dtype=x.dtype)

        # Example manipulation: double the input and write into new tensor
        new_tensor[:, :, :, :] = 2 * x

        return new_tensor

model = TensorInjector()
example_input = torch.randn(1, 3, 64, 64)

trace_model(model, example_input)

code_contents = """\
import torch
import torch.nn as nn
from torchvista import trace_model

class TensorInjector(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Get the shape of the input
        B, C, H, W = x.shape

        # Create a new tensor filled with zeros, same shape as x
        new_tensor = torch.zeros((B, C, H, W), device=x.device, dtype=x.dtype)

        # Example manipulation: double the input and write into new tensor
        new_tensor[:, :, :, :] = 2 * x

        # Return the new tensor
        return new_tensor

model = TensorInjector()
example_input = torch.randn(1, 3, 64, 64)

trace_model(model, example_input)

"""
