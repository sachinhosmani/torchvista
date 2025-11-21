import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvista import trace_model

# Define a submodule that contains a Sequential of 4 identical Linear layers
class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, 2*dim),
            nn.Linear(2*dim, dim),
            nn.Linear(dim, dim),
            nn.Linear(dim, dim),
            nn.Linear(dim, 2*dim),
            nn.Linear(2*dim, dim),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return self.layers(x)

# Define the main model with 2 such blocks in a ModuleList
class MainModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.blocks = nn.ModuleList([Block(dim) for _ in range(2)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

# Instantiate the model
dim = 8
model = MainModel(dim)

# Example input tensor
x = torch.randn(4, dim)

trace_model(model, x)