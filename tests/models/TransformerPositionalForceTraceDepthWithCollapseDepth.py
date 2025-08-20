import torch
import torch.nn as nn

class PositionalTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(20, 32)
        self.pos_embed = nn.Parameter(torch.randn(10, 1, 32))
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=32, nhead=4)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        self.fc = nn.Linear(32, 5)

    def forward(self, x):
        x = self.embed(x) + self.pos_embed[:x.size(0)]
        x = self.encoder(x)
        return self.fc(x[0])  # classification using first token

model = PositionalTransformer()
example_input = torch.randn(10, 1, 20)
forced_module_tracing_depth = 1
collapse_modules_after_depth = 0

code_contents = """\
import torch
import torch.nn as nn
from torchvista import trace_model

class PositionalTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(20, 32)
        self.pos_embed = nn.Parameter(torch.randn(10, 1, 32))
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=32, nhead=4)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        self.fc = nn.Linear(32, 5)

    def forward(self, x):
        x = self.embed(x) + self.pos_embed[:x.size(0)]
        x = self.encoder(x)
        return self.fc(x[0])

model = PositionalTransformer()
example_input = torch.randn(10, 1, 20)

trace_model(model, example_input, forced_module_tracing_depth=1, collapse_modules_after_depth=0)
"""
