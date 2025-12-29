import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model=256, nhead=8, dim_feedforward=512):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.norm1(x + self.attn(x, x, x)[0])
        x = self.norm2(x + self.ffn(x))
        return x

class CompressionTest6(nn.Module):
    def __init__(self, num_layers=6, d_model=256, nhead=8):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, nhead) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

model = CompressionTest6()
example_input = torch.randn(2, 16, 256)  # (batch, seq_len, d_model)

show_compressed_view = True
collapse_modules_after_depth= 0

code_contents = """\
import torch
import torch.nn as nn
from torchvista import trace_model

class TransformerBlock(nn.Module):
    def __init__(self, d_model=256, nhead=8, dim_feedforward=512):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.norm1(x + self.attn(x, x, x)[0])
        x = self.norm2(x + self.ffn(x))
        return x

class CompressionTest6(nn.Module):
    def __init__(self, num_layers=6, d_model=256, nhead=8):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, nhead) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

model = CompressionTest6()
example_input = torch.randn(2, 16, 256)  # (batch, seq_len, d_model)

trace_model(model, example_input, collapse_modules_after_depth=0, show_compressed_view=True)

"""
