import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return self.net(x)

class AttentionBlock(nn.Module):
    def __init__(self, dim=128, nhead=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, nhead, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.mlp = MLP(dim)

    def forward(self, x):
        x = self.norm(x + self.attn(x, x, x)[0])
        x = x + self.mlp(x)
        return x

class EncoderStage(nn.Module):
    def __init__(self, dim=128, nhead=4, num_blocks=4):
        super().__init__()
        self.blocks = nn.ModuleList([
            AttentionBlock(dim, nhead) for _ in range(num_blocks)
        ])
        self.downsample = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.downsample(x)

class DecoderStage(nn.Module):
    def __init__(self, dim=128, nhead=4, num_blocks=4):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.blocks = nn.ModuleList([
            AttentionBlock(dim, nhead) for _ in range(num_blocks)
        ])

    def forward(self, x):
        x = self.upsample(x)
        for block in self.blocks:
            x = block(x)
        return x

class CompressionTest7(nn.Module):
    def __init__(self, dim=128, nhead=4, num_stages=3, blocks_per_stage=4):
        super().__init__()
        self.encoder = nn.ModuleList([
            EncoderStage(dim, nhead, blocks_per_stage) for _ in range(num_stages)
        ])
        self.middle = nn.ModuleList([
            AttentionBlock(dim, nhead) for _ in range(6)
        ])
        self.decoder = nn.ModuleList([
            DecoderStage(dim, nhead, blocks_per_stage) for _ in range(num_stages)
        ])
        self.head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        for stage in self.encoder:
            x = stage(x)
        for block in self.middle:
            x = block(x)
        for stage in self.decoder:
            x = stage(x)
        return self.head(x)

model = CompressionTest7()
example_input = torch.randn(2, 16, 128)  # (batch, seq_len, dim)

show_compressed_view = True
collapse_modules_after_depth = 0

code_contents = """\
import torch
import torch.nn as nn
from torchvista import trace_model

class MLP(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return self.net(x)

class AttentionBlock(nn.Module):
    def __init__(self, dim=128, nhead=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, nhead, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.mlp = MLP(dim)

    def forward(self, x):
        x = self.norm(x + self.attn(x, x, x)[0])
        x = x + self.mlp(x)
        return x

class EncoderStage(nn.Module):
    def __init__(self, dim=128, nhead=4, num_blocks=4):
        super().__init__()
        self.blocks = nn.ModuleList([
            AttentionBlock(dim, nhead) for _ in range(num_blocks)
        ])
        self.downsample = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.downsample(x)

class DecoderStage(nn.Module):
    def __init__(self, dim=128, nhead=4, num_blocks=4):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.blocks = nn.ModuleList([
            AttentionBlock(dim, nhead) for _ in range(num_blocks)
        ])

    def forward(self, x):
        x = self.upsample(x)
        for block in self.blocks:
            x = block(x)
        return x

class CompressionTest7(nn.Module):
    def __init__(self, dim=128, nhead=4, num_stages=3, blocks_per_stage=4):
        super().__init__()
        self.encoder = nn.ModuleList([
            EncoderStage(dim, nhead, blocks_per_stage) for _ in range(num_stages)
        ])
        self.middle = nn.ModuleList([
            AttentionBlock(dim, nhead) for _ in range(6)
        ])
        self.decoder = nn.ModuleList([
            DecoderStage(dim, nhead, blocks_per_stage) for _ in range(num_stages)
        ])
        self.head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        for stage in self.encoder:
            x = stage(x)
        for block in self.middle:
            x = block(x)
        for stage in self.decoder:
            x = stage(x)
        return self.head(x)

model = CompressionTest7()
example_input = torch.randn(2, 16, 128)  # (batch, seq_len, dim)

trace_model(model, example_input, collapse_modules_after_depth=0, show_compressed_view=True)

"""
