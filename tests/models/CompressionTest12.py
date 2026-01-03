import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, mult * dim),
            nn.GELU(),
            nn.Linear(mult * dim, dim),
        )

    def forward(self, x):
        return self.net(x)

class AttentionBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        # inner repetition: three identical FF nets
        self.ff_chain = nn.Sequential(*[FeedForward(dim) for _ in range(3)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + attn_out)
        x = self.ff_chain(x)
        return attn_out

class Stage(nn.Module):
    def __init__(self, dim, heads, depth):
        super().__init__()
        # mid-level repetition: N AttentionBlocks
        self.blocks = nn.Sequential(*[AttentionBlock(dim, heads) for _ in range(depth)])

    def forward(self, x):
        return self.blocks(x)

class CompressionTest12(nn.Module):
    """
    Nested repetition structure with three levels:
    - Outer: multiple Stages
    - Mid: each Stage has multiple AttentionBlocks
    - Inner: each AttentionBlock has multiple FeedForward nets

    Tests compression of deeply nested repeated modules.
    """
    def __init__(self, dim=64, heads=4, stage_depth=1, num_stages=2):
        super().__init__()
        self.stages = nn.Sequential(*[Stage(dim, heads, stage_depth) for _ in range(num_stages)])

    def forward(self, x):
        return self.stages(x)

model = CompressionTest12(dim=64, heads=4, stage_depth=1, num_stages=2)

example_input = torch.randn(2, 32, 64)

show_compressed_view = True
collapse_modules_after_depth = 6
forced_module_tracing_depth = 5
show_non_gradient_nodes = True

code_contents = """\
import torch
import torch.nn as nn
from torchvista import trace_model

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, mult * dim),
            nn.GELU(),
            nn.Linear(mult * dim, dim),
        )

    def forward(self, x):
        return self.net(x)

class AttentionBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        # inner repetition: three identical FF nets
        self.ff_chain = nn.Sequential(*[FeedForward(dim) for _ in range(3)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + attn_out)
        x = self.ff_chain(x)
        return attn_out

class Stage(nn.Module):
    def __init__(self, dim, heads, depth):
        super().__init__()
        # mid-level repetition: N AttentionBlocks
        self.blocks = nn.Sequential(*[AttentionBlock(dim, heads) for _ in range(depth)])

    def forward(self, x):
        return self.blocks(x)

class CompressionTest12(nn.Module):
    \"\"\"
    Nested repetition structure with three levels:
    - Outer: multiple Stages
    - Mid: each Stage has multiple AttentionBlocks
    - Inner: each AttentionBlock has multiple FeedForward nets

    Tests compression of deeply nested repeated modules.
    \"\"\"
    def __init__(self, dim=64, heads=4, stage_depth=1, num_stages=2):
        super().__init__()
        self.stages = nn.Sequential(*[Stage(dim, heads, stage_depth) for _ in range(num_stages)])

    def forward(self, x):
        return self.stages(x)

model = CompressionTest12(dim=64, heads=4, stage_depth=1, num_stages=2)
example_input = torch.randn(2, 32, 64)

trace_model(
    model,
    example_input,
    show_compressed_view=True,
    collapse_modules_after_depth=6,
    forced_module_tracing_depth=5,
    show_non_gradient_nodes=True,
)

"""
