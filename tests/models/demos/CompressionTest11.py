import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class CompressionTest11(nn.Module):
    """
    Two parallel branches from input:

    Branch 1 (imperfect): 4 MLP blocks with matching input/output dims (64->64),
    but the 3rd block has a different internal hidden_dim (256 instead of 128).
    This breaks the repetition pattern - blocks 1,2 repeat, block 3 is different,
    block 4 cannot repeat with block 3.

    Branch 2 (perfect): 4 identical MLP blocks with all dims matching (64->128->64).
    All 4 should be detected as repeating.
    """
    def __init__(self):
        super().__init__()
        # Branch 1: imperfect repetition (3rd block has different hidden_dim)
        self.branch_imperfect = nn.Sequential(
            MLP(64, 128, 64),  # block 0
            MLP(64, 128, 64),  # block 1 - matches block 0
            MLP(64, 256, 64),  # block 2 - different hidden_dim!
            MLP(64, 128, 64),  # block 3 - does not match block 2
        )

        # Branch 2: perfect repetition (all identical)
        self.branch_perfect = nn.Sequential(
            MLP(64, 128, 64),
            MLP(64, 128, 64),
            MLP(64, 128, 64),
            MLP(64, 128, 64),
        )

    def forward(self, x):
        out1 = self.branch_imperfect(x)
        out2 = self.branch_perfect(x)
        return out1 + out2

model = CompressionTest11()

example_input = torch.randn(2, 64)

show_compressed_view = True
collapse_modules_after_depth = 2

code_contents = """\
import torch
import torch.nn as nn
from torchvista import trace_model

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class CompressionTest11(nn.Module):
    \"\"\"
    Two parallel branches from input:

    Branch 1 (imperfect): 4 MLP blocks with matching input/output dims (64->64),
    but the 3rd block has a different internal hidden_dim (256 instead of 128).
    This breaks the repetition pattern - blocks 1,2 repeat, block 3 is different,
    block 4 cannot repeat with block 3.

    Branch 2 (perfect): 4 identical MLP blocks with all dims matching (64->128->64).
    All 4 should be detected as repeating.
    \"\"\"
    def __init__(self):
        super().__init__()
        # Branch 1: imperfect repetition (3rd block has different hidden_dim)
        self.branch_imperfect = nn.Sequential(
            MLP(64, 128, 64),  # block 0
            MLP(64, 128, 64),  # block 1 - matches block 0
            MLP(64, 256, 64),  # block 2 - different hidden_dim!
            MLP(64, 128, 64),  # block 3 - does not match block 2
        )

        # Branch 2: perfect repetition (all identical)
        self.branch_perfect = nn.Sequential(
            MLP(64, 128, 64),
            MLP(64, 128, 64),
            MLP(64, 128, 64),
            MLP(64, 128, 64),
        )

    def forward(self, x):
        out1 = self.branch_imperfect(x)
        out2 = self.branch_perfect(x)
        return out1 + out2

model = CompressionTest11()
example_input = torch.randn(2, 64)

trace_model(model, example_input, show_compressed_view=True, collapse_modules_after_depth=2)

"""
