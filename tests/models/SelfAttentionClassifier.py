import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** 0.5

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attn = torch.softmax((q @ k.transpose(-2, -1)) / self.scale, dim=-1)
        return attn @ v

class AttentionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(32, 64)
        self.attn = SelfAttention(64)
        self.classifier = nn.Linear(64, 5)

    def forward(self, x):
        x = self.embed(x)
        x = self.attn(x)
        return self.classifier(x.mean(dim=1))

model = AttentionClassifier()
example_input = torch.randn(2, 10, 32)

code_contents = """\
import torch
import torch.nn as nn
from torchvista import trace_model

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** 0.5

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attn = torch.softmax((q @ k.transpose(-2, -1)) / self.scale, dim=-1)
        return attn @ v

class AttentionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(32, 64)
        self.attn = SelfAttention(64)
        self.classifier = nn.Linear(64, 5)

    def forward(self, x):
        x = self.embed(x)
        x = self.attn(x)
        return self.classifier(x.mean(dim=1))

model = AttentionClassifier()
example_input = torch.randn(2, 10, 32)

trace_model(model, example_input)

"""
