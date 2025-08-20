import torch
import torch.nn as nn


class DynamicLayerSelector(nn.Module):
    def __init__(self):
        super(DynamicLayerSelector, self).__init__()
        
    def forward(self, x):        
        results = torch.zeros(x.size(0), x.size(1)).to(x.device)
        x = x + results
        
        return results

class MainModule(nn.Module):
    def __init__(self):
        super(MainModule, self).__init__()
        self.dls = DynamicLayerSelector()

    def forward(self, x):
        return self.dls(x)

model = MainModule()

example_input = torch.ones(3,  4)

code_contents = """\
import torch
import torch.nn as nn
from torchvista import trace_model

class DynamicLayerSelector(nn.Module):
    def __init__(self):
        super(DynamicLayerSelector, self).__init__()
        
    def forward(self, x):        
        results = torch.zeros(x.size(0), x.size(1)).to(x.device)
        x = x + results
        
        return results

class MainModule(nn.Module):
    def __init__(self):
        super(MainModule, self).__init__()
        self.dls = DynamicLayerSelector()

    def forward(self, x):
        return self.dls(x)

model = MainModule()

example_input = torch.ones(3,  4)

trace_model(model, example_input)

"""
