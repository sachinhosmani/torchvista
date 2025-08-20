import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMultiPathModule(nn.Module):
    def __init__(self, in_features=10, out_features=5):
        super(SimpleMultiPathModule, self).__init__()
        
        self.linear1 = nn.Linear(in_features, out_features)
        
        self.path_weights = nn.ParameterDict({
            'path1_weight': nn.Parameter(torch.tensor(0.5)),
        })
        
        self.global_biases = nn.ParameterList([
            nn.Parameter(torch.randn(1)),
            nn.Parameter(torch.randn(1))
        ])
    
    def forward(self, x, x2, x3):
        
        output1 = self.linear1(x)
        x += x2

        x *= F.relu(x3)
        
        results = torch.zeros(x.size(0), self.linear1.out_features, device=x.device)
                
        bias = 5 + self.global_biases[0] + self.global_biases[1] + output1[:,:]
        results = results + bias + output1[-5]
        
        return results

model = SimpleMultiPathModule()

x = torch.randn(3, 10)
y = torch.randn(3, 10)
z = torch.randn(3, 10)
example_input = (x, y, z)

code_contents = """\
import torch
import torch.nn as nn
from torchvista import trace_model

class SimpleMultiPathModule(nn.Module):
    def __init__(self, in_features=10, out_features=5):
        super(SimpleMultiPathModule, self).__init__()
        
        self.linear1 = nn.Linear(in_features, out_features)
        
        self.path_weights = nn.ParameterDict({
            'path1_weight': nn.Parameter(torch.tensor(0.5)),
        })
        
        self.global_biases = nn.ParameterList([
            nn.Parameter(torch.randn(1)),
            nn.Parameter(torch.randn(1))
        ])
    
    def forward(self, x, x2, x3):
        
        output1 = self.linear1(x)
        x += x2

        x *= F.relu(x3)
        
        results = torch.zeros(x.size(0), self.linear1.out_features, device=x.device)
                
        bias = 5 + self.global_biases[0] + self.global_biases[1] + output1[:,:]
        results = results + bias + output1[-5]
        
        return results

model = SimpleMultiPathModule()

x = torch.randn(3, 10)
y = torch.randn(3, 10)
z = torch.randn(3, 10)
example_input = (x, y, z)


trace_model(model, example_input)

"""
